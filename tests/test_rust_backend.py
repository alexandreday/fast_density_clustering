"""Tests for Rust backend (fdc_rs) — parity with Python and direct API tests."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as nmi

try:
    import fdc_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="fdc_rs not installed")


@pytest.fixture
def simple_blobs():
    X, y = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=0.4, random_state=42)
    return StandardScaler().fit_transform(X), y


class TestKnnQuery:
    def test_returns_correct_shapes(self, simple_blobs):
        X, _ = simple_blobs
        k = 20
        nn_dist, nn_idx = fdc_rs.knn_query(X, k)
        assert nn_dist.shape == (X.shape[0] * k,)
        assert nn_idx.shape == (X.shape[0] * k,)

    def test_self_is_nearest(self, simple_blobs):
        X, _ = simple_blobs
        nn_dist, nn_idx = fdc_rs.knn_query(X, 10)
        nn_dist = nn_dist.reshape(X.shape[0], 10)
        nn_idx = nn_idx.reshape(X.shape[0], 10)
        # First neighbor should be self (distance 0)
        np.testing.assert_allclose(nn_dist[:, 0], 0.0, atol=1e-10)
        np.testing.assert_array_equal(nn_idx[:, 0], np.arange(X.shape[0]))

    def test_distances_sorted(self, simple_blobs):
        X, _ = simple_blobs
        nn_dist, _ = fdc_rs.knn_query(X, 15)
        nn_dist = nn_dist.reshape(X.shape[0], 15)
        # Distances should be non-decreasing
        for i in range(X.shape[0]):
            assert np.all(np.diff(nn_dist[i]) >= -1e-10)

    def test_parity_with_sklearn(self, simple_blobs):
        from sklearn.neighbors import NearestNeighbors
        X, _ = simple_blobs
        k = 20
        # sklearn
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
        sk_dist, sk_idx = nbrs.kneighbors(X)
        # Rust
        rs_dist, rs_idx = fdc_rs.knn_query(X, k)
        rs_dist = rs_dist.reshape(X.shape[0], k)
        rs_idx = rs_idx.reshape(X.shape[0], k)
        # Distances should match closely
        np.testing.assert_allclose(rs_dist, sk_dist, atol=1e-10)


class TestKnnQueryCross:
    def test_cross_query_shapes(self, simple_blobs):
        X, _ = simple_blobs
        X_train, X_test = X[:400], X[400:]
        k = 10
        nn_dist, nn_idx = fdc_rs.knn_query_cross(X_train, X_test, k)
        assert nn_dist.shape == (100 * k,)
        assert nn_idx.shape == (100 * k,)


class TestEpanechnikovKde:
    def test_output_shape(self, simple_blobs):
        X, _ = simple_blobs
        nn_dist, _ = fdc_rs.knn_query(X, 20)
        nn_dist = nn_dist.reshape(X.shape[0], 20)
        rho = fdc_rs.epanechnikov_kde(nn_dist, 0.3, 2, X.shape[0])
        assert rho.shape == (X.shape[0],)

    def test_output_finite(self, simple_blobs):
        X, _ = simple_blobs
        nn_dist, _ = fdc_rs.knn_query(X, 20)
        nn_dist = nn_dist.reshape(X.shape[0], 20)
        rho = fdc_rs.epanechnikov_kde(nn_dist, 0.3, 2, X.shape[0])
        assert np.all(np.isfinite(rho))

    def test_parity_with_python(self, simple_blobs):
        from fdc.density_estimation import epanechnikov_kde_from_nn
        X, _ = simple_blobs
        nn_dist, _ = fdc_rs.knn_query(X, 20)
        nn_dist = nn_dist.reshape(X.shape[0], 20)
        h = 0.3
        py_rho = epanechnikov_kde_from_nn(nn_dist, h, 2, n_total=X.shape[0])
        rs_rho = np.asarray(fdc_rs.epanechnikov_kde(nn_dist, h, 2, X.shape[0]))
        np.testing.assert_allclose(rs_rho, py_rho, atol=1e-10)


class TestComputeDelta:
    def test_returns_correct_types(self, simple_blobs):
        X, _ = simple_blobs
        nn_dist, nn_idx = fdc_rs.knn_query(X, 20)
        nn_dist = nn_dist.reshape(X.shape[0], 20)
        nn_idx = nn_idx.reshape(X.shape[0], 20)
        rho = np.asarray(fdc_rs.epanechnikov_kde(nn_dist, 0.3, 2, X.shape[0]))
        delta, nn_delta, centers, dg = fdc_rs.compute_delta(X, rho, nn_idx, nn_dist)
        assert delta.shape == (X.shape[0],)
        assert nn_delta.shape == (X.shape[0],)
        assert len(centers) > 0
        assert len(dg) == X.shape[0]


class TestAssignCluster:
    def test_all_assigned(self):
        idx_centers = np.array([0, 3], dtype=np.int64)
        nn_delta = np.array([-1, 0, 0, -1, 3], dtype=np.int64)
        density_graph = [[1, 2], [], [], [4], []]
        labels = np.asarray(fdc_rs.assign_cluster(idx_centers, nn_delta, density_graph))
        assert labels.shape == (5,)
        assert np.all(labels >= 0)

    def test_centers_labeled_correctly(self):
        idx_centers = np.array([0, 2], dtype=np.int64)
        nn_delta = np.array([-1, 0, -1], dtype=np.int64)
        density_graph = [[1], [], []]
        labels = np.asarray(fdc_rs.assign_cluster(idx_centers, nn_delta, density_graph))
        assert labels[0] == 0
        assert labels[2] == 1


class TestRoundFloat:
    def test_basic(self):
        assert fdc_rs.round_float(0.00345) == pytest.approx(0.001)
        assert fdc_rs.round_float(0.0) == 0.0

    def test_large(self):
        assert fdc_rs.round_float(123.4) == pytest.approx(100.0)


class TestEndToEnd:
    def test_rust_backend_produces_good_clusters(self, simple_blobs):
        """Full end-to-end: FDC with Rust backend should produce high NMI."""
        from fdc import FDC
        X, y = simple_blobs
        model = FDC(eta=0.5, verbose=0, random_state=42)
        model.fit(X)
        score = nmi(y, model.cluster_label)
        assert score > 0.9, f"NMI too low: {score:.3f}"

    def test_rust_python_same_cluster_count(self, simple_blobs):
        """Rust and Python backends should find the same number of clusters."""
        import fdc.fdc as fdc_mod
        import fdc.density_estimation as kde_mod
        from fdc import FDC
        X, y = simple_blobs

        # Rust
        model_rust = FDC(eta=0.5, verbose=0, random_state=42)
        model_rust.fit(X)
        n_rust = len(np.unique(model_rust.cluster_label))

        # Python (temporarily disable Rust)
        orig_fdc = fdc_mod._HAS_RUST
        orig_kde = kde_mod._HAS_RUST
        fdc_mod._HAS_RUST = False
        kde_mod._HAS_RUST = False
        try:
            model_py = FDC(eta=0.5, verbose=0, random_state=42)
            model_py.fit(X)
            n_py = len(np.unique(model_py.cluster_label))
        finally:
            fdc_mod._HAS_RUST = orig_fdc
            kde_mod._HAS_RUST = orig_kde

        assert n_rust == n_py, f"Rust found {n_rust} clusters, Python found {n_py}"

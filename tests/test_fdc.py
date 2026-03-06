import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as nmi

from fdc import FDC
from fdc.fdc import assign_cluster


@pytest.fixture
def simple_blobs():
    """Well-separated Gaussian blobs — easy clustering case."""
    X, y = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=0.4, random_state=42)
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.fixture
def model():
    return FDC(eta=0.5, verbose=0, random_state=42)


class TestFDCFit:
    def test_fit_returns_self(self, model, simple_blobs):
        X, _ = simple_blobs
        result = model.fit(X)
        assert result is model

    def test_cluster_labels_assigned(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert model.cluster_label is not None

    def test_cluster_labels_shape(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert model.cluster_label.shape == (len(X),)

    def test_cluster_labels_no_unassigned(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert np.all(model.cluster_label >= 0)

    def test_finds_correct_cluster_count(self, simple_blobs):
        X, _ = simple_blobs
        model = FDC(eta=0.5, verbose=0, random_state=42)
        model.fit(X)
        n_clusters = len(np.unique(model.cluster_label))
        assert n_clusters == 5

    def test_high_nmi_on_clean_blobs(self, simple_blobs):
        X, y = simple_blobs
        model = FDC(eta=0.5, verbose=0, random_state=42)
        model.fit(X)
        score = nmi(y, model.cluster_label)
        assert score > 0.9, f"NMI too low: {score:.3f}"

    def test_explicit_bandwidth(self, simple_blobs):
        X, _ = simple_blobs
        model = FDC(eta=0.5, verbose=0, bandwidth=0.3)
        model.fit(X)
        assert model.cluster_label is not None
        assert model.bandwidth == 0.3

    def test_too_few_samples_raises(self, model):
        X = np.random.rand(5, 2)
        with pytest.raises(AssertionError):
            model.fit(X)


class TestFDCAttributes:
    def test_rho_shape(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert model.rho.shape == (len(X),)

    def test_bandwidth_positive(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert model.bandwidth > 0

    def test_idx_centers_nonempty(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert len(model.idx_centers) > 0

    def test_idx_centers_valid_indices(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert np.all(model.idx_centers < len(X))


class TestFDCNJob:
    def test_explicit_n_job_1(self, simple_blobs):
        X, _ = simple_blobs
        model = FDC(eta=0.5, verbose=0, n_job=1, random_state=42)
        model.fit(X)
        assert model.cluster_label is not None

    def test_n_job_capped_at_cpu_count(self):
        import multiprocessing
        model = FDC(n_job=99999)
        assert model.n_job <= multiprocessing.cpu_count()


class TestFDCSearchSizeClamping:
    def test_search_size_clamped_to_nh_size(self, simple_blobs):
        X, _ = simple_blobs
        model = FDC(eta=0.5, verbose=0, nh_size=20, search_size=100, random_state=42)
        model.fit(X)
        assert model.search_size <= model.nh_size


class TestFDCSaveLoad:
    def test_save_returns_path(self, model, simple_blobs, tmp_path):
        X, _ = simple_blobs
        model.fit(X)
        fname = str(tmp_path / "model.pkl")
        result = model.save(fname)
        assert result == fname

    def test_load_restores_cluster_labels(self, model, simple_blobs, tmp_path):
        X, _ = simple_blobs
        model.fit(X)
        fname = str(tmp_path / "model.pkl")
        model.save(fname)
        loaded = FDC().load(fname)
        np.testing.assert_array_equal(loaded.cluster_label, model.cluster_label)

    def test_make_file_name_format(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        fname = model.make_file_name()
        assert fname.startswith("fdc_")
        assert fname.endswith(".pkl")


class TestFDCCoarseGrain:
    def test_coarse_grain_returns_self(self, simple_blobs):
        X, _ = simple_blobs
        model = FDC(eta=0.0, verbose=0, random_state=42)
        model.fit(X)
        result = model.coarse_grain(np.linspace(0, 0.5, 5))
        assert result is model

    def test_coarse_grain_does_not_increase_clusters(self, simple_blobs):
        X, _ = simple_blobs
        model = FDC(eta=0.0, verbose=0, random_state=42)
        model.fit(X)
        n_before = len(model.idx_centers)
        model.coarse_grain(np.linspace(0, 0.5, 10))
        assert len(model.idx_centers) <= n_before

    def test_coarse_grain_sets_noise_range(self, simple_blobs):
        X, _ = simple_blobs
        model = FDC(eta=0.0, verbose=0, random_state=42)
        model.fit(X)
        noise = list(np.linspace(0, 0.3, 5))
        model.coarse_grain(noise)
        assert model.noise_range == noise


class TestFDCComputeDelta:
    def test_compute_delta_without_rho_uses_self_rho(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        # Re-run compute_delta without passing rho — hits the rho=None branch
        model.compute_delta(X)
        assert model.delta is not None
        assert model.nn_delta is not None
        assert model.idx_centers_unmerged is not None


class TestFDCEstimateEta:
    def test_estimate_eta_returns_float(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        eta = model.estimate_eta()
        assert isinstance(eta, float)


class TestFDCReset:
    def test_reset_clears_bandwidth(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert model.bandwidth is not None
        model.reset()
        assert model.bandwidth is None


class TestFDCFindNHv1:
    def test_find_nh_v1_returns_array(self, model, simple_blobs):
        X, _ = simple_blobs
        model.fit(X)
        assert model.cluster_label is not None
        nh = model.find_NH_tree_search_v1(0, model.rho[0] - 0.1, model.cluster_label)
        assert isinstance(nh, np.ndarray)


class TestFDCDuplicatePoints:
    def test_duplicate_points_do_not_crash(self):
        """Exact duplicate points should not cause bandwidth divergence."""
        X, _ = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.5, random_state=42)
        # Add exact duplicates
        X = np.vstack([X, X[:50]])
        model = FDC(eta=0.5, verbose=0, random_state=42)
        model.fit(X)
        assert model.cluster_label is not None
        assert model.cluster_label.shape == (250,)

    def test_all_identical_points(self):
        """All-identical dataset: should not crash, returns single cluster."""
        X = np.ones((100, 2))
        model = FDC(eta=0.5, verbose=0, bandwidth=0.5)
        model.fit(X)
        assert model.cluster_label is not None
        assert model.cluster_label.shape == (100,)


class TestAssignCluster:
    def test_all_samples_assigned(self):
        idx_centers = np.array([0, 3], dtype=int)
        nn_delta = np.array([-1, 0, 0, -1, 3], dtype=int)
        density_graph: list[list[int]] = [[1, 2], [], [], [4], []]
        labels = assign_cluster(idx_centers, nn_delta, density_graph)
        assert labels.shape == (5,)
        assert np.all(labels >= 0)

    def test_centers_get_correct_labels(self):
        idx_centers = np.array([0, 2], dtype=int)
        nn_delta = np.array([-1, 0, -1], dtype=int)
        density_graph: list[list[int]] = [[1], [], []]
        labels = assign_cluster(idx_centers, nn_delta, density_graph)
        assert labels[0] == 0
        assert labels[2] == 1

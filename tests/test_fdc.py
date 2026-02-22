import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as nmi

from fdc import FDC


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

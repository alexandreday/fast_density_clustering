import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from fdc.density_estimation import KDE


@pytest.fixture
def X():
    X, _ = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)
    return StandardScaler().fit_transform(X)


class TestKDEFit:
    def test_fit_returns_self(self, X):
        kde = KDE()
        result = kde.fit(X)
        assert result is kde

    def test_bandwidth_set_after_fit(self, X):
        kde = KDE()
        kde.fit(X)
        assert kde.bandwidth is not None
        assert kde.bandwidth > 0

    def test_explicit_bandwidth_preserved(self, X):
        kde = KDE(bandwidth=0.5)
        kde.fit(X)
        assert kde.bandwidth == 0.5


class TestKDEEvaluateDensity:
    def test_output_shape(self, X):
        kde = KDE()
        kde.fit(X)
        rho = kde.evaluate_density(X)
        assert rho.shape == (len(X),)

    def test_output_is_log_density(self, X):
        """Output should be finite log-density values."""
        kde = KDE()
        kde.fit(X)
        rho = kde.evaluate_density(X)
        assert np.all(np.isfinite(rho))

    def test_denser_region_has_higher_density(self):
        """Points at the center of a blob should have higher density than outliers."""
        rng = np.random.default_rng(42)
        center = rng.normal(loc=0, scale=0.2, size=(200, 2))
        outliers = rng.uniform(low=3, high=4, size=(20, 2))
        X = np.vstack([center, outliers])

        kde = KDE()
        kde.fit(X)
        rho = kde.evaluate_density(X)

        mean_center_rho = rho[:200].mean()
        mean_outlier_rho = rho[200:].mean()
        assert mean_center_rho > mean_outlier_rho

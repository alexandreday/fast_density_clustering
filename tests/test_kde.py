import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
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


class TestKDEKernels:
    @pytest.mark.parametrize("kernel", ["gaussian", "tophat", "linear", "epanechnikov"])
    def test_kernel_fits_and_produces_finite_density(self, X, kernel):
        kde = KDE(bandwidth=0.5, kernel=kernel)
        kde.fit(X)
        rho = kde.evaluate_density(X)
        assert rho.shape == (len(X),)
        assert np.all(np.isfinite(rho))


class TestKDEBandwidthEstimate:
    def test_returns_tuple_of_three_floats(self, X):
        X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
        kde = KDE()
        result = kde.bandwidth_estimate(X_train, X_test)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)

    def test_h_min_positive_and_h_max_greater(self, X):
        X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
        kde = KDE()
        h_est, h_min, h_max = kde.bandwidth_estimate(X_train, X_test)
        assert h_min > 0
        assert h_max > h_min

    def test_works_with_pre_supplied_nn_dist(self, X):
        from sklearn.neighbors import NearestNeighbors

        X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
        nn.fit(X_train)
        nn_dist, _ = nn.kneighbors(X_test, n_neighbors=2, return_distance=True)

        kde = KDE(nn_dist=nn_dist)
        h_est, h_min, h_max = kde.bandwidth_estimate(X_train, X_test)
        assert h_min > 0
        assert h_max > h_min
        assert np.isfinite(h_est)

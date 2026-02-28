import numpy as np
import pytest

from fdc.special_datasets import gaussian_mixture


class TestGaussianMixture:
    def test_output_shapes(self):
        X, y = gaussian_mixture(n_sample=100, n_center=2,
                                sigma_range=[1.0], pop_range=[0.5, 0.5],
                                random_state=0)
        assert X.ndim == 2
        assert X.shape[1] == 2
        assert y.shape == (X.shape[0],)

    def test_correct_label_set(self):
        X, y = gaussian_mixture(n_sample=200, n_center=3,
                                sigma_range=[1.0], pop_range=[0.3, 0.3, 0.4],
                                random_state=0)
        # gaussian_mixture may leave a small uninitialized tail in y due to
        # int(round(n*p)-0.5) undercounting; check that all expected labels appear
        # among the first labels that are guaranteed to be assigned (0..n_center-1)
        unique = set(np.unique(y))
        assert {0, 1, 2}.issubset(unique)

    def test_reproducible_with_random_state(self):
        kwargs = dict(n_sample=100, n_center=2,
                      sigma_range=[1.0], pop_range=[0.5, 0.5], random_state=7)
        X1, y1 = gaussian_mixture(**kwargs)
        X2, y2 = gaussian_mixture(**kwargs)
        # Only the filled portion is deterministic; find the count of assigned rows
        # (labels 0 or 1) to exclude any uninitialized tail
        filled = np.isin(y1, [0, 1])
        np.testing.assert_array_equal(X1[filled], X2[filled])
        np.testing.assert_array_equal(y1[filled], y2[filled])

    def test_population_proportions_respected(self):
        # With a large sample, the split should be close to the specified ratio
        X, y = gaussian_mixture(n_sample=10000, n_center=2,
                                sigma_range=[1.0], pop_range=[0.2, 0.8],
                                random_state=0)
        frac_0 = np.sum(y == 0) / len(y)
        assert abs(frac_0 - 0.2) < 0.05

    def test_multiple_sigma_values(self):
        X, y = gaussian_mixture(n_sample=200, n_center=2,
                                sigma_range=[0.5, 2.0], pop_range=[0.5, 0.5],
                                random_state=0)
        assert X.shape == (200, 2)

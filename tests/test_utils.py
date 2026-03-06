import numpy as np
import pytest

from fdc.fdc import index_greater, chunkIt
from fdc.density_estimation import round_float
from fdc.utils import transform


class TestIndexGreater:
    def test_finds_first_greater(self):
        arr = np.array([1.0, 0.5, 2.0, 3.0])
        assert index_greater(arr) == 2

    def test_returns_none_when_none_greater(self):
        arr = np.array([3.0, 2.0, 1.0, 0.5])
        assert index_greater(arr) is None

    def test_returns_none_for_equal_values(self):
        arr = np.array([1.0, 1.0, 1.0])
        assert index_greater(arr) is None

    def test_respects_precision(self):
        # difference smaller than default prec (1e-8) should not be detected
        arr = np.array([1.0, 1.0 + 1e-10])
        assert index_greater(arr) is None


class TestChunkIt:
    def test_correct_number_of_chunks(self):
        chunks = chunkIt(100, 4)
        assert len(chunks) == 4

    def test_chunks_cover_full_range(self):
        chunks = chunkIt(100, 4)
        assert chunks[0][0] == 0
        assert chunks[-1][1] == 100

    def test_chunks_are_contiguous(self):
        chunks = chunkIt(100, 4)
        for i in range(len(chunks) - 1):
            assert chunks[i][1] == chunks[i + 1][0]

    def test_single_chunk(self):
        chunks = chunkIt(50, 1)
        assert len(chunks) == 1
        assert chunks[0] == [0, 50]


class TestRoundFloat:
    def test_rounds_to_first_significant_digit(self):
        # returns the order of magnitude of the first significant digit
        assert round_float(0.00345) == 0.001

    def test_handles_small_values(self):
        assert round_float(0.001) == 0.001

    def test_returns_float(self):
        assert isinstance(round_float(0.0056), float)

    def test_zero_returns_zero(self):
        assert round_float(0.0) == 0.0


class TestTransform:
    def test_identity_transform(self):
        X = np.arange(6.0).reshape(2, 3)
        result = transform(X, [(lambda x: x, slice(None))])
        np.testing.assert_array_equal(result, X)

    def test_scaling_transform(self):
        X = np.ones((3, 2))
        result = transform(X, [(lambda x: x * 5, slice(None))])
        np.testing.assert_array_equal(result, np.full((3, 2), 5.0))

    def test_does_not_modify_original(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = X.copy()
        transform(X, [(lambda x: x * 0, slice(None))])
        np.testing.assert_array_equal(X, original)

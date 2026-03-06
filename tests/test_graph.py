"""Tests for fdc.graph module (DGRAPH class and helpers)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from fdc import FDC
from fdc.graph import DGRAPH, edge_info, merge_info


# ---------------------------------------------------------------------------
# cluster_label_standard
# ---------------------------------------------------------------------------

class TestClusterLabelStandard:
    """Tests for DGRAPH.cluster_label_standard()."""

    def _make_dgraph(self) -> DGRAPH:
        return DGRAPH()

    def test_labels_with_gaps(self):
        dg = self._make_dgraph()
        y = np.array([0, 0, 5, 5, 10])
        result = dg.cluster_label_standard(y=y)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2])

    def test_labels_already_standard(self):
        dg = self._make_dgraph()
        y = np.array([0, 0, 1, 1, 2, 2])
        result = dg.cluster_label_standard(y=y)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2])

    def test_single_cluster(self):
        dg = self._make_dgraph()
        y = np.array([7, 7, 7, 7])
        result = dg.cluster_label_standard(y=y)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_reverse_order_labels(self):
        dg = self._make_dgraph()
        y = np.array([10, 10, 5, 5, 0, 0])
        result = dg.cluster_label_standard(y=y)
        # unique sorts: [0, 5, 10] -> 0->0, 5->1, 10->2
        np.testing.assert_array_equal(result, [2, 2, 1, 1, 0, 0])

    def test_uses_self_cluster_label_when_y_is_none(self):
        dg = self._make_dgraph()
        dg.cluster_label = np.array([3, 3, 8, 8])
        result = dg.cluster_label_standard()
        np.testing.assert_array_equal(result, [0, 0, 1, 1])

    def test_raises_when_no_label_and_y_is_none(self):
        dg = self._make_dgraph()
        dg.cluster_label = None
        with pytest.raises(AssertionError):
            dg.cluster_label_standard()

    def test_output_dtype_is_int64(self):
        dg = self._make_dgraph()
        y = np.array([100, 200, 300])
        result = dg.cluster_label_standard(y=y)
        assert result.dtype == np.int64


# ---------------------------------------------------------------------------
# edge_info / merge_info (print helpers)
# ---------------------------------------------------------------------------

class TestPrintHelpers:
    def test_edge_info_robust(self, capsys):
        edge_info((0, 1), cv_score=0.95, std_score=0.02, min_score=0.5)
        captured = capsys.readouterr()
        assert "robust edge" in captured.out
        assert "0.95" in captured.out

    def test_edge_info_reject(self, capsys):
        edge_info((2, 3), cv_score=0.4, std_score=0.1, min_score=0.5)
        captured = capsys.readouterr()
        assert "reject edge" in captured.out

    def test_merge_info_prints(self, capsys):
        merge_info(c1=0, c2=1, score=0.55, new_c=5, n_cluster=3)
        captured = capsys.readouterr()
        assert "merge edge" in captured.out
        assert "n_cluster=" in captured.out


# ---------------------------------------------------------------------------
# DGRAPH integration (fit via FDC on simple blobs)
# ---------------------------------------------------------------------------

class TestDGRAPHIntegration:
    """Integration tests that fit DGRAPH through a real FDC model."""

    @pytest.fixture()
    def fdc_blobs(self):
        """Fit FDC on nearby blobs so DGRAPH finds inter-cluster edges."""
        X, _ = make_blobs(
            n_samples=300,
            centers=np.array([[0, 0], [2, 0], [1, 2]]),
            cluster_std=0.8,
            random_state=42,
        )
        model = FDC(verbose=0)
        model.fit(X)
        return model, X

    def test_fit_returns_self(self, fdc_blobs):
        model, X = fdc_blobs
        dg = DGRAPH()
        result = dg.fit(model, X)
        assert result is dg

    def test_cluster_label_shape(self, fdc_blobs):
        model, X = fdc_blobs
        dg = DGRAPH()
        dg.fit(model, X)
        assert dg.cluster_label is not None
        assert dg.cluster_label.shape == (X.shape[0],)

    def test_nn_list_populated(self, fdc_blobs):
        model, X = fdc_blobs
        dg = DGRAPH()
        dg.fit(model, X)
        assert len(dg.nn_list) > 0

    def test_graph_has_edges(self, fdc_blobs):
        model, X = fdc_blobs
        dg = DGRAPH()
        dg.fit(model, X)
        assert len(dg.graph) > 0

    def test_edge_score_has_entries(self, fdc_blobs):
        model, X = fdc_blobs
        dg = DGRAPH()
        dg.fit(model, X)
        assert len(dg.edge_score) > 0
        for key, val in dg.edge_score.items():
            assert len(key) == 2
            assert len(val) == 2  # [cv_score, cv_score_std]


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    @staticmethod
    def _make_picklable_dgraph() -> DGRAPH:
        """Create a DGRAPH and remove its unpicklable file handle."""
        dg = DGRAPH()
        dg.fout.close()
        del dg.fout
        return dg

    def test_round_trip(self, tmp_path):
        dg = self._make_picklable_dgraph()
        dg.cluster_label = np.array([0, 0, 1, 1, 2, 2])
        dg.init_n_cluster = 3

        fpath = str(tmp_path / "dgraph_test.pkl")
        dg.save(fpath)

        loaded = self._make_picklable_dgraph()
        loaded.load(fpath)
        np.testing.assert_array_equal(loaded.cluster_label, dg.cluster_label)
        assert loaded.init_n_cluster == 3

    def test_default_filename(self, tmp_path, monkeypatch):
        """save()/load() with no name uses make_file_name() default."""
        monkeypatch.chdir(tmp_path)
        dg = self._make_picklable_dgraph()
        dg.cluster_label = np.array([5, 5, 10, 10])
        dg.save()

        loaded = self._make_picklable_dgraph()
        loaded.load()
        np.testing.assert_array_equal(loaded.cluster_label, dg.cluster_label)

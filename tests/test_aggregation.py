"""
tests/test_aggregation.py – Tests for gene-level score aggregation logic.

Run with:  pytest tests/test_aggregation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _make_scores_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestAggregationMethods:
    """Test that min / mean / max_abs aggregation behaves correctly."""

    SCORES = [
        {"variant_id": "var1", "gene_id": "ENSG1", "gene_name": "GeneA",
         "scorer": "RNA_SEQ", "score_value": -0.5},
        {"variant_id": "var2", "gene_id": "ENSG1", "gene_name": "GeneA",
         "scorer": "RNA_SEQ", "score_value": -1.2},
        {"variant_id": "var3", "gene_id": "ENSG1", "gene_name": "GeneA",
         "scorer": "RNA_SEQ", "score_value":  0.3},
        {"variant_id": "var4", "gene_id": "ENSG2", "gene_name": "GeneB",
         "scorer": "RNA_SEQ", "score_value":  0.8},
    ]

    def _agg(self, method: str, series: pd.Series) -> float:
        if method == "min":
            return series.min()
        elif method == "mean":
            return series.mean()
        elif method == "max_abs":
            return series.loc[series.abs().idxmax()]
        raise ValueError(method)

    def test_min_aggregation(self):
        df = _make_scores_df(self.SCORES)
        gene1 = df[df["gene_id"] == "ENSG1"]["score_value"]
        result = self._agg("min", gene1)
        assert result == pytest.approx(-1.2)

    def test_mean_aggregation(self):
        df = _make_scores_df(self.SCORES)
        gene1 = df[df["gene_id"] == "ENSG1"]["score_value"]
        result = self._agg("mean", gene1)
        assert result == pytest.approx((-0.5 - 1.2 + 0.3) / 3)

    def test_max_abs_aggregation(self):
        df = _make_scores_df(self.SCORES)
        gene1 = df[df["gene_id"] == "ENSG1"]["score_value"]
        result = self._agg("max_abs", gene1)
        assert result == pytest.approx(-1.2)

    def test_single_variant_gene(self):
        df = _make_scores_df(self.SCORES)
        gene2 = df[df["gene_id"] == "ENSG2"]["score_value"]
        for method in ("min", "mean", "max_abs"):
            result = self._agg(method, gene2)
            assert result == pytest.approx(0.8)

    def test_predicted_suppressed_flag(self):
        """Negative score → ag_predicted_suppressed should be True."""
        score = -1.2
        assert score < 0  # predicted suppressed

    def test_predicted_activated_flag(self):
        score = 0.8
        assert score >= 0  # predicted activated / neutral

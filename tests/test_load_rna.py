"""
tests/test_load_rna.py – Unit tests for RNA expression loading.

Run with:  pytest tests/test_load_rna.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import log2_tpm, classify_expressed, strip_version


class TestLog2TPM:
    def test_zero(self):
        s = pd.Series([0.0])
        result = log2_tpm(s, pseudo=1.0)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_one(self):
        s = pd.Series([1.0])
        result = log2_tpm(s, pseudo=1.0)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_vectorised(self):
        s = pd.Series([0.0, 1.0, 3.0])
        result = log2_tpm(s)
        assert len(result) == 3
        assert all(result >= 0)


class TestClassifyExpressed:
    def test_above_threshold(self):
        s = pd.Series([0.0, 0.5, 1.0, 2.0, 10.0])
        result = classify_expressed(s, threshold=1.0)
        expected = pd.Series([False, False, True, True, True])
        pd.testing.assert_series_equal(result, expected)

    def test_default_threshold(self):
        s = pd.Series([0.99, 1.0, 1.01])
        result = classify_expressed(s)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert result.iloc[2]


class TestLoadRnaIntegration:
    @pytest.mark.skipif(
        not Path("data/raw/Example_RNA.xlsx").exists(),
        reason="data/raw/Example_RNA.xlsx not present",
    )
    def test_loads_without_error(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "load_rna", Path("src/2_load_rna.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cfg = {
            "data": {"rna_xlsx": "data/raw/Example_RNA.xlsx"},
            "output": {"results_dir": str(tmp_path)},
            "comparison": {
                "tpm_column": "tpm_unstranded",
                "expressed_threshold": 1.0,
                "log_transform_tpm": True,
            },
        }
        df = mod.run(cfg)
        assert len(df) > 0
        assert "gene_id" in df.columns
        assert "tpm_unstranded" in df.columns
        assert "log2_tpm" in df.columns
        assert "is_expressed" in df.columns
        # gene_ids should be unversioned
        assert not df["gene_id"].str.contains(r"\.\d+$", regex=True).any()

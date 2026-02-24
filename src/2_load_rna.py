"""
2_load_rna.py – Load and clean the TCGA RNA-seq expression data.

Input:  data/raw/Example_RNA.xlsx  (GENCODE v36 gene expression table)
Output: results/rna_expression.parquet

Key steps:
  1. Read the Excel file (header is on row 2; row 1 is a metadata comment).
  2. Drop summary rows (N_unmapped, N_multimapping, etc.).
  3. Strip Ensembl version suffixes from gene_id.
  4. Apply a log2(TPM + 1) transform for later use in correlations.
  5. Classify genes as expressed (TPM ≥ threshold) or not.
  6. Save the cleaned table.

GENCODE v36 is used in the RNA file; the VEP annotations use GENCODE v43-46
gene IDs.  Ensembl gene IDs are stable across versions (ENSG…), so matching
on the stripped ID (no version suffix) is safe for the vast majority of genes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, strip_version, log2_tpm, classify_expressed,
    save_parquet, setup_logging, ensure_dir,
)

# Summary rows present at the top of TCGA STAR-Counts files
_SUMMARY_PREFIXES = ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(config: dict) -> pd.DataFrame:
    xlsx_path = Path(config["data"]["rna_xlsx"])
    results_dir = ensure_dir(config["output"]["results_dir"])
    out_path = results_dir / "rna_expression.parquet"
    tpm_col = config["comparison"]["tpm_column"]
    tpm_threshold = config["comparison"]["expressed_threshold"]
    log_transform = config["comparison"]["log_transform_tpm"]

    logger.info(f"Loading RNA expression data: {xlsx_path}")

    # ---- Step 1: Read Excel (header on row 2, index 1) ----------
    # The first row is a metadata comment: "# gene-model: GENCODE v36"
    df_raw = pd.read_excel(xlsx_path, header=1, engine="openpyxl")

    # Rename columns to standard names (they already match after header=1)
    expected_cols = [
        "gene_id", "gene_name", "gene_type",
        "unstranded", "stranded_first", "stranded_second",
        "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded",
    ]
    df_raw.columns = expected_cols

    logger.info(f"Raw table: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    # ---- Step 2: Drop summary rows ------------------------------
    summary_mask = df_raw["gene_id"].str.startswith(_SUMMARY_PREFIXES, na=False)
    df = df_raw[~summary_mask].copy()
    logger.info(f"After dropping summary rows: {len(df):,} genes")

    # ---- Step 3: Clean Ensembl IDs ------------------------------
    df["gene_id_versioned"] = df["gene_id"].astype(str)
    df["gene_id"] = df["gene_id_versioned"].apply(strip_version)

    # ---- Step 4: Numeric types ----------------------------------
    numeric_cols = ["unstranded", "stranded_first", "stranded_second",
                    "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where TPM is NaN (shouldn't happen for well-formed files)
    before = len(df)
    df = df.dropna(subset=[tpm_col])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df):,} rows with NaN {tpm_col}")

    # ---- Step 5: Derived columns --------------------------------
    df["log2_tpm"] = log2_tpm(df[tpm_col])
    df["is_expressed"] = classify_expressed(df[tpm_col], tpm_threshold)

    # ---- Step 6: Summary ----------------------------------------
    _print_summary(df, tpm_col, tpm_threshold)

    # ---- Save ---------------------------------------------------
    save_parquet(df, out_path)
    return df


def _print_summary(df: pd.DataFrame, tpm_col: str, tpm_threshold: float) -> None:
    n_total = len(df)
    n_expressed = df["is_expressed"].sum()
    n_silent = n_total - n_expressed

    logger.info("=== RNA Expression Summary ===")
    logger.info(f"  Total genes          : {n_total:,}")
    logger.info(f"  Expressed (TPM≥{tpm_threshold}): {n_expressed:,} ({100*n_expressed/n_total:.1f}%)")
    logger.info(f"  Silent    (TPM<{tpm_threshold}) : {n_silent:,} ({100*n_silent/n_total:.1f}%)")
    logger.info(f"  Median TPM           : {df[tpm_col].median():.3f}")
    logger.info(f"  Mean TPM (expressed) : {df.loc[df['is_expressed'], tpm_col].mean():.1f}")

    # Gene type breakdown
    top_types = (
        df.groupby("gene_type")["is_expressed"]
        .agg(["sum", "count"])
        .assign(pct=lambda x: 100 * x["sum"] / x["count"])
        .sort_values("count", ascending=False)
        .head(5)
    )
    logger.info("  Top gene types (expressed %):")
    for gtype, row in top_types.iterrows():
        logger.info(f"    {gtype:<30} {row['sum']:>6}/{row['count']:>6} ({row['pct']:.0f}%)")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load TCGA RNA expression data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_file"], cfg["logging"]["level"])
    run(cfg)

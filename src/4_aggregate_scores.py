"""
4_aggregate_scores.py – Aggregate per-variant AlphaGenome scores to gene level.

Input:
  results/alphagenome_scores.parquet   (per-variant, per-scorer scores)
  results/vcf_parsed.parquet           (variant → gene mapping)

Output:
  results/gene_scores.parquet          (per-gene aggregated score)

Aggregation rationale
---------------------
A single gene can harbour multiple somatic variants.  To represent the
gene-level predicted expression impact we need to collapse variant scores.

Three approaches are available (config: aggregation.method):

  "min"       : Most negative ∆RNA-seq across variants per gene.
                Assumes a tumour-suppressor-like model: the worst observed
                suppression dominates.  Appropriate when asking "is this gene
                likely to be silenced?".

  "mean"      : Average ∆RNA-seq.  Blends enhancing and suppressing effects;
                reasonable for a gene with many small-effect variants.

  "max_abs"   : Largest absolute change.  Captures the most impactful variant
                regardless of direction; useful when exploring both activating
                and silencing effects.

The primary scorer (config: aggregation.primary_scorer) is used for the
headline gene score; scores from other scorers are also retained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, load_parquet, save_parquet, setup_logging, ensure_dir


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(config: dict) -> pd.DataFrame:
    results_dir = ensure_dir(config["output"]["results_dir"])
    out_path = results_dir / "gene_scores.parquet"

    agg_method = config["aggregation"]["method"]
    primary_scorer = config["aggregation"]["primary_scorer"]  # e.g. "rna_seq"

    # ---- Load inputs --------------------------------------------
    df_scores = load_parquet(results_dir / "alphagenome_scores.parquet")
    df_vcf = load_parquet(results_dir / "vcf_parsed.parquet")

    logger.info(f"Score rows: {len(df_scores):,}, VCF rows: {len(df_vcf):,}")

    # ---- Map variants → genes -----------------------------------
    # df_vcf has one row per (variant, gene) pair
    variant_gene_map = (
        df_vcf[["variant_id", "gene_id", "gene_name"]]
        .dropna(subset=["gene_id"])
        .query("gene_id != ''")
        .drop_duplicates()
    )

    # Join scores with gene info
    df = df_scores.merge(variant_gene_map, on="variant_id", how="inner")
    logger.info(f"After joining to gene map: {len(df):,} rows, "
                f"{df['gene_id'].nunique():,} genes")

    # ---- Identify score column ----------------------------------
    # AlphaGenome tidy_scores() produces columns like:
    #   variant_id | scorer | score_name | score_value | …
    # We pivot on scorer / score_name to get one column per scorer.
    if "score_value" not in df.columns:
        raise ValueError("Expected 'score_value' column in AlphaGenome scores. "
                         "Check the output format of variant_scorers.tidy_scores().")

    # Filter to primary scorer rows for headline aggregation
    primary_mask = df["scorer"].str.lower().str.contains(primary_scorer.lower(), na=False)
    df_primary = df[primary_mask].copy()
    if df_primary.empty:
        logger.warning(
            f"No rows matched primary scorer '{primary_scorer}'. "
            "Using all scorers for aggregation."
        )
        df_primary = df.copy()

    # ---- Aggregate to gene level --------------------------------
    def _agg(series: pd.Series) -> float:
        if agg_method == "min":
            return series.min()
        elif agg_method == "mean":
            return series.mean()
        elif agg_method == "max_abs":
            return series.loc[series.abs().idxmax()]
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method!r}")

    gene_agg = (
        df_primary
        .groupby(["gene_id", "gene_name"])["score_value"]
        .agg(
            ag_score_primary=_agg,
            ag_score_min=lambda x: x.min(),
            ag_score_mean=lambda x: x.mean(),
            ag_score_max=lambda x: x.max(),
            n_variants=lambda x: x.count(),
        )
        .reset_index()
    )

    gene_agg.rename(columns={"ag_score_primary": f"ag_score_{agg_method}"}, inplace=True)

    # Also compute per-scorer gene-level scores (for auxiliary analysis)
    scorer_agg = (
        df
        .groupby(["gene_id", "gene_name", "scorer"])["score_value"]
        .agg(_agg)
        .unstack("scorer")
        .reset_index()
    )
    scorer_agg.columns = [
        f"ag_{col.lower()}" if col not in ("gene_id", "gene_name") else col
        for col in scorer_agg.columns
    ]

    df_gene = gene_agg.merge(scorer_agg, on=["gene_id", "gene_name"], how="left")

    # ---- Classify variants by predicted direction ---------------
    # Negative score → predicted to reduce expression
    df_gene["ag_predicted_suppressed"] = df_gene[f"ag_score_{agg_method}"] < 0
    df_gene["ag_score_magnitude"] = df_gene[f"ag_score_{agg_method}"].abs()

    logger.info(f"Gene-level scores: {len(df_gene):,} genes")
    logger.info(f"  Predicted suppressed: "
                f"{df_gene['ag_predicted_suppressed'].sum():,} "
                f"({100*df_gene['ag_predicted_suppressed'].mean():.1f}%)")

    save_parquet(df_gene, out_path)
    return df_gene


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate AlphaGenome scores to gene level")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_file"], cfg["logging"]["level"])
    run(cfg)

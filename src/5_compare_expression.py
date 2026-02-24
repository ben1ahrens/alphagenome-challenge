"""
5_compare_expression.py – Compare AlphaGenome predictions with observed RNA expression.

Inputs:
  results/gene_scores.parquet      (AlphaGenome gene-level scores, step 4)
  results/rna_expression.parquet   (observed TPM per gene, step 2)

Outputs:
  results/merged.parquet           (merged predicted + observed table)
  results/figures/scatter_predicted_vs_observed.png
  results/figures/roc_curve.png
  results/figures/tpm_stratified_boxplot.png
  results/figures/volcano_ag_score.png

Analysis design
---------------
The core question is: do variants predicted by AlphaGenome to reduce expression
associate with lower observed TPM in the same patient?

We approach this in three complementary ways:

1. Correlation (continuous × continuous)
   Spearman ρ between the AlphaGenome gene score and log2(TPM + 1).
   Spearman is preferred over Pearson because TPM distributions are
   heavily right-skewed and scores may not be normally distributed.

2. Classification (TPM threshold)
   Treat "expressed" (TPM ≥ 1) as the positive label.
   Ask: can AlphaGenome score discriminate expressed from silent genes?
   → ROC-AUC using ag_score as a predictor (higher score = more expressed).

3. Stratified comparison (TPM >1 vs ≤1)
   Compare the distribution of AlphaGenome scores in expressed vs silent genes.
   Expected: genes predicted to have reduced expression (lower ag_score)
   should be enriched in the silent group.

Caveats
-------
- Most somatic variants are in intergenic/intronic regions and may have small
  predicted effects. We expect a modest, noisy correlation.
- The RNA data comes from a single tumour sample; many genes with low expression
  could be silenced by copy-number loss or promoter methylation rather than
  the variants captured in this VCF.
- AlphaGenome predicts regulatory potential from sequence alone and does not
  model copy-number variation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_parquet, save_parquet,
    setup_logging, ensure_dir,
)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(config: dict) -> pd.DataFrame:
    results_dir = ensure_dir(config["output"]["results_dir"])
    figures_dir = ensure_dir(config["output"]["figures_dir"])
    out_path = results_dir / "merged.parquet"

    tpm_col = config["comparison"]["tpm_column"]
    tpm_threshold = config["comparison"]["expressed_threshold"]
    agg_method = config["aggregation"]["method"]
    score_col = f"ag_score_{agg_method}"

    # ---- Load data ---------------------------------------------
    df_scores = load_parquet(results_dir / "gene_scores.parquet")
    df_rna = load_parquet(results_dir / "rna_expression.parquet")

    logger.info(f"Gene scores: {len(df_scores):,} genes")
    logger.info(f"RNA genes  : {len(df_rna):,} genes")

    # ---- Merge -------------------------------------------------
    df = df_scores.merge(
        df_rna[["gene_id", "gene_name", tpm_col, "log2_tpm", "is_expressed", "gene_type"]],
        on="gene_id",
        how="inner",
        suffixes=("_ag", "_rna"),
    )
    df["gene_name"] = df["gene_name_ag"].fillna(df["gene_name_rna"])

    logger.info(f"Merged table: {len(df):,} genes with both scores and expression")

    if df.empty:
        logger.error("Merge produced empty table. Check gene_id formats match "
                     "(both should be unversioned Ensembl IDs, e.g. ENSG00000157764).")
        return df

    # ---- 1. Spearman correlation --------------------------------
    valid = df[[score_col, "log2_tpm"]].dropna()
    rho, pval = stats.spearmanr(valid[score_col], valid["log2_tpm"])
    logger.info(f"Spearman ρ (ag_score vs log2_tpm) = {rho:.3f}  (p = {pval:.2e}, n = {len(valid)})")

    # ---- 2. AUC ------------------------------------------------
    auc_df = df[[score_col, "is_expressed"]].dropna()
    if auc_df["is_expressed"].nunique() == 2:
        auc = roc_auc_score(auc_df["is_expressed"], auc_df[score_col])
        logger.info(f"ROC-AUC (expressed vs silent) = {auc:.3f}")
    else:
        auc = float("nan")
        logger.warning("Only one class in is_expressed; AUC not computed.")

    # ---- 3. Stratified comparison -------------------------------
    expressed_scores = df.loc[df["is_expressed"], score_col].dropna()
    silent_scores = df.loc[~df["is_expressed"], score_col].dropna()
    mw_stat, mw_pval = stats.mannwhitneyu(
        expressed_scores, silent_scores, alternative="two-sided"
    )
    logger.info(
        f"Mann–Whitney U (expressed vs silent scores): "
        f"U = {mw_stat:.0f}, p = {mw_pval:.2e}"
    )

    # Add summary stats to df
    df["spearman_rho"] = rho
    df["roc_auc"] = auc

    # ---- Figures -----------------------------------------------
    _plot_scatter(df, score_col, tpm_col, rho, pval, figures_dir)
    if not np.isnan(auc):
        _plot_roc(auc_df, score_col, auc, figures_dir)
    _plot_boxplot(df, score_col, tpm_threshold, figures_dir)
    _plot_volcano(df, score_col, tpm_col, figures_dir)

    # ---- Save --------------------------------------------------
    save_parquet(df, out_path)
    logger.info("Comparison complete.")
    return df


# ------------------------------------------------------------------ #
# Figures                                                              #
# ------------------------------------------------------------------ #

def _plot_scatter(df, score_col, tpm_col, rho, pval, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        df[score_col], df["log2_tpm"],
        alpha=0.35, s=12, edgecolors="none", c="steelblue",
    )
    ax.set_xlabel(f"AlphaGenome score ({score_col})", fontsize=12)
    ax.set_ylabel("log₂(TPM + 1) [observed]", fontsize=12)
    ax.set_title(
        f"Predicted ∆RNA-seq vs Observed Expression\n"
        f"Spearman ρ = {rho:.3f}  (p = {pval:.1e},  n = {len(df):,})",
        fontsize=11,
    )
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="tomato", linewidth=0.8, linestyle="--", label="TPM = 1 threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = figures_dir / "scatter_predicted_vs_observed.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def _plot_roc(auc_df, score_col, auc, figures_dir):
    fpr, tpr, _ = roc_curve(auc_df["is_expressed"], auc_df[score_col])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="steelblue",
            label=f"AlphaGenome (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Predicted Score → Expressed (TPM ≥ 1)")
    ax.legend()
    plt.tight_layout()
    path = figures_dir / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def _plot_boxplot(df, score_col, tpm_threshold, figures_dir):
    plot_df = df[[score_col, "is_expressed"]].dropna().copy()
    plot_df["Expression"] = plot_df["is_expressed"].map(
        {True: f"Expressed\n(TPM ≥ {tpm_threshold})",
         False: f"Silent\n(TPM < {tpm_threshold})"}
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(
        data=plot_df, x="Expression", y=score_col, ax=ax,
        palette={"Expressed\n(TPM ≥ 1.0)": "steelblue",
                 "Silent\n(TPM < 1.0)": "tomato"},
        order=[f"Expressed\n(TPM ≥ {tpm_threshold})",
               f"Silent\n(TPM < {tpm_threshold})"],
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"AlphaGenome score ({score_col})")
    ax.set_title("AlphaGenome Score by Expression Status\n(TCGA-LUAD, TCGA-05-4384)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    path = figures_dir / "tpm_stratified_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def _plot_volcano(df, score_col, tpm_col, figures_dir):
    """Scatter: AlphaGenome score (x) vs observed log2 TPM (y), highlight extremes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background points
    ax.scatter(df[score_col], df["log2_tpm"],
               alpha=0.2, s=10, c="grey", edgecolors="none")

    # Highlight strongly predicted-suppressed genes with low observed expression
    suppressed = df[(df[score_col] < df[score_col].quantile(0.1)) &
                    (df["log2_tpm"] < df["log2_tpm"].quantile(0.25))]
    ax.scatter(suppressed[score_col], suppressed["log2_tpm"],
               alpha=0.8, s=25, c="tomato", edgecolors="none",
               label=f"Suppressed candidates (n={len(suppressed)})")

    # Annotate top candidates by name
    top = suppressed.nsmallest(10, score_col)
    for _, r in top.iterrows():
        ax.annotate(r.get("gene_name", ""), (r[score_col], r["log2_tpm"]),
                    fontsize=7, color="darkred", xytext=(5, 5),
                    textcoords="offset points")

    ax.set_xlabel(f"AlphaGenome ∆RNA-seq score ({score_col})", fontsize=11)
    ax.set_ylabel("log₂(TPM + 1) observed", fontsize=11)
    ax.set_title("Cancer Vaccine Antigen Candidate Identification\nSuppress-predicted & low-observed genes",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    plt.tight_layout()
    path = figures_dir / "volcano_ag_score.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare AlphaGenome predictions with RNA expression")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_file"], cfg["logging"]["level"])
    run(cfg)

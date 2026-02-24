"""
6_gtex_context.py – Contextualise results with GTEx reference expression.

Inputs:
  results/merged.parquet     (merged predicted + observed, step 5)

Outputs:
  results/gtex_context.parquet
  results/figures/gtex_vs_tcga_tpm.png
  results/figures/gtex_predicted_heatmap.png

Purpose
-------
GTEx provides reference RNA expression across healthy tissues, including lung.
Comparing AlphaGenome predictions and the TCGA tumour expression against GTEx
normal lung helps us identify:

  1. Genes with normally high expression that appear suppressed in the tumour
     (potential tumour suppressors or regulatory disruption targets).

  2. Whether AlphaGenome predictions track normal lung expression, suggesting
     the model has some lung tissue specificity despite not being cancer-specific.

API
---
We use the public GTEx REST API v2 (no authentication required):
  https://gtexportal.org/api/v2/expression/geneExpression
  Returns median TPM per tissue for a gene.

For bulk download, the GTEx portal (https://gtexportal.org) provides pre-computed
median expression matrices.  This script uses the API for convenience with
the genes actually observed in our merged dataset.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_parquet, save_parquet,
    setup_logging, ensure_dir,
)

GTEX_API = "https://gtexportal.org/api/v2/expression/geneExpression"
GTEX_TISSUE = "Lung"
SLEEP_BETWEEN_REQUESTS = 0.3   # seconds; be polite to GTEx API


# ------------------------------------------------------------------ #
# GTEx data fetching                                                   #
# ------------------------------------------------------------------ #

def fetch_gtex_median_tpm(gene_ids: list[str],
                           tissue: str = GTEX_TISSUE,
                           batch_size: int = 50) -> pd.DataFrame:
    """
    Fetch median TPM from GTEx for a list of Ensembl gene IDs.

    Returns a DataFrame with columns: gene_id, gtex_median_tpm.
    Genes not found in GTEx are returned with NaN.

    Parameters
    ----------
    gene_ids : Ensembl gene IDs (unversioned, e.g. 'ENSG00000157764')
    tissue   : GTEx tissue name (e.g. 'Lung')
    batch_size : number of gene IDs per API request
    """
    results: list[dict] = []
    n_batches = (len(gene_ids) + batch_size - 1) // batch_size

    logger.info(f"Fetching GTEx {tissue} expression for {len(gene_ids):,} genes "
                f"({n_batches} batches)…")

    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i : i + batch_size]
        params = {
            "tissueSiteDetailId": tissue,
            "gencodeId": ",".join(batch),
            "attributeSubset": "median",
        }
        try:
            resp = requests.get(GTEX_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"GTEx API error for batch {i//batch_size + 1}: {exc}")
            for gid in batch:
                results.append({"gene_id": gid, "gtex_median_tpm": float("nan")})
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        # Parse response
        gene_data = data.get("data", [])
        found_ids = set()
        for item in gene_data:
            # GTEx returns versioned Ensembl IDs; strip version
            raw_id = item.get("gencodeId", "")
            gid = raw_id.split(".")[0]
            median_tpm = item.get("median", float("nan"))
            results.append({"gene_id": gid, "gtex_median_tpm": float(median_tpm)})
            found_ids.add(gid)

        # Fill not-found genes
        for gid in batch:
            if gid not in found_ids:
                results.append({"gene_id": gid, "gtex_median_tpm": float("nan")})

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return pd.DataFrame(results).drop_duplicates("gene_id")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(config: dict) -> pd.DataFrame:
    results_dir = ensure_dir(config["output"]["results_dir"])
    figures_dir = ensure_dir(config["output"]["figures_dir"])
    out_path = results_dir / "gtex_context.parquet"

    gtex_tissue = config["gtex"]["tissue"]
    top_n = config["gtex"]["top_n_genes"]
    agg_method = config["aggregation"]["method"]
    score_col = f"ag_score_{agg_method}"
    tpm_col = config["comparison"]["tpm_column"]

    # ---- Load merged table -------------------------------------
    df = load_parquet(results_dir / "merged.parquet")
    logger.info(f"Loaded merged data: {len(df):,} genes")

    gene_ids = df["gene_id"].unique().tolist()

    # ---- Fetch GTEx --------------------------------------------
    gtex_df = fetch_gtex_median_tpm(gene_ids, tissue=gtex_tissue)
    gtex_df["log2_gtex_tpm"] = np.log2(gtex_df["gtex_median_tpm"] + 1)

    df = df.merge(gtex_df, on="gene_id", how="left")
    logger.info(f"GTEx data retrieved for "
                f"{df['gtex_median_tpm'].notna().sum():,}/{len(df):,} genes")

    # ---- Analysis 1: TCGA tumour vs GTEx normal lung -----------
    both_valid = df[["log2_tpm", "log2_gtex_tpm"]].dropna()
    if not both_valid.empty:
        from scipy import stats
        rho, pval = stats.spearmanr(both_valid["log2_tpm"], both_valid["log2_gtex_tpm"])
        logger.info(f"TCGA tumour vs GTEx normal lung: Spearman ρ = {rho:.3f} (p={pval:.1e})")

    # ---- Analysis 2: AlphaGenome predictions vs GTEx normal ----
    ag_gtex_valid = df[[score_col, "log2_gtex_tpm"]].dropna()
    if not ag_gtex_valid.empty:
        from scipy import stats
        rho2, pval2 = stats.spearmanr(
            ag_gtex_valid[score_col], ag_gtex_valid["log2_gtex_tpm"]
        )
        logger.info(f"AlphaGenome score vs GTEx normal lung: "
                    f"Spearman ρ = {rho2:.3f} (p={pval2:.1e})")

    # ---- Figures -----------------------------------------------
    _plot_tcga_vs_gtex(df, top_n, tpm_col, figures_dir)
    _plot_heatmap(df, score_col, top_n, figures_dir)

    save_parquet(df, out_path)
    return df


# ------------------------------------------------------------------ #
# Figures                                                              #
# ------------------------------------------------------------------ #

def _plot_tcga_vs_gtex(df: pd.DataFrame, top_n: int, tpm_col: str,
                        figures_dir: Path) -> None:
    """Scatter: TCGA tumour TPM vs GTEx normal lung TPM."""
    plot_df = df[["gene_name", "log2_tpm", "log2_gtex_tpm"]].dropna()
    if plot_df.empty:
        logger.warning("No data for TCGA vs GTEx scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(plot_df["log2_gtex_tpm"], plot_df["log2_tpm"],
               alpha=0.25, s=10, c="steelblue", edgecolors="none")

    # Highlight genes most discordant (highly expressed in GTEx but low in tumour)
    plot_df["delta"] = plot_df["log2_gtex_tpm"] - plot_df["log2_tpm"]
    discordant = plot_df.nlargest(top_n, "delta")
    ax.scatter(discordant["log2_gtex_tpm"], discordant["log2_tpm"],
               s=20, c="tomato", edgecolors="none", alpha=0.8,
               label=f"Tumour-suppressed candidates (n={top_n})")
    for _, r in discordant.head(10).iterrows():
        ax.annotate(str(r["gene_name"]), (r["log2_gtex_tpm"], r["log2_tpm"]),
                    fontsize=7, color="darkred", xytext=(4, 4),
                    textcoords="offset points")

    lim_min = min(plot_df[["log2_tpm", "log2_gtex_tpm"]].min())
    lim_max = max(plot_df[["log2_tpm", "log2_gtex_tpm"]].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.8, alpha=0.5)

    ax.set_xlabel("log₂(GTEx Lung TPM + 1) [normal]", fontsize=11)
    ax.set_ylabel("log₂(TCGA TPM + 1) [tumour]", fontsize=11)
    ax.set_title("Tumour vs Normal Lung Expression\n(TCGA-LUAD vs GTEx Lung)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = figures_dir / "gtex_vs_tcga_tpm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def _plot_heatmap(df: pd.DataFrame, score_col: str, top_n: int,
                   figures_dir: Path) -> None:
    """Heatmap of top-N genes: AlphaGenome score | TCGA TPM | GTEx TPM."""
    cols_needed = [score_col, "log2_tpm", "log2_gtex_tpm", "gene_name"]
    plot_df = df[cols_needed].dropna().copy()
    if plot_df.empty:
        logger.warning("No complete data for heatmap.")
        return

    # Top genes by AlphaGenome suppression score (most negative)
    top = plot_df.nsmallest(top_n, score_col).set_index("gene_name")
    heat_data = top[[score_col, "log2_tpm", "log2_gtex_tpm"]].rename(columns={
        score_col: "AG ∆RNA-seq",
        "log2_tpm": "TCGA log₂TPM",
        "log2_gtex_tpm": "GTEx log₂TPM",
    })
    # Z-score normalise for visual comparability
    heat_z = (heat_data - heat_data.mean()) / heat_data.std()

    fig, ax = plt.subplots(figsize=(6, max(4, top_n // 3)))
    sns.heatmap(
        heat_z, ax=ax, cmap="RdBu_r", center=0,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Z-score"},
    )
    ax.set_title(
        f"Top {top_n} AlphaGenome-Suppressed Genes\n"
        "(ranked by ∆RNA-seq score)",
        fontsize=11,
    )
    ax.set_xlabel("")
    plt.tight_layout()
    path = figures_dir / "gtex_predicted_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GTEx context analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_file"], cfg["logging"]["level"])
    run(cfg)

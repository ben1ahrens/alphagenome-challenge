"""
Exploratory Analysis Notebook – AlphaGenome Serova Challenge
============================================================
Convert to notebook:  jupytext --to notebook notebooks/exploratory_analysis.py
Or run interactively: jupyter notebook

This notebook walks through:
  1. Loading and exploring the VCF and RNA data
  2. Running AlphaGenome on a small subset (example variant)
  3. Visualising the results
  4. Statistical analysis
  5. GTEx contextualisation

Requirements: run setup.sh first and set ALPHA_GENOME_API_KEY.
"""
# %% [markdown]
# # AlphaGenome × TCGA-LUAD Exploratory Analysis
# **Serova Tech Challenge — Benjamin Ahrens**

# %% [markdown]
# ## 0. Setup

# %%
import os, sys
from pathlib import Path
sys.path.insert(0, "../src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
# Load API key from .env
from dotenv import load_dotenv
load_dotenv("../.env")
API_KEY = os.environ.get("ALPHA_GENOME_API_KEY", "")
if not API_KEY:
    print("⚠️  ALPHA_GENOME_API_KEY not set — AlphaGenome cells will be skipped")

# %%
# Load config
from utils import load_config
cfg = load_config("../config.yaml")
RESULTS = Path("../results")

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False})

# %% [markdown]
# ## 1. Explore the VCF Data

# %%
# Step 1 must be run first to produce this file
vcf_df = pd.read_parquet(RESULTS / "vcf_parsed.parquet")
print(f"Shape: {vcf_df.shape}")
vcf_df.head()

# %%
# Variant consequence breakdown
cons_counts = (
    vcf_df.dropna(subset=["consequence"])
    .assign(main_csq=lambda d: d["consequence"].str.split("&").str[0])
    .groupby("main_csq")["variant_id"].nunique()
    .sort_values(ascending=False)
)
fig, ax = plt.subplots(figsize=(9, 4))
cons_counts.head(12).plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Number of unique variants")
ax.set_title("TCGA-LUAD Somatic Variant Consequences (PASS, VEP)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %%
# Genes hit by most variants
top_genes = vcf_df.groupby("gene_name")["variant_id"].nunique().sort_values(ascending=False)
print("Top 10 genes by variant count:")
print(top_genes.head(10).to_string())

# %% [markdown]
# ## 2. Explore the RNA Expression Data

# %%
rna_df = pd.read_parquet(RESULTS / "rna_expression.parquet")
print(f"Shape: {rna_df.shape}")
rna_df.head()

# %%
# TPM distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Raw TPM
axes[0].hist(np.log2(rna_df["tpm_unstranded"].clip(lower=0.01) + 1), bins=80,
             color="steelblue", edgecolor="none", alpha=0.8)
axes[0].set_xlabel("log₂(TPM + 1)")
axes[0].set_title("RNA Expression Distribution (TCGA-05-4384)")
axes[0].axvline(np.log2(2), color="tomato", linestyle="--", label="TPM=1")
axes[0].legend()

# Expressed vs silent
expr_counts = rna_df["is_expressed"].value_counts()
axes[1].pie(expr_counts.values,
            labels=[f"Silent (TPM<1)\n{expr_counts[False]:,}",
                    f"Expressed (TPM≥1)\n{expr_counts[True]:,}"],
            colors=["#d9534f", "#5bc0de"], startangle=90, autopct="%1.0f%%")
axes[1].set_title("Expression Status (TPM threshold = 1)")
plt.tight_layout()
plt.show()

# %%
# Gene type breakdown
fig, ax = plt.subplots(figsize=(8, 4))
(
    rna_df.groupby("gene_type")["is_expressed"]
    .agg(["sum", "count"])
    .assign(pct=lambda d: 100 * d["sum"] / d["count"])
    .sort_values("count", ascending=False)
    .head(8)["pct"]
    .sort_values()
    .plot(kind="barh", ax=ax, color="steelblue")
)
ax.set_xlabel("% expressed (TPM ≥ 1)")
ax.set_title("Expression Rate by Gene Type")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. AlphaGenome — Single Variant Example
#
# This section demonstrates AlphaGenome on one variant. The full batch run
# is in `src/3_run_alphagenome.py`.

# %%
if not API_KEY:
    print("Skipping AlphaGenome demo — API key not set")
else:
    from alphagenome.data import genome as ag_genome
    from alphagenome.models import dna_client, variant_scorers

    model = dna_client.create(API_KEY)

    # Pick a representative missense variant from the VCF
    example_var = (
        vcf_df[vcf_df["consequence"].str.contains("missense", na=False)]
        .iloc[0]
    )
    print(f"Example variant: {example_var['variant_id']}")
    print(f"Gene: {example_var['gene_name']} ({example_var['gene_id']})")
    print(f"Consequence: {example_var['consequence']}")

    variant = ag_genome.Variant(
        chromosome=example_var["chrom"],
        position=int(example_var["pos"]),
        reference_bases=example_var["ref"],
        alternate_bases=example_var["alt"],
        name=example_var["variant_id"],
    )
    seq_len = dna_client.SUPPORTED_SEQUENCE_LENGTHS["SEQUENCE_LENGTH_1MB"]
    interval = variant.reference_interval.resize(seq_len)

    scorers = [variant_scorers.RNASeqScorer(), variant_scorers.CAGEScorer()]
    print("\nScoring variant with AlphaGenome (RNA-seq + CAGE)…")
    result = model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=scorers,
        organism="human",
    )
    df_ex = variant_scorers.tidy_scores([result])
    print(df_ex)

# %% [markdown]
# ## 4. Compare AlphaGenome Predictions with Observed Expression

# %%
# Requires steps 3–5 to have been run
try:
    merged = pd.read_parquet(RESULTS / "merged.parquet")
    print(f"Merged table: {len(merged):,} genes")
    merged.head()
except FileNotFoundError:
    print("results/merged.parquet not found — run the full pipeline first.")
    merged = None

# %%
if merged is not None:
    score_col = f"ag_score_{cfg['aggregation']['method']}"

    # Spearman correlation
    valid = merged[[score_col, "log2_tpm"]].dropna()
    rho, pval = stats.spearmanr(valid[score_col], valid["log2_tpm"])
    print(f"Spearman ρ = {rho:.3f}  (p = {pval:.2e},  n = {len(valid):,})")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged[score_col], merged["log2_tpm"],
               alpha=0.3, s=10, c="steelblue", edgecolors="none")
    ax.set_xlabel(f"AlphaGenome ∆RNA-seq score")
    ax.set_ylabel("log₂(TPM + 1) [observed]")
    ax.set_title(f"AlphaGenome vs Observed Expression\n(ρ = {rho:.3f}, p = {pval:.1e})")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.axhline(1, color="tomato", lw=0.8, ls="--", label="TPM=1")
    ax.legend()
    plt.tight_layout()
    plt.show()

# %%
if merged is not None:
    # TPM-stratified boxplot
    plot_df = merged[[score_col, "is_expressed"]].dropna().copy()
    plot_df["Group"] = plot_df["is_expressed"].map(
        {True: "Expressed (TPM≥1)", False: "Silent (TPM<1)"}
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(data=plot_df, x="Group", y=score_col, ax=ax,
                palette={"Expressed (TPM≥1)": "steelblue", "Silent (TPM<1)": "tomato"},
                flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title("AlphaGenome Score by Expression Group")
    ax.set_xlabel("")
    ax.set_ylabel(f"AlphaGenome score ({score_col})")
    plt.tight_layout()
    plt.show()

    # Mann–Whitney test
    expr = plot_df.loc[plot_df["is_expressed"], score_col]
    silent = plot_df.loc[~plot_df["is_expressed"], score_col]
    u_stat, u_pval = stats.mannwhitneyu(expr, silent, alternative="two-sided")
    print(f"Mann–Whitney U = {u_stat:.0f}, p = {u_pval:.2e}")
    print(f"Median score — Expressed: {expr.median():.4f}, Silent: {silent.median():.4f}")

# %% [markdown]
# ## 5. GTEx Context

# %%
try:
    gtex_df = pd.read_parquet(RESULTS / "gtex_context.parquet")
    print(f"GTEx-merged table: {len(gtex_df):,} genes")
except FileNotFoundError:
    print("results/gtex_context.parquet not found — run step 6 first.")
    gtex_df = None

# %%
if gtex_df is not None and "gtex_median_tpm" in gtex_df.columns:
    valid = gtex_df[["log2_tpm", "log2_gtex_tpm"]].dropna()
    rho_gtex, _ = stats.spearmanr(valid["log2_tpm"], valid["log2_gtex_tpm"])
    print(f"Spearman ρ (TCGA tumour vs GTEx normal lung) = {rho_gtex:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(valid["log2_gtex_tpm"], valid["log2_tpm"],
               alpha=0.2, s=8, c="steelblue", edgecolors="none")
    ax.set_xlabel("log₂(GTEx Lung TPM + 1)")
    ax.set_ylabel("log₂(TCGA TPM + 1)")
    ax.set_title(f"Normal Lung vs Tumour Expression\n(ρ = {rho_gtex:.3f})")
    lim = [min(valid.min()), max(valid.max())]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Key Findings Summary

# %%
print("=" * 55)
print("KEY FINDINGS")
print("=" * 55)
if merged is not None:
    valid = merged[[score_col, "log2_tpm"]].dropna()
    rho, pval = stats.spearmanr(valid[score_col], valid["log2_tpm"])
    from sklearn.metrics import roc_auc_score
    auc_df = merged[[score_col, "is_expressed"]].dropna()
    if auc_df["is_expressed"].nunique() == 2:
        auc = roc_auc_score(auc_df["is_expressed"], auc_df[score_col])
    else:
        auc = float("nan")

    print(f"\nAlphaGenome vs Observed TPM:")
    print(f"  Spearman ρ = {rho:.3f}  (p = {pval:.2e})")
    print(f"  ROC-AUC   = {auc:.3f}")
    print(f"\nGenes analysed: {len(merged):,}")
    print(f"  Expressed (TPM≥1): {merged['is_expressed'].sum():,}")
    print(f"  Predicted suppressed: {(merged[score_col] < 0).sum():,}")
    print(f"  Both (suppressed & silent): "
          f"{((merged[score_col] < 0) & (~merged['is_expressed'])).sum():,}")

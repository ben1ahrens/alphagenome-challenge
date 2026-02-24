"""
1_parse_vcf.py – Parse and filter the TCGA-LUAD somatic VCF.

Input:  data/raw/VCF_File.gz  (tar.gz containing the inner VCF.gz + .tbi)
Output: results/vcf_parsed.parquet

Key steps:
  1. Extract the inner VCF.gz from the tar archive.
  2. Read all PASS variants.
  3. Parse VEP CSQ annotations to extract gene-level information.
  4. Save a tidy per-variant table ready for AlphaGenome scoring.

Notes:
  - The VCF is GRCh38 (hg38) with chromosomes prefixed "chr".
  - VEP CSQ field contains multiple transcripts per variant; we retain the
    first canonical/MANE-Select transcript per gene per variant.
  - Very long indels (> max_indel_length bp) are dropped; AlphaGenome handles
    SNVs and short indels most reliably.
"""

from __future__ import annotations

import gzip
import io
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd
from loguru import logger

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, parse_csq_header, parse_csq_entry,
    strip_version, save_parquet, setup_logging, ensure_dir,
)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(config: dict) -> pd.DataFrame:
    vcf_gz_path = Path(config["data"]["vcf_gz"])
    inner_vcf_path = config["vcf"]["vcf_inner_path"]
    pass_only = config["vcf"]["filter_pass_only"]
    max_indel = config["vcf"]["max_indel_length"]
    results_dir = ensure_dir(config["output"]["results_dir"])
    out_path = results_dir / "vcf_parsed.parquet"

    logger.info(f"Reading archive: {vcf_gz_path}")

    # ---- Step 1: Extract inner VCF.gz from tar.gz ----------------
    with tarfile.open(vcf_gz_path, "r:gz") as tar:
        member = tar.getmember(inner_vcf_path)
        vcf_bytes = tar.extractfile(member).read()

    logger.info(f"Extracted inner VCF ({len(vcf_bytes)/1e6:.1f} MB)")

    # ---- Step 2: Parse VCF header & variants ---------------------
    header_lines: list[str] = []
    records: list[dict] = []

    with gzip.open(io.BytesIO(vcf_bytes), "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("##"):
                header_lines.append(line)
                continue
            if line.startswith("#CHROM"):
                col_names = line.lstrip("#").split("\t")
                continue

            parts = line.split("\t")
            row = dict(zip(col_names, parts))

            # Filter: PASS only
            if pass_only and row.get("FILTER", "") != "PASS":
                continue

            chrom = row["CHROM"]
            pos = int(row["POS"])
            ref = row["REF"]
            alt_field = row["ALT"]

            # Handle multi-allelic sites: take first ALT
            alt = alt_field.split(",")[0]

            # Filter very long indels
            max_len = max(len(ref), len(alt))
            if max_len > max_indel:
                logger.debug(f"Skipping long indel at {chrom}:{pos} (len={max_len})")
                continue

            records.append({
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "filter": row.get("FILTER", ""),
                "info": row.get("INFO", ""),
                "variant_id": f"{chrom}_{pos}_{ref}_{alt}_b38",
            })

    logger.info(f"Read {len(records):,} PASS variants (after indel filter)")

    df = pd.DataFrame(records)

    # ---- Step 3: Parse VEP CSQ annotations ----------------------
    csq_fields = parse_csq_header(header_lines)
    if not csq_fields:
        logger.warning("Could not parse CSQ field names from VCF header. "
                       "Gene annotation will be empty.")
    else:
        logger.info(f"VEP CSQ has {len(csq_fields)} fields: {csq_fields[:5]}…")

    gene_rows: list[dict] = []

    for _, row in df.iterrows():
        info = row["info"]
        csq_match = re.search(r"CSQ=([^;]+)", info)
        if not csq_match:
            gene_rows.append({
                "variant_id": row["variant_id"],
                "gene_id": "",
                "gene_name": "",
                "consequence": "",
                "is_canonical": False,
            })
            continue

        transcripts = parse_csq_entry(csq_match.group(1), csq_fields)

        # Prefer MANE_SELECT (marked as canonical) transcripts
        canonical = [t for t in transcripts
                     if t.get("CANONICAL", "") == "YES" or t.get("MANE_SELECT", "")]
        chosen = canonical if canonical else transcripts

        # Deduplicate by gene
        seen_genes: set[str] = set()
        for t in chosen:
            gid = strip_version(t.get("Gene", ""))
            if not gid or gid in seen_genes:
                continue
            seen_genes.add(gid)
            gene_rows.append({
                "variant_id": row["variant_id"],
                "gene_id": gid,
                "gene_name": t.get("SYMBOL", ""),
                "consequence": t.get("Consequence", ""),
                "is_canonical": t.get("CANONICAL", "") == "YES",
            })

    df_genes = pd.DataFrame(gene_rows)

    # ---- Step 4: Merge variants with gene annotations -----------
    df_merged = df.merge(df_genes, on="variant_id", how="left")
    df_merged["gene_id"] = df_merged["gene_id"].fillna("")
    df_merged["gene_name"] = df_merged["gene_name"].fillna("")

    # Tidy types
    df_merged["pos"] = df_merged["pos"].astype(int)

    logger.info(f"Final table: {len(df_merged):,} rows "
                f"({df_merged['variant_id'].nunique():,} unique variants, "
                f"{df_merged['gene_id'].nunique():,} unique genes)")

    # ---- Step 5: Summary stats ----------------------------------
    _print_summary(df_merged)

    # ---- Save ---------------------------------------------------
    save_parquet(df_merged, out_path)
    return df_merged


def _print_summary(df: pd.DataFrame) -> None:
    """Log a brief summary of the parsed VCF."""
    n_variants = df["variant_id"].nunique()
    n_genes = df[df["gene_id"] != ""]["gene_id"].nunique()
    n_intergenic = (df["gene_id"] == "").sum()

    # Consequence breakdown
    consequences = (
        df.dropna(subset=["consequence"])
        .groupby(df["consequence"].str.split("&").str[0])
        ["variant_id"].nunique()
        .sort_values(ascending=False)
        .head(8)
    )

    logger.info("=== VCF Summary ===")
    logger.info(f"  Unique PASS variants : {n_variants:,}")
    logger.info(f"  Genes annotated      : {n_genes:,}")
    logger.info(f"  Intergenic variants  : {n_intergenic:,}")
    logger.info("  Top consequences:")
    for cons, n in consequences.items():
        logger.info(f"    {cons:<40} {n:>5}")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCGA-LUAD somatic VCF")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_file"], cfg["logging"]["level"])
    run(cfg)

"""
3_run_alphagenome.py – Score somatic variants with the AlphaGenome API.

Input:  results/vcf_parsed.parquet  (from step 1)
Output: results/alphagenome_scores.parquet

AlphaGenome API overview
------------------------
The model takes a 1 Mb window of DNA sequence centred on a variant and predicts
thousands of functional genomic tracks.  For variant scoring, we compare predictions
under the reference (REF) and alternate (ALT) alleles.

Scorers used
------------
  RNA-SEQ  : Predicted RNA-seq coverage change (primary expression signal)
  CAGE     : Predicted CAGE signal change (transcription initiation proxy)
  ATAC     : Predicted ATAC-seq change (chromatin accessibility)

Tissue
------
  UBERON:0002048 (lung) – the closest normal-tissue equivalent to LUAD.

Variant scoring strategy
------------------------
  For each PASS variant we call model.score_variant() with a 1 MB interval.
  AlphaGenome internally runs REF and ALT sequences through the model and
  returns per-track ∆scores.  The variant_scorers.tidy_scores() helper
  converts this to a tidy DataFrame.

Rate limiting
-------------
  The free API has per-request rate limits.  We sleep `rate_limit_sleep_s`
  between calls and retry up to `max_retries` times with exponential back-off.

Caching
-------
  Completed results are checkpointed to a local Parquet file after every
  `checkpoint_every` variants so the run can be resumed if interrupted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, get_api_key, save_parquet, load_parquet,
    setup_logging, ensure_dir, retry_with_backoff,
)

CHECKPOINT_EVERY = 50  # Save progress every N variants


# ------------------------------------------------------------------ #
# AlphaGenome imports (deferred so the module can be imported without
# the SDK installed, e.g. for testing).                               #
# ------------------------------------------------------------------ #

def _import_alphagenome():
    try:
        from alphagenome.data import genome as ag_genome
        from alphagenome.models import dna_client, variant_scorers, dna_model
        return ag_genome, dna_client, variant_scorers, dna_model
    except ImportError as e:
        raise ImportError(
            "AlphaGenome SDK not found. Install it with:\n"
            "  git clone https://github.com/google-deepmind/alphagenome.git\n"
            "  pip install -e ./alphagenome_sdk\n"
            f"Original error: {e}"
        ) from e


# ------------------------------------------------------------------ #
# Scorer factory                                                       #
# ------------------------------------------------------------------ #

def _build_scorers(scorer_cfg: dict, variant_scorers_mod, organism: str = "human") -> list:
    """Build the list of AlphaGenome variant scorer objects from config."""
    vs = variant_scorers_mod

    # Select the scorers most relevant to expression prediction
    # RNA_SEQ: gene-level log fold change (primary expression signal)
    # RNA_SEQ_ACTIVE: whether gene is actively transcribed
    # CAGE: transcription initiation proxy
    # ATAC: chromatin accessibility
    keys = ["RNA_SEQ", "RNA_SEQ_ACTIVE", "CAGE", "ATAC"]
    scorers = [vs.RECOMMENDED_VARIANT_SCORERS[k] for k in keys]
    logger.info(f"Using scorers: {keys}")
    return scorers


# ------------------------------------------------------------------ #
# Checkpoint helpers                                                   #
# ------------------------------------------------------------------ #

def _load_checkpoint(path: Path) -> set[str]:
    """Return set of variant_ids already scored (from checkpoint file)."""
    if path.exists():
        df = pd.read_parquet(path)
        done = set(df["variant_id"].unique())
        logger.info(f"Checkpoint: {len(done):,} variants already scored, resuming…")
        return done
    return set()


def _append_checkpoint(path: Path, new_rows: list[dict]) -> None:
    """Append new rows to checkpoint Parquet file."""
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(config: dict) -> pd.DataFrame:
    results_dir = ensure_dir(config["output"]["results_dir"])
    out_path = results_dir / "alphagenome_scores.parquet"
    checkpoint_path = results_dir / "_checkpoint_alphagenome.parquet"

    ag_cfg = config["alphagenome"]
    seq_len_key = f"SEQUENCE_LENGTH_{ag_cfg['sequence_length']}"
    sleep_s = ag_cfg["rate_limit_sleep_s"]
    max_retries = ag_cfg["max_retries"]

    # Load AlphaGenome first so dna_model is available before use
    ag_genome, dna_client, variant_scorers, dna_model = _import_alphagenome()
    organism = dna_model.Organism.HOMO_SAPIENS

    # Load variants
    vcf_df = load_parquet(results_dir / "vcf_parsed.parquet")
    variants_to_score = (
        vcf_df[["variant_id", "chrom", "pos", "ref", "alt"]]
        .drop_duplicates("variant_id")
        .reset_index(drop=True)
    )
    logger.info(f"Variants to score: {len(variants_to_score):,}")

    api_key = get_api_key(config)
    model = dna_client.create(api_key)
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[seq_len_key]
    scorers = _build_scorers(ag_cfg["scorers"], variant_scorers, organism)

    # Resume from checkpoint
    already_done = _load_checkpoint(checkpoint_path)
    pending = variants_to_score[~variants_to_score["variant_id"].isin(already_done)]
    logger.info(f"Pending variants: {len(pending):,}")

    all_rows: list[dict] = []
    buffer: list[dict] = []

    for i, (_, row) in enumerate(pending.iterrows(), start=1):
        variant_id = row["variant_id"]
        chrom = row["chrom"]
        pos = int(row["pos"])
        ref = row["ref"]
        alt = row["alt"]

        logger.debug(f"[{i}/{len(pending)}] Scoring {variant_id}")

        variant = ag_genome.Variant(
            chromosome=chrom,
            position=pos,
            reference_bases=ref,
            alternate_bases=alt,
            name=variant_id,
        )
        interval = variant.reference_interval.resize(sequence_length)

        def _score():
            return model.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=scorers,
                organism=organism,
            )

        try:
            scores = retry_with_backoff(_score, max_retries=max_retries, sleep_s=sleep_s)
        except Exception as exc:
            logger.error(f"Failed to score {variant_id}: {exc}")
            buffer.append({
                "variant_id": variant_id,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "scorer": "ERROR",
                "score_name": "ERROR",
                "score_value": float("nan"),
                "error": str(exc),
            })
            continue

        tidy = variant_scorers.tidy_scores([scores])
        tidy["variant_id"] = variant_id
        if "scored_interval" in tidy.columns:
            tidy["scored_interval"] = tidy["scored_interval"].astype(str)
        buffer.extend(tidy.to_dict(orient="records"))

        time.sleep(sleep_s)

        if i % CHECKPOINT_EVERY == 0:
            _append_checkpoint(checkpoint_path, buffer)
            all_rows.extend(buffer)
            buffer = []
            logger.info(f"  Checkpoint saved ({i}/{len(pending)} variants done)")

    if buffer:
        _append_checkpoint(checkpoint_path, buffer)
        all_rows.extend(buffer)

    if checkpoint_path.exists():
        df_scores = pd.read_parquet(checkpoint_path)
    else:
        df_scores = pd.DataFrame(all_rows)

    logger.info(f"Scoring complete: {df_scores['variant_id'].nunique():,} variants, "
                f"{len(df_scores):,} score rows")

    save_parquet(df_scores, out_path)
    return df_scores


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaGenome variant scoring")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_file"], cfg["logging"]["level"])
    run(cfg)

"""
utils.py – Shared utilities for the AlphaGenome Serova pipeline.
"""

from __future__ import annotations

import os
import re
import sys
import time
import yaml
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger


# ------------------------------------------------------------------ #
# Logging setup                                                        #
# ------------------------------------------------------------------ #

def setup_logging(log_file: str | None = None, level: str = "INFO") -> None:
    """Configure loguru for console and optional file output."""
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, rotation="10 MB",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")


# ------------------------------------------------------------------ #
# Configuration loading                                                #
# ------------------------------------------------------------------ #

def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration file and merge with environment variables."""
    load_dotenv()
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get_api_key(cfg: dict | None = None) -> str:
    """Return AlphaGenome API key from environment or config."""
    load_dotenv()
    key = os.environ.get("ALPHA_GENOME_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ALPHA_GENOME_API_KEY not set. "
            "Export it or add it to your .env file.\n"
            "Obtain a free key at: https://deepmind.google.com/science/alphagenome"
        )
    return key


# ------------------------------------------------------------------ #
# VCF utilities                                                        #
# ------------------------------------------------------------------ #

# VEP CSQ fields for TCGA LUAD annotation (order from ##INFO CSQ header)
# We parse dynamically from the VCF header in parse_vcf.py, but expose the
# commonly needed field indices as constants for documentation.
VEP_GENE_ID_FIELD = "Gene"       # Ensembl gene ID  (e.g., ENSG00000157764)
VEP_GENE_NAME_FIELD = "SYMBOL"   # gene symbol      (e.g., TP53)
VEP_CONSEQUENCE_FIELD = "Consequence"


def parse_csq_header(vcf_header_lines: list[str]) -> list[str]:
    """
    Extract VEP CSQ sub-field names from VCF header.

    Parameters
    ----------
    vcf_header_lines : list of raw header lines (starting with '##')

    Returns
    -------
    List of CSQ sub-field names in order.
    """
    for line in vcf_header_lines:
        m = re.search(r'##INFO=<ID=CSQ,.*Format: ([^"]+)">', line)
        if m:
            return m.group(1).split("|")
    return []


def parse_csq_entry(csq_string: str, csq_fields: list[str]) -> list[dict]:
    """
    Parse a CSQ INFO field into a list of dicts (one per transcript).

    Parameters
    ----------
    csq_string : raw CSQ value (comma-separated transcripts, pipe-separated fields)
    csq_fields : ordered field names from the VCF header

    Returns
    -------
    List of dicts, one per VEP transcript annotation.
    """
    records = []
    for transcript in csq_string.split(","):
        parts = transcript.split("|")
        # Pad to expected length
        parts += [""] * (len(csq_fields) - len(parts))
        records.append(dict(zip(csq_fields, parts)))
    return records


def strip_version(ensembl_id: str) -> str:
    """
    Remove Ensembl version suffix.
    e.g. 'ENSG00000157764.13'  →  'ENSG00000157764'
    """
    return ensembl_id.split(".")[0]


# ------------------------------------------------------------------ #
# Expression utilities                                                 #
# ------------------------------------------------------------------ #

import numpy as np


def log2_tpm(tpm: pd.Series, pseudo: float = 1.0) -> pd.Series:
    """Return log2(TPM + pseudo)."""
    return np.log2(tpm + pseudo)


def classify_expressed(tpm: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Return boolean Series: True if TPM >= threshold."""
    return tpm >= threshold


# ------------------------------------------------------------------ #
# I/O helpers                                                          #
# ------------------------------------------------------------------ #

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path, verbose: bool = True) -> None:
    """
    Save DataFrame as Parquet if pyarrow/fastparquet is available,
    otherwise fall back to compressed CSV.
    """
    p = Path(path)
    ensure_dir(p.parent)
    try:
        df.to_parquet(p, index=False)
        if verbose:
            logger.info(f"Saved {len(df):,} rows → {p}")
    except ImportError:
        csv_path = p.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        if verbose:
            logger.info(f"Saved {len(df):,} rows → {csv_path} (parquet unavailable; using CSV)")


def load_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load a Parquet file produced by this pipeline.
    Falls back to CSV if the Parquet file does not exist.
    """
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    # Fallback: try CSV
    csv_path = p.with_suffix(".csv.gz")
    if csv_path.exists():
        logger.debug(f"Loading CSV fallback: {csv_path}")
        return pd.read_csv(csv_path, compression="gzip")
    raise FileNotFoundError(
        f"Expected file not found: {p}\n"
        f"(also tried {csv_path})\n"
        "Have you run the previous pipeline steps?"
    )


# ------------------------------------------------------------------ #
# Rate-limited API call wrapper                                        #
# ------------------------------------------------------------------ #

def retry_with_backoff(func, max_retries: int = 3, sleep_s: float = 1.0):
    """
    Call `func()` with exponential back-off on failure.

    Parameters
    ----------
    func : callable (zero-argument lambda)
    max_retries : number of attempts
    sleep_s : base sleep duration (doubles on each retry)
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            wait = sleep_s * (2 ** (attempt - 1))
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {exc}. "
                           f"Retrying in {wait:.1f}s…")
            time.sleep(wait)

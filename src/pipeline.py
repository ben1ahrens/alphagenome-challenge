"""
pipeline.py – Orchestrate the full AlphaGenome Serova pipeline.

Usage
-----
    python src/pipeline.py                        # use default config.yaml
    python src/pipeline.py --config config.yaml
    python src/pipeline.py --steps 1 2 3          # run only specified steps
    python src/pipeline.py --skip-alphagenome      # skip step 3 (e.g. no API key yet)

Steps
-----
    1  parse_vcf           Parse and filter the somatic VCF
    2  load_rna            Clean RNA expression table
    3  run_alphagenome     Score variants with AlphaGenome API  [API key required]
    4  aggregate_scores    Collapse variant scores to gene level
    5  compare_expression  Correlation & classification analysis
    6  gtex_context        GTEx normal lung contextualisation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logging


# Map step number → (module name, function name)
STEP_MAP = {
    1: ("1_parse_vcf", "run"),
    2: ("2_load_rna", "run"),
    3: ("3_run_alphagenome", "run"),
    4: ("4_aggregate_scores", "run"),
    5: ("5_compare_expression", "run"),
    6: ("6_gtex_context", "run"),
}

STEP_LABELS = {
    1: "Parse VCF",
    2: "Load RNA expression",
    3: "Run AlphaGenome",
    4: "Aggregate scores",
    5: "Compare expression",
    6: "GTEx context",
}


def import_step(step_num: int):
    """Dynamically import the run function for a pipeline step."""
    import importlib
    module_name, func_name = STEP_MAP[step_num]
    # Import from the same directory as this file
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).parent / f"{module_name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def main():
    parser = argparse.ArgumentParser(
        description="AlphaGenome Serova Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--steps", nargs="+", type=int,
                        default=list(STEP_MAP.keys()),
                        help="Steps to run (default: all)")
    parser.add_argument("--skip-alphagenome", action="store_true",
                        help="Skip step 3 (AlphaGenome scoring)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["log_file"], cfg["logging"]["level"])

    steps_to_run = sorted(set(args.steps))
    if args.skip_alphagenome and 3 in steps_to_run:
        steps_to_run.remove(3)
        logger.warning("Skipping step 3 (AlphaGenome scoring) as requested.")

    logger.info("=" * 55)
    logger.info("  AlphaGenome Serova Pipeline")
    logger.info(f"  Steps: {steps_to_run}")
    logger.info("=" * 55)

    for step_num in steps_to_run:
        label = STEP_LABELS[step_num]
        logger.info(f"\n{'─' * 50}")
        logger.info(f"  STEP {step_num}: {label}")
        logger.info(f"{'─' * 50}")
        t0 = time.perf_counter()
        try:
            run_fn = import_step(step_num)
            run_fn(cfg)
            elapsed = time.perf_counter() - t0
            logger.info(f"  ✓ Step {step_num} completed in {elapsed:.1f}s")
        except Exception as exc:
            logger.error(f"  ✗ Step {step_num} FAILED: {exc}")
            raise

    logger.info("\n" + "=" * 55)
    logger.info("  Pipeline complete.  Results → results/")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()

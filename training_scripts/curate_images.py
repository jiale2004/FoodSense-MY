#!/usr/bin/env python3
"""Curate class-folder images without modifying the source dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from curation import (
    CurationConfig,
    ImageCurator,
    OpenClipScorer,
    load_manual_decisions,
    train_calibration_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "curation.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate, and semantically curate scraped images"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/curation"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--materialize",
        choices=["none", "hardlink", "copy"],
        default="hardlink",
        help="Create run-scoped status views without changing source images",
    )
    parser.add_argument("--run-id", help="Stable output run ID; defaults to a UTC timestamp")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="Pilot mode: process only the first N sorted images per class",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Run technical validation and deduplication only; unique images go to review",
    )
    parser.add_argument(
        "--decisions-from",
        type=Path,
        help=(
            "Completed pilot run whose accepted/rejected folders are authoritative; "
            "enables calibrated one-queue routing"
        ),
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.98,
        help="Required out-of-fold precision for calibrated auto-acceptance (default: 0.98)",
    )
    parser.add_argument(
        "--calibration-folds",
        type=int,
        default=5,
        help="Stratified cross-validation folds for pilot calibration (default: 5)",
    )
    parser.add_argument(
        "--minimum-auto-accept",
        type=int,
        default=5,
        help="Minimum pilot predictions needed to enable auto-acceptance for a class",
    )
    args = parser.parse_args()

    config = CurationConfig.load(args.config)
    scorer = None
    if config.semantic.enabled and not args.skip_semantic:
        scorer = OpenClipScorer(config.classes, config.semantic)

    manual_decisions = []
    calibration = None
    if args.decisions_from:
        if scorer is None:
            parser.error("--decisions-from requires semantic filtering; remove --skip-semantic")
        manual_decisions = load_manual_decisions(args.decisions_from)
        calibration = train_calibration_model(
            decisions=manual_decisions,
            scorer=scorer,
            source_run=args.decisions_from,
            target_precision=args.target_precision,
            folds=args.calibration_folds,
            minimum_predictions=args.minimum_auto_accept,
            batch_size=config.semantic.batch_size,
        )

    curator = ImageCurator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        scorer=scorer,
        materialize=args.materialize,
        run_id=args.run_id,
        limit_per_class=args.limit_per_class,
        manual_decisions=manual_decisions,
        calibration=calibration,
    )
    _, summary = curator.run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

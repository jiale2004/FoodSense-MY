#!/usr/bin/env python3
"""Copy an approved YOLO checkpoint into data/weights/best.pt for local inference.

Production weights are gitignored. After cloning or pulling a new run, promote
the approved checkpoint before starting uvicorn:

    python training_scripts/promote_weights.py

Default source is the interim v8_n_mg production checkpoint. Override with
--source when promoting a newer approved run.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "dataset3_interim_v8_n_mg"
    / "weights"
    / "best.pt"
)
DEFAULT_DEST = PROJECT_ROOT / "data" / "weights" / "best.pt"
EXPECTED_SHA256 = "f0cda9e12125326f24d61bab789e6e09118855a8fd56cb8b0a96e4eec95ee412"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(source: Path, dest: Path, expect_sha256: str | None) -> int:
    if not source.is_file():
        print(f"ERROR: source checkpoint not found: {source}", file=sys.stderr)
        return 1

    source_hash = sha256_file(source)
    if expect_sha256 and source_hash != expect_sha256:
        print(
            "ERROR: source SHA-256 mismatch.\n"
            f"  expected: {expect_sha256}\n"
            f"  actual:   {source_hash}",
            file=sys.stderr,
        )
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    dest_hash = sha256_file(dest)
    if dest_hash != source_hash:
        print("ERROR: destination hash does not match source after copy.", file=sys.stderr)
        return 1

    print(f"Promoted {source} → {dest}")
    print(f"SHA-256: {dest_hash}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Approved checkpoint to copy (default: interim v8_n_mg best.pt)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Application weights path (default: data/weights/best.pt)",
    )
    parser.add_argument(
        "--expect-sha256",
        default=EXPECTED_SHA256,
        help="Expected source SHA-256 (empty string to skip verification)",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip SHA-256 verification (needed when promoting a new approved run)",
    )
    args = parser.parse_args()

    expect = None if args.skip_hash_check or args.expect_sha256 == "" else args.expect_sha256
    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    dest = args.dest if args.dest.is_absolute() else PROJECT_ROOT / args.dest
    return promote(source, dest, expect)


if __name__ == "__main__":
    raise SystemExit(main())

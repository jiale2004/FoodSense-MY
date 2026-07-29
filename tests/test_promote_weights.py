"""Tests for training_scripts/promote_weights.py."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_scripts"))

from promote_weights import promote  # noqa: E402


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_promote_copies_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "best.pt"
    dest = tmp_path / "out" / "best.pt"
    digest = _write(source, b"checkpoint-bytes")
    assert promote(source, dest, digest) == 0
    assert dest.read_bytes() == b"checkpoint-bytes"


def test_promote_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "best.pt"
    dest = tmp_path / "out" / "best.pt"
    _write(source, b"checkpoint-bytes")
    assert promote(source, dest, "0" * 64) == 1
    assert dest.exists() is False


def test_promote_missing_source(tmp_path: Path) -> None:
    assert promote(tmp_path / "missing.pt", tmp_path / "out.pt", None) == 1

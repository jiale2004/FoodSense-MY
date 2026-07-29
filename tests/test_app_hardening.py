"""Unit tests for upload retention cleanup and vision device / weights policy."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.security import cleanup_uploads
from app.services.vision_service import VisionProcessor


def test_cleanup_uploads_removes_old_and_excess(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    now = time.time()

    old = uploads / "old.jpg"
    mid = uploads / "mid.jpg"
    new = uploads / "new.jpg"
    keep_txt = uploads / "readme.txt"
    for path, age in ((old, 100_000), (mid, 10), (new, 1)):
        path.write_bytes(b"jpeg")
        # st_mtime tweak
        import os

        os.utime(path, (now - age, now - age))
    keep_txt.write_text("ignore")

    settings = Settings(
        upload_retention_hours=1.0,  # 3600 seconds
        upload_max_files=1,
    )
    # Point cleanup at tmp uploads via a thin proxy
    class Proxy:
        upload_retention_hours = settings.upload_retention_hours
        upload_max_files = settings.upload_max_files

        @property
        def uploads_dir(self):
            return uploads

    result = cleanup_uploads(Proxy())  # type: ignore[arg-type]
    assert result["removed_by_age"] == 1
    assert old.exists() is False
    assert mid.exists() or new.exists()
    # max_files=1 keeps only the newest remaining image
    remaining_images = [p for p in uploads.iterdir() if p.suffix == ".jpg"]
    assert len(remaining_images) == 1
    assert remaining_images[0].name == "new.jpg"
    assert keep_txt.exists()
    assert result["removed_by_count"] == 1


def test_vision_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr("app.services.vision_service.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "app.services.vision_service.torch.backends.mps.is_available", lambda: True
    )
    settings = Settings(device="auto")
    processor = VisionProcessor(settings)
    assert processor.device == "cuda"


def test_vision_explicit_cpu(monkeypatch) -> None:
    monkeypatch.setattr("app.services.vision_service.torch.cuda.is_available", lambda: True)
    settings = Settings(device="cpu")
    processor = VisionProcessor(settings)
    assert processor.device == "cpu"


def test_vision_missing_weights_fail_hard(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    settings = Settings(model_weights_path=missing, device="cpu")
    processor = VisionProcessor(settings)
    with pytest.raises(FileNotFoundError, match="promote_weights"):
        processor.load_model()

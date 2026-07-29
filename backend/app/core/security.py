"""Upload validation and retention cleanup for prediction images."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import Depends, Header, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


async def validate_upload(file: UploadFile, settings: Settings | None = None) -> None:
    settings = settings or get_settings()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    extension = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {file.content_type}",
        )

    content = await file.read()
    await file.seek(0)

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )


def cleanup_uploads(settings: Settings | None = None) -> dict[str, int]:
    """Remove expired and excess upload files under settings.uploads_dir.

    Age cleanup uses ``upload_retention_hours`` (0 disables). Count cleanup
    keeps the newest ``upload_max_files`` images (0 disables). Returns counts
    of files removed by each policy.
    """
    settings = settings or get_settings()
    uploads_dir = settings.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for path in uploads_dir.iterdir()
        if path.is_file() and path.suffix.lower() in UPLOAD_SUFFIXES
    ]
    removed_by_age = 0
    removed_by_count = 0
    now = time.time()

    if settings.upload_retention_hours > 0:
        max_age_seconds = settings.upload_retention_hours * 3600.0
        survivors: list[Path] = []
        for path in files:
            try:
                age = now - path.stat().st_mtime
            except OSError:
                survivors.append(path)
                continue
            if age > max_age_seconds:
                try:
                    path.unlink()
                    removed_by_age += 1
                except OSError:
                    logger.warning("Failed to delete expired upload: %s", path)
            else:
                survivors.append(path)
        files = survivors

    if settings.upload_max_files > 0 and len(files) > settings.upload_max_files:
        files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files_sorted[settings.upload_max_files :]:
            try:
                path.unlink()
                removed_by_count += 1
            except OSError:
                logger.warning("Failed to delete excess upload: %s", path)

    if removed_by_age or removed_by_count:
        logger.info(
            "Upload cleanup removed %d by age, %d by count under %s",
            removed_by_age,
            removed_by_count,
            uploads_dir,
        )
    return {"removed_by_age": removed_by_age, "removed_by_count": removed_by_count}


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    # Settings must come from Depends(get_settings). A bare
    # `settings: Settings | None = None` makes FastAPI treat Settings as a
    # request-body field and breaks JSON endpoints like /api/chat (422).
    if not settings.api_key_enabled:
        return

    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

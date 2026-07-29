"""API smoke tests for health, classes, predict, and upload validation.

Vision inference is stubbed so the suite stays CPU-light and does not require
a live Ultralytics predict. Knowledge-base lookup uses the real JSON.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.models.schemas import BoundingBox, DetectionResult
from app.services.data_service import KnowledgeRetriever, get_data_service
from app.services.llm_service import AdvisoryGenerator, get_llm_service
from app.services.vision_service import VisionProcessor, get_vision_service, reset_vision_service


class StubVision(VisionProcessor):
    """VisionProcessor that skips YOLO load and returns a fixed detection."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = object()
        self.model_path = str(settings.model_weights_path)
        self.device = "cpu"
        self._detections = [
            DetectionResult(
                class_name="nasi_lemak",
                confidence=0.91,
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=120),
            )
        ]

    def load_model(self) -> None:
        return None

    def detect(self, image_path: Path) -> list[DetectionResult]:
        return list(self._detections)


class SettingsProxy:
    """Delegate to real Settings while overriding uploads_dir for isolation."""

    def __init__(self, base: Settings, uploads_dir: Path) -> None:
        self._base = base
        self._uploads_dir = uploads_dir

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    @property
    def uploads_dir(self) -> Path:
        return self._uploads_dir


def _jpeg_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:, :] = (40, 120, 200)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    get_settings.cache_clear()
    reset_vision_service()

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    base = get_settings()
    proxy = SettingsProxy(base, uploads)
    vision = StubVision(base)
    data = KnowledgeRetriever(base)
    data.load()
    llm = AdvisoryGenerator(base)

    def _settings() -> SettingsProxy:
        return proxy

    monkeypatch.setattr("app.main.get_vision_service", lambda: vision)
    monkeypatch.setattr("app.main.get_data_service", lambda: data)
    monkeypatch.setattr(
        "app.main.cleanup_uploads",
        lambda: {"removed_by_age": 0, "removed_by_count": 0},
    )
    monkeypatch.setattr("app.api.routes.get_settings", _settings)
    monkeypatch.setattr("app.api.routes.cleanup_uploads", lambda settings=None: None)
    monkeypatch.setattr("app.core.security.get_settings", _settings)

    app.dependency_overrides[get_vision_service] = lambda: vision
    app.dependency_overrides[get_data_service] = lambda: data
    app.dependency_overrides[get_llm_service] = lambda: llm

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    get_settings.cache_clear()
    reset_vision_service()


def test_health_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["knowledge_base_loaded"] is True
    assert payload["knowledge_base_entries"] == 6
    assert payload["device"] == "cpu"


def test_classes_lists_six_dishes(client: TestClient) -> None:
    response = client.get("/api/classes")
    assert response.status_code == 200
    classes = response.json()["classes"]
    assert set(classes) == {
        "nasi_lemak",
        "roti_canai",
        "char_kuey_teow",
        "chicken_rice",
        "laksa",
        "mee_goreng",
    }


def test_predict_happy_path(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("plate.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["class_name"] == "nasi_lemak"
    assert payload["nutrition"][0]["found_in_kb"] is True
    assert payload["nutrition"][0]["display_name"] == "Nasi Lemak"
    assert "educational purposes" in payload["disclaimer"].lower()
    assert payload["image_url"].startswith("/uploads/")
    assert payload["processing_ms"] >= 0
    assert payload["advisory_text"]


def test_predict_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_predict_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Empty" in response.json()["detail"]


def test_predict_rejects_oversized_file(tmp_path, monkeypatch) -> None:
    get_settings.cache_clear()
    reset_vision_service()

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    get_settings.cache_clear()
    base = get_settings()
    assert base.max_upload_size_mb == 1

    proxy = SettingsProxy(base, uploads)
    vision = StubVision(base)
    data = KnowledgeRetriever(base)
    data.load()
    llm = AdvisoryGenerator(base)

    monkeypatch.setattr("app.main.get_vision_service", lambda: vision)
    monkeypatch.setattr("app.main.get_data_service", lambda: data)
    monkeypatch.setattr(
        "app.main.cleanup_uploads",
        lambda: {"removed_by_age": 0, "removed_by_count": 0},
    )
    monkeypatch.setattr("app.api.routes.get_settings", lambda: proxy)
    monkeypatch.setattr("app.api.routes.cleanup_uploads", lambda settings=None: None)
    monkeypatch.setattr("app.core.security.get_settings", lambda: proxy)
    app.dependency_overrides[get_vision_service] = lambda: vision
    app.dependency_overrides[get_data_service] = lambda: data
    app.dependency_overrides[get_llm_service] = lambda: llm

    oversized = b"\xff\xd8\xff" + (b"x" * (1024 * 1024 + 64))
    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/predict",
                files={"file": ("big.jpg", oversized, "image/jpeg")},
            )
        assert response.status_code == 413
        assert "maximum size" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        reset_vision_service()

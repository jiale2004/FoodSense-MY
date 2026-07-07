import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.config import Settings, get_settings
from app.models.schemas import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "yolo11n.pt"
MAX_IMAGE_DIM = 640


class VisionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model: YOLO | None = None
        self.model_path: str = ""

    def load_model(self) -> None:
        weights_path = self.settings.model_weights_path
        if weights_path.exists():
            self.model_path = str(weights_path)
            logger.info("Loading custom YOLOv11n weights from %s", self.model_path)
        else:
            self.model_path = FALLBACK_MODEL
            logger.warning(
                "Custom weights not found at %s. Falling back to %s (COCO classes).",
                weights_path,
                FALLBACK_MODEL,
            )

        self.model = YOLO(self.model_path)

    def preprocess(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        height, width = image.shape[:2]
        max_dim = max(height, width)
        if max_dim > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def detect(self, image_path: Path) -> list[DetectionResult]:
        if self.model is None:
            raise RuntimeError("Vision model is not loaded.")

        processed = self.preprocess(image_path)
        results = self.model.predict(
            source=processed,
            conf=self.settings.confidence_threshold,
            verbose=False,
        )

        detections: list[DetectionResult] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names or {}
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = names.get(class_id, str(class_id))

            detections.append(
                DetectionResult(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


_vision_service: VisionService | None = None


def get_vision_service() -> VisionService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.core.config import Settings, get_settings
from app.models.schemas import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)

MAX_IMAGE_DIM = 640


class VisionProcessor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model: YOLO | None = None
        self.model_path: str = ""
        self.device: str = self._select_device()

    def _select_device(self) -> str:
        # Prefer explicit setting, else CUDA > MPS > CPU.
        if self.settings.device != "auto":
            return self.settings.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Load the approved six-class YOLO weights onto the selected device.

        Missing weights fail hard. Promote an approved checkpoint first:

            python training_scripts/promote_weights.py
        """
        weights_path = self.settings.model_weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Custom weights not found at {weights_path}. "
                "Promote the approved checkpoint before starting the API:\n"
                "  python training_scripts/promote_weights.py\n"
                "Or set MODEL_WEIGHTS_PATH to an existing .pt file. "
                "COCO pretrained fallback is disabled."
            )

        self.model_path = str(weights_path)
        logger.info("Loading custom YOLOv11n weights from %s", self.model_path)
        self.model = YOLO(self.model_path)
        logger.info("VisionProcessor using device: %s", self.device)

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
            iou=self.settings.iou_threshold,
            device=self.device,
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


_vision_processor: VisionProcessor | None = None


def get_vision_service() -> VisionProcessor:
    global _vision_processor
    if _vision_processor is None:
        _vision_processor = VisionProcessor()
    return _vision_processor


def reset_vision_service() -> None:
    global _vision_processor
    _vision_processor = None

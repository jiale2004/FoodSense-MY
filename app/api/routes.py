import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import get_settings
from app.core.security import validate_upload, verify_api_key
from app.models.schemas import (
    ClassListResponse,
    HealthResponse,
    PredictResponse,
)
from app.services.data_service import get_data_service
from app.services.llm_service import get_llm_service
from app.services.vision_service import get_vision_service

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    vision = get_vision_service()
    data = get_data_service()
    llm = get_llm_service()

    return HealthResponse(
        status="ok",
        model_loaded=vision.model is not None,
        model_path=vision.model_path,
        knowledge_base_loaded=data.is_loaded,
        knowledge_base_entries=data.entry_count,
        llm_provider=settings.llm_provider,
        llm_configured=llm.is_configured,
    )


@router.get("/classes", response_model=ClassListResponse)
async def list_classes() -> ClassListResponse:
    data = get_data_service()
    return ClassListResponse(classes=data.list_classes())


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    settings = get_settings()
    await validate_upload(file)

    uploads_dir = settings.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{extension}"
    save_path = uploads_dir / filename

    content = await file.read()
    save_path.write_bytes(content)

    start = time.perf_counter()
    vision = get_vision_service()
    data = get_data_service()
    llm = get_llm_service()

    detections = vision.detect(save_path)
    class_names = [d.class_name for d in detections]
    nutrition = data.lookup_many(class_names)
    advisory_text = await llm.generate_advisory(detections, nutrition)

    processing_ms = int((time.perf_counter() - start) * 1000)

    return PredictResponse(
        detections=detections,
        nutrition=nutrition,
        advisory_text=advisory_text,
        image_url=f"/uploads/{filename}",
        processing_ms=processing_ms,
    )

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.security import cleanup_uploads
from app.services.data_service import get_data_service
from app.services.vision_service import get_vision_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FoodSense-MY...")
    vision = get_vision_service()
    data = get_data_service()
    vision.load_model()
    data.load()
    cleanup_uploads()
    logger.info("FoodSense-MY ready.")
    yield
    logger.info("Shutting down FoodSense-MY.")


app = FastAPI(
    title="FoodSense-MY",
    description="Malaysian Food Object Detection and Nutritional Advisory System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

app.mount("/uploads", StaticFiles(directory=STATIC_DIR / "uploads"), name="uploads")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

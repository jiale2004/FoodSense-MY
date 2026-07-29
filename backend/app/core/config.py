import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# backend/app/core/config.py -> parents[1] = backend/app, parents[2] = backend,
# parents[3] = repository root. Data, weights, and the .env file live at the
# repository root; uploaded images live inside the backend app package.
APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

TARGET_CLASSES: list[str] = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["openai", "gemini"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None
    # gemini-2.5-* is blocked for many new API keys; flash-lite-latest stays current.
    gemini_model: str = "gemini-flash-lite-latest"

    model_weights_path: Path = Path("data/weights/best.pt")
    knowledge_base_path: Path = Path("data/knowledge_base.json")

    # Calibrated on the dataset3-interim-v5 val split for v8_n_mg
    # (macro-F1 optimum; see runs/detect/dataset3_interim_v8_n_mg_calibration/).
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    max_upload_size_mb: int = 10
    # Retain uploaded prediction images for this many hours; 0 disables age cleanup.
    upload_retention_hours: float = 24.0
    # Keep at most this many upload files; oldest are removed first. 0 disables.
    upload_max_files: int = 200

    api_key_enabled: bool = False
    api_key: str | None = None

    @field_validator("model_weights_path", "knowledge_base_path", mode="before")
    @classmethod
    def resolve_relative_paths(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return APP_DIR / "static" / "uploads"

    def validate_llm_config(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. LLM advisory will use template fallback."
            )
        elif self.llm_provider == "gemini" and not self.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. LLM advisory will use template fallback."
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_llm_config()
    return settings

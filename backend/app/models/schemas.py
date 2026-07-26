from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class NutritionInfo(BaseModel):
    class_name: str
    display_name: str
    calories_kcal: float | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    sodium_mg: float | None = None
    allergens: list[str] = Field(default_factory=list)
    dietary_tags: list[str] = Field(default_factory=list)
    health_notes: str = ""
    found_in_kb: bool = True


class PredictResponse(BaseModel):
    detections: list[DetectionResult]
    nutrition: list[NutritionInfo]
    advisory_text: str
    disclaimer: str
    image_url: str
    processing_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    knowledge_base_loaded: bool
    knowledge_base_entries: int
    llm_provider: str
    llm_configured: bool


class ClassListResponse(BaseModel):
    classes: list[str]


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatContext(BaseModel):
    """Optional last prediction context so the chat can ground answers."""

    detections: list[DetectionResult] = Field(default_factory=list)
    nutrition: list[NutritionInfo] = Field(default_factory=list)
    advisory_text: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    context: ChatContext | None = None


class ChatResponse(BaseModel):
    reply: str
    llm_used: bool
    disclaimer: str

import json
import logging
import re
from pathlib import Path

from app.core.config import Settings, get_settings
from app.models.schemas import NutritionInfo

logger = logging.getLogger(__name__)


def normalize_class_name(name: str) -> str:
    """Normalize a class name to lowercase underscore format."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


class KnowledgeRetriever:
    """Loads and looks up verified nutritional data from the JSON knowledge base."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._entries: dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        """Load and cache the knowledge base JSON file."""
        kb_path = self.settings.knowledge_base_path
        if not kb_path.exists():
            logger.warning("Knowledge base not found at %s", kb_path)
            self._entries = {}
            self._loaded = True
            return

        with kb_path.open(encoding="utf-8") as f:
            raw = json.load(f)

        dishes = raw.get("dishes", raw)
        self._entries = {
            normalize_class_name(key): value for key, value in dishes.items()
        }
        self._loaded = True
        logger.info("Loaded %d knowledge base entries from %s", len(self._entries), kb_path)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def list_classes(self) -> list[str]:
        """Return sorted list of known dish class names."""
        return sorted(self._entries.keys())

    def lookup(self, class_name: str) -> NutritionInfo:
        """Look up nutrition info for a single detected class name."""
        key = normalize_class_name(class_name)
        entry = self._entries.get(key)

        if entry is None:
            return NutritionInfo(
                class_name=class_name,
                display_name=class_name.replace("_", " ").title(),
                found_in_kb=False,
                health_notes="No verified nutritional data available for this dish.",
            )

        return NutritionInfo(
            class_name=key,
            display_name=entry.get("display_name", key.replace("_", " ").title()),
            calories_kcal=entry.get("calories_kcal"),
            protein_g=entry.get("protein_g"),
            carbs_g=entry.get("carbs_g"),
            fat_g=entry.get("fat_g"),
            sodium_mg=entry.get("sodium_mg"),
            allergens=entry.get("allergens", []),
            dietary_tags=entry.get("dietary_tags", []),
            health_notes=entry.get("health_notes", ""),
            found_in_kb=True,
        )

    def lookup_many(self, class_names: list[str]) -> list[NutritionInfo]:
        """Look up nutrition info for multiple class names, deduplicated."""
        seen: set[str] = set()
        results: list[NutritionInfo] = []
        for name in class_names:
            key = normalize_class_name(name)
            if key in seen:
                continue
            seen.add(key)
            results.append(self.lookup(name))
        return results


_knowledge_retriever: KnowledgeRetriever | None = None


def get_data_service() -> KnowledgeRetriever:
    """Return the singleton KnowledgeRetriever instance."""
    global _knowledge_retriever
    if _knowledge_retriever is None:
        _knowledge_retriever = KnowledgeRetriever()
    return _knowledge_retriever

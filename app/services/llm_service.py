import json
import logging

from app.core.config import Settings, get_settings
from app.models.schemas import DetectionResult, NutritionInfo

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful Malaysian food nutrition assistant for FoodSense-MY.
Provide clear, friendly, and safe dietary information based ONLY on the verified data provided.
Rules:
- Do NOT invent or estimate nutritional numbers; use only the values given.
- Mention allergens when present.
- Include a brief disclaimer that this is informational, not medical advice.
- Keep the response concise (2-4 short paragraphs).
- If no dishes were detected, explain that and suggest trying a clearer photo."""


class LLMService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def is_configured(self) -> bool:
        if self.settings.llm_provider == "openai":
            return bool(self.settings.openai_api_key)
        return bool(self.settings.gemini_api_key)

    def _build_user_prompt(
        self,
        detections: list[DetectionResult],
        nutrition_entries: list[NutritionInfo],
    ) -> str:
        payload = {
            "detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence, 3)}
                for d in detections
            ],
            "nutrition": [
                {
                    "display_name": n.display_name,
                    "calories_kcal": n.calories_kcal,
                    "protein_g": n.protein_g,
                    "carbs_g": n.carbs_g,
                    "fat_g": n.fat_g,
                    "sodium_mg": n.sodium_mg,
                    "allergens": n.allergens,
                    "dietary_tags": n.dietary_tags,
                    "health_notes": n.health_notes,
                    "found_in_kb": n.found_in_kb,
                }
                for n in nutrition_entries
            ],
        }
        return (
            "Analyze the following Malaysian food detection results and provide a "
            "user-friendly nutritional advisory. Use ONLY the verified data below:\n\n"
            + json.dumps(payload, indent=2)
        )

    def _fallback_advisory(
        self,
        detections: list[DetectionResult],
        nutrition_entries: list[NutritionInfo],
    ) -> str:
        if not detections:
            return (
                "No Malaysian dishes were detected in your image. "
                "Try uploading a clearer, well-lit photo of the food.\n\n"
                "Disclaimer: This information is for educational purposes only and is not medical advice."
            )

        lines = ["Here is a summary based on our verified nutrition database:\n"]
        for entry in nutrition_entries:
            if not entry.found_in_kb:
                lines.append(f"- {entry.display_name}: No verified data available.")
                continue
            macros = []
            if entry.calories_kcal is not None:
                macros.append(f"{entry.calories_kcal:.0f} kcal")
            if entry.protein_g is not None:
                macros.append(f"{entry.protein_g:.0f}g protein")
            if entry.carbs_g is not None:
                macros.append(f"{entry.carbs_g:.0f}g carbs")
            if entry.fat_g is not None:
                macros.append(f"{entry.fat_g:.0f}g fat")
            macro_str = ", ".join(macros) if macros else "macros unavailable"
            allergen_str = (
                f" Allergens: {', '.join(entry.allergens)}." if entry.allergens else ""
            )
            lines.append(f"- {entry.display_name}: {macro_str}.{allergen_str}")
            if entry.health_notes:
                lines.append(f"  {entry.health_notes}")

        lines.append(
            "\nDisclaimer: This information is for educational purposes only and is not medical advice."
        )
        return "\n".join(lines)

    async def _call_openai(self, user_prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        response = await client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content or ""

    async def _call_gemini(self, user_prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )
        response = await model.generate_content_async(user_prompt)
        return response.text or ""

    async def generate_advisory(
        self,
        detections: list[DetectionResult],
        nutrition_entries: list[NutritionInfo],
    ) -> str:
        if not self.is_configured:
            return self._fallback_advisory(detections, nutrition_entries)

        user_prompt = self._build_user_prompt(detections, nutrition_entries)

        try:
            if self.settings.llm_provider == "openai":
                return await self._call_openai(user_prompt)
            return await self._call_gemini(user_prompt)
        except Exception:
            logger.exception("LLM call failed; using template fallback.")
            return self._fallback_advisory(detections, nutrition_entries)


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

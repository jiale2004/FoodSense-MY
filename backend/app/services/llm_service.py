import json
import logging

from app.core.config import Settings, get_settings
from app.models.schemas import DetectionResult, NutritionInfo

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This information is for educational purposes only and is not medical advice. "
    "Consult a healthcare professional for dietary guidance."
)

SYSTEM_PROMPT = """You are a text-formatting assistant for FoodSense-MY.
Your ONLY role is to summarize the verified JSON data provided below into clear, friendly language.

STRICT RULES:
- Do NOT invent, estimate, or modify any calorie counts, macros, or nutritional numbers.
- Do NOT generate ingredient lists or medical advice beyond what is in the data.
- Do NOT add information that is not present in the provided JSON.
- Summarize ONLY the fields given: display_name, calories, macros, allergens, dietary_tags, health_notes.
- Keep the response concise (2-3 short paragraphs).
- If no dishes were detected, state that and suggest trying a clearer photo."""

CHAT_SYSTEM_PROMPT = """You are FoodSense-MY's helpful chat assistant for Malaysian food detection and nutrition.

Users may chat freely at any time — a photo upload is optional, not required.

STRICT RULES:
- Answer questions about Malaysian dishes, calories/macros, allergens, dietary tags, and how to use the app.
- If a knowledge-base JSON is provided, use those verified numbers. Do NOT invent calories or macros.
- If a latest-scan (meal context) JSON is provided, prefer it for questions about "my meal" / "this photo".
- If a number is not in the provided JSON, say you do not have verified data rather than guessing.
- Do NOT give medical diagnosis or personalized medical advice. Suggest consulting a professional when relevant.
- Keep replies concise (a few short paragraphs or bullets).
- If the user asks something unrelated, briefly redirect to food/nutrition topics.
- Never refuse to answer just because no image was uploaded."""


class AdvisoryGenerator:
    """Generates nutritional advisory text using an LLM as a strict formatting layer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def is_configured(self) -> bool:
        """Return True if the active LLM provider has an API key configured."""
        if self.settings.llm_provider == "openai":
            return bool(self.settings.openai_api_key)
        return bool(self.settings.gemini_api_key)

    def _build_user_prompt(
        self,
        detections: list[DetectionResult],
        nutrition_entries: list[NutritionInfo],
    ) -> str:
        """Build the fixed prompt template injecting verified JSON data."""
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
            "Format the following verified Malaysian food detection data into a "
            "user-friendly summary. Use ONLY the data below — do not add or modify any values:\n\n"
            + json.dumps(payload, indent=2)
        )

    def _fallback_advisory(
        self,
        detections: list[DetectionResult],
        nutrition_entries: list[NutritionInfo],
    ) -> str:
        """Generate a template-based advisory when LLM is unavailable."""
        if not detections:
            return (
                "No Malaysian dishes were detected in your image. "
                "Try uploading a clearer, well-lit photo of the food."
            )

        lines = ["Summary based on our verified nutrition database:\n"]
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

        return "\n".join(lines)

    async def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API for advisory text generation."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        response = await client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content or ""

    async def _call_gemini(self, user_prompt: str) -> str:
        """Call Google Gemini API for advisory text generation."""
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
        """Generate advisory text using LLM or template fallback."""
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

    def _chat_fallback(
        self,
        message: str,
        context_json: str | None,
        *,
        reason: str = "unconfigured",
    ) -> str:
        """Deterministic reply when the LLM is unavailable."""
        if reason == "error":
            base = (
                "The AI assistant hit an error (often an unavailable Gemini model id). "
                "Check GEMINI_MODEL in .env — try gemini-flash-lite-latest — then restart."
            )
        else:
            base = (
                "I can help with Malaysian dish nutrition once an LLM key is configured "
                "(set GEMINI_API_KEY or OPENAI_API_KEY in .env and restart the server)."
            )
        if context_json:
            return (
                f"{base}\n\nYour latest detection context is available on the next "
                "successful AI reply.\n\n"
                f"Your message: {message}"
            )
        return (
            f"{base}\n\n"
            "Tip: upload a meal photo first so I can ground answers in verified nutrition data."
        )

    async def generate_chat_reply(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        context_payload: dict | None = None,
        knowledge_base: dict | None = None,
    ) -> tuple[str, bool]:
        """Return (reply, llm_used) for the chat widget."""
        context_json = (
            json.dumps(context_payload, indent=2) if context_payload else None
        )
        kb_json = json.dumps(knowledge_base, indent=2) if knowledge_base else None
        if not self.is_configured:
            return self._chat_fallback(message, context_json, reason="unconfigured"), False

        parts: list[str] = []
        if kb_json:
            parts.append(
                "Verified nutrition knowledge base for supported dishes "
                "(use these numbers; do not invent macros):\n"
                + kb_json
            )
        if context_json:
            parts.append(
                "Latest meal detection from the user's photo "
                "(prefer this for questions about their current scan):\n"
                + context_json
            )
        else:
            parts.append(
                "No photo has been uploaded yet. Still answer general Malaysian "
                "food and nutrition questions using the knowledge base above."
            )
        parts.append(f"User question:\n{message}")
        user_content = "\n\n".join(parts)

        try:
            if self.settings.llm_provider == "openai":
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=self.settings.openai_api_key)
                messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
                for turn in (history or [])[-12:]:
                    role = turn.get("role")
                    content = turn.get("content")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
                messages.append({"role": "user", "content": user_content})
                response = await client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=700,
                )
                return response.choices[0].message.content or "", True

            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(
                model_name=self.settings.gemini_model,
                system_instruction=CHAT_SYSTEM_PROMPT,
            )
            # Gemini history uses "model" for assistant turns.
            gemini_history = []
            for turn in (history or [])[-12:]:
                role = turn.get("role")
                content = turn.get("content")
                if not content:
                    continue
                if role == "user":
                    gemini_history.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    gemini_history.append({"role": "model", "parts": [content]})
            chat = model.start_chat(history=gemini_history)
            response = await chat.send_message_async(user_content)
            return response.text or "", True
        except Exception:
            logger.exception("Chat LLM call failed; using template fallback.")
            return self._chat_fallback(message, context_json, reason="error"), False


_advisory_generator: AdvisoryGenerator | None = None


def get_llm_service() -> AdvisoryGenerator:
    """Return the singleton AdvisoryGenerator instance."""
    global _advisory_generator
    if _advisory_generator is None:
        _advisory_generator = AdvisoryGenerator()
    return _advisory_generator

import os
import logging
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / '.env')
logger = logging.getLogger(__name__)

class GroqClient:
    """Base client for Groq API with auto-model discovery."""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
        self.client = None
        self.ranker_model = "llama-3.3-70b-versatile"           # Best on Groq
        self.optimizer_model = "llama-3.3-70b-versatile"         # Best on Groq
        self.vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Official vision model
        self._available = False

        if self.api_key:
            try:
                self.client = Groq(
                    api_key=self.api_key,
                    max_retries=5,
                    timeout=120.0
                )
                self._available = True
                self._discover_models()
            except Exception as e:
                logger.error(f"Groq Client Init Error: {e}")

    def _discover_models(self):
        """Find best available models on Groq - always prefer newest."""
        try:
            models = [m.id for m in self.client.models.list().data]

            # Priority: gpt-oss-120b > llama-4 > llama-3.3-70b
            def best_text_model(candidates):
                for preferred in ["gpt-oss-120b", "llama-4", "70b"]:
                    matches = [m for m in candidates if preferred in m.lower()]
                    if matches:
                        return sorted(matches, reverse=True)[0]
                return None

            text_models = [m for m in models if "scout" not in m.lower() and "vision" not in m.lower() and "guard" not in m.lower() and "whisper" not in m.lower()]
            
            best = best_text_model(text_models)
            if best:
                self.ranker_model = best
                self.optimizer_model = best
                logger.info(f"Using Groq Text Model: {best}")

            # Vision: prefer llama-4-scout
            scout_models = [m for m in models if "scout" in m.lower()]
            vision_models = [m for m in models if "vision" in m.lower()]
            best_vision = scout_models or vision_models
            if best_vision:
                self.vision_model = sorted(best_vision, reverse=True)[0]
                logger.info(f"Using Groq Vision Model: {self.vision_model}")

        except Exception as e:
            logger.warning(f"Groq Auto-Discovery failed, using defaults: {e}")

    @property
    def available(self):
        return self._available and self.client is not None

    # Fallback model if primary fails (rate limit, unavailable, etc.)
    FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192"]

    def call(self, system_prompt, user_prompt, model, temperature=0.0, json_mode=False):
        """Centralized API call handler with fallback models."""
        if not self.available:
            logger.error("Groq client not available. API key may be missing or invalid.")
            return None

        # Try primary model first, then fallbacks
        models_to_try = [model] + [m for m in self.FALLBACK_MODELS if m != model]

        for attempt_model in models_to_try:
            try:
                kwargs = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "model": attempt_model,
                    "temperature": temperature,
                    "seed": 42
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    if attempt_model != model:
                        logger.info(f"Groq: succeeded with fallback model {attempt_model}")
                    return content
            except Exception as e:
                logger.error(f"Groq API Error with model '{attempt_model}': {e}")
                continue

        logger.error("Groq: all models failed.")
        return None

    def call_vision(self, user_prompt, base64_image, model=None):
        """Call Groq Vision model with an image."""
        if not self.available: return None
        try:
            target_model = model or self.vision_model
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                model=target_model,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq Vision Error: {e}")
            return None

# Singleton base client
groq_base = GroqClient()

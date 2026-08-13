import os
import logging
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / '.env')
logger = logging.getLogger(__name__)

class GroqClient:
    """Base client for Groq API with auto-model discovery."""
    
    def __init__(self):
        raw_key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
        self.api_key = raw_key.strip() if raw_key else None
        self.ranker_model = "llama-3.3-70b-versatile"           # Best on Groq
        self.optimizer_model = "llama-3.3-70b-versatile"         # Best on Groq
        self.vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Official vision model
        self._available = False

        if self.api_key:
            self._available = True
            self._discover_models()
        else:
            logger.error("Groq API key is missing.")

    def _discover_models(self):
        """Find best available models on Groq via REST API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            models = [m["id"] for m in response.json()["data"]]

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
        return self._available

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
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "model": attempt_model,
                    "temperature": temperature,
                    "seed": 42
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                
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
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
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
                "model": target_model,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq Vision Error: {e}")
            return None

# Singleton base client
groq_base = GroqClient()

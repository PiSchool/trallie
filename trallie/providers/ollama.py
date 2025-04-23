import os
import requests
from functools import lru_cache
from typing import Any

from trallie.providers import (
    BaseProvider,
    ProviderInitializationError,
    register_provider,
)


def ollama_api_call(default_return_value: Any):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except requests.RequestException as e:
                print(f"[ollama] API request failed: {e}")
                return default_return_value
        return wrapper
    return decorator


@register_provider("ollama")
class OllamaProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__()
        self.base_url = os.environ.get("OLLAMA_API_BASE_URL", "http://localhost:11434")

    @ollama_api_call(default_return_value=[])
    @lru_cache
    def list_available_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]

    @ollama_api_call(default_return_value="")
    @lru_cache
    def do_chat_completion(
        self, system_prompt: str, user_prompt: str, model_name: str
    ) -> str:
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
            "stream": False 
        }
        response = requests.post(f"{self.base_url}/api/chat", json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")

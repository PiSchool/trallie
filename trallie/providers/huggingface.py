import os
from functools import lru_cache
from typing import Any

import huggingface_hub
import requests

from trallie.providers import (
    BaseProvider,
    ProviderInitializationError,
    register_provider,
)


def hf_api_call(default_return_value: Any):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (
                huggingface_hub.InferenceTimeoutError,
                requests.exceptions.HTTPError,
            ) as e:
                if isinstance(e, huggingface_hub.InferenceTimeoutError):
                    print("[huggingface] model unavailable/request timed out.")
                    print(f"[huggingface] {e.response.text}")
                if isinstance(e, requests.exceptions.HTTPError):
                    print(
                        "[huggingface] HTTP status code "
                        f"{e.response.status_code} received."
                    )
                    print(f"[huggingface] {e.response.text}")
                return default_return_value

        return wrapper

    return decorator


@register_provider("huggingface")
class HuggingFaceProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__()
        if "HF_TOKEN" not in os.environ:
            raise ProviderInitializationError(
                "Must set the HF_TOKEN environment variable to use the "
                "'huggingface' provider."
            )
        self.client = huggingface_hub.InferenceClient(
            token=os.environ["HF_TOKEN"], timeout=3.0
        )

    @hf_api_call(default_return_value=[])
    @lru_cache
    def list_available_models(self) -> list[str]:
        return self.client.list_deployed_models()["text-generation"]

    @hf_api_call(default_return_value="")
    @lru_cache
    def do_chat_completion(
        self, system_prompt: str, user_prompt: str, model_name: str
    ) -> str:
        chat_completion = self.client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            model=model_name,
            max_tokens=100,
            stop=["\n"],
        )
        return chat_completion.choices[0].message.content

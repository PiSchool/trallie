import os
from functools import lru_cache
from typing import Any

import groq

from trallie.providers import (
    BaseProvider,
    ProviderInitializationError,
    register_provider,
)


def groq_api_call(default_return_value: Any):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (
                groq.APIConnectionError,
                groq.RateLimitError,
                groq.APIStatusError,
            ) as e:
                if isinstance(e, groq.APIConnectionError):
                    print("[groq] server could not be reached.")
                if isinstance(e, groq.RateLimitError):
                    print("[groq] rate limit exceeded.")
                if isinstance(e, groq.APIStatusError):
                    print(f"[groq] HTTP status code {e.status_code} received.")
                    print(f"[groq] {e.response.text}")
                return default_return_value

        return wrapper

    return decorator


@register_provider("groq")
class GroqProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__()
        if "GROQ_API_KEY" not in os.environ:
            raise ProviderInitializationError(
                "Must set the GROQ_API_KEY environment variable to use the "
                "'groq' provider."
            )
        self.client = groq.Groq(
            api_key=os.environ["GROQ_API_KEY"], timeout=30.0, max_retries=2
        )

    @groq_api_call(default_return_value=[])
    @lru_cache
    def list_available_models(self) -> list[str]:
        return [model.id for model in self.client.models.list().data if model.active]

    @groq_api_call(default_return_value="")
    @lru_cache
    def do_chat_completion(
        self, system_prompt: str, user_prompt: str, model_name: str
    ) -> str:
        chat_completion = self.client.chat.completions.create(
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
            temperature=0, 
            seed=4285, 
            response_format={"type": "json_object"}
        )
        return chat_completion.choices[0].message.content

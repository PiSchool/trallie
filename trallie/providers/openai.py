import os
from functools import lru_cache
from typing import Any

import openai

from trallie.providers import (
    BaseProvider,
    ProviderInitializationError,
    register_provider,
)


def openai_api_call(default_return_value: Any):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.RateLimitError,
            ) as e:
                if isinstance(e, openai.APITimeoutError):
                    print(f"[openai] API request timed out: {e}.")
                if isinstance(e, openai.APIError):
                    print(f"[openai] API returned an API Error: {e}.")
                if isinstance(e, openai.APIConnectionError):
                    print(f"[openai] API request failed to connect: {e}.")
                if isinstance(e, openai.BadRequestError):
                    print(f"[openai] API request was invalid: {e}.")
                if isinstance(e, openai.AuthenticationError):
                    print(f"[openai] API request was not authorized: {e}.")
                if isinstance(e, openai.PermissionDeniedError):
                    print(f"[openai] API request was not permitted: {e}.")
                if isinstance(e, openai.RateLimitError):
                    print(f"[openai] API request exceeded rate limit: {e}.")
                return default_return_value

        return wrapper

    return decorator


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__()
        if "OPENAI_API_KEY" not in os.environ:
            raise ProviderInitializationError(
                "Must set the OPENAI_API_KEY environment variable to use the "
                "'openai' provider."
            )
        self.client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], timeout=30.0, max_retries=2
        )

    @openai_api_call(default_return_value=[])
    @lru_cache
    def list_available_models(self) -> list[str]:
        return [model.id for model in self.client.models.list().data if model.active]

    @openai_api_call(default_return_value="")
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

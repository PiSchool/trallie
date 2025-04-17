from trallie.providers import get_provider
from trallie.providers import ProviderInitializationError
from trallie.prompts import (
    ZERO_SHOT_EXTRACTION_SYSTEM_PROMPT,
    FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_DE,
    FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_FR,
    FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_ES,
    FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_IT
)
from trallie.data_handlers import DataHandler

import json

import re

# Post processing for a reasoning model 
def post_process_response(response: str) -> str:
    """
    Removes <think>...</think> content from the response.
    """
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

class DataExtractor:
    LANGUAGE_PROMPT_MAP = {
        "en": ZERO_SHOT_EXTRACTION_SYSTEM_PROMPT,
        "de": FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_DE,
        "fr": FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_FR,
        "es": FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_ES,
        "it": FEW_SHOT_EXTRACTION_SYSTEM_PROMPT_IT,
    }

    ALLOWED_NON_EN_MODELS = {"gpt-4o", "llama-3.3-70b-versatile"}
    ALLOWED_NON_EN_PROVIDERS = {"openai", "groq"}
    ALLOWED_REASONING_MODELS = {"deepseek-r1-distill-llama-70b"}
    
    def __init__(self, provider, model_name, system_prompt=None, language="en", reasoning_mode=False):
        self.provider = provider
        self.model_name = model_name
        self.client = get_provider(self.provider)
        self.language = language
        self.reasoning_mode = reasoning_mode

        if self.reasoning_mode and self.model_name not in self.ALLOWED_REASONING_MODELS:
            raise ValueError(
                f"`reasoning_mode=True` is not supported for model '{self.model_name}'. "
            )
    
        if self.language == "en":
            self.system_prompt = system_prompt or self.LANGUAGE_PROMPT_MAP["en"]
        else:
            # Enforce allowed providers/models for non-English
            if self.provider not in self.ALLOWED_NON_EN_PROVIDERS:
                raise ValueError(f"Provider '{self.provider}' is not supported for language '{self.language}'.")

            if self.model_name not in self.ALLOWED_NON_EN_MODELS:
                raise ValueError(f"Model '{self.model_name}' is not allowed for non-English extraction.")

            self.system_prompt = system_prompt or self.LANGUAGE_PROMPT_MAP.get(self.language)
            if not self.system_prompt:
                raise ValueError(f"No prompt available for language '{self.language}'.")

    def extract_attributes(self, schema, record, max_retries=3):
        """
        Extracts attributes for a given record and schema.
        """
        user_prompt = f"""
            Following is the record: {record} and the attribute schema for extraction: {schema}
            Provide the extracted attributes. Avoid any words at the beginning and end.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.do_chat_completion(
                    self.system_prompt, user_prompt, self.model_name
                )
                # Validate if response is a valid JSON
                # print(response)
                if self.reasoning_mode:
                    response = post_process_response(response)
                response = json.loads(response)
                return response
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Invalid JSON response (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                print(f"Error: {e}")
                return None

    def extract_data(self, schema, record, max_retries=3, from_text=False):
        """
        Processes record and returns extracted attributes.
        """
        record_text = DataHandler(record, from_text=from_text).get_text()
        return self.extract_attributes(schema, record_text, max_retries)

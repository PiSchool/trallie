from trallie.providers import get_provider
from trallie.providers import ProviderInitializationError
from trallie.prompts import (
    FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT,
    FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_DE,
    FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_ES,
    FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_FR,
    FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_IT
)
from trallie.data_handlers import DataHandler

from collections import Counter
import json

import re

# Post processing for a reasoning model 
def post_process_response(response: str) -> str:
    """
    Removes <think>...</think> content from the response.
    """
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

class SchemaGenerator:
    LANGUAGE_PROMPT_MAP = {
        "en": FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT,
        "de": FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_DE,
        "fr": FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_FR,
        "es": FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_ES,
        "it": FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT_IT,
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
        self.attribute_counter = Counter()

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

    def extract_schema(self, description, record, max_retries=5):
        """
        Extract schema from a single document
        """
        user_prompt = f"""
            The data collection has the following description: {description}. 
            Following is the record: {record}
            Provide the schema/set of attributes in a JSON format. 
            Avoid any words at the beginning and end.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.do_chat_completion(
                    self.system_prompt, user_prompt, self.model_name
                )
                # Validate if response is a valid JSON
                if self.reasoning_mode:
                    response = post_process_response(response)
                schema = json.loads(response)
                return schema
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Invalid JSON response (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                print(f"Error: {e}")
                return None

    def update_schema_collection(self, description, record):
        """
        Updates schema collection with attributes from a single document.
        """
        schema = self.extract_schema(description, record)
        if schema:
            attributes = schema.keys() if isinstance(schema, dict) else []
            self.attribute_counter.update(attributes)

    def get_top_k_attributes(self, top_k=10):
        """
        Returns the top k most frequent attributes across multiple documents.
        """
        return [attr for attr, _ in self.attribute_counter.most_common(top_k)]

    def discover_schema(self, description, records, num_records=10, from_text=False):
        """
        Processes multiple documents for creation of the schema
        """
        num_records = min(num_records, len(records))

        for record in records[:num_records]:
            record_content = DataHandler(record, from_text=from_text).get_text()
            self.update_schema_collection(description, record_content)

        return self.get_top_k_attributes()


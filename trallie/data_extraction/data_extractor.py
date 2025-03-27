from trallie.providers import get_provider
from trallie.providers import ProviderInitializationError
from trallie.prompts import (
    FEW_SHOT_EXTRACTION_SYSTEM_PROMPT,
    ZERO_SHOT_EXTRACTION_SYSTEM_PROMPT,
)

import json
from trallie.data_handlers import DataHandler


class DataExtractor:
    def __init__(self, provider, model_name, system_prompt=None):
        self.provider = provider
        self.model_name = model_name
        self.client = get_provider(self.provider)
        self.system_prompt = system_prompt or FEW_SHOT_EXTRACTION_SYSTEM_PROMPT

    def extract_attributes(self, schema, record, max_retries=3):
        """
        Extracts attributes for a given record and schema.
        """
        user_prompt = f"""
            Following is the attribute schema for extraction: {schema} and the record: {record}
            Provide the extracted attributes. Avoid any words at the beginning and end.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.do_chat_completion(
                    self.system_prompt, user_prompt, self.model_name
                )
                # Validate if response is a valid JSON
                response = json.loads(response)
                return response
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Invalid JSON response (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                print(f"Error: {e}")
                return None

    def extract_data(self, schema, record, max_retries=3):
        """
        Processes record and returns extracted attributes.
        """
        record_text = DataHandler(record).get_text()
        return self.extract_attributes(schema, record_text, max_retries)

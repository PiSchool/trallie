from trallie.providers import get_provider
from trallie.providers import ProviderInitializationError
from trallie.prompts import (
    FEW_SHOT_GENERATION_SYSTEM_PROMPT,
    ZERO_SHOT_GENERATION_SYSTEM_PROMPT,
    FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT
)
from trallie.data_handlers import DataHandler

from collections import Counter
import json

class SchemaGenerator:
    def __init__(self, provider, model_name, system_prompt=None):
        self.provider = provider
        self.model_name = model_name
        self.client = get_provider(self.provider)
        self.system_prompt = system_prompt or FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT
        self.attribute_counter = Counter()

    def extract_schema(self, description, record, max_retries=3):
        '''
        Extract schema from a single document
        '''
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
        '''
        Updates schema collection with attributes from a single document.
        '''
        schema = self.extract_schema(description, record)
        if schema:
            attributes = schema.keys() if isinstance(schema, dict) else []
            self.attribute_counter.update(attributes)

    def get_top_k_attributes(self, top_k=10):
        '''
        Returns the top k most frequent attributes across multiple documents. 
        '''
        return [attr for attr, _ in self.attribute_counter.most_common(top_k)]

    def discover_schema(self, description, records, num_records=5):
        '''
        Processes multiple documents for creation of the schema
        '''
        num_records = min(num_records, len(records))

        for record in records[:num_records]:
            record_content = DataHandler(record).get_text()
            self.update_schema_collection(description, record)
        
        return self.get_top_k_attributes()

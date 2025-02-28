from trallie.providers import get_provider
from trallie.providers import ProviderInitializationError
from trallie.prompts import FEW_SHOT_EXTRACTION_SYSTEM_PROMPT, ZERO_SHOT_EXTRACTION_SYSTEM_PROMPT

class DataExtractor:
    def __init__(self, provider, model_name):
        self.provider = provider 
        self.model_name = model_name
        self.client = get_provider(self.provider)
     
    def extract_data_zero_shot(self, schema, record):
        user_prompt = f"""
            Following is the record: {record} and the attribute schema for extraction: {schema}
            Provide the extracted attributes. Avoid any words at the beginning and end.
        """
        try: 
            response = self.client.do_chat_completion(
                ZERO_SHOT_EXTRACTION_SYSTEM_PROMPT, user_prompt, self.model_name
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

    def extract_data_few_shot(self, schema, record):
        user_prompt = f"""
            Following is the record: {record} and the attribute schema for extraction: {schema}
            Provide the extracted attributes. Avoid any words at the beginning and end.
        """
        try: 
            response = self.client.do_chat_completion(
                FEW_SHOT_EXTRACTION_SYSTEM_PROMPT, user_prompt, self.model_name
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

    def extract_data_custom(self, system_prompt, user_prompt):
        try: 
            response = self.client.do_chat_completion(
                system_prompt, user_prompt, self.model_name
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None






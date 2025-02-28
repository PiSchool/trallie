
from trallie.providers import get_provider
from trallie.providers import ProviderInitializationError
from trallie.prompts import (FEW_SHOT_GENERATION_SYSTEM_PROMPT, 
                            ZERO_SHOT_GENERATION_SYSTEM_PROMPT,
                            TOPIC_MODELLING_PROMPT, 
                            CLUSTER_LABELLING_PROMPT
                            SCHEMA_REFINEMENT_PROMPT)

class SchemaGenerator:
    def __init__(self, provider, model_name):
        self.provider = provider 
        self.model_name = model_name
        self.client = get_provider(self.provider)
     
    def discover_schema_zero_shot(self, description, records):
        user_prompt = f"""
            The data collection has the following description: {description}. 
            Following is the list of records: {records}
            Provide the schema/set of attributes in a JSON format. 
            Avoid any words at the beginning and end.
        """
        try: 
            response = self.client.do_chat_completion(
                ZERO_SHOT_GENERATION_SYSTEM_PROMPT, user_prompt, self.model_name
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

    def discover_schema_few_shot(self, description, records):
        user_prompt = f"""
            The data collection has the following description: {description}. 
            Following is the list of records: {records}
            Provide the schema/set of attributes in a JSON format. 
            Avoid any words at the beginning and end.
        """
        try: 
            response = self.client.do_chat_completion(
                FEW_SHOT_GENERATION_SYSTEM_PROMPT, user_prompt, self.model_name
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

    def discover_schema_topic_modelling(self, description, records):
        #TODO: with keyphrase transformer
        pass

    def discover_schema_custom(self, system_prompt, user_prompt):
        try: 
            response = self.client.do_chat_completion(
                system_prompt, user_prompt, self.model_name
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None





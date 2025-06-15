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
from typing import List, Dict, Any, Optional

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

    def discover_schema_large_document(self, 
                                     description: str, 
                                     record: str, 
                                     chunk_size: int = 100000, 
                                     overlap_size: int = 10000,
                                     max_retries: int = 5, 
                                     from_text: bool = False,
                                     top_k: int = 10) -> List[str]:
        """
        Discover schema from a large document by processing it in chunks.
        
        Args:
            description: Description of the data collection
            record: The document path or text to process
            chunk_size: Size of each chunk in characters
            overlap_size: Size of overlap between chunks
            max_retries: Maximum number of retries for each chunk
            from_text: Whether the record is text or a file path
            top_k: Number of top attributes to return
            
        Returns:
            List of top-k most frequent attributes
        """
        # Create a data handler for the document
        data_handler = DataHandler(record, from_text=from_text)
        
        # Define the LLM processor function for each chunk
        def llm_processor(chunk_text: str) -> Dict[str, Any]:
            return self.extract_schema(description, chunk_text, max_retries)
        
        # Process the large document using chunking
        chunk_results = data_handler.process_large_document(
            llm_processor=llm_processor,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            combine_results=False  # We want individual results to count frequencies
        )
        
        # Update the attribute counter with results from all chunks
        if isinstance(chunk_results, dict) and "error" not in chunk_results:
            # If we got a combined result, process it
            if chunk_results:
                attributes = chunk_results.keys() if isinstance(chunk_results, dict) else []
                self.attribute_counter.update(attributes)
        else:
            # Process individual chunk results
            for result in chunk_results:
                if result and isinstance(result, dict):
                    attributes = result.keys()
                    self.attribute_counter.update(attributes)
        
        return self.get_top_k_attributes(top_k)

    def discover_schema_with_chunking(self, 
                                    description: str, 
                                    records: List[str], 
                                    num_records: int = 10, 
                                    from_text: bool = False,
                                    chunk_size: int = 100000, 
                                    overlap_size: int = 10000,
                                    auto_detect_large_docs: bool = True,
                                    top_k: int = 10) -> List[str]:
        """
        Discover schema with automatic chunking for large documents.
        
        Args:
            description: Description of the data collection
            records: List of document paths or texts to process
            num_records: Number of records to process
            from_text: Whether the records are text or file paths
            chunk_size: Size of each chunk in characters
            overlap_size: Size of overlap between chunks
            auto_detect_large_docs: Whether to automatically use chunking for large documents
            top_k: Number of top attributes to return
            
        Returns:
            List of top-k most frequent attributes
        """
        num_records = min(num_records, len(records))
        
        for record in records[:num_records]:
            if auto_detect_large_docs:
                # Check if the document is large enough to warrant chunking
                data_handler = DataHandler(record, from_text=from_text)
                full_text = data_handler.get_text()
                
                if full_text and not full_text.startswith("Error:"):
                    if len(full_text) > chunk_size:
                        print(f"Document is large ({len(full_text)} chars), using chunking for schema discovery...")
                        self.discover_schema_large_document(
                            description, record, chunk_size, overlap_size, 5, from_text, top_k
                        )
                        continue
                    else:
                        print(f"Document is small ({len(full_text)} chars), processing normally for schema discovery...")
            
            # Use the original method for smaller documents
            record_content = DataHandler(record, from_text=from_text).get_text()
            self.update_schema_collection(description, record_content)
        
        return self.get_top_k_attributes(top_k)


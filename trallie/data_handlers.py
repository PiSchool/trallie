import json
import re
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Optional, Callable
import math

def infer_datatype(func):
    """
    Decorator to infer the datatype of the document and set it.
    """

    def wrapper(self, *args, **kwargs):
        self.datatype = self.document.split(".")[-1].lower()
        return func(self, *args, **kwargs)

    return wrapper
    
class DataHandler:
    def __init__(self, document, from_text=False):
        self.document = document
        self.from_text = from_text
        self.datatype = None
        self.length = None  # Length will be set by the decorator
        self.text = None  # Contains extracted text

    def get_text_from_pdf(self):
        # Use a pdf extractor
        try:
            # Open and read the PDF file
            reader = PdfReader(self.document)
            text = ""
            # Iterate through all pages and extract text
            for page in reader.pages:
                text += page.extract_text()
            return text.strip() if text else "Error: No text found in the PDF"
        except FileNotFoundError:
            return "Error: File not found"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"
        return "Extracted text from PDF"

    def get_text_from_html(self):
        # Use an HTML parser
        try:
            with open(self.document, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return "Error: File not found"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"
        return "Extracted text from HTML"

    def get_text_from_json(self):
        # Extract text from JSON
        try:
            with open(self.document, "r", encoding="utf-8") as file:
                data = json.load(file)
            # Convert the JSON object to a pretty-printed string for readability
            return json.dumps(data, indent=4)
        except FileNotFoundError:
            return "Error: File not found"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        return "Extracted text from JSON"

    def get_text_from_txt(self):
        try:
            with open(self.document, "r") as file:
                return file.read()
        except FileNotFoundError:
            return "Error: File not found"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"
        
    #def chunk_text(self, char_limit=100000):
    #     """
    #     Decorator to chunk longer document if it exceeds the word length.
    #     """
    #     if self.length > char_limit:
    #         self.text = self.text[:char_limit]
    #     return self.text

    # @infer_datatype
    # def get_text(self):
    #     """
    #     Get text from the document based on its datatype.
    #     """
    #     if self.from_text:
    #         self.datatype = "txt"
    #         self.text = self.document
    #     else:
    #         if self.datatype == "pdf":
    #             self.text = self.get_text_from_pdf()
    #         elif self.datatype in {"html", "htm"}:
    #             self.text = self.get_text_from_html()
    #         elif self.datatype == "json":
    #             self.text = self.get_text_from_json()
    #         elif self.datatype == "txt":
    #             self.text = self.get_text_from_txt()
    #         else:
    #             return "Unsupported file type"

    #     # Chunk text
    #     self.length = len(self.text)
    #     self.chunk_text()
    #     return self.text

    @infer_datatype
    def get_text(self):
        """
        Get text from the document based on its datatype.
        """
        if self.from_text:
            self.datatype = "txt"
            self.text = self.document
        else:
            if self.datatype == "pdf":
                self.text = self.get_text_from_pdf()
            elif self.datatype in {"html", "htm"}:
                self.text = self.get_text_from_html()
            elif self.datatype == "json":
                self.text = self.get_text_from_json()
            elif self.datatype == "txt":
                self.text = self.get_text_from_txt()
            else:
                return "Unsupported file type"

        self.length = len(self.text)
        return self.text

    def create_overlapping_chunks(self, text: str, chunk_size: int = 100000, overlap_size: int = 10000) -> List[str]:
        """
        Create overlapping chunks from text to maintain context between chunks.
        
        Args:
            text: The text to chunk
            chunk_size: Size of each chunk in characters
            overlap_size: Size of overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the first chunk, include overlap from previous chunk
            if start > 0:
                chunk_start = start - overlap_size
            else:
                chunk_start = start
                
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 1000 characters
                search_start = max(end - 1000, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start + chunk_size // 2:  # Only break if we find a reasonable sentence end
                    end = sentence_end + 1
            
            chunk = text[chunk_start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            start = end
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks

    def process_chunks_with_llm(self, 
                              chunks: List[str], 
                              llm_processor: Callable[[str], Dict[str, Any]], 
                              combine_results: bool = True,
                              combine_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process each chunk with the provided LLM processor and optionally combine results.
        
        Args:
            chunks: List of text chunks to process
            llm_processor: Function that takes a text chunk and returns extracted data
            combine_results: Whether to combine results from all chunks
            combine_prompt: Custom prompt for combining results (optional)
            
        Returns:
            Combined results from all chunks
        """
        if not chunks:
            return {}
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            try:
                print(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
                result = llm_processor(chunk)
                if result:
                    chunk_results.append(result)
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                continue
        
        if not chunk_results:
            return {}
        
        # If only one chunk or no combination needed, return the first result
        if len(chunk_results) == 1 or not combine_results:
            return chunk_results[0]
        
        # Combine results from all chunks
        return self.combine_chunk_results(chunk_results, combine_prompt)
    
    def combine_chunk_results(self, 
                            chunk_results: List[Dict[str, Any]], 
                            custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Combine results from multiple chunks using an LLM to make a final decision.
        
        Args:
            chunk_results: List of results from individual chunks
            custom_prompt: Custom prompt for combining results
            
        Returns:
            Combined result
        """
        if not chunk_results:
            return {}
            
        # First, collect all values for each attribute
        collected_results = {}
        for result in chunk_results:
            for key, value in result.items():
                if key not in collected_results:
                    collected_results[key] = []
                
                # Handle different value types
                if isinstance(value, list):
                    collected_results[key].extend(value)
                elif isinstance(value, (str, int, float, bool)):
                    if value not in collected_results[key]:
                        collected_results[key].append(value)
                else:
                    collected_results[key].append(value)
        
        # Log the collected results before LLM consolidation
        print("\nCollected results from chunks before final LLM consolidation:")
        print(json.dumps(collected_results, indent=2))
        print("\nPerforming final LLM consolidation...")
        
        # Create a prompt for the LLM to make final decisions
        prompt = custom_prompt or """
        You are an expert at analyzing and consolidating information from multiple sources.
        Below are the results found for different queries in a document. For each query, multiple values were found.
        Your task is to analyze these values and determine the most accurate and complete answer for each query.
        Consider the following:
        1. If values are consistent, use the most detailed one
        2. If values conflict, choose the most specific and technically accurate one
        3. If values complement each other, combine them appropriately
        4. If a value seems out of context or incorrect, exclude it
        
        Here are the results to analyze:
        {results}
        
        Please provide your final decision for each query in JSON format, maintaining the same structure.
        """
        
        # Format the results for the prompt
        formatted_results = json.dumps(collected_results, indent=2)
        final_prompt = prompt.format(results=formatted_results)
        
        try:
            # Get the LLM to make final decisions
            from trallie.providers import get_provider
            provider = get_provider("openai")  # Using OpenAI for final decision
            response = provider.do_chat_completion(
                system_prompt="You are an expert at analyzing and consolidating technical information.",
                user_prompt=final_prompt,
                model_name="gpt-4o"
            )
            
            # Parse the response
            final_results = json.loads(response)
            return final_results
            
        except Exception as e:
            print(f"Error in final LLM consolidation: {e}")
            # Fallback to simple merging if LLM consolidation fails
            final_results = {}
            for key, values in collected_results.items():
                # Remove duplicates while preserving order
                seen = set()
                unique_values = []
                for value in values:
                    if value not in seen:
                        seen.add(value)
                        unique_values.append(value)
                
                # If only one value, simplify
                if len(unique_values) == 1:
                    final_results[key] = unique_values[0]
                else:
                    final_results[key] = unique_values
            
            return final_results

    def process_large_document(self, 
                             llm_processor: Callable[[str], Dict[str, Any]], 
                             chunk_size: int = 100000, 
                             overlap_size: int = 10000,
                             combine_results: bool = True) -> Dict[str, Any]:
        """
        Process a large document by chunking it and processing each chunk with an LLM.
        
        Args:
            llm_processor: Function that takes text and returns extracted data
            chunk_size: Size of each chunk in characters
            overlap_size: Size of overlap between chunks
            combine_results: Whether to combine results from all chunks
            
        Returns:
            Combined results from processing all chunks
        """
        # Get the full text
        full_text = self.get_text()
        
        if not full_text or full_text.startswith("Error:"):
            return {"error": full_text}
        
        # Create chunks
        chunks = self.create_overlapping_chunks(full_text, chunk_size, overlap_size)
        
        print(f"Document length: {len(full_text)} characters")
        print(f"Created {len(chunks)} chunks")
        
        # Process chunks
        return self.process_chunks_with_llm(chunks, llm_processor, combine_results)

    

""" Example Usage:
        python cli_document_search.py document.pdf
        python cli_document_search.py document.pdf --provider groq --model llama-3.3-70b-versatile
        python cli_document_search.py document.pdf --output results.json --chunk-size 50000
    """


import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

from trallie.data_extraction.data_extractor import DataExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hardcoded queries to search for
DEFAULT_QUERIES = {
    "maximum_wind_speed": "What is the maximum speed of wind in m/s?",
    "heater_sensing_range": "What is the heater sensing range",
    "acoustic_insulation": " What is Acoustic insulation for all ducts"
}

def setup_environment():
    """Setup environment variables for API keys."""
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = None # ENTER GROQ KEY HERE
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = None # ENTER OPENAI KEY HERE

def create_search_schema():
    """Create a schema for the hardcoded queries."""
    schema = {}
    for query, description in DEFAULT_QUERIES.items():
        schema[query] = description
    return schema

def process_document(file_path: str, provider: str, model_name: str, chunk_size: int = 100000, overlap_size: int = 10000) -> Dict[str, Any]:
    """
    Process a document using chunking to extract specific information.
    
    Args:
        file_path: Path to the document file
        provider: LLM provider to use
        model_name: Model name to use
        chunk_size: Size of each chunk in characters
        overlap_size: Size of overlap between chunks
        
    Returns:
        Dictionary containing extracted information
    """
    try:
        # Create schema for the queries
        schema = create_search_schema()
        
        # Initialize data extractor
        data_extractor = DataExtractor(
            provider=provider, 
            model_name=model_name,
            reasoning_mode=False
        )
        
        logger.info(f"Processing document: {file_path}")
        logger.info(f"Using provider: {provider}, model: {model_name}")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {overlap_size}")
        
        # Process the document with chunking
        logger.info("Starting initial document processing with chunking...")
        extracted_data = data_extractor.extract_data_with_chunking(
            schema=schema,
            record=file_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            combine_results=True,
            auto_detect_large_docs=True,
            max_retries=3
        )
        
        logger.info("Initial processing complete. Performing final LLM consolidation...")
        
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {"error": str(e)}

def display_results(extracted_data: Dict[str, Any], output_file: str = None):
    """Display the extracted results in a formatted way."""
    logger.info("Processing search results...")
    
    if "error" in extracted_data:
        logger.error(f"Error: {extracted_data['error']}")
        return
    
    if not extracted_data:
        logger.warning("No information found for the specified queries.")
        return
    
    # Format results for JSON output
    results = {}
    for query, description in DEFAULT_QUERIES.items():
        query_display_name = query.replace('_', ' ').title()
        
        if query in extracted_data:
            value = extracted_data[query]
            
            # Format the value for display
            if isinstance(value, list):
                if len(value) == 1:
                    display_value = str(value[0])
                else:
                    display_value = [str(v) for v in value]
            else:
                display_value = str(value)
            
            results[query] = {
                "description": description,
                "value": display_value,
                "confidence": "high" if isinstance(value, str) else "multiple_values"
            }
        else:
            results[query] = {
                "description": description,
                "value": None,
                "confidence": "none"
            }
    
    # Print results in JSON format
    print(json.dumps(results, indent=2))
    
    # Save results to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Could not save results to file: {e}")

def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Search documents for specific technical specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_document_search.py document.pdf
  python cli_document_search.py document.pdf --provider groq --model llama-3.3-70b-versatile
  python cli_document_search.py document.pdf --output results.json --chunk-size 50000
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the document file to search (PDF, TXT, JSON, HTML)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["openai", "groq"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--model",
        help="Model name to use (default: gpt-4o for OpenAI, llama-3.3-70b-versatile for Groq)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Size of each chunk in characters (default: 100000)"
    )
    
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=10000,
        help="Size of overlap between chunks (default: 10000)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file to save results as JSON"
    )
    
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="List the available search queries and exit"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # List queries if requested
    if args.list_queries:
        logger.info("Available Search Queries:")
        for query, description in DEFAULT_QUERIES.items():
            print(f"• {query.replace('_', ' ').title()}: {description}")
        return
    
    # Set default model based on provider
    if not args.model:
        args.model = "gpt-4o" if args.provider == "openai" else "llama-3.3-70b-versatile"
    
    # Process document
    logger.info("Starting document search...")
    extracted_data = process_document(
        args.file_path,
        args.provider,
        args.model,
        args.chunk_size,
        args.overlap_size
    )
    
    # Display results
    display_results(extracted_data, args.output)
    logger.info("Document search completed.")

if __name__ == "__main__":
    main() 
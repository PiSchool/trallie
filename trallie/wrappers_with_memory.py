"""
Enhanced wrappers with memory support for OpenIE
"""
import os
import json

from trallie import SchemaGenerator
from trallie import DataExtractor


def openie_with_memory(description, records, provider, model_name, reasoning_mode=False, dataset_name="dataset", memory=False):
    """
    OpenIE with configurable memory support.
    
    Args:
        description: Dataset description
        records: List of document paths
        provider: LLM provider
        model_name: Model name
        reasoning_mode: Whether to use reasoning mode
        dataset_name: Name for output file
        memory: Whether to enable memory in schema generation
        
    Returns:
        Dictionary of extracted data
    """
    # Ensure the 'results' directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the schema generator with configurable memory
    schema_generator = SchemaGenerator(
        provider=provider, 
        model_name=model_name, 
        reasoning_mode=reasoning_mode,
        memory=memory  # KEY: Memory parameter
    )
    
    # Feed records to the LLM and discover schema
    print(f"Discovering schema with memory={'ON' if memory else 'OFF'}...")
    schema = schema_generator.discover_schema(description, records)
    print(f"Generated schema: {schema}")
    
    # Initialize data extractor with a provider and model
    data_extractor = DataExtractor(provider=provider, model_name=model_name)
    
    # Extract values from the text based on the schema
    print("Extracting data from every record:")
    extracted_jsons = {}
    for record in records:
        record_name = os.path.basename(record)
        try:
            extracted_json = data_extractor.extract_data(schema, record)
            extracted_jsons[record_name] = extracted_json
            print(f"Record: {record_name}, processed!")
        except Exception as e:
            print(f"Error processing {record_name}: {e}")
            extracted_jsons[record_name] = None

    print("Writing results to a file")
    memory_suffix = "memory" if memory else "nomemory"
    output_file = f"{results_dir}/{model_name}_{dataset_name}_openie_{memory_suffix}_predicted_table.json"
    with open(output_file, "w") as json_file:
        json.dump(extracted_jsons, json_file, indent=4)

    print(f"OpenIE completed! Results saved to {output_file}")
    return extracted_jsons, schema


def closedie(records, schema, provider, model_name, reasoning_mode=False, dataset_name="dataset"):
    """
    ClosedIE with predefined schema.
    """
    # Ensure the 'results' directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract values from the text based on the schema
    data_extractor = DataExtractor(provider=provider, model_name=model_name, reasoning_mode=reasoning_mode)
    print("Extracting data from every record:")
    extracted_jsons = {}
    for record in records:
        record_name = os.path.basename(record)
        try:
            extracted_json = data_extractor.extract_data(schema, record)
            extracted_jsons[record_name] = extracted_json
            print(f"Record: {record}, processed!")
        except Exception as e:
            print(f"Error processing {record_name}: {e}")
            extracted_jsons[record_name] = None

    print("Writing results to a file")
    with open(f"{results_dir}/{model_name}_{dataset_name}_closedie_predicted_table.json", "w") as json_file:
        json.dump(extracted_jsons, json_file, indent=4)

    print("ClosedIE completed!")
    return extracted_jsons


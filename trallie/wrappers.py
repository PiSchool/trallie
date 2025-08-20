import os
import json

from trallie import SchemaGenerator
from trallie import DataExtractor


def openie(description, records, provider, model_name, reasoning_mode, dataset_name):
    # Ensure the 'results' directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    # Initialize the schema generator with a provider and model
    schema_generator = SchemaGenerator(provider=provider, model_name=model_name, reasoning_mode=reasoning_mode)
    # Feed records to the LLM and discover schema
    schema = schema_generator.discover_schema(description, records)
    print("Generated a schema for the records!")
    # Initialize data extractor with a provider and model
    data_extractor = DataExtractor(provider=provider, model_name=model_name)
    # Extract values from the text based on the schema
    print("Extracting data from every record:")
    extracted_jsons = {}
    for record in records:
        record_name = os.path.basename(record)
        extracted_json = data_extractor.extract_data(schema, record)
        extracted_jsons[record_name] = extracted_json
        print(f"Record: {record}, processed!")

    print("Writing results to a file")
    with open(f"results/{model_name}_{dataset_name}_openie_predicted_table.json", "w") as json_file:
        json.dump(extracted_jsons, json_file, indent=4)

    print("OpenIE completed!")
    return extracted_jsons


def closedie(records, schema, provider, model_name, reasoning_mode, dataset_name):
    # Ensure the 'results' directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    # Extract values from the text based on the schema
    data_extractor = DataExtractor(provider=provider, model_name=model_name, reasoning_mode=reasoning_mode)
    print("Extracting data from every record:")
    extracted_jsons = {}
    for record in records:
        record_name = os.path.basename(record)
        extracted_json = data_extractor.extract_data(schema, record)
        extracted_jsons[record_name] = extracted_json
        print(f"Record: {record}, processed!")

    print("Writing results to a file")
    with open(f"results/{model_name}_{dataset_name}_closedie_predicted_table.json", "w") as json_file:
        json.dump(extracted_jsons, json_file, indent=4)

    print("ClosedIE completed!")
    return extracted_jsons

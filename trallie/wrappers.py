import os
import json

from trallie.data_handlers import (
    create_records_for_schema_generation,
    create_record_for_data_extraction,
)

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor


def openie(description, records, provider, model_name, dataset_name):
    # Create records from the data collection (max records are 5)
    schema_records = create_records_for_schema_generation(records)
    # Initialize the schema generator with a provider and model
    schema_generator = SchemaGenerator(provider=provider, model_name=model_name)
    # Feed records to the LLM and discover schema
    schema = schema_generator.discover_schema_few_shot(description, schema_records)
    print("Generated a schema for the records!")
    # Initialize data extractor with a provider and model
    data_extractor = DataExtractor(provider=provider, model_name=model_name)
    # Extract values from the text based on the schema
    print("Extracting data from every record:")
    extracted_jsons = {}
    for record in records:
        record_name = os.path.basename(record)
        extraction_record = create_record_for_data_extraction(record)
        extracted_json = data_extractor.extract_data_few_shot(schema, extraction_record)
        extracted_jsons[record_name] = extracted_json
        print(f"Record: {record}, processed!")

    print("Writing results to a file")
    with open(f"{dataset_name}_openie_predicted_table.json", "w") as json_file:
        json.dump(extracted_jsons, json_file, indent=4)

    print("OpenIE completed!")
    return extracted_jsons


def closedie(records, schema, provider, model_name, dataset_name):
    # Extract values from the text based on the schema
    data_extractor = DataExtractor(provider=provider, model_name=model_name)
    print("Extracting data from every record:")
    extracted_jsons = {}
    for record in records:
        record_name = os.path.basename(record)
        extraction_record = create_record_for_data_extraction(record)
        extracted_json = data_extractor.extract_data_zero_shot(
            schema, extraction_record
        )
        extracted_jsons[record_name] = extracted_json
        print(f"Record: {record}, processed!")

    print("Writing results to a file")
    with open(f"{dataset_name}_closedie_predicted_table.json", "w") as json_file:
        json.dump(extracted_jsons, json_file, indent=4)

    print("ClosedIE completed!")
    return extracted_jsons

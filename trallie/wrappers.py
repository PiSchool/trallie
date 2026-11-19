import os
import json

from trallie import SchemaGenerator
from trallie import DataExtractor
from trallie.data_handlers import DataHandlerError


def _extract_all(data_extractor, schema, records):
    extracted_jsons = {}
    for record in records:
        record_name = os.path.basename(record)
        try:
            extracted_jsons[record_name] = data_extractor.extract_data(schema, record)
        except DataHandlerError as e:
            print(f"Record: {record}, skipped: {e}")
            extracted_jsons[record_name] = None
            continue
        print(f"Record: {record}, processed!")
    return extracted_jsons


def result_tag(model_name):
    # Model ids such as "openai/gpt-oss-120b" must not create subdirectories
    return model_name.replace("/", "-")


def _write_results(results_dir, model_name, dataset_name, mode, extracted_jsons):
    os.makedirs(results_dir, exist_ok=True)
    print("Writing results to a file")
    with open(f"{results_dir}/{result_tag(model_name)}_{dataset_name}_{mode}_predicted_table.json", "w") as json_file:
        json.dump(extracted_jsons, json_file, indent=4)


def openie(description, records, provider, model_name, reasoning_mode, dataset_name,
           top_k=20, results_dir="results"):
    # Initialize the schema generator with a provider and model
    schema_generator = SchemaGenerator(provider=provider, model_name=model_name, reasoning_mode=reasoning_mode)
    # Feed records to the LLM and discover schema (names plus descriptions)
    schema = schema_generator.discover_schema(description, records, top_k=top_k, return_descriptions=True)
    print("Generated a schema for the records!")
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/{result_tag(model_name)}_{dataset_name}_openie_schema.json", "w") as schema_file:
        json.dump(schema, schema_file, indent=4)
    # Initialize data extractor with a provider and model
    data_extractor = DataExtractor(provider=provider, model_name=model_name, reasoning_mode=reasoning_mode)
    # Extract values from the text based on the schema
    print("Extracting data from every record:")
    extracted_jsons = _extract_all(data_extractor, schema, records)
    _write_results(results_dir, model_name, dataset_name, "openie", extracted_jsons)
    print("OpenIE completed!")
    return extracted_jsons


def closedie(records, schema, provider, model_name, reasoning_mode, dataset_name, results_dir="results"):
    # Extract values from the text based on the schema
    data_extractor = DataExtractor(provider=provider, model_name=model_name, reasoning_mode=reasoning_mode)
    print("Extracting data from every record:")
    extracted_jsons = _extract_all(data_extractor, schema, records)
    _write_results(results_dir, model_name, dataset_name, "closedie", extracted_jsons)
    print("ClosedIE completed!")
    return extracted_jsons

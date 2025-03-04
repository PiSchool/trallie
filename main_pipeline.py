import os
from trallie.data_handlers import (
    create_records_for_schema_generation,
    create_record_for_schema_filling,
)

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor

os.environ["GROQ_API_KEY"] = None #ENTER_GROQ_KEY_HERE
os.environ["OPENAI_API_KEY"] = None #ENTER OPENAI KEY HERE

# Define the path to a set of documents/a data collection for inference
records = [
    "data/use-cases/EO_papers/pdf_0808.3837.pdf",
    "data/use-cases/EO_papers/pdf_1001.4405.pdf",
    "data/use-cases/EO_papers/pdf_1002.3408.pdf",
]

# Provide a description of the data collection
description = "A dataset of Earth observation papers"
# Create records from the data collection (max records are 5)
schema_records = create_records_for_schema_generation(records)

# Initialize the schema generator with a provider and model
schema_generator = SchemaGenerator(
    provider="groq", model_name="llama-3.3-70b-versatile"
)
# Feed records to the LLM and discover schema
print("SCHEMA GENERATION IN ACTION ...")
schema = schema_generator.discover_schema_few_shot(description, schema_records)
print("Inferred schema", schema)


# Initialize data extractor with a provider and model 
data_extractor = DataExtractor(
    provider="groq", model_name="llama-3.3-70b-versatile"
)
# Extract values from the text based on the schema
print("SCHEMA COMPLETION IN ACTION ...")
for record in records:
    extraction_record = create_record_for_schema_filling(record)
    extracted_json = data_extractor.extract_data_few_shot(schema, extraction_record)
    print("Extracted attributes:", extracted_json)

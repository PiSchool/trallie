import os

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor

os.environ["GROQ_API_KEY"] = None #ENTER GROQ KEY HERE
os.environ["OPENAI_API_KEY"] = None #ENTER OPENAI KEY HERE

# Define the path to a set of documents/a data collection for inference
records = [
    "data/use-cases/EO_papers/pdf_0808.3837.pdf",
    "data/use-cases/EO_papers/pdf_1001.4405.pdf",
    "data/use-cases/EO_papers/pdf_1002.3408.pdf",
]

# Provide a description of the data collection
description = "A dataset of Earth observation papers"

# Initialize the schema generator with a provider and model
schema_generator = SchemaGenerator(provider="openai", model_name="gpt-4o")
# Feed records to the LLM and discover schema
print("SCHEMA GENERATION IN ACTION ...")
schema = schema_generator.discover_schema(description, records)
print("Inferred schema", schema)

# Initialize data extractor with a provider and model
data_extractor = DataExtractor(provider="openai", model_name="gpt-4o")
# Extract values from the text based on the schema
print("SCHEMA COMPLETION IN ACTION ...")
for record in records:
    extracted_json = data_extractor.extract_data(schema, record)
    print("Extracted attributes:", extracted_json)

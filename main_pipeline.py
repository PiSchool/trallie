import os

from trallie import SchemaGenerator
from trallie import DataExtractor

# os.environ["GROQ_API_KEY"] = None #ENTER GROQ KEY HERE
# os.environ["OPENAI_API_KEY"] = None #ENTER OPENAI KEY HERE


# Define the path to a set of documents/a data collection for inference
records = [
    "data/evaluation/data/fda_510ks/data/evaporate/fda-ai-pmas/510k/K141114.txt",
]

# Provide a description of the data collection
description = "A dataset of Earth observation papers"

# Initialize the schema generator with a provider and model
schema_generator = SchemaGenerator(provider="groq", model_name="llama-3.3-70b-versatile")
# Feed records to the LLM and discover schema
print("SCHEMA GENERATION IN ACTION ...")
schema = schema_generator.discover_schema(description, records)
print("Inferred schema", schema)

# Initialize data extractor with a provider and model
data_extractor = DataExtractor(provider="groq", model_name="llama-3.3-70b-versatile")
# Extract values from the text based on the schema
print("SCHEMA COMPLETION IN ACTION ...")
for record in records:
    extracted_json = data_extractor.extract_data(schema, record)
    print("Extracted attributes:", extracted_json)

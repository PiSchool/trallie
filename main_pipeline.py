import os
from trallie.data_handlers import (create_records_for_schema_generation, 
                                    create_record_for_schema_filling)
 
from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor

os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

# Define a set of records for schema inference
records = ['../trallie-core/data/raw/billsum_data/text_0.txt',
            '../trallie-core/data/raw/billsum_data/text_1.txt',
            '../trallie-core/data/raw/billsum_data/text_2.txt',
            '../trallie-core/data/raw/billsum_data/text_3.txt',
            '../trallie-core/data/raw/billsum_data/text_4.txt',
            '../trallie-core/data/raw/billsum_data/text_5.txt',
            '../trallie-core/data/raw/billsum_data/text_6.txt',
            '../trallie-core/data/raw/billsum_data/text_7.txt'
]

pdf_records = ['../trallie-core/data/raw/resume_data/data/ACCOUNTANT/10554236.pdf',
            '../trallie-core/data/raw/resume_data/data/SALES/10138632.pdf',
            '../trallie-core/data/raw/resume_data/data/TEACHER/11616482.pdf',
            '../trallie-core/data/raw/resume_data/data/FITNESS/10235429.pdf',
            '../trallie-core/data/raw/resume_data/data/ENGINEERING/10030015.pdf',
]

papers = ["data/use-cases/EO_papers/pdf_0808.3837.pdf",
        "data/use-cases/EO_papers/pdf_1001.4405.pdf",
        "data/use-cases/EO_papers/pdf_1002.3408.pdf"]


# Provide a description of the data collection
description = "A dataset of Earth observation papers"
# Create records from the data collection (max records are 5)
schema_records = create_records_for_schema_generation(papers)

schema_generator = SchemaGenerator(provider="groq", model_name="llama-3.3-70b-versatile")
#Feed records to the LLM and discover schema
print("SCHEMA GENERATION IN ACTION ...")
schema = schema_generator.discover_schema_zero_shot(description, schema_records)
print("Inferred schema", schema)

data_extractor = DataExtractor(provider="groq", model_name="llama-3.3-70b-versatile")
# Extract values from the text based on the schema 
print("SCHEMA COMPLETION IN ACTION ...")
for record in papers:
    extraction_record = create_record_for_schema_filling(record)
    extracted_json = data_extractor.extract_data_few_shot(schema, extraction_record)
    print("Extracted attributes:", extracted_json)


schema_generator = SchemaGenerator(provider="openai", model_name="gpt-4o")
#Feed records to the LLM and discover schema
print("SCHEMA GENERATION IN ACTION ...")
schema = schema_generator.discover_schema_zero_shot(description, schema_records)
print("Inferred schema", schema)

data_extractor = DataExtractor(provider="groq", model_name="llama-3.3-70b-versatile")
# Extract values from the text based on the schema 
print("SCHEMA COMPLETION IN ACTION ...")
for record in papers:
    extraction_record = create_record_for_schema_filling(record)
    extracted_json = data_extractor.extract_data_few_shot(schema, extraction_record)
    print("Extracted attributes:", extracted_json)

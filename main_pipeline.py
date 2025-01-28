from trallie.data_handlers import (create_records_for_schema_generation, 
                                    create_record_for_schema_filling)

from trallie.schema_generation.prompts import (FEW_SHOT_INFERENCE_PROMPT, 
                                                ZERO_SHOT_INFERENCE_PROMPT) 
from trallie.schema_generation.schema_generator import discover_schema

from trallie.schema_filling.prompts import (FEW_SHOT_FILLING_PROMPT, 
                                            ZERO_SHOT_FILLING_PROMPT) 
from trallie.schema_filling.schema_filler import fill_schema

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

# Provide a description of the data collection
description = "A collection of US congress bills"
# Create records from the data collection (max records are 5)
schema_records = create_records_for_schema_generation(records[:8])

# Feed records to the LLM and discover schema
print("SCHEMA GENERATION IN ACTION ...")
schema = discover_schema(description, schema_records, FEW_SHOT_INFERENCE_PROMPT)
print("Inferred schema", schema)

# Extract values from the text based on the schema 
print("SCHEMA COMPLETION IN ACTION ...")
for record in records:
    extraction_record = create_record_for_schema_filling(record)
    extracted_json = fill_schema(schema, record, ZERO_SHOT_FILLING_PROMPT)
    print("Extracted attributes:", extracted_json)


# Provide a description of the data collection
description = "A collection of resumes"
# Create records from the data collection (max records are 5)
schema_records = create_records_for_schema_generation(pdf_records[:3])

# Feed records to the LLM and discover schema
print("SCHEMA GENERATION IN ACTION ...")
schema = discover_schema(description, schema_records, FEW_SHOT_INFERENCE_PROMPT)
print("Inferred schema", schema)

# cExtract values from the text based on the schema 
print("SCHEMA COMPLETION IN ACTION ...")
for record in pdf_records:
    extraction_record = create_record_for_schema_filling(record)
    extracted_json = fill_schema(schema, record, ZERO_SHOT_FILLING_PROMPT)
    print("Extracted attributes:", extracted_json)

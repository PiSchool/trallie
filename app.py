import streamlit as st
import pandas as pd
import tempfile

from trallie.data_handlers import (create_records_for_schema_generation, 
                                    create_record_for_schema_filling)

from trallie.schema_generation.prompts import (FEW_SHOT_INFERENCE_PROMPT, 
                                                ZERO_SHOT_INFERENCE_PROMPT) 
from trallie.schema_generation.schema_generator import discover_schema

from trallie.schema_filling.prompts import (FEW_SHOT_FILLING_PROMPT, 
                                            ZERO_SHOT_FILLING_PROMPT) 
from trallie.schema_filling.schema_filler import fill_schema


# Streamlit App
st.set_page_config(page_title="Trallie", layout="centered")
st.image("assets\logo-pischool-transparent.svg", width=200)

# Header
st.title("Trallie")
st.subheader("Information Structuring: turning free-form text into tables")

# Schema Name Input
schema_name = st.text_input("Schema Name *", placeholder="e.g., InfoSynth StreamFlow")

# Description Input
description = st.text_input("Description *", placeholder="Provide a description of the data collection, e.g., 'A collection of resumes'")

# File Upload
uploaded_files = st.file_uploader("Upload files", type=["pdf", "json", "txt"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
            temp_file.write(uploaded_file.read())
            file_paths.append(temp_file.name)  # Store the file path
    

# Schema Generation and Completion
if st.button("Generate Schema"):
    if not schema_name:
        st.error("Please provide a schema name.")
    elif not uploaded_files:
        st.error("Please upload at least one file.")
    elif not description:
        st.error("Please provide a description.")
    else:
        # Generate Schema
        st.info("Generating schema...")
        schema_records = create_records_for_schema_generation(file_paths[:3])
        schema = discover_schema(description, schema_records, FEW_SHOT_INFERENCE_PROMPT)
        #st.text(schema)
        #st.success("Schema created successfully!")
        # Schema Editor
        st.subheader("Schema *")
        # Display schema in a text area
        st.text_area("Generated Schema", schema, height=300)  # Adjust the height as needed

        # Extract Values
        st.info("Extracting values based on the schema...")
        extracted_data = []
        for record in file_paths:
            extraction_record = create_record_for_schema_filling(record)
            extracted_json = fill_schema(schema, extraction_record, ZERO_SHOT_FILLING_PROMPT)
            extracted_data.append(extracted_json)
        
        st.write("Extracted Attributes:")
        st.json(extracted_data)


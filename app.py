import streamlit as st
import pandas as pd
import tempfile

from trallie.data_handlers import (
    create_records_for_schema_generation,
    create_record_for_data_extraction,
)

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor


# Streamlit App
st.set_page_config(page_title="Trallie", layout="centered")
st.image("assets\logo-pischool-transparent.svg", width=200)

# Header
st.title("Trallie")
st.subheader("Information Structuring: turning free-form text into tables")

# Schema Name Input
schema_name = st.text_input("Schema Name *", placeholder="e.g., InfoSynth StreamFlow")

# Description Input
description = st.text_input(
    "Description *",
    placeholder="Provide a description of the data collection, e.g., 'A collection of resumes'",
)

# File Upload
uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "json", "txt"], accept_multiple_files=True
)

if uploaded_files:
    file_paths = []

    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=uploaded_file.name
        ) as temp_file:
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
        schema_generator = SchemaGenerator(
            provider="groq", model_name="llama-3.3-70b-versatile"
        )
        schema_records = create_records_for_schema_generation(file_paths[:3])
        schema = schema_generator.discover_schema_few_shot(description, schema_records)
        # st.text(schema)
        # st.success("Schema created successfully!")
        # Schema Editor
        st.subheader("Schema *")
        # Display schema in a text area
        st.text_area(
            "Generated Schema", schema, height=300
        )  # Adjust the height as needed

        # Extract Values
        st.info("Extracting values based on the schema...")
        extracted_data = []
        data_extractor = DataExtractor(
            provider="groq", model_name="llama-3.3-70b-versatile"
        )
        for record in file_paths:
            extraction_record = create_record_for_data_extraction(record)
            extracted_json = data_extractor.extract_data_few_shot(
                schema, extraction_record
            )
            extracted_data.append(extracted_json)

        st.write("Extracted Attributes:")
        st.json(extracted_data)

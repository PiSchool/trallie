import os
import streamlit as st
import pandas as pd
import tempfile

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor

os.environ["GROQ_API_KEY"] = None # ENTER GROQ KEY HERE
os.environ["OPENAI_API_KEY"] =   None # ENTER OPENAI KEY HERE

# TODO add dropdown for providers and models
logo_image = os.path.join("assets", "logo-pischool-transparent.svg")

st.set_page_config(page_title="Trallie", layout="centered")
st.image(logo_image, width=200)

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
    "Upload files", type=["pdf", "json", "txt", "html", "htm"], accept_multiple_files=True
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
            provider="openai", model_name="gpt-4o"
        )
        schema = schema_generator.discover_schema(description, file_paths)
        st.subheader("Schema *")
        # Display schema in a text area
        st.text_area(
            "Generated Schema", schema, height=300
        )  # Adjust the height as needed

        # Extract Values
        st.info("Extracting values based on the schema...")
        extracted_data = []
        data_extractor = DataExtractor(
            provider="openai", model_name="gpt-4o"
        )
        for record in file_paths:
            extracted_json = data_extractor.extract_data(
                schema, record
            )
            extracted_data.append(extracted_json)

        st.write("Extracted Attributes:")
        st.json(extracted_data)
        st.success("Data extracted successfully!")   
import os
import streamlit as st
import pandas as pd
import tempfile

from trallie import SchemaGenerator
from trallie import DataExtractor

# Set OPENAI_API_KEY in your environment before running.

# TODO add dropdown for providers and models
logo_image = os.path.join("assets", "logo-pischool-transparent.svg")

st.set_page_config(page_title="Trallie", layout="centered")

# Custom CSS styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F2F7F7;
    }
    
    /* Header styling */
    h1 {
        color: #232323;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #232323;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #FAFDFF;
        border: 1px solid #A3A3A3;
        border-radius: 8px;
        color: #232323;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #009DDE;
        box-shadow: 0 0 0 1px #009DDE;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #FAFDFF;
        border: 2px dashed #009DDE;
        border-radius: 12px;
        padding: 2rem;
    }
    
    .stFileUploader label {
        color: #232323;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #009DDE;
        color: #FAFDFF;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0088c4;
        box-shadow: 0 4px 12px rgba(0, 157, 222, 0.3);
    }
    
    /* Text area */
    .stTextArea > div > div > textarea {
        background-color: #FAFDFF;
        border: 1px solid #A3A3A3;
        border-radius: 8px;
        color: #232323;
    }
    
    /* Info/Success boxes */
    .stAlert {
        background-color: #FAFDFF;
        border-left: 4px solid #009DDE;
        border-radius: 8px;
    }
    
    /* JSON display */
    .stJson {
        background-color: #FAFDFF;
        border: 1px solid #A3A3A3;
        border-radius: 8px;
    }
    
    /* Labels */
    label {
        color: #232323 !important;
        font-weight: 500;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #009DDE;
    }
    
    /* Subheader under title */
    .stApp > header + div h3 {
        color: #232323;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

st.image(logo_image, width=200)

# Header
st.title("Trallie")
st.subheader("Shaping Unstructured Data into Valuable Information")

# Schema Name Input
schema_name = st.text_input("Schema name *", placeholder="Resume")

# Description Input
description = st.text_input(
    "Description *",
    placeholder="A collection of resumes",
)

# File Upload
uploaded_files = st.file_uploader(
    "Drop your file here or Click to browse", 
    type=["pdf", "json", "txt", "html", "htm"], 
    accept_multiple_files=True,
    help="Supported files: PDF, JSON, TXT, HTML, HTM | Maximum size: 200MB"
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
        with st.spinner("Generating schema..."):
            schema_generator = SchemaGenerator(
                provider="openai", model_name="gpt-4o"
            )
            schema = schema_generator.discover_schema(description, file_paths)
        
        st.subheader("Generated Schema")
        # Display schema in a text area
        st.text_area(
            "Schema", schema, height=150, label_visibility="collapsed"
        )

        # Extract Values
        with st.spinner("Extracting values based on the schema..."):
            extracted_data = []
            data_extractor = DataExtractor(
                provider="openai", model_name="gpt-4o"
            )
            
            # Progress bar
            progress_bar = st.progress(0)
            total_files = len(file_paths)
            
            for idx, record in enumerate(file_paths):
                extracted_json = data_extractor.extract_data(schema, record)
                extracted_data.append(extracted_json)
                progress_bar.progress((idx + 1) / total_files)

        st.subheader("Extracted Attributes:")
        st.json(extracted_data)
        st.success("Data extracted successfully!")
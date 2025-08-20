# Trallie Documentation

<p align="center">
  <img src="assets/trallie.png" alt="Image description" style="width:250px; height:auto;">
</p>

**Trallie** (Transfer Learning for Information Extraction) is an LLM-based framework that enables structured data extraction from unstructured text. 

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Configuration](#configuration)
6. [Extending Trallie](#extending-trallie)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## 🚀 Features

1. Supports for several document types like **PDF, HTML and TXT as well as raw text**.  

2. Support for multiple LLM providers : **OpenAI, Groq, HuggingFace Endpoints and Ollama**. Supports regular + reasoning models!

3. Modular framework: Extract data from your documents according to a **pre-defined format or auto-infer schema** from documents.

4. Supports inputs and outputs in **5 languages : English (EN), Italian (IT), French (FR), German (DE) and Spanish (ES)**.


## 📦 Installation

### Install Trallie from source:

```bash
git clone https://github.com/PiSchool/trallie.git
cd trallie
pip install -e .
```

### Install Trallie via pip 
```bash
pip install trallie
```

## ⚡ Quick Start

Here's a minimal example to extract information:

```python
import os

from trallie import SchemaGenerator
from trallie import DataExtractor

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
```

### Output (example)

```json
{"location": ["Paris, France"], "name": "John Doe", "age": 24}
```

---

## Core Components

### Schema Generation

### Data Extraction 

Trallie handles everything through its `Extractor` interface, which:

- Receives input text and rules
- Formats prompts
- Sends them to the selected LLM backend
- Normalizes the results into structured output


## ⚙️ Configuration

You can customize Trallie using:

- Python function parameters
- Prompt templates
- Model selection (OpenAI or others)
- Language selection 

Add a `.env` file for API configuration:

```
OPENAI_API_KEY=your_key_here
```

<!-- ## 🧩 Extending Trallie

Ways to extend the framework:

- Customize prompt templates
- Add new extractors or normalizers
- Integrate with your NLP or ETL pipelines

-->

<!-- ## 🛠️ Troubleshooting

- **Invalid Schema**: Ensure your rules match expected output formats.
- **Poor Results**: Adjust your prompts or verify model configuration.
- **Rate Limits**: Use batching or rate-limiting with external APIs.

-->

## 📄 License

Trallie is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/PiSchool/trallie/blob/main/LICENSE) file for details.

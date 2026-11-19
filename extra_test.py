import os

from trallie import SchemaGenerator, DataExtractor
from starter_schemas import STARTER_SCHEMAS

# os.environ["GROQ_API_KEY"] = None #ENTER GROQ KEY HERE
# os.environ["OPENAI_API_KEY"] = None #ENTER OPENAI KEY HERE
# Set OPENAI_API_KEY in your environment before running.


FEW_SHOT_GENERATION_SELF_REFLECTION_SYSTEM_PROMPT = """
    You are a helpful database creation assistant, an important part of an AI-powered 
    unstructured to a searchable, queryable structured database creation system. Your 
    job is to discover an entity schema that identified important attributes 
    across different records of a collection of documents and provide it to the user. 
    You must focus on attributes having a precise entity-based answer. Go through the 
    following step-by-step to arrive at the answer:

    Reference Starter Schemas:
    {STARTER_SCHEMAS}

    Step 1: You must identify a set of keywords that contain relevant terms in each 
    document. Combine these terms across all the records in a set. Generate a set of 
    maximum 100 keywords per document. 
    Step 2: Transform the keywords into a set of generic topics to avoid niche attribute 
    names that are specific to a record. 
    Step 3: Identify a set of 10-20 attributes for the schema.
    Step 4: Self-Reflection Phase - Before finalizing your schema, critically evaluate 
    your work by asking yourself:
        - Are all attributes truly generic enough to apply across different records?
        - Do the attributes capture the most important information for database queries?
        - Are there any redundant or overly similar attributes that should be merged?
        - Are there any critical attributes missing that would be valuable for search/filtering?
        - Is each attribute description clear and unambiguous?
        - Would this schema work well for both data entry and retrieval purposes?
    Step 5: Revise and finalize your schema based on the self-reflection insights.

    You must provide the schema in JSON format as below.

    {
    "attribute1":"brief description of what entity/value to extract",
    "attribute2":"brief description of what entity/value to extract"
    }

    You will be provided with a few records from a data collection along with a brief 
    description of the collection to aid you in the process. Following is an example to 
    help you:

    "Wyoming Oil Deal 35 BOPD $1.7m
    Current production: 35 BOPD
    Location: BYRON, Wyoming
    680 Acres N0N-Contigious.
    4-leases with 5 wells.
    Upside is room to drill 7 more wells.
    Producing from Phosphoria formation, and Tensleep Formation.
    NRI average of all 4 leases 79.875%
    Asking $1.7 Million"

    {
    "projectname": "name of the project"
    "industry": "industry or vertical of the project"
    "projectlocation" : "location of the project"
    "projecttype" : "type of the project"
    "productionstatus" : "status of the project"
    "dealtype" : "type of deal"
    "amount" : "amount for the deal"
    }

    In order to respond in valid JSON adhere to the following rules:
        1. Avoid backticks ``` or ```json at the beginning and end of the response.
        2. Enclose all properties in the JSON in double quotes only
        3. Avoid any additional content at the beginning and end of the response.  
        4. Always start and end with curly braces. 

    Now infer the schema of another document.
    You must only return a single final JSON output and not the intermediate outputs.
    Only respond with valid JSON. 
    """


# Define the path to a set of documents/a data collection for inference
records = [
    "data/evaluation/data/fda_510ks/data/evaporate/fda-ai-pmas/510k/K141114.txt",
]

# Provide a description of the data collection
description = "A dataset of Earth observation papers"

# Initialize the schema generator with a provider and model
schema_generator = SchemaGenerator(provider="openai", model_name="gpt-4o", system_prompt=FEW_SHOT_GENERATION_SELF_REFLECTION_SYSTEM_PROMPT)
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

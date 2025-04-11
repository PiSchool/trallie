import os
import glob

from trallie.evaluation.evaluation import evaluate_openie, evaluate_closedie
from trallie.evaluation.evaluation_params import get_evaluation_params, get_dataset_schema
from trallie.wrappers import openie, closedie

from pathlib import Path

os.environ["GROQ_API_KEY"] = None
os.environ["OPENAI_API_KEY"] = None 

base_path = Path("data/evaluation/evaporate/data")
datasets = get_evaluation_params(base_path)

for dataset_name, dataset_info in datasets.items():
    dataset_info["schema"] = get_dataset_schema(dataset_info["ground_truth"])
    records = glob.glob(str(dataset_info["files"]))
    extracted_jsons = openie(dataset_info['description'], records, provider="groq", model_name="llama-3.3-70b-versatile", dataset_name=dataset_name)
    result = evaluate_openie(dataset_info['ground_truth'], f"{dataset_name}_openie_predicted_table.json")
    print(result)
    schema = get_dataset_schema(dataset_info['ground_truth'])
    extracted_jsons = closedie(records, schema, provider="groq", model_name="llama-3.3-70b-versatile", dataset_name=dataset_name)
    result = evaluate_closedie(dataset_info["ground_truth"], f"{dataset_name}_closedie_predicted_table.json")
    print(result)
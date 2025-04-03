import os
import glob

from trallie.evaluation.evaluation import evaluate_openie_sbert, evaluate_value_f1
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
    extracted_jsons = openie(dataset_info['description'], records, provider="openai", model_name="gpt-4o", dataset_name=dataset_name)
    result = evaluate_openie_sbert(dataset_info['ground_truth'], f"{dataset_name}_openie_predicted_table.json")
    schema = get_dataset_schema(dataset_info['ground_truth'])
    extracted_jsons = closedie(records, schema, provider="openai", model_name="gpt-4o", dataset_name=dataset_name)
    result = evaluate_value_f1(fda_ground_truth, f"{dataset_name}_closedie_predicted_table.json")
   
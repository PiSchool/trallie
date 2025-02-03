from trallie.evaluation.schema_evaluation import evaluate_schema_f1_with_llm, evaluate_schema_f1
from trallie.evaluation.evaluation_helpers import *
from trallie.evaluation.gold_schemas import *

import pandas as pd
from datasets import load_dataset

# Industry data Evaluation
print("Industry Report Dataset")
description = "A collection of industry report data"
industry_data_path = "data\evaluation\projectsample_20160630_maxy_v1.csv"
df = pd.read_csv(industry_data_path)
descriptions = df["Description"].dropna().tolist()

grouped_descriptions = group_descriptions(descriptions, group_size=10)
micro_f1, macro_f1 = evaluate_schema_f1_with_llm(description, grouped_descriptions, industry_data_entity_gold_schema)
print("Micro F1 :", micro_f1, "Macro F1 :", macro_f1)

# InLegalNER Evaluation
print("InLegalNER Dataset")
description = "A collection of sentences of legal data"
dataset = load_dataset("opennyaiorg/InLegalNER")
text = dataset['dev']['data']

grouped_sentences = group_descriptions(text, group_size=10)
micro_f1, macro_f1 = evaluate_schema_f1_with_llm(description, grouped_sentences, inlegalner_gold_entity_schema)
print("Micro F1 :", micro_f1, "Macro F1 :", macro_f1)

# MIT Restaurants Evaluation
print("MIT Restaurants Data")
description = "A collection of restaurant reviews"
mit_restaurants_dataset = load_dataset("tner/mit_restaurant")
grouped_eval_input = prepare_eval_input(mit_restaurants_dataset)

micro_f1, macro_f1 = evaluate_schema_f1_with_llm(description, grouped_eval_input, restaurant_data_gold_schema)
print("Micro F1 :", micro_f1, "Macro F1 :", macro_f1)

# MIT Movies Evaluation
print("MIT Movie Data")
description = "A collection of restaurant reviews"
mit_movies_dataset = load_dataset("tner/mit_movie_trivia")
grouped_eval_input = prepare_eval_input(mit_movies_dataset)

micro_f1, macro_f1 = evaluate_schema_f1_with_llm(description, grouped_eval_input, movie_data_gold_schema)
print("Micro F1 :", micro_f1, "Macro F1 :", macro_f1)

# CADEC Dataset
print("CADEC Dataset")
description = "A collection of adverse drug event annotations"
dataset = load_dataset("KevinSpaghetti/cadec")
filtered_dataset = filter_duplicate_text_columns(dataset['test'], 'text')
text = filtered_dataset['text']
grouped_sentences = group_descriptions(text, group_size=10)
micro_f1, macro_f1 = evaluate_schema_f1_with_llm(description, grouped_sentences, medical_entity_schema)
print("Micro F1 :", micro_f1, "Macro F1 :", macro_f1)

# FDA
print("FDA Dataset")
description = "A collection of documents from the FDA 510(k)"
dataset = load_dataset("hazyresearch/fda")
grouped_data = group_text_by_file_name(dataset)
grouped_sentences = group_descriptions(grouped_data, group_size=5)
micro_f1, macro_f1 = evaluate_schema_f1_with_llm(description, grouped_sentences, medical_device_submission_schema)
print("Micro F1 :", micro_f1, "Macro F1 :", macro_f1)

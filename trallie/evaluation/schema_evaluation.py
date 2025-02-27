from trallie.prompts import (FEW_SHOT_INFERENCE_PROMPT, 
                            ZERO_SHOT_INFERENCE_PROMPT,
                            SCHEMA_GENERATION_JUDGE_PROMPT) 
 
from trallie.schema_generation.schema_generator import discover_schema, discover_schema_openai
from trallie.schema_filling.schema_filler import fill_schema
from trallie.evaluation.evaluation_helpers import *

import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
   
def evaluate_schema_f1(description, group_records, golden_schema):
    """
    Pipeline to evaluate schema generation from a CSV file.
    
    Args:
        group_records (List[List[str]]): List of grouped records to evaluate.
        golden_schema (dict): The ground-truth schema for evaluation.
        
    Returns:
        Tuple[float, float]: Micro and Macro F1 scores.
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    f1_scores = []
    golden_keys = extract_keys(golden_schema)
    
    for group in tqdm(group_records):
        # Send group to the LLM via the helper function
        llm_output = discover_schema(description, group, FEW_SHOT_INFERENCE_PROMPT)
        #print(llm_output)
        
        if not validate_json(llm_output):
            continue
        
        predicted_schema = json.loads(llm_output)
        predicted_keys = extract_keys(predicted_schema)
        
        # Compute TP, FP, FN
        tp = len(golden_keys & predicted_keys)
        fp = len(predicted_keys - golden_keys)
        fn = len(golden_keys - predicted_keys)
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

        #print(f1)
       
    # Micro F1 Calculation
    precision_micro = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall_micro = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)) if (precision_micro + recall_micro) > 0 else 0
    
    # Macro F1 Calculation
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return f1_micro, f1_macro


def embedding_sim_sbert(string1, string2, model, threshold=0.5):
    emb1 = model.encode(string1, convert_to_tensor=True)
    emb2 = model.encode(string2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity, similarity >= threshold

def evaluate_schema_f1_with_llm(description, group_records, golden_schema, threshold=0.5):
    """
    Pipeline to evaluate schema generation from a CSV file with semantic key matching.
    
    Args:
        description (str): Description of the schema discovery task.
        group_records (List[List[str]]): List of grouped records to evaluate.
        golden_schema (dict): The ground-truth schema for evaluation.
        threshold (float): Cosine similarity threshold for synonyms.
        
    Returns:
        Tuple[float, float]: Micro and Macro F1 scores.
    """
    # Load SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_tp, all_fp, all_fn = 0, 0, 0
    f1_scores = []
    golden_keys = extract_keys(golden_schema)
    
    for group in tqdm(group_records):
        llm_output = discover_schema(description, group, FEW_SHOT_INFERENCE_PROMPT)
        
        if not validate_json(llm_output):
            continue
        
        predicted_schema = json.loads(llm_output)
        predicted_keys = extract_keys(predicted_schema)
        
        matched_keys = set()
        
        for pred_key in predicted_keys:
            for gold_key in golden_keys:
                similarity, is_synonym = embedding_sim_sbert(pred_key, gold_key, model, threshold)
                if is_synonym:
                    matched_keys.add(pred_key)
                    break
        
        tp = len(matched_keys)
        fp = len(predicted_keys - matched_keys)
        fn = len(golden_keys - matched_keys)
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # Micro F1 Calculation
    precision_micro = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall_micro = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)) if (precision_micro + recall_micro) > 0 else 0
    
    # Macro F1 Calculation
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return f1_micro, f1_macro

###################
def evaluate_entity_f1(records, schema, prompt, model, text_field):
    """
    Evaluates entity-level F1 scores for extracted data against ground truth.
    
    Parameters:
        records (list): List of records, each containing 'full_text', 'keys', and 'values'.
        schema (dict): Schema to be filled.
        prompt (str): Prompt to use for zero-shot filling.
        model: SBERT model for embedding similarity.

    Returns:
        dict: F1-scores per entity, macro F1-score, and micro F1-score.
    """
    entity_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for record in tqdm(records):
        extracted_json = fill_schema(schema, record[text_field], prompt)
        if not validate_json(extracted_json):
            continue
        extracted_data = json.loads(extracted_json)

        for key, ground_truth_value in zip(record['keys'], record['values']):
            extracted_value = extracted_data.get(key, 'N/A')
            score, match_value = embedding_sim_sbert(ground_truth_value, extracted_value, model)
        
            if match_value:
                entity_metrics[key]["tp"] += 1  # True Positive
            else:
                if extracted_value is not None:
                    entity_metrics[key]["fp"] += 1  # False Positive
                entity_metrics[key]["fn"] += 1  # False Negative
    
    entity_f1_scores = {}
    all_tps, all_fps, all_fns = 0, 0, 0

    for entity, metrics in entity_metrics.items():
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        entity_f1_scores[entity] = f1
        all_tps += tp
        all_fps += fp
        all_fns += fn

    macro_f1 = sum(entity_f1_scores.values()) / len(entity_f1_scores) if entity_f1_scores else 0
    micro_precision = all_tps / (all_tps + all_fps) if (all_tps + all_fps) > 0 else 0
    micro_recall = all_tps / (all_tps + all_fns) if (all_tps + all_fns) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {
        "entity_f1_scores": entity_f1_scores,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }

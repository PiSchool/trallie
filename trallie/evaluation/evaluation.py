import json
import os
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from trallie.evaluation.evaluation_helpers import *


def embedding_sim_sbert(text1, text2, model, threshold=0.5):
    """
    Compute SBERT cosine similarity between two text values.

    Returns:
        score (float): Cosine similarity score.
        match (bool): Whether the similarity score is above the threshold.
    """
    if not text1 or not text2:
        return 0.0, False

    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()

    return score, score >= threshold


## Closed IE Evaluation
def evaluate_value_f1(ground_truth_path, predicted_path, value_threshold=0.5):
    """
    Evaluates value-level F1 scores using SBERT for fuzzy matching.

    Parameters:
        ground_truth_path (str): Path to the ground truth JSON file.
        predicted_path (str): Path to the predicted JSON file.
        value_threshold (float): Similarity threshold for value matching.

    Returns:
        dict: F1-score for value correctness.
    """
    # Load SBERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load JSON files
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    with open(predicted_path, "r") as f:
        predicted = json.load(f)

    # Convert predicted JSON string values to dict (if necessary)
    predicted = {
        k: json.loads(v) if isinstance(v, str) else v for k, v in predicted.items()
    }
    ground_truth = {os.path.basename(k): v for k, v in ground_truth.items()}

    # Find common filenames
    common_files = set(ground_truth.keys()) & set(predicted.keys())

    metrics = {"tp": 0, "fp": 0, "fn": 0}

    for filename in tqdm(common_files):
        gt_entities = ground_truth[filename]
        pred_entities = predicted[filename]

        gt_values = [v for val in gt_entities.values() for v in ensure_list(val)]
        pred_values = [v for val in pred_entities.values() for v in ensure_list(val)]

        matched_preds = set()  # To keep track of matched predictions

        for gt_value in gt_values:
            match_found = False
            for idx, pred_value in enumerate(pred_values):
                if idx in matched_preds:
                    continue  # Skip already matched predictions

                score, value_match = embedding_sim_sbert(
                    gt_value, pred_value, model, value_threshold
                )
                if value_match:
                    metrics["tp"] += 1
                    matched_preds.add(idx)
                    match_found = True
                    break  # Stop searching once a match is found

            if not match_found:
                metrics["fn"] += 1  # False Negative

        # False Positives: Remaining unmatched predicted values
        metrics["fp"] += len(pred_values) - len(matched_preds)

    # Compute F1-score
    precision = (
        metrics["tp"] / (metrics["tp"] + metrics["fp"])
        if (metrics["tp"] + metrics["fp"]) > 0
        else 0
    )
    recall = (
        metrics["tp"] / (metrics["tp"] + metrics["fn"])
        if (metrics["tp"] + metrics["fn"]) > 0
        else 0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


## OpenIE Evaluation
def find_best_matching_key(gt_key, pred_keys, model, key_threshold=0.5):
    """
    Finds the best matching key from predicted keys using SBERT similarity.

    Returns:
        best_key (str): Best-matching key from predicted JSON.
        best_score (float): Similarity score.
    """
    if not pred_keys:
        return None, 0.0

    gt_embedding = model.encode(gt_key, convert_to_tensor=True)
    pred_embeddings = model.encode(pred_keys, convert_to_tensor=True)

    scores = util.cos_sim(gt_embedding, pred_embeddings)[0].cpu().numpy()
    best_idx = scores.argmax()
    best_score = scores[best_idx]

    return (
        (pred_keys[best_idx], best_score)
        if best_score >= key_threshold
        else (None, 0.0)
    )


def evaluate_openie_sbert(
    ground_truth_path, predicted_path, key_threshold=0.5, value_threshold=0.5
):
    """
    Evaluates entity and value-level F1 scores using SBERT for fuzzy matching.

    Parameters:
        ground_truth_path (str): Path to the ground truth JSON file.
        predicted_path (str): Path to the predicted JSON file.
        key_threshold (float): Similarity threshold for key matching.
        value_threshold (float): Similarity threshold for value matching.

    Returns:
        dict: F1-score for entity-value correctness.
    """
    # Load SBERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load JSON files
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    with open(predicted_path, "r") as f:
        predicted = json.load(f)

    # Convert predicted JSON string values to dict (if necessary)
    predicted = {
        k: json.loads(v) if isinstance(v, str) else v for k, v in predicted.items()
    }
    ground_truth = {os.path.basename(k): v for k, v in ground_truth.items()}

    # Find common filenames
    common_files = set(ground_truth.keys()) & set(predicted.keys())

    entity_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for filename in tqdm(common_files):
        gt_entities = ground_truth[filename]
        pred_entities = predicted[filename]
        pred_keys = list(pred_entities.keys())

        for gt_key, gt_value in gt_entities.items():
            best_pred_key, key_score = find_best_matching_key(
                gt_key, pred_keys, model, key_threshold
            )

            if best_pred_key:
                pred_value = pred_entities[best_pred_key]
                value_score, value_match = embedding_sim_sbert(
                    gt_value, pred_value, model, value_threshold
                )

                if value_match:
                    entity_metrics[gt_key]["tp"] += 1  # True Positive
                else:
                    entity_metrics[gt_key]["fp"] += 1  # False Positive
                    entity_metrics[gt_key]["fn"] += 1  # False Negative
            else:
                entity_metrics[gt_key]["fn"] += 1  # False Negative (no matching key)

    # Compute F1-scores
    entity_f1_scores = {}
    all_tps, all_fps, all_fns = 0, 0, 0

    for entity, metrics in entity_metrics.items():
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )

        entity_f1_scores[entity] = f1
        all_tps += tp
        all_fps += fp
        all_fns += fn

    macro_f1 = (
        sum(entity_f1_scores.values()) / len(entity_f1_scores)
        if entity_f1_scores
        else 0
    )
    micro_precision = all_tps / (all_tps + all_fps) if (all_tps + all_fps) > 0 else 0
    micro_recall = all_tps / (all_tps + all_fns) if (all_tps + all_fns) > 0 else 0
    micro_f1 = (
        (2 * micro_precision * micro_recall / (micro_precision + micro_recall))
        if (micro_precision + micro_recall) > 0
        else 0
    )

    return {
        "entity_f1_scores": entity_f1_scores,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }

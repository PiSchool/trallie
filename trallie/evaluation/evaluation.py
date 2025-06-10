import json
import os
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def embedding_match(text1, text2, model, threshold=0.5, use_exact_match=False):
    # Return 0 if either text is missing
    if not text1 or not text2:
        return 0.0, False

    # If exact match is enabled, check first
    if use_exact_match and text1.strip().lower() == text2.strip().lower():
        return 1.0, True

    # Fall back to S-BERT similarity
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score, score >= threshold

# def embedding_sim_sbert(text1, text2, model, threshold=0.5):
#     # Compute S-BERT between two pieces of text
#     if not text1 or not text2:
#         return 0.0, False
#     embeddings = model.encode([text1, text2], convert_to_tensor=True)
#     score = util.cos_sim(embeddings[0], embeddings[1]).item()
#     return score, score >= threshold

def evaluate_closedie(ground_truth_path, predicted_path, value_threshold=0.5, use_exact_match=False):
    # Evaluate performance of ClosedIE task
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    with open(predicted_path, "r") as f:
        predicted = json.load(f)
    predicted = {
        k: json.loads(v) if isinstance(v, str) else v for k, v in predicted.items()
    }
    ground_truth = {os.path.basename(k): v for k, v in ground_truth.items()}
    common_files = set(ground_truth.keys()) & set(predicted.keys())
    entity_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for filename in tqdm(common_files):
        gt_entities = ground_truth[filename] or {}
        pred_entities = predicted.get(filename) or {}
        for gt_key, gt_vals in gt_entities.items():
            gt_values = gt_vals if isinstance(gt_vals, list) else [gt_vals]
            pred_raw = pred_entities.get(gt_key, [])
            pred_values = pred_raw if isinstance(pred_raw, list) else [pred_raw]
            if not pred_values:
                entity_metrics[gt_key]["fn"] += len(gt_values)
                continue
            matched_preds = set()
            for gt_value in gt_values:
                match_found = False
                for idx, pred_value in enumerate(pred_values):
                    if idx in matched_preds:
                        continue
                    score, match = embedding_match(
                        gt_value, pred_value, model, value_threshold, use_exact_match=use_exact_match
                    )
                    if match:
                        entity_metrics[gt_key]["tp"] += 1
                        matched_preds.add(idx)
                        match_found = True
                        break
                if not match_found:
                    entity_metrics[gt_key]["fn"] += 1
            entity_metrics[gt_key]["fp"] += len(pred_values) - len(matched_preds)
    return _compute_f1_metrics(entity_metrics)

def find_best_matching_key(gt_key, pred_keys, model, key_threshold=0.8):
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

def evaluate_openie(ground_truth_path, predicted_path, key_threshold=0.5, value_threshold=0.5, use_exact_match=False):
    # Evaluate OpenIE task
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    with open(predicted_path, "r") as f:
        predicted = json.load(f)
    predicted = {
        k: json.loads(v) if isinstance(v, str) else v for k, v in predicted.items()
    }
    ground_truth = {os.path.basename(k): v for k, v in ground_truth.items()}
    common_files = set(ground_truth.keys()) & set(predicted.keys())
    entity_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for filename in tqdm(common_files):
        gt_entities = ground_truth[filename] or {}
        pred_entities = predicted.get(filename) or {}
        pred_keys = list(pred_entities.keys())
        for gt_key, gt_value in gt_entities.items():
            best_pred_key, key_score = find_best_matching_key(
                gt_key, pred_keys, model, key_threshold
            )
            if best_pred_key:
                pred_value = pred_entities[best_pred_key]
                value_score, value_match = embedding_match(
                    gt_value, pred_value, model, value_threshold, use_exact_match=use_exact_match
                )
                if value_match:
                    entity_metrics[gt_key]["tp"] += 1
                else:
                    entity_metrics[gt_key]["fp"] += 1
                    entity_metrics[gt_key]["fn"] += 1
            else:
                entity_metrics[gt_key]["fn"] += 1
    return _compute_f1_metrics(entity_metrics)

def _compute_f1_metrics(entity_metrics):
    entity_f1_scores = {}
    all_tps, all_fps, all_fns = 0, 0, 0
    for entity, metrics in entity_metrics.items():
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
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
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
    }

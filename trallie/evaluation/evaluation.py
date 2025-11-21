"""
Evaluation utilities for OpenIE and ClosedIE
"""
import json
from typing import Dict, Any
from collections import defaultdict


def normalize_value(value: Any) -> str:
    """Normalize a value for comparison"""
    if value is None:
        return ""
    return str(value).strip().lower()


def calculate_field_match(pred_value: Any, gt_value: Any, use_exact_match: bool = True) -> float:
    """
    Calculate match score between predicted and ground truth values.
    
    Args:
        pred_value: Predicted value
        gt_value: Ground truth value
        use_exact_match: If True, use exact matching; otherwise use partial matching
        
    Returns:
        Match score (1.0 for exact match, 0.0 for no match, or partial score)
    """
    pred_norm = normalize_value(pred_value)
    gt_norm = normalize_value(gt_value)
    
    if not pred_norm or not gt_norm:
        return 0.0
    
    if pred_norm == gt_norm:
        return 1.0
    
    if not use_exact_match:
        # Partial matching using containment
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return 0.8
        
        # Jaccard similarity for word-level matching
        pred_words = set(pred_norm.split())
        gt_words = set(gt_norm.split())
        
        if pred_words and gt_words:
            intersection = len(pred_words & gt_words)
            union = len(pred_words | gt_words)
            return intersection / union if union > 0 else 0.0
    
    return 0.0


def evaluate_openie(ground_truth_path: str, predicted_path: str, use_exact_match: bool = True) -> Dict[str, Any]:
    """
    Evaluate OpenIE results against ground truth.
    
    Args:
        ground_truth_path: Path to ground truth JSON file
        predicted_path: Path to predicted results JSON file
        use_exact_match: Whether to use exact matching
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load ground truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Load predictions
    with open(predicted_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    total_fields = 0
    matched_fields = 0
    partial_matches = 0
    field_scores = defaultdict(list)
    
    # Evaluate each document
    for doc_id, gt_data in ground_truth.items():
        # Find matching prediction (handle different path formats)
        pred_data = None
        doc_basename = doc_id.split('/')[-1]
        
        for pred_id, pred in predictions.items():
            if doc_basename in pred_id or pred_id in doc_basename:
                pred_data = pred
                break
        
        if pred_data is None:
            # No prediction found for this document
            total_fields += len(gt_data)
            continue
        
        # Compare fields
        for gt_field, gt_value in gt_data.items():
            total_fields += 1
            
            # Try to find matching field in predictions (case-insensitive)
            best_score = 0.0
            for pred_field, pred_value in pred_data.items():
                if gt_field.lower() == pred_field.lower():
                    score = calculate_field_match(pred_value, gt_value, use_exact_match)
                    best_score = max(best_score, score)
                    field_scores[gt_field].append(score)
            
            if best_score >= 0.8:
                matched_fields += 1
            elif best_score >= 0.4:
                partial_matches += 1
    
    # Calculate metrics
    precision = matched_fields / total_fields if total_fields > 0 else 0.0
    recall = matched_fields / total_fields if total_fields > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Field-level metrics
    field_accuracy = {}
    for field, scores in field_scores.items():
        field_accuracy[field] = sum(scores) / len(scores) if scores else 0.0
    
    return {
        'total_fields': total_fields,
        'matched_fields': matched_fields,
        'partial_matches': partial_matches,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': matched_fields / total_fields if total_fields > 0 else 0.0,
        'field_accuracy': field_accuracy
    }


def evaluate_closedie(ground_truth_path: str, predicted_path: str, use_exact_match: bool = True) -> Dict[str, Any]:
    """
    Evaluate ClosedIE results against ground truth.
    Same as evaluate_openie but with explicit schema matching.
    """
    return evaluate_openie(ground_truth_path, predicted_path, use_exact_match)


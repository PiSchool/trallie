#!/usr/bin/env python3
"""
Trallie Benchmarking Tool
Core evaluation functionality for comparing model performance.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import pandas as pd


class TrallieBenchmarker:
    """
    Benchmarking tool for Trallie evaluation results.
    """
    
    def __init__(self, eval_results_dir: str = "trallie-eval-result", 
                 ground_truth_dir: str = "data/evaporate/data"):
        """
        Initialize the benchmarker.
        
        Args:
            eval_results_dir: Directory containing evaluation result files
            ground_truth_dir: Directory containing ground truth data
        """
        self.eval_results_dir = Path(eval_results_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.eval_results = {}
        self.ground_truth = {}
        self.metrics = {}
    
    def load_evaluation_results(self) -> Dict[str, Any]:
        """
        Load evaluation results from JSON files.
        
        Returns:
            Dictionary mapping model keys to evaluation data
        """
        print("Loading evaluation results...")
        
        if not self.eval_results_dir.exists():
            print(f"Evaluation results directory not found: {self.eval_results_dir}")
            return {}
        
        # Load all JSON files in the directory
        for json_file in self.eval_results_dir.glob("*.json"):
            if json_file.name.endswith('_predicted_table.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract model and dataset info from filename
                    filename = json_file.stem.replace('_predicted_table', '')
                    parts = filename.split('_')
                    
                    # Find the approach (openie/closedie) in the filename
                    approach = None
                    if 'openie' in filename.lower():
                        approach = 'openie'
                    elif 'closedie' in filename.lower():
                        approach = 'closedie'
                    
                    # Extract dataset name (everything between model and approach)
                    dataset = None
                    if approach:
                        # Remove the approach part to get dataset
                        dataset_part = filename.replace(f'_{approach}', '')
                        # Known dataset patterns
                        known_datasets = [
                            'fda_510ks',
                            'swde_movie_allmovie', 
                            'swde_movie_amctv',
                            'swde_movie_hollywood',
                            'swde_movie_iheartmovies'
                        ]
                        for known_dataset in known_datasets:
                            if known_dataset in dataset_part:
                                dataset = known_dataset
                                break
                    
                    if not dataset:
                        dataset = "unknown"
                    
                    # Model name is everything before the dataset
                    model_name = filename.replace(f'_{dataset}_{approach}', '') if approach else filename
                    
                    self.eval_results[filename] = {
                        'model': model_name,
                        'dataset': dataset,
                        'extraction_type': approach or 'unknown',
                        'data': data
                    }
                    
                    print(f"Loaded: {filename} ({len(data)} documents)")
                    
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.eval_results)} evaluation result files")
        return self.eval_results
    
    def load_ground_truth(self) -> Dict[str, Dict[str, Any]]:
        """
        Load ground truth data from JSON files.
        
        Returns:
            Dictionary mapping dataset names to ground truth data
        """
        if not self.ground_truth_dir.exists():
            print(f"Ground truth directory not found: {self.ground_truth_dir}")
            return {}
        
        ground_truth = {}
        
        # Load ground truth for each dataset
        for dataset_dir in self.ground_truth_dir.iterdir():
            if dataset_dir.is_dir():
                table_file = dataset_dir / "table.json"
                if table_file.exists():
                    try:
                        # Try multiple encodings
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                with open(table_file, 'r', encoding=encoding) as f:
                                    data = json.load(f)
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            print(f"Error loading ground truth {table_file}: encoding issues")
                            continue
                        
                        ground_truth[dataset_dir.name] = data
                        print(f"Loaded ground truth for dataset: {dataset_dir.name}")
                        
                    except Exception as e:
                        print(f"Error loading ground truth {table_file}: {e}")
        
        print(f"Loaded {len(ground_truth)} ground truth datasets")
        return ground_truth
    
    def normalize_field_names(self, field_name: str) -> str:
        """
        Normalize field names for comparison.
        
        Args:
            field_name: Original field name
            
        Returns:
            Normalized field name
        """
        # Remove special characters and convert to lowercase
        normalized = field_name.lower()
        normalized = normalized.replace('_', ' ').replace('-', ' ')
        normalized = ' '.join(normalized.split())  # Remove extra whitespace
        
        # Common field name mappings
        field_mappings = {
            'company': 'company',
            'organization': 'organization',
            'org': 'organization',
            'name': 'name',
            'title': 'title',
            'date': 'date',
            'time': 'time',
            'location': 'location',
            'address': 'address',
            'phone': 'phone',
            'email': 'email',
            'url': 'url',
            'website': 'website',
            'price': 'price',
            'cost': 'price',
            'rating': 'rating',
            'score': 'rating',
            'genre': 'genre',
            'category': 'category',
            'type': 'type',
            'status': 'status'
        }
        
        return field_mappings.get(normalized, normalized)
    
    def calculate_field_similarity(self, pred_value: Any, gt_value: Any) -> float:
        """
        Calculate similarity between predicted and ground truth values.
        
        Args:
            pred_value: Predicted value
            gt_value: Ground truth value
            
        Returns:
            Similarity score between 0 and 1
        """
        if pred_value is None or gt_value is None:
            return 0.0
        
        # Convert to strings for comparison
        pred_str = str(pred_value).strip().lower()
        gt_str = str(gt_value).strip().lower()
        
        if not pred_str or not gt_str:
            return 0.0
        
        # Exact match
        if pred_str == gt_str:
            return 1.0
        
        # Check if predicted value contains ground truth or vice versa
        if pred_str in gt_str or gt_str in pred_str:
            return 0.8
        
        # Calculate Jaccard similarity for partial matches
        pred_words = set(pred_str.split())
        gt_words = set(gt_str.split())
        
        if not pred_words or not gt_words:
            return 0.0
        
        intersection = len(pred_words & gt_words)
        union = len(pred_words | gt_words)
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_single_model(self, model_name: str, dataset: str, extraction_type: str, 
                            predicted_data: Dict[str, Any], ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a single model's performance.
        
        Args:
            model_name: Name of the model
            dataset: Dataset name
            extraction_type: Type of extraction (openie/closedie)
            predicted_data: Predicted data from the model
            ground_truth: Ground truth data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Find matching ground truth dataset
        gt_dataset = None
        for gt_name in ground_truth.keys():
            # More precise matching for SWDE movie datasets
            if dataset == gt_name:
                gt_dataset = gt_name
                break
            # For SWDE movie datasets, check exact match first
            elif dataset.startswith('swde_movie_') and gt_name.startswith('swde_movie_'):
                if dataset == gt_name:
                    gt_dataset = gt_name
                    break
            # For other datasets, use flexible matching
            elif (dataset.lower() in gt_name.lower() or 
                  gt_name.lower() in dataset.lower()):
                gt_dataset = gt_name
                break
        
        if gt_dataset is None:
            print(f"Ground truth not found for dataset: {dataset}")
            return {}
        
        print(f"Using ground truth dataset: {gt_dataset} for evaluation")
        
        # Initialize metrics
        total_documents = 0
        total_fields = 0
        correct_fields = 0
        partial_matches = 0
        field_metrics = defaultdict(list)
        
        # Process each document
        for doc_id, pred_doc in predicted_data.items():
            # Find corresponding ground truth document
            gt_doc = None
            for gt_doc_id, gt_doc_data in ground_truth[gt_dataset].items():
                if doc_id in gt_doc_id or gt_doc_id in doc_id:
                    gt_doc = gt_doc_data
                    break
                    
            if gt_doc is None:
                continue
                
            # Check if pred_doc is valid
            if pred_doc is None:
                continue
                
            total_documents += 1
            
            # Compare fields
            for gt_field, gt_value in gt_doc.items():
                total_fields += 1
                
                # Find best matching predicted field
                best_similarity = 0.0
                best_pred_field = None
                best_pred_value = None
                
                for pred_field, pred_value in pred_doc.items():
                    # Normalize field names
                    norm_gt_field = self.normalize_field_names(gt_field)
                    norm_pred_field = self.normalize_field_names(pred_field)
                    
                    if norm_gt_field == norm_pred_field:
                        similarity = self.calculate_field_similarity(pred_value, gt_value)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_pred_field = pred_field
                            best_pred_value = pred_value
                
                # Record field-level metrics
                field_metrics[gt_field].append({
                    'gt_value': gt_value,
                    'pred_value': best_pred_value,
                    'similarity': best_similarity
                })
                
                if best_similarity >= 0.8:
                    correct_fields += 1
                elif best_similarity >= 0.4:
                    partial_matches += 1
        
        # Calculate overall metrics
        precision = correct_fields / total_fields if total_fields > 0 else 0
        recall = correct_fields / total_fields if total_fields > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        partial_accuracy = (correct_fields + partial_matches * 0.5) / total_fields if total_fields > 0 else 0
        
        # Calculate field-level accuracy
        field_accuracy = {}
        for field, metrics_list in field_metrics.items():
            if metrics_list:
                avg_similarity = sum(m['similarity'] for m in metrics_list) / len(metrics_list)
                field_accuracy[field] = avg_similarity
        
        return {
            'model': model_name,
            'dataset': dataset,
            'extraction_type': extraction_type,
            'total_documents': total_documents,
            'total_fields': total_fields,
            'correct_fields': correct_fields,
            'partial_matches': partial_matches,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'partial_accuracy': partial_accuracy,
            'field_accuracy': field_accuracy
        }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all models and datasets.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("Running comprehensive evaluation...")
        
        # Load data if not already loaded
        if not self.eval_results:
            self.load_evaluation_results()
        if not self.ground_truth:
            self.load_ground_truth()
        
        # Evaluate each model
        for model_key, model_data in self.eval_results.items():
            model_name = model_data['model']
            dataset = model_data['dataset']
            extraction_type = model_data['extraction_type']
            predicted_data = model_data['data']
            
            print(f"Evaluating {model_key}...")
            self.metrics[model_key] = self.evaluate_single_model(model_name, dataset, extraction_type, predicted_data, self.ground_truth)
        
        return self.metrics

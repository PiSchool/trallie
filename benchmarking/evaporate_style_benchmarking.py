#!/usr/bin/env python3
"""
Evaporate-Style Benchmarking for Trallie
Implements token-level similarity matching like Evaporate does
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter, defaultdict
import numpy as np


class EvaporateStyleBenchmarker:
    """
    Evaporate's approach:
    - Token-level similarity matching (SQuAD-style)
    - Content-focused evaluation, not schema-focused
    - Partial credit for overlapping content
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
    
    def clean_comparison(self, text: str, field: str = "") -> str:
        """
        Clean text for comparison (following Evaporate's approach).
        
        Args:
            text: Text to clean
            field: Field name (for context)
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        field = field.lower()
        
        # Remove punctuation and special characters (like Evaporate does)
        for char in ["'", ":", "<", ">", '"', "none", ",", ".", "?", "!", ";", "(", ")", "[", "]", "{", "}", "-", "\n", "\t", "\r"]:
            text = text.replace(char, " ")
        
        # Split into tokens and rejoin
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t]
        return " ".join(tokens)
    
    def normalize_value_type(self, metadata: Any, attribute: str = "") -> List[str]:
        """
        Normalize metadata to list of strings (following Evaporate's approach).
        
        Args:
            metadata: Metadata to normalize
            attribute: Attribute name
            
        Returns:
            List of normalized strings
        """
        cleaned_items = []
        
        if isinstance(metadata, str):
            metadata = [metadata]
        elif isinstance(metadata, (int, float)):
            metadata = [str(metadata)]
        elif metadata is None:
            return []
        elif not isinstance(metadata, (list, tuple)):
            metadata = [str(metadata)]
        
        for item in metadata:
            if isinstance(item, list):
                item = [str(i) for i in item]
                item = ", ".join(item)
            elif isinstance(item, tuple):
                item = list(item)
                item = [str(i) for i in item]
                item = ", ".join(item)
            elif item is None:
                item = ''
            elif not isinstance(item, str):
                item = str(item)
            
            if item:
                cleaned_items.append(item)
        
        return cleaned_items
    
    def text_f1(self, preds: List[str], golds: List[str], 
                extraction_fraction: float = 1.0, 
                attribute: str = None,
                extraction_fraction_thresh: float = 0.8,
                use_abstension: bool = True) -> tuple:
        """
        Compute average F1 of text spans (following Evaporate's SQuAD-style approach).
        
        Args:
            preds: List of predicted texts
            golds: List of gold standard texts
            extraction_fraction: Fraction of extractions
            attribute: Attribute name
            extraction_fraction_thresh: Threshold for abstension
            use_abstension: Whether to use abstension
            
        Returns:
            Tuple of (average_f1, median_f1)
        """
        total_f1 = 0
        total_recall = 0
        total_prec = 0
        f1s = []
        total = 0
        
        # Handle abstensions (like Evaporate does)
        if extraction_fraction >= extraction_fraction_thresh and use_abstension:
            new_preds = []
            new_golds = []
            for pred, gold in zip(preds, golds):
                if pred:
                    new_preds.append(pred)
                    new_golds.append(gold)
            preds = new_preds
            golds = new_golds
            if not preds:
                return 0.0, 0.0
        
        for pred, gold in zip(preds, golds):
            if isinstance(pred, str):
                pred_toks = pred.split()
            else:
                pred_toks = pred
            
            if isinstance(gold, str):
                gold_toks_list = [gold.split()]
            else:
                gold_toks_list = gold
            
            if isinstance(gold_toks_list, list) and gold_toks_list:
                for gold_toks in gold_toks_list:
                    # Handle single token case (like Evaporate does)
                    if len(gold_toks) == 1 and len(pred_toks) == 1:
                        gold_toks = gold_toks[0].split()
                        pred_toks = pred_toks[0].split()
                    
                    # Count common tokens (intersection)
                    common = Counter(pred_toks) & Counter(gold_toks)
                    num_same = sum(common.values())
                    
                    if len(gold_toks) == 0 or len(pred_toks) == 0:
                        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                        total_f1 += int(gold_toks == pred_toks)
                        f1s.append(int(gold_toks == pred_toks))
                        total_recall += int(gold_toks == pred_toks)
                    elif num_same == 0:
                        total_f1 += 0
                        f1s.append(0)
                    else:
                        # Calculate precision and recall
                        precision = 1.0 * num_same / len(pred_toks)
                        recall = 1.0 * num_same / len(gold_toks)
                        f1 = (2 * precision * recall) / (precision + recall)
                        total_f1 += f1
                        total_recall += recall
                        total_prec += precision
                        f1s.append(f1)
                    
                    total += 1
        
        if not total:
            return 0.0, 0.0
        
        f1_avg = total_f1 / total
        f1_median = np.percentile(f1s, 50) if f1s else 0.0
        
        return f1_avg, f1_median
    
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
                    
                    # Extract dataset name
                    dataset = None
                    if approach:
                        dataset_part = filename.replace(f'_{approach}', '')
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
    
    def evaluate_single_model_evaporate_style(self, model_name: str, dataset: str, 
                                            extraction_type: str, predicted_data: Dict[str, Any], 
                                            ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a single model using Evaporate's approach.
        
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
            if dataset == gt_name:
                gt_dataset = gt_name
                break
            elif dataset.startswith('swde_movie_') and gt_name.startswith('swde_movie_'):
                if dataset == gt_name:
                    gt_dataset = gt_name
                    break
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
            
            # For each ground truth field, find best matching predicted field
            for gt_field, gt_value in gt_doc.items():
                # Normalize ground truth value
                gt_values = self.normalize_value_type(gt_value, gt_field)
                gt_cleaned = [self.clean_comparison(str(v), gt_field) for v in gt_values]
                
                best_f1 = 0.0
                best_pred_field = None
                best_pred_values = []
                
                # Try to match with any predicted field (content-based, not field-name-based)
                for pred_field, pred_value in pred_doc.items():
                    # Normalize predicted value
                    pred_values = self.normalize_value_type(pred_value, pred_field)
                    pred_cleaned = [self.clean_comparison(str(v), pred_field) for v in pred_values]
                    
                    # Calculate F1 using Evaporate's approach
                    f1, _ = self.text_f1(pred_cleaned, gt_cleaned, attribute=gt_field)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_pred_field = pred_field
                        best_pred_values = pred_cleaned
                
                # Record field-level metrics
                field_metrics[gt_field].append({
                    'gt_value': gt_cleaned,
                    'pred_value': best_pred_values,
                    'f1_score': best_f1,
                    'matched_field': best_pred_field
                })
        
        # Calculate overall metrics
        all_f1_scores = []
        for field, metrics_list in field_metrics.items():
            for metric in metrics_list:
                all_f1_scores.append(metric['f1_score'])
        
        if all_f1_scores:
            avg_f1 = np.mean(all_f1_scores)
            median_f1 = np.median(all_f1_scores)
        else:
            avg_f1 = 0.0
            median_f1 = 0.0
        
        return {
            'model': model_name,
            'dataset': dataset,
            'extraction_type': extraction_type,
            'total_documents': total_documents,
            'total_fields_evaluated': len(all_f1_scores),
            'average_f1': avg_f1,
            'median_f1': median_f1,
            'field_metrics': dict(field_metrics)
        }
    
    def run_evaporate_style_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation using Evaporate's approach.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("Running Evaporate-style evaluation...")
        
        # Load data if not already loaded
        if not self.eval_results:
            self.load_evaluation_results()
        if not self.ground_truth:
            self.ground_truth = self.load_ground_truth()
        
        # Evaluate each model
        for model_key, model_data in self.eval_results.items():
            model_name = model_data['model']
            dataset = model_data['dataset']
            extraction_type = model_data['extraction_type']
            predicted_data = model_data['data']
            
            print(f"Evaluating {model_key} using Evaporate-style approach...")
            self.metrics[model_key] = self.evaluate_single_model_evaporate_style(
                model_name, dataset, extraction_type, predicted_data, self.ground_truth
            )
        
        return self.metrics
    
    def display_results(self):
        """
        Display the evaluation results in a table format.
        """
        if not self.metrics:
            print("❌ No metrics available for display")
            return
        
        print("=" * 100)
        print("🏆 EVAPORATE-STYLE BENCHMARKING RESULTS")
        print("=" * 100)
        
        # Create table data
        table_data = []
        
        for model_key, metrics in self.metrics.items():
            if not metrics:
                continue
                
            model = metrics.get('model', 'Unknown')
            dataset = metrics.get('dataset', 'Unknown')
            approach = metrics.get('extraction_type', 'Unknown').upper()
            avg_f1 = metrics.get('average_f1', 0.0)
            median_f1 = metrics.get('median_f1', 0.0)
            total_fields = metrics.get('total_fields_evaluated', 0)
            
            table_data.append({
                'Model': model,
                'Dataset': dataset,
                'Approach': approach,
                'Avg F1': f"{avg_f1:.3f}",
                'Median F1': f"{median_f1:.3f}",
                'Fields': total_fields
            })
        
        # Display the table
        if table_data:
            # Print header
            print(f"{'Model':<40} {'Dataset':<25} {'Approach':<12} {'Avg F1':<10} {'Median F1':<12} {'Fields':<8}")
            print("-" * 100)
            
            # Print rows
            for row in table_data:
                print(f"{row['Model']:<40} {row['Dataset']:<25} {row['Approach']:<12} {row['Avg F1']:<10} {row['Median F1']:<12} {row['Fields']:<8}")
            
            print("-" * 100)
            print()
            
            # Summary statistics
            print("📈 SUMMARY STATISTICS:")
            print("-" * 50)
            
            # Group by approach
            closedie_metrics = [m for m in self.metrics.values() if m and 'closedie' in m.get('extraction_type', '').lower()]
            openie_metrics = [m for m in self.metrics.values() if m and 'openie' in m.get('extraction_type', '').lower()]
            
            if closedie_metrics:
                avg_closedie_f1 = np.mean([m.get('average_f1', 0.0) for m in closedie_metrics])
                print(f"🔒 CLOSEDIE Average F1 Score: {avg_closedie_f1:.3f} ({len(closedie_metrics)} models)")
            
            if openie_metrics:
                avg_openie_f1 = np.mean([m.get('average_f1', 0.0) for m in openie_metrics])
                print(f"🔓 OPENIE Average F1 Score: {avg_openie_f1:.3f} ({len(openie_metrics)} models)")
            
            # Best performing model
            best_model = max([m for m in self.metrics.values() if m], key=lambda x: x.get('average_f1', 0.0))
            print(f"🏆 Best Overall Model: {best_model['model']} (F1: {best_model.get('average_f1', 0.0):.3f})")
            
            print(f"\n💡 Key Insight: This approach evaluates CONTENT similarity, not field name alignment!")
            print(f"   OPENIE models should now score much better than with field-level matching.")
        else:
            print("❌ No data available for display")


def main():
    """Main function to run Evaporate-style benchmarking."""
    
    print("🔍 Loading Trallie evaluation results with Evaporate-style approach...")
    
    # Initialize the benchmarker
    benchmarker = EvaporateStyleBenchmarker()
    
    # Load evaluation results
    benchmarker.load_evaluation_results()
    
    # Load ground truth
    benchmarker.ground_truth = benchmarker.load_ground_truth()
    
    if not benchmarker.eval_results:
        print("❌ No evaluation results found!")
        return
    
    if not benchmarker.ground_truth:
        print("❌ No ground truth data found!")
        return
    
    print("✅ Data loaded successfully!")
    print(f"📊 Found {len(benchmarker.eval_results)} evaluation result files")
    print(f"🎯 Found {len(benchmarker.ground_truth)} ground truth datasets")
    print()
    
    # Run Evaporate-style evaluation
    print("🔄 Running Evaporate-style evaluation...")
    benchmarker.run_evaporate_style_evaluation()
    
    if not benchmarker.metrics:
        print("❌ No metrics generated!")
        return
    
    print("✅ Evaluation completed!")
    print()
    
    # Display results
    benchmarker.display_results()


if __name__ == "__main__":
    main()

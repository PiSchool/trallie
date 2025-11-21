#!/usr/bin/env python3
"""Evaluate all datasets using SBERT evaluation for both no-memory and memory."""

import json
from pathlib import Path
from tqdm import tqdm

from trallie.evaluation.evaluation_params import get_evaluation_params

# Import SBERT evaluation
import importlib.util
eval_sbert_path = Path(__file__).parent.parent / "eval_sbert" / "eval_sbert.py"
spec = importlib.util.spec_from_file_location("eval_sbert", eval_sbert_path)
eval_sbert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_sbert_module)
evaluate_openie_sbert = eval_sbert_module.evaluate_openie

BASE_PATH = Path("evaporate/data")
RESULTS_ROOT = Path("results")
SKIP_DATASETS = {""}  # Add dataset names to skip if needed
MODEL_NAME = "gpt-4o-mini"


def evaluate_all_datasets():
    """Evaluate all datasets using SBERT evaluation."""
    # Get all datasets
    datasets = get_evaluation_params(BASE_PATH)
    
    # Get all result directories
    result_dirs = [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and d.name not in SKIP_DATASETS]
    
    print(f"Found {len(result_dirs)} datasets to evaluate (excluding {', '.join(SKIP_DATASETS)})")
    print()
    
    results_summary = {}
    
    for dataset_dir in tqdm(sorted(result_dirs), desc="Evaluating datasets"):
        dataset_name = dataset_dir.name
        
        if dataset_name not in datasets:
            print(f"Warning: {dataset_name} not found in evaluation params, skipping...")
            continue
        
        dataset_info = datasets[dataset_name]
        ground_truth_path = dataset_info["ground_truth"]
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*70}")
        
        results_summary[dataset_name] = {}
        
        # Evaluate no-memory
        nomemory_pred_path = dataset_dir / f"{MODEL_NAME}_openie_nomemory" / "predictions.json"
        if nomemory_pred_path.exists():
            print(f"\n[NO MEMORY] Running SBERT evaluation...")
            try:
                eval_sbert = evaluate_openie_sbert(
                    ground_truth_path,
                    str(nomemory_pred_path)
                )
                output_dir = dataset_dir / f"{MODEL_NAME}_openie_nomemory"
                with open(output_dir / "sbert_eval.json", "w", encoding="utf-8") as f:
                    json.dump(eval_sbert, f, indent=2)
                
                print(f"  Micro F1: {eval_sbert['micro_f1']:.4f}")
                print(f"  Micro Precision: {eval_sbert['micro_precision']:.4f}")
                print(f"  Micro Recall: {eval_sbert['micro_recall']:.4f}")
                print(f"  Macro F1: {eval_sbert['macro_f1']:.4f}")
                
                results_summary[dataset_name]["no_memory"] = eval_sbert
            except Exception as e:
                print(f"  Error evaluating no-memory: {e}")
                results_summary[dataset_name]["no_memory"] = {"error": str(e)}
        else:
            print(f"[NO MEMORY] Predictions file not found: {nomemory_pred_path}")
        
        # Evaluate memory
        memory_pred_path = dataset_dir / f"{MODEL_NAME}_openie_memory" / "predictions.json"
        if memory_pred_path.exists():
            print(f"\n[MEMORY] Running SBERT evaluation...")
            try:
                eval_sbert = evaluate_openie_sbert(
                    ground_truth_path,
                    str(memory_pred_path)
                )
                output_dir = dataset_dir / f"{MODEL_NAME}_openie_memory"
                with open(output_dir / "sbert_eval.json", "w", encoding="utf-8") as f:
                    json.dump(eval_sbert, f, indent=2)
                
                print(f"  Micro F1: {eval_sbert['micro_f1']:.4f}")
                print(f"  Micro Precision: {eval_sbert['micro_precision']:.4f}")
                print(f"  Micro Recall: {eval_sbert['micro_recall']:.4f}")
                print(f"  Macro F1: {eval_sbert['macro_f1']:.4f}")
                
                results_summary[dataset_name]["memory"] = eval_sbert
            except Exception as e:
                print(f"  Error evaluating memory: {e}")
                results_summary[dataset_name]["memory"] = {"error": str(e)}
        else:
            print(f"[MEMORY] Predictions file not found: {memory_pred_path}")
    
    # Save summary
    summary_path = RESULTS_ROOT / "all_datasets_sbert_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("All evaluations complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Dataset':<40} {'No-Memory Micro F1':<20} {'Memory Micro F1':<20}")
    print("-" * 80)
    for dataset_name, results in sorted(results_summary.items()):
        no_mem_f1 = results.get("no_memory", {}).get("micro_f1", "N/A")
        mem_f1 = results.get("memory", {}).get("micro_f1", "N/A")
        if isinstance(no_mem_f1, float):
            no_mem_f1 = f"{no_mem_f1:.4f}"
        if isinstance(mem_f1, float):
            mem_f1 = f"{mem_f1:.4f}"
        print(f"{dataset_name:<40} {no_mem_f1:<20} {mem_f1:<20}")


if __name__ == "__main__":
    evaluate_all_datasets()


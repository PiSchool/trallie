#!/usr/bin/env python3
"""
Trallie Benchmark Display Script
Shows a single, focused table comparing closedie vs openie performance across models.
"""

import sys
import os
from pathlib import Path
from benchmarking import TrallieBenchmarker


def main():
    """Display a single benchmarking table focusing on closedie vs openie performance."""

    # Initialize the benchmarker
    benchmarker = TrallieBenchmarker()

    print("🔍 Loading Trallie evaluation results...")

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

    # Run comprehensive evaluation
    print("🔄 Running comprehensive evaluation...")
    benchmarker.run_comprehensive_evaluation()

    if not benchmarker.metrics:
        print("❌ No metrics generated!")
        return

    print("✅ Evaluation completed!")
    print()

    # Generate and display the main performance table
    print("=" * 100)
    print("🏆 TRALLIE BENCHMARKING RESULTS - CLOSEDIE vs OPENIE PERFORMANCE")
    print("=" * 100)

    # Create a focused table showing closedie vs openie performance
    table_data = []

    for model_key, metrics in benchmarker.metrics.items():
        if not metrics:  # Skip empty metrics
            continue
            
        # Use the metadata already stored in metrics
        model = metrics.get('model', 'Unknown')
        dataset = metrics.get('dataset', 'Unknown')
        approach = metrics.get('extraction_type', 'Unknown').upper()
        
        # Extract key metrics
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1_score = metrics.get('f1_score', 0.0)
        partial_accuracy = metrics.get('partial_accuracy', 0.0)

        table_data.append({
            'Model': model,
            'Dataset': dataset,
            'Approach': approach,
            'Precision': f"{precision:.3f}",
            'Recall': f"{recall:.3f}",
            'F1 Score': f"{f1_score:.3f}",
            'Partial Acc.': f"{partial_accuracy:.3f}"
        })

    # Display the table
    if table_data:
        # Print header with proper spacing
        print(f"{'Model':<40} {'Dataset':<25} {'Approach':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Partial Acc.':<12}")
        print("-" * 100)
        
        # Print rows with proper spacing
        for row in table_data:
            print(f"{row['Model']:<40} {row['Dataset']:<25} {row['Approach']:<12} {row['Precision']:<12} {row['Recall']:<12} {row['F1 Score']:<12} {row['Partial Acc.']:<12}")
        
        print("-" * 100)
        print()

        # Summary statistics
        print("📈 SUMMARY STATISTICS:")
        print("-" * 50)

        # Group by approach using the extraction_type from metrics
        closedie_metrics = [m for m in benchmarker.metrics.values() if m and 'closedie' in m.get('extraction_type', '').lower()]
        openie_metrics = [m for m in benchmarker.metrics.values() if m and 'openie' in m.get('extraction_type', '').lower()]

        if closedie_metrics:
            avg_closedie_f1 = sum(m.get('f1_score', 0.0) for m in closedie_metrics) / len(closedie_metrics)
            print(f"🔒 CLOSEDIE Average F1 Score: {avg_closedie_f1:.3f} ({len(closedie_metrics)} datasets)")

        if openie_metrics:
            avg_openie_f1 = sum(m.get('f1_score', 0.0) for m in openie_metrics) / len(openie_metrics)
            print(f"🔓 OPENIE Average F1 Score: {avg_openie_f1:.3f} ({len(openie_metrics)} datasets)")

        # Best performing model
        best_model = max([m for m in benchmarker.metrics.values() if m], key=lambda x: x.get('f1_score', 0.0))
        print(f"🏆 Best Overall Model: {best_model['model']} (F1: {best_model.get('f1_score', 0.0):.3f})")

    else:
        print("❌ No data available for display")


if __name__ == "__main__":
    main()

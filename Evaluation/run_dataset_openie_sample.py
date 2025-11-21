#!/usr/bin/env python3
"""General solution to run OpenIE (memory and no-memory) on any dataset with sampling."""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to ensure trallie can be imported
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tqdm import tqdm

from trallie.data_extraction.data_extractor import DataExtractor
from trallie.data_handlers import DataHandler
from trallie.evaluation.evaluation import evaluate_openie
from trallie.evaluation.evaluation_params import (
    DATASET_DESCRIPTIONS,
    extract_docs_from_tar,
    get_evaluation_params,
)
from trallie.providers import get_provider
from trallie.providers.openai import OpenAIProvider
from trallie.providers.groq import GroqProvider
from trallie.schema_generation.schema_generator import SchemaGenerator

RESULTS_ROOT = Path("results")
BASE_PATH = Path("evaporate/data")

# Chunking configuration
CHUNK_SIZE_THRESHOLD = 100000  # Characters - threshold for "large document"
CHUNK_SIZE = 100000            # Characters per chunk
OVERLAP_SIZE = 10000           # Characters overlap between chunks

# Model pricing (per 1M tokens, as of 2024)
MODEL_PRICING = {
    # OpenAI models
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    # Groq models (free tier, but we'll use approximate pricing for tracking)
    "llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},  # Free tier
    "llama-3.1-8b-instant": {"input": 0.00, "output": 0.00},  # Free tier
    "llama-3.1-70b-versatile": {"input": 0.00, "output": 0.00},  # Free tier
    "deepseek-r1-distill-llama-70b": {"input": 0.00, "output": 0.00},  # Free tier
    "mixtral-8x7b-32768": {"input": 0.00, "output": 0.00},  # Free tier
}

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
}

# Popular models list
POPULAR_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4"],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "deepseek-r1-distill-llama-70b",
    ],
}


def ensure_api_key(provider: str) -> None:
    """Ensure API key is set for the specified provider."""
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    elif provider == "groq":
        if "GROQ_API_KEY" not in os.environ:
            raise SystemExit("GROQ_API_KEY environment variable is not set.")
    else:
        raise SystemExit(f"Unsupported provider: {provider}. Supported: openai, groq")


def get_dataset_info(dataset_name: str) -> Dict:
    """Get dataset information including description and document paths."""
    datasets = get_evaluation_params(BASE_PATH)
    
    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise SystemExit(
            f"Dataset '{dataset_name}' not found. Available datasets: {available}"
        )
    
    dataset_info = datasets[dataset_name]
    dataset_dir = Path(dataset_info["dataset_dir"])
    
    # Extract docs.tar.gz if needed
    print(f"Checking/extracting documents for {dataset_name}...")
    if not extract_docs_from_tar(dataset_dir):
        raise SystemExit(
            f"Could not extract documents for {dataset_name}. "
            f"Make sure docs.tar.gz exists in {dataset_dir}"
        )
    
    # Search recursively in dataset directory for extracted documents
    all_files = (
        list(dataset_dir.rglob("*.htm"))
        + list(dataset_dir.rglob("*.html"))
        + list(dataset_dir.rglob("*.txt"))
        + list(dataset_dir.rglob("*.json"))
    )
    
    # Filter out table.json (ground truth file)
    all_files = [f for f in all_files if f.name != "table.json"]
    
    if not all_files:
        raise SystemExit(
            f"No document files found in {dataset_dir} after extraction. "
            f"Make sure docs.tar.gz was extracted successfully."
        )
    
    print(f"Found {len(all_files)} documents in {dataset_dir}")
    
    return {
        "description": dataset_info["description"],
        "docs_dir": dataset_dir,
        "dataset_dir": dataset_dir,
        "ground_truth": dataset_info.get("ground_truth"),
    }


def sample_documents(docs_dir: Path, percentage: float, seed: int) -> List[Path]:
    """Sample a percentage of documents using a fixed seed for reproducibility."""
    all_files = sorted(
        [
            f
            for f in docs_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in {".txt", ".html", ".htm", ".json"}
        ]
    )
    
    if not all_files:
        raise SystemExit(f"No document files found in {docs_dir} or subdirectories")
    
    num_samples = max(1, int(len(all_files) * percentage))
    rng = random.Random(seed)
    sampled = rng.sample(all_files, num_samples)
    
    return sorted(sampled)


def get_document_size(record_path: Path) -> int:
    """Get document size in characters."""
    try:
        data_handler = DataHandler(str(record_path), from_text=False)
        text = data_handler.get_text()
        if text and not text.startswith("Error:"):
            return len(text)
        return 0
    except Exception as e:
        print(f"Warning: Could not get size for {record_path}: {e}")
        return 0


def get_document_category(doc_size: int) -> str:
    """Categorize document by size."""
    if doc_size < CHUNK_SIZE_THRESHOLD:
        return "short"
    else:
        return "long"


def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a specific model."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Default to gpt-4o-mini pricing if model not found
    print(f"Warning: Pricing not found for {model}, using gpt-4o-mini pricing")
    return MODEL_PRICING["gpt-4o-mini"]


def calculate_cost(
    prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini"
) -> Dict[str, float]:
    """Calculate cost based on token usage and model pricing."""
    pricing = get_model_pricing(model)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }


def get_usage_tracker(provider: str):
    """Get usage tracker for the specified provider."""
    if provider == "openai":
        return OpenAIProvider
    elif provider == "groq":
        return GroqProvider
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def summarize_usage(usage_log: List[Dict[str, Optional[int]]]) -> Dict[str, int]:
    """Summarize token usage from usage log."""
    summary = {
        "requests": len(usage_log),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for entry in usage_log:
        summary["prompt_tokens"] += entry.get("prompt_tokens") or 0
        summary["completion_tokens"] += entry.get("completion_tokens") or 0
        summary["total_tokens"] += entry.get("total_tokens") or 0
    return summary


def track_per_call_metrics(
    usage_log_before: List[Dict],
    usage_log_after: List[Dict],
    time_start: float,
    time_end: float,
    call_type: str,
    document_name: str = None,
    call_id: int = None,
) -> Dict:
    """Extract per-call metrics from usage log."""
    new_calls = usage_log_after[len(usage_log_before):]
    
    per_call_metrics = []
    for idx, call in enumerate(new_calls):
        call_time = time_end - time_start if len(new_calls) == 1 else None
        prompt_tokens = call.get("prompt_tokens") or 0
        completion_tokens = call.get("completion_tokens") or 0
        total_tokens = call.get("total_tokens") or 0
        
        cost = calculate_cost(prompt_tokens, completion_tokens)
        
        per_call_metrics.append({
            "call_id": call_id or (len(usage_log_before) + idx),
            "call_type": call_type,
            "document_name": document_name,
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "time_seconds": round(call_time, 4) if call_time else None,
            "cost_usd": cost,
        })
    
    return per_call_metrics


def calculate_chunking_stats(
    documents: List[Path], per_call_metrics: List[Dict]
) -> Dict:
    """Calculate chunking statistics for documents."""
    doc_sizes = {doc.name: get_document_size(doc) for doc in documents}
    
    long_docs = [name for name, size in doc_sizes.items() if size >= CHUNK_SIZE_THRESHOLD]
    short_docs = [name for name, size in doc_sizes.items() if size < CHUNK_SIZE_THRESHOLD]
    
    # Group per-call metrics by document
    doc_metrics = defaultdict(list)
    for call in per_call_metrics:
        if call.get("document_name"):
            doc_metrics[call["document_name"]].append(call)
    
    # Count chunked documents (documents with multiple extraction calls)
    chunked_docs = []
    chunks_per_doc = []
    for doc_name, calls in doc_metrics.items():
        extraction_calls = [c for c in calls if c["call_type"] == "extraction"]
        if len(extraction_calls) > 1:
            chunked_docs.append(doc_name)
            chunks_per_doc.append(len(extraction_calls))
    
    # Calculate averages for long vs short documents
    def calc_avg_metrics(doc_names: List[str], metric_type: str) -> Dict:
        relevant_calls = [
            call
            for call in per_call_metrics
            if call.get("document_name") in doc_names and call["call_type"] == metric_type
        ]
        if not relevant_calls:
            return {
                "count": 0,
                "avg_tokens": 0,
                "avg_time": 0,
                "avg_cost": 0,
            }
        
        total_tokens = sum(c["tokens"]["total_tokens"] for c in relevant_calls)
        total_time = sum(c["time_seconds"] or 0 for c in relevant_calls)
        total_cost = sum(c["cost_usd"]["total_cost_usd"] for c in relevant_calls)
        count = len(relevant_calls)
        
        return {
            "count": count,
            "avg_tokens": round(total_tokens / count, 2) if count > 0 else 0,
            "avg_time": round(total_time / count, 4) if count > 0 else 0,
            "avg_cost": round(total_cost / count, 6) if count > 0 else 0,
        }
    
    return {
        "chunk_size_threshold": CHUNK_SIZE_THRESHOLD,
        "chunk_size": CHUNK_SIZE,
        "overlap_size": OVERLAP_SIZE,
        "long_documents_count": len(long_docs),
        "short_documents_count": len(short_docs),
        "chunked_documents_count": len(chunked_docs),
        "avg_chunks_per_document": round(sum(chunks_per_doc) / len(chunks_per_doc), 2) if chunks_per_doc else 0,
        "long_docs_schema": calc_avg_metrics(long_docs, "schema"),
        "short_docs_schema": calc_avg_metrics(short_docs, "schema"),
        "long_docs_extraction": calc_avg_metrics(long_docs, "extraction"),
        "short_docs_extraction": calc_avg_metrics(short_docs, "extraction"),
        "document_sizes": {name: size for name, size in doc_sizes.items()},
    }


def run_evaluation(
    dataset_name: str, predictions_path: Path, ground_truth_path: str, output_dir: Path
) -> Dict:
    """Run F1 evaluation and save results."""
    if not ground_truth_path or not Path(ground_truth_path).exists():
        print(f"Warning: Ground truth not found at {ground_truth_path}, skipping evaluation")
        return {}
    
    print(f"\nRunning evaluation...")
    
    try:
        # Exact match evaluation
        eval_exact = evaluate_openie(ground_truth_path, str(predictions_path), use_exact_match=True)
        with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(eval_exact, f, indent=2)
        
        # Partial match evaluation
        eval_partial = evaluate_openie(ground_truth_path, str(predictions_path), use_exact_match=False)
        with open(output_dir / "evaluation_partial.json", "w", encoding="utf-8") as f:
            json.dump(eval_partial, f, indent=2)
        
        print(f"  Exact Match - F1: {eval_exact['f1_score']:.4f}, Precision: {eval_exact['precision']:.4f}, Recall: {eval_exact['recall']:.4f}")
        print(f"  Partial Match - F1: {eval_partial['f1_score']:.4f}, Precision: {eval_partial['precision']:.4f}, Recall: {eval_partial['recall']:.4f}")
        
        return {
            "exact_match": eval_exact,
            "partial_match": eval_partial,
        }
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_openie_with_metrics(
    memory: bool,
    records: List[Path],
    dataset_name: str,
    dataset_description: str,
    model: str,
    provider: str,
    seed: int,
    ground_truth_path: str = None,
    reasoning_mode: bool = False,
) -> Dict:
    """Run OpenIE pipeline and collect metrics."""
    suffix = "memory" if memory else "nomemory"
    all_per_call_metrics = []
    call_counter = 0
    
    # Get usage tracker for the provider
    UsageTracker = get_usage_tracker(provider)
    
    # Reset usage tracking
    if hasattr(UsageTracker, "reset_usage_log"):
        UsageTracker.reset_usage_log()
    
    # Schema generation with chunking
    print(f"\n[{suffix.upper()}] Initializing schema generator (memory={memory}, reasoning={reasoning_mode})")
    schema_generator = SchemaGenerator(
        provider=provider,
        model_name=model,
        reasoning_mode=reasoning_mode,
        memory=memory,
    )
    
    schema_start = time.perf_counter()
    print(f"[{suffix.upper()}] Discovering schema from {len(records)} records (with chunking)...")
    
    # Use chunking for schema discovery (processes all records)
    schema_usage_before = UsageTracker.get_usage_log() if hasattr(UsageTracker, "get_usage_log") else []
    schema = schema_generator.discover_schema_with_chunking(
        dataset_description,
        [str(r) for r in records],
        num_records=len(records),
        from_text=False,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE,
        auto_detect_large_docs=True,
        top_k=10,
    )
    schema_usage_after = UsageTracker.get_usage_log() if hasattr(UsageTracker, "get_usage_log") else []
    
    # Track schema generation calls (aggregate all schema calls)
    schema_calls = schema_usage_after[len(schema_usage_before):]
    for idx, call in enumerate(schema_calls):
        all_per_call_metrics.append({
            "call_id": call_counter + idx,
            "call_type": "schema",
            "document_name": None,  # Schema discovery processes multiple docs
            "tokens": {
                "prompt_tokens": call.get("prompt_tokens") or 0,
                "completion_tokens": call.get("completion_tokens") or 0,
                "total_tokens": call.get("total_tokens") or 0,
            },
            "time_seconds": None,  # Can't track per-call time for batch processing
            "cost_usd": calculate_cost(
                call.get("prompt_tokens") or 0,
                call.get("completion_tokens") or 0,
                model
            ),
        })
    call_counter += len(schema_calls)
    schema_time = time.perf_counter() - schema_start
    print(f"[{suffix.upper()}] Schema discovered with {len(schema)} attributes")
    schema_usage = UsageTracker.get_usage_log() if hasattr(UsageTracker, "get_usage_log") else []
    
    # Data extraction with chunking
    print(f"[{suffix.upper()}] Starting extraction across {len(records)} documents (with chunking)...")
    data_extractor = DataExtractor(
        provider=provider,
        model_name=model,
        reasoning_mode=reasoning_mode,
    )
    
    predictions: Dict[str, Optional[Dict[str, object]]] = {}
    extraction_start = time.perf_counter()
    extraction_usage_before = schema_usage
    
    for record in tqdm(records, desc=f"[{suffix.upper()}] Extracting", unit="doc"):
        call_start = time.perf_counter()
        
        # Use chunking for extraction
        result = data_extractor.extract_data_with_chunking(
            schema,
            str(record),
            chunk_size=CHUNK_SIZE,
            overlap_size=OVERLAP_SIZE,
            max_retries=3,
            from_text=False,
            combine_results=True,
            auto_detect_large_docs=True,
        )
        
        call_end = time.perf_counter()
        extraction_usage_after = UsageTracker.get_usage_log() if hasattr(UsageTracker, "get_usage_log") else []
        
        # Track per-call metrics for extraction
        call_metrics = track_per_call_metrics(
            extraction_usage_before,
            extraction_usage_after,
            call_start,
            call_end,
            "extraction",
            document_name=record.name,
            call_id=call_counter,
        )
        all_per_call_metrics.extend(call_metrics)
        call_counter += len(call_metrics)
        extraction_usage_before = extraction_usage_after
        
        predictions[record.name] = result
    
    extraction_time = time.perf_counter() - extraction_start
    total_time = time.perf_counter() - schema_start
    
    total_usage_log = UsageTracker.get_usage_log() if hasattr(UsageTracker, "get_usage_log") else []
    extraction_usage_log = total_usage_log[len(schema_usage):]
    
    schema_summary = summarize_usage(schema_usage)
    extraction_summary = summarize_usage(extraction_usage_log)
    total_summary = summarize_usage(total_usage_log)
    
    # Calculate costs
    schema_cost = calculate_cost(
        schema_summary["prompt_tokens"], schema_summary["completion_tokens"], model
    )
    extraction_cost = calculate_cost(
        extraction_summary["prompt_tokens"],
        extraction_summary["completion_tokens"],
        model,
    )
    total_cost = calculate_cost(
        total_summary["prompt_tokens"], total_summary["completion_tokens"], model
    )
    
    # Calculate chunking stats
    chunking_stats = calculate_chunking_stats(records, all_per_call_metrics)
    
    metrics = {
        "dataset": dataset_name,
        "model": model,
        "provider": provider,
        "memory": memory,
        "num_records": len(records),
        "sample_percentage": len(records) / len(records) if records else 0,
        "seed": seed,
        "schema_time_seconds": round(schema_time, 2),
        "extraction_time_seconds": round(extraction_time, 2),
        "total_time_seconds": round(total_time, 2),
        "schema_usage": schema_summary,
        "extraction_usage": extraction_summary,
        "total_usage": total_summary,
        "schema_cost": schema_cost,
        "extraction_cost": extraction_cost,
        "cost": total_cost,
        "per_call_metrics": all_per_call_metrics,
        "chunking_stats": chunking_stats,
    }
    
    return {
        "schema": schema,
        "predictions": predictions,
        "metrics": metrics,
    }


def save_results(
    output_dir: Path, schema: List[str], predictions: Dict, metrics: Dict
) -> None:
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run OpenIE on a dataset with sampling"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset (e.g., 'swde_movie_allmovie')",
    )
    parser.add_argument(
        "--sample_percentage",
        type=float,
        default=0.1,
        help="Percentage of documents to sample (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gpt-4o-mini for OpenAI, llama-3.3-70b-versatile for Groq)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "groq"],
        default="openai",
        help="Provider name (default: openai)",
    )
    parser.add_argument(
        "--reasoning_mode",
        action="store_true",
        help="Enable reasoning mode (only for supported models like deepseek-r1-distill-llama-70b)",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List popular models for each provider and exit",
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\nPopular Models by Provider:")
        print("=" * 60)
        for prov, models in POPULAR_MODELS.items():
            print(f"\n{prov.upper()}:")
            for model in models:
                pricing = get_model_pricing(model)
                cost_str = f"${pricing['input']:.2f}/${pricing['output']:.2f} per 1M tokens"
                if pricing['input'] == 0 and pricing['output'] == 0:
                    cost_str = "Free tier"
                print(f"  - {model:40} ({cost_str})")
        print("\n" + "=" * 60)
        return
    
    ensure_api_key(args.provider)
    
    # Set default model based on provider if not specified
    if not args.model:
        args.model = DEFAULT_MODELS.get(args.provider, DEFAULT_MODELS["openai"])
        print(f"Using default model for {args.provider}: {args.model}")
    
    # Validate model for provider (warn but don't fail)
    if args.provider == "groq" and args.model not in POPULAR_MODELS.get("groq", []):
        print(f"Warning: {args.model} may not be a standard Groq model. Continuing anyway...")
    elif args.provider == "openai" and args.model not in POPULAR_MODELS.get("openai", []):
        print(f"Warning: {args.model} may not be a standard OpenAI model. Continuing anyway...")
    
    # Get dataset info
    print(f"Loading dataset: {args.dataset_name}")
    dataset_info = get_dataset_info(args.dataset_name)
    
    # Get all documents (recursively)
    all_files = sorted(
        [
            f
            for f in dataset_info["docs_dir"].rglob("*")
            if f.is_file() and f.suffix.lower() in {".txt", ".html", ".htm", ".json"}
        ]
    )
    print(f"Found {len(all_files)} total documents")
    
    # Sample documents
    sampled_records = sample_documents(
        dataset_info["docs_dir"], args.sample_percentage, args.seed
    )
    print(
        f"Sampled {len(sampled_records)} documents ({args.sample_percentage*100:.1f}%) "
        f"using seed {args.seed}"
    )
    
    total_docs = len(all_files)
    ground_truth_path = dataset_info.get("ground_truth")
    
    # Run no-memory
    print(f"\n{'='*60}")
    print(f"Running OpenIE WITHOUT memory")
    print(f"{'='*60}")
    no_memory_result = run_openie_with_metrics(
        memory=False,
        records=sampled_records,
        dataset_name=args.dataset_name,
        dataset_description=dataset_info["description"],
        model=args.model,
        provider=args.provider,
        seed=args.seed,
        ground_truth_path=ground_truth_path,
        reasoning_mode=args.reasoning_mode,
    )
    no_memory_result["metrics"]["total_documents"] = total_docs
    no_memory_result["metrics"]["sampled_documents"] = len(sampled_records)
    no_memory_result["metrics"]["sample_percentage"] = args.sample_percentage
    
    output_dir_nomemory = (
        RESULTS_ROOT
        / args.dataset_name
        / f"{args.provider}_{args.model}_openie_nomemory".replace("/", "_").replace(":", "_")
    )
    save_results(
        output_dir_nomemory,
        no_memory_result["schema"],
        no_memory_result["predictions"],
        no_memory_result["metrics"],
    )
    
    # Run evaluation for no-memory
    eval_results_nomemory = run_evaluation(
        args.dataset_name,
        output_dir_nomemory / "predictions.json",
        ground_truth_path,
        output_dir_nomemory,
    )
    no_memory_result["metrics"]["evaluation"] = eval_results_nomemory
    
    # Update summary with evaluation
    with open(output_dir_nomemory / "summary.json", "w", encoding="utf-8") as f:
        json.dump(no_memory_result["metrics"], f, indent=2)
    
    print(f"\nNo-memory summary:")
    print(f"  Time: {no_memory_result['metrics']['total_time_seconds']:.2f}s")
    print(f"  Cost: ${no_memory_result['metrics']['cost']['total_cost_usd']:.4f}")
    print(f"  Tokens: {no_memory_result['metrics']['total_usage']['total_tokens']:,}")
    if eval_results_nomemory and "exact_match" in eval_results_nomemory:
        print(f"  F1 Score: {eval_results_nomemory['exact_match']['f1_score']:.4f}")
    
    # Run with memory
    print(f"\n{'='*60}")
    print(f"Running OpenIE WITH memory")
    print(f"{'='*60}")
    memory_result = run_openie_with_metrics(
        memory=True,
        records=sampled_records,
        dataset_name=args.dataset_name,
        dataset_description=dataset_info["description"],
        model=args.model,
        provider=args.provider,
        seed=args.seed,
        ground_truth_path=ground_truth_path,
        reasoning_mode=args.reasoning_mode,
    )
    memory_result["metrics"]["total_documents"] = total_docs
    memory_result["metrics"]["sampled_documents"] = len(sampled_records)
    memory_result["metrics"]["sample_percentage"] = args.sample_percentage
    
    output_dir_memory = (
        RESULTS_ROOT
        / args.dataset_name
        / f"{args.provider}_{args.model}_openie_memory".replace("/", "_").replace(":", "_")
    )
    save_results(
        output_dir_memory,
        memory_result["schema"],
        memory_result["predictions"],
        memory_result["metrics"],
    )
    
    # Run evaluation for memory
    eval_results_memory = run_evaluation(
        args.dataset_name,
        output_dir_memory / "predictions.json",
        ground_truth_path,
        output_dir_memory,
    )
    memory_result["metrics"]["evaluation"] = eval_results_memory
    
    # Update summary with evaluation
    with open(output_dir_memory / "summary.json", "w", encoding="utf-8") as f:
        json.dump(memory_result["metrics"], f, indent=2)
    
    print(f"\nMemory summary:")
    print(f"  Time: {memory_result['metrics']['total_time_seconds']:.2f}s")
    print(f"  Cost: ${memory_result['metrics']['cost']['total_cost_usd']:.4f}")
    print(f"  Tokens: {memory_result['metrics']['total_usage']['total_tokens']:,}")
    if eval_results_memory and "exact_match" in eval_results_memory:
        print(f"  F1 Score: {eval_results_memory['exact_match']['f1_score']:.4f}")
    
    print(f"\n{'='*60}")
    print("All results saved successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

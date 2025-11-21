#!/usr/bin/env python3
"""
General batched evaluation script for Trallie.
Similar to run_fda_openie_gpt4omini.py but with batching and configurable parameters.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor
from trallie.providers import get_provider


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batched evaluation on any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Groq (fast and reliable)
  python simple_batched_eval/run_evaluation.py --dataset fda_510ks \\
    --provider groq --model llama-3.3-70b-versatile --batch-size 10

  # Run with OpenAI
  python simple_batched_eval/run_evaluation.py --dataset fda_510ks \\
    --provider openai --model gpt-4o-mini --batch-size 20

  # Run with Ollama (local)
  python simple_batched_eval/run_evaluation.py --dataset fda_510ks \\
    --provider ollama --model llama3.1:8b --batch-size 5
        """
    )
    
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g., fda_510ks)")
    parser.add_argument("--provider", default="groq",
                        choices=["groq", "openai", "ollama"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Model name (default: llama-3.3-70b-versatile)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of parallel extraction workers (default: 10)")
    parser.add_argument("--num-docs", type=int, default=None,
                        help="Limit number of documents (default: all)")
    parser.add_argument("--memory", action="store_true",
                        help="Enable schema memory mode")
    parser.add_argument("--output-dir", default="simple_results",
                        help="Output directory (default: simple_results)")
    parser.add_argument("--schema", default=None,
                        help="JSON file with predefined schema for ClosedIE (skips schema discovery)")
    parser.add_argument("--schema-json", default=None,
                        help="JSON string with predefined schema for ClosedIE")
    
    return parser.parse_args()


def load_dataset(dataset_name: str, num_docs: Optional[int] = None) -> tuple:
    """Load documents from a dataset."""
    base_path = Path("evaporate/data")
    dataset_dir = base_path / dataset_name
    
    # Find documents
    docs_dir = dataset_dir / "data/evaporate"
    
    # Try to find documents in various subdirectories
    doc_paths = []
    for pattern in ["**/*.txt", "**/*.html", "**/*.json"]:
        found = list(docs_dir.glob(pattern))
        if found:
            doc_paths.extend(found)
    
    if not doc_paths:
        raise SystemExit(f"No documents found in {docs_dir}. Make sure docs.tar.gz has been extracted.")
    
    # Sort and limit if requested
    doc_paths = sorted(doc_paths)
    if num_docs:
        doc_paths = doc_paths[:num_docs]
    
    # Get dataset description
    table_path = dataset_dir / "data/table.json"
    description = f"{dataset_name} dataset"
    if table_path.exists():
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
                if isinstance(table_data, dict) and 'description' in table_data:
                    description = table_data['description']
        except:
            pass
    
    return doc_paths, description


def extract_single_doc(extractor: DataExtractor, schema: Dict, doc_path: Path, 
                       idx: int, total: int) -> tuple:
    """Extract data from a single document with automatic chunking."""
    try:
        if idx % 10 == 1 or idx == total:
            print(f"  Processing {idx}/{total}: {doc_path.name}")
        
        # Use extract_data_with_chunking for automatic large document handling
        result = extractor.extract_data_with_chunking(
            schema=schema,
            record=str(doc_path),
            chunk_size=8000,  # Same as schema generation
            overlap_size=1000,
            max_retries=3,
            from_text=False,
            combine_results=True,
            auto_detect_large_docs=True
        )
        return (doc_path.name, result, None)
    except Exception as e:
        print(f"  Error on {doc_path.name}: {e}")
        return (doc_path.name, None, str(e))


def run_evaluation(args):
    """Run the evaluation pipeline."""
    # Load predefined schema if provided (ClosedIE mode)
    predefined_schema = None
    if args.schema:
        with open(args.schema, 'r') as f:
            predefined_schema = json.load(f)
        eval_mode = "ClosedIE (Predefined Schema)"
    elif args.schema_json:
        predefined_schema = json.loads(args.schema_json)
        eval_mode = "ClosedIE (Predefined Schema)"
    else:
        eval_mode = "OpenIE (Schema Discovery)"
    
    print("\n" + "="*80)
    print(f"BATCHED EVALUATION - {args.dataset}")
    print("="*80)
    print(f"Mode:         {eval_mode}")
    print(f"Provider:     {args.provider}")
    print(f"Model:        {args.model}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Memory:       {args.memory}")
    print("="*80 + "\n")
    
    # Verify API keys
    if args.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    elif args.provider == "groq" and "GROQ_API_KEY" not in os.environ:
        raise SystemExit("GROQ_API_KEY environment variable is not set.")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    doc_paths, description = load_dataset(args.dataset, args.num_docs)
    print(f"✓ Loaded {len(doc_paths)} documents")
    print(f"  Description: {description}\n")
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.provider}_{args.model.replace(':', '_')}_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}\n")
    
    # PHASE 1: Schema Generation (skip if predefined schema provided)
    if predefined_schema:
        print("="*80)
        print("PHASE 1: USING PREDEFINED SCHEMA (ClosedIE)")
        print("="*80)
        schema = predefined_schema
        schema_time = 0
        
        # Handle both list and dict schema formats
        schema_len = len(schema)
        if isinstance(schema, dict):
            schema_preview = ', '.join(list(schema.keys())[:5])
        else:  # list
            schema_preview = ', '.join(schema[:5])
        
        print(f"✓ Using predefined schema with {schema_len} attributes")
        print(f"  Attributes: {schema_preview}{'...' if schema_len > 5 else ''}\n")
        
        schema_start = time.perf_counter()  # Start timer for total time calculation
    else:
        print("="*80)
        print("PHASE 1: SCHEMA GENERATION (OpenIE)")
        print("="*80)
        
        print(f"Initializing schema generator (memory={args.memory})")
        schema_generator = SchemaGenerator(
            provider=args.provider,
            model_name=args.model,
            reasoning_mode=False,
            memory=args.memory,
        )
        
        schema_start = time.perf_counter()
        print(f"Discovering schema from {len(doc_paths)} documents...")
        print("  (Using auto-chunking for large documents)\n")
        
        # Use discover_schema_with_chunking to handle large documents automatically
        schema = schema_generator.discover_schema_with_chunking(
            description,
            [str(p) for p in doc_paths],
            num_records=len(doc_paths),
            from_text=False,
            chunk_size=8000,  # Smaller chunks to stay under API limits
            overlap_size=1000,
            auto_detect_large_docs=True,
            top_k=15
        )
        
        schema_end = time.perf_counter()
        schema_time = schema_end - schema_start
        
        if not schema:
            raise SystemExit("Schema generation failed - no attributes discovered")
        
        # Handle both list and dict schema formats
        schema_len = len(schema)
        if isinstance(schema, dict):
            schema_preview = ', '.join(list(schema.keys())[:5])
        else:  # list
            schema_preview = ', '.join(schema[:5])
        
        print(f"✓ Schema discovered with {schema_len} attributes")
        print(f"  Time: {schema_time:.2f}s")
        print(f"  Attributes: {schema_preview}{'...' if schema_len > 5 else ''}\n")
    
    # PHASE 2: Data Extraction (with batching)
    print("="*80)
    print("PHASE 2: DATA EXTRACTION")
    print("="*80)
    
    print(f"Initializing data extractor")
    data_extractor = DataExtractor(
        provider=args.provider,
        model_name=args.model,
        reasoning_mode=False,
    )
    
    extraction_start = time.perf_counter()
    print(f"Extracting data from {len(doc_paths)} documents (batch_size={args.batch_size})...")
    
    predictions: Dict[str, Optional[Dict]] = {}
    errors: Dict[str, str] = {}
    
    # Parallel extraction with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(extract_single_doc, data_extractor, schema, path, idx, len(doc_paths)): path
            for idx, path in enumerate(doc_paths, start=1)
        }
        
        for future in as_completed(futures):
            doc_name, result, error = future.result()
            if error:
                errors[doc_name] = error
            else:
                predictions[doc_name] = result
    
    extraction_end = time.perf_counter()
    extraction_time = extraction_end - extraction_start
    total_time = extraction_end - schema_start
    
    print(f"\n✓ Extraction complete")
    print(f"  Time: {extraction_time:.2f}s")
    print(f"  Speed: {len(doc_paths) / extraction_time:.2f} docs/sec")
    if errors:
        print(f"  ⚠ Errors: {len(errors)} documents failed")
    
    # Create summary
    metrics = {
        "evaluation_mode": "ClosedIE" if predefined_schema else "OpenIE",
        "dataset": args.dataset,
        "provider": args.provider,
        "model": args.model,
        "memory": args.memory,
        "batch_size": args.batch_size,
        "num_documents": len(doc_paths),
        "successful_extractions": len(predictions) - len(errors),
        "failed_extractions": len(errors),
        "timing": {
            "schema_time_seconds": round(schema_time, 2),
            "extraction_time_seconds": round(extraction_time, 2),
            "total_time_seconds": round(total_time, 2),
            "docs_per_second": round(len(doc_paths) / extraction_time, 2),
        }
    }
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    with open(output_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"✓ Saved schema.json")
    
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Saved predictions.json")
    
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved summary.json")
    
    if errors:
        with open(output_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"✓ Saved errors.json")
    
    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mode:              {eval_mode}")
    print(f"Dataset:           {args.dataset}")
    print(f"Documents:         {len(doc_paths)}")
    print(f"Schema Attributes: {schema_len}")
    if predefined_schema:
        print(f"Schema Time:       0.00s (predefined)")
    else:
        print(f"Schema Time:       {schema_time:.2f}s")
    print(f"Extraction Time:   {extraction_time:.2f}s")
    print(f"Total Time:        {total_time:.2f}s")
    print(f"Extraction Speed:  {len(doc_paths) / extraction_time:.2f} docs/sec")
    print(f"\nResults saved to: {output_dir}")
    print("="*80 + "\n")
    
    return output_dir


def main():
    """Main entry point."""
    try:
        args = parse_args()
        output_dir = run_evaluation(args)
        return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


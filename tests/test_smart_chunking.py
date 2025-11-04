import os
import json
import time
from trallie import DataExtractor
from trallie.data_extraction.smart_chunker import SmartChunker

# Set API key for Groq
os.environ["GROQ_API_KEY"] = ""

# Define schema for FDA 510k documents
schema = [
    "purpose for submission",
    "measurand", 
    "type of test",
    "classification",
    "product code",
    "panel",
    "intended use",
    "510k number",
    "applicant",
    "predicate device name"
]

def test_chunking_methods():
    print("="*80)
    print("SMART CHUNKING vs TRADITIONAL CHUNKING COMPARISON")
    print("="*80)
    print()
    
    # Load data from JSON (relative to project root)
    json_file = "evaporate/data/fda_510ks/table.json"
    # If running from tests directory, go up one level
    if not os.path.exists(json_file):
        json_file = os.path.join("..", json_file)
    if not os.path.exists(json_file):
        print(f"ERROR: JSON file not found: {json_file}")
        return
    
    # Load FDA documents
    with open(json_file, 'r', encoding='utf-8') as f:
        fda_data = json.load(f)
    
    # Get all documents and find a large one for testing
    print("Finding a large document for chunking test...")
    large_docs = []
    for key, value in fda_data.items():
        if isinstance(value, dict):
            content_len = sum(len(str(v)) for v in value.values())
            large_docs.append((key, value, content_len))
    
    # Sort by size and get top 5 largest
    large_docs.sort(key=lambda x: x[2], reverse=True)
    
    # Use 5th largest document (big enough for chunking but not too big)
    if len(large_docs) >= 5:
        test_key, doc_content, size = large_docs[4]  # 5th largest
        print(f"Selected large document: {size:,} characters")
    else:
        # Fallback to first document
        test_key = list(fda_data.keys())[0]
        doc_content = fda_data[test_key]
    
    # Convert dict to string if needed
    if isinstance(doc_content, dict):
        doc_content = '\n'.join([f"{k}: {v}" for k, v in doc_content.items()])
    
    print(f"Test File: {test_key}")
    print(f"Document Size: {len(doc_content):,} characters")
    print()
    
    # Test smart chunker strategy detection (no LLM calls)
    print("="*80)
    print("STRATEGY DETECTION TEST (No LLM calls)")
    print("="*80)
    smart_chunker = SmartChunker()
    detected_strategy = smart_chunker.detect_document_structure(doc_content)
    print(f"Detected Chunking Strategy: {detected_strategy}")
    print(f"This detection uses only regex patterns - no LLM calls!")
    print()
    
    # Initialize extractors
    extractor_traditional = DataExtractor(
        provider="groq",
        model_name="llama-3.3-70b-versatile"
    )
    
    extractor_smart = DataExtractor(
        provider="groq",
        model_name="llama-3.3-70b-versatile"
    )
    
    # Save to temp file for processing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(doc_content)
        temp_file = f.name
    
    # Test 1: Traditional Chunking (Character-based)
    print("="*80)
    print("TEST 1: TRADITIONAL CHUNKING (Character-based)")
    print("="*80)
    print("Strategy: Simple character-based splitting with sentence boundary detection")
    start_time = time.time()
    
    try:
        # Traditional: use standard chunking (character-based)
        # Force chunking by using smaller chunk size
        result_traditional = extractor_traditional.extract_data_with_chunking(
            schema=schema,
            record=temp_file,
            chunk_size=1500,  # Small chunk size to force chunking
            overlap_size=150,
            combine_results=True,
            auto_detect_large_docs=False  # Force chunking even for small docs
        )
        
        time_traditional = time.time() - start_time
        print(f"Time taken: {time_traditional:.2f}s")
        print(f"Extracted fields: {len([k for k, v in (result_traditional or {}).items() if v])}")
        print(f"Results preview: {json.dumps({k: str(v)[:50] + '...' if len(str(v)) > 50 else v for k, v in (result_traditional or {}).items()}, indent=2)}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result_traditional = None
        time_traditional = 0
    
    print()
    
    # Test 2: Smart Chunking (Generic Regex-based)
    print("="*80)
    print("TEST 2: SMART CHUNKING (Generic Regex-based)")
    print("="*80)
    print(f"Detected Strategy: {detected_strategy}")
    print("Uses generic regex patterns to detect document structure")
    print("Works with ANY dataset, not just FDA documents")
    start_time = time.time()
    
    try:
        result_smart = extractor_smart.extract_data_smart_chunking(
            schema=schema,
            record=temp_file,
            chunk_size=1500,  # Same chunk size for fair comparison
            overlap_size=150,
            combine_results=True,
            use_smart_chunking=True,  # Use new generic smart chunking
            auto_detect_large_docs=False  # Force chunking even for small docs
        )
        
        time_smart = time.time() - start_time
        print(f"Time taken: {time_smart:.2f}s")
        print(f"Extracted fields: {len([k for k, v in (result_smart or {}).items() if v])}")
        print(f"Results preview: {json.dumps({k: str(v)[:50] + '...' if len(str(v)) > 50 else v for k, v in (result_smart or {}).items()}, indent=2)}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result_smart = None
        time_smart = 0
    
    print()
    
    # Comparison
    print()
    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<30} {'Traditional':<20} {'Smart (Generic)':<20}")
    print("-" * 80)
    print(f"{'Time taken:':<30} {time_traditional:.2f}s{'':<10} {time_smart:.2f}s")
    
    if result_traditional and result_smart:
        traditional_fields = len([k for k, v in result_traditional.items() if v and str(v).strip()])
        smart_fields = len([k for k, v in result_smart.items() if v and str(v).strip()])
        
        print(f"{'Fields extracted:':<30} {traditional_fields:<20} {smart_fields:<20}")
        
        # Count non-empty values
        traditional_non_empty = sum(1 for k, v in result_traditional.items() if v and str(v).strip() and str(v).strip() not in ['', 'None', 'Not explicitly stated', 'not specified', 'Unknown'])
        smart_non_empty = sum(1 for k, v in result_smart.items() if v and str(v).strip() and str(v).strip() not in ['', 'None', 'Not explicitly stated', 'not specified', 'Unknown'])
        
        print(f"{'Non-empty values:':<30} {traditional_non_empty:<20} {smart_non_empty:<20}")
        
        print()
        if smart_non_empty > traditional_non_empty:
            print(f"[SUCCESS] Smart chunking extracted {smart_non_empty - traditional_non_empty} more non-empty fields!")
        elif traditional_non_empty > smart_non_empty:
            print(f"[WARNING] Traditional chunking extracted {traditional_non_empty - smart_non_empty} more non-empty fields")
        else:
            print("[INFO] Both methods extracted similar number of non-empty fields")
        
        # Show which fields were extracted by each method
        traditional_keys = set(k for k, v in result_traditional.items() if v and str(v).strip() and str(v).strip() not in ['', 'None', 'Not explicitly stated', 'not specified', 'Unknown'])
        smart_keys = set(k for k, v in result_smart.items() if v and str(v).strip() and str(v).strip() not in ['', 'None', 'Not explicitly stated', 'not specified', 'Unknown'])
        
        only_traditional = traditional_keys - smart_keys
        only_smart = smart_keys - traditional_keys
        
        if only_traditional:
            print(f"\nFields only in Traditional: {list(only_traditional)}")
        if only_smart:
            print(f"Fields only in Smart: {list(only_smart)}")
        
    elif result_traditional:
        print("[WARNING] Smart chunking failed, but traditional succeeded")
    elif result_smart:
        print("[WARNING] Traditional chunking failed, but smart succeeded")
    else:
        print("[ERROR] Both methods failed")
    
    print()
    print("="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print("Results saved to:")
    print("- tests/traditional_results.json")
    print("- tests/smart_results.json")
    print("- tests/comparison_results.json")
    print()
    
    # Get the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save individual results
    with open(os.path.join(test_dir, "traditional_results.json"), "w") as f:
        json.dump(result_traditional, f, indent=2)
    
    with open(os.path.join(test_dir, "smart_results.json"), "w") as f:
        json.dump(result_smart, f, indent=2)
    
    # Save comparison
    comparison = {
        "test_file": test_key,
        "document_size": len(doc_content),
        "detected_strategy": detected_strategy,
        "traditional": {
            "time_seconds": time_traditional,
            "fields_extracted": len([k for k, v in (result_traditional or {}).items() if v]) if result_traditional else 0,
            "results": result_traditional
        },
        "smart": {
            "time_seconds": time_smart,
            "fields_extracted": len([k for k, v in (result_smart or {}).items() if v]) if result_smart else 0,
            "results": result_smart
        }
    }
    
    with open(os.path.join(test_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Clean up temp file
    if os.path.exists(temp_file):
        os.unlink(temp_file)

if __name__ == "__main__":
    test_chunking_methods()

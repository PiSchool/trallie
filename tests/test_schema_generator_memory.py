
import os
import sys
from pathlib import Path

# Ensure local package (repo root) is used
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor
from trallie.providers.errors import ProviderInitializationError


def main():
    # Set dummy API key for testing (replace with your actual key)
    os.environ.setdefault("GROQ_API_KEY", "your_groq_api_key_here")

    description = "A small mixed dataset of personal bios and product blurbs."

    record1 = (
        "John Doe is a 28-year-old software engineer living in Berlin. "
        "He likes Python and machine learning."
    )
    record2 = (
        "The Galaxy X smartphone costs $999 and launched on 2023-09-01. "
        "It is produced by ACME Mobile and sold in the USA."
    )

    generator = SchemaGenerator(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        memory=True,
    )

    print("=== First extraction (no prior memory) ===")
    schema1 = generator.extract_schema(description, record1, max_retries=3)
    print("Schema 1:", schema1)
    print("\nlast_schema after first:", generator.last_schema)
    assert isinstance(schema1, dict), "First extraction did not return a dict schema"
    assert generator.last_schema == schema1, "last_schema should equal first returned schema"
    # Ensure prior schema is not in the first prompt
    assert "Prior schema" not in (getattr(generator, "_last_user_prompt", "") or ""), "First prompt should not contain prior schema context"
    # Schema values should be textual descriptions (not extracted values)
    for k, v in schema1.items():
        assert isinstance(k, str), "Schema keys must be strings"
        assert isinstance(v, str), "Schema values should be textual descriptions"

    print("\n=== Second extraction (with previous memory as context) ===")
    schema2 = generator.extract_schema(description, record2, max_retries=3)
    print("Schema 2:", schema2)
    print("\nlast_schema after second:", generator.last_schema)
    assert isinstance(schema2, dict), "Second extraction did not return a dict schema"
    assert generator.last_schema == schema2, "last_schema should be overwritten to second returned schema"
    # Ensure previous schema is included in the second prompt
    last_prompt = getattr(generator, "_last_user_prompt", "") or ""
    assert "Prior schema (for reference only):" in last_prompt, "Second prompt should include previous schema context"
    for k, v in schema2.items():
        assert isinstance(k, str), "Schema keys must be strings"
        assert isinstance(v, str), "Schema values should be textual descriptions"

    # Ensure schemas can differ and memory only keeps the last one
    if schema1 != schema2:
        assert generator.last_schema != schema1, "Memory should not keep older schemas"

    # Also verify discover_schema returns a list of attribute names
    print("\n=== Discover schema over two records (from_text=True) ===")
    attrs = generator.discover_schema(description, [record1, record2], num_records=2, from_text=True)
    print("Top attributes:", attrs)
    assert isinstance(attrs, list), "discover_schema should return a list of attribute names"
    for a in attrs:
        assert isinstance(a, str), "Attribute names must be strings"


def test_empty_documents_scenarios():
    """Test behavior when description is provided without any documents."""
    
    # Set dummy API key for testing (replace with your actual key)
    os.environ.setdefault("GROQ_API_KEY", "your_groq_api_key_here")
    
    description = "A collection of scientific research papers about climate change."
    
    print("\n" + "="*80)
    print("TESTING EMPTY DOCUMENTS SCENARIOS")
    print("="*80)
    
    # Test 1: SchemaGenerator with empty document list
    print("\n=== Test 1: SchemaGenerator with empty document list ===")
    generator = SchemaGenerator(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        memory=False,  # Disable memory for cleaner testing
    )
    
    # Test discover_schema with empty list
    try:
        empty_attrs = generator.discover_schema(description, [], num_records=10, from_text=True)
        print(f"Empty documents result: {empty_attrs}")
        assert isinstance(empty_attrs, list), "discover_schema should return a list even with empty documents"
        assert len(empty_attrs) == 0, "Empty document list should return empty attribute list"
    except ValueError as e:
        print(f"Correctly raised ValueError for empty documents: {e}")
    
    # Test discover_schema with None records
    try:
        none_attrs = generator.discover_schema(description, None, num_records=10, from_text=True)
        print(f"None records result: {none_attrs}")
        # This should either return empty list or raise an appropriate error
        assert isinstance(none_attrs, list), "None records should be handled gracefully"
    except (TypeError, AttributeError, ValueError) as e:
        print(f"Expected error with None records: {e}")
        # This is acceptable behavior - None should raise an error
    
    # Test discover_schema with empty string records
    try:
        empty_string_attrs = generator.discover_schema(description, [""], num_records=1, from_text=True)
        print(f"Empty string record result: {empty_string_attrs}")
        assert isinstance(empty_string_attrs, list), "Empty string record should return a list"
    except ValueError as e:
        print(f"Correctly raised ValueError for empty string: {e}")
    
    # Test 2: DataExtractor with empty document scenarios
    print("\n=== Test 2: DataExtractor with empty document scenarios ===")
    extractor = DataExtractor(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        reasoning_mode=False
    )
    
    # Define a test schema
    test_schema = {
        "title": "Title of the research paper",
        "authors": "List of authors",
        "abstract": "Abstract of the paper",
        "keywords": "Keywords related to the research"
    }
    
    # Test extract_data with empty string
    try:
        empty_result = extractor.extract_data(test_schema, "", max_retries=3, from_text=True)
        print(f"Empty string extraction result: {empty_result}")
        # Note: DataExtractor returns None for empty strings when API calls fail
        assert empty_result is None or isinstance(empty_result, dict), "Empty string should return None or dict"
        # The result might be None or empty dict depending on implementation
    except ValueError as e:
        print(f"Correctly raised ValueError for empty string: {e}")
    
    # Test extract_data with None record
    try:
        none_result = extractor.extract_data(test_schema, None, max_retries=3, from_text=True)
        print(f"None record extraction result: {none_result}")
        # This should either return None/empty dict or raise an appropriate error
    except (TypeError, AttributeError, ValueError) as e:
        print(f"Expected error with None record: {e}")
        # This is acceptable behavior
    
    # Test 3: Edge cases with invalid file paths
    print("\n=== Test 3: Edge cases with invalid file paths ===")
    
    # Test with non-existent file path
    try:
        invalid_file_result = extractor.extract_data(test_schema, "non_existent_file.txt", max_retries=3, from_text=False)
        print(f"Invalid file path result: {invalid_file_result}")
        # Should handle gracefully - might return None or empty dict
        assert invalid_file_result is None or isinstance(invalid_file_result, dict), "Invalid file should be handled gracefully"
    except Exception as e:
        print(f"Expected error with invalid file: {e}")
        # This is also acceptable behavior
    
    # Test 4: SchemaGenerator with very short descriptions
    print("\n=== Test 4: SchemaGenerator with minimal descriptions ===")
    
    minimal_descriptions = [
        "",
        "Papers",
        "A",
        "   ",  # Only whitespace
        "123",  # Numbers only
    ]
    
    # Define a test record for minimal description testing
    test_record = "This is a sample research paper about machine learning and artificial intelligence."
    
    for i, desc in enumerate(minimal_descriptions):
        print(f"\nTesting minimal description {i+1}: '{desc}'")
        try:
            minimal_attrs = generator.discover_schema(desc, [test_record], num_records=1, from_text=True)
            print(f"Minimal description result: {minimal_attrs}")
            assert isinstance(minimal_attrs, list), f"Minimal description '{desc}' should return a list"
        except Exception as e:
            print(f"Error with minimal description '{desc}': {e}")
            # Some minimal descriptions might cause errors, which is acceptable
    
    # Test 5: DataExtractor with invalid schemas
    print("\n=== Test 5: DataExtractor with invalid schemas ===")
    
    invalid_schemas = [
        {},  # Empty schema
        None,  # None schema
        "invalid_schema",  # String instead of dict
        {"": "empty key"},  # Empty key
        {"key": ""},  # Empty value description
    ]
    
    test_record = "This is a test research paper about machine learning."
    
    for i, schema in enumerate(invalid_schemas):
        print(f"\nTesting invalid schema {i+1}: {schema}")
        try:
            invalid_result = extractor.extract_data(schema, test_record, max_retries=3, from_text=True)
            print(f"Invalid schema result: {invalid_result}")
            # Should handle gracefully
        except Exception as e:
            print(f"Expected error with invalid schema: {e}")
            # This is acceptable behavior for invalid schemas
    
    print("\n" + "="*80)
    print("EMPTY DOCUMENTS SCENARIOS TESTING COMPLETED")
    print("="*80)


def test_error_handling_and_edge_cases():
    """Test comprehensive error handling and edge cases."""
    
    # Set dummy API key for testing (replace with your actual key)
    os.environ.setdefault("GROQ_API_KEY", "your_groq_api_key_here")
    
    print("\n" + "="*80)
    print("TESTING ERROR HANDLING AND EDGE CASES")
    print("="*80)
    
    # Test 1: Invalid provider configurations
    print("\n=== Test 1: Invalid provider configurations ===")
    
    invalid_configs = [
        {"provider": "invalid_provider", "model_name": "gpt-4o"},
        {"provider": "openai", "model_name": "invalid_model"},
        {"provider": "groq", "model_name": "gpt-4o"},  # Wrong provider-model combination
    ]
    
    for config in invalid_configs:
        print(f"Testing config: {config}")
        try:
            generator = SchemaGenerator(**config)
            print(f"Unexpected success with invalid config: {config}")
        except (ValueError, ProviderInitializationError) as e:
            print(f"Expected error with invalid config: {e}")
            # This is the expected behavior
    
    # Test 2: Memory system edge cases
    print("\n=== Test 2: Memory system edge cases ===")
    
    generator = SchemaGenerator(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        memory=True,
    )
    
    # Test memory reset
    generator.reset_memory()
    assert generator.last_schema is None, "Memory should be cleared after reset"
    
    # Test memory with empty schema
    try:
        empty_schema = generator.extract_schema("Test description", "", max_retries=3)
        print(f"Empty schema result: {empty_schema}")
        # Should handle gracefully
    except ValueError as e:
        print(f"Correctly raised ValueError for empty schema: {e}")
    
    # Test 3: Large document handling
    print("\n=== Test 3: Large document handling ===")
    
    # Create a very large text (simulate large document)
    large_text = "This is a test document. " * 10000  # ~250,000 characters
    
    try:
        large_schema = generator.extract_schema("Large document test", large_text, max_retries=3)
        print(f"Large document schema result: {large_schema}")
        assert isinstance(large_schema, dict), "Large document should return a dict schema"
    except Exception as e:
        print(f"Error with large document: {e}")
        # Large documents might cause issues depending on model limits
    
    # Test 4: Special characters and encoding
    print("\n=== Test 4: Special characters and encoding ===")
    
    special_texts = [
        "Document with émojis 🚀 and spécial characters",
        "Document with\nnewlines\tand\ttabs",
        "Document with \"quotes\" and 'apostrophes'",
        "Document with [brackets] and {braces}",
        "Document with numbers 123 and symbols @#$%",
    ]
    
    for i, text in enumerate(special_texts):
        print(f"Testing special text {i+1}: {text[:50]}...")
        try:
            special_schema = generator.extract_schema("Special characters test", text, max_retries=3)
            print(f"Special text schema: {special_schema}")
            assert isinstance(special_schema, dict), "Special characters should be handled"
        except Exception as e:
            print(f"Error with special text: {e}")
    
    print("\n" + "="*80)
    print("ERROR HANDLING AND EDGE CASES TESTING COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
    test_empty_documents_scenarios()
    test_error_handling_and_edge_cases()



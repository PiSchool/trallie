
import os
import sys
from pathlib import Path

# Ensure local package (repo root) is used
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trallie.schema_generation.schema_generator import SchemaGenerator


def main():
    # Set the Groq API key
    #os.environ.setdefault("GROQ_API_KEY", "None")

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


if __name__ == "__main__":
    main()



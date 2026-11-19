#!/usr/bin/env python3
"""
Run structured extraction with NuMind NuExtract (cloud API).

Requires ``NUMIND_API_KEY`` (or pass ``--api-key``). Install: ``pip install numind``.

Example::

    python scripts/nuextract_gliner/run_nuextract.py \\
        --input sample.txt \\
        --schema-json path/to/schema.json \\
        --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import schema_utils  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Extract structured fields with NuExtract (NuMind API).")
    p.add_argument("--input", type=Path, required=True, help="Path to .txt/.md or .pdf")
    p.add_argument(
        "--schema-json",
        type=Path,
        required=True,
        help="JSON file: trallie-style {attributes: [...]} or a JSON list of field names",
    )
    p.add_argument("--output", type=Path, help="Write JSON result to this path (default: stdout)")
    p.add_argument(
        "--verbatim",
        action="store_true",
        help="Use verbatim-string for all fields (must appear literally in source)",
    )
    p.add_argument("--api-key", default=os.environ.get("NUMIND_API_KEY"), help="NuMind API key")
    args = p.parse_args()

    if not args.api_key:
        print("Missing API key: set NUMIND_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    try:
        from numind import NuMind
    except ImportError as e:
        print("Install numind: pip install numind", file=sys.stderr)
        raise SystemExit(1) from e

    text = schema_utils.load_text_from_path(args.input)
    schema = schema_utils.load_json(args.schema_json)
    attrs = schema_utils.attributes_from_schema(schema)
    template = schema_utils.nuextract_template_from_attributes(
        attrs, field_type="verbatim-string" if args.verbatim else "string"
    )

    client = NuMind(api_key=args.api_key)
    result = client.extract_structured_data(template=template, input_text=text)

    out_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json + "\n", encoding="utf-8")
    else:
        print(out_json)


if __name__ == "__main__":
    main()

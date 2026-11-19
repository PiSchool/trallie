#!/usr/bin/env python3
"""
Run zero-shot entity extraction with GLiNER (local model).

Install: ``pip install gliner`` (brings PyTorch). Example::

    python scripts/nuextract_gliner/run_gliner.py \\
        --input sample.txt \\
        --labels invoice_number,invoice_date,grand_total \\
        --output entities.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import schema_utils  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Extract entities with GLiNER (zero-shot NER).")
    p.add_argument("--input", type=Path, required=True, help="Path to .txt/.md or .pdf")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--labels",
        type=str,
        help="Comma-separated entity types to detect (e.g. person,date,amount)",
    )
    g.add_argument(
        "--schema-json",
        type=Path,
        help="JSON file with 'attributes' list; each attribute becomes a GLiNER label",
    )
    p.add_argument(
        "--model",
        default="urchade/gliner_medium-v2.1",
        help="HF model id for GLiNER.from_pretrained",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", type=Path, help="Write JSON to this path (default: stdout)")
    args = p.parse_args()

    try:
        from gliner import GLiNER
    except ImportError as e:
        print("Install gliner: pip install gliner", file=sys.stderr)
        raise SystemExit(1) from e

    text = schema_utils.load_text_from_path(args.input)
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    else:
        schema = schema_utils.load_json(args.schema_json)
        labels = schema_utils.attributes_from_schema(schema)

    if not labels:
        print("No labels provided.", file=sys.stderr)
        sys.exit(1)

    model = GLiNER.from_pretrained(args.model)
    entities = model.predict_entities(text, labels, threshold=args.threshold)

    payload = {
        "model": args.model,
        "threshold": args.threshold,
        "labels": labels,
        "entities": entities,
    }
    out_json = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json + "\n", encoding="utf-8")
    else:
        print(out_json)


if __name__ == "__main__":
    main()

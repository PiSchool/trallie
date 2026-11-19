#!/usr/bin/env python3
"""
ClosedIE baseline: run NuMind's open NuExtract-2.0 checkpoint locally on a
benchmark dataset and write predictions in Trallie's predicted_table format.

    python benchmarking/baselines/run_nuextract_local.py --dataset fda_510ks --limit 100

Output: trallie-eval-result/nuextract-2.0-2b_<dataset>_closedie_predicted_table.json
Resumable: already-processed records are skipped.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_params import get_dataset_schema, get_evaluation_params  # noqa: E402


def build_template(fields, field_type):
    return json.dumps({f: field_type for f in fields}, indent=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fda_510ks")
    p.add_argument("--model", default="numind/NuExtract-2.0-2B")
    p.add_argument("--field-type", default="string", choices=["string", "verbatim-string"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-input-tokens", type=int, default=8000)
    p.add_argument("--out-dir", default="trallie-eval-result")
    args = p.parse_args()

    datasets = get_evaluation_params(Path("data/evaluation/data"))
    info = datasets[args.dataset]
    fields = get_dataset_schema(info["ground_truth"])
    template = build_template(fields, args.field_type)
    # Same record selection as evaluation_main.py (first N of the glob), minus macOS "._" files
    records = sorted(r for r in glob.glob(str(info["files"])) if not os.path.basename(r).startswith("._"))
    records = records[: args.limit]

    tag = args.model.split("/")[-1].lower()
    if args.field_type == "verbatim-string":
        tag += "-verbatim"
    out_path = Path(args.out_dir) / f"{tag}_{args.dataset}_closedie_predicted_table.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    tok = processor.tokenizer

    for i, record in enumerate(records):
        name = os.path.basename(record)
        if name in results:
            continue
        text = Path(record).read_text(encoding="utf-8", errors="ignore")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_input_tokens:
            text = tok.decode(ids[: args.max_input_tokens])
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}], template=template, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=None, padding=True, return_tensors="pt").to(device)
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1536)
            decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results[name] = json.loads(decoded)
        except json.JSONDecodeError:
            results[name] = None
        except RuntimeError as e:  # MPS OOM on long inputs: count as a failed record
            print(f"{name}: {str(e)[:120]}", flush=True)
            results[name] = None
        print(f"[{i + 1}/{len(records)}] {name} in_tok={inputs['input_ids'].shape[1]} "
              f"{time.time() - t0:.1f}s ok={results[name] is not None}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=4, ensure_ascii=False))
        if device == "mps":
            torch.mps.empty_cache()

    print(f"Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()

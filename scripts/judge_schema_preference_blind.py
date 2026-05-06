"""
Blind pairwise LLM preference between two attribute-name lists (no labels that reveal origin).

Exactly TWO gpt-4o-mini calls per run:
  1) Schema A vs Schema B only (order randomized).
  2) Same A vs B plus a reference document excerpt (same ordering).

The model never sees "gold", "ground truth", "human", "Evaporate", or "predicted".
It must score **each** schema on the same rubric (parallel dimension_scores), assign a
**holistic final score** (0–1) to each schema, then pick preferred A/B/tie. The saved artifact
maps A/B back to run_schema vs labeled_reference_fields with both dimension and final scores.

Usage:
  $env:OPENAI_API_KEY="..."
  python scripts/judge_schema_preference_blind.py --run gpt4omini_openie_nomemory --dataset fda_510ks

Optional: --doc-id, --out, --seed 42, --char-limit 6000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_EVAL_ROOT = REPO_ROOT / "schema_eval_data"

EVAPORATE_DATASET_NAMES = (
    "fda_510ks",
    "enron",
    "wiki_nba_players",
    "swde_movie_allmovie",
    "swde_movie_amctv",
    "swde_movie_hollywood",
    "swde_movie_iheartmovies",
    "swde_movie_imdb",
    "swde_movie_metacritic",
    "swde_movie_rottentomatoes",
    "swde_movie_yahoo",
    "swde_university_collegeprowler",
    "swde_university_ecampustours",
    "swde_university_embark",
    "swde_university_matchcollege",
    "swde_university_usnews",
)


def _pred_root(dataset: str) -> Path:
    return SCHEMA_EVAL_ROOT / "predicted" / dataset


def _gold_schema_path(dataset: str) -> Path:
    return SCHEMA_EVAL_ROOT / "ground_truth" / dataset / "ground_truth.json"


def _evaporate_docs_tar(dataset: str) -> Path:
    return REPO_ROOT / "evaporate" / "data" / dataset / "docs.tar.gz"


def _evaporate_table(dataset: str) -> Path:
    return REPO_ROOT / "evaporate" / "data" / dataset / "table.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing_pred_schema(dataset: str, run: str) -> Path:
    p = _pred_root(dataset) / run / "schema.json"
    if not p.is_file():
        root = _pred_root(dataset)
        available = sorted({d.name for d in root.iterdir() if d.is_dir()}) if root.is_dir() else []
        raise FileNotFoundError(f"Predicted schema not found at {p}. Available runs: {available}")
    return p


def _load_gold_schema_names(dataset: str) -> list[str]:
    path = _gold_schema_path(dataset)
    gold = _load_json(path)
    names = gold.get("attribute_names")
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"Invalid attribute list at {path}")
    return names


def _load_pred_schema_names(path: Path) -> list[str]:
    pred = _load_json(path)
    if not isinstance(pred, list) or not all(isinstance(x, str) for x in pred):
        raise ValueError(f"Invalid predicted schema list at {path}")
    return pred


def _pick_doc_id_from_table(dataset: str) -> str:
    table = _load_json(_evaporate_table(dataset))
    if not isinstance(table, dict) or not table:
        raise ValueError(f"Unexpected table format at {_evaporate_table(dataset)}")
    k = next(iter(table.keys()))
    return Path(k).name


def _extract_doc_text_from_tar(dataset: str, doc_id: str, char_limit: int) -> str:
    tar_path = _evaporate_docs_tar(dataset)
    if not tar_path.is_file():
        raise FileNotFoundError(f"Missing {tar_path}")

    member_name: Optional[str] = None
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if Path(m.name).name == doc_id:
                member_name = m.name
                break
        if not member_name:
            raise FileNotFoundError(f"Could not find {doc_id} inside {tar_path}")
        f = tf.extractfile(member_name)
        if f is None:
            raise FileNotFoundError(f"Could not extract member {member_name}")
        raw = f.read()

    text = ""
    for enc in ("utf-8", "latin-1"):
        try:
            candidate = raw.decode(enc)
            if "\x00" not in candidate:
                text = candidate
                break
        except UnicodeDecodeError:
            continue

    if not text:
        try:
            from trallie.data_handlers import DataHandler

            tmp_dir = REPO_ROOT / ".tmp_schema_judge"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / doc_id
            tmp_path.write_bytes(raw)
            text = DataHandler(str(tmp_path)).get_text()
        except Exception:
            text = ""

    text = text.strip()
    if len(text) > char_limit:
        text = text[:char_limit] + "\n...[truncated]...\n"
    return text


def _reference_document_for_judge(doc_id: str, text: str) -> str:
    header = f"[reference_document_id: {doc_id}]"
    body = (text or "").strip()
    if not body:
        return f"{header}\n\n[no extractable text from this file]"
    return f"{header}\n\n{body}"


def _extract_first_json_object(content: str) -> dict[str, Any]:
    content = (content or "").strip()
    if not content:
        raise ValueError("Model returned empty content.")
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        brace = content.find("{", idx)
        if brace == -1:
            break
        try:
            data, end = decoder.raw_decode(content, brace)
            if isinstance(data, dict):
                return data
            idx = max(end, brace + 1)
        except json.JSONDecodeError:
            idx = brace + 1
    raise ValueError(f"Model did not return a JSON object. Got: {content[:2000]}")


# Same keys for Schema A and Schema B (blind-safe: no cross-list "recall vs gold").
SCHEMA_SCORE_DIMENSION_KEYS = (
    "naming_consistency",
    "redundancy_quality",
    "domain_plausibility",
    "field_specificity",
    "schema_coherence",
    "estimated_extraction_utility",
)


def _validate_dimension_block(label: str, block: Any) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"dimension_scores.{label} must be an object")
    for k in SCHEMA_SCORE_DIMENSION_KEYS:
        if k not in block:
            raise ValueError(f"dimension_scores.{label} missing '{k}'")
        v = block[k]
        if not isinstance(v, (int, float)):
            raise ValueError(f"dimension_scores.{label}.{k} must be a number")
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"dimension_scores.{label}.{k} must be in [0,1], got {v}")


def _validate_preference_output(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Judge output must be a JSON object")
    if "dimension_scores" not in data:
        raise ValueError("Missing 'dimension_scores'")
    ds = data["dimension_scores"]
    if not isinstance(ds, dict):
        raise ValueError("dimension_scores must be an object")
    if "A" not in ds or "B" not in ds:
        raise ValueError("dimension_scores must contain 'A' and 'B'")
    _validate_dimension_block("A", ds["A"])
    _validate_dimension_block("B", ds["B"])

    if "final_scores" not in data:
        raise ValueError("Missing 'final_scores'")
    fs = data["final_scores"]
    if not isinstance(fs, dict):
        raise ValueError("final_scores must be an object")
    for label in ("A", "B"):
        if label not in fs:
            raise ValueError(f"final_scores missing '{label}'")
        v = fs[label]
        if not isinstance(v, (int, float)):
            raise ValueError(f"final_scores.{label} must be a number")
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"final_scores.{label} must be in [0,1], got {v}")

    if "preferred" not in data:
        raise ValueError("Missing 'preferred'")
    pref = data["preferred"]
    if pref not in ("A", "B", "tie"):
        raise ValueError(f"preferred must be A, B, or tie; got {pref!r}")
    if "rationale_brief" not in data or not isinstance(data["rationale_brief"], str):
        raise ValueError("Missing or invalid rationale_brief")
    if "confidence" not in data:
        raise ValueError("Missing 'confidence'")
    c = data["confidence"]
    if not isinstance(c, (int, float)) or not 0.0 <= float(c) <= 1.0:
        raise ValueError("confidence must be a number in [0,1]")


BLIND_PAIRWISE_SYSTEM_PROMPT = """
Assume the role of an impartial expert in information extraction and schema design for
unstructured document collections (e.g. regulatory filings, web pages, directories).

You will compare two anonymous candidate attribute lists, labeled only as **Schema A** and
**Schema B**. You do NOT know how either list was produced. Do not infer or state that either
list is "official", "human", "benchmark", "model", or "ground truth".

WORKFLOW (must follow in order)
1) Score **Schema A alone** and **Schema B alone** on the six dimensions below (each 0.0–1.0,
   higher is better). Score each list on its own merits — do not peek at one list to excuse
   weaknesses in the other.
2) Assign **final_scores** for A and for B: each a single holistic 0.0–1.0 summary of that
   schema's quality, consistent with its six dimension scores (do not give A a 0.95 final if
   all its dimensions are near zero).
3) Choose **preferred**: "A", "B", or "tie" for which list is the better attribute set for
   downstream structured extraction in this **dataset_name** (domain context only).
4) **confidence** (0.0–1.0) reflects how certain you are about **preferred** (use low values
   when close).
5) **preferred** must be broadly consistent with **final_scores** and dimension scores (do not
   pick B if both final_scores and dimensions strongly favor A unless you explain in comparison_notes).

SIX DIMENSIONS (apply separately to A and to B)
• naming_consistency — uniform style (casing, delimiters); mixed styles score lower.
• redundancy_quality — 1.0 if field names are mutually distinct in meaning; lower if
  near-duplicates or overlapping concepts appear within the same list.
• domain_plausibility — do names look like plausible attributes for documents in dataset_name?
• field_specificity — penalize vague catch-alls ("info", "details", "misc", "other").
• schema_coherence — do the fields read as a coherent extraction checklist for the domain?
• estimated_extraction_utility — expected usefulness for a downstream extraction step.
  If the user message includes **reference_document**, let that excerpt inform this dimension
  for BOTH schemas (how well attributes align with what the excerpt discusses). If there is no
  excerpt, judge utility from names + dataset_name only.

Return ONLY valid JSON (no markdown fences, no text before or after) with exactly this shape:
{
  "dimension_scores": {
    "A": {
      "naming_consistency": 0.0,
      "redundancy_quality": 0.0,
      "domain_plausibility": 0.0,
      "field_specificity": 0.0,
      "schema_coherence": 0.0,
      "estimated_extraction_utility": 0.0
    },
    "B": {
      "naming_consistency": 0.0,
      "redundancy_quality": 0.0,
      "domain_plausibility": 0.0,
      "field_specificity": 0.0,
      "schema_coherence": 0.0,
      "estimated_extraction_utility": 0.0
    }
  },
  "final_scores": {
    "A": <holistic summary 0.0–1.0 for Schema A>,
    "B": <holistic summary 0.0–1.0 for Schema B>
  },
  "preferred": "A" | "B" | "tie",
  "confidence": <number from 0.0 to 1.0>,
  "rationale_brief": "<short paragraph>",
  "comparison_notes": "<optional short notes; use if dimensions conflict with preferred>"
}
""".strip()


def _call_gpt_4o_mini(user_payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in your environment.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": BLIND_PAIRWISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = _extract_first_json_object(content)
    _validate_preference_output(data)
    return data


def _default_output_path(dataset: str, run: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = SCHEMA_EVAL_ROOT / "judge_outputs" / dataset / run
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"judge_schema_preference_blind_{dataset}_{ts}.json"


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_winner(
    preferred: Literal["A", "B", "tie"],
    a_origin: Literal["run_schema", "labeled_reference_fields"],
) -> dict[str, Any]:
    """Map A/B preference back to our two sources (for the artifact only)."""
    if preferred == "tie":
        return {"outcome": "tie", "preferred_source": None}
    if preferred == "A":
        return {"outcome": "preferred_A", "preferred_source": a_origin}
    b_origin: Literal["run_schema", "labeled_reference_fields"] = (
        "labeled_reference_fields" if a_origin == "run_schema" else "run_schema"
    )
    return {"outcome": "preferred_B", "preferred_source": b_origin}


def _dimension_scores_by_source(
    output: dict[str, Any],
    a_origin: Literal["run_schema", "labeled_reference_fields"],
) -> dict[str, Any]:
    """Remap A/B scores to stable researcher labels (not shown to the model)."""
    ds = output["dimension_scores"]
    scores_a = ds["A"]
    scores_b = ds["B"]
    if a_origin == "run_schema":
        return {
            "run_schema": scores_a,
            "labeled_reference_fields": scores_b,
        }
    return {
        "run_schema": scores_b,
        "labeled_reference_fields": scores_a,
    }


def _final_scores_by_source(
    output: dict[str, Any],
    a_origin: Literal["run_schema", "labeled_reference_fields"],
) -> dict[str, float]:
    fs = output["final_scores"]
    fa = float(fs["A"])
    fb = float(fs["B"])
    if a_origin == "run_schema":
        return {"run_schema": fa, "labeled_reference_fields": fb}
    return {"run_schema": fb, "labeled_reference_fields": fa}


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind pairwise schema preference (2 LLM calls).")
    ap.add_argument("--run", required=True, help="Run folder under schema_eval_data/predicted/<dataset>/")
    ap.add_argument("--dataset", default="fda_510ks", choices=EVAPORATE_DATASET_NAMES)
    ap.add_argument("--doc-id", default=None, help="Reference doc basename for call 2.")
    ap.add_argument("--out", default=None, help="Output JSON path.")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for A/B ordering (default: nondeterministic).")
    ap.add_argument("--char-limit", type=int, default=6000, help="Max chars of reference document body.")
    args = ap.parse_args()

    dataset = args.dataset
    pred_path = _first_existing_pred_schema(dataset, args.run)
    run_schema = _load_pred_schema_names(pred_path)
    labeled_list = _load_gold_schema_names(dataset)

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    # True -> Schema A is the run output; False -> Schema A is the labeled reference list
    a_is_run_schema = rng.getrandbits(1) == 1
    if a_is_run_schema:
        schema_a, schema_b = run_schema, labeled_list
        a_origin: Literal["run_schema", "labeled_reference_fields"] = "run_schema"
    else:
        schema_a, schema_b = labeled_list, run_schema
        a_origin = "labeled_reference_fields"

    blind_payload_base = {
        "dataset_name": dataset,
        "comparison_task": "anonymous_pairwise_attribute_list_preference",
        "schema_a_field_names": schema_a,
        "schema_b_field_names": schema_b,
        "instruction": (
            "Choose whether Schema A or Schema B is better suited as the attribute set for "
            "extracting structured fields from documents in this dataset. If equally good, "
            "respond with tie."
        ),
    }

    # Call 1: schemas only
    payload_1 = dict(blind_payload_base)
    print("\n=== CALL 1 (blind, schemas only) — model input (no origin labels) ===")
    print(json.dumps(payload_1, ensure_ascii=False, indent=2))
    out1 = _call_gpt_4o_mini(payload_1)
    print("\n=== CALL 1 OUTPUT ===")
    print(json.dumps(out1, indent=2))

    doc_id = args.doc_id or _pick_doc_id_from_table(dataset)
    doc_body = _extract_doc_text_from_tar(dataset, doc_id, args.char_limit)
    ref = _reference_document_for_judge(doc_id, doc_body)

    payload_2 = dict(blind_payload_base)
    payload_2["reference_document"] = ref
    payload_2["instruction"] = (
        blind_payload_base["instruction"]
        + " You may use the reference_document only as topical context; do not treat it as a label for either schema."
    )

    print("\n=== CALL 2 (blind, schemas + reference doc) — model input ===")
    print(json.dumps(payload_2, ensure_ascii=False, indent=2))
    out2 = _call_gpt_4o_mini(payload_2)
    print("\n=== CALL 2 OUTPUT ===")
    print(json.dumps(out2, indent=2))

    provenance = {
        "schema_a_was": a_origin,
        "schema_b_was": (
            "labeled_reference_fields" if a_origin == "run_schema" else "run_schema"
        ),
        "random_seed_used": args.seed,
        "interpretation": (
            "schema_a_was=run_schema means your pipeline schema was shown as 'Schema A' for this run; "
            "labeled_reference_fields is the union of attribute names from schema_eval_data/ground_truth/.../ground_truth.json"
        ),
    }

    artifact = {
        "evaluation_type": "blind_pairwise_preference",
        "dataset": dataset,
        "model": "gpt-4o-mini",
        "run": args.run,
        "provenance_for_researchers_only": provenance,
        "call_1": {
            "input_sent_to_model": payload_1,
            "output": out1,
            "resolved": _resolve_winner(out1["preferred"], a_origin),
            "dimension_scores_by_source": _dimension_scores_by_source(out1, a_origin),
            "final_scores_by_source": _final_scores_by_source(out1, a_origin),
        },
        "call_2": {
            "input_sent_to_model": payload_2,
            "output": out2,
            "resolved": _resolve_winner(out2["preferred"], a_origin),
            "dimension_scores_by_source": _dimension_scores_by_source(out2, a_origin),
            "final_scores_by_source": _final_scores_by_source(out2, a_origin),
        },
        "reference_doc_id": doc_id,
        "pred_schema_path": str(pred_path.relative_to(REPO_ROOT)),
        "labeled_fields_path": str(_gold_schema_path(dataset).relative_to(REPO_ROOT)),
        "system_prompt": BLIND_PAIRWISE_SYSTEM_PROMPT,
    }

    out_path = Path(args.out) if args.out else _default_output_path(dataset, args.run)
    _write_artifact(out_path, artifact)
    print(f"\nSaved artifact (includes blind mapping for analysis): {out_path}")

    print("\n=== RESOLVED + SCORES BY SOURCE (researcher-only mapping) ===")
    print(
        json.dumps(
            {
                "call_1": {
                    "resolved": artifact["call_1"]["resolved"],
                    "dimension_scores_by_source": artifact["call_1"]["dimension_scores_by_source"],
                    "final_scores_by_source": artifact["call_1"]["final_scores_by_source"],
                },
                "call_2": {
                    "resolved": artifact["call_2"]["resolved"],
                    "dimension_scores_by_source": artifact["call_2"]["dimension_scores_by_source"],
                    "final_scores_by_source": artifact["call_2"]["final_scores_by_source"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

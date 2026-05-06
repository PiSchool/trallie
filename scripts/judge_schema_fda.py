"""
LLM-as-a-judge schema evaluation for Evaporate datasets (schema only).

Runs exactly TWO model calls per invocation. Each response is structured JSON containing:
mappings, missing/extra field lists, duplicate groups, per-dimension scores
(gold_coverage_recall, pred_precision, semantic_alignment, granularity_match,
domain_appropriateness, redundancy_quality, naming_consistency), rationale, and
final_score.

  Call 1 — predicted schema + gold schema + judge prompt  (no document)
  Call 2 — predicted schema + gold schema + judge prompt + one reference document text

Usage (PowerShell):
  $env:OPENAI_API_KEY="..."
  python scripts/judge_schema_fda.py --run gpt4omini_openie_nomemory

Optional flags:
  --doc-id   somefile.pdf      # pick a specific doc from docs.tar.gz
  --dataset  fda_510ks         # default; change for other Evaporate datasets
  --out      path/to/out.json  # override output path
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_EVAL_ROOT = REPO_ROOT / "schema_eval_data"

# ── Paths are parameterised by dataset; resolved in main() ──────────────────
def _pred_root(dataset: str) -> Path:
    return SCHEMA_EVAL_ROOT / "predicted" / dataset

def _gold_schema_path(dataset: str) -> Path:
    return SCHEMA_EVAL_ROOT / "ground_truth" / dataset / "ground_truth.json"

def _evaporate_docs_tar(dataset: str) -> Path:
    return REPO_ROOT / "evaporate" / "data" / dataset / "docs.tar.gz"

def _evaporate_table(dataset: str) -> Path:
    return REPO_ROOT / "evaporate" / "data" / dataset / "table.json"


# ── Judge prompt ─────────────────────────────────────────────────────────────
SCHEMA_GENERATION_JUDGE_PROMPT = """
You are a senior expert in structured information extraction and schema evaluation. You have
deep familiarity with heterogeneous document collections, including regulatory filings
(FDA 510(k) submissions), email corpora (Enron), sports encyclopedias (NBA player bios),
movie databases (IMDb, Rotten Tomatoes, Metacritic, AllMovie, AMC, Hollywood.com,
iHeartMovies, Yahoo Movies), and university directories (CollegeProwler, eCampusTours,
Embark, MatchCollege, US News).

You evaluate whether a MODEL-GENERATED flat schema (a list of field names) aligns with a
GROUND-TRUTH flat schema derived from labeled extractions in the Evaporate benchmark
(Arora et al., VLDB 2023). The benchmark spans 16 real-world settings; each setting has its
own domain vocabulary, so your evaluation must be grounded in the declared dataset_name.

The user message is JSON with:
  - dataset_name        (string, e.g. "fda_510ks", "wiki_nba_players", "swde_movie_imdb",
                         "swde_university_usnews", "enron")
  - predicted_schema    (list of strings)
  - ground_truth_schema (list of strings)
  - reference_document  (optional — plain text from one document; may begin with a single-line
                         header "[reference_document_id: <basename>]" before the excerpt)

════════════════════════════════════════════════════════════════════════
DOMAIN VOCABULARY GUIDE  (use to inform semantic alignment and appropriateness)
════════════════════════════════════════════════════════════════════════
fda_510ks          → device trade name, applicant, K-number, device classification,
                     product code, predicate device name/code, indications for use,
                     substantial equivalence, submission type, decision date, contact
enron              → sender, recipients, date, subject, cc, bcc, body, attachments,
                     thread id, folder, organization
wiki_nba_players   → player name, position, draft year, team, college, nationality,
                     height, weight, points per game, rebounds, assists, career stats
swde_movie_*       → title, director, genre, rating (MPAA), runtime, release date,
                     cast, producer, studio, user score, critic score, synopsis
swde_university_*  → university name, location, tuition, enrollment, acceptance rate,
                     SAT/ACT scores, ranking, housing, financial aid, majors

If dataset_name is absent or unrecognized, infer domain from field names in both schemas
and state the inference in domain_notes.

════════════════════════════════════════════════════════════════════════
TASK
════════════════════════════════════════════════════════════════════════
Grade predicted_schema against ground_truth_schema. Produce explicit field alignments,
then score seven dimensions on a continuous scale 0.0 (poor) → 1.0 (excellent), plus a
final_score on 0.0–1.0.

════════════════════════════════════════════════════════════════════════
MATCHING PROTOCOL  (apply before scoring)
════════════════════════════════════════════════════════════════════════
1. Align predicted fields to gold fields using exact names, clear synonyms, or obvious
   abbreviations (e.g., "trade name" ↔ "device trade name", "pts_per_game" ↔ "points per
   game"). Minor punctuation/casing/underscore differences do not matter.
2. One-to-one: each predicted field maps to at most one gold field; each gold field to at
   most one predicted field. On ties, choose the single best semantic pair; leave the rest
   unmatched.
3. link_type for each aligned pair:
     "exact"   – same concept, trivially equivalent names
     "synonym" – same concept, clearly different but unambiguous names
     "partial" – related but imperfect (overlapping scope or one is a subset)
     "none"    – no meaningful relation (use only when marking extra_pred or missing_gold)
4. missing_gold   → every gold field with no aligned predicted field.
5. extra_pred     → every predicted field with no aligned gold field.
6. duplicate_pred_groups → lists of predicted names that are near-duplicates or redundant
   within the predicted schema (e.g., both "movie title" and "film name" present). Use []
   if none.

════════════════════════════════════════════════════════════════════════
NUMBERED EVALUATION CRITERIA
════════════════════════════════════════════════════════════════════════
1. gold_coverage_recall
   Fraction of gold fields covered by an aligned predicted field.
   • exact/synonym alignment counts fully.
   • partial alignment counts as 0.5 of a covered field.
   • Anchor: 1.0 ≈ virtually all gold concepts present; 0.3 ≈ most gold concepts missing.

2. pred_precision
   Fraction of predicted fields that map to a genuine gold concept (not extra_pred, not
   vacuous). Penalize vague catch-alls ("details", "info", "other", "misc").
   • Anchor: 1.0 ≈ every predicted slot earns its place; 0.2 ≈ mostly hallucinated labels.

3. semantic_alignment
   Among linked pairs (exact/synonym/partial), how well do predicted names reflect the
   same meaning as gold names — not just string overlap? Penalize overly generic labels
   even when the link_type is "synonym".
   • Anchor: 1.0 ≈ crisp, domain-appropriate wording throughout;
             0.3 ≈ many weak, vague, or misleading links.

4. granularity_match
   Does the predicted schema avoid improper merging or splitting?
   • Penalize: one predicted field bundling multiple distinct gold concepts (merge error),
     or one gold concept fragmented across multiple predicted fields (split error).
   • Explain specific merge/split issues in granularity_notes.
   • Anchor: 1.0 ≈ every predicted field maps cleanly to one gold concept and vice versa.

5. domain_appropriateness
   Do predicted names look like attributes a data analyst or domain expert would use on
   real documents from dataset_name?
   • If reference_document is present, ground your judgment in topics that actually appear
     (reason at schema level; do not quote the document).
   • If reference_document is absent, infer from dataset_name + field names and note the
     limitation in domain_notes.
   • For fda_510ks: apply strict 510(k) regulatory vocabulary.
   • For swde_movie_* / swde_university_*: apply SWDE web-extraction conventions.
   • For wiki_nba_players / enron: apply encyclopedia / email conventions respectively.
   • Anchor: 1.0 ≈ every predicted field looks native to the domain;
             0.3 ≈ generic or wrong-domain labels throughout.

6. redundancy_quality
   1.0 if predicted fields are mutually distinct in purpose. Penalize proportionally to
   the size and number of duplicate_pred_groups.
   • Anchor: 1.0 ≈ no redundancy; 0.4 ≈ several pairs of near-duplicate predicted fields.

7. naming_consistency
   Are predicted field names internally consistent in style (casing, delimiter, verbosity)?
   Mixed styles (snake_case alongside Title Case alongside abbreviations) reduce clarity
   for downstream use.
   • Anchor: 1.0 ≈ uniform naming style; 0.5 ≈ noticeable style variation;
             0.2 ≈ chaotic, unsystematic names.

════════════════════════════════════════════════════════════════════════
FINAL SCORE
════════════════════════════════════════════════════════════════════════
final_score is your overall judgment. It must be CONSISTENT with the seven dimension scores
(do not output 0.9 if recall and precision are both below 0.5).
When reference_document is absent, apply a modest conservative cap to domain_appropriateness
and final_score and state this in domain_notes.

════════════════════════════════════════════════════════════════════════
CALIBRATION ANCHORS  (illustrative; do not copy into output)
════════════════════════════════════════════════════════════════════════
• Predicted renames most gold fields clearly → high gold_coverage_recall and pred_precision.
• Predicted adds "misc_info" and "text_blob" → pred_precision and semantic_alignment drop.
• One predicted "device_info" maps to three gold fields mentally → granularity_match drops.
• Dataset is swde_movie_imdb; predicted schema omits rating and cast → domain_appropriateness
  drops when reference_document confirms those appear in docs.
• All predicted fields use snake_case → naming_consistency near 1.0.
• "movie_title" and "film_title" both in predicted → duplicate_pred_groups non-empty,
  redundancy_quality drops.

════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════════════════════
Return ONLY valid JSON — no markdown fences, no preamble, no commentary after.
Use the exact field name strings from the user JSON inside mappings, missing_gold,
extra_pred, and duplicate_pred_groups.

Required JSON shape:
{
  "dataset_name_used": "<echo dataset_name back, or state inferred domain>",
  "mappings": [
    {
      "predicted_field": "<string from predicted_schema or null>",
      "gold_field": "<string from ground_truth_schema or null>",
      "link_type": "exact | synonym | partial | none",
      "alignment_note": "<short string>"
    }
  ],
  "missing_gold": [],
  "extra_pred": [],
  "duplicate_pred_groups": [],
  "granularity_notes": "",
  "domain_notes": "",
  "dimension_scores": {
    "gold_coverage_recall": 0.0,
    "pred_precision": 0.0,
    "semantic_alignment": 0.0,
    "granularity_match": 0.0,
    "domain_appropriateness": 0.0,
    "redundancy_quality": 0.0,
    "naming_consistency": 0.0
  },
  "rationale_brief": "",
  "final_score": 0.0
}
""".strip()


# ── Dimension keys — must stay in sync with the prompt's dimension_scores block ──
DIMENSION_KEYS = (
    "gold_coverage_recall",
    "pred_precision",
    "semantic_alignment",
    "granularity_match",
    "domain_appropriateness",   # renamed from fda_appropriateness
    "redundancy_quality",
    "naming_consistency",       # new 7th dimension
)

# All 16 Evaporate dataset names for reference / --dataset validation
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


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing_pred_schema(dataset: str, run: str) -> Path:
    p = _pred_root(dataset) / run / "schema.json"
    if not p.is_file():
        root = _pred_root(dataset)
        available = sorted({d.name for d in root.iterdir() if d.is_dir()}) if root.is_dir() else []
        raise FileNotFoundError(
            f"Predicted schema not found at {p}. Available runs: {available}"
        )
    return p


def _load_gold_schema_names(dataset: str) -> list[str]:
    path = _gold_schema_path(dataset)
    gold = _load_json(path)
    names = gold.get("attribute_names")
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"Invalid gold schema at {path}")
    return names


def _load_pred_schema_names(path: Path) -> list[str]:
    pred = _load_json(path)
    if not isinstance(pred, list) or not all(isinstance(x, str) for x in pred):
        raise ValueError(f"Invalid predicted schema list at {path}")
    return pred


def _pick_doc_id_from_table(dataset: str) -> str:
    table = _load_json(_evaporate_table(dataset))
    if not isinstance(table, dict) or not table:
        raise ValueError(f"Unexpected evaporate table format at {_evaporate_table(dataset)}")
    k = next(iter(table.keys()))
    return Path(k).name


def _extract_doc_text_from_tar(dataset: str, doc_id: str, char_limit: int = 6000) -> str:
    """
    Pull a single file from docs.tar.gz without full extraction.
    Returns a (possibly truncated) plain-text string — NOT a dict.
    The new judge prompt expects reference_document to be a plain text string.
    """
    tar_path = _evaporate_docs_tar(dataset)
    if not tar_path.is_file():
        raise FileNotFoundError(
            f"Missing {tar_path}. Ensure Git LFS finished downloading the Evaporate clone."
        )

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
            raise FileNotFoundError(f"Could not extract member {member_name} from tar")
        raw = f.read()

    # Best-effort text decoding
    text = ""
    for enc in ("utf-8", "latin-1"):
        try:
            candidate = raw.decode(enc)
            if "\x00" not in candidate:
                text = candidate
                break
        except UnicodeDecodeError:
            continue

    # Fallback: DataHandler (e.g. for PDFs), if available in the project
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

    # Return plain string — the judge prompt expects reference_document as a text snippet,
    # NOT a {"doc_id": ..., "text": ...} dict.
    return text


def _reference_document_for_judge(doc_id: str, text: str) -> str:
    """Single string so the model knows which file the excerpt came from."""
    header = f"[reference_document_id: {doc_id}]"
    body = (text or "").strip()
    if not body:
        return f"{header}\n\n[no extractable text from this file]"
    return f"{header}\n\n{body}"


def _extract_first_json_object(content: str) -> dict[str, Any]:
    """
    Parse the first JSON object from model output.
    Tries full-string parse, then scans for a balanced object via JSONDecoder.raw_decode
    (avoids greedy-regex failures when extra braces appear).
    """
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


# ── OpenAI call ───────────────────────────────────────────────────────────────

def _call_gpt_4o_mini(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in your environment.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = _extract_first_json_object(content)
    _validate_judge_output(data)
    return data


# ── Output validation ─────────────────────────────────────────────────────────

def _validate_judge_output(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Judge output must be a JSON object, got: {type(data)}")

    required_top_level = (
        "dataset_name_used",        # new field — echoes back the dataset
        "mappings",
        "missing_gold",
        "extra_pred",
        "duplicate_pred_groups",
        "granularity_notes",
        "domain_notes",             # renamed from fda_notes
        "dimension_scores",
        "rationale_brief",
        "final_score",
    )
    for key in required_top_level:
        if key not in data:
            raise ValueError(f"Judge output missing required key '{key}'. Keys present: {list(data.keys())}")

    if not isinstance(data["mappings"], list):
        raise ValueError("mappings must be a list")

    for i, entry in enumerate(data["mappings"]):
        if not isinstance(entry, dict):
            raise ValueError(f"mappings[{i}] must be a dict")
        for field in ("predicted_field", "gold_field", "link_type", "alignment_note"):
            if field not in entry:
                raise ValueError(f"mappings[{i}] missing '{field}'")
        if entry["link_type"] not in ("exact", "synonym", "partial", "none"):
            raise ValueError(f"mappings[{i}].link_type invalid: {entry['link_type']!r}")

    if not isinstance(data["missing_gold"], list):
        raise ValueError("missing_gold must be a list")
    if not isinstance(data["extra_pred"], list):
        raise ValueError("extra_pred must be a list")
    if not isinstance(data["duplicate_pred_groups"], list):
        raise ValueError("duplicate_pred_groups must be a list")

    ds = data["dimension_scores"]
    if not isinstance(ds, dict):
        raise ValueError("dimension_scores must be an object")

    for k in DIMENSION_KEYS:   # validates all 7 keys, including domain_appropriateness and naming_consistency
        if k not in ds:
            raise ValueError(f"dimension_scores missing '{k}'")
        v = ds[k]
        if not isinstance(v, (int, float)):
            raise ValueError(f"dimension_scores.{k} must be a number, got {type(v)}")
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"dimension_scores.{k} must be in [0,1], got {v}")

    fs = data["final_score"]
    if not isinstance(fs, (int, float)):
        raise ValueError(f"final_score must be a number, got {type(fs)}")
    if not 0.0 <= float(fs) <= 1.0:
        raise ValueError(f"final_score must be in [0,1], got {fs}")


# ── Output path ───────────────────────────────────────────────────────────────

def _default_output_path(dataset: str, run: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = SCHEMA_EVAL_ROOT / "judge_outputs" / dataset / run
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"judge_schema_{dataset}_{ts}.json"


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="Run folder under schema_eval_data/predicted/<dataset> (e.g. gpt4omini_openie_nomemory).",
    )
    ap.add_argument(
        "--dataset",
        default="fda_510ks",
        choices=EVAPORATE_DATASET_NAMES,
        help="Evaporate dataset name (default: fda_510ks).",
    )
    ap.add_argument(
        "--doc-id",
        default=None,
        help="Optional basename of a reference doc inside evaporate/data/<dataset>/docs.tar.gz.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path. If omitted, a timestamped file is created under schema_eval_data/judge_outputs/.",
    )
    args = ap.parse_args()

    dataset = args.dataset
    pred_schema_path = _first_existing_pred_schema(dataset, args.run)
    predicted = _load_pred_schema_names(pred_schema_path)
    gold = _load_gold_schema_names(dataset)

    # ── Call 1: schema-only (no reference document) ──────────────────────────
    payload_1 = {
        "dataset_name": dataset,          # key the prompt expects
        "predicted_schema": predicted,
        "ground_truth_schema": gold,
    }
    user_1 = json.dumps(payload_1, ensure_ascii=False, indent=2)
    print("\n=== CALL 1 (schema-only) INPUT ===")
    print(user_1)
    out1 = _call_gpt_4o_mini(SCHEMA_GENERATION_JUDGE_PROMPT, user_1)
    print("\n=== CALL 1 (schema-only) OUTPUT ===")
    print(json.dumps(out1, indent=2))

    # ── Call 2: schema + one reference document ───────────────────────────────
    doc_id = args.doc_id or _pick_doc_id_from_table(dataset)
    # reference_document is a plain text string, matching the prompt spec
    doc_text = _extract_doc_text_from_tar(dataset, doc_id)
    payload_2 = {
        "dataset_name": dataset,          # key the prompt expects
        "predicted_schema": predicted,
        "ground_truth_schema": gold,
        "reference_document": _reference_document_for_judge(doc_id, doc_text),
    }
    user_2 = json.dumps(payload_2, ensure_ascii=False, indent=2)
    print("\n=== CALL 2 (with reference doc) INPUT ===")
    print(user_2)
    out2 = _call_gpt_4o_mini(SCHEMA_GENERATION_JUDGE_PROMPT, user_2)
    print("\n=== CALL 2 (with reference doc) OUTPUT ===")
    print(json.dumps(out2, indent=2))

    # ── Persist full artifact ─────────────────────────────────────────────────
    evaporate_table = _evaporate_table(dataset)
    evaporate_tar = _evaporate_docs_tar(dataset)
    artifact = {
        "dataset": dataset,
        "model": "gpt-4o-mini",
        "run": args.run,
        "pred_schema_path": str(pred_schema_path.relative_to(REPO_ROOT)),
        "gold_schema_path": str(_gold_schema_path(dataset).relative_to(REPO_ROOT)),
        "evaporate_table_path": str(evaporate_table.relative_to(REPO_ROOT)) if evaporate_table.exists() else None,
        "evaporate_docs_tar": str(evaporate_tar.relative_to(REPO_ROOT)) if evaporate_tar.exists() else None,
        "reference_doc_id": doc_id,
        "judge_prompt": SCHEMA_GENERATION_JUDGE_PROMPT,
        "call_1": {"input": payload_1, "output": out1},
        "call_2": {"input": payload_2, "output": out2},
    }

    out_path = Path(args.out) if args.out else _default_output_path(dataset, args.run)
    _write_artifact(out_path, artifact)
    print(f"\nSaved artifact to: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(
        json.dumps(
            {
                "call_1_final_score": out1.get("final_score"),
                "call_2_final_score": out2.get("final_score"),
                "call_1_dimension_scores": out1.get("dimension_scores"),
                "call_2_dimension_scores": out2.get("dimension_scores"),
                "call_1_dataset_name_used": out1.get("dataset_name_used"),
                "call_2_dataset_name_used": out2.get("dataset_name_used"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
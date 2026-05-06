"""
Compare predicted OpenIE schemas (attribute name lists) to gold attribute_names
in schema_eval_data/ground_truth/*/ground_truth.json.

Normalization: lowercased, underscores and hyphens -> spaces, collapsed whitespace.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "schema_eval_data"
PRED = SCHEMA_ROOT / "predicted"
GT = SCHEMA_ROOT / "ground_truth"


def norm(name: str) -> str:
    s = name.lower().replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def load_predicted(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}")
    return {norm(x) for x in raw if isinstance(x, str) and x.strip()}


def load_gold(path: Path) -> tuple[set[str], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") == "missing_source_table" or not data.get("attribute_names"):
        return set(), data
    return {norm(x) for x in data["attribute_names"]}, data


def prf1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main() -> None:
    if not PRED.is_dir():
        print("No schema_eval_data/predicted; run scripts/build_schema_eval_data.py first.")
        return

    rows: list[tuple[str, str, float, float, float, int, int, int]] = []

    for schema_path in sorted(PRED.glob("*/*/schema.json")):
        rel = schema_path.relative_to(PRED)
        dataset, run = rel.parts[0], rel.parts[1]
        gt_path = GT / dataset / "ground_truth.json"
        if not gt_path.is_file():
            continue

        gold_set, meta = load_gold(gt_path)
        pred_set = load_predicted(schema_path)
        prec, rec, f1 = prf1(pred_set, gold_set)
        tp = len(pred_set & gold_set)
        rows.append(
            (dataset, run, prec, rec, f1, tp, len(pred_set), len(gold_set))
        )

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"{'dataset':<38} {'run':<32} {'P':>7} {'R':>7} {'F1':>7} {'tp':>5} {'|P|':>5} {'|G|':>5}")
    print("-" * 120)
    for dataset, run, prec, rec, f1, tp, npred, ngold in rows:
        print(
            f"{dataset:<38} {run:<32} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} "
            f"{tp:>5} {npred:>5} {ngold:>5}"
        )
    print("-" * 120)
    print(f"rows={len(rows)}  (skipped runs with no ground_truth.json)")


if __name__ == "__main__":
    main()

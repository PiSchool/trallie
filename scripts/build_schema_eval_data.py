"""
Populate schema_eval_data/: predicted schemas from results + ground-truth attribute sets.
Re-run after adding new result runs or datasets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_GLOB = REPO_ROOT / "results (1)" / "results"
EVAPORATE_DATA = REPO_ROOT / "evaporate" / "data"
OUT_ROOT = REPO_ROOT / "schema_eval_data"


def copy_predicted_schemas() -> int:
    OUT_PRED = OUT_ROOT / "predicted"
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    n = 0
    if not RESULTS_GLOB.is_dir():
        return 0
    for schema_path in RESULTS_GLOB.glob("*/*/schema.json"):
        parts = schema_path.relative_to(RESULTS_GLOB).parts
        if len(parts) != 3:
            continue
        dataset, run_name, _ = parts
        dest_dir = OUT_PRED / dataset / run_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(schema_path, dest_dir / "schema.json")
        n += 1
    return n


def extract_ground_truth_attributes(table_path: Path) -> tuple[list[str], int]:
    with open(table_path, "r", encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table, dict):
        raise ValueError(f"Expected object at top level: {table_path}")
    names: set[str] = set()
    for row in table.values():
        if isinstance(row, dict):
            names.update(row.keys())
    return sorted(names), len(table)


def write_ground_truth_files() -> int:
    OUT_GT = OUT_ROOT / "ground_truth"
    OUT_GT.mkdir(parents=True, exist_ok=True)
    n = 0
    for dataset_dir in sorted(EVAPORATE_DATA.iterdir()):
        if not dataset_dir.is_dir():
            continue
        table_path = dataset_dir / "table.json"
        if not table_path.is_file():
            continue
        attrs, num_docs = extract_ground_truth_attributes(table_path)
        rel = table_path.relative_to(REPO_ROOT).as_posix()
        payload = {
            "dataset": dataset_dir.name,
            "source_table_relative": rel,
            "num_documents": num_docs,
            "attribute_names": attrs,
            "note": "attribute_names is the union of keys across all rows in source table.json (gold schema).",
        }
        dest = OUT_GT / dataset_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        n += 1
    return n


def write_missing_ground_truth_stubs() -> int:
    """For predicted datasets with no evaporate table, write a stub ground_truth.json."""
    OUT_PRED = OUT_ROOT / "predicted"
    OUT_GT = OUT_ROOT / "ground_truth"
    n = 0
    if not OUT_PRED.is_dir():
        return 0
    for dataset_dir in OUT_PRED.iterdir():
        if not dataset_dir.is_dir():
            continue
        name = dataset_dir.name
        if (EVAPORATE_DATA / name / "table.json").is_file():
            continue
        stub = OUT_GT / name / "ground_truth.json"
        if stub.is_file():
            continue
        dest = OUT_GT / name
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": name,
            "source_table_relative": None,
            "num_documents": None,
            "attribute_names": [],
            "status": "missing_source_table",
            "note": "No evaporate/data/{}/table.json in this workspace when stub was created.".format(
                name
            ),
        }
        with open(stub, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        n += 1
    return n


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pc = copy_predicted_schemas()
    gc = write_ground_truth_files()
    sc = write_missing_ground_truth_stubs()
    print(
        f"schema_eval_data: copied {pc} predicted schema(s); "
        f"wrote {gc} ground_truth.json from evaporate; "
        f"{sc} stub(s) for datasets without table.json."
    )


if __name__ == "__main__":
    main()

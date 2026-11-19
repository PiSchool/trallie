"""
Score prediction tables with EVAPORATE's own evaluate_synthetic.py (unmodified),
by converting them to the files its evaluators read.

    git clone --depth 1 https://github.com/HazyResearch/evaporate.git /tmp/evaporate
    python benchmarking/evaporate_official_eval.py /tmp/evaporate . /tmp/evap_work [results_dir]
"""
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

evap_repo, trallie_repo, work = map(Path, sys.argv[1:4])
results_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else trallie_repo / "trallie-eval-result"
work.mkdir(parents=True, exist_ok=True)

# evaporate/utils.py imports `manifest`; evaluate_synthetic only needs get_file_attribute from it.
pkg = types.ModuleType("evaporate")
pkg.__path__ = []
sys.modules["evaporate"] = pkg


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


esu = load("evaporate.evaluate_synthetic_utils", evap_repo / "evaporate/evaluate_synthetic_utils.py")
stub = types.ModuleType("evaporate.utils")
stub.get_file_attribute = esu.get_file_attribute
sys.modules["evaporate.utils"] = stub
es = load("evaporate.evaluate_synthetic", evap_repo / "evaporate/evaluate_synthetic.py")

gold_file = trallie_repo / "data/evaluation/data/fda_510ks/table.json"
gold = json.loads(gold_file.read_text())
gold_dir = "/fda"  # evaluate_extraction_quality re-keys gold as <dir>/<basename>
attrs = sorted({a for v in gold.values() for a in v})
raw_key = {k.rsplit("/", 1)[-1]: k for k in gold}
args = SimpleNamespace(generative_index_path=str(work), set_dicts=None)

rows = []
for f in sorted(results_dir.glob("*fda_510ks_*_predicted_table.json")):
    stem = f.stem.replace("_predicted_table", "")
    mode = "openie" if stem.endswith("openie") else "closedie"
    pred = {k.rsplit("/", 1)[-1]: v for k, v in json.loads(f.read_text()).items()}
    run = stem.replace("/", "_")
    if mode == "closedie":
        for a in attrs:
            # Every file Trallie processed is present; a failed doc becomes "" (scored 0 vs non-empty gold)
            f2m = {f"{gold_dir}/{d}": ((v or {}).get(a) or "") for d, v in pred.items() if d in raw_key}
            (work / f"{run}_{esu.get_file_attribute(a)}_file2metadata.json").write_text(json.dumps(f2m))
        res = es.evaluate_extraction_quality(run, args, str(gold_file), gold_dir, gold_attributes=attrs)
        rows.append((stem, mode, 100 * res["f1"], res["total_attributes"]))
    else:
        f2e = {raw_key[d]: {k: ("" if v is None else v) for k, v in (row or {}).items()}
               for d, row in pred.items() if d in raw_key}
        (work / f"{run}_file2extractions.json").write_text(json.dumps(f2e))
        res = es.evaluate_openie_quality(run, args, str(gold_file))
        rows.append((stem, mode, 100 * res["f1"], f"P={100*res['precision']:.1f} R={100*res['recall']:.1f}"))

print("\n=== EVAPORATE's own evaluator ===")
for stem, mode, f1, extra in rows:
    print(f"{mode:9} {stem:60} F1={f1:5.1f}  {extra}")

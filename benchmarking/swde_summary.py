#!/usr/bin/env python3
"""
Per-site and per-vertical SWDE scores (EVAPORATE metrics), averaged over the
sites in each vertical, next to the EVAPORATE paper's numbers.

    python benchmarking/swde_summary.py --results-dir results_swde
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_baselines import basename, closedie_f1, load_gold, openie_pair_f1, parse_name  # noqa: E402

# Paper Tables 1-2 (text-davinci-003). Aggregation over sites is not stated in the paper.
PAPER = {
    "movie": {"EVAPORATE-Code+": (79.5, 56.8), "EVAPORATE-Direct": (84.4, 45.2), "DOM-LM": (71.9, 54.1)},
    "university": {"EVAPORATE-Code+": (73.7, 59.0), "EVAPORATE-Direct": (72.6, 53.8), "DOM-LM": (68.0, 55.2)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_swde")
    args = ap.parse_args()
    res = Path(args.results_dir)

    per_site = defaultdict(dict)  # (model, mode) -> site -> (f1, failed, docs)
    for f in sorted(res.glob("*_swde_*_predicted_table.json")):
        site = "swde_" + f.name.split("_swde_", 1)[1].rsplit("_", 3)[0]
        model, mode = parse_name(f, site)
        pred = {basename(k): v for k, v in json.loads(f.read_text()).items()}
        gold = {d: v for d, v in load_gold(site).items() if d in pred}
        failed = sum(1 for d in gold if not pred.get(d))
        if mode == "closedie":
            f1 = closedie_f1(pred, gold)[0]
        else:
            schema = f.with_name(f.name.replace("_predicted_table.json", "_schema.json"))
            ranked = list(json.loads(schema.read_text())) if schema.exists() else None
            f1 = openie_pair_f1(pred, gold, ranked)[0]
        n_attrs = len({a for v in gold.values() for a in v})
        per_site[(model, mode)][site] = (f1, failed, len(gold), n_attrs)

    print(f"{'site':32} {'gold attrs':>10} " + " ".join(f"{m[:14]+'/'+md[:6]:>22}" for m, md in sorted(per_site)))
    sites = sorted({s for v in per_site.values() for s in v})
    for s in sites:
        n = next(v[s][3] for v in per_site.values() if s in v)
        cells = []
        for key in sorted(per_site):
            v = per_site[key].get(s)
            cells.append(f"{v[0]:16.1f} f={v[1]:<3}" if v else f"{'-':>22}")
        print(f"{s:32} {n:10} " + " ".join(cells))

    print("\n== Vertical averages (mean over sites) ==")
    for vertical in ("movie", "university"):
        print(f"\n-- SWDE {vertical} --   ClosedIE F1   OpenIE F1")
        models = sorted({m for m, _ in per_site})
        for m in models:
            vals = []
            for mode in ("closedie", "openie"):
                xs = [v[0] for s, v in per_site.get((m, mode), {}).items() if f"_{vertical}_" in s]
                vals.append(f"{sum(xs) / len(xs):9.1f} ({len(xs)} sites)" if xs else f"{'-':>17}")
            print(f"Trallie + {m:28} {vals[0]}  {vals[1]}")
        for name, (c, o) in PAPER[vertical].items():
            print(f"{name + ' [paper]':38} {c:9.1f}            {o:9.1f}")


if __name__ == "__main__":
    main()

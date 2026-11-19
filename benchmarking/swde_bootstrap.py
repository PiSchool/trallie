#!/usr/bin/env python3
"""
Bootstrap 95% CIs for Trallie's SWDE vertical means: resample pages (with
replacement) within each site, rescore, average over sites.

    python benchmarking/swde_bootstrap.py --results-dir results_swde --n 200
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_baselines import basename, closedie_f1, load_gold, openie_pair_f1, parse_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_swde")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    runs = defaultdict(list)  # (model, vertical, mode) -> [(pred, gold, ranked)]
    for f in sorted(Path(args.results_dir).glob("*_swde_*_predicted_table.json")):
        site = "swde_" + f.name.split("_swde_", 1)[1].rsplit("_", 3)[0]
        model, mode = parse_name(f, site)
        pred = {basename(k): v for k, v in json.loads(f.read_text()).items()}
        gold = {d: v for d, v in load_gold(site).items() if d in pred}
        schema = f.with_name(f.name.replace("_predicted_table.json", "_schema.json"))
        ranked = list(json.loads(schema.read_text())) if schema.exists() else None
        runs[(model, site.split("_")[1], mode)].append((pred, gold, ranked))

    def score(mode, pred, gold, ranked):
        return closedie_f1(pred, gold)[0] if mode == "closedie" else openie_pair_f1(pred, gold, ranked)[0]

    for (model, vertical, mode), sites in sorted(runs.items()):
        point = sum(score(mode, *s) for s in sites) / len(sites)
        boots = []
        for _ in range(args.n):
            total = 0.0
            for pred, gold, ranked in sites:
                docs = list(gold)
                sample = [rng.choice(docs) for _ in docs]
                # rename duplicates so a page drawn twice counts twice
                g = {f"{d}#{i}": gold[d] for i, d in enumerate(sample)}
                p = {f"{d}#{i}": pred.get(d) for i, d in enumerate(sample)}
                total += score(mode, p, g, ranked)
            boots.append(total / len(sites))
        boots.sort()
        lo, hi = boots[int(0.025 * args.n)], boots[int(0.975 * args.n) - 1]
        print(f"{model:24} SWDE {vertical:10} {mode:8} {point:5.1f}  95% CI [{lo:5.1f}, {hi:5.1f}]  ({len(sites)} sites)")


if __name__ == "__main__":
    main()

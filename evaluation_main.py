"""
Run Trallie OpenIE/ClosedIE over the EVAPORATE benchmark datasets, then score
with EVAPORATE's metrics (benchmarking/compare_baselines.py).

API keys are read from the environment (GROQ_API_KEY / OPENAI_API_KEY).

    python evaluation_main.py --datasets fda_510ks --models groq:llama-3.3-70b-versatile openai:gpt-4o
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmarking"))
from evaluation_params import get_dataset_schema, get_evaluation_params  # noqa: E402

from trallie.wrappers import closedie, openie, result_tag  # noqa: E402

REASONING_MODELS = {"deepseek-r1-distill-llama-70b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["fda_510ks"])
    ap.add_argument("--models", nargs="+", default=["groq:llama-3.3-70b-versatile", "groq:llama-3.1-8b-instant"],
                    help="provider:model pairs")
    ap.add_argument("--modes", nargs="+", default=["openie", "closedie"], choices=["openie", "closedie"])
    ap.add_argument("--limit", type=int, default=100, help="Records per dataset")
    ap.add_argument("--top-k", type=int, default=20, help="OpenIE schema size")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    datasets = get_evaluation_params(Path("data/evaluation/data"))
    for dataset_name in args.datasets:
        dataset_info = datasets[dataset_name]
        records = sorted(glob.glob(str(dataset_info["files"])))[: args.limit]
        for spec in args.models:
            provider, model_name = spec.split(":", 1)
            reasoning_mode = model_name in REASONING_MODELS
            for mode in args.modes:
                out = f"{args.results_dir}/{result_tag(model_name)}_{dataset_name}_{mode}_predicted_table.json"
                if os.path.exists(out):
                    print(f"Skipping existing {out}")
                    continue
                if mode == "openie":
                    openie(dataset_info["description"], records, provider=provider, model_name=model_name,
                           reasoning_mode=reasoning_mode, dataset_name=dataset_name,
                           top_k=args.top_k, results_dir=args.results_dir)
                else:
                    closedie(records, get_dataset_schema(dataset_info["ground_truth"]), provider=provider,
                             model_name=model_name, reasoning_mode=reasoning_mode,
                             dataset_name=dataset_name, results_dir=args.results_dir)

        subprocess.run([sys.executable, "benchmarking/compare_baselines.py", "--dataset", dataset_name,
                        "--results-dir", args.results_dir], check=False)


if __name__ == "__main__":
    main()

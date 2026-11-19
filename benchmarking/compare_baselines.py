#!/usr/bin/env python3
"""
Score Trallie and baseline predictions with EVAPORATE's own metrics so the
numbers are comparable to the paper (Arora et al., VLDB 2024, arXiv:2304.09433).

- ClosedIE: SQuAD-style token F1 per (doc, attribute), averaged over docs per
  attribute, then macro-averaged over attributes (evaporate/evaluate_synthetic.py).
- OpenIE: Pair F1 over exact (doc, attribute, value) tuples, attributes matched
  by normalised name only, top-k attributes with k = #gold attributes.
- OpenIE (value-mapped): diagnostic only. Each gold attribute is mapped to the
  single predicted column whose values give the best average token F1 across
  docs, then scored as ClosedIE. Separates "schema naming" from "extraction".

    python benchmarking/compare_baselines.py --dataset fda_510ks
"""

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Published EVAPORATE results on the same data (text-davinci-003).
PAPER_BASELINES = {
    "fda_510ks": {
        "EVAPORATE-Code+": {"closedie": 80.1, "openie": 62.8},
        "EVAPORATE-Direct": {"closedie": 78.0, "openie": 45.5},
    }
}


def clean(value, attribute=""):
    """Port of EVAPORATE's clean_comparison (evaluate_synthetic_utils.py)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ", ".join(dict.fromkeys(str(v) for v in value if v is not None))
    elif isinstance(value, dict):
        value = ", ".join(f"{k} {v}" for k, v in value.items())
    text = str(value)
    if "<" in text and ">" in text:
        text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().replace("  ", " ").lower()
    # EVAPORATE removes the attribute name (and "(s)" variants) anywhere in the value
    for a in (f"{attribute}(s)".lower(), f"{attribute.strip()}(s)".lower(), attribute.lower()):
        if a:
            text = text.replace(a, "").strip()
    for ch in ["'", '"', "(", ")", ",", "/", "]", "[", ":"]:
        text = text.replace(ch, "")
    text = html.unescape(text)
    for ch in ["&amp;", "&", "-", "_", "\n", "\t", "http:", "<", ">"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def token_f1(pred, gold):
    p, g = pred.split(), gold.split()
    if g and p and len(p) > len(g) and p[: len(g)] == g:
        p = p[: len(g)]  # EVAPORATE's prefix truncation
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def norm_key(k):
    return re.sub(r"[^a-z0-9]", "", k.lower())


def basename(doc_id):
    return doc_id.rsplit("/", 1)[-1]


def load_gold(dataset):
    gold = json.loads(Path(f"data/evaluation/data/{dataset}/table.json").read_text())
    return {basename(k): v for k, v in gold.items()}


def closedie_f1(pred, gold):
    attrs = sorted({a for d in gold.values() for a in d})
    per_attr = {}
    for a in attrs:
        scores = []
        for doc, g in gold.items():
            p = (pred.get(doc) or {})
            # tolerate case/format differences in returned keys
            pv = p.get(a)
            if pv is None:
                pv = next((v for k, v in p.items() if norm_key(k) == norm_key(a)), None)
            gv = g.get(a)
            # EVAPORATE: comma lists equal up to order count as equal
            if isinstance(pv, str) and isinstance(gv, str) and "," in pv and "," in gv and \
                    sorted(x.strip() for x in pv.split(",")) == sorted(x.strip() for x in gv.split(",")):
                gv = pv
            pc, gc = clean(pv, a), clean(gv, a)
            # EVAPORATE: character-level prefix check, then truncate pred to gold's token count
            # (so an empty gold value always matches)
            if pc.strip(".").startswith(gc.strip(".")):
                pc = " ".join(pc.split()[: len(gc.split())])
            scores.append(token_f1(pc, gc))
        per_attr[a] = 100 * sum(scores) / len(scores)
    return sum(per_attr.values()) / len(per_attr), per_attr


def openie_pair_f1(pred, gold, ranked_attributes=None):
    """ranked_attributes: the system's own schema order (best first), if saved."""
    k = len({a for d in gold.values() for a in d})
    if ranked_attributes:
        top = {norm_key(a) for a in ranked_attributes[:k]}
    else:
        counts = Counter(norm_key(a) for d in pred.values() if d for a, v in d.items() if v not in (None, ""))
        top = {a for a, _ in counts.most_common(k)}
    pred_keys = {norm_key(a) for d in pred.values() if d for a in d}

    def gold_key(a):
        # EVAPORATE: the page's entity name is matched to a predicted "name" attribute
        if a == "topic_entity_name" and "name" in pred_keys:
            return "name"
        # EVAPORATE's second mapping pass: singular/plural variants (it uses str.strip('s'))
        for variant in (a, a.strip("s"), a + "s"):
            if norm_key(variant) in pred_keys:
                return norm_key(variant)
        return norm_key(a)

    gold_t = {(doc, gold_key(a), clean(v, a).replace(" ", ""))
              for doc, d in gold.items() for a, v in d.items() if clean(v, a)}
    pred_t = {(doc, norm_key(a), clean(v, a).replace(" ", ""))
              for doc, d in pred.items() if doc in gold and d
              for a, v in d.items() if norm_key(a) in top and clean(v, a)}
    if not gold_t or not pred_t:
        return 0.0, 0.0, 0.0
    tp = len(gold_t & pred_t)
    p, r = tp / len(pred_t), tp / len(gold_t)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * f, 100 * p, 100 * r


def openie_value_mapped(pred, gold):
    attrs = sorted({a for d in gold.values() for a in d})
    pred_keys = sorted({k for d in pred.values() if d for k in d})
    per_attr, mapping = {}, {}
    for a in attrs:
        best, best_k = 0.0, None
        for k in pred_keys:
            s = sum(token_f1(clean((pred.get(doc) or {}).get(k), a), clean(g.get(a), a))
                    for doc, g in gold.items()) / len(gold)
            if s > best:
                best, best_k = s, k
        per_attr[a], mapping[a] = 100 * best, best_k
    return sum(per_attr.values()) / len(per_attr), per_attr, mapping


def parse_name(path, dataset):
    stem = path.stem.replace("_predicted_table", "")
    mode = "openie" if stem.endswith("_openie") else "closedie"
    model = stem.replace(f"_{dataset}_{mode}", "").replace(f"{dataset}_{mode}", "") or "unlabelled-run"
    return model, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fda_510ks")
    ap.add_argument("--results-dir", default="trallie-eval-result")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--docs-from", default=None,
                    help="Score only the docs present (non-null) in this predictions file, "
                         "e.g. a partially finished baseline run")
    args = ap.parse_args()

    gold = load_gold(args.dataset)
    if args.docs_from:
        keep = {basename(k) for k, v in json.loads(Path(args.docs_from).read_text()).items() if v is not None}
        gold = {d: v for d, v in gold.items() if d in keep}
        print(f"Scoring on {len(gold)}-doc subset from {args.docs_from} (paper rows are full-set numbers)")
    rows, details = [], {}
    for f in sorted(Path(args.results_dir).glob(f"*{args.dataset}_*_predicted_table.json")):
        model, mode = parse_name(f, args.dataset)
        pred = {basename(k): v for k, v in json.loads(f.read_text()).items()}
        # Score the docs this run processed (failed ones count as 0), as EVAPORATE does;
        # e.g. a 100-page sample of a 2000-page SWDE site
        run_gold = {d: v for d, v in gold.items() if d in pred}
        n_null = sum(1 for d in run_gold if not pred.get(d))
        row = {"system": model, "mode": mode, "docs": len(run_gold), "failed_docs": n_null}
        if not run_gold:
            continue
        if mode == "closedie":
            row["f1"], per = closedie_f1(pred, run_gold)
            details[f"{model}/{mode}"] = {"per_attribute_f1": per}
        else:
            schema_file = f.with_name(f.name.replace("_predicted_table.json", "_schema.json"))
            ranked = list(json.loads(schema_file.read_text())) if schema_file.exists() else None
            row["f1"], row["precision"], row["recall"] = openie_pair_f1(pred, run_gold, ranked)
            row["value_mapped_f1"], per, mapping = openie_value_mapped(pred, run_gold)
            details[f"{model}/{mode}"] = {"per_attribute_value_mapped_f1": per, "mapping": mapping}
        rows.append(row)

    for name, scores in PAPER_BASELINES.get(args.dataset, {}).items():
        for mode, f1 in scores.items():
            rows.append({"system": name + " [paper]", "mode": mode, "f1": f1})

    for mode in ("closedie", "openie"):
        print(f"\n== {mode.upper()} ({args.dataset}) ==")
        for r in sorted((r for r in rows if r["mode"] == mode), key=lambda r: -r["f1"]):
            extra = ""
            if mode == "openie" and "precision" in r:
                extra = f"  P={r['precision']:.1f} R={r['recall']:.1f}  value-mapped F1={r['value_mapped_f1']:.1f}"
            fails = f"  failed={r['failed_docs']}" if "failed_docs" in r else ""
            print(f"{r['system']:<40} F1={r['f1']:5.1f}{extra}{fails}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows, "details": details}, indent=2))


if __name__ == "__main__":
    main()

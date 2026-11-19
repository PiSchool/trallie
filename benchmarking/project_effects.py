"""Project the effect of each Trallie change from saved (pre-change) outputs.

    python benchmarking/project_effects.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_baselines import clean, closedie_f1, load_gold, openie_pair_f1, token_f1  # noqa: E402

R = Path("trallie-eval-result")
gold = load_gold("fda_510ks")
attrs = sorted({a for d in gold.values() for a in d})
docs_text = {p.name: p.read_text(errors="ignore") for p in
             Path("data/evaluation/data/fda_510ks/data/evaporate/fda-ai-pmas/510k").glob("K*.txt")}


def load(name):
    return {k.rsplit("/", 1)[-1]: v for k, v in json.loads((R / name).read_text()).items()}


def rename_to_gold(pred):
    """Rename each predicted column to the gold attribute its values best match (one column per gold)."""
    cols = sorted({k for d in pred.values() if d for k in d})
    best = {}
    for c in cols:
        scores = {a: sum(token_f1(clean((pred.get(d) or {}).get(c), a), clean(g[a], a))
                         for d, g in gold.items()) / len(gold) for a in attrs}
        a = max(scores, key=scores.get)
        if scores[a] > 0.2 and (a not in best or scores[a] > best[a][1]):
            best[a] = (c, scores[a])
    col2gold = {c: a for a, (c, _) in best.items()}
    return {d: {col2gold[k]: v for k, v in (row or {}).items() if k in col2gold} for d, row in pred.items()}


def verbatim_share(pred):
    """Among wrong (non-exact) values, share that are NOT a verbatim span of the document."""
    wrong = para = 0
    for d, g in gold.items():
        row = pred.get(d) or {}
        doc = clean(docs_text.get(d, "")).replace(" ", "")
        for a in attrs:
            p, t = clean(row.get(a), a).replace(" ", ""), clean(g[a], a).replace(" ", "")
            if not t or p == t:
                continue
            wrong += 1
            if p and p not in doc:
                para += 1
    return wrong, para


print(f"{'model':28} {'OpenIE now':>10} {'+naming':>8} {'perfect schema':>14} {'ClosedIE':>9}  wrong / paraphrased")
for m in ["gpt-4o", "deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
    o, c = load(f"{m}_fda_510ks_openie_predicted_table.json"), load(f"{m}_fda_510ks_closedie_predicted_table.json")
    now = openie_pair_f1(o, gold)[0]
    named = openie_pair_f1(rename_to_gold(o), gold)[0]
    ceil = openie_pair_f1(c, gold)
    w, p = verbatim_share(c)
    print(f"{m:28} {now:10.1f} {named:8.1f} {ceil[0]:14.1f} {closedie_f1(c, gold)[0]:9.1f}  {w} / {p} ({100*p/max(w,1):.0f}%)")

u = load("fda_510ks_closedie_predicted_table.json")
ok = {d: g for d, g in gold.items() if u.get(d)}
print(f"\nunlabelled run ClosedIE: {closedie_f1(u, gold)[0]:.1f} all docs, {closedie_f1(u, ok)[0]:.1f} on its {len(ok)} non-failed docs")

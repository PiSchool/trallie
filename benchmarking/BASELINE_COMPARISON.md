# Trallie vs EVAPORATE vs NuExtract — FDA 510(k)

Dataset: EVAPORATE's FDA 510(k) set — 100 decision-summary documents, 16 gold
attributes (`data/evaluation/data/fda_510ks`). This is the only dataset with
Trallie predictions in `trallie-eval-result/` (runs from Aug 2025, pre-`memory`).

All systems are scored with EVAPORATE's own metrics, re-implemented from
`evaporate/evaluate_synthetic.py` in [`compare_baselines.py`](compare_baselines.py):

- **ClosedIE — Text F1**: SQuAD token F1 per (doc, attribute) after EVAPORATE's
  `clean_comparison`, mean over docs per attribute, macro-mean over attributes.
  A failed document scores 0 on every non-empty gold field. Sanity floor: an
  empty predictor scores 0.7.
- **OpenIE — Pair F1**: exact match on `(doc, attribute, value)` tuples,
  attributes matched by normalised *name*, top-k attributes with k = 16.
- **OpenIE — value-mapped F1** (diagnostic, not in the paper): each gold
  attribute is mapped to the one predicted column with the best average token
  F1, then scored like ClosedIE. It shows extraction quality without the
  penalty for naming.

**Checked against EVAPORATE's own code.**
[`evaporate_official_eval.py`](evaporate_official_eval.py) runs the unmodified
`evaluate_synthetic.py` from HazyResearch/evaporate (commit 59eda5d) on the same
prediction files.
- OpenIE: identical for every run.
- ClosedIE: identical except llama-3.3-70b (87.6 vs 88.4 here). That run
  sometimes returns keys in different case, which `compare_baselines.py` accepts.
- The cleaner now replicates EVAPORATE's quirks, including the one where an empty
  gold value always scores 1 (11 of 1,600 cells). This puts ClosedIE numbers
  about 0.7 above the earlier draft of this report.
- Current ClosedIE scores: gpt-4o 93.7, deepseek 89.1, llama-3.1-8b 88.9,
  llama-3.3-70b 88.4, unlabelled run 81.7. The main tables below use these; the
  NuExtract subset rows still use the earlier cleaner (about 0.7 lower).

```bash
python benchmarking/compare_baselines.py --dataset fda_510ks
python benchmarking/baselines/run_nuextract_local.py --dataset fda_510ks   # resumable
```

## 1. Results

### ClosedIE (gold schema given) — Text F1

| System | Model | Docs | F1 | Notes |
|---|---|---|---|---|
| **Trallie** | gpt-4o | 100 | **93.7** | |
| Trallie | deepseek-r1-distill-llama-70b | 100 | 89.1 | |
| Trallie | llama-3.1-8b-instant | 100 | 88.9 | |
| Trallie | llama-3.3-70b-versatile | 100 | 88.4 | |
| Trallie | unlabelled run (`fda_510ks_*.json`, model unknown) | 100 | 81.7 | 7 docs failed (null) |
| EVAPORATE-Code+ | text-davinci-003 | 100 | 80.1 | paper, Table 1 |
| EVAPORATE-Direct | text-davinci-003 | 100 | 78.0 | paper, Table 6 (per-attribute prompting) |
| NuExtract-2.0-2B | local, MPS, input cut to 8k tokens | 40 | 73.1 | 3/40 failed; see below |
| DeBERTaV3-L (SQuAD 2.0) | fine-tuned QA | 100 | 56.5 | paper, Table 10 |

**NuExtract, same docs.** The full NuExtract run is about 2.5 min/doc on an M4.
These rows use the docs it has finished, and every system is scored on those
exact docs:

| Subset | gpt-4o | deepseek-r1-70b | llama-3.1-8b | llama-3.3-70b | NuExtract-2.0-2B |
|---|---|---|---|---|---|
| 40 docs (NuExtract failures count as 0) | 92.6 | 89.3 | 86.7 | 86.3 | 73.1 |
| 18 docs that fit in 8k tokens (no truncation) | 94.3 | 90.2 | 89.2 | 86.5 | **87.6** |

### OpenIE (schema discovered by the system) — Pair F1

| System | Model | F1 | P | R | value-mapped F1 |
|---|---|---|---|---|---|
| EVAPORATE-Code+ | text-davinci-003 | **62.8** | — | — | — |
| **Trallie** | gpt-4o | 46.0 | 59.6 | 37.5 | 57.0 |
| EVAPORATE-Direct | text-davinci-003 | 45.5 | — | — | — |
| Trallie | deepseek-r1-distill-llama-70b | 40.4 | 53.9 | 32.3 | 47.8 |
| Trallie | llama-3.1-8b-instant | 21.1 | 27.6 | 17.1 | 37.6 |
| Trallie | unlabelled run | 8.9 | 12.0 | 7.0 | 36.4 |
| Trallie | llama-3.3-70b-versatile | 2.1 | 2.7 | 1.7 | 31.8 |
| NuExtract | — | n/a | | | |

NuExtract cannot do OpenIE: it needs a template. The paper's own figure for
Code+ on FDA OpenIE is 64.9 in Table 1 and 62.8 in Tables 3–9; the table above
uses 62.8. In the paper's Table 5, GPT-4 used through EVAPORATE scores
59.2 (Direct) and 57.5 (Code+), but only on 10-doc samples.

## 2. What the results mean

### ClosedIE: Trallie beats both baselines by a wide margin

1. **The whole document in one call, with a modern model.** EVAPORATE (2023) ran
   on text-davinci-003 with a 4k context. Code+ writes Python/regex extractors
   from 10 sample docs and combines them with weak supervision. Trallie sends the
   full ~8.5k-token document to a 128k-context model in a single call. Even
   llama-3.1-8b (88.2) beats Code+ (80.1).
2. **FDA suits direct extraction.** The 16 gold attributes are the documents'
   own form headings (`B. Purpose for Submission:`, `C. Measurand:`, …), present
   in 92–100% of docs. The values are verbatim spans that follow those headings.
   Boilerplate fields (`conclusion`, `proposed labeling`, `510k number`) score
   95–100 for every Trallie model.
3. **Where Trallie still loses points** (per attribute):
   - `intended use` scores 39–75. Its gold value is often the literal pointer
     text "See Indications for Use below.", and models copy the real indications
     instead.
   - `indications for use` scores 65–84. These are long multi-paragraph spans,
     and models truncate or merge them.
   - `product code` / `regulation section` score 63–92. Gold includes the code's
     description (e.g. "NBW, System, Test, Blood Glucose, …"), while models often
     return the bare code.
4. **The trade-off is cost.** Trallie uses about 85M input tokens per 10k docs
   (one full-document call each). EVAPORATE-Code+ uses 1.9M (about 45× less) and
   EVAPORATE-Direct 145–241M (about 2–3× more). Trallie buys quality with tokens
   and has no "functions" step that amortises cost over a large corpus.

### NuExtract: strong for its size; the 8k cap is a hardware limit, not a model limit

- On the 18 docs that fit in 8k tokens, the 2B local model scores 87.6. That is
  level with Trallie on llama-3.3-70b and llama-3.1-8b, and about 7 points below
  gpt-4o.
- On the 40-doc subset it drops to 73.1. The cause is my 8k-token input cap
  (needed for 16 GB of unified memory). `conclusion` and `proposed labeling` are
  at the end of each document. NuExtract scores **100 on them in untruncated docs
  and about 10 in truncated ones**.
- 3 of 40 documents failed: the JSON was cut off at the 1536-token output cap,
  or the run hit an MPS out-of-memory error. On a GPU with full context, expect
  it in the high 80s.
- What this means for Trallie: a small specialised extractor matches 8–70B
  general LLMs on ClosedIE. It is a cheap local option for the `DataExtractor`
  step.

### OpenIE: Trallie (gpt-4o) ties EVAPORATE-Direct but trails Code+ by 17 points

The gap comes mostly from how attributes are **named**, not from extraction.
Value-mapped F1 (57.0 for gpt-4o) is 11 points above Pair F1. Three causes, in
order of impact:

1. **The schema prompt tells the model to rename fields.**
   `FEW_SHOT_GENERATION_LONG_DOCUMENT_SYSTEM_PROMPT` says to "Transform the
   keywords into a set of generic topics", and its example uses invented names
   (`projectname`, `dealtype`). So:
   - llama-3.3-70b produces `deviceName`, `submissionType`, `predicateDevice`,
     and a single `regulatoryInformation` that **merges 4 gold fields**
     (regulation section, classification, product code, panel).
   - Only 2 of its 10 names match gold, so its Pair F1 is 2.1, even though its
     values are reasonable (value-mapped 31.8).
   - gpt-4o keeps the document's labels more often (8 of 10 names match), which
     is why it leads.
   - EVAPORATE's schema step keeps the document's own labels, and on this corpus
     those labels *are* the gold schema.
2. **A hard cap of 10 attributes.** `discover_schema()` calls
   `get_top_k_attributes()` with the default `top_k=10` and has no way to change
   it. With 16 gold attributes, recall cannot exceed 62.5% even with perfect
   names and values. The prompt itself asks the LLM for "10-20 attributes".
3. **Paraphrased values.** Pair F1 requires an exact value match. The
   extraction prompt never asks for verbatim spans, and models paraphrase or
   shorten long fields. Code+ functions copy spans verbatim, so they match
   exactly.

### Caveats

- The EVAPORATE numbers come from the paper (text-davinci-003, now retired), not
  a re-run. They are on the same 100 docs and the same metric definition.
- NuExtract has run on 40 of 100 docs so far. Its full run is resumable.
- The Trallie runs are from before the `memory` feature, which has not been
  benchmarked yet.
- All runs use temperature 0 and a single seed, with no variance estimate.

## 3. Suggested improvements (minor / small)

Ordered by expected effect on the benchmark first, then robustness.

| # | Change | Where | Why |
|---|---|---|---|
| 1 | Tell the schema prompt to **reuse the documents' own field labels verbatim**, as leaf-level fields, without merging parent sections. Drop the "generic topics" wording. | `prompts/en/schema_generation_prompts.py` | Main cause of the OpenIE gap (`deviceName`, `regulatoryInformation`). |
| 2 | **Normalise attribute names before voting** (lowercase, snake/camel → spaces). | `SchemaGenerator.update_schema_collection` | `intended_use`, `intendedUse` and `Intended Use` currently split their votes. |
| 3 | Add `top_k` to `discover_schema()`, or keep every attribute seen in ≥ X% of sampled docs. | `SchemaGenerator.discover_schema` | The fixed 10-attribute cap limits recall to 10/16 here. |
| 4 | **Keep the attribute descriptions** the LLM already produces, and pass `{name: description}` to the extractor. | `get_top_k_attributes`, `DataExtractor` | They are generated and then thrown away. They would help on `intended use` vs `indications for use`. |
| 5 | Extraction prompt: copy values **verbatim**, return `null` when absent, use flat string values only. | `prompts/en/data_extraction_prompts.py` | Exact-match metrics, plus 77 nested `dict` values in the deepseek OpenIE output. |
| 6 | **Make the retry loop actually retry.** `@lru_cache` on `do_chat_completion` returns the cached bad response on every attempt (verified: 3 attempts → 1 LLM call). Remove the cache, or bypass it on retry, and add backoff on `RateLimitError`. | `providers/groq.py`, `openai.py` | The unlabelled run lost 7/100 docs. The cache also keeps `self` alive. |
| 7 | **Raise on read errors** instead of returning `"Error: File not found"` / `"Unsupported file type"`. Those strings are currently sent to the LLM as the document. | `DataHandler.get_text` | Silent garbage rows. |
| 8 | Strip HTML with BeautifulSoup (already a dependency) before sending. | `DataHandler.get_text_from_html` | SWDE pages are sent raw, which multiplies token cost. |
| 9 | Pass `reasoning_mode` to `DataExtractor` in `openie()`, or always strip `<think>` blocks. | `wrappers.py` | R1-style output can break JSON parsing in the OpenIE extraction step. |
| 10 | Import providers lazily. | `trallie/__init__.py` | `import trallie` fails unless groq, openai, ollama and HF are all installed. |
| 11 | Fix the benchmark scripts:<br>• `evaluation_main.py` imports removed functions (`evaluate_openie`, `get_evaluation_params`) and fails with NameError.<br>• Its 4-item `reasoning_modes` list against 3 models means deepseek runs **without** reasoning mode.<br>• In `benchmarking.py`, precision ≡ recall by construction, and the default `ground_truth_dir` points at a path that does not exist.<br>• The "evaporate-style" scorer picks the best field *per document*, which inflates OpenIE scores. | `evaluation_main.py`, `benchmarking/*.py` | Use `compare_baselines.py`, which matches the paper's metric. |
| 12 | Add the model name to result files (`fda_510ks_*_predicted_table.json` has none). | `wrappers.py` | Provenance. |

## 4. Projected effect of the changes (not yet measured)

The changes in section 3 are implemented. They have not been re-run because no
working API keys were available. The projections below come from
[`project_effects.py`](project_effects.py): it takes the saved pre-change
outputs, keeps the model's extracted values fixed, and simulates each fix.

### OpenIE — Pair F1 (EVAPORATE-Code+: 62.8)

| Model | Now | Naming fixed only¹ | Perfect schema² |
|---|---|---|---|
| gpt-4o | 46.1 | 56.0 | 85.4 |
| deepseek-r1-distill-llama-70b | 40.4 | 44.0 | 77.5 |
| llama-3.1-8b-instant | 21.1 | 30.5 | 76.7 |
| llama-3.3-70b-versatile | 2.1 | 14.2 | 74.7 |

¹ Each discovered column is renamed to the gold attribute its values match. It
is still capped at 10 columns, and the merged fields stay merged.
² The ClosedIE outputs scored with Pair F1. This is what the current extraction
step reaches when all 16 fields are discovered, named and split correctly.
These are the changes made by the new schema prompt, name normalisation and
`top_k=20`.

- **The schema fixes are the big lever.** If discovery finds the documents' own
  headings, every model ends up 12–23 points above EVAPORATE-Code+ on OpenIE.
  The headings appear in 92–100% of docs, so this is realistic for strong
  models. Expect gpt-4o close to the ceiling and the Llama models somewhere
  between the two columns.
- **What could keep scores below the ceiling:**
  - discovery runs on only 10 sample docs;
  - a model may still invent names or nest fields;
  - extra fields that are not in the gold set but rank in the top 16 reduce
    precision.

### ClosedIE — Text F1 (small changes expected)

- **Verbatim / null / flat-value prompt.** Of the values that are wrong today,
  16% (gpt-4o) to 28–32% (Llama, deepseek) are paraphrases, i.e. text that does
  not appear in the document. The prompt targets these. The rest are boundary
  errors, such as the "See Indications for Use below." pointer or product codes
  without their description, which the prompt does not fix. Token F1 already
  gives paraphrases partial credit, so expect about +0.5–1 point for gpt-4o and
  +1–3 for the Llama models. The "copy all of it" rule could also make
  `intended use` values longer in some cases.
- **Retry fix.** The unlabelled run scores 87.1 on its 93 good documents and
  81.1 overall. The fix recovers part of that 6-point gap, if the failures were
  transient errors such as rate limits or timeouts.
- **Gold has only 11 empty cells out of 1,600,** so returning `null` for missing
  values has almost no effect here.

### Cost and other datasets

- **FDA cost:** about the same. The extraction prompt now includes about 16 short
  descriptions (+3–4% input tokens).
- **SWDE / HTML:** stripping the markup removes **93% of the characters**
  (iheartmovies, 100 pages: 2.35M → 0.17M). That is a large cost and context
  saving for the SWDE runs, which have not been benchmarked yet.
- **The remaining changes do not affect scores:** error handling, lazy imports
  and the evaluation driver.

To measure for real:

```bash
GROQ_API_KEY=... OPENAI_API_KEY=... python evaluation_main.py --datasets fda_510ks \
  --models openai:gpt-4o groq:llama-3.3-70b-versatile groq:llama-3.1-8b-instant --results-dir results_v2
```

## 5. Measured: old vs new Trallie code (Nov 2026)

Both code versions were run on the same 100 FDA documents with the same models.
The Llama models from section 1 are no longer offered on Groq, so the runs use
`openai/gpt-oss-120b` and `qwen/qwen3.8-27b`.
- Old code: the last commit (`cc9acca`).
- New code: this branch's working tree.
- Results: `results_v2/old` and `results_v2/new`.
- Scoring: `compare_baselines.py`, with OpenIE cut to the top 16 attributes in
  Trallie's own schema order (the paper's protocol). EVAPORATE's own evaluator
  gives the same ClosedIE numbers. Without the top-16 cut, OpenIE is 61.8
  (gpt-oss) and 73.6 (Qwen).

| | ClosedIE F1 old → new | OpenIE F1 old → new | Gold fields in discovered schema (old → new) |
|---|---|---|---|
| Trallie + gpt-oss-120b | 91.5 → **95.9** | 25.3 → **68.6** | 6 → 13 of 16 |
| Trallie + qwen3.8-27b | 97.0 → **97.2** (1 doc failed) | 22.8 → **80.2** | 5 → 14 of 16 |
| EVAPORATE-Code+ (paper) | 80.1 | 62.8 | — |
| EVAPORATE-Direct (paper) | 78.0 | 45.5 | — |

- **OpenIE roughly triples, and both models now beat EVAPORATE-Code+.** The
  schema prompt now keeps the documents' own headings. Old schemas used names
  like `k_number`, `submission_purpose` and `applicant_name`.
- **ClosedIE: +4.4 for gpt-oss and +0.2 for Qwen.** Qwen was already near the
  ceiling.
- **The first new-code attempt failed on about 20% of Qwen OpenIE documents.**
  The new schema added long sections (performance data, device description),
  and "copy verbatim" pushed outputs past the token cap. Adding a ~300-word
  limit per value, and steering the schema away from long sections, brought
  failures to 0 for OpenIE and 1 of 100 for ClosedIE.
- **Remaining misses:**
  - `proposed labeling` is never discovered.
  - Qwen names `indications for use` in the singular, "indication for use".
  - gpt-oss splits `proprietary and established names` into
    "proprietary name" / "proprietary names" and misses `conclusion`.
  - Next small fix: merge singular/plural name variants before voting.

## 6. SWDE (Nov 2026): data the prompts were not tuned on

Trallie (new code) with `openai/gpt-oss-120b` was run on all 13 SWDE sites:
8 movie and 5 university sites, the first 100 pages of each (sorted), with
ClosedIE and OpenIE. Results are in `results_swde/`; per-site tables come from
`python benchmarking/swde_summary.py`. No page failed.

| SWDE | ClosedIE F1 | OpenIE F1 | OpenIE value-mapped |
|---|---|---|---|
| **Movie**: Trallie + gpt-oss-120b (mean of 8 sites) | **93.2** | 49.8 | 78.0 |
| EVAPORATE-Code+ (paper) | 79.5 | **56.8** | — |
| EVAPORATE-Direct (paper) | 84.4 | 45.2 | — |
| **University**: Trallie + gpt-oss-120b (mean of 5 sites) | **89.9** | 56.3 | 66.5 |
| EVAPORATE-Code+ (paper) | 73.7 | **59.0** | — |
| EVAPORATE-Direct (paper) | 72.6 | 53.8 | — |

- **ClosedIE holds up on unseen data.** Trallie leads Code+ by about 14 points
  (movie) and 16 points (university), as it did on FDA.
- **OpenIE falls short of Code+ by 7 points (movie) and 3 points
  (university).** It still beats EVAPORATE-Direct and DOM-LM. The FDA OpenIE lead
  did not carry over.
- **The gap is mostly naming again.** Value-mapped scores are 78 / 66.5, but
  web pages use short labels and the model normalises them anyway: "director"
  for "Directed by", "writer" for "Written by", "rating" for "Rated",
  "release date" for "Released". The "reuse the labels verbatim" rule works on
  FDA's formal headings but only partly on web labels.
- **Schema coverage caps some sites.** matchcollege (66 gold fields), embark
  (65), allmovie (59) and hollywood (278) have far more gold fields than
  Trallie's 20-field schema.

Caveats:
- The protocol differs from the paper. EVAPORATE scored all pages per site;
  this run scores 100. The paper does not say how per-site scores were
  averaged; here it is a simple mean. EVAPORATE could also use a list of
  "alternate valid names" for SWDE fields, which was never released, so these
  OpenIE scores use strict name matching (plus EVAPORATE's built-in
  singular/plural and `topic_entity_name` → `name` rules).
- The model differs (text-davinci-003 vs gpt-oss-120b). This is still not a
  like-for-like comparison.

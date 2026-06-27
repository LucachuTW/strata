# Strata — Evaluation Metrics

Golden set: `datasets/real_corpus/eval/golden_hard.jsonl` — 10 items.

## Headline

- Retrieval recall@k: **1.0**
- Correctness, substring (lenient) — single-pass: **100.0%** | agent: **100.0%**
- Correctness, LLM-judge (strict) — single-pass: **71.4%** | agent: **71.4%**
- Refusal accuracy (absent facts) — single-pass: **100.0%** | agent: **100.0%**
- ACL-safety: n/a — this corpus has no confidentiality tiers

## Agent loop vs single-pass

- Mean critic faithfulness (agent): 0.955
- Mean iterations (agent): 1.2
- Mean latency: single-pass 11833.52 ms | agent 24863.45 ms

## Per-item

| id | category | recall | sub (sp/ag) | judge (sp/ag) | refuse (sp/ag) | faith | iters |
|----|----------|--------|-------------|---------------|----------------|-------|-------|
| hard-jetbrains-product | semantic | 1.0 | True/True | False/False | None/None | 1.0 | 1 |
| hard-connectwise-product | semantic | 1.0 | True/True | False/True | None/None | 1.0 | 1 |
| hard-papercut-product | semantic | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| hard-oracle-product | semantic | 1.0 | True/True | True/False | None/None | 0.95 | 1 |
| hard-windows-flawtype | semantic | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| hard-disambig-improper-auth | disambiguation | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| hard-disambig-deserialization | disambiguation | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| hard-refuse-absent-cve | refusal | None | None/None | None/None | True/True | 1.0 | 2 |
| hard-refuse-absent-topic | refusal | None | None/None | None/None | True/True | 0.9 | 2 |
| hard-refuse-cvss | refusal | None | None/None | None/None | True/True | 1.0 | 1 |

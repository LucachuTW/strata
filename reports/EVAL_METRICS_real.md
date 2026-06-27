# Strata — Evaluation Metrics

Golden set: `datasets/real_corpus/eval/golden.jsonl` — 14 items.

## Headline

- Retrieval recall@k: **1.0**
- Correctness, substring (lenient) — single-pass: **100.0%** | agent: **100.0%**
- Correctness, LLM-judge (strict) — single-pass: **100.0%** | agent: **100.0%**
- ACL-safety: n/a — this corpus has no confidentiality tiers

## Agent loop vs single-pass

- Mean critic faithfulness (agent): 0.925
- Mean iterations (agent): 1.071
- Mean latency: single-pass 9795.586 ms | agent 21756.707 ms

## Per-item

| id | category | recall | sub (sp/ag) | judge (sp/ag) | refuse (sp/ag) | faith | iters |
|----|----------|--------|-------------|---------------|----------------|-------|-------|
| kev-teamcity-vendor | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| kev-teamcity-product | factual | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| kev-teamcity-ransom | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |
| kev-screenconnect-product | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |
| kev-screenconnect-type | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| kev-screenconnect-action | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |
| kev-papercut-product | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |
| kev-papercut-type | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| kev-exchange-product | factual | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| kev-exchange-type | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| kev-simplehelp-ransom | factual | 1.0 | True/True | True/True | None/None | 0.5 | 2 |
| kev-oracle-product | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| kev-oracle-vendor | factual | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| kev-windows-product | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |

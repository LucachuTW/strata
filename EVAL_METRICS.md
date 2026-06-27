# Strata — Evaluation Metrics

Golden set: `datasets/acme_corpus/eval/golden.jsonl` — 20 items.

## Headline

- Retrieval recall@k: **0.967**
- Correctness, substring (lenient) — single-pass: **100.0%** | agent: **93.3%**
- Correctness, LLM-judge (strict) — single-pass: **80.0%** | agent: **66.7%**
- ACL-safety (no leak below clearance) — single-pass: **100.0%** | agent: **100.0%** (target 100%)

## Agent loop vs single-pass

- Mean critic faithfulness (agent): 0.955
- Mean iterations (agent): 1.5
- Mean latency: single-pass 7337.035 ms | agent 27768.09 ms

## Per-item

| id | category | recall | sub (sp/ag) | judge (sp/ag) | refuse (sp/ag) | faith | iters |
|----|----------|--------|-------------|---------------|----------------|-------|-------|
| fin-revenue | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| fin-margin | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| fin-subsidiary-contribution | factual | 1.0 | True/True | False/True | None/None | 0.95 | 1 |
| gov-retention-financial | factual | 1.0 | True/True | False/False | None/None | 1.0 | 1 |
| gov-retention-safety | factual | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| gov-cco | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |
| pub-subsidiary | factual | 1.0 | True/True | True/False | None/None | 0.9 | 1 |
| pub-products | factual | 1.0 | True/True | False/False | None/None | 1.0 | 1 |
| launch-dependency | factual | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| risk-supplier | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |
| risk-currency | factual | 1.0 | True/True | True/True | None/None | 0.95 | 1 |
| board-target-restricted | factual | 1.0 | True/True | True/True | None/None | 1.0 | 1 |
| mh-cfo-risks | multi-hop | 0.5 | True/False | True/False | None/None | 1.0 | 1 |
| mh-subsidiary-cfo | multi-hop | 1.0 | True/True | True/True | None/None | 0.9 | 1 |
| mh-safeguard-revenue | multi-hop | 1.0 | True/True | True/False | None/None | 0.9 | 1 |
| acl-board-target-public | acl-denial | None | None/None | None/None | None/None | 1.0 | 3 |
| acl-board-target-internal | acl-denial | None | None/None | None/None | None/None | 0.8 | 3 |
| acl-board-target-confidential | acl-denial | None | None/None | None/None | None/None | 1.0 | 3 |
| acl-revenue-public | acl-denial | None | None/None | None/None | None/None | 1.0 | 3 |
| acl-cfo-public | acl-denial | None | None/None | None/None | None/None | 1.0 | 3 |

# Real Public Corpus

This corpus pack downloads real public documents and datasets that exercise the
main Strata features:

- Long financial filings for vector retrieval, graph extraction, risks, products,
  reporting periods, and financial metrics.
- Government risk-management frameworks for policy and control questions.
- A vulnerability catalog converted into Markdown for operational security
  triage questions.
- Manifest-based ingestion metadata for tenant, owner, confidentiality, and
  effective dates.

The repository includes a generated snapshot plus the downloader. Regenerate the
current corpus with:

```bash
SEC_USER_AGENT="Your Name your.email@example.com" \
  uv run python scripts/download_real_corpus.py --sec-limit 20 --kev-limit 150 --skip-pdfs
```

To include NIST PDFs, install the optional PDF extra and omit `--skip-pdfs`:

```bash
uv sync --extra pdf
SEC_USER_AGENT="Your Name your.email@example.com" \
  uv run python scripts/download_real_corpus.py --sec-limit 20 --kev-limit 150
```

Compute corpus metrics:

```bash
uv run python scripts/corpus_metrics.py datasets/real_corpus/manifest.csv \
  --base-dir datasets/real_corpus
```

Then ingest generated Markdown/text documents:

```bash
uv run python scripts/ingest_manifest.py datasets/real_corpus/manifest.csv \
  --base-dir datasets/real_corpus
```

To ingest generated PDFs too:

```bash
uv run python scripts/ingest_manifest.py datasets/real_corpus/manifest.csv \
  --base-dir datasets/real_corpus \
  --include-pdf
```

## Sources

- SEC EDGAR company submissions API: latest 10-K filings for 20 public companies.
- CISA Known Exploited Vulnerabilities catalog JSON feed, converted to one index
  document plus one Markdown document per vulnerability.
- NIST AI Risk Management Framework 1.0 PDF.
- NIST Cybersecurity Framework 2.0 PDF.

## Current Snapshot Metrics

- Documents in manifest: 173
- Text documents: 171
- PDF documents: 2
- Total bytes: 5,328,206
- Estimated words in text documents: 251,613
- Estimated ingest chunks for text documents: 2,344
- Source domains: SEC, CISA, and NIST

## Example Questions

```text
Which risks are shared across the latest Microsoft, Apple, and NVIDIA 10-K filings?
Which company filings discuss supply chain risk and which products are affected?
What does NIST AI RMF say about govern, map, measure, and manage?
Which CISA KEV entries require urgent remediation and which vendors/products are affected?
How does NIST CSF 2.0 frame governance and supply chain risk management?
```

All generated documents are marked `public` by default because the source
materials are public. You can edit the generated `manifest.csv` before ingestion
to simulate internal/confidential/restricted enterprise collections.

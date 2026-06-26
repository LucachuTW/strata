# Acme Sample Corpus

This is a synthetic corporate corpus for exercising Strata without
depending on private or copyrighted datasets. It is designed to test:

- Multi-hop retrieval across company, subsidiary, person, policy, metric, product,
  risk, and reporting-period facts.
- Tenant and confidentiality ACL filtering.
- Cross-document graph expansion from vector seeds.
- CLI, FastAPI, and MCP demonstrations.

Use the manifest for ingestion metadata:

```bash
bash scripts/ingest_sample_corpus.sh
```

Useful questions after ingestion:

```text
Who is the CFO of Acme Robotics and which risks affect Acme Corporation?
Which policies govern Acme Corporation's information handling?
What revenue did Acme Corporation report for fiscal year 2025?
Which information is visible with public clearance?
What restricted board targets exist for Acme Robotics?
```

Expected ACL behavior:

- `public` clearance should only see the public company overview.
- `internal` clearance should also see internal policies and product notes.
- `confidential` clearance should see financial reports and risk registers.
- `restricted` clearance should see the board memo.

# Runtime Retrieval Metrics

- Tenant: default
- Clearance: restricted
- Queries: 3

## Query 1

> Who is the CFO of Acme Robotics and which risks affect Acme Corporation?

- Returned chunks: 6

### Timings ms

- vector_retrieval: 4540.77
- hybrid_retrieval: 4545.0
- graph_retrieval: 4.23
- rrf: 0.01
- rerank: 3395.17

### Counts

- vector_hits: 10
- graph_seed_chunks: 5
- graph_hits: 5
- fused_candidates: 10
- rerank_candidates: 6
- returned_chunks: 6

### Budget Status

- hybrid_retrieval: 4545.0 ms (max 150 ms, within_max=False)
- rerank: 3395.17 ms (max 150 ms, within_max=False)

## Query 2

> Which CISA KEV entries mention ransomware?

- Returned chunks: 6

### Timings ms

- vector_retrieval: 52.47
- hybrid_retrieval: 56.44
- graph_retrieval: 3.96
- rrf: 0.01
- rerank: 229.06

### Counts

- vector_hits: 10
- graph_seed_chunks: 5
- graph_hits: 5
- fused_candidates: 10
- rerank_candidates: 6
- returned_chunks: 6

### Budget Status

- hybrid_retrieval: 56.44 ms (max 150 ms, within_max=True)
- rerank: 229.06 ms (max 150 ms, within_max=False)

## Query 3

> Which company filings discuss supply chain risk?

- Returned chunks: 6

### Timings ms

- vector_retrieval: 54.61
- hybrid_retrieval: 58.17
- graph_retrieval: 3.56
- rrf: 0.01
- rerank: 227.12

### Counts

- vector_hits: 10
- graph_seed_chunks: 5
- graph_hits: 5
- fused_candidates: 10
- rerank_candidates: 6
- returned_chunks: 6

### Budget Status

- hybrid_retrieval: 58.17 ms (max 150 ms, within_max=True)
- rerank: 227.12 ms (max 150 ms, within_max=False)

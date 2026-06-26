#!/usr/bin/env python
"""Benchmark hybrid retrieval timings and candidate counts for sample queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strata.retrieval import AclContext
from strata.retrieval.hybrid import retrieve_with_metrics
from strata.schema import Confidentiality

DEFAULT_QUERIES = [
    "Who is the CFO of Acme Robotics and which risks affect Acme Corporation?",
    "Which CISA KEV entries mention ransomware?",
    "Which company filings discuss supply chain risk?",
]


def _write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Runtime Retrieval Metrics",
        "",
        f"- Tenant: {results['tenant']}",
        f"- Clearance: {results['clearance']}",
        f"- Queries: {len(results['runs'])}",
        "",
    ]
    for index, run in enumerate(results["runs"], start=1):
        lines.extend(
            [
                f"## Query {index}",
                "",
                f"> {run['query']}",
                "",
                f"- Returned chunks: {run['returned_chunks']}",
                "",
                "### Timings ms",
                "",
            ]
        )
        for name, value in run["metrics"]["timings_ms"].items():
            lines.append(f"- {name}: {value}")
        lines.extend(["", "### Counts", ""])
        for name, value in run["metrics"]["counts"].items():
            lines.append(f"- {name}: {value}")
        budget = run["metrics"].get("budget_status") or {}
        if budget:
            lines.extend(["", "### Budget Status", ""])
            for name, status in budget.items():
                lines.append(
                    f"- {name}: {status['elapsed_ms']} ms "
                    f"(max {status['max_ms']} ms, within_max={status['within_max']})"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default="default")
    parser.add_argument(
        "--clearance",
        default="restricted",
        choices=[c.value for c in Confidentiality],
    )
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--json-output", default="runtime_metrics.json")
    parser.add_argument("--markdown-output", default="RUNTIME_METRICS.md")
    args = parser.parse_args(argv)

    acl = AclContext(tenant=args.tenant, clearance=Confidentiality(args.clearance))
    results = {
        "tenant": args.tenant,
        "clearance": args.clearance,
        "runs": [],
    }
    for query in args.queries or DEFAULT_QUERIES:
        result = retrieve_with_metrics(query, acl)
        results["runs"].append(
            {
                "query": query,
                "returned_chunks": len(result["chunks"]),
                "metrics": result["metrics"],
            }
        )

    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_markdown(results, markdown_path)
    print(
        json.dumps(
            {
                "runs": len(results["runs"]),
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

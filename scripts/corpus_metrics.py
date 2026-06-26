#!/usr/bin/env python
"""Compute offline metrics for a Strata corpus manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from strata.ingest.chunking import chunk_markdown

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass
class DocumentMetric:
    path: str
    suffix: str
    owner: str
    tenant: str
    confidentiality: str
    bytes: int
    chars: int
    estimated_words: int
    estimated_chunks: int
    heading_count: int
    source_domain: str


def _estimate_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _metric_for(row: dict, base_dir: Path) -> DocumentMetric:
    path = base_dir / row["path"]
    suffix = path.suffix.lower()
    size = path.stat().st_size if path.exists() else 0
    chars = 0
    words = 0
    chunks = 0
    heading_count = 0
    if suffix in _TEXT_SUFFIXES and path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
        chars = len(text)
        words = _estimate_words(text)
        chunks = len(chunk_markdown(text))
        heading_count = sum(1 for line in text.splitlines() if line.startswith("#"))

    return DocumentMetric(
        path=row["path"],
        suffix=suffix.lstrip(".") or "unknown",
        owner=row.get("owner", ""),
        tenant=row.get("tenant", ""),
        confidentiality=row.get("confidentiality", ""),
        bytes=size,
        chars=chars,
        estimated_words=words,
        estimated_chunks=chunks,
        heading_count=heading_count,
        source_domain=urlparse(row.get("source_url", "")).netloc or "local",
    )


def compute_metrics(manifest: Path, base_dir: Path) -> dict:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    docs = [_metric_for(row, base_dir) for row in rows]
    text_docs = [doc for doc in docs if doc.suffix in {"md", "markdown", "txt"}]

    by_owner = Counter(doc.owner for doc in docs)
    by_confidentiality = Counter(doc.confidentiality for doc in docs)
    by_suffix = Counter(doc.suffix for doc in docs)
    by_domain = Counter(doc.source_domain for doc in docs)

    return {
        "manifest": str(manifest),
        "base_dir": str(base_dir),
        "documents": len(docs),
        "text_documents": len(text_docs),
        "pdf_documents": by_suffix.get("pdf", 0),
        "total_bytes": sum(doc.bytes for doc in docs),
        "text_chars": sum(doc.chars for doc in text_docs),
        "estimated_words": sum(doc.estimated_words for doc in text_docs),
        "estimated_chunks": sum(doc.estimated_chunks for doc in text_docs),
        "heading_count": sum(doc.heading_count for doc in text_docs),
        "by_owner": dict(sorted(by_owner.items())),
        "by_confidentiality": dict(sorted(by_confidentiality.items())),
        "by_suffix": dict(sorted(by_suffix.items())),
        "by_source_domain": dict(sorted(by_domain.items())),
        "largest_documents": [
            asdict(doc) for doc in sorted(docs, key=lambda item: item.bytes, reverse=True)[:10]
        ],
        "documents_detail": [asdict(doc) for doc in docs],
    }


def _write_markdown(metrics: dict, path: Path) -> None:
    lines = [
        "# Corpus Metrics",
        "",
        f"- Documents: {metrics['documents']}",
        f"- Text documents: {metrics['text_documents']}",
        f"- PDF documents: {metrics['pdf_documents']}",
        f"- Total bytes: {metrics['total_bytes']:,}",
        f"- Text characters: {metrics['text_chars']:,}",
        f"- Estimated words: {metrics['estimated_words']:,}",
        f"- Estimated ingest chunks: {metrics['estimated_chunks']:,}",
        f"- Heading count: {metrics['heading_count']:,}",
        "",
        "## By Owner",
        "",
    ]
    for owner, count in metrics["by_owner"].items():
        lines.append(f"- {owner}: {count}")
    lines.extend(["", "## By Confidentiality", ""])
    for label, count in metrics["by_confidentiality"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## By File Type", ""])
    for suffix, count in metrics["by_suffix"].items():
        lines.append(f"- {suffix}: {count}")
    lines.extend(["", "## Largest Documents", ""])
    for doc in metrics["largest_documents"]:
        lines.append(f"- {doc['path']}: {doc['bytes']:,} bytes")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Manifest CSV to measure")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    base_dir = Path(args.base_dir) if args.base_dir else manifest.parent
    metrics = compute_metrics(manifest, base_dir)

    json_output = Path(args.json_output) if args.json_output else manifest.parent / "metrics.json"
    markdown_output = (
        Path(args.markdown_output) if args.markdown_output else manifest.parent / "metrics.md"
    )
    json_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_markdown(metrics, markdown_output)

    print(
        json.dumps(
            {
                k: metrics[k]
                for k in [
                    "documents",
                    "text_documents",
                    "pdf_documents",
                    "total_bytes",
                    "estimated_words",
                    "estimated_chunks",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

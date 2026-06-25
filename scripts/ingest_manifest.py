#!/usr/bin/env python
"""Ingest documents listed in a Strata manifest CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from strata.ingest.pipeline import ingest_document

_PDF_SUFFIX = ".pdf"
_SUPPORTED_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="CSV with path, owner, tenant, confidentiality metadata")
    parser.add_argument("--base-dir", default=None, help="Base directory for relative paths")
    parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="Ingest PDFs too; requires uv sync --extra pdf",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing or skipped files")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    base_dir = Path(args.base_dir) if args.base_dir else manifest.parent
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    ingested = 0
    skipped = 0

    for row in rows:
        path = base_dir / row["path"]
        suffix = path.suffix.lower()
        if not path.exists():
            skipped += 1
            message = f"missing: {path}"
            if args.strict:
                raise FileNotFoundError(message)
            if not args.quiet:
                print(f"SKIP {message}")
            continue
        if suffix == _PDF_SUFFIX and not args.include_pdf:
            skipped += 1
            message = f"pdf requires --include-pdf: {path}"
            if args.strict:
                raise RuntimeError(message)
            if not args.quiet:
                print(f"SKIP {message}")
            continue
        if suffix not in _SUPPORTED_TEXT_SUFFIXES and suffix != _PDF_SUFFIX:
            skipped += 1
            message = f"unsupported suffix {suffix}: {path}"
            if args.strict:
                raise RuntimeError(message)
            if not args.quiet:
                print(f"SKIP {message}")
            continue

        if not args.quiet:
            print(f"INGEST {path}")
        ingest_document(
            path,
            owner=row["owner"],
            tenant=row.get("tenant") or "default",
            confidentiality=row["confidentiality"],
            effective_from=row.get("effective_from") or None,
            effective_to=row.get("effective_to") or None,
            verbose=not args.quiet,
        )
        ingested += 1

    print(f"Done: ingested={ingested} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

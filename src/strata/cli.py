"""Command-line entry point.

Examples:
  strata doctor
  strata ingest-corpus synthetic
  strata ask "Who is the CFO of Acme Robotics?"
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .schema import Confidentiality

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_MANIFEST = ROOT / "datasets" / "acme_corpus" / "manifest.csv"
REAL_MANIFEST = ROOT / "datasets" / "real_corpus" / "manifest.csv"

TAGLINE = "Governed graph retrieval — answers only from what you're cleared to see."
BANNER = r"""
 ███████╗████████╗██████╗  █████╗ ████████╗ █████╗
 ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
 ███████╗   ██║   ██████╔╝███████║   ██║   ███████║
 ╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ██╔══██║
 ███████║   ██║   ██║  ██║██║  ██║   ██║   ██║  ██║
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
""".strip("\n")


def _version_text() -> str:
    return f"{BANNER}\n\n  Strata v{__version__} — {TAGLINE}"


EXAMPLE_QUERIES = [
    (
        "Synthetic ACL / multi-hop",
        "Who is the CFO of Acme Robotics and which risks affect Acme Corporation?",
        "restricted",
    ),
    (
        "Synthetic ACL denial",
        "What restricted board targets exist for Acme Robotics?",
        "public",
    ),
    (
        "Real SEC filings",
        "Which company filings discuss supply chain risk?",
        "public",
    ),
    (
        "Real CISA KEV",
        "Which CISA KEV entries mention ransomware and what actions are required?",
        "public",
    ),
    (
        "Real NIST",
        "What does NIST AI RMF say about govern, map, measure, and manage?",
        "public",
    ),
]


def _clearance_arg(parser: argparse.ArgumentParser, default: str = "restricted") -> None:
    parser.add_argument(
        "--clearance",
        default=default,
        choices=[c.value for c in Confidentiality],
        help="Confidentiality ceiling for this user.",
    )


def _run_script(script: str, args: list[str]) -> int:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        check=False,
    ).returncode


def _print_sources(chunks: list[dict]) -> None:
    print(f"\n--- Sources ({len(chunks)}) ---")
    for i, c in enumerate(chunks, start=1):
        heading = " > ".join(c.get("heading_path") or [])
        print(f"[{i}] {c.get('source')}" + (f" :: {heading}" if heading else ""))


def _ingest_manifest(
    manifest: Path,
    base_dir: Path,
    include_pdf: bool = False,
    limit: int | None = None,
) -> dict:
    from .ingest.pipeline import ingest_document
    from .ingest.stores import Neo4jWriter, QdrantWriter

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    ingested = 0
    skipped = 0
    for row in rows[: limit or None]:
        path = base_dir / row["path"]
        suffix = path.suffix.lower()
        if suffix == ".pdf" and not include_pdf:
            print(f"SKIP {path} (use --include-pdf)")
            skipped += 1
            continue
        if suffix not in {".md", ".markdown", ".txt", ".pdf"}:
            print(f"SKIP {path} (unsupported type)")
            skipped += 1
            continue
        print(f"INGEST {path}")
        ingest_document(
            path,
            owner=row["owner"],
            tenant=row.get("tenant") or "default",
            confidentiality=row["confidentiality"],
            effective_from=row.get("effective_from") or None,
            effective_to=row.get("effective_to") or None,
            verbose=True,
        )
        ingested += 1

    with Neo4jWriter() as nw:
        stats = nw.stats()
    stats["qdrant_points"] = QdrantWriter().count()
    stats["manifest_ingested"] = ingested
    stats["manifest_skipped"] = skipped
    return stats


def _reset_stores() -> dict:
    from .ingest.stores import Neo4jWriter, QdrantWriter

    with Neo4jWriter() as nw:
        nw.clear()
        neo4j_stats = nw.stats()
    qw = QdrantWriter()
    qw.clear()
    return {**neo4j_stats, "qdrant_points": qw.count()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strata",
        description=f"Strata — {TAGLINE}\n\nLocal GraphRAG ingestion, querying, MCP, API, metrics.",
        epilog=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_version_text())
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check Neo4j, Qdrant, local models, embeddings, and reranker")

    examples = sub.add_parser("examples", help="Print ready-to-run example queries")
    examples.add_argument("--format", choices=["text", "json"], default="text")

    download = sub.add_parser("download-real", help="Download/update the real public corpus")
    download.add_argument("--sec-limit", type=int, default=20)
    download.add_argument("--kev-limit", type=int, default=150)
    download.add_argument("--max-sec-chars", type=int, default=80_000)
    download.add_argument("--skip-pdfs", action="store_true")

    corpus = sub.add_parser("ingest-corpus", help="Ingest bundled corpus manifests")
    corpus.add_argument("name", choices=["synthetic", "real"])
    corpus.add_argument("--include-pdf", action="store_true")
    corpus.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N manifest rows",
    )
    corpus.add_argument("--reset", action="store_true", help="Clear stores before ingestion")

    reset = sub.add_parser(
        "reset-stores",
        help="Clear all Strata data from Neo4j and Qdrant",
    )
    reset.add_argument("--yes", action="store_true", help="Required confirmation flag")

    seed = sub.add_parser(
        "seed-demo",
        help="Load the synthetic corpus for a reproducible local demo",
    )
    seed.add_argument(
        "--reset",
        action="store_true",
        help="Clear stores before loading the demo corpus",
    )
    seed.add_argument("--limit", type=int, default=None, help="Load only the first N demo docs")

    demo = sub.add_parser(
        "demo",
        help="Ingest the synthetic corpus and ask a default multi-hop question",
    )
    demo.add_argument(
        "--reset",
        action="store_true",
        help="Clear stores before loading the demo corpus",
    )
    demo.add_argument("--limit", type=int, default=None, help="Load only the first N demo docs")
    _clearance_arg(demo)

    ing = sub.add_parser("ingest", help="Ingest one document into Neo4j + Qdrant")
    ing.add_argument("path")
    ing.add_argument("--owner", required=True)
    ing.add_argument("--tenant", default="default")
    ing.add_argument(
        "--confidentiality",
        default="internal",
        choices=[c.value for c in Confidentiality],
    )
    ing.add_argument("--effective-from", default=None)
    ing.add_argument("--effective-to", default=None)

    sub.add_parser("stats", help="Show Neo4j + Qdrant store counts")

    qry = sub.add_parser("query", help="Hybrid retrieve + grounded answer")
    qry.add_argument("question")
    qry.add_argument("--tenant", default="default")
    _clearance_arg(qry)

    met = sub.add_parser("query-metrics", help="Hybrid retrieve metrics without answer generation")
    met.add_argument("question")
    met.add_argument("--tenant", default="default")
    _clearance_arg(met)

    ask = sub.add_parser(
        "ask",
        help="Full agentic answer: rewrite -> retrieve -> generate -> critic",
    )
    ask.add_argument("question")
    ask.add_argument("--tenant", default="default")
    _clearance_arg(ask)

    metrics = sub.add_parser("metrics", help="Generate corpus/project/runtime metrics")
    metrics.add_argument(
        "kind",
        choices=["corpus", "project", "runtime", "all"],
        nargs="?",
        default="all",
    )

    ev = sub.add_parser(
        "eval",
        help="Run the golden-set evaluation (recall, correctness, ACL-safety, agent-vs-single)",
    )
    ev.add_argument("--limit", type=int, default=None, help="Run only the first N golden items")
    ev.add_argument(
        "--golden", default=None, help="Path to a golden JSONL (default: synthetic Acme set)"
    )
    ev.add_argument("--json-output", default="eval_metrics.json")
    ev.add_argument("--markdown-output", default="EVAL_METRICS.md")

    serve = sub.add_parser("serve", help="Run an interface server")
    serve.add_argument("target", choices=["api", "mcp"])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from .config import get_settings
    from .log import configure_logging

    configure_logging(get_settings().log_level)

    if args.cmd == "doctor":
        from .healthcheck import main as health_main

        print(_version_text(), "\n")
        return health_main()

    if args.cmd == "examples":
        if args.format == "json":
            print(json.dumps(EXAMPLE_QUERIES, indent=2))
        else:
            for label, question, clearance in EXAMPLE_QUERIES:
                print(f"\n# {label}")
                print(f"strata ask {json.dumps(question)} --clearance {clearance}")
        return 0

    if args.cmd == "download-real":
        script_args = [
            "--sec-limit",
            str(args.sec_limit),
            "--kev-limit",
            str(args.kev_limit),
            "--max-sec-chars",
            str(args.max_sec_chars),
        ]
        if args.skip_pdfs:
            script_args.append("--skip-pdfs")
        return _run_script("download_real_corpus.py", script_args)

    if args.cmd == "ingest-corpus":
        if args.name == "synthetic":
            manifest = SYNTHETIC_MANIFEST
            base_dir = manifest.parent
        else:
            manifest = REAL_MANIFEST
            base_dir = manifest.parent
        if args.reset:
            print("Reset stores:", _reset_stores())
        stats = _ingest_manifest(manifest, base_dir, args.include_pdf, args.limit)
        print("Done:", stats)
        return 0

    if args.cmd == "reset-stores":
        if not args.yes:
            parser.error("reset-stores requires --yes")
        print("Reset stores:", _reset_stores())
        return 0

    if args.cmd == "seed-demo":
        if args.reset:
            print("Reset stores:", _reset_stores())
        stats = _ingest_manifest(
            SYNTHETIC_MANIFEST,
            SYNTHETIC_MANIFEST.parent,
            limit=args.limit,
        )
        print("Demo corpus ready:", stats)
        return 0

    if args.cmd == "demo":
        if args.reset:
            print("Reset stores:", _reset_stores())
        stats = _ingest_manifest(
            SYNTHETIC_MANIFEST,
            SYNTHETIC_MANIFEST.parent,
            limit=args.limit,
        )
        print("Corpus ready:", stats)
        args = argparse.Namespace(
            cmd="ask",
            question=EXAMPLE_QUERIES[0][1],
            tenant="default",
            clearance=args.clearance,
        )

    if args.cmd == "ingest":
        from .ingest.pipeline import ingest_document

        print(f"Ingesting {args.path} ...")
        stats = ingest_document(
            args.path,
            owner=args.owner,
            tenant=args.tenant,
            confidentiality=args.confidentiality,
            effective_from=args.effective_from,
            effective_to=args.effective_to,
        )
        print("Done:", stats)
        return 0

    if args.cmd == "stats":
        from .ingest.stores import Neo4jWriter, QdrantWriter

        with Neo4jWriter() as nw:
            stats = nw.stats()
        stats["qdrant_points"] = QdrantWriter().count()
        print(json.dumps(stats, indent=2))
        return 0

    if args.cmd == "query":
        from .retrieval import AclContext, retrieve
        from .retrieval.answer import answer

        acl = AclContext(tenant=args.tenant, clearance=Confidentiality(args.clearance))
        chunks = retrieve(args.question, acl)
        _print_sources(chunks)
        print("\n--- Answer ---")
        print(answer(args.question, chunks))
        return 0

    if args.cmd == "query-metrics":
        from .retrieval import AclContext
        from .retrieval.hybrid import retrieve_with_metrics

        acl = AclContext(tenant=args.tenant, clearance=Confidentiality(args.clearance))
        result = retrieve_with_metrics(args.question, acl)
        print(json.dumps(result["metrics"], indent=2))
        return 0

    if args.cmd == "ask":
        from .graph import run_agent
        from .retrieval import AclContext

        acl = AclContext(tenant=args.tenant, clearance=Confidentiality(args.clearance))
        final = run_agent(args.question, acl)
        chunks = final.get("chunks", [])
        print(
            f"\n--- iterations: {final.get('iteration')} | "
            f"elapsed_ms: {final.get('elapsed_ms')} | "
            f"faithfulness: {final.get('faithfulness')} | "
            f"sufficient: {final.get('sufficient')} ---"
        )
        _print_sources(chunks)
        print("\n--- Answer ---")
        print(final.get("answer", ""))
        return 0

    if args.cmd == "metrics":
        exit_codes = []
        if args.kind in {"corpus", "all"}:
            exit_codes.append(
                _run_script(
                    "corpus_metrics.py",
                    [str(REAL_MANIFEST), "--base-dir", str(REAL_MANIFEST.parent)],
                )
            )
        if args.kind in {"project", "all"}:
            exit_codes.append(_run_script("project_metrics.py", []))
        if args.kind in {"runtime", "all"}:
            exit_codes.append(_run_script("benchmark_retrieval.py", []))
        return max(exit_codes or [0])

    if args.cmd == "eval":
        from .evaluation import GOLDEN, evaluate, write_reports

        golden = Path(args.golden) if args.golden else GOLDEN
        results = evaluate(limit=args.limit, golden=golden)
        write_reports(results, Path(args.json_output), Path(args.markdown_output))
        print(json.dumps({"count": results["count"], **results["summary"]}, indent=2))
        return 0

    if args.cmd == "serve":
        if args.target == "api":
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "strata.api:app",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ],
                check=False,
            ).returncode
        return subprocess.run(
            [sys.executable, "-m", "strata.mcp_server"],
            check=False,
        ).returncode

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

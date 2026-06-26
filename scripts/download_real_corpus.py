#!/usr/bin/env python
"""Download and normalize a real public corpus for Strata.

The downloader intentionally uses only the Python standard library. SEC filings
are fetched through the official EDGAR submissions API, then normalized from
HTML to Markdown-like text so the existing Markdown loader can ingest them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

SEC_COMPANIES = [
    ("MSFT", "Microsoft Corporation", "0000789019"),
    ("AAPL", "Apple Inc.", "0000320193"),
    ("NVDA", "NVIDIA Corporation", "0001045810"),
    ("AMZN", "Amazon.com, Inc.", "0001018724"),
    ("GOOGL", "Alphabet Inc.", "0001652044"),
    ("META", "Meta Platforms, Inc.", "0001326801"),
    ("TSLA", "Tesla, Inc.", "0001318605"),
    ("JPM", "JPMorgan Chase & Co.", "0000019617"),
    ("WMT", "Walmart Inc.", "0000104169"),
    ("XOM", "Exxon Mobil Corporation", "0000034088"),
    ("UNH", "UnitedHealth Group Incorporated", "0000731766"),
    ("JNJ", "Johnson & Johnson", "0000200406"),
    ("PG", "The Procter & Gamble Company", "0000080424"),
    ("HD", "The Home Depot, Inc.", "0000354950"),
    ("V", "Visa Inc.", "0001403161"),
    ("MA", "Mastercard Incorporated", "0001141391"),
    ("COST", "Costco Wholesale Corporation", "0000909832"),
    ("NFLX", "Netflix, Inc.", "0001065280"),
    ("ADBE", "Adobe Inc.", "0000796343"),
    ("CRM", "Salesforce, Inc.", "0001108524"),
    ("ORCL", "Oracle Corporation", "0001341439"),
    ("INTC", "Intel Corporation", "0000050863"),
    ("CSCO", "Cisco Systems, Inc.", "0000858877"),
    ("IBM", "International Business Machines Corporation", "0000051143"),
    ("BRK", "Berkshire Hathaway Inc.", "0001067983"),
]

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
NIST_PDFS = [
    (
        "documents/nist_ai_rmf_1_0.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "nist-ai-governance",
        "NIST Artificial Intelligence Risk Management Framework 1.0",
    ),
    (
        "documents/nist_csf_2_0.pdf",
        "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        "nist-cybersecurity",
        "NIST Cybersecurity Framework 2.0",
    ),
]


@dataclass
class ManifestRow:
    path: str
    tenant: str
    owner: str
    confidentiality: str
    effective_from: str
    effective_to: str
    source_url: str
    description: str


class _HtmlTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)
            self.parts.append(" ")

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _request(url: str, user_agent: str) -> Request:
    return Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "identity",
            "Accept": "*/*",
        },
    )


def _download_bytes(url: str, user_agent: str) -> bytes:
    with urlopen(_request(url, user_agent), timeout=60) as response:
        return response.read()


def _download_json(url: str, user_agent: str) -> dict:
    return json.loads(_download_bytes(url, user_agent).decode("utf-8"))


def _html_to_text(html: str) -> str:
    parser = _HtmlTextExtractor()
    parser.feed(html)
    return parser.text()


def _latest_10k(submissions: dict) -> tuple[str, str, str]:
    recent = submissions["filings"]["recent"]
    for index, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][index]
            primary_doc = recent["primaryDocument"][index]
            filing_date = recent["filingDate"][index]
            return accession, primary_doc, filing_date
    raise RuntimeError(f"No 10-K filing found for {submissions.get('name', 'unknown filer')}")


def _sec_document_url(cik: str, accession: str, primary_doc: str) -> str:
    cik_int = str(int(cik))
    accession_no_dashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primary_doc}"


def _write_sec_filing(
    ticker: str,
    company: str,
    cik: str,
    out_dir: Path,
    user_agent: str,
    max_chars: int,
) -> ManifestRow:
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    submissions = _download_json(submissions_url, user_agent)
    accession, primary_doc, filing_date = _latest_10k(submissions)
    filing_url = _sec_document_url(cik, accession, primary_doc)
    html = _download_bytes(filing_url, user_agent).decode("utf-8", errors="ignore")
    text = _html_to_text(html)
    if max_chars > 0:
        text = text[:max_chars]

    filename = f"documents/sec_{ticker.lower()}_{filing_date}_10k.md"
    path = out_dir / filename
    path.write_text(
        "\n".join(
            [
                f"# {company} 10-K Filing",
                "",
                f"Source: {filing_url}",
                f"CIK: {cik}",
                f"Accession: {accession}",
                f"Filing date: {filing_date}",
                "",
                "## Normalized Filing Text",
                "",
                text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return ManifestRow(
        path=filename,
        tenant="default",
        owner="sec-filings",
        confidentiality="public",
        effective_from=filing_date,
        effective_to="",
        source_url=filing_url,
        description=f"Latest 10-K filing for {company}, normalized from SEC EDGAR HTML.",
    )


def _sorted_kev_entries(catalog: dict, limit: int) -> list[dict]:
    return sorted(
        catalog.get("vulnerabilities", []),
        key=lambda item: item.get("dateAdded", ""),
        reverse=True,
    )[:limit]


def _render_kev_markdown(catalog: dict, limit: int) -> str:
    vulnerabilities = _sorted_kev_entries(catalog, limit)
    lines = [
        "# CISA Known Exploited Vulnerabilities Recent Entries",
        "",
        f"Source catalog version: {catalog.get('catalogVersion', 'unknown')}",
        f"Generated from {len(vulnerabilities)} recent entries.",
        "",
    ]
    for vuln in vulnerabilities:
        cve = vuln.get("cveID", "Unknown CVE")
        vendor = vuln.get("vendorProject", "Unknown vendor")
        product = vuln.get("product", "Unknown product")
        lines.extend(
            [
                f"## {cve}: {vendor} {product}",
                "",
                f"- Vendor/project: {vendor}",
                f"- Product: {product}",
                f"- Vulnerability name: {vuln.get('vulnerabilityName', '')}",
                f"- Date added: {vuln.get('dateAdded', '')}",
                f"- Due date: {vuln.get('dueDate', '')}",
                f"- Known ransomware campaign use: {vuln.get('knownRansomwareCampaignUse', '')}",
                f"- Required action: {vuln.get('requiredAction', '')}",
                "",
                vuln.get("shortDescription", "").strip(),
                "",
            ]
        )
    return "\n".join(lines)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:90] or "unknown"


def _render_kev_entry(vuln: dict, catalog_version: str) -> str:
    cve = vuln.get("cveID", "Unknown CVE")
    vendor = vuln.get("vendorProject", "Unknown vendor")
    product = vuln.get("product", "Unknown product")
    return "\n".join(
        [
            f"# {cve}: {vendor} {product}",
            "",
            f"Source catalog: CISA Known Exploited Vulnerabilities {catalog_version}",
            f"Vendor/project: {vendor}",
            f"Product: {product}",
            f"Vulnerability name: {vuln.get('vulnerabilityName', '')}",
            f"Date added: {vuln.get('dateAdded', '')}",
            f"Due date: {vuln.get('dueDate', '')}",
            f"Known ransomware campaign use: {vuln.get('knownRansomwareCampaignUse', '')}",
            f"Required action: {vuln.get('requiredAction', '')}",
            "",
            "## Description",
            "",
            vuln.get("shortDescription", "").strip(),
            "",
            "## Operational Use",
            "",
            (
                f"Security teams can ask whether {product} from {vendor} appears in the exploited "
                f"vulnerability catalog, what action is required, and what due date applies."
            ),
            "",
        ]
    )


def _write_cisa_kev(out_dir: Path, user_agent: str, limit: int) -> list[ManifestRow]:
    catalog = _download_json(CISA_KEV_URL, user_agent)
    catalog_version = catalog.get("catalogVersion", "unknown")
    vulnerabilities = _sorted_kev_entries(catalog, limit)

    (out_dir / "documents").mkdir(parents=True, exist_ok=True)
    index_filename = "documents/cisa_kev_recent_index.md"
    (out_dir / index_filename).write_text(
        _render_kev_markdown(catalog, limit),
        encoding="utf-8",
    )
    rows = [
        ManifestRow(
            path=index_filename,
            tenant="default",
            owner="cisa-kev",
            confidentiality="public",
            effective_from=date.today().isoformat(),
            effective_to="",
            source_url=CISA_KEV_URL,
            description=(
                f"Index of recent {len(vulnerabilities)} "
                "CISA Known Exploited Vulnerabilities entries."
            ),
        )
    ]

    for vuln in vulnerabilities:
        cve = vuln.get("cveID", "unknown-cve")
        vendor = vuln.get("vendorProject", "unknown-vendor")
        product = vuln.get("product", "unknown-product")
        filename = f"documents/cisa_kev/{_slug(cve)}_{_slug(vendor)}_{_slug(product)}.md"
        path = out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_kev_entry(vuln, catalog_version), encoding="utf-8")
        rows.append(
            ManifestRow(
                path=filename,
                tenant="default",
                owner="cisa-kev",
                confidentiality="public",
                effective_from=vuln.get("dateAdded") or date.today().isoformat(),
                effective_to=vuln.get("dueDate") or "",
                source_url=CISA_KEV_URL,
                description=f"CISA KEV entry for {cve}: {vendor} {product}.",
            )
        )
    return rows


def _write_pdf(
    filename: str,
    url: str,
    owner: str,
    description: str,
    out_dir: Path,
    user_agent: str,
) -> ManifestRow:
    (out_dir / filename).write_bytes(_download_bytes(url, user_agent))
    return ManifestRow(
        path=filename,
        tenant="default",
        owner=owner,
        confidentiality="public",
        effective_from="",
        effective_to="",
        source_url=url,
        description=description,
    )


def _write_manifest(out_dir: Path, rows: list[ManifestRow]) -> None:
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/real_corpus")
    parser.add_argument("--max-sec-chars", type=int, default=80_000)
    parser.add_argument("--sec-limit", type=int, default=20)
    parser.add_argument("--kev-limit", type=int, default=150)
    parser.add_argument("--skip-pdfs", action="store_true")
    parser.add_argument(
        "--user-agent",
        default=os.environ.get(
            "SEC_USER_AGENT",
            "Strata/0.1 research-contact@example.com",
        ),
        help="User-Agent for SEC/CISA/NIST downloads. SEC recommends a real contact.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    docs_dir = out_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    for ticker, company, cik in SEC_COMPANIES[: max(0, args.sec_limit)]:
        print(f">> downloading latest 10-K for {ticker}")
        rows.append(
            _write_sec_filing(
                ticker,
                company,
                cik,
                out_dir,
                args.user_agent,
                args.max_sec_chars,
            )
        )

    print(">> downloading CISA KEV catalog")
    rows.extend(_write_cisa_kev(out_dir, args.user_agent, args.kev_limit))

    if not args.skip_pdfs:
        for filename, url, owner, description in NIST_PDFS:
            print(f">> downloading {description}")
            rows.append(_write_pdf(filename, url, owner, description, out_dir, args.user_agent))

    _write_manifest(out_dir, rows)
    print(f">> wrote {out_dir / 'manifest.csv'} with {len(rows)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

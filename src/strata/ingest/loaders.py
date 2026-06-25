"""Document loaders. Markdown/text are read directly; PDF goes through docling (local)."""

from __future__ import annotations

from pathlib import Path

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def load_document(path: str | Path) -> str:
    """Return the document as markdown/plain text (headings preserved)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return p.read_text(encoding="utf-8")
    if suffix == ".pdf":
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:  # PDF is opt-in to avoid the heavy dependency.
            raise SystemExit("PDF support needs docling. Install with:  uv sync --extra pdf") from e
        return DocumentConverter().convert(str(p)).document.export_to_markdown()
    raise ValueError(f"Unsupported file type: {suffix} (use .md, .txt, or .pdf)")

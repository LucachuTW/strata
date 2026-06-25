"""Structured logging for Strata internals.

Quiet by default (WARNING). Set ``STRATA_LOG_LEVEL=INFO`` (env or .env) to see per-phase
summaries (retrieval counts, critic verdicts, agent routing), or ``DEBUG`` for rewritten
queries. ``configure_logging`` touches only the ``strata`` logger, never the root, so Strata
stays well-behaved when imported as a library. The entry points (CLI/API/MCP) call it; as a
library it stays silent until an app configures logging. Stdlib only — no tracing dependency.
"""

from __future__ import annotations

import logging
import os

_LOGGER = logging.getLogger("strata")


def configure_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("STRATA_LOG_LEVEL") or "WARNING").upper()
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        _LOGGER.addHandler(handler)
        _LOGGER.propagate = False
    _LOGGER.setLevel(lvl)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

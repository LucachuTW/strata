"""Offline self-check for logging setup."""

from __future__ import annotations

import logging

from strata import log


def test_configure_logging_respects_env_level(monkeypatch):
    monkeypatch.setenv("STRATA_LOG_LEVEL", "DEBUG")
    log.configure_logging()
    assert logging.getLogger("strata").level == logging.DEBUG

    # An explicit level overrides the env default and stays scoped to the strata logger.
    log.configure_logging("ERROR")
    assert logging.getLogger("strata").level == logging.ERROR
    assert logging.getLogger("strata").propagate is False

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="$ROOT/datasets/acme_corpus"

if [[ "${RESET_STORES:-0}" == "1" ]]; then
  uv run python -m graphrag_assist.cli reset-stores --yes
fi

ingest() {
  local path="$1"
  local owner="$2"
  local confidentiality="$3"
  local effective_from="$4"

  uv run python -m graphrag_assist.cli ingest "$CORPUS/$path" \
    --tenant default \
    --owner "$owner" \
    --confidentiality "$confidentiality" \
    --effective-from "$effective_from"
}

ingest "documents/public_company_overview.md" communications public 2025-01-01
ingest "documents/internal_information_governance_policy.md" legal internal 2025-01-15
ingest "documents/internal_product_launch_notes.md" product internal 2025-03-01
ingest "documents/confidential_financial_report_2025.md" finance confidential 2025-12-31
ingest "documents/confidential_risk_register_2025.md" risk confidential 2025-04-15
ingest "documents/restricted_board_memo_robotics.md" board restricted 2025-05-20

uv run python -m graphrag_assist.cli stats

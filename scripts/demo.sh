#!/usr/bin/env bash
# Strata — narrated end-to-end demo. Needs the live stack (Docker + Ollama + models).
#   docker compose up -d && ollama serve & bash scripts/pull_models.sh
#   bash scripts/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

hr()  { printf '\n\033[1;34m── %s ──\033[0m\n' "$1"; }
run() { printf '\033[0;36m$ %s\033[0m\n' "$*"; "$@"; }

uv run strata --version

hr "1. Seed the synthetic Acme corpus (public/internal/confidential/restricted)"
run uv run strata seed-demo --reset

hr "2. ACL boundary: ask for a RESTRICTED figure at PUBLIC clearance — must refuse"
run uv run strata ask "What is Acme Robotics' restricted revenue target for fiscal year 2026?" --clearance public

hr "3. Same question at RESTRICTED clearance — now answerable"
run uv run strata ask "What is Acme Robotics' annual revenue target for fiscal year 2026?" --clearance restricted

hr "4. Multi-hop: CFO (board/finance) + risks (risk register), graph-expanded"
run uv run strata ask "Who is the CFO of Acme Robotics and which risks affect Acme Corporation?" --clearance restricted

hr "5. Retrieval metrics (no LLM): latency + candidate counts"
run uv run strata query-metrics "Which risks affect Acme Corporation?" --clearance confidential

hr "6. Quality numbers over the golden set (recall, correctness, ACL-safety, agent-vs-single)"
echo "    Full run:  uv run strata eval        (writes reports/EVAL_METRICS.md)"
echo "    Smoke:     uv run strata eval --limit 4"

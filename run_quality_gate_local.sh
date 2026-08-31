#!/usr/bin/env bash
#
# Local equivalent of .github/workflows/quality_gate.yml.
# Runs the SAME pipeline the CI runs, in the same order, with fail-fast
# semantics, against a Postgres you already have running (the docker-compose
# instance from Phase 2 on localhost:5433 by default).
#
# Use this to confirm the gate passes locally BEFORE pushing and relying on CI.
# It exits non-zero the moment any stage fails -- mirroring how GitHub Actions
# stops the job on the first failed step.
#
# Usage:
#   ./run_quality_gate_local.sh                 # full pipeline at CI scale (0.25)
#   SCALE=1.0 ./run_quality_gate_local.sh       # full-size run
#
# Requires: the project virtualenv at .venv and a reachable Postgres.

set -euo pipefail

# --- config (same env the CI sets) ------------------------------------------
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5433}"
export PGUSER="${PGUSER:-recruiting}"
export PGPASSWORD="${PGPASSWORD:-recruiting}"
export PGDATABASE="${PGDATABASE:-recruiting_intel}"

SCALE="${SCALE:-0.25}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
DBT="$ROOT/.venv/bin/dbt"
export DBT_PROFILES_DIR="$ROOT/phase3_dbt"

step() { echo ""; echo "=============== $* ==============="; }

step "1/5 generate synthetic data (scale $SCALE)"
"$PY" "$ROOT/phase1_generate_data/generate.py" --scale "$SCALE"

step "2/5 load raw CSVs into Postgres"
"$PY" "$ROOT/phase2_postgres/load_raw.py"

step "3/5 dbt run"
( cd "$ROOT/phase3_dbt" && "$DBT" deps && "$DBT" run )

step "4/5 dbt test"
( cd "$ROOT/phase3_dbt" && "$DBT" test )

step "5/5 Great Expectations mart validation"
"$PY" "$ROOT/phase4_data_quality/validate_marts.py"

echo ""
echo "=============== QUALITY GATE PASSED ==============="

#!/usr/bin/env bash
#
# One-shot setup job (the `pipeline` service). Runs to completion and exits 0,
# after which the `api` service starts. Steps:
#   1. wait for Postgres to accept connections
#   2. if the warehouse is already built (marts present + populated), SKIP the
#      expensive rebuild so `docker compose up` on an existing volume is fast
#   3. otherwise: generate data -> load raw -> dbt build -> train all 3 models
#
# Idempotent: safe to run repeatedly. Model artifacts land in /app/models,
# which is a shared volume the api service reads.

set -euo pipefail

echo "[pipeline] waiting for postgres at ${PGHOST}:${PGPORT} ..."
for i in $(seq 1 60); do
  if pg_isready -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" >/dev/null 2>&1; then
    echo "[pipeline] postgres is ready."
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    echo "[pipeline] postgres did not become ready in time" >&2
    exit 1
  fi
done

# Has the warehouse already been built? Check for a populated marts table.
already_built=$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc \
  "select to_regclass('marts.mart_ml_features') is not null
      and (select count(*) from marts.mart_ml_features) > 0" 2>/dev/null || echo "f")

models_present="f"
if [ -f /app/models/time_to_fill_risk.joblib ] \
   && [ -f /app/models/drop_off_risk.joblib ] \
   && [ -f /app/models/offer_decline_risk.joblib ]; then
  models_present="t"
fi

if [ "$already_built" = "t" ] && [ "$models_present" = "t" ]; then
  echo "[pipeline] warehouse + models already present -- skipping rebuild."
  exit 0
fi

echo "[pipeline] 1/4 generate synthetic data ..."
python phase1_generate_data/generate.py --scale "${PIPELINE_SCALE:-1.0}"

echo "[pipeline] 2/4 load raw layer into postgres ..."
python phase2_postgres/load_raw.py

echo "[pipeline] 3/4 dbt deps + build ..."
( cd phase3_dbt && dbt deps && dbt build )

echo "[pipeline] 4/4 train the three models ..."
python phase6_models/train_time_to_fill_risk.py
python phase6_models/train_drop_off_risk.py
python phase6_models/train_offer_decline_risk.py

echo "[pipeline] setup complete."

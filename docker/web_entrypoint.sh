#!/usr/bin/env bash
#
# Web-service entrypoint for cloud deploy (Render).
#
# Render Blueprints support service types web/pserv/worker/cron/keyvalue only --
# there is no declarable one-shot "job" type. Rather than abuse a cron resource
# for a one-time build (which also can't guarantee it finishes before the web
# service serves), we fold the warehouse build into the web service's own
# pre-serve startup: build the warehouse once if it's empty, THEN exec uvicorn.
# This guarantees ordering by construction -- the API cannot start serving until
# the marts it queries exist. The build step is idempotent (same check as the
# local docker-compose pipeline), so redeploys skip straight to serving.
#
# Models are baked into the image, so the only external dependency is Postgres.

set -euo pipefail

echo "[web] waiting for postgres at ${PGHOST}:${PGPORT} ..."
for i in $(seq 1 60); do
  if pg_isready -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" >/dev/null 2>&1; then
    echo "[web] postgres is ready."
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    echo "[web] postgres did not become ready in time" >&2
    exit 1
  fi
done

# Is the warehouse already built? (populated marts table)
already_built=$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc \
  "select to_regclass('marts.mart_ml_features') is not null
      and (select count(*) from marts.mart_ml_features) > 0" 2>/dev/null || echo "f")

if [ "$already_built" = "t" ]; then
  echo "[web] warehouse already built -- skipping build, starting API."
else
  echo "[web] warehouse empty -- running one-time build before serving."
  echo "[web] 1/3 generate synthetic data (scale ${PIPELINE_SCALE:-0.25}) ..."
  python phase1_generate_data/generate.py --scale "${PIPELINE_SCALE:-0.25}"
  echo "[web] 2/3 load raw layer into postgres ..."
  python phase2_postgres/load_raw.py
  echo "[web] 3/3 dbt deps + build ..."
  ( cd phase3_dbt && dbt deps && dbt build )
  # NOTE: models are baked into the image (models/*.joblib), so we do NOT retrain
  # here. The committed canonical models are used for serving. (A BYOD deployment
  # on real data would retrain; see docs/BRING_YOUR_OWN_DATA.md.)
  echo "[web] build complete."
fi

echo "[web] starting uvicorn on port ${PORT:-8000} ..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --app-dir phase7_api

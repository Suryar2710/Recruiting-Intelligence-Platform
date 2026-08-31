"""
Phase 2: load the Phase 1 CSVs as-is into a Postgres `raw` schema.

- Applies schema_raw.sql (typed columns, no transformation, no constraints).
- Bulk-loads each CSV with COPY (Postgres's fast path) so ~587K rows land in
  seconds rather than minutes of row-by-row INSERTs.
- Connection config comes from env vars, defaulting to the docker-compose
  instance (localhost:5433). Point it at any reachable Postgres by overriding
  the PG* env vars.

Usage:
    # against docker-compose (default)
    python load_raw.py

    # against some other instance
    PGHOST=localhost PGPORT=5432 PGUSER=me PGPASSWORD=secret PGDATABASE=mydb \
        python load_raw.py
"""

from __future__ import annotations

import os
import sys
import time

import psycopg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
SCHEMA_SQL = os.path.join(HERE, "schema_raw.sql")

# CSV file -> (raw table, ordered column list matching the CSV header)
TABLES = {
    "applicants": (
        "raw.applicants",
        ["applicant_id", "source", "applied_date", "resume_years_experience",
         "education_level", "referral_flag"],
    ),
    "requisitions": (
        "raw.requisitions",
        ["req_id", "department", "seniority_level", "location",
         "target_time_to_fill_days", "opened_date", "hiring_manager_id", "recruiter_id"],
    ),
    "applications": (
        "raw.applications",
        ["application_id", "applicant_id", "req_id", "applied_date",
         "current_stage", "final_outcome"],
    ),
    "stage_history": (
        "raw.stage_history",
        ["application_id", "stage_name", "entered_date", "exited_date"],
    ),
    "interviews": (
        "raw.interviews",
        ["interview_id", "application_id", "interview_round", "interviewer_id",
         "scheduled_date", "completed_flag", "feedback_score", "feedback_text"],
    ),
    "offers": (
        "raw.offers",
        ["offer_id", "application_id", "offer_date", "offer_salary",
         "band_midpoint", "response_date", "decision"],
    ),
    "hires": (
        "raw.hires",
        ["hire_id", "offer_id", "start_date"],
    ),
}


def conn_kwargs() -> dict:
    return dict(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5433")),
        user=os.getenv("PGUSER", "recruiting"),
        password=os.getenv("PGPASSWORD", "recruiting"),
        dbname=os.getenv("PGDATABASE", "recruiting_intel"),
    )


def apply_schema(conn: psycopg.Connection) -> None:
    with open(SCHEMA_SQL, "r") as f:
        ddl = f.read()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print("Applied schema_raw.sql (schema + typed tables created).")


def copy_csv(conn: psycopg.Connection, csv_name: str, table: str, columns: list[str]) -> int:
    path = os.path.join(DATA_DIR, f"{csv_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run phase1_generate_data/generate.py first."
        )

    col_list = ", ".join(columns)
    # COPY ... FROM STDIN WITH CSV HEADER: Postgres parses the file directly.
    copy_sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT csv, HEADER true)"

    with conn.cursor() as cur, open(path, "rb") as fh:
        with cur.copy(copy_sql) as copy:
            while chunk := fh.read(1 << 20):  # 1 MB chunks
                copy.write(chunk)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return cur.fetchone()[0]


def main() -> int:
    kw = conn_kwargs()
    print(f"Connecting to postgres://{kw['user']}@{kw['host']}:{kw['port']}/{kw['dbname']} ...")
    try:
        conn = psycopg.connect(**kw, connect_timeout=10)
    except psycopg.OperationalError as e:
        print("\nCould not connect to Postgres.", file=sys.stderr)
        print("Start it first:  cd phase2_postgres && docker compose up -d", file=sys.stderr)
        print(f"psycopg error: {e}", file=sys.stderr)
        return 1

    with conn:
        apply_schema(conn)

        print("\nLoading CSVs via COPY:")
        total = 0
        t0 = time.time()
        for csv_name, (table, columns) in TABLES.items():
            n = copy_csv(conn, csv_name, table, columns)
            print(f"  {table:22s} {n:>10,d} rows")
            total += n
        dt = time.time() - t0
        print(f"\nLoaded {total:,} rows across {len(TABLES)} tables in {dt:.1f}s.")

        # quick post-load sanity read so we know the data is actually queryable
        with conn.cursor() as cur:
            cur.execute("""
                SELECT final_outcome, count(*)
                FROM raw.applications
                GROUP BY final_outcome
                ORDER BY count(*) DESC
            """)
            print("\nSanity check -- raw.applications by final_outcome:")
            for outcome, cnt in cur.fetchall():
                print(f"  {outcome:12s} {cnt:>9,d}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

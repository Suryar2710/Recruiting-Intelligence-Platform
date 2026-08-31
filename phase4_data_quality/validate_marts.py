"""
Phase 4: Great Expectations validation of the MART layer.

Why this shape (a single programmatic runner, not the GX project/notebook scaffold):
  For a CI quality gate we need a self-contained script that connects to the
  warehouse, runs a fixed set of expectations against the actual mart tables, and
  exits non-zero if ANY expectation fails. The 0.18 fluent API lets us do exactly
  that without committing a great_expectations.yml + checkpoints tree, and it is
  far more stable across environments. It validates the tables in Postgres (the
  real marts dbt built), not CSVs.

What it validates (all against schema `marts`):
  1. Referential integrity: every fact FK exists in its dim (via a set/relationship
     check expressed as "fact FK values are a subset of dim key values").
  2. No negative time-to-fill (fact_applications, fact_requisitions) and no
     negative days-to-respond (fact_offers).
  3. feedback_score carried into fact_applications (min/avg) stays within 1..5.
  4. offer decision in the expected value set.
  5. Completeness: primary keys and foreign keys are non-null.

Exit code: 0 if all suites pass, 1 otherwise (so CI blocks on failure).

Run locally:
    # marts must be built first (dbt run) against a reachable Postgres
    python phase4_data_quality/validate_marts.py
"""

from __future__ import annotations

import os
import sys

# Disable tqdm progress bars (GX emits one per metric batch) so CI logs are clean.
os.environ.setdefault("TQDM_DISABLE", "1")

import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest


def pg_url() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5433")
    user = os.getenv("PGUSER", "recruiting")
    password = os.getenv("PGPASSWORD", "recruiting")
    db = os.getenv("PGDATABASE", "recruiting_intel")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


# ---- expectation definitions per mart table --------------------------------- #
# Each entry: a query that materializes the table (from the marts schema) and a
# function that adds expectations to a validator over that table.

def expect_dim_applicant(v):
    v.expect_column_values_to_not_be_null("applicant_key")
    v.expect_column_values_to_be_unique("applicant_key")
    v.expect_column_values_to_not_be_null("applicant_id")


def expect_dim_requisition(v):
    v.expect_column_values_to_not_be_null("requisition_key")
    v.expect_column_values_to_be_unique("requisition_key")
    v.expect_column_values_to_not_be_null("req_id")


def expect_dim_recruiter(v):
    v.expect_column_values_to_not_be_null("recruiter_key")
    v.expect_column_values_to_be_unique("recruiter_key")


def expect_dim_date(v):
    v.expect_column_values_to_not_be_null("date_key")
    v.expect_column_values_to_be_unique("date_key")


def expect_fact_applications(v, dim_keys):
    # completeness on PK + FKs
    v.expect_column_values_to_not_be_null("application_id")
    v.expect_column_values_to_be_unique("application_id")
    v.expect_column_values_to_not_be_null("applicant_key")
    v.expect_column_values_to_not_be_null("requisition_key")
    v.expect_column_values_to_not_be_null("recruiter_key")
    v.expect_column_values_to_not_be_null("applied_date_key")

    # referential integrity: FK values must be a subset of dim key sets
    v.expect_column_values_to_be_in_set("applicant_key", dim_keys["applicant"])
    v.expect_column_values_to_be_in_set("requisition_key", dim_keys["requisition"])
    v.expect_column_values_to_be_in_set("recruiter_key", dim_keys["recruiter"])
    v.expect_column_values_to_be_in_set("applied_date_key", dim_keys["date"])

    # no negative time-to-fill (null allowed: unfilled/not-hired applications)
    v.expect_column_values_to_be_between(
        "actual_time_to_fill_days", min_value=0, max_value=1000, mostly=1.0
    )

    # feedback carried into the fact stays in 1..5 (nulls allowed: no interviews)
    v.expect_column_values_to_be_between("min_feedback_score", min_value=1, max_value=5)
    v.expect_column_values_to_be_between("avg_feedback_score", min_value=1, max_value=5)

    # outcome domain
    v.expect_column_values_to_be_in_set(
        "final_outcome", ["hired", "rejected", "withdrawn", "in_progress"]
    )


def expect_fact_offers(v, dim_keys):
    v.expect_column_values_to_not_be_null("offer_id")
    v.expect_column_values_to_be_unique("offer_id")
    v.expect_column_values_to_not_be_null("application_id")
    v.expect_column_values_to_not_be_null("requisition_key")
    v.expect_column_values_to_not_be_null("recruiter_key")
    v.expect_column_values_to_not_be_null("offer_date_key")

    # referential integrity
    v.expect_column_values_to_be_in_set("requisition_key", dim_keys["requisition"])
    v.expect_column_values_to_be_in_set("recruiter_key", dim_keys["recruiter"])
    v.expect_column_values_to_be_in_set("offer_date_key", dim_keys["date"])
    v.expect_column_values_to_be_in_set("application_id", dim_keys["application"])

    # no negative days-to-respond
    v.expect_column_values_to_be_between("days_to_respond", min_value=0, max_value=365)

    # decision domain
    v.expect_column_values_to_be_in_set("decision", ["accepted", "declined", "expired"])


def expect_fact_requisitions(v, dim_keys):
    v.expect_column_values_to_not_be_null("req_id")
    v.expect_column_values_to_be_unique("req_id")
    v.expect_column_values_to_not_be_null("requisition_key")
    v.expect_column_values_to_not_be_null("recruiter_key")
    v.expect_column_values_to_not_be_null("opened_date_key")

    # referential integrity
    v.expect_column_values_to_be_in_set("requisition_key", dim_keys["requisition"])
    v.expect_column_values_to_be_in_set("recruiter_key", dim_keys["recruiter"])
    v.expect_column_values_to_be_in_set("opened_date_key", dim_keys["date"])

    # no negative aggregated time-to-fill (null allowed: unfilled reqs)
    v.expect_column_values_to_be_between(
        "avg_time_to_fill_days", min_value=0, max_value=1000, mostly=1.0
    )
    v.expect_column_values_to_be_between(
        "max_time_to_fill_days", min_value=0, max_value=1000, mostly=1.0
    )

    # logical consistency between the two miss flags: all_hires_missed_target
    # implies any_hire_missed_target (all-missed must imply any-missed).
    #
    # GX 0.18 does not register a working column-pair ">= " expectation for
    # boolean columns on the SQL engine, so we express the rule with the
    # registered pair expectation via casting to int (False->0, True->1) inside a
    # derived-column check. We implement it as a SQL unexpected-rows check in
    # check_flag_consistency() (called from main) rather than here, because that
    # is the portable, unambiguous equivalent. See that function.


TABLES = [
    ("dim_applicant", expect_dim_applicant, False),
    ("dim_requisition", expect_dim_requisition, False),
    ("dim_recruiter", expect_dim_recruiter, False),
    ("dim_date", expect_dim_date, False),
    ("fact_applications", expect_fact_applications, True),
    ("fact_offers", expect_fact_offers, True),
    ("fact_requisitions", expect_fact_requisitions, True),
]


def load_dim_keys(context, datasource_name):
    """Pull dimension key sets used for referential-integrity subset checks."""
    import sqlalchemy as sa

    engine = sa.create_engine(pg_url())
    keys = {}
    with engine.connect() as conn:
        keys["applicant"] = [r[0] for r in conn.execute(
            sa.text("select applicant_key from marts.dim_applicant"))]
        keys["requisition"] = [r[0] for r in conn.execute(
            sa.text("select requisition_key from marts.dim_requisition"))]
        keys["recruiter"] = [r[0] for r in conn.execute(
            sa.text("select recruiter_key from marts.dim_recruiter"))]
        keys["date"] = [r[0] for r in conn.execute(
            sa.text("select date_key from marts.dim_date"))]
        keys["application"] = [r[0] for r in conn.execute(
            sa.text("select application_id from marts.fact_applications"))]
    engine.dispose()
    return keys


def get_validator(context, datasource_name, table):
    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector",
        data_asset_name=table,
        runtime_parameters={"query": f"select * from marts.{table}"},
        batch_identifiers={"default_identifier_name": table},
    )
    return context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=f"suite_{table}",
    )


def check_flag_consistency() -> tuple[bool, int]:
    """
    Logical-consistency check on fact_requisitions:
      all_hires_missed_target = True  implies  any_hire_missed_target = True.

    Equivalent to a column-pair "any >= all" expectation (booleans ordered
    False < True). We count rows that violate it -- i.e. all-missed is true but
    any-missed is not -- and pass only if that count is zero. Unfilled reqs have
    both flags null and are naturally excluded by the WHERE clause.

    Returns (ok, violation_count).
    """
    import sqlalchemy as sa

    engine = sa.create_engine(pg_url())
    with engine.connect() as conn:
        violations = conn.execute(
            sa.text(
                """
                select count(*)
                from marts.fact_requisitions
                where all_hires_missed_target = true
                  and (any_hire_missed_target is distinct from true)
                """
            )
        ).scalar_one()
    engine.dispose()
    return (violations == 0, int(violations))


def main() -> int:
    context = gx.get_context()
    # Silence GX's per-metric tqdm progress bars so CI logs stay readable.
    try:
        from great_expectations.data_context.types.base import ProgressBarsConfig
        context.variables.progress_bars = ProgressBarsConfig(
            globally=False, profilers=False, metric_calculations=False
        )
    except Exception:
        pass

    datasource_name = "recruiting_marts"
    context.add_datasource(
        name=datasource_name,
        class_name="Datasource",
        execution_engine={
            "class_name": "SqlAlchemyExecutionEngine",
            "connection_string": pg_url(),
        },
        data_connectors={
            "default_runtime_data_connector": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"],
            }
        },
    )

    print("Loading dimension key sets for referential-integrity checks ...")
    dim_keys = load_dim_keys(context, datasource_name)
    print(
        f"  applicant={len(dim_keys['applicant'])}, "
        f"requisition={len(dim_keys['requisition'])}, "
        f"recruiter={len(dim_keys['recruiter'])}, "
        f"date={len(dim_keys['date'])}, "
        f"application={len(dim_keys['application'])}"
    )

    all_ok = True
    print("\nValidating mart tables:")
    for table, add_expectations, needs_keys in TABLES:
        context.add_or_update_expectation_suite(f"suite_{table}")
        validator = get_validator(context, datasource_name, table)
        if needs_keys:
            add_expectations(validator, dim_keys)
        else:
            add_expectations(validator)
        result = validator.validate()
        n = result.statistics["evaluated_expectations"]
        passed = result.statistics["successful_expectations"]
        ok = result.success
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  {table:20s} {status}  ({passed}/{n} expectations)")
        if not ok:
            for r in result.results:
                if not r.success:
                    exp = r.expectation_config
                    print(f"      FAILED: {exp.expectation_type} "
                          f"on {exp.kwargs.get('column', '(table)')}")

    # Cross-column logical-consistency check (all-missed implies any-missed).
    flag_ok, flag_violations = check_flag_consistency()
    all_ok = all_ok and flag_ok
    status = "PASS" if flag_ok else "FAIL"
    print(f"  {'fact_requisitions*':20s} {status}  "
          f"(all_hires_missed => any_hire_missed; {flag_violations} violations)")

    print()
    if all_ok:
        print("All mart expectation suites PASSED.")
        return 0
    print("One or more mart expectation suites FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

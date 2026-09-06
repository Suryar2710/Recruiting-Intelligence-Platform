# Bring Your Own Data

## What this project actually hands you

This is **not** a pre-trained model you can point at any company's ATS and get
useful predictions. That wouldn't be honest or defensible: every ATS has a
different schema, every company defines its funnel stages differently, and a
model trained on this repo's *synthetic* data has learned this repo's synthetic
correlations — it would not transfer to your real hiring outcomes.

What you get is the **validated pipeline and methodology**:

- a typed raw-landing schema and fast bulk loader (`phase2_postgres/`),
- dbt transformations from raw → staging → intermediate → star-schema marts,
  with 105+ data tests (`phase3_dbt/`),
- a Great Expectations quality gate wired into CI that blocks on failure
  (`phase4_data_quality/`),
- a leakage-audited feature mart (`phase5` / `mart_ml_features`),
- three model-training scripts with documented leakage discipline and honest
  metrics/ceiling analysis (`phase6_models/`),
- a FastAPI service and a one-command Docker stack.

To use it on real data you **map your ATS export to the schema below, then
re-run the pipeline (including retraining the three models) on your data.** The
portable asset is the pipeline, not the model weights.

---

## Required schema

Map your ATS export to these seven tables and load them into the `raw` schema
(see `phase2_postgres/schema_raw.sql` for the exact DDL and
`phase2_postgres/load_raw.py` for the loader). Types are what the loader/dbt
expect. Nothing here needs to be renamed inside your ATS — you produce CSVs (or
tables) that match these column names and types.

### `applicants` — one row per candidate
| column | type | notes |
|---|---|---|
| applicant_id | text | natural key, unique |
| source | text | acquisition channel (see accepted values below) |
| applied_date | date | when the candidate entered the pool |
| resume_years_experience | numeric | |
| education_level | text | free-ish text; lowercased in staging |
| referral_flag | boolean | true if a referral |

### `requisitions` — one row per open role
| column | type | notes |
|---|---|---|
| req_id | text | natural key, unique |
| department | text | |
| seniority_level | text | see accepted values below |
| location | text | |
| target_time_to_fill_days | integer | your SLA/target for this role |
| opened_date | date | when the req opened (anchors time-to-fill) |
| hiring_manager_id | text | |
| recruiter_id | text | used for recruiter-load feature |

### `applications` — one row per candidate × requisition
| column | type | notes |
|---|---|---|
| application_id | text | natural key, unique |
| applicant_id | text | FK → applicants |
| req_id | text | FK → requisitions |
| applied_date | date | |
| current_stage | text | where the application currently sits |
| final_outcome | text | see accepted values below |

### `stage_history` — one row per stage an application passed through
| column | type | notes |
|---|---|---|
| application_id | text | FK → applications |
| stage_name | text | see accepted values below |
| entered_date | date | |
| exited_date | date | |

### `interviews` — one row per interview
| column | type | notes |
|---|---|---|
| interview_id | text | natural key, unique |
| application_id | text | FK → applications |
| interview_round | integer | 1, 2, 3, … |
| interviewer_id | text | |
| scheduled_date | date | |
| completed_flag | boolean | |
| feedback_score | integer | **must be 1–5** (enforced by a dbt test) |
| feedback_text | text | optional |

### `offers` — one row per extended offer
| column | type | notes |
|---|---|---|
| offer_id | text | natural key, unique |
| application_id | text | FK → applications |
| offer_date | date | |
| offer_salary | numeric | |
| band_midpoint | numeric | midpoint of the pay band (drives decline signal) |
| response_date | date | |
| decision | text | see accepted values below |

### `hires` — one row per hire
| column | type | notes |
|---|---|---|
| hire_id | text | natural key, unique |
| offer_id | text | FK → offers |
| start_date | date | anchors actual time-to-fill |

---

## Categorical values you must map

The dbt staging tests enforce these value sets. If your ATS uses different
labels, map them to these (or update the `accepted_values` lists in
`phase3_dbt/models/staging/_staging.yml` to match your vocabulary — the tests
are the contract, so keep them and your data in sync):

| field | accepted values |
|---|---|
| `applicants.source` | referral, job_board, linkedin, university, agency |
| `requisitions.seniority_level` | junior, mid, senior, staff, principal |
| `applications.final_outcome` | hired, rejected, withdrawn, in_progress |
| `stage_history.stage_name` | applied, screen, phone_interview, onsite, final |
| `offers.decision` | accepted, declined, expired |
| `interviews.feedback_score` | integers 1–5 |

Two derived concepts the pipeline computes for you (no need to supply them):
- **is_niche_location** — currently hardcoded to `{boise, reno, berlin}` as this
  dataset's thin talent markets, in two dbt models
  (`phase3_dbt/models/marts/dim_requisition.sql` and
  `phase3_dbt/models/marts/mart_ml_features.sql`). For real data, edit that set
  in both files to your genuinely hard-to-fill locations, or replace it with a
  data-driven flag. (`int_hire_pace` derives the miss-rate signal from actual vs
  target fill times and does not reference the niche set directly.)
- **max_stage_dwell_days**, **recruiter_concurrent_open_reqs** — computed in the
  feature mart from `stage_history` and `requisitions`.

---

## Steps to run on real data

1. **Connection is already config-driven.** `load_raw.py`, dbt, and the API all
   read `PGHOST / PGPORT / PGUSER / PGPASSWORD / PGDATABASE` env vars — point
   them at your Postgres. No code change needed for connection.
2. **Produce CSVs** matching the seven schemas above (filenames = table names,
   e.g. `applicants.csv`) in the `data/` directory, **instead of** running
   `phase1_generate_data/generate.py`. (The generator exists only to fabricate
   the demo dataset; on real data you skip it entirely.)
3. **Load**: `python phase2_postgres/load_raw.py`.
4. **Transform + test**: `cd phase3_dbt && dbt build`. Fix any `accepted_values`
   / relationship test failures — those are telling you where your data's
   vocabulary or referential integrity differs from the contract.
5. **Validate**: `python phase4_data_quality/validate_marts.py` (the same gate CI
   runs).
6. **Retrain** the three models on *your* marts:
   `python phase6_models/train_time_to_fill_risk.py` (and the drop-off and
   offer-decline scripts). The scripts read from `marts.mart_ml_features`, so
   they retrain on whatever data you loaded. New model cards are written with
   *your* metrics.
7. **Serve**: the FastAPI service loads the freshly retrained `models/*.joblib`.

## Why retraining is non-negotiable

The three models encode relationships learned from this repo's synthetic
generator (e.g. referrals accept ~15pp more, offers below 90% of band decline
far more, niche-location senior roles miss their fill target). Your real
correlations will differ in strength, direction, and which features matter. The
leakage discipline baked into the training scripts still applies — e.g. the
time-to-fill model only uses features known at req-open time, the offer model
excludes post-response fields — so retraining gives you honest, leakage-safe
models on your data, which is the whole point.

## What is intentionally *not* over-engineered

`load_raw.py` hardcodes the seven table/column definitions on purpose: that
mapping **is** the schema contract a BYOD user must satisfy, so making it fully
dynamic would hide the very thing you need to conform to. Connection settings —
the thing that genuinely varies per environment — are already env-var driven.

-- Phase 2: raw landing schema.
-- These tables mirror the Phase 1 CSVs 1:1 with explicit types. No transformation,
-- no constraints beyond types -- cleaning/typing/renaming is dbt's job in Phase 3.
-- We deliberately DO NOT declare PK/FK constraints here: the "raw" layer should
-- ingest whatever the source produced, and quality gates live downstream (dbt
-- tests + Great Expectations). Declaring types (not just text) still gives us a
-- sane starting point and catches gross corruption at load time.

CREATE SCHEMA IF NOT EXISTS raw;

-- CASCADE so reloads don't fail when downstream dbt staging views already
-- reference these tables (they get rebuilt by `dbt run` after the reload).
DROP TABLE IF EXISTS raw.applicants CASCADE;
CREATE TABLE raw.applicants (
    applicant_id             text,
    source                   text,
    applied_date             date,
    resume_years_experience  numeric,
    education_level          text,
    referral_flag            boolean
);

DROP TABLE IF EXISTS raw.requisitions CASCADE;
CREATE TABLE raw.requisitions (
    req_id                    text,
    department                text,
    seniority_level           text,
    location                  text,
    target_time_to_fill_days  integer,
    opened_date               date,
    hiring_manager_id         text,
    recruiter_id              text
);

DROP TABLE IF EXISTS raw.applications CASCADE;
CREATE TABLE raw.applications (
    application_id  text,
    applicant_id    text,
    req_id          text,
    applied_date    date,
    current_stage   text,
    final_outcome   text
);

DROP TABLE IF EXISTS raw.stage_history CASCADE;
CREATE TABLE raw.stage_history (
    application_id  text,
    stage_name      text,
    entered_date    date,
    exited_date     date
);

DROP TABLE IF EXISTS raw.interviews CASCADE;
CREATE TABLE raw.interviews (
    interview_id     text,
    application_id   text,
    interview_round  integer,
    interviewer_id   text,
    scheduled_date   date,
    completed_flag   boolean,
    feedback_score   integer,
    feedback_text    text
);

DROP TABLE IF EXISTS raw.offers CASCADE;
CREATE TABLE raw.offers (
    offer_id        text,
    application_id  text,
    offer_date      date,
    offer_salary    numeric,
    band_midpoint   numeric,
    response_date   date,
    decision        text
);

DROP TABLE IF EXISTS raw.hires CASCADE;
CREATE TABLE raw.hires (
    hire_id     text,
    offer_id    text,
    start_date  date
);

-- mart_ml_features
-- Grain: one row per application.
--
-- The single feature table the three Phase 6 models train on. Built as a dbt
-- mart (not a Python script) because it is entirely a join + computed columns
-- over existing staging/intermediate models: no iterative or stateful logic that
-- needs Python. As a mart it gets free lineage (dbt docs), rides the existing
-- test/CI pipeline, and materializes as a Postgres table the ML scripts query
-- directly -- no extra runtime hop or second DB connection.
--
-- Feature groups:
--   candidate      : source, referral_flag, resume_years_experience
--   process        : interview_rounds_completed, avg_feedback_score,
--                     max_gap_between_rounds_days, interviewer_count
--   requisition    : department, seniority_level, location, is_niche_location,
--                     recruiter_concurrent_open_reqs (point-in-time load)
--   offer (subset) : offer_to_band_ratio, days_to_respond  -- only for
--                     applications that reached the offer stage; null otherwise
--
-- Labels (targets):
--   missed_target  : time-to-fill risk. Non-null only for hired applications
--                    (a fill actually happened). applications->hires is 1:1, so
--                    this attaches cleanly at the application grain.
--   was_dropped    : candidate drop-off (from int_application_funnel).
--   is_offer_declined : offer-decline label; non-null only where an offer exists.

with applications as (
    select * from {{ ref('stg_applications') }}
),

applicants as (
    select * from {{ ref('stg_applicants') }}
),

requisitions as (
    select * from {{ ref('stg_requisitions') }}
),

funnel as (
    select * from {{ ref('int_application_funnel') }}
),

offer_outcomes as (
    select * from {{ ref('int_offer_outcomes') }}
),

hire_pace as (
    select application_id, missed_target
    from {{ ref('int_hire_pace') }}
),

-- process features: interview rollups incl. max gap between consecutive rounds
interview_gaps as (
    select
        application_id,
        interview_round,
        scheduled_date,
        feedback_score,
        interviewer_id,
        scheduled_date
            - lag(scheduled_date) over (
                partition by application_id order by interview_round
            ) as gap_from_prev_round
    from {{ ref('stg_interviews') }}
),

interview_features as (
    select
        application_id,
        count(*)                                as interview_rounds_completed,
        round(avg(feedback_score), 3)           as avg_feedback_score,
        min(feedback_score)                     as min_feedback_score,
        count(distinct interviewer_id)          as interviewer_count,
        max(gap_from_prev_round)                as max_gap_between_rounds_days
    from interview_gaps
    group by application_id
),

-- requisition feature: recruiter's concurrent OPEN-req load at the moment this
-- application was submitted. A req counts as "open at time T" for the same
-- recruiter if it opened on/before T and (it was never filled OR it was filled
-- after T). We exclude the application's own requisition from the count.
app_req as (
    select
        a.application_id,
        a.applied_date,
        r.req_id        as this_req_id,
        r.recruiter_id
    from applications a
    join requisitions r on a.req_id = r.req_id
),

-- earliest fill date per requisition (null if never filled)
req_fill as (
    select
        app.req_id,
        min(h.start_date) as filled_date
    from {{ ref('stg_hires') }} h
    join {{ ref('stg_offers') }} o        on h.offer_id = o.offer_id
    join {{ ref('stg_applications') }} app on o.application_id = app.application_id
    group by app.req_id
),

concurrent_load as (
    select
        ar.application_id,
        count(*) as recruiter_concurrent_open_reqs
    from app_req ar
    join requisitions r2
        on r2.recruiter_id = ar.recruiter_id
       and r2.req_id <> ar.this_req_id
       and r2.opened_date <= ar.applied_date
    left join req_fill rf2 on r2.req_id = rf2.req_id
    where rf2.filled_date is null
       or rf2.filled_date > ar.applied_date
    group by ar.application_id
),

final as (
    select
        f.application_id,

        -- ---- candidate features ----
        ap.source                                       as candidate_source,
        ap.is_referral                                  as referral_flag,
        ap.resume_years_experience,

        -- ---- process features ----
        coalesce(iff.interview_rounds_completed, 0)     as interview_rounds_completed,
        iff.avg_feedback_score,
        iff.min_feedback_score,
        coalesce(iff.interviewer_count, 0)              as interviewer_count,
        iff.max_gap_between_rounds_days,
        -- max dwell in any single funnel stage (days). Unlike the interview-round
        -- gap, this exists for EVERY application (incl. those that dropped before
        -- any interview) and is where the generator's ">10-day gap -> drop-off"
        -- signal actually lives. Non-interview stages count too.
        greatest(
            coalesce(f.days_in_applied, 0),
            coalesce(f.days_in_screen, 0),
            coalesce(f.days_in_phone_interview, 0),
            coalesce(f.days_in_onsite, 0),
            coalesce(f.days_in_final, 0)
        )                                               as max_stage_dwell_days,

        -- ---- requisition features ----
        r.department,
        r.seniority_level,
        r.location,
        (r.location in ('boise', 'reno', 'berlin'))     as is_niche_location,
        coalesce(cl.recruiter_concurrent_open_reqs, 0)  as recruiter_concurrent_open_reqs,

        -- ---- offer features (only where an offer exists) ----
        oo.offer_to_band_ratio,
        oo.days_to_respond,
        (oo.offer_id is not null)                       as reached_offer_stage,

        -- ---- context / grain helpers ----
        f.current_stage,
        f.final_outcome,
        f.applied_date,

        -- ---- LABELS ----
        -- time-to-fill risk: non-null only for hired applications
        hp.missed_target                                as label_missed_target,
        -- candidate drop-off
        f.was_dropped                                   as label_was_dropped,
        -- offer decline: non-null only where an offer exists
        case
            when oo.offer_id is not null
                then (oo.decision = 'declined')
            else null
        end                                             as label_offer_declined

    from funnel f
    join applications a       on f.application_id = a.application_id
    left join applicants ap   on a.applicant_id = ap.applicant_id
    left join requisitions r  on a.req_id = r.req_id
    left join interview_features iff on f.application_id = iff.application_id
    left join concurrent_load cl     on f.application_id = cl.application_id
    left join offer_outcomes oo      on f.application_id = oo.application_id
    left join hire_pace hp           on f.application_id = hp.application_id
)

select * from final

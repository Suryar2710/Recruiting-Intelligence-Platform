-- fact_applications
-- Grain: one row per application.
--
-- The central fact for candidate/application-level analysis and the Phase 6
-- drop-off model. Carries:
--   * dimension foreign keys (applicant, requisition, recruiter, dates)
--   * funnel measures from int_application_funnel (days per stage, total days to
--     decision, was_dropped, drop_stage)
--   * interview aggregates (rounds, avg feedback score) rolled up per application
--   * the per-hire time-to-fill fact for HIRED applications.
--
-- Why the hire-level time-to-fill fact lives here (not only on fact_requisitions):
--   applications -> offers -> hires is a strict 1:1 chain in this data (verified:
--   1 offer per application, 1 hire per offer). So a hired application maps to
--   exactly one hire and one unambiguous fill date. That makes fact_applications
--   the natural, grain-safe home for per-hire actual_time_to_fill. fact_requisitions
--   instead AGGREGATES these hire events up to the req grain (a req can have many
--   hires), so the two facts answer different questions without double-counting.

with funnel as (
    select * from {{ ref('int_application_funnel') }}
),

applications as (
    select application_id, applicant_id, req_id from {{ ref('stg_applications') }}
),

requisitions as (
    select req_id, recruiter_id from {{ ref('stg_requisitions') }}
),

-- interview rollup per application
interviews as (
    select
        application_id,
        count(*)                                as interview_count,
        max(interview_round)                    as max_interview_round,
        round(avg(feedback_score), 3)           as avg_feedback_score,
        min(feedback_score)                     as min_feedback_score,
        count(distinct interviewer_id)          as distinct_interviewers
    from {{ ref('stg_interviews') }}
    group by application_id
),

-- per-hire fill fact (1:1 with a hired application)
hire_pace as (
    select
        application_id,
        actual_time_to_fill_days,
        days_over_target,
        missed_target
    from {{ ref('int_hire_pace') }}
),

final as (
    select
        f.application_id,

        -- dimension foreign keys (surrogate keys mirror the dims' key logic)
        {{ dbt_utils.generate_surrogate_key(['f.applicant_id']) }}  as applicant_key,
        {{ dbt_utils.generate_surrogate_key(['f.req_id']) }}        as requisition_key,
        {{ dbt_utils.generate_surrogate_key(['r.recruiter_id']) }}  as recruiter_key,
        cast(to_char(f.applied_date, 'YYYYMMDD') as integer)        as applied_date_key,

        -- natural keys retained for lineage
        f.applicant_id,
        f.req_id,
        r.recruiter_id,
        f.applied_date,

        -- funnel measures
        f.current_stage,
        f.final_outcome,
        f.days_in_applied,
        f.days_in_screen,
        f.days_in_phone_interview,
        f.days_in_onsite,
        f.days_in_final,
        f.total_days_to_decision,
        f.stages_recorded,
        f.was_dropped,
        f.drop_stage,

        -- interview measures (0 / null where the application had no interviews)
        coalesce(i.interview_count, 0)          as interview_count,
        i.max_interview_round,
        i.avg_feedback_score,
        i.min_feedback_score,
        coalesce(i.distinct_interviewers, 0)    as distinct_interviewers,

        -- outcome flags
        (f.final_outcome = 'hired')             as is_hired,

        -- per-hire time-to-fill fact (null unless this application was hired)
        hp.actual_time_to_fill_days,
        hp.days_over_target,
        hp.missed_target

    from funnel f
    left join requisitions r on f.req_id = r.req_id
    left join interviews i   on f.application_id = i.application_id
    left join hire_pace hp   on f.application_id = hp.application_id
)

select * from final

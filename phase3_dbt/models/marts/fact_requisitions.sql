-- fact_requisitions
-- Grain: one row per requisition.
--
-- The fact for requisition-level analysis and the Phase 6 requisition
-- time-to-fill-risk model (which scores OPEN reqs: "will this miss its target?").
--
-- Grain decision on hire-level time-to-fill:
--   int_hire_pace is at the HIRE grain (a req can have many hires, ~4.6 per
--   filled req here). To keep this fact strictly one-row-per-req, we AGGREGATE
--   int_hire_pace up to the req grain rather than selecting from it directly:
--     - hire_count
--     - avg / min / max actual_time_to_fill_days
--     - any_hire_missed_target (worst-case: did ANY hire on this req miss target)
--     - all_hires_missed_target
--   Selecting int_hire_pace directly would fan the req out to one row per hire
--   and break the grain (and double-count reqs in BI). These aggregates are
--   historical/outcome measures; the live risk model in Phase 6 scores still-open
--   reqs using requisition-grain features (dept/seniority/location/recruiter load)
--   that exist before any hire, so it does not depend on these backward-looking
--   aggregates -- they are for BI and for constructing training labels.
--
-- We also roll up application-funnel activity per req (applications received,
-- offers extended, drop-offs) to support the "which reqs/stages are bottlenecks"
-- business question.

with requisitions as (
    select * from {{ ref('stg_requisitions') }}
),

applications as (
    select application_id, req_id, final_outcome from {{ ref('stg_applications') }}
),

funnel as (
    select application_id, req_id, was_dropped from {{ ref('int_application_funnel') }}
),

hire_pace as (
    select * from {{ ref('int_hire_pace') }}
),

-- application-activity rollup per requisition
app_rollup as (
    select
        req_id,
        count(*)                                                    as applications_received,
        count(*) filter (where final_outcome = 'hired')             as hired_count,
        count(*) filter (where final_outcome = 'rejected')          as rejected_count,
        count(*) filter (where final_outcome = 'withdrawn')         as withdrawn_count,
        count(*) filter (where final_outcome = 'in_progress')       as in_progress_count
    from applications
    group by req_id
),

-- drop-off rollup per requisition (candidate withdrawals)
drop_rollup as (
    select
        req_id,
        count(*) filter (where was_dropped) as dropped_count
    from funnel
    group by req_id
),

-- hire-pace AGGREGATED from the hire grain up to the req grain
pace_rollup as (
    select
        req_id,
        count(*)                                        as hire_count,
        round(avg(actual_time_to_fill_days), 1)         as avg_time_to_fill_days,
        min(actual_time_to_fill_days)                   as min_time_to_fill_days,
        max(actual_time_to_fill_days)                   as max_time_to_fill_days,
        bool_or(missed_target)                          as any_hire_missed_target,
        bool_and(missed_target)                         as all_hires_missed_target
    from hire_pace
    group by req_id
),

final as (
    select
        r.req_id,

        -- dimension foreign keys
        {{ dbt_utils.generate_surrogate_key(['r.req_id']) }}        as requisition_key,
        {{ dbt_utils.generate_surrogate_key(['r.recruiter_id']) }}  as recruiter_key,
        cast(to_char(r.opened_date, 'YYYYMMDD') as integer)         as opened_date_key,

        -- descriptive keys/attributes for convenience
        r.recruiter_id,
        r.hiring_manager_id,
        r.department,
        r.seniority_level,
        r.location,
        r.opened_date,
        r.target_time_to_fill_days,

        -- application-activity measures
        coalesce(ar.applications_received, 0)   as applications_received,
        coalesce(ar.hired_count, 0)             as hired_count,
        coalesce(ar.rejected_count, 0)          as rejected_count,
        coalesce(ar.withdrawn_count, 0)         as withdrawn_count,
        coalesce(ar.in_progress_count, 0)       as in_progress_count,
        coalesce(dr.dropped_count, 0)           as dropped_count,

        -- pace measures aggregated from int_hire_pace (null if never filled)
        coalesce(pr.hire_count, 0)              as hire_count,
        (pr.hire_count is not null)             as is_filled,
        pr.avg_time_to_fill_days,
        pr.min_time_to_fill_days,
        pr.max_time_to_fill_days,
        pr.any_hire_missed_target,
        pr.all_hires_missed_target

    from requisitions r
    left join app_rollup ar  on r.req_id = ar.req_id
    left join drop_rollup dr on r.req_id = dr.req_id
    left join pace_rollup pr on r.req_id = pr.req_id
)

select * from final

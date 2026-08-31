-- int_hire_pace
-- Grain: one row per HIRE event (not per requisition).
--
-- Why per-hire, not per-req:
--   A requisition can produce multiple hires (~4.6 per filled req in this
--   dataset). Computing time-to-fill as MIN(hire.start_date) per requisition
--   only captures the single fastest hire and discards the rest, which deflated
--   the target-miss rate to ~14.6% and broke alignment with the validated signal.
--   The honest unit for "how long did this fill take" is the individual hire, so
--   we grain here at one row per hire and compute actual_time_to_fill_days,
--   days_over_target, and missed_target for each hire event, carrying the
--   requisition's department / seniority / location / target onto every row.
--   This reproduces the validated per-hire miss-rate split
--   (~39.5% baseline / ~67.8% senior+niche).
--
-- Lineage: hire -> offer -> application -> requisition.
-- Every row is by definition a filled/hired outcome, so there is no is_filled
-- flag here (it would be true throughout and convey no grain-level meaning);
-- open/unfilled requisitions are simply absent from this model.

with hires as (
    select * from {{ ref('stg_hires') }}
),

offers as (
    select offer_id, application_id from {{ ref('stg_offers') }}
),

applications as (
    select application_id, req_id from {{ ref('stg_applications') }}
),

requisitions as (
    select * from {{ ref('stg_requisitions') }}
),

final as (
    select
        h.hire_id,
        h.offer_id,
        app.application_id,
        r.req_id,

        -- requisition attributes carried onto every hire row for grouping
        r.department,
        r.seniority_level,
        r.location,
        r.recruiter_id,
        r.hiring_manager_id,

        r.opened_date,
        h.start_date                                                as filled_date,
        r.target_time_to_fill_days,

        -- per-hire time-to-fill: this hire's start date minus the req open date
        (h.start_date - r.opened_date)                              as actual_time_to_fill_days,

        -- variance vs target (positive = slower than target) and a miss flag,
        -- computed individually for every hire event
        ((h.start_date - r.opened_date) - r.target_time_to_fill_days) as days_over_target,
        ((h.start_date - r.opened_date) > r.target_time_to_fill_days) as missed_target

    from hires h
    join offers o        on h.offer_id = o.offer_id
    join applications app on o.application_id = app.application_id
    join requisitions r   on app.req_id = r.req_id
)

select * from final

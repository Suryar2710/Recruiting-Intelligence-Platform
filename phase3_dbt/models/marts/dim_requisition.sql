-- dim_requisition
-- Grain: one row per requisition.
-- Descriptive attributes of the role being filled. Time-to-fill / pace metrics
-- live on fact_requisitions, not here -- a dimension holds descriptors, a fact
-- holds measures. is_niche_location is a derived attribute used heavily by the
-- Phase 6 requisition-risk model, so it is pre-computed here once.

with requisitions as (
    select * from {{ ref('stg_requisitions') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['req_id']) }}       as requisition_key,
    req_id,
    department,
    seniority_level,
    location,
    (location in ('boise', 'reno', 'berlin'))               as is_niche_location,
    recruiter_id,
    hiring_manager_id,
    target_time_to_fill_days,
    opened_date
from requisitions

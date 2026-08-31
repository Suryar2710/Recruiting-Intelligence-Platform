-- dim_recruiter
-- Grain: one row per recruiter.
-- There is no standalone recruiter source table; recruiters are referenced by
-- recruiter_id on requisitions. We derive the dimension from the distinct
-- recruiters seen on requisitions and attach a lightweight descriptive count of
-- how many requisitions they own overall. (The time-varying "concurrent open-req
-- load" feature the Phase 6 model needs is a point-in-time measure and belongs in
-- the ML feature mart in Phase 5, not on this slowly-changing dimension.)

with requisitions as (
    select * from {{ ref('stg_requisitions') }}
),

recruiters as (
    select
        recruiter_id,
        count(*) as total_requisitions_owned
    from requisitions
    where recruiter_id is not null
    group by recruiter_id
)

select
    {{ dbt_utils.generate_surrogate_key(['recruiter_id']) }} as recruiter_key,
    recruiter_id,
    total_requisitions_owned
from recruiters

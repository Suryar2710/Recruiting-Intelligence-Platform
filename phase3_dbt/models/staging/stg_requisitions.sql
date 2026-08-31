-- Staging: requisitions. 1:1 with raw.requisitions.

with source as (
    select * from {{ source('raw', 'requisitions') }}
),

renamed as (
    select
        cast(req_id as varchar)                          as req_id,
        lower(nullif(trim(department), ''))              as department,
        lower(nullif(trim(seniority_level), ''))         as seniority_level,
        lower(nullif(trim(location), ''))                as location,
        cast(target_time_to_fill_days as integer)        as target_time_to_fill_days,
        cast(opened_date as date)                        as opened_date,
        cast(hiring_manager_id as varchar)               as hiring_manager_id,
        cast(recruiter_id as varchar)                    as recruiter_id
    from source
)

select * from renamed

-- Staging: stage_history. 1:1 with raw.stage_history (one row per stage an
-- application passed through). No dedup/aggregation here -- that is intermediate's job.

with source as (
    select * from {{ source('raw', 'stage_history') }}
),

renamed as (
    select
        cast(application_id as varchar)                  as application_id,
        lower(nullif(trim(stage_name), ''))              as stage_name,
        cast(entered_date as date)                       as entered_date,
        cast(exited_date as date)                        as exited_date
    from source
)

select * from renamed

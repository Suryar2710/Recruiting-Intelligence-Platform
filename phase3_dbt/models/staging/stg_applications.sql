-- Staging: applications. 1:1 with raw.applications.

with source as (
    select * from {{ source('raw', 'applications') }}
),

renamed as (
    select
        cast(application_id as varchar)                  as application_id,
        cast(applicant_id as varchar)                    as applicant_id,
        cast(req_id as varchar)                          as req_id,
        cast(applied_date as date)                       as applied_date,
        lower(nullif(trim(current_stage), ''))           as current_stage,
        lower(nullif(trim(final_outcome), ''))           as final_outcome
    from source
)

select * from renamed

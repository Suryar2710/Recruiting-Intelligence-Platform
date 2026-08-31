-- Staging: interviews. 1:1 with raw.interviews.

with source as (
    select * from {{ source('raw', 'interviews') }}
),

renamed as (
    select
        cast(interview_id as varchar)                    as interview_id,
        cast(application_id as varchar)                  as application_id,
        cast(interview_round as integer)                 as interview_round,
        cast(interviewer_id as varchar)                  as interviewer_id,
        cast(scheduled_date as date)                     as scheduled_date,
        coalesce(cast(completed_flag as boolean), false) as is_completed,
        cast(feedback_score as integer)                  as feedback_score,
        nullif(trim(feedback_text), '')                  as feedback_text
    from source
)

select * from renamed

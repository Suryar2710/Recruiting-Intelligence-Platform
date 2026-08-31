-- Staging: applicants. 1:1 with raw.applicants.
-- Type casting, light text normalization, empty-string -> null. No business logic.

with source as (
    select * from {{ source('raw', 'applicants') }}
),

renamed as (
    select
        cast(applicant_id as varchar)                             as applicant_id,
        lower(nullif(trim(source), ''))                           as source,
        cast(applied_date as date)                                as applied_date,
        cast(resume_years_experience as numeric)                  as resume_years_experience,
        lower(nullif(trim(education_level), ''))                  as education_level,
        -- referral_flag is already boolean in raw; coalesce nulls to false since a
        -- missing referral flag means "not a referral".
        coalesce(cast(referral_flag as boolean), false)           as is_referral
    from source
)

select * from renamed

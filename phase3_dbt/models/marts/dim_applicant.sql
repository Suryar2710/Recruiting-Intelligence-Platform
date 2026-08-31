-- dim_applicant
-- Grain: one row per applicant.
-- Conformed dimension describing a candidate: acquisition source, experience,
-- education, referral flag. A surrogate key is generated for star-schema joins,
-- while the natural key (applicant_id) is retained for lineage/debugging.

with applicants as (
    select * from {{ ref('stg_applicants') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['applicant_id']) }}  as applicant_key,
    applicant_id,
    source                    as applicant_source,
    resume_years_experience,
    education_level,
    is_referral
from applicants

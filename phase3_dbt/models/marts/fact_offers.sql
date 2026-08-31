-- fact_offers
-- Grain: one row per offer.
--
-- The fact for offer-economics analysis and the Phase 6 offer-decline model.
-- Carries offer measures (salary, band delta, offer-to-band ratio, days to
-- respond, decision flags) plus dimension foreign keys resolved through the
-- offer's application (application -> applicant / requisition / recruiter).

with offers as (
    select * from {{ ref('int_offer_outcomes') }}
),

applications as (
    select application_id, applicant_id, req_id from {{ ref('stg_applications') }}
),

requisitions as (
    select req_id, recruiter_id from {{ ref('stg_requisitions') }}
),

final as (
    select
        o.offer_id,

        -- dimension foreign keys
        {{ dbt_utils.generate_surrogate_key(['a.applicant_id']) }}  as applicant_key,
        {{ dbt_utils.generate_surrogate_key(['a.req_id']) }}        as requisition_key,
        {{ dbt_utils.generate_surrogate_key(['r.recruiter_id']) }}  as recruiter_key,
        cast(to_char(o.offer_date, 'YYYYMMDD') as integer)          as offer_date_key,

        -- natural keys for lineage
        o.application_id,
        a.applicant_id,
        a.req_id,
        r.recruiter_id,

        -- offer measures
        o.offer_date,
        o.response_date,
        o.offer_salary,
        o.band_midpoint,
        o.offer_vs_band_delta,
        o.offer_to_band_ratio,
        o.is_below_band,
        o.days_to_respond,

        -- outcome
        o.decision,
        o.is_accepted,
        o.is_declined

    from offers o
    left join applications a on o.application_id = a.application_id
    left join requisitions r on a.req_id = r.req_id
)

select * from final

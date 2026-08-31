-- int_offer_outcomes
-- Grain: one row per offer.
--
-- Derives the offer economics and response behavior the offer-decline model
-- (Phase 6) and the mart layer need:
--   * offer_to_band_ratio / offer_vs_band_delta -- how competitive the offer was
--   * is_below_band (offer < 90% of midpoint) -- the threshold Phase 1 baked the
--     decline signal around
--   * days_to_respond -- how long the candidate took to respond
--   * is_accepted / is_declined flags (expired is treated as a non-acceptance)

with offers as (
    select * from {{ ref('stg_offers') }}
),

final as (
    select
        offer_id,
        application_id,
        offer_date,
        response_date,
        offer_salary,
        band_midpoint,

        -- economics vs band midpoint
        (offer_salary - band_midpoint)                              as offer_vs_band_delta,
        case
            when band_midpoint is null or band_midpoint = 0 then null
            else round(offer_salary / band_midpoint, 4)
        end                                                         as offer_to_band_ratio,
        case
            when band_midpoint is null or band_midpoint = 0 then null
            else (offer_salary < 0.90 * band_midpoint)
        end                                                         as is_below_band,

        -- response timing
        (response_date - offer_date)                                as days_to_respond,

        -- outcome flags
        decision,
        (decision = 'accepted')                                     as is_accepted,
        (decision in ('declined', 'expired'))                       as is_declined

    from offers
)

select * from final

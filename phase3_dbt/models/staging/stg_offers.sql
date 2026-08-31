-- Staging: offers. 1:1 with raw.offers.

with source as (
    select * from {{ source('raw', 'offers') }}
),

renamed as (
    select
        cast(offer_id as varchar)                        as offer_id,
        cast(application_id as varchar)                  as application_id,
        cast(offer_date as date)                         as offer_date,
        cast(offer_salary as numeric)                    as offer_salary,
        cast(band_midpoint as numeric)                   as band_midpoint,
        cast(response_date as date)                      as response_date,
        lower(nullif(trim(decision), ''))                as decision
    from source
)

select * from renamed

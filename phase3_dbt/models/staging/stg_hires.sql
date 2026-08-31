-- Staging: hires. 1:1 with raw.hires.

with source as (
    select * from {{ source('raw', 'hires') }}
),

renamed as (
    select
        cast(hire_id as varchar)                         as hire_id,
        cast(offer_id as varchar)                        as offer_id,
        cast(start_date as date)                         as start_date
    from source
)

select * from renamed

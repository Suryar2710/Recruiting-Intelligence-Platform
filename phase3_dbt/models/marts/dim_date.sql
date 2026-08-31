-- dim_date
-- Grain: one row per calendar day.
-- Conformed date dimension spanning the full range of fact dates (applications,
-- requisitions opened, offers, hires) with a buffer. Built from a date spine so
-- every day exists even if no event landed on it -- required for correct
-- time-series rollups in BI. date_key is an integer YYYYMMDD surrogate, the
-- conventional grain-safe join key for date dimensions.

with spine as (
    {{ dbt_utils.date_spine(
        datepart="day",
        start_date="cast('2024-01-01' as date)",
        end_date="cast('2026-01-01' as date)"
    ) }}
),

dates as (
    select cast(date_day as date) as date_day
    from spine
)

select
    cast(to_char(date_day, 'YYYYMMDD') as integer)  as date_key,
    date_day,
    extract(year   from date_day)::int              as year,
    extract(quarter from date_day)::int             as quarter,
    extract(month  from date_day)::int              as month,
    to_char(date_day, 'Month')                      as month_name,
    extract(day    from date_day)::int              as day_of_month,
    extract(dow    from date_day)::int              as day_of_week,
    to_char(date_day, 'Day')                        as day_name,
    (extract(dow from date_day) in (0, 6))          as is_weekend,
    date_trunc('month', date_day)::date             as first_day_of_month
from dates

-- int_application_funnel
-- Grain: one row per application.
--
-- Collapses the per-stage stage_history rows into a single application-level row
-- with days spent in each stage, total days to decision, and drop-off flags.
--
-- Notes on the source data:
--   * stage_history records the stages an application entered/exited:
--     applied -> screen -> phone_interview -> onsite -> final.
--   * The "offer" step is not a stage_history row; reaching it is represented by
--     applications.current_stage = 'offer' and the offers table. So the funnel
--     timing here covers application -> final decision within the interview loop.
--   * was_dropped = the candidate withdrew (final_outcome = 'withdrawn'). That is
--     the candidate-initiated drop-off the drop-off model in Phase 6 predicts.
--     A rejection is a company-initiated decision, not a drop.
--   * drop_stage = the stage the application was sitting in at the drop
--     (applications.current_stage), null when the candidate did not drop.

with applications as (
    select * from {{ ref('stg_applications') }}
),

stage_history as (
    select * from {{ ref('stg_stage_history') }}
),

-- one row per application with each stage's duration pivoted into its own column
stage_durations as (
    select
        application_id,
        max(case when stage_name = 'applied'
            then (exited_date - entered_date) end)          as days_in_applied,
        max(case when stage_name = 'screen'
            then (exited_date - entered_date) end)          as days_in_screen,
        max(case when stage_name = 'phone_interview'
            then (exited_date - entered_date) end)          as days_in_phone_interview,
        max(case when stage_name = 'onsite'
            then (exited_date - entered_date) end)          as days_in_onsite,
        max(case when stage_name = 'final'
            then (exited_date - entered_date) end)          as days_in_final,
        -- funnel span from first entry to last exit across all recorded stages
        min(entered_date)                                   as first_stage_entered,
        max(exited_date)                                    as last_stage_exited,
        count(*)                                            as stages_recorded
    from stage_history
    group by application_id
),

final as (
    select
        a.application_id,
        a.applicant_id,
        a.req_id,
        a.applied_date,
        a.current_stage,
        a.final_outcome,

        -- per-stage durations (null where the application never reached the stage)
        sd.days_in_applied,
        sd.days_in_screen,
        sd.days_in_phone_interview,
        sd.days_in_onsite,
        sd.days_in_final,

        -- total days from application to the last recorded funnel event.
        -- coalesce the span endpoints to applied_date so applications with a
        -- single recorded stage still get 0 rather than null.
        (coalesce(sd.last_stage_exited, a.applied_date)
            - coalesce(sd.first_stage_entered, a.applied_date))  as total_days_to_decision,

        sd.stages_recorded,

        -- drop-off flags: a drop is a candidate withdrawal.
        (a.final_outcome = 'withdrawn')                          as was_dropped,
        case
            when a.final_outcome = 'withdrawn' then a.current_stage
            else null
        end                                                      as drop_stage

    from applications a
    left join stage_durations sd
        on a.application_id = sd.application_id
)

select * from final

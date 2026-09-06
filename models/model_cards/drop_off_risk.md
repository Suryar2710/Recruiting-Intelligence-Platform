# Model Card: Candidate Drop-off Risk

## Target variable
`label_was_dropped` from `marts.mart_ml_features` - a boolean that is True when a
candidate **withdrew** from the process before a decision (candidate-initiated
drop-off, distinct from a company rejection). Predicted as a probability in [0, 1].

## Features used
Candidate, requisition, and interview-activity ("process") features known while a
candidate is still active:

- **Candidate:** candidate_source, referral_flag, resume_years_experience
- **Requisition:** department, seniority_level, location, is_niche_location,
  recruiter_concurrent_open_reqs (recruiter's open-req load at application time)
- **Process:** interview_rounds_completed, avg_feedback_score, min_feedback_score,
  interviewer_count, max_gap_between_rounds_days, max_stage_dwell_days

**Deliberately excluded to prevent label leakage:** final_outcome, current_stage,
the offer-decline / time-to-fill labels, and offer-stage features. These encode or
follow the outcome and would inflate metrics unrealistically.

## Train/test split strategy
Single stratified hold-out split: 75% train / 25% test, `random_state=42`,
stratified on the target so the drop-off rate is preserved in both sets
(train n=90,000, test n=30,000). Class imbalance is handled with
`class_weight="balanced"`.

## Models compared
A **logistic-regression baseline** (with median imputation + standardization) vs a
**HistGradientBoostingClassifier** (NaN-aware trees). The model reported below is
the one with the higher test ROC AUC.

**Selected model: HistGradientBoosting**

## Final metrics (test set)
- **ROC AUC:** 0.6713
- **F1 (positive class = dropped):** 0.2518

## Top 5 feature importances (plain language)
Permutation importance on the test set (mean drop in ROC AUC when the feature is
shuffled):

1. **max_stage_dwell_days** - the longest time (in days) the candidate sat in any single funnel stage -- long stalls are the main drop-off signal (mean AUC drop 0.0953 when shuffled)
2. **interview_rounds_completed** - how many interview rounds the candidate has completed so far (mean AUC drop 0.0407 when shuffled)
3. **interviewer_count** - how many distinct interviewers the candidate has seen (mean AUC drop 0.0364 when shuffled)
4. **max_gap_between_rounds_days** - the longest gap (in days) between interview rounds (mean AUC drop 0.0040 when shuffled)
5. **avg_feedback_score** - the candidate's average interviewer feedback score (mean AUC drop 0.0013 when shuffled)

## Reproducibility
Models are trained **inside the Docker container** (Python 3.11, numpy 1.26.4),
which is the project's canonical environment -- the same one CI and
`docker compose up` reproduce. This matters because the synthetic data generator
is sensitive to the numpy major version: the same fixed seed produces a different
dataset under numpy 2.x than under 1.26.x (a seed fixes the bit-stream, not the
output across numpy versions). Training on a host machine with a different numpy
version is not guaranteed to reproduce these exact numbers. numpy is pinned to
`1.26.4` in requirements.txt to prevent silent drift.

## Limitations
- **Moderate AUC reflects a genuinely noisy signal.** In the synthetic generator,
  drop-off is a per-stage event with a modest coefficient and substantial noise,
  so it is inherently harder to separate than the time-to-fill or offer-decline
  outcomes. Gradient-boosted trees edge out logistic regression here (they capture
  the non-linear interaction between funnel depth and stage dwell), but the gap is
  small -- the achievable AUC is bounded more by the data's signal-to-noise than
  by model choice. A much higher number would indicate leakage, not skill.
- **Synthetic data.** Trained on generated data with hand-designed correlations,
  not real hiring outcomes. Patterns reflect the generator's assumptions, so
  metrics do not transfer to production without retraining on real data.
- **Class imbalance.** The positive (dropped) class is only ~10.0% of
  applications. Metrics are reported with this in mind; a naive accuracy figure
  would be misleading, which is why AUC/F1 are used and class weighting is applied.
- **Not validated against a real company's process.** No external or temporal
  validation; the split is a random hold-out on one synthetic snapshot.
- **Process-feature assumption.** Interview-activity features represent
  "activity so far." In a live setting these must be computed as-of the scoring
  time to avoid leaking future events; here they are taken from the completed
  (synthetic) record.

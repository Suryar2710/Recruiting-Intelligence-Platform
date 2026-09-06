# Model Card: Requisition Time-to-Fill Risk

## Target variable
`label_missed_target` from `marts.mart_ml_features` - a boolean that is True when
a requisition's actual time-to-fill exceeded its target. Non-null only for the
10,836 hired applications (rows where a fill actually happened).
Predicted as a probability in [0, 1].

## Features used
Requisition-grain features **known when a req opens** (so the model can score
still-open reqs at prediction time):

- **Requisition:** department, seniority_level, location, is_niche_location,
  recruiter_concurrent_open_reqs (recruiter's open-req load at application time)

**Deliberately excluded to prevent leakage / grain mismatch:** all interview and
process features, all offer features, candidate-level features
(resume_years_experience, source), current_stage, final_outcome, and the other
two labels. These do not exist when a requisition is still open, so using them
would train the model on information unavailable at scoring time.

## Train/test split strategy
Single stratified hold-out split: 75% train / 25% test, `random_state=42`,
stratified on the target (train n=8,127, test n=2,709). The label is
reasonably balanced (~42% positive), so no class weighting is required,
though it is applied for consistency with the drop-off model.

## Models compared
A **logistic-regression baseline** (imputation + standardization) vs a
**HistGradientBoostingClassifier** (NaN-aware trees). Reported model is the one
with the higher test ROC AUC.

**Selected model: HistGradientBoosting**

## Final metrics (test set)
- **ROC AUC:** 0.7069
- **F1 (positive class = missed target):** 0.6091

## Top 5 feature importances (plain language)
Permutation importance on the test set (mean drop in ROC AUC when the feature is
shuffled):

1. **is_niche_location** - whether the role is in a thin talent-market location -- by far the dominant driver of missing the target (mean AUC drop 0.1652 when shuffled)
2. **recruiter_concurrent_open_reqs** - how many other reqs the recruiter had open at the time (recruiter load) (mean AUC drop 0.0304 when shuffled)
3. **department** - the hiring department (mean AUC drop 0.0283 when shuffled)
4. **location** - the role location (niche markets fill much slower) (mean AUC drop 0.0220 when shuffled)
5. **seniority_level** - the seniority of the role (mean AUC drop 0.0176 when shuffled)

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
- **Synthetic data.** Trained on generated data with hand-designed correlations,
  not real hiring outcomes. In the generator, senior + niche-location roles
  reliably blow past target; the model largely recovers that rule. Metrics will
  not transfer to production without retraining on real data.
- **Label available only for filled reqs.** The target exists only where a hire
  occurred, so the model is trained on filled requisitions and applied to open
  ones. This is the standard survivorship caveat for time-to-fill models; a
  production version would also model still-open reqs via survival analysis
  (censoring), not just a binary hit/miss on completed fills.
- **Not validated against a real company's process.** No external or temporal
  validation; the split is a random hold-out on one synthetic snapshot.
- **Dominant single feature.** Performance is driven mostly by niche-location.
  On real data the driver mix would differ, and a single-feature-dominated model
  is brittle to distribution shift in that feature.

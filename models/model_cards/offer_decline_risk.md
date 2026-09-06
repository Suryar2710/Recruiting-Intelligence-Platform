# Model Card: Offer Decline Risk

## Target variable
`label_offer_declined` from `marts.mart_ml_features` - a boolean that is True when
an extended offer was **declined or expired** (i.e. not accepted). Non-null only
for the 16,209 applications that reached the offer stage. Predicted as
a probability in [0, 1].

## Scoring context
This model is designed to score **at the moment an offer is extended**, using only
information available at that time. It answers the question: "given this offer's
economics and the candidate/requisition profile, how likely is the candidate to
decline?"

## Features used
All features are known at offer-extension time:

- **Offer economics:** offer_to_band_ratio (offer salary / band midpoint)
- **Candidate:** referral_flag, resume_years_experience, candidate_source
- **Requisition context:** department, seniority_level, location, is_niche_location

**Deliberately excluded:**
- **days_to_respond** — only known after the candidate responds, so including it
  would violate the at-offer-extension scoring contract. Its permutation importance
  was negligible (0.0010) in the version that included it, so removing it costs
  nothing.
- Interview/process features, current_stage, final_outcome, and the time-to-fill
  and drop-off labels.

## Train/test split strategy
Single stratified hold-out split: 75% train / 25% test, `random_state=42`,
stratified on the target (train n=12,156, test n=4,053). Label is
reasonably balanced (~33% positive); class weighting applied for
consistency with the other models.

## Models compared
A **logistic-regression baseline** (imputation + standardization) vs a
**HistGradientBoostingClassifier**. Reported model is the higher test ROC AUC.

**Selected model: LogisticRegression**

## Final metrics (test set)
- **ROC AUC:** 0.6413
- **F1 (positive class = declined):** 0.4988

## Top 5 feature importances (plain language)
Permutation importance on the test set (mean drop in ROC AUC when the feature is
shuffled):

1. **offer_to_band_ratio** - how the offer compares to the salary band midpoint -- offers below ~90% of midpoint are declined far more often (mean AUC drop 0.0837 when shuffled)
2. **candidate_source** - the acquisition channel the candidate came from (mean AUC drop 0.0188 when shuffled)
3. **referral_flag** - whether the candidate was a referral -- referrals accept notably more often (mean AUC drop 0.0109 when shuffled)
4. **location** - the role location (mean AUC drop 0.0018 when shuffled)
5. **seniority_level** - the seniority of the role (mean AUC drop 0.0007 when shuffled)

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
  not real hiring outcomes. The generator makes below-band offers decline more and
  referrals accept more; the model recovers those rules. Metrics will not transfer
  to production without retraining on real data.
- **Class imbalance.** The declined class is ~33% of offers - milder than
  the drop-off model but still handled with class weighting; AUC/F1 reported rather
  than accuracy.
- **Not validated against a real company's process.** No external or temporal
  validation; the split is a random hold-out on one synthetic snapshot.

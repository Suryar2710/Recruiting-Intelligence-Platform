"""
Phase 6 - Model 2: Candidate drop-off risk.

Predicts the probability that an in-pipeline candidate ABANDONS (withdraws)
before a decision. Target: mart_ml_features.label_was_dropped (True = the
candidate withdrew).

Modeling notes
--------------
Grain: one row per application (all 120k rows have this label -- drop-off is
defined for every candidate, unlike the offer/time-to-fill labels which only
exist at later funnel stages).

Leakage control (important): we EXCLUDE columns that encode the outcome:
  - final_outcome (literally contains 'withdrawn'),
  - current_stage (where the funnel stopped),
  - the offer-decline and time-to-fill labels.
We keep candidate, requisition, and process (interview-activity) features. The
process features represent "activity so far", which is the information a real
drop-off model would have when scoring a still-active candidate. This
assumption is called out in the model card.

Approach: a logistic-regression baseline first, then a gradient-boosted trees
model (HistGradientBoostingClassifier -- handles NaNs natively, so no imputation
needed for the tree model and no extra XGBoost/LightGBM dependency), and we
compare. Metrics: ROC AUC and F1 (positive class = dropped). Feature importance
via permutation importance on the test set. Trained model saved with joblib.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sqlalchemy as sa

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "models")
CARD_DIR = os.path.join(HERE, "..", "models", "model_cards")
RANDOM_STATE = 42
TARGET = "label_was_dropped"

# Features grouped by type. Deliberately excludes leakage columns.
NUMERIC_FEATURES = [
    "resume_years_experience",
    "interview_rounds_completed",
    "avg_feedback_score",
    "min_feedback_score",
    "interviewer_count",
    "max_gap_between_rounds_days",
    "max_stage_dwell_days",
    "recruiter_concurrent_open_reqs",
]
CATEGORICAL_FEATURES = [
    "candidate_source",
    "department",
    "seniority_level",
    "location",
]
BOOLEAN_FEATURES = ["referral_flag", "is_niche_location"]

# Columns explicitly NOT used (leakage or wrong grain).
LEAKAGE_EXCLUDED = [
    "final_outcome", "current_stage", "reached_offer_stage",
    "offer_to_band_ratio", "days_to_respond",
    "label_missed_target", "label_offer_declined",
    "application_id", "applied_date",
]


def pg_url() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5433")
    user = os.getenv("PGUSER", "recruiting")
    password = os.getenv("PGPASSWORD", "recruiting")
    db = os.getenv("PGDATABASE", "recruiting_intel")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def load_data() -> pd.DataFrame:
    engine = sa.create_engine(pg_url())
    df = pd.read_sql("select * from marts.mart_ml_features", engine)
    engine.dispose()
    return df


def build_preprocessor(for_linear: bool) -> ColumnTransformer:
    """Linear model needs scaling + imputation; tree model handles raw + NaN."""
    if for_linear:
        from sklearn.impute import SimpleImputer
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical = OneHotEncoder(handle_unknown="ignore")
        return ColumnTransformer([
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ])
    # tree model: one-hot categoricals, pass numerics through (NaN-aware)
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("bool", "passthrough", BOOLEAN_FEATURES),
    ])


def evaluate(name, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    print(f"\n[{name}]  ROC AUC = {auc:.4f}   F1 (dropped) = {f1:.4f}")
    print(classification_report(y_test, pred, target_names=["stayed", "dropped"], digits=3))
    return {"auc": auc, "f1": f1}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CARD_DIR, exist_ok=True)

    print("Loading feature mart ...")
    df = load_data()
    print(f"  {len(df):,} rows")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy()
    # booleans -> ints for sklearn
    for b in BOOLEAN_FEATURES:
        X[b] = X[b].astype(int)
    y = df[TARGET].astype(int)

    pos_rate = y.mean()
    print(f"  positive (dropped) rate = {pos_rate:.3%}  (class imbalance)")

    # stratified split preserves the drop-off rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train={len(X_train):,}  test={len(X_test):,}")

    # ---- baseline: logistic regression ----
    logit = Pipeline([
        ("prep", build_preprocessor(for_linear=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logit.fit(X_train, y_train)
    logit_metrics = evaluate("baseline logistic regression", logit, X_test, y_test)

    # ---- gradient-boosted trees ----
    hgb = Pipeline([
        ("prep", build_preprocessor(for_linear=False)),
        ("clf", HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, max_iter=300, learning_rate=0.08,
            class_weight="balanced",
        )),
    ])
    hgb.fit(X_train, y_train)
    hgb_metrics = evaluate("HistGradientBoosting", hgb, X_test, y_test)

    # ---- pick winner by AUC ----
    if hgb_metrics["auc"] >= logit_metrics["auc"]:
        best_name, best_model, best_metrics = "HistGradientBoosting", hgb, hgb_metrics
    else:
        best_name, best_model, best_metrics = "LogisticRegression", logit, logit_metrics
    print(f"\nSelected model: {best_name} (higher test AUC)")

    # ---- feature importance (permutation on the test set) ----
    print("\nComputing permutation importances (this takes a moment) ...")
    feat_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    perm = permutation_importance(
        best_model, X_test, y_test, scoring="roc_auc",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importances = sorted(
        zip(feat_names, perm.importances_mean, perm.importances_std),
        key=lambda t: t[1], reverse=True,
    )
    print("\nTop feature importances (mean AUC drop when shuffled):")
    for name, mean, std in importances[:10]:
        print(f"  {name:32s} {mean:.4f} +/- {std:.4f}")

    # ---- persist model ----
    model_path = os.path.join(MODEL_DIR, "drop_off_risk.joblib")
    joblib.dump(
        {"model": best_model, "features": feat_names,
         "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
         "boolean": BOOLEAN_FEATURES, "target": TARGET},
        model_path,
    )
    print(f"\nSaved model -> {model_path}")

    # ---- example predictions for interpretability ----
    print("\nExample predictions (feature values -> predicted drop-off prob):")
    ex = X_test.head(4).copy()
    ex_proba = best_model.predict_proba(ex)[:, 1]
    for i, (_, row) in enumerate(ex.iterrows()):
        print(f"  #{i+1} src={row['candidate_source']}, referral={bool(row['referral_flag'])}, "
              f"rounds={row['interview_rounds_completed']}, "
              f"max_gap={row['max_gap_between_rounds_days']}, "
              f"avg_fb={row['avg_feedback_score']} -> P(drop)={ex_proba[i]:.3f}  "
              f"(actual={'dropped' if y_test.iloc[i] else 'stayed'})")

    write_model_card(best_name, best_metrics, importances, pos_rate, len(X_train), len(X_test))
    print(f"\nWrote model card -> {os.path.join(CARD_DIR, 'drop_off_risk.md')}")


def _plain_language(feature: str) -> str:
    m = {
        "max_gap_between_rounds_days": "the longest gap (in days) between interview rounds",
        "max_stage_dwell_days": "the longest time (in days) the candidate sat in any single funnel stage -- long stalls are the main drop-off signal",
        "interview_rounds_completed": "how many interview rounds the candidate has completed so far",
        "avg_feedback_score": "the candidate's average interviewer feedback score",
        "min_feedback_score": "the candidate's lowest interviewer feedback score",
        "interviewer_count": "how many distinct interviewers the candidate has seen",
        "recruiter_concurrent_open_reqs": "how many other reqs the recruiter had open at application time (recruiter load)",
        "resume_years_experience": "the candidate's years of experience",
        "referral_flag": "whether the candidate came in as a referral",
        "candidate_source": "the acquisition channel the candidate came from",
        "seniority_level": "the seniority of the role",
        "department": "the hiring department",
        "location": "the role location",
        "is_niche_location": "whether the role is in a thin talent-market location",
    }
    return m.get(feature, feature)


def write_model_card(model_name, metrics, importances, pos_rate, n_train, n_test):
    top5 = importances[:5]
    top5_md = "\n".join(
        f"{i+1}. **{name}** - {_plain_language(name)} "
        f"(mean AUC drop {mean:.4f} when shuffled)"
        for i, (name, mean, std) in enumerate(top5)
    )
    card = f"""# Model Card: Candidate Drop-off Risk

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
(train n={n_train:,}, test n={n_test:,}). Class imbalance is handled with
`class_weight="balanced"`.

## Models compared
A **logistic-regression baseline** (with median imputation + standardization) vs a
**HistGradientBoostingClassifier** (NaN-aware trees). The model reported below is
the one with the higher test ROC AUC.

**Selected model: {model_name}**

## Final metrics (test set)
- **ROC AUC:** {metrics['auc']:.4f}
- **F1 (positive class = dropped):** {metrics['f1']:.4f}

## Top 5 feature importances (plain language)
Permutation importance on the test set (mean drop in ROC AUC when the feature is
shuffled):

{top5_md}

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
- **Class imbalance.** The positive (dropped) class is only ~{pos_rate:.1%} of
  applications. Metrics are reported with this in mind; a naive accuracy figure
  would be misleading, which is why AUC/F1 are used and class weighting is applied.
- **Not validated against a real company's process.** No external or temporal
  validation; the split is a random hold-out on one synthetic snapshot.
- **Process-feature assumption.** Interview-activity features represent
  "activity so far." In a live setting these must be computed as-of the scoring
  time to avoid leaking future events; here they are taken from the completed
  (synthetic) record.
"""
    with open(os.path.join(CARD_DIR, "drop_off_risk.md"), "w") as f:
        f.write(card)


if __name__ == "__main__":
    main()

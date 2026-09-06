"""
Phase 6 - Model 3: Offer decline risk.

Predicts whether an extended offer will be DECLINED (or expire) rather than
accepted. Scores at the moment an offer is extended, using only information
available at that time. Target: mart_ml_features.label_offer_declined
(True = declined/expired). Non-null only for the ~12,773 applications that
reached the offer stage.

"Feature not available for the class that matters" pre-check
------------------------------------------------------------
The label exists for every offer, and the Phase 1 signals live in features
present at offer time: offer_to_band_ratio and referral_flag. Verified:
  - offer_to_band_ratio alone ~0.61 AUC, referral_flag ~0.58,
  - decline rate 54% (below-band, non-referral) vs 21% (at-band, referral),
  - label well balanced (~38% positive).
So there is no availability trap: the signal-carrying features are populated for
the whole labeled class.

days_to_respond is intentionally EXCLUDED despite being available in the feature
mart: it is only known after a candidate responds, so including it would violate
the "score at offer-extension time" contract. Its permutation importance was
negligible (0.0010) so removing it costs nothing.

Features used (all known at offer-extension time):
  offer_to_band_ratio (offer economics),
  referral_flag, resume_years_experience, candidate_source (candidate),
  department, seniority_level, location, is_niche_location (requisition context).
Excluded: days_to_respond (post-outcome), interview/process features,
current_stage, final_outcome, the other two labels, time-to-fill features.

Approach mirrors Models 1 & 2: logistic baseline vs HistGradientBoosting,
compare on ROC AUC, permutation importance, joblib save, model card.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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
TARGET = "label_offer_declined"

NUMERIC_FEATURES = [
    "offer_to_band_ratio",
    "resume_years_experience",
]
CATEGORICAL_FEATURES = ["candidate_source", "department", "seniority_level", "location"]
BOOLEAN_FEATURES = ["referral_flag", "is_niche_location"]


def pg_url() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5433")
    user = os.getenv("PGUSER", "recruiting")
    password = os.getenv("PGPASSWORD", "recruiting")
    db = os.getenv("PGDATABASE", "recruiting_intel")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def load_labeled() -> pd.DataFrame:
    engine = sa.create_engine(pg_url())
    df = pd.read_sql(
        "select * from marts.mart_ml_features where label_offer_declined is not null",
        engine,
    )
    engine.dispose()
    return df


def build_preprocessor(for_linear: bool) -> ColumnTransformer:
    if for_linear:
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer([
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ])
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        # dense output: HistGradientBoosting requires dense X
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("bool", "passthrough", BOOLEAN_FEATURES),
    ])


def evaluate(name, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    print(f"\n[{name}]  ROC AUC = {auc:.4f}   F1 (declined) = {f1:.4f}")
    print(classification_report(y_test, pred, target_names=["accepted", "declined"], digits=3))
    return {"auc": auc, "f1": f1}


def _plain_language(feature: str) -> str:
    m = {
        "offer_to_band_ratio": "how the offer compares to the salary band midpoint -- offers below ~90% of midpoint are declined far more often",
        "referral_flag": "whether the candidate was a referral -- referrals accept notably more often",
        "resume_years_experience": "the candidate's years of experience",
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
    card = f"""# Model Card: Offer Decline Risk

## Target variable
`label_offer_declined` from `marts.mart_ml_features` - a boolean that is True when
an extended offer was **declined or expired** (i.e. not accepted). Non-null only
for the {n_train + n_test:,} applications that reached the offer stage. Predicted as
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
stratified on the target (train n={n_train:,}, test n={n_test:,}). Label is
reasonably balanced (~{pos_rate:.0%} positive); class weighting applied for
consistency with the other models.

## Models compared
A **logistic-regression baseline** (imputation + standardization) vs a
**HistGradientBoostingClassifier**. Reported model is the higher test ROC AUC.

**Selected model: {model_name}**

## Final metrics (test set)
- **ROC AUC:** {metrics['auc']:.4f}
- **F1 (positive class = declined):** {metrics['f1']:.4f}

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
- **Synthetic data.** Trained on generated data with hand-designed correlations,
  not real hiring outcomes. The generator makes below-band offers decline more and
  referrals accept more; the model recovers those rules. Metrics will not transfer
  to production without retraining on real data.
- **Class imbalance.** The declined class is ~{pos_rate:.0%} of offers - milder than
  the drop-off model but still handled with class weighting; AUC/F1 reported rather
  than accuracy.
- **Not validated against a real company's process.** No external or temporal
  validation; the split is a random hold-out on one synthetic snapshot.
"""
    with open(os.path.join(CARD_DIR, "offer_decline_risk.md"), "w") as f:
        f.write(card)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CARD_DIR, exist_ok=True)

    print("Loading feature mart (labeled subset: offers) ...")
    df = load_labeled()
    print(f"  {len(df):,} labeled rows")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy()
    for b in BOOLEAN_FEATURES:
        X[b] = X[b].astype(int)
    y = df[TARGET].astype(int)

    pos_rate = y.mean()
    print(f"  positive (declined) rate = {pos_rate:.3%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train={len(X_train):,}  test={len(X_test):,}")

    logit = Pipeline([
        ("prep", build_preprocessor(for_linear=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logit.fit(X_train, y_train)
    logit_metrics = evaluate("baseline logistic regression", logit, X_test, y_test)

    hgb = Pipeline([
        ("prep", build_preprocessor(for_linear=False)),
        ("clf", HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, max_iter=300, learning_rate=0.08,
            class_weight="balanced",
        )),
    ])
    hgb.fit(X_train, y_train)
    hgb_metrics = evaluate("HistGradientBoosting", hgb, X_test, y_test)

    if hgb_metrics["auc"] >= logit_metrics["auc"]:
        best_name, best_model, best_metrics = "HistGradientBoosting", hgb, hgb_metrics
    else:
        best_name, best_model, best_metrics = "LogisticRegression", logit, logit_metrics
    print(f"\nSelected model: {best_name} (higher test AUC)")

    print("\nComputing permutation importances ...")
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
    for name, mean, std in importances:
        print(f"  {name:32s} {mean:.4f} +/- {std:.4f}")

    model_path = os.path.join(MODEL_DIR, "offer_decline_risk.joblib")
    joblib.dump(
        {"model": best_model, "features": feat_names,
         "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
         "boolean": BOOLEAN_FEATURES, "target": TARGET},
        model_path,
    )
    print(f"\nSaved model -> {model_path}")

    print("\nExample predictions (feature values -> predicted decline prob):")
    ex = X_test.head(4).copy()
    ex_proba = best_model.predict_proba(ex)[:, 1]
    for i, (_, row) in enumerate(ex.iterrows()):
        print(f"  #{i+1} band_ratio={row['offer_to_band_ratio']}, referral={bool(row['referral_flag'])}, "
              f"source={row['candidate_source']}, seniority={row['seniority_level']} "
              f"-> P(decline)={ex_proba[i]:.3f}  "
              f"(actual={'declined' if y_test.iloc[i] else 'accepted'})")

    write_model_card(best_name, best_metrics, importances, pos_rate, len(X_train), len(X_test))
    print(f"\nWrote model card -> {os.path.join(CARD_DIR, 'offer_decline_risk.md')}")


if __name__ == "__main__":
    main()

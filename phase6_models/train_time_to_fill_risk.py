"""
Phase 6 - Model 1: Requisition time-to-fill risk.

Predicts whether a requisition will MISS its time-to-fill target.
Target: mart_ml_features.label_missed_target (True = actual fill time exceeded
the target). Non-null only for the 7,957 hired applications (a fill happened).

The "feature not available for the class that matters" trap (checked first)
--------------------------------------------------------------------------
This model's job is to score requisitions that are STILL OPEN (no hire yet).
So it must only use features that exist WHEN A REQ OPENS -- requisition-grain
attributes -- not candidate/interview/offer features that only materialize
later in the funnel. Training on those would be leakage: the model would learn
from information unavailable at scoring time.

We verified (pre-check) that the requisition-grain signal is strong and well
distributed across the labeled class:
  - is_niche_location alone reaches ~0.68 ROC AUC,
  - miss rate ~30% (non-niche) vs ~65-72% (niche) across all seniorities,
  - label is well balanced (~44% positive),
so unlike the drop-off model there is no signal-availability trap here.

Features used (requisition-grain, known at open time):
  department, seniority_level, location, is_niche_location,
  recruiter_concurrent_open_reqs.
Explicitly excluded (leakage / wrong grain): all interview/process features,
all offer features, candidate-level features, current_stage, final_outcome,
and the other two labels.

Approach mirrors Model 2: logistic-regression baseline vs HistGradientBoosting,
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
TARGET = "label_missed_target"

# Requisition-grain features only -- known when a req opens.
NUMERIC_FEATURES = ["recruiter_concurrent_open_reqs"]
CATEGORICAL_FEATURES = ["department", "seniority_level", "location"]
BOOLEAN_FEATURES = ["is_niche_location"]


def pg_url() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5433")
    user = os.getenv("PGUSER", "recruiting")
    password = os.getenv("PGPASSWORD", "recruiting")
    db = os.getenv("PGDATABASE", "recruiting_intel")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def load_labeled() -> pd.DataFrame:
    engine = sa.create_engine(pg_url())
    # only rows with a time-to-fill outcome (hired applications)
    df = pd.read_sql(
        "select * from marts.mart_ml_features where label_missed_target is not null",
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
    print(f"\n[{name}]  ROC AUC = {auc:.4f}   F1 (missed) = {f1:.4f}")
    print(classification_report(y_test, pred, target_names=["hit_target", "missed_target"], digits=3))
    return {"auc": auc, "f1": f1}


def _plain_language(feature: str) -> str:
    m = {
        "is_niche_location": "whether the role is in a thin talent-market location -- by far the dominant driver of missing the target",
        "location": "the role location (niche markets fill much slower)",
        "seniority_level": "the seniority of the role",
        "department": "the hiring department",
        "recruiter_concurrent_open_reqs": "how many other reqs the recruiter had open at the time (recruiter load)",
    }
    return m.get(feature, feature)


def write_model_card(model_name, metrics, importances, pos_rate, n_train, n_test):
    top5 = importances[:5]
    top5_md = "\n".join(
        f"{i+1}. **{name}** - {_plain_language(name)} "
        f"(mean AUC drop {mean:.4f} when shuffled)"
        for i, (name, mean, std) in enumerate(top5)
    )
    card = f"""# Model Card: Requisition Time-to-Fill Risk

## Target variable
`label_missed_target` from `marts.mart_ml_features` - a boolean that is True when
a requisition's actual time-to-fill exceeded its target. Non-null only for the
{n_train + n_test:,} hired applications (rows where a fill actually happened).
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
stratified on the target (train n={n_train:,}, test n={n_test:,}). The label is
reasonably balanced (~{pos_rate:.0%} positive), so no class weighting is required,
though it is applied for consistency with the drop-off model.

## Models compared
A **logistic-regression baseline** (imputation + standardization) vs a
**HistGradientBoostingClassifier** (NaN-aware trees). Reported model is the one
with the higher test ROC AUC.

**Selected model: {model_name}**

## Final metrics (test set)
- **ROC AUC:** {metrics['auc']:.4f}
- **F1 (positive class = missed target):** {metrics['f1']:.4f}

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
"""
    with open(os.path.join(CARD_DIR, "time_to_fill_risk.md"), "w") as f:
        f.write(card)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CARD_DIR, exist_ok=True)

    print("Loading feature mart (labeled subset: hired applications) ...")
    df = load_labeled()
    print(f"  {len(df):,} labeled rows")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES].copy()
    for b in BOOLEAN_FEATURES:
        X[b] = X[b].astype(int)
    y = df[TARGET].astype(int)

    pos_rate = y.mean()
    print(f"  positive (missed target) rate = {pos_rate:.3%}")

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

    model_path = os.path.join(MODEL_DIR, "time_to_fill_risk.joblib")
    joblib.dump(
        {"model": best_model, "features": feat_names,
         "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
         "boolean": BOOLEAN_FEATURES, "target": TARGET},
        model_path,
    )
    print(f"\nSaved model -> {model_path}")

    print("\nExample predictions (feature values -> predicted miss-target prob):")
    ex = X_test.head(4).copy()
    ex_proba = best_model.predict_proba(ex)[:, 1]
    for i, (_, row) in enumerate(ex.iterrows()):
        print(f"  #{i+1} dept={row['department']}, seniority={row['seniority_level']}, "
              f"loc={row['location']}, niche={bool(row['is_niche_location'])}, "
              f"rec_load={row['recruiter_concurrent_open_reqs']} "
              f"-> P(miss)={ex_proba[i]:.3f}  "
              f"(actual={'missed' if y_test.iloc[i] else 'hit'})")

    write_model_card(best_name, best_metrics, importances, pos_rate, len(X_train), len(X_test))
    print(f"\nWrote model card -> {os.path.join(CARD_DIR, 'time_to_fill_risk.md')}")


if __name__ == "__main__":
    main()

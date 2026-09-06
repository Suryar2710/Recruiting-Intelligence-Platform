"""
Phase 7: FastAPI service exposing predictions from the three Phase 6 models
and KPI summary from the dbt marts.

Endpoints:
  GET /requisitions/at-risk     — unfilled reqs ranked by predicted fill-risk score
  GET /applications/{id}/dropout-risk — single application drop-off prediction
  GET /offers/{id}/decline-risk       — single offer decline prediction
  GET /kpis/summary                   — aggregate KPIs from the mart layer

Run locally:
    cd phase7_api
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from decimal import Decimal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sqlalchemy as sa


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def pg_url() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5433")
    user = os.getenv("PGUSER", "recruiting")
    password = os.getenv("PGPASSWORD", "recruiting")
    db = os.getenv("PGDATABASE", "recruiting_intel")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


# --------------------------------------------------------------------------- #
# Globals loaded at startup
# --------------------------------------------------------------------------- #
_models: dict = {}
_engine: sa.Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and create DB engine once at startup, dispose at shutdown."""
    global _models, _engine
    for name in ("time_to_fill_risk", "drop_off_risk", "offer_decline_risk"):
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if not os.path.exists(path):
            raise RuntimeError(f"Model not found: {path}. Train it first (Phase 6).")
        _models[name] = joblib.load(path)
    _engine = sa.create_engine(pg_url(), pool_pre_ping=True)
    yield
    if _engine:
        _engine.dispose()


app = FastAPI(
    title="Recruiting Intelligence Platform",
    description="Predictions and KPIs from the recruiting funnel data pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _coerce_serializable(val):
    """Convert numpy/pandas/Decimal types to plain Python for JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, Decimal)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


def _predict_one(model_key: str, features_df: pd.DataFrame) -> float:
    """Run a single-row prediction through a loaded model pipeline."""
    m = _models[model_key]
    pipeline = m["model"]
    bools = m.get("boolean", [])
    for b in bools:
        if b in features_df.columns:
            features_df[b] = features_df[b].astype(int)
    proba = pipeline.predict_proba(features_df)[:, 1]
    return round(float(proba[0]), 4)


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #
class ReqAtRisk(BaseModel):
    req_id: str
    department: str
    seniority_level: str
    location: str
    is_niche_location: bool
    target_time_to_fill_days: int
    recruiter_concurrent_open_reqs: int
    fill_risk_score: float


class DropoutRiskResponse(BaseModel):
    application_id: str
    dropout_risk_score: float
    features: dict


class DeclineRiskResponse(BaseModel):
    offer_id: str
    application_id: str
    decline_risk_score: float
    features: dict


class KPISummary(BaseModel):
    total_applications: int
    total_offers: int
    total_hires: int
    total_requisitions: int
    overall_hire_rate: float
    overall_offer_accept_rate: float
    overall_dropout_rate: float
    avg_time_to_fill_days: float | None
    target_miss_rate: float | None
    referral_accept_rate: float | None
    non_referral_accept_rate: float | None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/requisitions/at-risk", response_model=list[ReqAtRisk])
def requisitions_at_risk(limit: int = 20):
    """
    List unfilled (open) requisitions ranked by predicted fill-risk score.
    Uses Model 1 (time-to-fill risk) to score each unfilled req.
    """
    with _engine.connect() as conn:
        # Pull unfilled reqs with their features from fact_requisitions
        df = pd.read_sql(sa.text("""
            select
                fr.req_id,
                fr.department,
                fr.seniority_level,
                fr.location,
                (dr.is_niche_location)::int as is_niche_location,
                fr.target_time_to_fill_days,
                -- recruiter load: total other reqs owned by this recruiter
                -- (approximation available at query time)
                coalesce(rc.concurrent_load, 0) as recruiter_concurrent_open_reqs
            from marts.fact_requisitions fr
            join marts.dim_requisition dr on fr.req_id = dr.req_id
            left join (
                select recruiter_id, count(*) - 1 as concurrent_load
                from staging.stg_requisitions
                group by recruiter_id
            ) rc on fr.recruiter_id = rc.recruiter_id
            where not fr.is_filled
        """), conn)

    if df.empty:
        return []

    m = _models["time_to_fill_risk"]
    feat_cols = m["numeric"] + m["categorical"] + m["boolean"]
    X = df[feat_cols].copy()
    for b in m["boolean"]:
        X[b] = X[b].astype(int)
    probas = m["model"].predict_proba(X)[:, 1]
    df["fill_risk_score"] = np.round(probas, 4)
    df = df.sort_values("fill_risk_score", ascending=False).head(limit)
    df["is_niche_location"] = df["is_niche_location"].astype(bool)

    return [
        ReqAtRisk(**{k: _coerce_serializable(v) for k, v in row.items()})
        for _, row in df.iterrows()
    ]


@app.get("/applications/{application_id}/dropout-risk", response_model=DropoutRiskResponse)
def application_dropout_risk(application_id: str):
    """
    Predict the drop-off (withdrawal) probability for a single application.
    Uses Model 2 (candidate drop-off risk).
    """
    m = _models["drop_off_risk"]
    feat_cols = m["numeric"] + m["categorical"] + m["boolean"]

    with _engine.connect() as conn:
        df = pd.read_sql(
            sa.text("select * from marts.mart_ml_features where application_id = :aid"),
            conn,
            params={"aid": application_id},
        )

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Application {application_id} not found.")

    X = df[feat_cols].copy()
    score = _predict_one("drop_off_risk", X)

    return DropoutRiskResponse(
        application_id=application_id,
        dropout_risk_score=score,
        features={k: _coerce_serializable(df.iloc[0][k]) for k in feat_cols},
    )


@app.get("/offers/{offer_id}/decline-risk", response_model=DeclineRiskResponse)
def offer_decline_risk(offer_id: str):
    """
    Predict the decline probability for a single offer.
    Uses Model 3 (offer decline risk). Scores using information available at
    offer-extension time only.
    """
    m = _models["offer_decline_risk"]
    feat_cols = m["numeric"] + m["categorical"] + m["boolean"]

    with _engine.connect() as conn:
        # offer -> application -> features
        df = pd.read_sql(sa.text("""
            select fo.offer_id, mf.*
            from marts.fact_offers fo
            join marts.mart_ml_features mf on fo.application_id = mf.application_id
            where fo.offer_id = :oid
        """), conn, params={"oid": offer_id})

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Offer {offer_id} not found.")

    X = df[feat_cols].copy()
    score = _predict_one("offer_decline_risk", X)

    return DeclineRiskResponse(
        offer_id=offer_id,
        application_id=str(df.iloc[0]["application_id"]),
        decline_risk_score=score,
        features={k: _coerce_serializable(df.iloc[0][k]) for k in feat_cols},
    )


@app.get("/kpis/summary", response_model=KPISummary)
def kpis_summary():
    """
    Aggregate KPIs from the mart layer, returned as JSON. Designed for a BI
    dashboard landing page or an interview demo.
    """
    with _engine.connect() as conn:
        r = conn.execute(sa.text("""
            select
                (select count(*) from marts.fact_applications)                    as total_applications,
                (select count(*) from marts.fact_offers)                          as total_offers,
                (select count(*) from marts.fact_applications
                 where final_outcome = 'hired')                                   as total_hires,
                (select count(*) from marts.fact_requisitions)                    as total_requisitions,

                -- rates
                (select round(100.0 * count(*) filter (where final_outcome = 'hired')
                              / nullif(count(*), 0), 2)
                 from marts.fact_applications)                                    as overall_hire_rate,

                (select round(100.0 * count(*) filter (where is_accepted)
                              / nullif(count(*), 0), 2)
                 from marts.fact_offers)                                          as overall_offer_accept_rate,

                (select round(100.0 * count(*) filter (where was_dropped)
                              / nullif(count(*), 0), 2)
                 from marts.fact_applications)                                    as overall_dropout_rate,

                -- time-to-fill (from hired applications)
                (select round(avg(actual_time_to_fill_days)::numeric, 1)
                 from marts.fact_applications
                 where actual_time_to_fill_days is not null)                      as avg_time_to_fill_days,

                (select round(100.0 * count(*) filter (where missed_target)
                              / nullif(count(*), 0), 2)
                 from marts.fact_applications
                 where missed_target is not null)                                 as target_miss_rate,

                -- referral vs non-referral accept (from fact_offers + dim_applicant)
                (select round(100.0 * count(*) filter (where fo.is_accepted)
                              / nullif(count(*), 0), 2)
                 from marts.fact_offers fo
                 join marts.dim_applicant da on fo.applicant_key = da.applicant_key
                 where da.is_referral)                                            as referral_accept_rate,

                (select round(100.0 * count(*) filter (where fo.is_accepted)
                              / nullif(count(*), 0), 2)
                 from marts.fact_offers fo
                 join marts.dim_applicant da on fo.applicant_key = da.applicant_key
                 where not da.is_referral)                                        as non_referral_accept_rate
        """)).fetchone()

    return KPISummary(**{
        k: _coerce_serializable(v) for k, v in zip(KPISummary.model_fields.keys(), r)
    })

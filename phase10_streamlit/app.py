"""
Phase 10: Streamlit demo for the Recruiting Intelligence Platform.

Calls the deployed FastAPI service (Phase 7 / Render) and turns each prediction
into a decision aid: the risk score, the plain-language recommended action
(Step 2 heuristics), and a chart of the model's own top feature importances for
context.

Config:
  API_BASE_URL — the deployed API. Read from st.secrets or env, defaults to the
  live Render URL. Override in Streamlit Cloud via app secrets, or locally via
  the environment.

Run locally:
    streamlit run phase10_streamlit/app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_API = "https://recruiting-intel-api.onrender.com"


def api_base() -> str:
    # st.secrets first (Streamlit Cloud), then env, then the live default.
    try:
        if "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass
    return os.getenv("API_BASE_URL", DEFAULT_API).rstrip("/")


API = api_base()

# The three models' top feature importances (from the canonical model cards,
# trained in the Docker/CI environment). Shown as context, not recomputed here.
FEATURE_IMPORTANCE = {
    "time_to_fill": [
        ("is_niche_location", 0.1652),
        ("recruiter_concurrent_open_reqs", 0.0304),
        ("department", 0.0283),
        ("location", 0.0220),
        ("seniority_level", 0.0176),
    ],
    "drop_off": [
        ("max_stage_dwell_days", 0.0953),
        ("interview_rounds_completed", 0.0407),
        ("interviewer_count", 0.0364),
        ("max_gap_between_rounds_days", 0.0040),
        ("avg_feedback_score", 0.0013),
    ],
    "offer_decline": [
        ("offer_to_band_ratio", 0.0837),
        ("candidate_source", 0.0188),
        ("referral_flag", 0.0109),
        ("location", 0.0018),
        ("seniority_level", 0.0007),
    ],
}

MODEL_META = {
    "time_to_fill": {"auc": 0.7069, "algo": "HistGradientBoosting"},
    "drop_off": {"auc": 0.6713, "algo": "HistGradientBoosting"},
    "offer_decline": {"auc": 0.6413, "algo": "LogisticRegression"},
}

BAND_COLOR = {"high": "#d64545", "elevated": "#e0a458", "low": "#3f9142"}


# --------------------------------------------------------------------------- #
# API helpers (with cold-start handling)
# --------------------------------------------------------------------------- #
def call_api(path: str, timeout: int = 90):
    """
    GET the API. The free Render service sleeps on idle, so the first request
    after a while can take 30-60s to cold-start. We use a generous timeout and
    surface a friendly message rather than a raw error.
    """
    url = f"{API}{path}"
    try:
        r = requests.get(url, timeout=timeout)
    except requests.exceptions.Timeout:
        st.error("The API did not respond in time. The free-tier server may be "
                 "waking from sleep — wait ~30s and try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API}. ({e})")
        return None
    if r.status_code == 404:
        return {"__404__": True, "detail": r.json().get("detail", "Not found")}
    if r.status_code != 200:
        st.error(f"API returned HTTP {r.status_code}: {r.text[:200]}")
        return None
    return r.json()


def importance_chart(model_key: str):
    data = FEATURE_IMPORTANCE[model_key]
    df = pd.DataFrame(data, columns=["feature", "importance"]).set_index("feature")
    st.caption("Model's top feature importances (permutation, mean AUC drop when shuffled)")
    st.bar_chart(df, horizontal=True)


def render_recommendation(rec: dict):
    band = rec.get("risk_band", "low")
    color = BAND_COLOR.get(band, "#888")
    st.markdown(
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:6px;font-weight:600'>{band.upper()} RISK</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Recommended action:** {rec.get('recommendation', '—')}")
    reasons = rec.get("rationale", [])
    if reasons:
        st.markdown("**Why:**")
        for r in reasons:
            st.markdown(f"- {r}")
    if rec.get("is_heuristic"):
        st.caption("This recommendation is a transparent rule derived from the "
                   "model's own feature importances — not a second ML model.")


def score_metric(label: str, score: float):
    st.metric(label, f"{score:.1%}")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Recruiting Intelligence Platform", page_icon="📊", layout="wide")

st.title("Recruiting Intelligence Platform")
st.markdown(
    "An end-to-end recruiting-funnel pipeline: synthetic data → PostgreSQL → dbt "
    "star schema → data-quality gate → three risk models → this API + demo. "
    "Predictions below come live from the deployed FastAPI service; each pairs a "
    "risk score with a plain-language recommended action."
)
st.caption(f"API: {API}  ·  Models trained on synthetic data (see Bring-Your-Own-Data note in the repo).")

with st.sidebar:
    st.header("About the models")
    for key, meta in MODEL_META.items():
        st.markdown(f"**{key.replace('_', ' ').title()}** — {meta['algo']}, "
                    f"test AUC {meta['auc']:.3f}")
    st.divider()
    st.markdown("Free-tier note: the API sleeps when idle. The first request may "
                "take ~30–60s to wake it — subsequent calls are fast.")

tab_kpi, tab_fill, tab_drop, tab_offer = st.tabs(
    ["📈 KPIs", "🧩 Requisition fill-risk", "🚪 Candidate drop-off", "💸 Offer decline"]
)

# ---- KPIs ----
with tab_kpi:
    st.subheader("Warehouse KPIs")
    if st.button("Load KPIs", key="kpi_btn"):
        with st.spinner("Querying the API (may cold-start)…"):
            data = call_api("/kpis/summary")
        if data:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Applications", f"{data['total_applications']:,}")
            c2.metric("Offers", f"{data['total_offers']:,}")
            c3.metric("Hires", f"{data['total_hires']:,}")
            c4.metric("Requisitions", f"{data['total_requisitions']:,}")
            c1.metric("Hire rate", f"{data['overall_hire_rate']}%")
            c2.metric("Offer accept rate", f"{data['overall_offer_accept_rate']}%")
            c3.metric("Dropout rate", f"{data['overall_dropout_rate']}%")
            c4.metric("Target-miss rate", f"{data['target_miss_rate']}%")
            st.markdown("**Validated signal:** referral vs non-referral offer acceptance")
            st.bar_chart(pd.DataFrame({
                "accept_rate": {
                    "referral": data["referral_accept_rate"],
                    "non_referral": data["non_referral_accept_rate"],
                }
            }))

# ---- Requisition fill-risk ----
with tab_fill:
    st.subheader("Open requisitions at risk of missing their fill target")
    limit = st.slider("How many to show", 5, 25, 10, key="fill_limit")
    if st.button("Load at-risk requisitions", key="fill_btn"):
        with st.spinner("Scoring open requisitions (may cold-start)…"):
            data = call_api(f"/requisitions/at-risk?limit={limit}")
        if data:
            rows = [{
                "req_id": r["req_id"], "department": r["department"],
                "seniority": r["seniority_level"], "location": r["location"],
                "niche": r["is_niche_location"],
                "fill_risk": r["fill_risk_score"],
                "band": r["recommended_action"]["risk_band"],
            } for r in data]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            top = data[0]
            st.markdown(f"#### Top-risk requisition: `{top['req_id']}`")
            score_metric("Predicted fill-miss risk", top["fill_risk_score"])
            render_recommendation(top["recommended_action"])
    importance_chart("time_to_fill")

# ---- Candidate drop-off ----
with tab_drop:
    st.subheader("Candidate drop-off (withdrawal) risk")
    aid = st.text_input("Application ID", value="APN0000001", key="drop_id")
    if st.button("Predict drop-off risk", key="drop_btn"):
        with st.spinner("Scoring application (may cold-start)…"):
            data = call_api(f"/applications/{aid}/dropout-risk")
        if data and data.get("__404__"):
            st.warning(data["detail"])
        elif data:
            score_metric("Predicted drop-off risk", data["dropout_risk_score"])
            render_recommendation(data["recommended_action"])
            with st.expander("Feature values used"):
                st.json(data["features"])
    importance_chart("drop_off")

# ---- Offer decline ----
with tab_offer:
    st.subheader("Offer decline risk (scored at offer-extension time)")
    oid = st.text_input("Offer ID", value="OFF0000001", key="offer_id")
    if st.button("Predict decline risk", key="offer_btn"):
        with st.spinner("Scoring offer (may cold-start)…"):
            data = call_api(f"/offers/{oid}/decline-risk")
        if data and data.get("__404__"):
            st.warning(data["detail"])
        elif data:
            score_metric("Predicted decline risk", data["decline_risk_score"])
            render_recommendation(data["recommended_action"])
            with st.expander("Feature values used"):
                st.json(data["features"])
    importance_chart("offer_decline")

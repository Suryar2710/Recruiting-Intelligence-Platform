"""
Phase 1: Synthetic recruiting funnel data generator.

Design philosophy
------------------
The whole platform is only as good as the signal in this data. If the tables
were random noise, every model in Phase 6 would be meaningless. So instead of
sampling outcomes at random, we model the funnel as a *sequential probabilistic
process*: each stage's outcome is a deterministic function of latent candidate,
requisition, and process features passed through an explicit logistic (or linear)
link with hand-chosen coefficients, plus a modest amount of noise.

That means the following correlations are baked in *by construction* and are
recoverable by a model:
  - Referral candidates get a meaningful bump to offer-accept probability.
  - Long gaps (>10 days) between interview rounds raise drop-off probability.
  - Offers below 90% of band midpoint decline much more often.
  - Senior + niche-location roles blow past their time-to-fill target.
  - Low interview feedback scores drive rejection at that stage.

We still inject noise everywhere so the models can't hit 100% AUC (which would
itself be a red flag in an interview). The goal is "clearly learnable, not
trivially perfect."

Run:
    python generate.py            # default ~ target row count
    python generate.py --scale 1.0
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# --------------------------------------------------------------------------- #
# Config / knobs
# --------------------------------------------------------------------------- #
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

SOURCES = ["referral", "job_board", "linkedin", "university", "agency"]
SOURCE_WEIGHTS = [0.18, 0.30, 0.28, 0.14, 0.10]

EDUCATION = ["high_school", "associate", "bachelor", "master", "phd"]
EDUCATION_WEIGHTS = [0.08, 0.12, 0.48, 0.26, 0.06]

DEPARTMENTS = ["engineering", "sales", "marketing", "finance", "operations", "data", "support"]
SENIORITY = ["junior", "mid", "senior", "staff", "principal"]
SENIORITY_WEIGHTS = [0.30, 0.34, 0.22, 0.09, 0.05]

# Niche / hard-to-fill locations get a time-to-fill penalty.
LOCATIONS = ["remote", "new_york", "san_francisco", "austin", "boise", "reno", "london", "berlin"]
NICHE_LOCATIONS = {"boise", "reno", "berlin"}  # thin talent markets in this synthetic world

STAGES = ["applied", "screen", "phone_interview", "onsite", "final", "offer"]

# Target-ish table sizes. Applications is the driver; everything else scales off it.
BASE_APPLICATIONS = 120_000  # ~ produces 500K-650K total rows across all tables


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bounded_dates(start: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Add integer day offsets (as timedelta) to an array of numpy datetime64 dates."""
    return start + offsets.astype("int64").astype("timedelta64[D]")


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def make_requisitions(rng: np.random.Generator, n_reqs: int) -> pd.DataFrame:
    dept = rng.choice(DEPARTMENTS, size=n_reqs)
    seniority = rng.choice(SENIORITY, size=n_reqs, p=SENIORITY_WEIGHTS)
    location = rng.choice(LOCATIONS, size=n_reqs)

    # Baseline target time-to-fill scales with seniority; harder roles are given
    # generous-but-not-generous-enough targets so senior+niche reliably miss.
    seniority_idx = np.array([SENIORITY.index(s) for s in seniority])
    # Targets are set so that easy roles (junior/mid, common locations) usually
    # HIT their target, while senior + niche roles usually MISS it. This creates a
    # learnable contrast rather than "everyone misses".
    # Targets set a little above the *typical* actual fill time for easy roles so
    # they mostly land at/under target, while senior + niche roles (which run
    # slower per stage) reliably blow past. This gives the Phase 6 requisition
    # model a clean, learnable target-miss contrast.
    base_target = 72 + seniority_idx * 6  # junior ~72d, principal ~96d
    target_ttf = (base_target + rng.normal(0, 6, n_reqs)).round().astype(int).clip(45, 140)

    # Opened over an ~18-month window.
    start = np.datetime64("2024-01-01")
    opened_offset = rng.integers(0, 540, n_reqs)
    opened_date = bounded_dates(np.full(n_reqs, start), opened_offset)

    n_recruiters = max(8, n_reqs // 400)
    n_managers = max(12, n_reqs // 250)

    return pd.DataFrame(
        {
            "req_id": [f"REQ{i:06d}" for i in range(1, n_reqs + 1)],
            "department": dept,
            "seniority_level": seniority,
            "location": location,
            "target_time_to_fill_days": target_ttf,
            "opened_date": opened_date,
            "hiring_manager_id": [f"HM{rng.integers(1, n_managers + 1):04d}" for _ in range(n_reqs)],
            "recruiter_id": [f"REC{rng.integers(1, n_recruiters + 1):04d}" for _ in range(n_reqs)],
        }
    )


def make_applicants(rng: np.random.Generator, n_app: int) -> pd.DataFrame:
    source = rng.choice(SOURCES, size=n_app, p=SOURCE_WEIGHTS)
    referral_flag = (source == "referral")

    # Experience correlates loosely with education; add noise.
    education = rng.choice(EDUCATION, size=n_app, p=EDUCATION_WEIGHTS)
    edu_idx = np.array([EDUCATION.index(e) for e in education])
    years_exp = (rng.gamma(shape=2.0, scale=2.5, size=n_app) + edu_idx * 0.8).round(1).clip(0, 40)

    start = np.datetime64("2024-01-01")
    applied_offset = rng.integers(0, 560, n_app)
    applied_date = bounded_dates(np.full(n_app, start), applied_offset)

    return pd.DataFrame(
        {
            "applicant_id": [f"APP{i:07d}" for i in range(1, n_app + 1)],
            "source": source,
            "applied_date": applied_date,
            "resume_years_experience": years_exp,
            "education_level": education,
            "referral_flag": referral_flag,
        }
    )


def build_funnel(
    rng: np.random.Generator,
    applicants: pd.DataFrame,
    requisitions: pd.DataFrame,
    n_applications: int,
):
    """
    Core of Phase 1. Walk each application through the funnel stage-by-stage,
    with each transition driven by explicit coefficients on latent features.
    Returns application-level + child tables.
    """
    n_app = len(applicants)
    n_req = len(requisitions)

    # ---- assign applicants + requisitions to applications --------------------
    applicant_pick = rng.integers(0, n_app, n_applications)
    req_pick = rng.integers(0, n_req, n_applications)

    app_source = applicants["source"].to_numpy()[applicant_pick]
    app_referral = applicants["referral_flag"].to_numpy()[applicant_pick]
    app_exp = applicants["resume_years_experience"].to_numpy()[applicant_pick]
    app_applicant_id = applicants["applicant_id"].to_numpy()[applicant_pick]

    req_seniority = requisitions["seniority_level"].to_numpy()[req_pick]
    req_location = requisitions["location"].to_numpy()[req_pick]
    req_id = requisitions["req_id"].to_numpy()[req_pick]
    req_target = requisitions["target_time_to_fill_days"].to_numpy()[req_pick]
    req_opened = requisitions["opened_date"].to_numpy()[req_pick]

    seniority_idx = np.array([SENIORITY.index(s) for s in req_seniority])
    is_niche = np.isin(req_location, list(NICHE_LOCATIONS))

    # application applied_date: a person applies to a requisition *shortly after*
    # it opens (0-45 days), not hundreds of days later. We anchor the application
    # purely to req_opened. The previous approach took
    # max(applicant_applied, req_opened) with two independent ~540-day draws,
    # which meant the applicant's pool-entry date frequently landed hundreds of
    # days after the req opened and inflated time-to-fill (mean 150+d, no
    # seniority correlation). Anchoring on the req is what keeps TTF realistic
    # and preserves the seniority/niche signal.
    apply_lag = rng.integers(0, 46, n_applications)
    applied_date = bounded_dates(req_opened, apply_lag).astype("datetime64[D]")

    # Keep the applicant's own "entered the pool" date consistent: it should be
    # at or before this application. We overwrite the sampled applicant applied
    # dates so the applicants CSV reflects "first time we saw this person",
    # rather than an independent timestamp that could postdate their application.
    first_seen = (
        pd.DataFrame({"applicant_id": app_applicant_id, "applied_date": applied_date})
        .groupby("applicant_id", as_index=True)["applied_date"]
        .min()
    )
    applicants = applicants.copy()
    mapped = applicants["applicant_id"].map(first_seen)
    applicants["applied_date"] = mapped.fillna(applicants["applied_date"]).astype("datetime64[ns]")

    # A latent per-applicant "strength" that quietly influences advancement and
    # feedback. This gives models something coherent to learn beyond single flags.
    strength = rng.normal(0, 1, n_applications) + 0.15 * (app_exp - app_exp.mean()) / (app_exp.std() + 1e-9)

    # ---- walk the funnel -----------------------------------------------------
    application_id = np.array([f"APN{i:07d}" for i in range(1, n_applications + 1)])

    # State arrays
    current_stage = np.full(n_applications, "applied", dtype=object)
    final_outcome = np.full(n_applications, "in_progress", dtype=object)
    was_dropped = np.zeros(n_applications, dtype=bool)
    drop_stage = np.full(n_applications, None, dtype=object)
    alive = np.ones(n_applications, dtype=bool)  # still progressing

    # Track dates as we move through stages (per application)
    stage_entered = {STAGES[0]: applied_date.copy()}
    stage_records = []  # for stage_history
    interview_records = []  # for interviews
    interview_counter = 1

    # per-app running feedback aggregation (for offer stage + features later)
    feedback_sum = np.zeros(n_applications)
    feedback_count = np.zeros(n_applications)
    max_gap_between_rounds = np.zeros(n_applications)  # signal: long gaps -> dropout
    interview_round_count = np.zeros(n_applications, dtype=int)

    # applied stage exit
    prev_entered = applied_date.copy()

    for si in range(len(STAGES) - 1):
        stage = STAGES[si]
        next_stage = STAGES[si + 1]

        entered = stage_entered[stage]
        # time spent in this stage: gap until next round. This is the lever for the
        # ">10 day gap between rounds raises drop-off" signal.
        gap_days = rng.gamma(shape=2.0, scale=2.0, size=n_applications).round().astype(int)
        gap_days = np.clip(gap_days, 1, 45)
        # senior + niche roles move slower per stage (feeds time-to-fill signal).
        # This accumulates across ~5 transitions into a sizable TTF difference.
        gap_days = (
            gap_days
            + (seniority_idx * 1).astype(int)
            + is_niche.astype(int) * rng.integers(2, 7, n_applications)
        )

        exited = bounded_dates(entered, gap_days)

        # record stage_history for everyone currently alive at this stage
        idx_alive = np.where(alive)[0]
        for i in idx_alive:
            stage_records.append(
                (application_id[i], stage, entered[i], exited[i])
            )

        # For interview-type stages, create interview rows + feedback scores.
        is_interview_stage = stage in ("phone_interview", "onsite", "final")
        feedback_this_stage = np.full(n_applications, np.nan)
        if is_interview_stage:
            # feedback_score driven by latent strength -> low scores => rejection
            fb_lin = 3.0 + 0.7 * strength + rng.normal(0, 0.6, n_applications)
            feedback_this_stage = np.clip(np.round(fb_lin), 1, 5)
            for i in idx_alive:
                interview_records.append(
                    (
                        f"INT{interview_counter + int(i):08d}",  # placeholder, fixed below
                        application_id[i],
                        int(interview_round_count[i]) + 1,
                        f"IVR{rng.integers(1, 300):04d}",
                        entered[i],
                        True,
                        int(feedback_this_stage[i]),
                    )
                )
            interview_round_count[idx_alive] += 1
            feedback_sum[idx_alive] += feedback_this_stage[idx_alive]
            feedback_count[idx_alive] += 1

        # track gaps between interview rounds (only meaningful once we have >=1 round)
        # the gap leading *into* the next round is gap_days; use it as the round gap.
        newly_gap = np.where(alive & (interview_round_count > 0), gap_days, max_gap_between_rounds)
        max_gap_between_rounds = np.maximum(max_gap_between_rounds, np.where(alive, gap_days, 0))

        # ---- transition probability: advance vs (reject|withdraw) ------------
        # Higher strength -> advance. Low feedback at interview stage -> reject.
        # Long gaps -> withdraw (candidate drop-off).
        z_advance = (
            0.6
            + 0.9 * strength
            + 0.5 * app_referral.astype(float)          # referrals advance a bit more
            - 0.15 * seniority_idx                        # senior funnels are stricter
        )
        if is_interview_stage:
            z_advance += 0.8 * (feedback_this_stage - 3.0)  # low score -> big drop in advance prob

        p_advance = sigmoid(z_advance)

        # withdrawal (candidate-initiated drop) rises sharply with long gaps
        z_withdraw = -2.2 + 0.12 * (gap_days - 8) + 0.08 * (gap_days > 10) * gap_days
        p_withdraw = sigmoid(z_withdraw) * 0.5  # cap withdrawal influence

        u = rng.random(n_applications)
        # decide per alive application
        advance = alive & (u < p_advance) & ~(rng.random(n_applications) < p_withdraw)
        withdraw = alive & ~advance & (rng.random(n_applications) < (p_withdraw + 0.05))
        reject = alive & ~advance & ~withdraw

        # apply outcomes
        # withdrawn
        w_idx = np.where(withdraw)[0]
        final_outcome[w_idx] = "withdrawn"
        was_dropped[w_idx] = True
        drop_stage[w_idx] = stage
        alive[w_idx] = False
        # rejected
        r_idx = np.where(reject)[0]
        final_outcome[r_idx] = "rejected"
        drop_stage[r_idx] = stage
        alive[r_idx] = False
        # advanced
        a_idx = np.where(advance)[0]
        current_stage[a_idx] = next_stage
        next_entered = np.empty(n_applications, dtype="datetime64[D]")
        next_entered[:] = np.datetime64("NaT", "D")
        next_entered[a_idx] = exited[a_idx].astype("datetime64[D]")
        stage_entered[next_stage] = next_entered

        prev_entered = exited

    # Anyone who reached "offer" stage and is still alive gets an offer.
    reached_offer = alive & (current_stage == "offer")

    # ---- OFFERS --------------------------------------------------------------
    offer_idx = np.where(reached_offer)[0]
    n_offers = len(offer_idx)

    # band midpoint by seniority; offer salary is a noisy fraction of it.
    band_mid_by_sen = np.array([80_000, 110_000, 150_000, 190_000, 240_000])
    band_midpoint = band_mid_by_sen[seniority_idx[offer_idx]] * (
        1 + rng.normal(0, 0.05, n_offers)
    )
    band_midpoint = band_midpoint.round(-2)

    # offer_ratio: how competitive the offer is vs band midpoint.
    # We deliberately generate a spread including many below-90% offers.
    offer_ratio = np.clip(rng.normal(0.97, 0.10, n_offers), 0.75, 1.25)
    offer_salary = (band_midpoint * offer_ratio).round(-2)

    offer_entered = stage_entered["offer"][offer_idx]
    offer_date = offer_entered  # entered offer stage == offer extended
    days_to_respond = rng.gamma(shape=2.0, scale=3.0, size=n_offers).round().astype(int).clip(1, 45)
    response_date = bounded_dates(offer_date, days_to_respond)

    # ACCEPT probability: referral bump + strong dependence on offer_ratio.
    off_referral = app_referral[offer_idx].astype(float)
    # Intercept tuned so the *baseline* (non-referral, at-band) accept rate sits
    # around 60-65%, leaving room for the referral bump to show ~15-20pp.
    z_accept = (
        0.15
        + 4.0 * (offer_ratio - 0.90)     # below 90% of band -> much lower accept
        + 0.85 * off_referral            # referral accept bump (~15-20pp)
        - 0.03 * days_to_respond         # dragging out slightly lowers accept
        + 0.25 * strength[offer_idx]
    )
    p_accept = sigmoid(z_accept)
    u_off = rng.random(n_offers)

    decision = np.where(u_off < p_accept, "accepted", "declined").astype(object)
    # a slice of slow responders expire
    expire_mask = (decision == "declined") & (days_to_respond > 20) & (rng.random(n_offers) < 0.3)
    decision[expire_mask] = "expired"

    # write offer outcome back into application final_outcome
    for local_i, gi in enumerate(offer_idx):
        if decision[local_i] == "accepted":
            final_outcome[gi] = "hired"
            current_stage[gi] = "hired"
        else:
            final_outcome[gi] = "rejected"  # declined/expired offer => not hired
            drop_stage[gi] = "offer"

    offer_id = np.array([f"OFF{i:07d}" for i in range(1, n_offers + 1)])
    offers = pd.DataFrame(
        {
            "offer_id": offer_id,
            "application_id": application_id[offer_idx],
            "offer_date": offer_date,
            "offer_salary": offer_salary,
            "band_midpoint": band_midpoint,
            "response_date": response_date,
            "decision": decision,
        }
    )

    # ---- HIRES ---------------------------------------------------------------
    accepted_mask = decision == "accepted"
    hire_offer_ids = offer_id[accepted_mask]
    hire_response = response_date[accepted_mask]
    # Notice period before start; kept modest so it doesn't dominate time-to-fill.
    start_dates = bounded_dates(hire_response, rng.integers(7, 25, accepted_mask.sum()))
    hires = pd.DataFrame(
        {
            "hire_id": [f"HIRE{i:07d}" for i in range(1, len(hire_offer_ids) + 1)],
            "offer_id": hire_offer_ids,
            "start_date": start_dates,
        }
    )

    # ---- APPLICATIONS table --------------------------------------------------
    applications = pd.DataFrame(
        {
            "application_id": application_id,
            "applicant_id": app_applicant_id,
            "req_id": req_id,
            "applied_date": applied_date,
            "current_stage": current_stage,
            "final_outcome": final_outcome,
        }
    )

    # ---- STAGE HISTORY -------------------------------------------------------
    stage_history = pd.DataFrame(
        stage_records,
        columns=["application_id", "stage_name", "entered_date", "exited_date"],
    )

    # ---- INTERVIEWS (fix ids to be unique/sequential) ------------------------
    interviews = pd.DataFrame(
        interview_records,
        columns=[
            "interview_id",
            "application_id",
            "interview_round",
            "interviewer_id",
            "scheduled_date",
            "completed_flag",
            "feedback_score",
        ],
    )
    interviews["interview_id"] = [f"INT{i:08d}" for i in range(1, len(interviews) + 1)]
    # add a light feedback_text tied to the score so the column isn't empty
    text_map = {
        1: "strong no; significant concerns",
        2: "leaning no; gaps in key areas",
        3: "mixed; borderline",
        4: "leaning yes; solid signal",
        5: "strong yes; excellent fit",
    }
    interviews["feedback_text"] = interviews["feedback_score"].map(text_map)

    # expose a couple of latent signals we baked in, for later sanity checks only
    diag = pd.DataFrame(
        {
            "application_id": application_id,
            "_referral": app_referral,
            "_max_round_gap": max_gap_between_rounds,
            "_was_dropped": was_dropped,
            "_final_outcome": final_outcome,
        }
    )

    return (applications, stage_history, interviews, offers, hires,
            offer_ratio, decision, off_referral, diag, applicants)


# --------------------------------------------------------------------------- #
# Summary / sanity checks
# --------------------------------------------------------------------------- #
def summarize_time_to_fill(tables: dict):
    """Actual time-to-fill vs target, split by senior+niche vs the rest."""
    reqs = tables["requisitions"].copy()
    apps = tables["applications"].copy()
    offers = tables["offers"].copy()
    hires = tables["hires"].copy()

    # link hire -> offer -> application -> req, compute fill date = hire.start_date
    off = offers.merge(hires, on="offer_id", how="inner")[["application_id", "offer_id"]]
    off = off.merge(hires[["offer_id", "start_date"]], on="offer_id", how="left")
    filled = off.merge(apps[["application_id", "req_id"]], on="application_id", how="left")
    filled = filled.merge(
        reqs[["req_id", "opened_date", "target_time_to_fill_days", "seniority_level", "location"]],
        on="req_id",
        how="left",
    )
    filled["start_date"] = pd.to_datetime(filled["start_date"])
    filled["opened_date"] = pd.to_datetime(filled["opened_date"])
    filled["actual_ttf"] = (filled["start_date"] - filled["opened_date"]).dt.days
    filled = filled[filled["actual_ttf"] >= 0]

    senior_niche = filled["seniority_level"].isin(["senior", "staff", "principal"]) & filled[
        "location"
    ].isin(list(NICHE_LOCATIONS))
    miss_sn = (filled[senior_niche]["actual_ttf"] > filled[senior_niche]["target_time_to_fill_days"]).mean()
    miss_rest = (filled[~senior_niche]["actual_ttf"] > filled[~senior_niche]["target_time_to_fill_days"]).mean()

    # Per-seniority avg target vs avg actual (the check requested during review).
    order = pd.CategoricalDtype(SENIORITY, ordered=True)
    filled["seniority_level"] = filled["seniority_level"].astype(order)
    by_sen = (
        filled.groupby("seniority_level", observed=True)
        .agg(
            n_hires=("actual_ttf", "size"),
            avg_target=("target_time_to_fill_days", "mean"),
            avg_actual=("actual_ttf", "mean"),
            miss_rate=("actual_ttf", lambda s: (s > filled.loc[s.index, "target_time_to_fill_days"]).mean()),
        )
        .round(1)
        .sort_index()
    )
    return miss_sn, miss_rest, by_sen


def print_summary(tables: dict, offer_ratio, decision, off_referral, diag):
    print("\n" + "=" * 70)
    print("ROW COUNTS")
    print("=" * 70)
    total = 0
    for name, df in tables.items():
        print(f"  {name:16s} {len(df):>10,d} rows")
        total += len(df)
    print(f"  {'TOTAL':16s} {total:>10,d} rows")

    print("\n" + "=" * 70)
    print("SIGNAL SANITY CHECKS (these must show clear, non-random gaps)")
    print("=" * 70)

    # 1. Referral vs non-referral offer accept rate
    acc = pd.DataFrame({"decision": decision, "referral": off_referral.astype(bool)})
    ref_rate = (acc[acc.referral].decision.eq("accepted")).mean()
    non_rate = (acc[~acc.referral].decision.eq("accepted")).mean()
    print(f"\n1. Offer accept rate  referral={ref_rate:.1%}  non-referral={non_rate:.1%}"
          f"  (gap {(ref_rate - non_rate) * 100:+.1f}pp; target ~15-20pp bump)")

    # 2. Below-90%-band decline rate vs at/above
    below = offer_ratio < 0.90
    dec = pd.Series(decision)
    decl_below = dec[below].isin(["declined", "expired"]).mean()
    decl_above = dec[~below].isin(["declined", "expired"]).mean()
    print(f"2. Decline/expire rate  offer<90% band={decl_below:.1%}  >=90%={decl_above:.1%}"
          f"  (should be meaningfully higher below)")

    # 3. Long gap between rounds vs drop-off
    d = diag.copy()
    long_gap = d["_max_round_gap"] > 10
    drop_long = d[long_gap]["_was_dropped"].mean()
    drop_short = d[~long_gap & (d["_max_round_gap"] > 0)]["_was_dropped"].mean()
    print(f"3. Withdrawal (drop) rate  max round gap>10d={drop_long:.1%}  <=10d={drop_short:.1%}"
          f"  (should be higher with long gaps)")

    # 4. senior+niche time-to-fill target-miss rate
    miss_sn, miss_rest, by_sen = summarize_time_to_fill(tables)
    print(f"4. Time-to-fill target MISS rate  senior+niche={miss_sn:.1%}  others={miss_rest:.1%}"
          f"  (senior+niche should miss more often)")

    print("\n   actual_time_to_fill by seniority (avg_target vs avg_actual):")
    print(by_sen.to_string())

    print("\n5. Final outcome distribution")
    print(d["_final_outcome"].value_counts(normalize=True).mul(100).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="scale factor on base row counts")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    Faker.seed(SEED)

    n_applications = int(BASE_APPLICATIONS * args.scale)
    n_applicants = int(n_applications * 0.55)   # some applicants apply to multiple reqs
    n_reqs = max(500, int(n_applications * 0.02))

    print(f"Generating: {n_applications:,} applications, {n_applicants:,} applicants, {n_reqs:,} reqs ...")

    requisitions = make_requisitions(rng, n_reqs)
    applicants = make_applicants(rng, n_applicants)

    (applications, stage_history, interviews, offers, hires,
     offer_ratio, decision, off_referral, diag, applicants) = build_funnel(
        rng, applicants, requisitions, n_applications
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    tables = {
        "applicants": applicants,
        "requisitions": requisitions,
        "applications": applications,
        "stage_history": stage_history,
        "interviews": interviews,
        "offers": offers,
        "hires": hires,
    }
    for name, df in tables.items():
        path = os.path.join(OUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)

    print(f"\nWrote {len(tables)} CSVs to {os.path.abspath(OUT_DIR)}")
    print_summary(tables, offer_ratio, decision, off_referral, diag)


if __name__ == "__main__":
    main()

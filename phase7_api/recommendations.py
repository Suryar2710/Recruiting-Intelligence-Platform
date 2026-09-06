"""
Rule-based "recommended action" layer for the prediction endpoints.

These are deterministic HEURISTICS, not a second ML model. Each rule is derived
directly from the corresponding model's own top permutation-importance features
(documented per function), so a high-risk score is paired with the plausible
lever a recruiter could actually pull. The rules are intentionally simple and
transparent -- they turn a bare probability into a decision aid without
pretending to be additional learned intelligence.

Thresholds (RISK_THRESHOLD etc.) are chosen around each model's operating range
and are explicitly labelled as heuristics, not learned cutoffs.

Every function returns a dict:
    {
        "risk_band": "low" | "elevated" | "high",
        "recommendation": str,        # the primary suggested action
        "rationale": [str, ...],      # which signals drove it (plain language)
        "is_heuristic": True          # honesty flag surfaced in the API/UI
    }
"""

from __future__ import annotations

# Shared risk banding. Kept simple and identical across models so the UI reads
# consistently; tuned to sit near each model's positive-class operating range.
HIGH = 0.60
ELEVATED = 0.40


def _band(score: float) -> str:
    if score >= HIGH:
        return "high"
    if score >= ELEVATED:
        return "elevated"
    return "low"


def time_to_fill_action(score: float, *, is_niche_location: bool,
                        seniority_level: str | None = None,
                        recruiter_concurrent_open_reqs: int | None = None) -> dict:
    """
    Model 1 top feature by a wide margin: is_niche_location. Secondary signals:
    seniority and recruiter load. Rules key off those.
    """
    band = _band(score)
    rationale: list[str] = []
    rec = "Fill pace looks on track; no action needed."

    if band != "low":
        if is_niche_location:
            rationale.append("role is in a thin talent-market (niche) location, "
                             "the strongest driver of missing the fill target")
            rec = ("Consider widening the location radius (or opening the role to "
                   "remote) or increasing the fill-time target for this requisition.")
        else:
            rationale.append("predicted fill time exceeds the target for this profile")
            rec = ("Review sourcing channels and interview-loop scheduling to "
                   "compress time-to-fill for this requisition.")

        if seniority_level in ("senior", "staff", "principal"):
            rationale.append(f"{seniority_level}-level roles fill more slowly")
        if recruiter_concurrent_open_reqs is not None and recruiter_concurrent_open_reqs >= 250:
            rationale.append(f"recruiter is carrying a heavy load "
                             f"({recruiter_concurrent_open_reqs} concurrent open reqs) "
                             f"— consider rebalancing")

    return {"risk_band": band, "recommendation": rec,
            "rationale": rationale, "is_heuristic": True}


def dropout_action(score: float, *, max_stage_dwell_days: float | None = None,
                   interview_rounds_completed: int | None = None) -> dict:
    """
    Model 2 top features: interview_rounds_completed and max_stage_dwell_days.
    The actionable lever is stage dwell time (long stalls precede withdrawals).
    """
    band = _band(score)
    rationale: list[str] = []
    rec = "Candidate engagement looks stable; no action needed."

    if band != "low":
        stalled = max_stage_dwell_days is not None and max_stage_dwell_days >= 10
        if stalled:
            rationale.append(f"candidate has sat in a single stage for "
                             f"{int(max_stage_dwell_days)}+ days, longer than typical — "
                             f"long stalls precede withdrawals")
            rec = ("Candidate has been in their current stage longer than typical — "
                   "follow up or expedite the next step to reduce drop-off risk.")
        else:
            rationale.append("elevated drop-off probability for this candidate profile")
            rec = ("Proactively check in with the candidate and confirm timeline "
                   "expectations to keep them engaged.")

        if interview_rounds_completed is not None and interview_rounds_completed == 0:
            rationale.append("no interview rounds completed yet — early-funnel drop risk")

    return {"risk_band": band, "recommendation": rec,
            "rationale": rationale, "is_heuristic": True}


def offer_decline_action(score: float, *, offer_to_band_ratio: float | None = None,
                         referral_flag: bool | None = None) -> dict:
    """
    Model 3 top feature: offer_to_band_ratio (offers below ~0.90 of band midpoint
    decline far more often). Secondary: referral status.
    """
    band = _band(score)
    rationale: list[str] = []
    rec = "Offer looks likely to be accepted; no action needed."

    if band != "low":
        below_band = offer_to_band_ratio is not None and offer_to_band_ratio < 0.90
        if below_band:
            rationale.append(f"offer is at {offer_to_band_ratio:.0%} of band midpoint "
                             f"(below 90%), which correlates strongly with decline")
            rec = ("Offer is below 90% of band midpoint, which correlates strongly "
                   "with decline — consider increasing the offer or emphasizing "
                   "non-salary value (equity, growth, flexibility).")
        else:
            rationale.append("elevated decline probability for this offer profile")
            rec = ("Reinforce the offer with a personal close and highlight the "
                   "candidate's motivations before the response deadline.")

        if referral_flag is False:
            rationale.append("non-referral candidates accept at a lower rate here")

    return {"risk_band": band, "recommendation": rec,
            "rationale": rationale, "is_heuristic": True}

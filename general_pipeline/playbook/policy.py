"""Dataset-specific decision-time / leakage policies (HyperAck-style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class LeakagePolicy:
    """Post-outcome or contested features unavailable at honest decision time T0."""

    unsafe_only_features: List[str] = field(default_factory=list)
    rationale: str = "No classic post-outcome leakage columns identified."
    has_leakage: bool = False


# Features that inflate lab scores but are unavailable / contested at decision time.
LEAKAGE_POLICIES: Dict[str, LeakagePolicy] = {
    "adult": LeakagePolicy(
        unsafe_only_features=[],
        rationale=(
            "Census attributes are contemporaneous. No post-outcome leakage columns; "
            "safe and unsafe feature matrices are identical. Focus is FE + optimization ladder."
        ),
        has_leakage=False,
    ),
    "bank_marketing": LeakagePolicy(
        unsafe_only_features=["duration"],
        rationale=(
            "`duration` is call length known only after the call ends — classic target leakage "
            "for predicting subscription (y). Safe models must drop it (HyperAck analog of final fares)."
        ),
        has_leakage=True,
    ),
    "breast_cancer": LeakagePolicy(
        unsafe_only_features=[],
        rationale="All morphometry features are available at diagnosis time; no temporal leakage.",
        has_leakage=False,
    ),
    "heart_disease": LeakagePolicy(
        unsafe_only_features=[],
        rationale="Clinical attributes are pre-diagnosis; no post-outcome leakage identified.",
        has_leakage=False,
    ),
    "credit_default": LeakagePolicy(
        unsafe_only_features=[],
        rationale=(
            "Payment history / bill amounts are known before predicting next-month default; "
            "no post-outcome leakage identified."
        ),
        has_leakage=False,
    ),
    "german_credit": LeakagePolicy(
        unsafe_only_features=[],
        rationale="Applicant attributes are known at underwriting time; no leakage identified.",
        has_leakage=False,
    ),
    "mushroom": LeakagePolicy(
        unsafe_only_features=[],
        rationale="Physical traits are observed before edibility label; no leakage identified.",
        has_leakage=False,
    ),
    "spambase": LeakagePolicy(
        unsafe_only_features=[],
        rationale="Word/char frequencies are available at classification time; no leakage identified.",
        has_leakage=False,
    ),
    "online_shoppers": LeakagePolicy(
        unsafe_only_features=["PageValues"],
        rationale=(
            "`PageValues` is a Google Analytics metric tightly coupled to purchase intent and can "
            "act as a near-outcome proxy. Safe mode drops it for an honest pre-purchase decision model."
        ),
        has_leakage=True,
    ),
    "wine_quality": LeakagePolicy(
        unsafe_only_features=[],
        rationale="Physicochemical tests are available before quality rating; no leakage identified.",
        has_leakage=False,
    ),
}


def get_policy(key: str) -> LeakagePolicy:
    return LEAKAGE_POLICIES.get(key, LeakagePolicy())

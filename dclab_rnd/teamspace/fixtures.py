"""The invented inputs the deterministic team agents read in this release (W5, 2026-11-20).

Nothing here is a real customer, order or model. Each fixture stands in for what a later release reads for real:

- ``SAMPLES``: an invented table's columns, as Data Science's ``inspect_schema`` sees them (R3 reads the project's own
  tables). ``renewal_v1`` has the renewal date and mature outcomes; ``renewal_no_dates`` lacks the renewal date, and
  names the team that can supply it.
- ``MODELS``: the two model specialists' fixtures of the owner's food delivery walkthrough, for order 1042 at its first
  dispatch (18:00): ETA fixture E1 (arrival 18:32) and HyperAck acceptance fixture A1 (72% that a courier accepts the
  first offer). Every answer built from them is labelled simulated (``status=simulated``, ``simulated=True``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

SAMPLES: dict[str, dict[str, Any]] = {
    "sample:renewal_v1": {
        "columns": ("account_id", "plan", "tenure_months", "seats", "last_login_at", "support_tickets_90d", "renewal_date",
                    "renewed_by_date"),
        "renewal_date": "renewal_date",
        "outcome": "renewed_by_date",
        "outcomes_mature": True,  # every renewal date in the sample is past, so every outcome is known
        "known_after_renewal": ("renewed_by_date",),
    },
    "sample:renewal_no_dates": {
        "columns": ("account_id", "plan", "tenure_months", "seats", "last_login_at", "support_tickets_90d", "cancelled"),
        "renewal_date": None,
        "outcome": None,
        "outcomes_mature": False,
        "known_after_renewal": (),
        "suppliers": {"renewal_date": "business"},  # the contracts system is the Business team's
    },
}

# (specialist kind, dispatch snapshot) -> its fixture. The snapshot is the order's state when the question was asked.
DISPATCH = "dispatch:1042@18:00"
MODELS: dict[tuple[str, str], dict[str, Any]] = {
    ("eta", DISPATCH): {"id": "E1", "order": "1042", "at": "18:00", "arrival": "18:32", "assumption": "fixed route"},
    ("acceptance", DISPATCH): {"id": "A1", "order": "1042", "at": "18:00", "probability": 0.72},
}


def sample(ref: str) -> Optional[Mapping[str, Any]]:
    return SAMPLES.get(ref)


def feasibility(ref: str) -> dict[str, Any]:
    """What the sample can answer for a ranking seven days before renewal: ``ok``, or the missing field and who can
    supply it."""
    found = SAMPLES.get(ref)
    if found is None:
        return {"ok": False, "missing": "sample", "supplier": "business", "why": f"no sample {ref} is available"}
    if not found.get("renewal_date"):
        return {"ok": False, "missing": "renewal_date", "supplier": (found.get("suppliers") or {}).get("renewal_date", "business"),
                "why": "the sample has no renewal date, so the moment seven days before renewal cannot be fixed"}
    if not found.get("outcome") or not found.get("outcomes_mature"):
        return {"ok": False, "missing": "renewed_by_date", "supplier": "business",
                "why": "the sample's renewal outcomes are not known yet"}
    return {"ok": True, "renewal_date": found["renewal_date"], "outcome": found["outcome"],
            "excluded": tuple(found.get("known_after_renewal") or ())}


def model_fixture(kind: str, snapshot: str) -> Optional[Mapping[str, Any]]:
    return MODELS.get((kind, snapshot))


def eta_after(arrival: str, minutes: int) -> str:
    """An arrival moved by ``minutes`` under the fixed-route assumption (the kitchen's delay passes straight through)."""
    moment = datetime.strptime(arrival, "%H:%M") + timedelta(minutes=int(minutes))
    return moment.strftime("%H:%M")


__all__ = ["DISPATCH", "MODELS", "SAMPLES", "eta_after", "feasibility", "model_fixture", "sample"]

"""The invented inputs the deterministic team agents read in this release (W5, 2026-11-20).

Nothing here is a real customer, order or model. Each fixture stands in for what a later release reads for real:

- ``SAMPLES``: an invented table's columns, as Data Science's ``inspect_schema`` sees them (R3 reads the project's own
  tables). ``renewal_v1`` has the renewal date and mature outcomes; ``renewal_no_dates`` lacks the renewal date, and
  names the team that can supply it. 0.170.0 (T1): each sample declares which event each outcome column records
  (``labels``: ``renewed_by_date`` records a failure to renew by the contractual date, never a cancellation), so
  feasibility means the outcome a person confirmed (``dclab_core.target_contract.agreement``).
  ``renewal_with_cancellations`` also has the CRM's cancellation log (``cancelled_at``) and a column written only after
  a cancellation (``cancellation_reason``, a late feature); ``renewal_recent`` has renewal dates still ahead, so its
  outcomes are not known yet.
- ``MODELS``: the two model specialists' fixtures of the owner's food delivery walkthrough, for order 1042 at its first
  dispatch (18:00): ETA fixture E1 (arrival 18:32) and HyperAck acceptance fixture A1 (72% that a courier accepts the
  first offer). Every answer built from them is labelled simulated (``status=simulated``, ``simulated=True``).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

from dclab_core import target_contract as tc

LAPSE, CANCEL = "non_renewal_by_date", "explicit_cancellation"
_RENEWAL = ("account_id", "plan", "tenure_months", "seats", "last_login_at", "support_tickets_90d", "renewal_date", "renewed_by_date")
# who can supply a missing record, and from where (the contracts system and the CRM are the Business team's)
_SUPPLY = {"renewal_date": ("business", "the contracts system: each account's contractual renewal date"),
           CANCEL: ("business", "the CRM's cancellation log: one row per account that cancelled, with its date"),
           LAPSE: ("business", "the contracts system: whether each account renewed by its contractual renewal date")}

SAMPLES: dict[str, dict[str, Any]] = {
    "sample:renewal_v1": {
        "columns": _RENEWAL,
        "renewal_date": "renewal_date",
        "outcome": "renewed_by_date",
        "outcomes_mature": True,  # every renewal date in the sample is past, so every outcome is known
        "known_after_renewal": ("renewed_by_date",),
        "labels": ({"column": "renewed_by_date", "event": LAPSE},),  # a renewal record: it holds no cancellation
        "accounts": 1200,
        "do_not_contact": 3,  # accounts on the CRM's do-not-contact list (Sales removes them before any plan)
    },
    "sample:renewal_with_cancellations": {  # the same accounts with the CRM's cancellation log attached
        "columns": _RENEWAL + ("cancelled_at", "cancellation_reason"),
        "renewal_date": "renewal_date",
        "outcome": "renewed_by_date",
        "outcomes_mature": True,
        # cancellation_reason is written only after a cancellation: a late feature, never an input
        "known_after_renewal": ("renewed_by_date", "cancelled_at", "cancellation_reason"),
        "labels": ({"column": "renewed_by_date", "event": LAPSE}, {"column": "cancelled_at", "event": CANCEL}),
        "accounts": 1200,
        "do_not_contact": 3,
    },
    "sample:renewal_recent": {  # renewal dates still ahead: no outcome is known yet
        "columns": _RENEWAL,
        "renewal_date": "renewal_date",
        "outcome": "renewed_by_date",
        "outcomes_mature": False,
        "known_after_renewal": ("renewed_by_date",),
        "labels": ({"column": "renewed_by_date", "event": LAPSE},),
        "accounts": 300,
        "do_not_contact": 0,
    },
    "sample:renewal_no_dates": {
        "columns": ("account_id", "plan", "tenure_months", "seats", "last_login_at", "support_tickets_90d", "cancelled"),
        "renewal_date": None,
        "outcome": None,
        "outcomes_mature": False,
        "known_after_renewal": (),
        "labels": ({"column": "cancelled", "event": CANCEL},),
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


def labels(ref: str) -> list[tc.LabelDeclaration]:
    """What each outcome column of the sample records (declared here, never guessed from a column's name)."""
    found = SAMPLES.get(ref) or {}
    return [tc.LabelDeclaration(column=d["column"], event=d["event"], declared_by="fixture", source=ref,
                                mature=bool(d.get("mature", found.get("outcomes_mature")))) for d in found.get("labels") or ()]


def offered(ref: Optional[str] = None) -> tuple[str, ...]:
    """The events a person chooses from when the outcome needs a definition: the ones the sample records first, then the
    others DCLab reads (``target_contract.EVENTS``), so a person can name an event the table does not hold."""
    recorded = [d.event for d in labels(ref)] if ref else []
    return tuple(dict.fromkeys([*recorded, *tc.EVENTS]))


def feasibility(ref: str, target: Optional[tc.TargetContract] = None) -> dict[str, Any]:
    """What the sample can answer for a ranking seven days before renewal: ``ok``, or what is missing, why it matters and
    who can supply it (``missing``: the field or record; ``field``: its code; ``supplier``: the team; ``source``: where it
    comes from). With ``target`` (the outcome a person confirmed), a label column must record that event, with known
    outcomes (``target_contract.agreement``; ``verdict``). Without one, the sample's own outcome column is the outcome."""
    found = SAMPLES.get(ref)
    if found is None:
        return {"ok": False, "missing": "sample", "field": "sample", "supplier": "business", "why": f"no sample {ref} is available"}
    if not found.get("renewal_date"):
        supplier = (found.get("suppliers") or {}).get("renewal_date", "business")
        return {"ok": False, "missing": "renewal_date", "field": "renewal_date", "supplier": supplier, "source": _SUPPLY["renewal_date"][1],
                "why": "the sample has no renewal date, so the moment seven days before renewal cannot be fixed",
                "matters": "without it no row has a prediction moment, so nothing can be scored before renewal"}
    verdict = tc.agreement(target, labels(ref)) if target is not None else None
    if verdict is not None and verdict.kind in ("event_mismatch", "no_label"):
        said = tc.event_label(verdict.missing or "")
        held = "; ".join(f"{d.column} records {tc.event_label(d.event)}" for d in labels(ref)) or "no outcome column"
        supplier, source = _SUPPLY.get(verdict.missing or "", ("business", None))
        return {"ok": False, "missing": f"a record of {said}", "field": verdict.missing, "supplier": supplier, "source": source,
                "why": f"the outcome confirmed is {said}, but {ref} holds no record of it ({held})",
                "matters": f"a model trained on what the table records would learn {', '.join(tc.event_label(e) for e in verdict.recorded) or 'nothing'}, "
                           f"not {said}: its ranking would answer another question",
                "verdict": verdict}
    if (verdict is not None and verdict.kind == "immature") or (verdict is None and (not found.get("outcome") or not found.get("outcomes_mature"))):
        column = (verdict.column if verdict is not None else None) or found.get("outcome") or "renewed_by_date"
        supplier, source = _SUPPLY.get((verdict.missing if verdict is not None else None) or LAPSE, _SUPPLY[LAPSE])
        return {"ok": False, "missing": column, "field": column, "supplier": supplier, "source": source,
                "why": "the sample's renewal outcomes are not known yet",
                "matters": f"{column} is known only once each renewal date has passed: a model cannot be checked on outcomes nobody has seen",
                "verdict": verdict}
    outcome = verdict.column if verdict is not None else found["outcome"]
    return {"ok": True, "renewal_date": found["renewal_date"], "outcome": outcome,
            "excluded": tuple(found.get("known_after_renewal") or ()), "verdict": verdict}


_MOMENT = re.compile(r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|fourteen|thirty)\s+days?\s+before\s+"
                     r"(?:each\s+|the\s+)?renewal)\b", re.I)


def terms(ref: Optional[str], question: str = "") -> dict[str, Optional[str]]:
    """The rest of a confirmed outcome's contract for a renewal sample (0.170.0, T1): what one prediction is for, when it
    is made (read from the question, else unknown), by when the event counts, where the label comes from and when a row's
    outcome is known. Fixture facts, never guessed from a table."""
    when = _MOMENT.search(question or "")
    renewal = bool((SAMPLES.get(ref or "") or {}).get("renewal_date"))
    return {"entity": "account", "prediction_time": when.group(1) if when else None,
            "horizon": "by the account's contractual renewal date" if renewal else None,
            "label_source": ref if ref in SAMPLES else None,
            "maturity_rule": "an account's outcome is known once its renewal date has passed" if renewal else None}


def model_fixture(kind: str, snapshot: str) -> Optional[Mapping[str, Any]]:
    return MODELS.get((kind, snapshot))


def eta_after(arrival: str, minutes: int) -> str:
    """An arrival moved by ``minutes`` under the fixed-route assumption (the kitchen's delay passes straight through)."""
    moment = datetime.strptime(arrival, "%H:%M") + timedelta(minutes=int(minutes))
    return moment.strftime("%H:%M")


__all__ = ["DISPATCH", "MODELS", "SAMPLES", "eta_after", "feasibility", "labels", "model_fixture", "offered", "sample", "terms"]

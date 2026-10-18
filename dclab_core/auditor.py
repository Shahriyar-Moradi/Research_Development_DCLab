"""The auditor (package 28.5): before a holdout is read, the columns a solution would read are screened for a leak.

A column that alone separates the confirmed cases on the training rows (its single-column ROC AUC at least ``BAR``, or at
most ``1 - BAR``) is a leak suspect: a real signal of fraud, churn or failure is rarely that clean, and a column written
after the outcome often is. The screen reads only the rows it is given (the fit rows: never the validation or the holdout,
so screening chooses nothing on them), fits nothing and changes no number. Its verdict is evidence for the person, who
decides whether a suspect is kept out; the auditor never forbids a column itself.

``screen`` is the rule; ``AuditVerdict`` v1 the record of one audit (the suspects with their scores, what the moment says
is written after it, the verdict and the round); ``verdict`` assembles it. A host sends a solution back to its designer
when the verdict says so, at most ``ROUNDS`` times, and stops with the suspects as a named blocker after that.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

BAR = 0.98  # a single column this clean on the training rows is a leak suspect
ROUNDS = 2  # send-backs before the auditor stops with a named blocker
MIN_POSITIVES = 10  # fewer confirmed cases on the rows than this: the screen says nothing (an AUC on a handful is noise)
MAX_CODES = 50  # a code column with more distinct codes than this is not screened: each rare code alone would "separate" by chance


class Suspect(Contract):
    column: str = Field(min_length=1, max_length=200)
    auc: float = Field(ge=0.0, le=1.0, description="The column alone against the confirmed cases, on the screened rows.")
    reason: str = Field(max_length=600)


class MomentItem(Contract):
    column: str = Field(min_length=1, max_length=200)
    reason: str = Field(max_length=600)
    source: str = Field(max_length=40, description="audit, moment or model: who says it is written after the prediction moment.")


class AuditVerdict(Versioned):
    """One audit of a solution before its holdout is read: names, scores and counts, never a row."""

    schema_version: Literal["dclab_core.audit_verdict.v1"] = _version("audit_verdict")
    subject: str = Field(min_length=1, max_length=120, description="The project (or the stage) audited.")
    round: int = Field(default=1, ge=1, le=ROUNDS + 1)
    rows: int = Field(default=0, ge=0, description="The training rows screened.")
    positives: int = Field(default=0, ge=0)
    columns: int = Field(default=0, ge=0, description="The columns screened.")
    bar: float = Field(default=BAR, gt=0.5, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict, description="Every screened column's AUC alone (a suspect's and every other's).")
    inputs: str = Field(default="", max_length=80, description="The fingerprint of what was screened: a change to the tables, the labels or the columns kept out makes the verdict old.")
    suspects: list[Suspect] = Field(default_factory=list, max_length=200)
    after_moment: list[MomentItem] = Field(default_factory=list, max_length=200)
    verdict: Literal["pass", "send_back", "stop", "not_screened"] = "pass"
    note: str = Field(default="", max_length=600)


CONTRACTS["audit_verdict"] = AuditVerdict


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC of one column's values (ties shared), by ranks: no model, no threshold."""
    order = pd.Series(x).rank(method="average").to_numpy()
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((order[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def screen(frame: pd.DataFrame, labels: pd.Series, rows: Sequence[Any], columns: Sequence[str], bar: float = BAR) -> tuple[list[Suspect], dict[str, Any]]:
    """The leak suspects among ``columns`` on ``rows`` (the training rows): each column alone against ``labels`` (1 a
    confirmed case, 0 every other row). A missing value is its own value, below every other: a column that is filled only
    after the outcome separates by its gaps. Returns the suspects, strongest first, and the counts screened with every
    screened column's AUC (``counts["scores"]``)."""
    index = pd.Index(rows).intersection(frame.index)
    y = labels.reindex(index).fillna(0).astype(int).to_numpy()
    counts: dict[str, Any] = {"rows": int(len(index)), "positives": int(y.sum()), "columns": 0, "scores": {}}
    if counts["positives"] < MIN_POSITIVES or counts["positives"] == len(index):
        return [], counts
    out = []
    for c in columns:
        if c not in frame.columns:
            continue
        values = pd.to_numeric(frame.loc[index, c], errors="coerce")
        if values.notna().sum() == 0:
            if frame.loc[index, c].nunique(dropna=True) > MAX_CODES:
                continue
            codes, _ = pd.factorize(frame.loc[index, c].astype("string"), use_na_sentinel=True)  # a code column: its codes' order is arbitrary, so a code is read by its rate
            values = pd.Series(codes, index=index).replace(-1, np.nan)
            rate = pd.Series(y, index=index).groupby(values).mean()
            values = values.map(rate)
        counts["columns"] += 1
        filled = values.fillna(values.min() - 1 if values.notna().any() else 0).to_numpy(dtype=float)
        auc = _auc(filled, y)
        counts["scores"][str(c)[:200]] = round(auc, 4)
        strength = max(auc, 1.0 - auc)
        if strength >= bar:
            out.append(Suspect(column=str(c)[:200], auc=round(auc, 4),
                               reason=f"alone it separates the {counts['positives']} confirmed cases on the {counts['rows']} training rows "
                                      f"(AUC {auc:.3f}): a column this clean is usually written after the outcome"))
    return sorted(out, key=lambda s: -max(s.auc, 1 - s.auc)), counts


def verdict(subject: str, suspects: Sequence[Suspect], counts: dict[str, Any], after_moment: Sequence[dict[str, Any]] = (),
            round_: int = 1, bar: float = BAR, inputs: str = "", limits: str = "") -> AuditVerdict:
    """The audit's record: a send-back while suspects (or columns the moment says are written after it) remain and rounds
    are left, a stop after ``ROUNDS``, else a pass; not screened when the rows held too few confirmed cases."""
    items = [MomentItem(column=str(i.get("column"))[:200], reason=str(i.get("reason") or "")[:600], source=str(i.get("source") or "moment")[:40])
             for i in after_moment if i.get("column")]
    found = bool(suspects) or bool(items)
    if found:
        state = "send_back" if round_ <= ROUNDS else "stop"
    else:
        state = "not_screened" if counts.get("positives", 0) < MIN_POSITIVES else "pass"
    note = {"pass": "No column alone separates the confirmed cases, and the moment names none written after it.",
            "send_back": "Leak suspects: the person decides whether each is kept out before the holdout is read.",
            "stop": "Leak suspects remain after the rounds: the run stops here, with them as its blocker.",
            "not_screened": f"Fewer than {MIN_POSITIVES} confirmed cases on the training rows: the screen says nothing."}[state]
    note = f"{note} {limits}".strip()[:600]
    return AuditVerdict(subject=str(subject)[:120], round=min(round_, ROUNDS + 1), rows=counts.get("rows", 0), positives=counts.get("positives", 0),
                        columns=counts.get("columns", 0), bar=bar, scores=dict(counts.get("scores") or {}), inputs=str(inputs)[:80],
                        suspects=list(suspects), after_moment=items, verdict=state, note=note)


def as_lines(v: AuditVerdict) -> list[str]:
    """The verdict as the person reads it: one line a suspect."""
    return [f"{s.column}: {s.reason}" for s in v.suspects] + [f"{i.column}: {i.reason} ({i.source})" for i in v.after_moment]


__all__ = ["BAR", "ROUNDS", "MIN_POSITIVES", "MAX_CODES", "Suspect", "MomentItem", "AuditVerdict", "screen", "verdict", "as_lines"]

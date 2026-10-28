"""Was the holdout already looked at? (H8's open gap, closed 2026-10-28.)

An analysis in Home's chat relates columns on the person's table before any prediction exists; its exposure log says
which columns were related and which dates the answers covered (``query.exposure``, carried into the project by
``draft.exits.conversion``). Rule 2 says the holdout is used once. ``holdout_exposure`` reads the carried log against the
split that ran and says, in words built from the records, whether the holdout's rows were among the rows the chat looked
at:

- ``after``: every answer was bounded on the split's own time column and the holdout starts more than a day after the
  last date they covered (a day's margin: the answers read times in UTC, the split in the table's own clock): the holdout
  number is a first look.
- ``seen``: the answers covered dates the holdout holds (on its time column): the number is not a first look.
- ``may_hold``: the answers were not bounded on the split's time column (no period, another column, a random split, a
  kind whose machine does not say its holdout's dates), or the holdout starts the day after them: it may hold rows they
  covered.

It never moves the split (a deterministic split is chosen by the engine); it is said beside the holdout number. Pure:
no table, no row, no value; dates and names only.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pandas as pd

STATUSES = ("after", "seen", "may_hold")


def holdout_exposure(exposure: Optional[Mapping[str, Any]], time_column: Optional[str], holdout_range: Optional[Sequence[str]]
                     ) -> Optional[dict[str, Any]]:
    """The holdout against what the chat explored, or None when nothing was explored."""
    if not exposure or not exposure.get("answers"):
        return None
    related = [str(c) for c in exposure.get("related_to_target") or []]
    out: dict[str, Any] = {"answers": int(exposure["answers"]), "related_to_target": related}
    after = exposure.get("holdout_after") or {}
    start = pd.to_datetime(str(holdout_range[0]), errors="coerce") if holdout_range else pd.NaT
    until = pd.to_datetime(str(after.get("date")), errors="coerce") if after.get("date") else pd.NaT
    if time_column and after.get("column") == time_column and pd.notna(start) and pd.notna(until):
        out.update(explored_until=until.date().isoformat(), holdout_from=start.date().isoformat())
        if start.normalize() > until.normalize() + pd.Timedelta(days=1):
            out["status"] = "after"
            out["note"] = (f"The chat's answers covered {time_column} up to {out['explored_until']}; the holdout starts on "
                           f"{out['holdout_from']}, after them, so its number is a first look.")
        elif start.normalize() > until.normalize():
            # the day after: the answers read times in UTC and the split in the table's own clock, so the first hours of
            # the holdout may be among what they covered (the review: a false "after" is the dangerous direction)
            out["status"] = "may_hold"
            out["note"] = (f"The chat's answers covered {time_column} up to {out['explored_until']} and the holdout starts the day "
                           f"after ({out['holdout_from']}): with time zones its first hours may be among the rows they looked at, "
                           "so its number may not be a first look.")
        else:
            out["status"] = "seen"
            out["note"] = (f"The chat's answers covered {time_column} up to {out['explored_until']}, and the holdout starts on "
                           f"{out['holdout_from']}: some of its rows were looked at before the build, so its number is not a first look.")
    else:
        out["status"] = "may_hold"
        why = ("the split is not in time" if not time_column else
               f"the answers were not bounded on {time_column}" if after.get("column") != time_column else "the dates could not be read")
        out["note"] = (f"The chat's answers looked at the table before the build and {why}: the holdout may hold rows they covered, "
                       "so its number may not be a first look.")
    if related:
        out["note"] += f" Compared with the outcome in the chat: {', '.join(related)}."
    return out


def mark_claims(claims: list[dict[str, Any]], seen: Optional[Mapping[str, Any]]) -> None:
    """The note added once to the holdout's claim (the first claim citing a holdout metric, else the first claim), unless
    the holdout was a first look. Never to the evidence's limits too: a report lists both (the review: said twice)."""
    if not seen or seen.get("status") == "after" or not claims:
        return
    target = next((c for c in claims if any("holdout" in str(e) for e in c.get("evidence") or [])), claims[0])
    limits = target.setdefault("limitations", [])
    if seen["note"] not in limits:
        limits.append(seen["note"])


__all__ = ["STATUSES", "holdout_exposure", "mark_claims"]

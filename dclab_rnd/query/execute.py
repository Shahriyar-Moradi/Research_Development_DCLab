"""The plan executor, the row floor and the exposure log (package H6).

``execute(plan, frame)`` runs a validated plan with pandas and returns ``{columns, rows, suppressed_groups,
rows_scanned, rows_matched, plan_hash, version}``: groups with fewer rows than the floor are removed and counted, never
shown. ``run(store, draft_id, plan)`` opens the draft's cleaned table (only while it matches the fingerprint recorded
when it was written), validates, executes, and appends the exposure entry to the draft.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

from . import plan as plans
from .validate import PlanError, validate

FLOOR_DEFAULT = 5
FLOOR_MIN = 3  # a workspace may raise the floor, never lower it below this
EXTREME_FACTOR = 4  # a min, a max or a percentile beyond p5..p95 is one row's value: shown only for groups this many floors big


def model_reads_aggregates() -> bool:
    """Whether a model may read an answer's aggregate result (H7 writes the sentence from it): on in R&D unless
    DCLAB_MODEL_READS_AGGREGATES=0; groups under the floor are never in a result, so never read."""
    return os.environ.get("DCLAB_MODEL_READS_AGGREGATES", "1").strip() not in ("0", "false", "no", "off")


def floor() -> int:
    try:
        return max(FLOOR_MIN, int(os.environ.get("DCLAB_ROW_FLOOR", FLOOR_DEFAULT)))
    except ValueError:
        return FLOOR_DEFAULT


def _naive(s):
    """A datetime column without its time zone (compared with the plan's plain dates)."""
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert(None)
    return s


def _mask(frame, f: dict[str, Any]):
    import pandas as pd

    s, op, v = _naive(frame[f["column"]]), f["op"], f.get("value")
    if pd.api.types.is_datetime64_any_dtype(s) and op not in ("is_null", "not_null"):
        v = [pd.to_datetime(x) for x in v] if isinstance(v, list) else pd.to_datetime(v)
    if op == "eq":
        return s == v
    if op == "ne":
        return s != v
    if op == "in":
        return s.isin(v)
    if op == "not_in":
        return ~s.isin(v)
    if op == "gt":
        return s > v
    if op == "ge":
        return s >= v
    if op == "lt":
        return s < v
    if op == "le":
        return s <= v
    if op == "between":
        return s.between(v[0], v[1])
    if op == "is_null":
        return s.isna()
    return s.notna()


def _cell(value: Any) -> Any:
    import pandas as pd

    from decimal import Decimal

    if isinstance(value, Decimal):
        value = float(value)
    if value is None or value is pd.NaT or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value


_FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def execute(plan: dict[str, Any], frame) -> dict[str, Any]:
    """Run a plan that ``validate`` accepted. Every result is an aggregate; groups under the floor are suppressed."""
    import pandas as pd

    data = frame
    scanned = len(data)
    for f in plan.get("filters") or []:
        data = data[_mask(data, f)]
    keys = list(plan.get("group_by") or [])
    period = plan.get("period")
    if period:
        col = period["column"]
        stamps = _naive(pd.to_datetime(data[col], errors="coerce"))
        keep = stamps.notna()
        if period.get("from") is not None:
            keep &= stamps >= pd.to_datetime(period["from"])
        if period.get("to") is not None:
            keep &= stamps <= pd.to_datetime(period["to"]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        data = data[keep]
        if period.get("grain"):
            data = data.assign(period=stamps[keep].dt.to_period(_FREQ[period["grain"]]).astype(str))
            keys = ["period", *keys]
    matched = len(data)
    least = floor()
    from .validate import kinds as _kinds

    people = [c for c, kind in _kinds(frame).items() if kind == "identifier"]  # whose rows: a group needs as many of each
    extreme = {m["name"] for m in plan["measures"] if m["op"] in ("min", "max")
               or (plans.PERCENTILE.match(m["op"]) and not 5 <= int(plans.PERCENTILE.match(m["op"]).group(1)) <= 95)}
    withheld = 0

    def enough(part) -> bool:
        return len(part) >= least and all(part[c].nunique(dropna=True) >= least for c in people)

    def measures_of(part) -> dict[str, Any]:
        nonlocal withheld
        out: dict[str, Any] = {}
        for m in plan["measures"]:
            op = m["op"]
            if m["name"] in extreme and len(part) < EXTREME_FACTOR * least:
                out[m["name"]] = None  # one row's value in a small group: withheld, counted
                withheld += 1
                continue
            if op == "count":
                out[m["name"]] = int(len(part))
            elif op == "rate":
                out[m["name"]] = float(_mask(part, m["where"]).mean()) if len(part) else None
            else:
                s = part[m["column"]]
                if op == "count_distinct":
                    out[m["name"]] = int(s.nunique(dropna=True))
                elif op in ("sum", "mean", "median", "min", "max"):
                    out[m["name"]] = getattr(s, op)() if len(s.dropna()) else None
                else:
                    q = int(plans.PERCENTILE.match(op).group(1)) / 100
                    out[m["name"]] = s.quantile(q) if len(s.dropna()) else None
        return out

    rows: list[dict[str, Any]] = []
    suppressed = 0
    if keys:
        for key, part in data.groupby(keys, dropna=False, sort=True, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            if not enough(part):
                suppressed += 1
                continue
            rows.append({**dict(zip(keys, key)), **measures_of(part)})
    elif enough(data):
        rows.append(measures_of(data))
    else:
        suppressed = 1  # the whole answer would describe fewer rows than the floor
    def missing(v) -> bool:
        return v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT

    for s in reversed(plan.get("sort") or []):
        present = [r for r in rows if not missing(r.get(s["by"]))]
        absent = [r for r in rows if missing(r.get(s["by"]))]
        try:
            present.sort(key=lambda r: r.get(s["by"]), reverse=bool(s.get("desc")))
        except TypeError:  # mixed types in a group column: compare as text
            present.sort(key=lambda r: str(r.get(s["by"])), reverse=bool(s.get("desc")))
        rows = present + absent  # missing values last whatever the direction
    rows = rows[: plan.get("limit", 100)]
    columns = [*keys, *[m["name"] for m in plan["measures"]]]
    out = {"version": plans.SCHEMA_VERSION, "columns": columns, "rows": [[_cell(r.get(c)) for c in columns] for r in rows],
           "suppressed_groups": suppressed, "withheld_extremes": withheld, "rows_scanned": int(scanned), "floor": least,
           "plan_hash": plans.plan_hash(plan)}
    if not suppressed:
        out["rows_matched"] = int(matched)  # never beside a hidden group: an exact count would tell what the floor hid
    return out


class TableError(RuntimeError):
    pass


def table(store: Any, draft: dict[str, Any]):
    """The draft's cleaned table, only while it matches the fingerprint recorded when the pipeline wrote it."""
    import pandas as pd

    from ..storage.files import FileError, files_for

    asset = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset") and a.get("status") == "ready"), None)
    if asset is None:
        raise TableError("no table is ready yet")
    path = store.data_dir(draft["id"]) / asset.get("clean_file", "clean.parquet")
    try:  # the storage's own record of the file (written by the pipeline), and the fingerprint the asset kept
        files = files_for(store)
        checked = files.path(files.key_of(path), expected=asset.get("clean_sha256"))
    except FileError as exc:
        raise TableError(f"the prepared table cannot be read as it was written: {exc}") from None
    return pd.read_parquet(checked), asset


def exposure_entry(plan: dict[str, Any], result: dict[str, Any], asset: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"plan_hash": result["plan_hash"], "columns_related": [list(p) for p in plans.related(plan)],
            "asset": (asset or {}).get("id"), "table_sha256": (asset or {}).get("clean_sha256"),
            "rows": "filter" if plan.get("filters") or (plan.get("period") or {}).get("from") or (plan.get("period") or {}).get("to") else "all",
            "period": {k: (plan.get("period") or {}).get(k) for k in ("column", "from", "to")} if plan.get("period") else None,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def run(store: Any, draft_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Validate and execute a plan on the draft's table, and log what it related (the exposure log)."""
    draft = store.get(draft_id)
    frame, asset = table(store, draft)
    checked = validate(plan, frame)
    result = execute(checked, frame)
    entry = exposure_entry(checked, result, asset)
    store.update(draft_id, lambda d: d.setdefault("exposure", []).append(entry))
    return {**result, "plan": checked}


__all__ = ["execute", "run", "table", "floor", "exposure_entry", "PlanError", "TableError"]

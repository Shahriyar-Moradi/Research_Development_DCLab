"""A small feature language, compiled into fold-safe transformers (package 18.5).

A model may *propose* features; code decides what they mean and computes them. A program is a small JSON object:

    {"name": "biker_dropoff_repeat", "op": "count", "by": ["biker_id", {"place": "destination"}]}
    {"name": "biker_short_share", "op": "mean", "of": {"below": [{"distance": ["source", "destination"]}, 50]}, "by": ["biker_id"]}

Operations (``op``):

* ``count`` / ``share``: rows sharing the ``by`` keys (a share: over the rows this row may see);
* ``mean``: the mean of ``of`` over rows sharing the ``by`` keys; ``deviation``: the row's ``of`` minus that mean;
* ``ratio``: ``of`` over ``per`` (a floor keeps it finite); ``value``: an expression on its own;
* ``since_previous``: time since the previous row with the same ``by`` keys (past rows only, whatever the mode);
* ``rolling_mean``: the mean of ``of`` over the previous ``rows`` rows with the same ``by`` keys (past only; rows that
  share a timestamp are taken together, so the file's order never decides).

Expressions (``of``, ``per``, a ``by`` key): a column name; ``{"distance": [pair, pair]}`` (metres between two
coordinate pairs, named by prefix as in ``source_lat``/``source_lon``); ``{"place": pair}`` (a cell of about 11 m);
``{"day": true}`` (the row's day, from the time column); ``{"below": [expr, number]}`` / ``{"above": [expr, number]}``
(an indicator).

What makes it safe: :func:`check` refuses a program that names the target, a column the model may not use, or an
unknown operation; no operation reads a label (the language has no target rate); and a program is computed from the
frame it transforms, in the mode the prediction moment sets (``audit``: rows up to the end of the row's day; ``event``:
strictly earlier rows), so no test row changes a training row's feature and the event mode uses no later row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from dclab_core.behaviour import CELL_DIGITS, coordinate_pairs, _haversine

OPS = ("count", "share", "mean", "deviation", "ratio", "value", "since_previous", "rolling_mean")
MAX_PROGRAMS = 30
MAX_ROWS = 50


class ProgramError(ValueError):
    """A program the language refuses, and why."""


def _refs(expr: Any) -> list[str]:
    """The table columns an expression reads (pairs expand to their lat/lon columns later)."""
    if isinstance(expr, str):
        return [expr]
    if isinstance(expr, Mapping) and len(expr) == 1:
        (kind, arg), = expr.items()
        if kind in ("below", "above"):
            return _refs(arg[0]) if isinstance(arg, list) and len(arg) == 2 else []
        return []
    return []


class Language:
    """The columns a program may use, and the coordinate pairs and time column it may refer to."""

    def __init__(self, allowed: Sequence[str], *, target: Optional[str] = None, time_column: Optional[str] = None,
                 entities: Sequence[str] = ()):
        self.allowed = [str(c) for c in allowed]
        self.keys = set(self.allowed) | {str(e) for e in entities}  # entities group rows; they are never a feature value
        self.target, self.time_column = target, time_column
        self.pairs = {prefix: (lat, lon) for prefix, lat, lon in coordinate_pairs(self.allowed)}

    def _expr(self, expr: Any, as_key: bool = False) -> None:
        if isinstance(expr, str):
            if self.target is not None and expr == str(self.target):
                raise ProgramError("a program may not read the target")
            if expr not in (self.keys if as_key else set(self.allowed)):
                raise ProgramError(f"{expr!r} is not a column a model may use")
            return
        if not isinstance(expr, Mapping) or len(expr) != 1:
            raise ProgramError(f"not an expression: {str(expr)[:80]}")
        (kind, arg), = expr.items()
        if kind == "distance":
            if not (isinstance(arg, list) and len(arg) == 2 and all(a in self.pairs for a in arg)):
                raise ProgramError(f"distance needs two coordinate pairs of {sorted(self.pairs)}")
        elif kind == "place":
            if arg not in self.pairs:
                raise ProgramError(f"place needs a coordinate pair of {sorted(self.pairs)}")
        elif kind == "day":
            if not self.time_column:
                raise ProgramError("day needs a time column")
        elif kind in ("below", "above"):
            if not (isinstance(arg, list) and len(arg) == 2 and isinstance(arg[1], (int, float)) and not isinstance(arg[1], bool)):
                raise ProgramError(f"{kind} needs [expression, number]")
            self._expr(arg[0])
        else:
            raise ProgramError(f"unknown expression {kind!r}")

    def check(self, program: Mapping[str, Any]) -> dict[str, Any]:
        """The program, normalised, or a ProgramError saying why it is refused."""
        if not isinstance(program, Mapping):
            raise ProgramError("a program is an object")
        op = program.get("op")
        if op not in OPS:
            raise ProgramError(f"unknown operation {op!r}")
        name = str(program.get("name") or "").strip()
        if not name or len(name) > 80:
            raise ProgramError("a program needs a short name")
        by = program.get("by") or []
        if not isinstance(by, list) or len(by) > 3:
            raise ProgramError("by is a list of at most three keys")
        for key in by:
            self._expr(key, as_key=True)
        if op in ("count", "share", "since_previous") and not by:
            raise ProgramError(f"{op} needs by")
        if op in ("count", "share", "since_previous") and ("of" in program or "per" in program):
            raise ProgramError(f"{op} reads no of or per: say what it counts with by")
        if op != "ratio" and "per" in program:
            raise ProgramError("only ratio reads per")
        if op in ("mean", "deviation", "ratio", "value", "rolling_mean"):
            if "of" not in program:
                raise ProgramError(f"{op} needs of")
            self._expr(program["of"])
        if op in ("mean", "deviation", "rolling_mean") and not by:
            raise ProgramError(f"{op} needs by")
        if op == "ratio":
            if "per" not in program:
                raise ProgramError("ratio needs per")
            self._expr(program["per"])
        if op == "since_previous" and not self.time_column:
            raise ProgramError("since_previous needs a time column")
        rows = program.get("rows", 5)
        if op == "rolling_mean" and not (isinstance(rows, int) and 1 <= rows <= MAX_ROWS):
            raise ProgramError(f"rows is a whole number from 1 to {MAX_ROWS}")
        clean = {"name": name, "op": op, "by": list(by)}
        for k in ("of", "per"):
            if k in program:
                clean[k] = program[k]
        if op == "rolling_mean":
            clean["rows"] = rows
        return clean

    def check_all(self, programs: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """(the programs kept, the ones refused with the reason); names are made unique."""
        kept, refused, names = [], [], set()
        for program in list(programs)[:MAX_PROGRAMS]:
            try:
                clean = self.check(program)
            except ProgramError as exc:
                refused.append({"program": str(program)[:200], "why": str(exc)})
                continue
            if clean["name"] in names:
                refused.append({"program": str(program)[:200], "why": "a name used twice"})
                continue
            names.add(clean["name"])
            kept.append(clean)
        return kept, refused


class Programs:
    """The compiled transformer: fit learns nothing from labels; transform computes every program on the frame."""

    def __init__(self, language: Language, programs: Sequence[Mapping[str, Any]], mode: str = "event"):
        if mode not in ("audit", "event"):
            raise ValueError("mode is audit or event")
        self.language, self.mode = language, mode
        self.programs, self.refused = language.check_all(programs)

    def fit(self, frame: pd.DataFrame, y: Any = None) -> "Programs":
        """Learns no label: only a ratio's floor (1 % of its denominator's median size on the fit rows)."""
        f = frame.reset_index(drop=True)
        self.floors = {}
        for p in self.programs:
            if p["op"] == "ratio":
                den = pd.to_numeric(self._value(f, p["per"]), errors="coerce").to_numpy(dtype=float)
                self.floors[p["name"]] = max(float(np.nanmedian(np.abs(den))) * 0.01, 1e-9) if np.isfinite(den).any() else 1.0
        return self

    # ------------------------------------------------------------------ expressions
    def _value(self, f: pd.DataFrame, expr: Any) -> pd.Series:
        if isinstance(expr, str):
            return f[expr]
        (kind, arg), = expr.items()
        if kind == "distance":
            (lat1, lon1), (lat2, lon2) = self.language.pairs[arg[0]], self.language.pairs[arg[1]]
            return pd.Series(_haversine(f[lat1], f[lon1], f[lat2], f[lon2]), index=f.index)
        if kind == "place":
            lat, lon = self.language.pairs[arg]
            cell = f[lat].round(CELL_DIGITS).astype(str) + "_" + f[lon].round(CELL_DIGITS).astype(str)
            return cell.where(f[lat].notna() & f[lon].notna())
        if kind == "day":
            return pd.to_datetime(f[self.language.time_column], errors="coerce").dt.floor("D").astype(str)
        inner = pd.to_numeric(self._value(f, arg[0]), errors="coerce")
        return ((inner < arg[1]) if kind == "below" else (inner > arg[1])).astype(float).where(inner.notna())

    def _time(self, f: pd.DataFrame) -> Optional[pd.Series]:
        t = self.language.time_column
        return pd.to_datetime(f[t], errors="coerce") if t and t in f else None

    def _visible(self, f: pd.DataFrame, keys: pd.DataFrame, values: pd.Series, how: str) -> np.ndarray:
        """Sum or count over rows of the same keys this row may see: up to the end of its day (audit) or strictly
        earlier (event); without a time column, the frame (audit) or the rows above (event)."""
        keys = keys.astype(str).reset_index(drop=True)
        values = values.reset_index(drop=True).astype(float)
        cols = list(keys.columns)
        t = self._time(f)
        if t is None:  # no time: the window is the frame (audit), or the rows above (event)
            if self.mode == "audit":
                return values.groupby([keys[c] for c in cols], dropna=False).transform("sum").to_numpy(dtype=float)
            return (values.groupby([keys[c] for c in cols], dropna=False).cumsum() - values).to_numpy(dtype=float)
        stamp = (t.dt.floor("D") if self.mode == "audit" else t).reset_index(drop=True)
        return _window(keys, stamp, values, strict=self.mode == "event")

    def _keys(self, f: pd.DataFrame, by: Sequence[Any]) -> pd.DataFrame:
        return pd.DataFrame({f"k{i}": self._value(f, key).astype(str).to_numpy() for i, key in enumerate(by)})

    def _one(self, f: pd.DataFrame, p: Mapping[str, Any]) -> np.ndarray:
        op = p["op"]
        if op == "value":
            return pd.to_numeric(self._value(f, p["of"]), errors="coerce").to_numpy(dtype=float)
        if op == "ratio":
            num = pd.to_numeric(self._value(f, p["of"]), errors="coerce").to_numpy(dtype=float)
            den = pd.to_numeric(self._value(f, p["per"]), errors="coerce").to_numpy(dtype=float)
            floor = getattr(self, "floors", {}).get(p["name"], 1e-9)  # learned on the fit rows
            safe = np.where(np.abs(den) < floor, np.where(den < 0, -floor, floor), den)  # a tiny denominator is the floor, its sign kept
            return num / safe
        keys = self._keys(f, p["by"])
        ones = pd.Series(np.ones(len(f)))
        if op == "count":
            return self._visible(f, keys, ones, "sum")
        if op == "share":
            seen = self._visible(f, pd.DataFrame({"all": np.zeros(len(f))}), ones, "sum")
            return self._visible(f, keys, ones, "sum") / np.maximum(seen, 1.0)
        if op in ("mean", "deviation"):
            x = pd.to_numeric(self._value(f, p["of"]), errors="coerce").reset_index(drop=True)
            total = self._visible(f, keys, x.fillna(0), "sum")
            seen = self._visible(f, keys, x.notna().astype(float), "sum")
            mean = total / np.where(seen > 0, seen, np.nan)
            return mean if op == "mean" else x.to_numpy(dtype=float) - mean
        t = self._time(f)
        stamp = t.reset_index(drop=True) if t is not None else pd.Series(pd.to_datetime(np.arange(len(f)), unit="s"))
        x = None if op == "since_previous" else pd.to_numeric(self._value(f, p["of"]), errors="coerce").reset_index(drop=True)
        return _past(keys, stamp, x, op, int(p.get("rows", 5)))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One column per program, named ``prog__<name>`` (never colliding with a table column)."""
        f = frame.reset_index(drop=True)
        return pd.DataFrame({f"prog__{p['name']}": self._one(f, p) for p in self.programs}, index=f.index).astype(float)


def _window(keys: pd.DataFrame, stamp: pd.Series, values: pd.Series, strict: bool) -> np.ndarray:
    """Per row, the sum of ``values`` over rows of the same keys whose stamp is strictly earlier (``strict``) or on or
    before the row's stamp (a day, in audit mode). Rows with the same stamp never see each other in strict mode, and a
    row without a stamp gets no history (NaN)."""
    cols = list(keys.columns)
    frame = keys.assign(__t=stamp.to_numpy(), __v=values.to_numpy(dtype=float))
    known = frame["__t"].notna()
    out = np.full(len(frame), np.nan)
    if not known.any():
        return out
    sub = frame[known]
    per = sub.groupby(cols + ["__t"], dropna=False)["__v"].sum().rename("__tie").reset_index().sort_values(cols + ["__t"])
    per["__upto"] = per.groupby(cols, dropna=False)["__tie"].cumsum()
    merged = sub.reset_index().merge(per, on=cols + ["__t"], how="left").set_index("index")
    out[merged.index.to_numpy()] = (merged["__upto"] - merged["__tie"] if strict else merged["__upto"]).to_numpy(dtype=float)
    return out


def _past(keys: pd.DataFrame, stamp: pd.Series, x: Optional[pd.Series], op: str, rows: int) -> np.ndarray:
    """Time since the previous strictly earlier row with the same keys, or the mean of ``x`` over the last ``rows`` of
    them; rows with the same stamp are never each other's past, and a row without a stamp has none."""
    out = np.full(len(keys), np.nan)
    group = keys.astype(str).agg("|".join, axis=1).to_numpy()
    t = pd.to_datetime(stamp).to_numpy()
    for g in pd.unique(group):
        idx = np.flatnonzero(group == g)
        idx = idx[~pd.isna(t[idx])]
        if not len(idx):
            continue
        order = idx[np.argsort(t[idx], kind="stable")]
        times = t[order]
        earlier = np.searchsorted(times, times, side="left")  # rows strictly before each row's stamp
        for j, row in enumerate(order):
            n = earlier[j]
            if n == 0:
                continue
            if op == "since_previous":
                out[row] = (times[j] - times[n - 1]) / np.timedelta64(1, "s")
            else:
                start = max(0, n - rows)
                start = int(np.searchsorted(times, times[start], side="left"))  # whole timestamps: file order never decides
                window = x.to_numpy(dtype=float)[order[start:n]]
                out[row] = np.nanmean(window) if np.isfinite(window).any() else np.nan
    return out


def behaviour_programs(entities: Sequence[str], language: Language, numeric: Sequence[str] = ()) -> list[dict[str, Any]]:
    """15.4's behaviour recipe written in the language (without its label rates, which the language does not have):
    trip distance and shortness, money per metre, counts per entity, day, drop-off place and route, entity means."""
    pairs = list(language.pairs)
    out: list[dict[str, Any]] = []
    distance = {"distance": pairs[:2]} if len(pairs) >= 2 else None
    if distance:
        out.append({"name": "trip_metres", "op": "value", "of": distance})
        out.append({"name": "route_share", "op": "share", "by": [{"place": pairs[0]}, {"place": pairs[1]}]})
        for column in [c for c in numeric if c in language.allowed][:4]:
            out.append({"name": f"{column}_per_metre", "op": "ratio", "of": column, "per": distance})
    for entity in entities:
        out.append({"name": f"{entity}_count", "op": "count", "by": [entity]})
        if language.time_column:
            out.append({"name": f"{entity}_day_count", "op": "count", "by": [entity, {"day": True}]})
        if pairs:
            out.append({"name": f"{entity}_{pairs[-1]}_repeat", "op": "count", "by": [entity, {"place": pairs[-1]}]})
        if distance:
            out.append({"name": f"{entity}_short_share", "op": "mean", "of": {"below": [distance, 50]}, "by": [entity]})
    return out


# ------------------------------------------------------------------ the feature engineer (a role)

def _engineer_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    from dclab_core.analyse import _kind

    df, language = inputs["frame"], inputs["language"]
    usable = [c for c in df.columns if c in language.keys][:150]
    lines = [f"{str(c)[:200]} | {_kind(df[c])} | missing {float(df[c].isna().mean()):.2f} | distinct {float(df[c].nunique()) / max(len(df), 1):.2f}"
             for c in usable]
    return {"prediction_moment": str(inputs.get("moment") or "not stated")[:600], "task": inputs.get("task"),
            "target": str(language.target or "")[:200], "entities": [str(e) for e in inputs.get("entities", [])][:5],
            "time_column": language.time_column, "coordinate_pairs": sorted(language.pairs)[:10],
            "goal": str(inputs.get("goal") or "")[:600],
            "columns": {f"{i + 1}-{i + len(chunk)}": chunk for i, chunk in ((i, lines[i:i + 50]) for i in range(0, len(lines), 50))}}


def _engineer_parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    programs = answer.get("programs")
    if not isinstance(programs, list):
        raise ValueError("programs must be a list")
    kept, refused = inputs["language"].check_all(programs)
    return {"programs": kept, "refused": refused}


def _engineer_role():
    from dclab_core.agents import Role

    return Role(name="feature_engineer",
                description="Proposes feature programs in the feature language from the column names, the moment and the goal; "
                            "code checks and computes them, and the feature ladder keeps them only for a measured gain.",
                evidence=_engineer_evidence, parse=_engineer_parse, fallback=lambda inputs: {"programs": [], "refused": []})


FEATURE_ENGINEER = _engineer_role()


def engineer(advisor: Any, frame: pd.DataFrame, language: Language, *, moment: str = "", task: str = "", goal: str = "",
             entities: Sequence[str] = ()) -> dict[str, Any]:
    """The feature engineer's checked programs (empty with no advisor), with the refused ones and the reason for each."""
    from dclab_core.agents import run_role

    result = run_role(FEATURE_ENGINEER, advisor, {"frame": frame, "language": language, "moment": moment, "task": task,
                                                  "goal": goal, "entities": list(entities)})
    return {**result.output, "source": result.source, "reason": result.reason}

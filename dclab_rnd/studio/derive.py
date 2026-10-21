"""The Data engineer's changes to a project's table, applied only to a new version (the built project never changes).

Two kinds of change, each a small record kept on the version (``data_ops``) and applied when it is planned and built:

- ``add_column``: a new column from a formula over the table's own columns. A formula is parsed, never run as code: column
  names, numbers, ``+ - * / // %  **``, comparisons, ``and``/``or``/``not`` and a few functions (``FUNCTIONS``). A formula
  that reads the outcome, a forbidden column or a leak suspect is refused: what it adds would be known only after the
  prediction moment, or would carry the answer.
- ``join``: the columns of another table looked up by a key. Only a lookup is allowed: the key must be unique in the other
  table, so each row gets at most one match and no row is repeated. An event table (many rows per key: orders, tickets) is
  refused, because summing its rows needs a time cutoff at the prediction moment, or it leaks what happened after.

Every check reads column names and counts, never a value.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,59}$")
MAX_FORMULA = 300
FUNCTIONS = ("log1p", "abs", "sqrt", "round", "days_between", "year", "month", "weekday", "hour", "is_missing")
RAW_ARGS = ("days_between", "year", "month", "weekday", "hour", "is_missing")  # read the column as it is, not as a number
BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.FloorDiv: np.floor_divide,
          ast.Mod: np.mod, ast.Pow: np.power}
COMPARE = {ast.Gt: np.greater, ast.Lt: np.less, ast.GtE: np.greater_equal, ast.LtE: np.less_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}


class ChangeRefused(ValueError):
    """A change that cannot be made, said in words the person can act on."""


# ---------------------------------------------------------------------- formulas


def _quoted(formula: str) -> tuple[str, dict[str, str]]:
    """`column name` in backticks becomes a placeholder name, so a column with spaces can be named."""
    names: dict[str, str] = {}

    def swap(m: re.Match) -> str:
        key = f"__c{len(names)}"
        names[key] = m.group(1)
        return key
    return re.sub(r"`([^`]{1,200})`", swap, formula), names


def parse(formula: str, columns: Iterable[str]) -> tuple[ast.Expression, set[str]]:
    """The checked formula and the columns it reads. Raises ChangeRefused for anything but the allowed parts."""
    text = " ".join(str(formula or "").split())
    if not text:
        raise ChangeRefused("Give the formula, for example: spend / tenure_months")
    if len(text) > MAX_FORMULA:
        raise ChangeRefused(f"A formula is at most {MAX_FORMULA} characters")
    text, quoted = _quoted(text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ChangeRefused("The formula does not read: use column names, numbers, + - * / and the functions " + ", ".join(FUNCTIONS)) from None
    known, used = set(map(str, columns)), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = quoted.get(node.id, node.id)
            if name in FUNCTIONS and _is_called(tree, node):
                continue
            if name not in known:
                raise ChangeRefused(f"There is no column named {name!r} in this table")
            used.add(name)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ChangeRefused("Only these functions can be used: " + ", ".join(FUNCTIONS))
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ChangeRefused("Only numbers can be written in a formula, not text")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Load, ast.And, ast.Or, ast.Not,
                                   ast.USub, ast.UAdd, *BINARY, *COMPARE)):
            raise ChangeRefused(f"A formula cannot use {type(node).__name__.lower()}: only column names, numbers, operators and "
                                + ", ".join(FUNCTIONS))
    if not used:
        raise ChangeRefused("The formula reads no column")
    return _rename(tree, quoted), used


def _is_called(tree: ast.AST, name: ast.Name) -> bool:
    return any(isinstance(n, ast.Call) and n.func is name for n in ast.walk(tree))


def _rename(tree: ast.Expression, quoted: dict[str, str]) -> ast.Expression:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in quoted:
            node.id = quoted[node.id]
    return tree


def _times(values: pd.Series) -> pd.Series:
    """Dates from a column; a time whose year was not written (pandas 3 reads it as year 1) is missing."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stamps = pd.to_datetime(values, errors="coerce", format="mixed")
    return stamps.mask(stamps.dt.year < 1000) if pd.api.types.is_datetime64_any_dtype(stamps) else pd.Series(pd.NaT, index=values.index)


def _call(name: str, args: list[Any]) -> Any:
    one = args[0] if args else None
    if name == "days_between":
        if len(args) != 2:
            raise ChangeRefused("days_between takes two date columns: days_between(start, end)")
        return (_times(pd.Series(args[1])) - _times(pd.Series(args[0]))).dt.total_seconds() / 86400.0
    if len(args) != (2 if name == "round" and len(args) == 2 else 1):
        raise ChangeRefused(f"{name} takes one column")
    if name in ("year", "month", "weekday", "hour"):
        stamps = _times(pd.Series(one))
        return {"year": stamps.dt.year, "month": stamps.dt.month, "weekday": stamps.dt.weekday, "hour": stamps.dt.hour}[name].astype(float)
    if name == "is_missing":
        return pd.Series(one).isna().astype(float)
    number = pd.to_numeric(pd.Series(one), errors="coerce")
    if name == "round":
        return number.round(int(args[1]) if len(args) == 2 else 0)
    return {"log1p": np.log1p, "abs": np.abs, "sqrt": np.sqrt}[name](number)


def compute(frame: pd.DataFrame, formula: str) -> pd.Series:
    """The new column's values, one per row of ``frame``; a division by zero or an invalid number becomes missing."""
    tree, _ = parse(formula, frame.columns)

    def ev(node: ast.AST, raw: bool = False) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if raw and isinstance(node, ast.Name):  # is_missing and the date functions read the column as it is (text dates, text categories)
            return frame[node.id]
        if isinstance(node, ast.Name):
            col = frame[node.id]
            return col if pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_bool_dtype(col) else pd.to_numeric(col, errors="coerce")
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            return np.logical_not(value) if isinstance(node.op, ast.Not) else (-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BinOp):
            return BINARY[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Compare):
            out, left = None, ev(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = ev(right_node)
                step = COMPARE[type(op)](left, right)
                out, left = (step if out is None else np.logical_and(out, step)), right
            return out
        if isinstance(node, ast.BoolOp):
            values = [ev(v) for v in node.values]
            join = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            out = values[0]
            for v in values[1:]:
                out = join(out, v)
            return out
        if isinstance(node, ast.Call):
            raw_args = node.func.id in RAW_ARGS
            return _call(node.func.id, [ev(a, raw=raw_args) for a in node.args])
        raise ChangeRefused(f"A formula cannot use {type(node).__name__.lower()}")

    try:
        with np.errstate(all="ignore"):
            value = ev(tree)
    except ChangeRefused:
        raise
    except Exception:  # noqa: BLE001 - numpy's own refusals (an integer to a negative power, a type it cannot add)
        raise ChangeRefused("The formula cannot be computed on these columns: check their types") from None
    series = pd.Series(value, index=frame.index) if not isinstance(value, pd.Series) else value.reindex(frame.index)
    if pd.api.types.is_bool_dtype(series):
        series = series.astype(float)
    series = pd.to_numeric(series, errors="coerce").astype(float)
    return series.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------- checks against the problem


def _outcomes(target: Any) -> set[str]:
    return {target} if isinstance(target, str) else {str(t) for t in (target or []) if t}


def blocked_inputs(used: set[str], target: Any, forbidden: Iterable[str], suspects: Iterable[str] = ()) -> list[str]:
    """Why the columns a change reads cannot be used, one reason a column (empty when they can). ``target``: the outcome, or
    every outcome the versions under way name."""
    out, outcomes = [], _outcomes(target)
    for col in sorted(used):
        if col in outcomes:
            out.append(f"{col} is the outcome: a column built from it carries the answer")
        elif col in set(forbidden):
            out.append(f"{col} is forbidden in this project (unknown at the prediction moment)")
        elif col in set(suspects):
            out.append(f"{col} is a leak suspect in the data card: confirm it is known at the prediction moment first")
    return out


def check_add(frame: pd.DataFrame, name: str, formula: str, target: Optional[str], forbidden: Iterable[str],
              suspects: Iterable[str] = ()) -> dict[str, Any]:
    """A new column checked and measured (counts and its missing share only): raises ChangeRefused when it cannot be made."""
    name = str(name or "").strip()
    if not NAME.match(name):
        raise ChangeRefused("Name the new column with letters, digits and underscores, starting with a letter (at most 60)")
    if name in frame.columns:
        raise ChangeRefused(f"The table already has a column named {name}")
    _, used = parse(formula, frame.columns)
    blocked = blocked_inputs(used, target, forbidden, suspects)
    if blocked:
        raise ChangeRefused("; ".join(blocked))
    values = compute(frame, formula)
    missing = float(values.isna().mean()) if len(values) else 1.0
    if missing >= 0.999:
        raise ChangeRefused("The formula gives no value on any row: check the columns it reads")
    return {"op": "add_column", "name": name, "formula": " ".join(str(formula).split()), "reads": sorted(used),
            "missing_share": round(missing, 3), "distinct": int(values.nunique())}


def content(frame: pd.DataFrame) -> str:
    """A fingerprint of a table's content (values and column names), to know it is the table that was checked."""
    import hashlib

    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    digest.update("\x1f".join(map(str, frame.columns)).encode())
    return digest.hexdigest()[:24]


def check_join(frame: pd.DataFrame, other: pd.DataFrame, table: str, on: str, target: Any,
               forbidden: Iterable[str], suspects: Iterable[str] = ()) -> dict[str, Any]:
    """A lookup join checked (key, uniqueness, overlap, names): raises ChangeRefused when it cannot be made."""
    blocked = blocked_inputs({on}, target, forbidden, suspects)
    if blocked:  # a key that is the outcome (or after the moment) makes every joined column a copy of it
        raise ChangeRefused("; ".join(blocked))
    if on not in frame.columns:
        raise ChangeRefused(f"This project's table has no column {on!r} to join on")
    if on not in other.columns:
        raise ChangeRefused(f"The table {table} has no column {on!r} to join on")
    repeats = int(other[on].duplicated().sum())
    if repeats:
        raise ChangeRefused(f"{on} repeats in {table} ({repeats} rows share a value): that is a table of events, not a lookup. Joining it means "
                            "summing its rows up to the prediction moment, with a time cutoff, or it leaks what happened after; that is not done here yet")
    added = [c for c in other.columns if c != on]
    if not added:
        raise ChangeRefused(f"{table} has no column besides {on}")
    named = sorted(_outcomes(target) & set(added))
    if named:
        raise ChangeRefused(f"{table} has a column named like the outcome ({', '.join(named)}): it would carry the answer")
    clash = [c for c in added if c in frame.columns]
    taken = [f"{table}_{c}" for c in clash if f"{table}_{c}" in frame.columns]
    if taken:
        raise ChangeRefused(f"{table}'s columns would land on columns this table already has ({', '.join(taken[:4])}): rename them in {table} first")
    blocked = [c for c in added if c in set(forbidden)]
    if blocked:
        raise ChangeRefused(f"{table} brings columns this project forbids: {', '.join(blocked[:6])}")
    matched = float(frame[on].isin(set(other[on].dropna())).mean()) if len(frame) else 0.0
    if matched == 0:
        raise ChangeRefused(f"No row of this table finds its {on} in {table}: check the key")
    return {"op": "join", "table": table, "on": on, "columns": [f"{table}_{c}" if c in clash else c for c in added][:200],
            "source_columns": added[:200], "content": content(other), "matched_share": round(matched, 3)}


# ---------------------------------------------------------------------- applying a version's changes


def apply(frame: pd.DataFrame, ops: list[dict[str, Any]], table: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    """The table with the version's changes, in order (``table(name)`` reads a joined table). The row order and count stay."""
    out = frame.copy()
    for op in ops or []:
        if op.get("op") == "add_column":
            out[op["name"]] = compute(out, op["formula"])
        elif op.get("op") == "join":
            other = table(op["table"])
            if op.get("content") and content(other) != op["content"]:
                raise ChangeRefused(f"{op['table']} changed since the join was confirmed: propose the join again")
            keep = [op["on"], *(op.get("source_columns") or [c for c in other.columns if c != op["on"]])]
            if any(c not in other.columns for c in keep):
                raise ChangeRefused(f"{op['table']} no longer has the columns that were checked")
            other = other[keep]
            if other[op["on"]].duplicated().any():
                raise ChangeRefused(f"{op['table']} changed: {op['on']} repeats in it now")
            names = dict(zip(op.get("source_columns") or [], op.get("columns") or [])) or \
                {c: f"{op['table']}_{c}" if c in out.columns else c for c in other.columns if c != op["on"]}
            if any(n in out.columns for n in names.values()):
                raise ChangeRefused(f"{op['table']}'s columns would land on columns the table already has")
            other = other.rename(columns=names)
            key = op["on"]
            joined = out[[key]].merge(other, on=key, how="left", validate="many_to_one")
            for c in joined.columns:
                if c != key:
                    out[c] = joined[c].to_numpy()
        else:
            raise ChangeRefused(f"Unknown change {op.get('op')!r}")
    return out


def describe(op: dict[str, Any]) -> str:
    """One line for the person (names and counts only)."""
    if op.get("op") == "add_column":
        return f"add {op['name']} = {op['formula']} (missing on {round(100 * op.get('missing_share', 0))}% of rows)"
    if op.get("op") == "join":
        cols = op.get("columns") or []
        return (f"join {op['table']} on {op['on']}: {len(cols)} column{'s' if len(cols) != 1 else ''} ({', '.join(cols[:5])}{', …' if len(cols) > 5 else ''}), "
                f"found for {round(100 * op.get('matched_share', 0))}% of rows")
    return str(op.get("op"))


__all__ = ["ChangeRefused", "FUNCTIONS", "parse", "compute", "blocked_inputs", "check_add", "check_join", "content", "apply", "describe"]

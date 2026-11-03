"""One shape for every evaluation result (DCLab Bench, B1 of the 2026-11-03 plan), and the rule that reads a layer.

The suites that measure DCLab (agents, journeys, the lead-agent exam, decision models, retrieval, the engine's kinds ...)
each wrote their result in their own shape: a verdict as a string ("pass: 17 of 17"), as a dict with a ``status``, with a
``pass`` flag or a ``passed`` one, as a top-level ``result``, or not at all. :func:`verdict_of` reads every one of them into
``passed`` = True, False or None ("not scored"); a shape it does not know is None, never a silent failure. :func:`rows_of`
turns one result into :class:`BenchRow` rows: its verdict, its counts (``[k, n]`` pairs with their 95% interval, the one
the suite stored or Wilson's) and its numeric summary. :func:`layer_status` reads the latest result of each suite of a
layer: red when one of them failed, amber when none failed but one is unscored, thin when too few cases back it, green
otherwise. There is no weighted total: a red layer is never averaged away.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Versioned, _version

LAYERS: tuple[str, ...] = ("software", "agents", "lead_agent", "llm_roles", "team", "decision_models", "retrieval_memory",
                           "data_science", "system_tasks", "external")
SPLITS: tuple[str, ...] = ("dev", "sealed", "test", "live", "unknown")
STATUSES: tuple[str, ...] = ("green", "amber", "red", "thin", "empty")
THIN = 20  # fewer cases than this behind a layer's latest results and it is thin, whatever they say

_PASS = re.compile(r"^\s*(pass|passed|ok)\b", re.I)
_FAIL = re.compile(r"^\s*(fail|failed)\b", re.I)


class BenchRow(Versioned):
    """One measurement of one evaluation result: its verdict, a count with its interval, or a number of its summary."""

    schema_version: Literal["dclab_core.bench_row.v1"] = _version("bench_row")
    suite: str = Field(max_length=80, description="The suite (an evidence campaign, or a bench layer's own run).")
    ref: str = Field(max_length=120, description="The result's id (its file name).")
    layer: Literal[LAYERS]
    split: Literal[SPLITS] = "unknown"
    metric: str = Field("verdict", max_length=120)
    k: Optional[int] = Field(None, ge=0)
    n: Optional[int] = Field(None, ge=0)
    value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    passed: Optional[bool] = Field(None, description="The verdict: True, False, or None when the result was not scored.")
    verdict: str = Field("", max_length=200, description="The result's own verdict words, cut.")
    gate: str = Field("", max_length=80)
    cost_eur: Optional[float] = Field(None, ge=0)
    seconds: Optional[float] = Field(None, ge=0)
    model: str = Field("", max_length=120)
    prompt: str = Field("", max_length=80)
    commit: str = Field("", max_length=40)
    at: str = Field("", max_length=40)


CONTRACTS["bench_row"] = BenchRow


def _read(words: str) -> Optional[bool]:
    if re.match(r"^\s*ok\b.*\bbaseline\b", words, re.I):
        return None  # "ok: baseline recorded" is a measurement kept, not a gate passed
    if _PASS.match(words):
        return True
    if _FAIL.match(words):
        return False
    return None


def verdict_of(doc: Mapping[str, Any]) -> tuple[Optional[bool], str]:
    """The result's verdict as (passed, its words). Known shapes: ``verdict`` a string or a dict with ``pass``,
    ``passed`` or ``status``; else a top-level ``result`` string or ``passed`` flag; else ``passes`` (the policies that
    passed: none is a failure). Anything else is (None, ""): not scored."""
    v = doc.get("verdict")
    if isinstance(v, bool):
        return v, str(v).lower()
    if isinstance(v, str):
        return _read(v), v[:200]
    if isinstance(v, Mapping):
        for key in ("pass", "passed"):
            if isinstance(v.get(key), bool):
                gate = v.get("gate") if isinstance(v.get("gate"), str) else ""
                return v[key], ("pass" if v[key] else "fail") + (f" ({gate})" if gate else "")
        if isinstance(v.get("status"), str):
            return _read(v["status"]), v["status"][:200]
        words = {k: x for k, x in v.items() if isinstance(x, str)}  # {"pooled": "fail", "daily": "pass"}: one verdict per part
        reads = [_read(x) for x in words.values()]
        if words and all(r is not None for r in reads):
            return all(reads), "; ".join(f"{k}: {x}" for k, x in words.items())[:200]
        return None, ""
    if isinstance(doc.get("passed"), bool):
        return doc["passed"], str(doc["passed"]).lower()
    if isinstance(doc.get("result"), str):
        return _read(doc["result"]), doc["result"][:200]
    if isinstance(doc.get("passes"), list):
        return bool(doc["passes"]), f"{len(doc['passes'])} passed"
    return None, ""


def split_of(doc: Mapping[str, Any], name: str = "") -> str:
    """dev, sealed, test, live or unknown: the result's own ``split`` (a sealed version counts as sealed), else its name."""
    for value in (doc.get("split"), (doc.get("gate") or {}).get("split") if isinstance(doc.get("gate"), Mapping) else None, name):
        words = str(value or "").lower()
        for split in ("sealed", "test", "dev", "live"):
            if split in words:
                return split
    return "unknown"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if not n:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def _pair(value: Any) -> Optional[tuple[int, int]]:
    if (isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
            and 0 <= value[0] <= value[1] and value[1] > 0):
        return int(value[0]), int(value[1])
    return None


def _interval(value: Any) -> Optional[tuple[float, float]]:
    if (isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
            and all(math.isfinite(float(x)) for x in value)):
        return float(value[0]), float(value[1])
    return None


DEPTH = 3  # how deep counts are looked for (``summary.audit.leaks_caught`` is three down)
MAX_COUNTS = 200  # a result with more counts than this keeps its first ones (a per-case table is not a measurement)


def counts(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every ``[k, n]`` count of the result, up to ``DEPTH`` levels into its parts, with its interval: ``<name>_ci95`` or
    ``<name>_ci`` beside it when the suite stored one, else Wilson's 95%; and every finite number at the top of its
    ``summary``."""
    out: list[dict[str, Any]] = []

    def walk(part: Mapping[str, Any], prefix: str, depth: int) -> None:
        for key, value in part.items():
            if len(out) >= MAX_COUNTS:
                return
            name = f"{prefix}{key}"[:120]
            pair = _pair(value)
            if pair:
                stored = _interval(part.get(f"{key}_ci95")) or _interval(part.get(f"{key}_ci"))
                lo, hi = stored or wilson(*pair)
                out.append({"metric": name, "k": pair[0], "n": pair[1], "value": round(pair[0] / pair[1], 4), "lower": lo, "upper": hi})
            elif prefix == "summary." and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                out.append({"metric": name, "value": float(value)})
            elif isinstance(value, Mapping) and depth < DEPTH:
                walk(value, f"{name}.", depth + 1)

    walk(doc, "", 1)
    return out


def _text(value: Any, limit: int) -> str:
    if isinstance(value, Mapping):
        value = next((v for v in value.values() if isinstance(v, str)), "")
    return str(value or "")[:limit]


def rows_of(doc: Mapping[str, Any], *, suite: str, ref: str, layer: str, at: str = "", commit: str = "") -> list[BenchRow]:
    """One result as rows: its verdict first, then its counts and summary numbers, each carrying the result's split,
    gate, cost, model and prompt."""
    passed, words = verdict_of(doc)
    gate = doc.get("gate")
    base = {"suite": suite[:80], "ref": ref[:120], "layer": layer, "split": split_of(doc, ref),
            "gate": _text(gate.get("version") if isinstance(gate, Mapping) else gate, 80),
            "cost_eur": _cost(doc), "model": _text(doc.get("model") or doc.get("models"), 120),
            "prompt": _text(doc.get("prompt") or doc.get("prompt_hashes"), 80),
            "commit": commit[:40], "at": str(at or doc.get("at") or doc.get("completed_at") or "")[:40]}
    rows = [BenchRow(**base, metric="verdict", passed=passed, verdict=words)]
    rows += [BenchRow(**base, **c) for c in counts(doc)]
    return rows


def _cost(doc: Mapping[str, Any]) -> Optional[float]:
    for key in ("spent_eur", "cost_eur", "eur"):
        value = doc.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            return float(value)
    return None


_ID = re.compile(r"^[A-Za-z0-9]+-(\d+)(?:_(.*))?$")
_SPLIT_WORDS = re.compile(r"(?:^|_)(?:sealed|dev|test|live)(?:_v\d+)?(?=_|$)|(?:^|_)v\d+(?=_|$)")


def track_of(ref: str, split: str = "unknown") -> tuple[str, str]:
    """What a result is a result of, inside its suite: its split and its point (the name after the id, without split and
    version words). ``DMG-014_column.semantic_role`` is the point ``column.semantic_role``; ``JRN-025_sealed_v6`` and
    ``JRN-024_sealed_v6`` are the same sealed track, ``JRN-027_dev`` another. A later dev result never hides a failed sealed
    one, and one decision point's gate never hides another's."""
    m = _ID.match(ref)
    rest = (m.group(2) or "") if m else ""
    return split, _SPLIT_WORDS.sub("", rest).strip("_")


def order_of(row: "BenchRow") -> tuple[int, str, str]:
    """Newest last: the result's number in its suite first (ids grow with time; 17 results carry no date), then its date."""
    m = _ID.match(row.ref)
    return (int(m.group(1)) if m else -1, row.at, row.ref)


def layer_status(rows: Iterable[BenchRow], thin: int = THIN) -> dict[str, Any]:
    """A layer's state from its rows: the latest result of each track of each suite (:func:`track_of`, newest by
    :func:`order_of`) is what counts. Red when one of them failed; amber when none failed and one is not scored; thin when
    the cases behind the scored ones are fewer than ``thin`` (the largest count of each result); green otherwise; empty
    with no result. Each reason names its result."""
    rows = list(rows)
    latest: dict[tuple[str, tuple[str, str]], BenchRow] = {}
    for r in rows:
        if r.metric != "verdict":
            continue
        key = (r.suite, track_of(r.ref, r.split))
        if key not in latest or order_of(r) > order_of(latest[key]):
            latest[key] = r
    if not latest:
        return {"status": "empty", "reasons": [], "suites": 0, "cases": 0}
    keep = {(r.suite, r.ref) for r in latest.values()}
    verdicts = [r for r in rows if r.metric == "verdict" and (r.suite, r.ref) in keep]
    failed = [r for r in verdicts if r.passed is False]
    unscored = [r for r in verdicts if r.passed is None]
    cases = sum(max([x.n for x in rows if (x.suite, x.ref) == (r.suite, r.ref) and x.n] or [0]) for r in verdicts if r.passed is not None)
    reasons = [f"{r.suite} {r.ref}: failed ({r.verdict[:80]})" for r in failed]
    reasons += [f"{r.suite} {r.ref}: not scored (the result has no verdict of a gate)" for r in unscored]
    if failed:
        status = "red"
    elif unscored:
        status = "amber"
    elif cases < thin:
        status = "thin"
        reasons.append(f"{cases} cases behind the latest results, fewer than {thin}")
    else:
        status = "green"
    return {"status": status, "reasons": reasons, "suites": len({s for s, _ in latest}), "tracks": len(latest), "cases": cases,
            "passed": sum(r.passed is True for r in verdicts), "failed": len(failed), "unscored": len(unscored)}


__all__ = ["BenchRow", "LAYERS", "SPLITS", "STATUSES", "THIN", "verdict_of", "split_of", "counts", "rows_of", "track_of", "order_of",
           "layer_status", "wilson"]

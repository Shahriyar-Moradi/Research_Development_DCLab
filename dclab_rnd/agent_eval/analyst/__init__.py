"""The problem analyst, measured (package 18.2): does a model reading the description make the brief better, and never worse?

Twenty-four written problem descriptions with the fields a careful reader would take from them (``briefs_dev.json``
and ``briefs_sealed.json``; the sealed set's fingerprint is committed and checked, and it is run once). Each brief is
read three ways:

* **rules**: ``dclab_core.understand`` alone;
* **analyst**: the same rules, then the analyst role for the fields they left unknown, every proposal checked
  (the person's whole words, a quote that says what the value says, a column that exists);
* **plain**: the same model with the same prompt asked for every field, its answers taken as they come: what a
  prompted model gives without the rules and the checks.

Every field of every brief is scored: *right* (a value that matches), *wrong* (a value that does not, or any value where
the description supports none), *missed* (no value where one was expected) or *abstained* (no value, rightly). The gate
(``GATE``, fixed before any run) asks, on the sealed set: more fields right with the analyst than with the rules, and
no field wrong that the rules had left unknown. A run in which a request failed is reported, never scored.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "analyst_v1" / "results"
SPLITS = {"dev": HERE / "briefs_dev.json", "sealed": HERE / "briefs_sealed.json"}
FIELDS = ("goal", "action", "unit", "task", "target_column", "prediction_moment", "horizon", "primary_metric")
ARMS = ("rules", "analyst", "plain")

GATE = {
    "version": "analyst_gate_v1",  # fixed on 2026-10-12, before any live run
    "split": "sealed",
    "more_right": "fields right with the analyst > fields right with the rules alone",
    "no_new_wrong": "fields wrong with the analyst that the rules had left unknown == 0",
    "rules_kept": "fields the rules knew that the analyst changed == 0",
}


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_fingerprint() -> str:
    return (HERE / "sealed.sha256").read_text(encoding="utf-8").strip()


def load(split: str) -> list[dict[str, Any]]:
    path = SPLITS[split]
    if split == "sealed" and fingerprint(path) != recorded_fingerprint():
        raise ValueError("the sealed briefs changed after they were sealed: a sealed set is never edited")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ scoring


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _word_in(word: str, tokens: list[str]) -> bool:
    if len(word) <= 3:
        return word in tokens or word + "s" in tokens  # short words (pm, mae, roc, f1) match whole, or as a plural
    stem = word[:5]
    return any(t.startswith(stem) for t in tokens)


# A clause that says what is *not* known yet ("at checkout, before delivery") is not the moment it names: it is left
# out before the gold's forbidden phrases are looked for. A value that is only such a clause is read whole.
NOT_YET = re.compile(r"\b(before|until|prior to|ahead of|without|not)\b[^,;.]*", re.I)


def _plain(value: str) -> str:
    return value.strip().strip("`'\"").strip().casefold()


def _phrase_in(phrase: str, tokens: list[str]) -> bool:
    return all(_word_in(w, tokens) for w in _tokens(phrase))


def score(value: Optional[str], gold: Any, field: str = "") -> str:
    """right, wrong, missed or abstained for one field."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return "abstained" if gold is None else "missed"
    if gold is None:
        return "wrong"  # the description supports no value here
    if isinstance(gold, str):
        return "right" if _plain(str(value)) == gold.casefold() else "wrong"  # "Binary" or `cancelled` is not an error of judgement
    core = NOT_YET.sub(" ", str(value)) if field == "prediction_moment" else str(value)
    tokens = _tokens(core) or _tokens(str(value))
    if any(_phrase_in(p, tokens) for p in gold.get("none", [])):
        return "wrong"
    return "right" if any(_phrase_in(p, tokens) for p in gold.get("any", [])) else "wrong"


# ------------------------------------------------------------------------------------------------ the arms


def _values(brief: Any) -> dict[str, Optional[str]]:
    out = {}
    for field in FIELDS:
        fact = getattr(brief, field)
        out[field] = str(fact.value) if fact.known and fact.value is not None else None
    return out


def read_rules(case: Mapping[str, Any]) -> dict[str, Optional[str]]:
    from dclab_core.understand import understand

    return _values(understand(case["description"], case["files"]))


def read_analyst(case: Mapping[str, Any], advisor: Any) -> dict[str, Optional[str]]:
    from dclab_core.understand import understand

    return _values(understand(case["description"], case["files"], advisor=advisor))


def read_plain(case: Mapping[str, Any], advisor: Any) -> dict[str, Optional[str]]:
    """The same model and prompt asked for every field; its values taken as given (no quote, no column check)."""
    from dclab_core.understand import _analyst_evidence, _norm, _sentences

    text = " ".join(_sentences(case["description"]))
    evidence = _analyst_evidence({"description": case["description"], "text": text, "files": case["files"], "unknown": list(FIELDS)})
    from dclab_core.advisor import consult

    failures: list[BaseException] = []
    answer = consult(advisor, "problem_analyst", evidence, parse=lambda a: a,  # the evidence is checked as for the role
                     on_error=lambda kind, exc: failures.append(exc or ValueError(kind))) or {}
    if failures:
        raise failures[0]
    out: dict[str, Optional[str]] = {}
    for field in FIELDS:
        item = answer.get(field)
        value = item.get("value") if isinstance(item, Mapping) else item
        out[field] = _norm(value) if isinstance(value, str) and value.strip() else None
    return out


# ------------------------------------------------------------------------------------------------ a run


class Counting:
    """An advisor that counts its requests and the failed ones (a failed request makes a run unscored)."""

    def __init__(self, inner: Any):  # failures: a request that raised (cut at the token limit, no JSON, refused by a cap)
        self.inner, self.requests, self.failures = inner, 0, []

    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Any:
        if self.failures:
            raise RuntimeError("stopped: an earlier request failed, so this run will not be scored")
        self.requests += 1
        try:
            return self.inner.ask(purpose, evidence)
        except Exception as error:  # noqa: BLE001 — recorded, then the role falls back as it would in the product
            self.failures.append(f"{type(error).__name__}: {str(error)[:160]}")
            raise


def measure(cases: list[dict[str, Any]], advisor: Optional[Any]) -> dict[str, Any]:
    """Every case read by every arm (only the rules when there is no advisor), scored field by field."""
    counting = Counting(advisor) if advisor is not None else None
    rows = []
    for case in cases:
        values = {"rules": read_rules(case)}
        if counting is not None:
            values["analyst"] = read_analyst(case, counting)
            try:
                values["plain"] = read_plain(case, counting)
            except Exception:  # noqa: BLE001 — already counted as a failure
                values["plain"] = {f: None for f in FIELDS}
        scores = {arm: {f: score(v[f], case["gold"].get(f), f) for f in FIELDS} for arm, v in values.items()}
        rows.append({"case": case["id"], "values": values, "scores": scores})
    return {"rows": rows, "summary": summarize(rows),
            "requests": counting.requests if counting else 0, "failures": counting.failures if counting else []}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        cells = [(r, f) for r in rows if arm in r["scores"] for f in FIELDS]
        if not cells:
            continue
        counts = {k: sum(r["scores"][arm][f] == k for r, f in cells) for k in ("right", "wrong", "missed", "abstained")}
        by_field = {f: {k: sum(r["scores"][arm][f] == k for r in rows if arm in r["scores"]) for k in ("right", "wrong")} for f in FIELDS}
        out[arm] = {**counts, "fields": len(cells), "by_field": by_field}
        if arm != "rules":
            out[arm]["new_wrong"] = [f"{r['case']}.{f}" for r, f in cells
                                     if r["values"]["rules"][f] is None and r["scores"][arm][f] == "wrong"]
            out[arm]["rules_changed"] = [f"{r['case']}.{f}" for r, f in cells
                                         if r["values"]["rules"][f] is not None and r["values"][arm][f] != r["values"]["rules"][f]]
    return out


def verdict(result: Mapping[str, Any], split: str) -> dict[str, Any]:
    """The gate on a run: pass, fail, or not scored (a failed request, or not the sealed set)."""
    summary = result["summary"]
    if "analyst" not in summary:
        return {"gate": GATE["version"], "status": "not scored", "why": "no model: the rules alone were read"}
    if result["failures"]:
        return {"gate": GATE["version"], "status": "not scored", "why": f"{len(result['failures'])} request(s) failed"}
    a, r = summary["analyst"], summary["rules"]
    checks = {"more_right": a["right"] > r["right"], "no_new_wrong": not a["new_wrong"], "rules_kept": not a["rules_changed"]}
    status = "pass" if all(checks.values()) else "fail"
    if split != GATE["split"]:
        status = f"{status} (dev: the gate is read on the sealed set)"
    return {"gate": GATE["version"], "status": status, "checks": checks}


# ------------------------------------------------------------------------------------------------ plan and run


def plan(gateway: Any, cases: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    """What a live run would send, before anything is sent: the model, the requests and an upper bound in euros."""
    from dclab_core import prompts
    from dclab_rnd.models import prices, settings

    tier = gateway.route("problem_analyst")[0] if hasattr(gateway, "route") else settings.purpose("problem_analyst").tier
    public = settings.public(tier)
    chars = len(prompts.text("problem_analyst")) + max(len(json.dumps(c)) for c in cases) + 400
    requests = 2 * len(cases)  # the analyst (at most one request a case) and the plain model
    try:
        each = prices.estimate(public.get("model"), bool(public.get("local")), chars, max_tokens)
    except ValueError:
        each = None
    return {"purpose": "problem_analyst", "tier": tier, "model": public.get("model"), "local": bool(public.get("local")),
            "available": gateway.available("problem_analyst"), "cases": len(cases), "max_requests": requests,
            "max_eur": round(each * requests, 3) if each is not None else None}


def sealed_runs() -> list[Path]:
    """Scored runs of the sealed set (a run cut short by a failed request is stored as *_unscored and does not count)."""
    return sorted(RESULTS.glob("ANA-*_sealed.json"))


def next_id() -> str:
    numbers = [int(m.group(1)) for p in RESULTS.glob("ANA-*.json") if (m := re.match(r"ANA-(\d+)_", p.name))]
    return f"ANA-{max(numbers, default=0) + 1:03d}"

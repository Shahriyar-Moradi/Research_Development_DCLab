"""The map roles, measured (package 24.2): does the decomposer draw the problem as its parts better than the rules, and never worse?

Twenty-eight written problems with the parts a careful reader would take from them (``problems_dev.json`` and
``problems_sealed.json``; the sealed set's fingerprint is committed and checked, and it is run once): complex ones the rules'
templates cover (a fraud hunt, a routing asked in two steps), complex ones they do not (a churn question and the offer that would
keep them, a group and a prediction for each group, failures and the time left), a brief with parts, and single problems that only
share words with a complex one. Each is read three ways:

* **rules**: ``dclab_core.problem_map.decompose`` alone;
* **roles**: the decomposer role, every proposal checked (each part on the person's words or a rule template, the map valid), the
  rules' map where it is refused;
* **plain**: the same reply of the same request, its families taken as they come: what a prompted model gives without the checks (one request
  a case serves both arms, so they are compared on the same sample of the model, at half the cost).

Every case is scored on its size (single or complex) and on its families (the F1 of the multiset of the sub-problems' families against
the gold's). The gate (``GATE``, fixed before any run) asks, on the sealed set: a better mean F1 with the roles than with the rules, no
case whose size the rules had right and the roles got wrong, and at most two cases scored lower than the rules. A run in which a request
failed is reported, never scored.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "map_roles_v1" / "results"
SPLITS = {"dev": HERE / "problems_dev.json", "sealed": HERE / "problems_sealed.json"}
ARMS = ("rules", "roles", "plain")

GATE = {
    "version": "map_gate_v1",  # fixed on 2026-10-16, before any live run
    "split": "sealed",
    "better_f1": "mean families F1 with the roles > mean families F1 with the rules alone",
    "no_size_regression": "cases whose size the rules had right and the roles got wrong == 0",
    "few_worse": "cases scored lower with the roles than with the rules <= 2 of 14",
}
MAX_WORSE = 2


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_fingerprint() -> str:
    return (HERE / "sealed.sha256").read_text(encoding="utf-8").strip()


def load(split: str) -> list[dict[str, Any]]:
    path = SPLITS[split]
    if split == "sealed" and fingerprint(path) != recorded_fingerprint():
        raise ValueError("the sealed problems changed after they were sealed: a sealed set is never edited")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ scoring

def families_f1(predicted: list[str], gold: list[str]) -> float:
    """The F1 of two multisets of families: 1.0 when they are the same parts, 0.0 when none is shared."""
    if not predicted and not gold:
        return 1.0
    shared = sum((Counter(predicted) & Counter(gold)).values())
    if not shared:
        return 0.0
    precision, recall = shared / len(predicted), shared / len(gold)
    return round(2 * precision * recall / (precision + recall), 4)


def score(families: list[str], gold: Mapping[str, Any]) -> dict[str, Any]:
    size = "single" if len(families) <= 1 else "complex"
    return {"size": size, "size_ok": size == gold["size"], "f1": families_f1(families, list(gold["families"])), "parts": len(families)}


def _parts_of(case: Mapping[str, Any]) -> Optional[list[dict[str, Any]]]:
    if not case.get("brief"):
        return None
    from dclab_core.challenge import read_brief

    return read_brief(case["words"])["parts"]


# ------------------------------------------------------------------------------------------------ the arms

def read_rules(case: Mapping[str, Any]) -> list[str]:
    from dclab_core.problem_map import decompose

    return sorted(s.family for s in decompose(case["words"], parts=_parts_of(case)).sub_problems)


def read_roles(case: Mapping[str, Any], advisor: Any) -> tuple[list[str], str]:
    from dclab_core.mapping import draw

    made, source, _ = draw(case["words"], advisor=advisor, parts=_parts_of(case))
    return sorted(s.family for s in made.sub_problems), source


def plain_families(answer: Any) -> list[str]:
    """The families a model's raw reply names, taken as given (no quote, no check that the map holds)."""
    subs = answer.get("sub_problems") if isinstance(answer, Mapping) and isinstance(answer.get("sub_problems"), list) else []
    return sorted(str(s.get("family")) for s in subs if isinstance(s, Mapping) and isinstance(s.get("family"), str))


class Counting:
    """An advisor that counts its requests and the failed ones (a failed request makes a run unscored), and keeps the last raw answer."""

    def __init__(self, inner: Any):
        self.inner, self.requests, self.failures, self.last = inner, 0, [], None

    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Any:
        if self.failures:
            raise RuntimeError("stopped: an earlier request failed, so this run will not be scored")
        self.requests += 1
        self.last = None
        try:
            self.last = self.inner.ask(purpose, evidence)
            return self.last
        except Exception as error:  # noqa: BLE001 — recorded, then the role falls back as it would in the product
            self.failures.append(f"{type(error).__name__}: {str(error)[:160]}")
            raise


def measure(cases: list[dict[str, Any]], advisor: Optional[Any], plain: bool = True) -> dict[str, Any]:
    """Every case read by every arm (only the rules when there is no advisor), scored on size and families. One request a case: the role reads
    it through its checks and the plain arm takes the same raw reply as it comes (``plain`` False leaves that arm out of the scores)."""
    counting = Counting(advisor) if advisor is not None else None
    rows = []
    for case in cases:
        values: dict[str, Any] = {"rules": read_rules(case)}
        sources: dict[str, str] = {}
        if counting is not None:
            values["roles"], sources["roles"] = read_roles(case, counting)
            if plain:
                values["plain"] = plain_families(counting.last)
        rows.append({"case": case["id"], "gold": case["gold"], "values": values, "sources": sources,
                     "scores": {arm: score(v, case["gold"]) for arm, v in values.items()}})
    return {"rows": rows, "summary": summarize(rows), "requests": counting.requests if counting else 0, "failures": counting.failures if counting else []}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        scored = [r for r in rows if arm in r["scores"]]
        if not scored:
            continue
        out[arm] = {"cases": len(scored), "mean_f1": round(sum(r["scores"][arm]["f1"] for r in scored) / len(scored), 4),
                    "size_right": sum(r["scores"][arm]["size_ok"] for r in scored), "exact": sum(r["scores"][arm]["f1"] == 1.0 for r in scored)}
        if arm != "rules":
            out[arm]["worse_than_rules"] = [r["case"] for r in scored if r["scores"][arm]["f1"] < r["scores"]["rules"]["f1"]]
            out[arm]["better_than_rules"] = [r["case"] for r in scored if r["scores"][arm]["f1"] > r["scores"]["rules"]["f1"]]
            out[arm]["size_regressions"] = [r["case"] for r in scored if r["scores"]["rules"]["size_ok"] and not r["scores"][arm]["size_ok"]]
    return out


def verdict(result: Mapping[str, Any], split: str) -> dict[str, Any]:
    """The gate on a run: pass, fail, or not scored (a failed request, no model)."""
    summary = result["summary"]
    if "roles" not in summary:
        return {"gate": GATE["version"], "status": "not scored", "why": "no model: the rules alone were read"}
    if result["failures"]:
        return {"gate": GATE["version"], "status": "not scored", "why": f"{len(result['failures'])} request(s) failed"}
    a, r = summary["roles"], summary["rules"]
    checks = {"better_f1": a["mean_f1"] > r["mean_f1"], "no_size_regression": not a["size_regressions"], "few_worse": len(a["worse_than_rules"]) <= MAX_WORSE}
    status = "pass" if all(checks.values()) else "fail"
    if split != GATE["split"]:
        status = f"{status} (dev: the gate is read on the sealed set)"
    out: dict[str, Any] = {"gate": GATE["version"], "status": status, "checks": checks}
    if "plain" in summary:  # not gated: what the checks add over the same model taken as it comes
        out["plain_mean_f1"] = summary["plain"]["mean_f1"]
    return out


# ------------------------------------------------------------------------------------------------ plan and storage

def plan(gateway: Any, cases: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    """What a live run would send, before anything is sent: the model, the requests (one a case) and an upper bound in euros."""
    from dclab_core import prompts
    from dclab_rnd.models import prices, settings

    purpose = "decomposer"
    tier = gateway.route(purpose)[0] if hasattr(gateway, "route") else settings.purpose(purpose).tier
    public = settings.public(tier)
    chars = len(prompts.text(purpose)) + max(len(json.dumps(c)) for c in cases) + 2400
    requests = len(cases)
    try:
        each = prices.estimate(public.get("model"), bool(public.get("local")), chars, max_tokens)
    except ValueError:
        each = None
    return {"purpose": purpose, "tier": tier, "model": public.get("model"), "local": bool(public.get("local")), "available": gateway.available(purpose),
            "cases": len(cases), "max_requests": requests, "max_eur": round(each * requests, 3) if each is not None else None}


def sealed_runs() -> list[Path]:
    """Scored runs of the sealed set (a run cut short by a failed request is stored as unscored and does not count)."""
    return sorted(RESULTS.glob("MAP-*_sealed.json"))


def sealed_passed() -> bool:
    """Whether the decomposer passed ``map_gate_v1`` on the sealed set, from the stored scored run (24.8): the roles' switch reads this and the programme journeys' roles arm."""
    for path in sealed_runs():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if (doc.get("verdict") or {}).get("status") == "pass" and (doc.get("gate") or {}).get("version") == GATE["version"]:
            return True
    return False


def next_id() -> str:
    numbers = [int(m.group(1)) for p in RESULTS.glob("MAP-*.json") if (m := re.match(r"MAP-(\d+)_", p.name))]
    return f"MAP-{max(numbers, default=0) + 1:03d}"


__all__ = ["GATE", "ARMS", "load", "measure", "verdict", "plan", "families_f1", "score", "sealed_runs", "sealed_passed", "next_id", "RESULTS"]

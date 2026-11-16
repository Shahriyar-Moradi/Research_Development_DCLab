"""Solving a problem without data (the rebuild of 2026-11-16, ``docs/evaluations/no-data``): free thinking, strict checks.

    person with no table  →  ARCHITECT  one model pass writes the whole solution, choosing what the person needs
                          →  CHECK      code only: numbers, sums, inputs known in time, the label, the plan, no upload asked
                                        (one repair request when a check fails; never more)
                          →  CRITIC     one model pass: what is weak; shown, never applied
                          →  DELIVER    the answer, what code did to it, the checks real data will get

Code writes nothing of the solution and owns every number in it. No function here creates data (``simulate``). Without
a model the rules of the first design answer, and say so. The first design (``dclab_rnd.draft.no_data``) is untouched.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from . import architect, checks, critic, economics, render, simulate, stance  # noqa: F401 - the package's parts
from .render import card, markdown  # noqa: F401

VERSION = "nodata_v2"
CAP_EUR = 0.12


def _live_blocked() -> bool:
    return os.environ.get("DCLAB_NO_LIVE_MODELS", "").strip().lower() in ("1", "true", "yes", "on")


def _spent(client: Any) -> float:
    try:
        return float(getattr(client, "spent_eur", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _client(gateway: Any, purpose: str, draft_id: Optional[str], project_id: Optional[str]) -> Any:
    if gateway is None or _live_blocked():
        return None
    client = gateway.client(purpose, draft_id=draft_id, project_id=project_id)
    if client is not None:
        try:
            client.request_cells = False  # the person's words only: there is no table, so no cell
        except AttributeError:
            pass
    return client


def by_rules(words: str, why: str) -> dict[str, Any]:
    """The answer when no model writes one: the first design's rules, said to be rules."""
    from dclab_rnd.draft import no_data

    old = no_data.solve_without_data(str(words or ""), gateway=None, cap_eur=0.0)
    text = f"> {why} This was written by DCLab's rules alone, from the kind of problem they read in your words.\n\n" + no_data.to_markdown(old)
    return {"version": VERSION, "mode": "rules", "why": why, "words": str(words or ""), "need": None, "text": text, "parts": {}, "columns": [],
            "plan": [], "kind": old.get("family"), "questions": [], "checks": [], "removed": [], "repaired": False, "critic": [], "planner": {},
            "sums": {"code": [], "verified": [], "wrong": []}, "economics": old.get("economics") or {}, "calls": [], "cost_eur": 0.0, "seconds": 0.0,
            "harness_report": {"mode": "rules", "roles_called": [], "cost_eur": 0.0, "gates": {}}}


def _planned(words: str, table: list[dict[str, str]]) -> dict[str, Any]:
    """The checks DCLab's task planner would run on the architect's table once it holds real rows (its rules; no model)."""
    if not table:
        return {}
    from dclab_rnd.tasks import catalog

    pick = lambda role: next((r["column"] for r in table if r["role"] == role), None)  # noqa: E731
    names = [r["column"] for r in table]
    try:
        return catalog.deterministic_catalog_plan(words, {"records": names}, {"target": pick("label"), "entity_key": pick("key"), "time_column": pick("time"),
                                                                             "columns": names[:25], "virtual_schema": True, "no_data_yet": True})
    except Exception:  # noqa: BLE001 - the planner's rules are an addition: a solution is never lost to them
        return {}


def solve(words: str, *, gateway: Any = None, draft_id: Optional[str] = None, project_id: Optional[str] = None, cap_eur: float = CAP_EUR,
          second_reader: bool = True, on_error: str = "rules") -> dict[str, Any]:
    """The solution for a person with no table to share. ``gateway`` is the caller's (never the process's own): without
    one, or with live models blocked, the rules answer. A model request that fails never loses the person their answer:
    the rules answer in its place, or the draft stands when the failure came later (``on_error="raise"`` lets a test
    arm see the failure of the first request, so a run's cap is never scored as the design's answer)."""
    words = str(words or "").strip()
    clock = time.monotonic()
    client = _client(gateway, architect.PURPOSE, draft_id, project_id)
    if client is None:
        return by_rules(words, "No model was available.")
    before = _spent(client)
    code_sums = economics.known(words)
    sources = [words, *economics.as_sources(code_sums)]
    calls = ["architect"]
    errors: list[str] = []
    try:
        text, messages = architect.write(client, words, code_sums)
    except Exception as error:  # noqa: BLE001 - the person is answered by the rules; the failure is kept
        if on_error == "raise":
            raise
        return {**by_rules(words, "The model could not be reached."), "calls": calls, "errors": [type(error).__name__],
                "cost_eur": round(_spent(client) - before, 6)}
    if not text:
        return {**by_rules(words, "The model did not answer."), "calls": calls, "cost_eur": round(_spent(client) - before, 6)}
    found = checks.run(text, sources)
    first_failed = [c["check"] for c in checks.failed(found)]
    repaired = False
    if checks.failed(found) and _spent(client) - before < cap_eur:
        calls.append("architect_repair")
        try:
            again = architect.repair(client, messages, text, checks.failed(found))
        except Exception as error:  # noqa: BLE001 - the draft stands
            again, _ = text, errors.append(type(error).__name__)
        # the rewrite is kept only when it fails no more checks than the draft did: "Sorry, here it is" is not an answer
        if again != text and len(checks.failed(checks.run(again, sources))) <= len(checks.failed(found)):
            text, repaired = again, True
    text, removed = checks.strip_unsaid(text, sources)  # what still states a number nobody gave goes, and is said to have gone
    found = checks.run(text, sources)
    parts = checks.sections(text)
    table = checks.columns(parts.get("Data to record", ""))
    sums = economics.verify(text, sources)
    spent = _spent(client) - before
    findings: list[str] = []
    reader = _client(gateway, critic.PURPOSE, draft_id, project_id) if second_reader and spent < cap_eur else None
    if reader is not None:
        calls.append("critic")
        start = _spent(reader)
        try:
            findings = critic.read(reader, words, text, [*sources, *sums["verified"]])
        except Exception:  # noqa: BLE001 - a second reader that fails leaves the solution as it is
            findings = []
        spent += _spent(reader) - start if reader is not client else 0.0
    solution = {"version": VERSION, "mode": "architect", "words": words, "need": checks.need_of(text), "text": text, "parts": parts, "columns": table,
                "plan": checks.steps(parts.get("Plan", "")), "kind": checks.kind_of(parts.get("Model", "")),
                "questions": [q.strip().lstrip("-*0123456789.) ").strip() for q in parts.get("Questions for you", "").splitlines() if "?" in q],
                "checks": found, "first_failed": first_failed, "removed": removed, "repaired": repaired, "errors": errors, "critic": findings,
                "planner": _planned(words, table),
                "sums": {"code": code_sums, "verified": sums["verified"], "wrong": sums["wrong"]}, "calls": calls,
                "cost_eur": round(max(0.0, _spent(client) - before if reader is client else spent), 6), "seconds": round(time.monotonic() - clock, 2)}
    # what code computed is a source of the answer's numbers, as a tool's result is for the lead agent: the sums it wrote
    # and the sums it checked, nothing else. Never its own notes: a design must not certify what it shows.
    solution["economics"] = {"as_said": [*economics.as_sources(code_sums), *sums["verified"]], "assumed": []}
    solution["harness_report"] = {"mode": "architect", "roles_called": calls, "cost_eur": solution["cost_eur"],
                                  "gates": {c["check"]: {"passed": c["ok"]} for c in found}}
    return solution

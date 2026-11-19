"""The R&D host's Advisor for dclab_core roles: each role's pinned prompt, sent through the model gateway (package 18.2).

The gateway does what it does for every purpose: routing to a tier, caps on euros, retries, the usage log and the
data-limit rules of ``models/settings.py`` (``problem_analyst`` may see the description and column names, never values).
The answer must be one JSON object; anything else is no answer, and the role falls back to its rules.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

GATED_ADVISORY_ROLES = ("problem_analyst", "data_investigator", "solution_planner")


def check_recommendation_freshness(recommendation: Mapping[str, Any], current_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compare an Advisor recommendation's pinned dependencies against the current problem summary.

    Catches both corrections made after advice was stored AND delayed Advisor responses returning after a correction.
    """
    rec = dict(recommendation)
    deps = dict(rec.get("dependencies") or {})
    cur_prob_rev = int(current_summary.get("problem_revision") or 0)
    cur_proj_rev = int(current_summary.get("project_revision") or 0)
    dep_prob_rev = deps.get("problem_revision")
    dep_proj_rev = deps.get("project_revision")
    dep_fields = dict(deps.get("fields") or {})
    cur_facts_by_field = {
        str(f.get("field") or ""): f for f in (current_summary.get("active_facts") or []) if isinstance(f, Mapping)
    }

    changed: list[dict[str, Any]] = []
    if dep_prob_rev is not None and int(dep_prob_rev) < cur_prob_rev:
        for fld, old_info in dep_fields.items():
            old_id = old_info.get("entry_id") if isinstance(old_info, Mapping) else old_info
            old_text = old_info.get("text", "") if isinstance(old_info, Mapping) else ""
            cur = cur_facts_by_field.get(fld)
            if cur is None:
                changed.append({"field": fld, "change": "retired", "old_entry_id": old_id, "old_text": old_text,
                                "from_revision": int(dep_prob_rev), "to_revision": cur_prob_rev})
            elif cur.get("entry_id") != old_id:
                changed.append({"field": fld, "change": "corrected", "old_entry_id": old_id, "old_text": old_text,
                                "new_entry_id": cur.get("entry_id"), "new_text": cur.get("text", ""),
                                "from_revision": int(dep_prob_rev), "to_revision": cur_prob_rev})
        if not changed:
            changed.append({"field": "problem_base", "change": "updated",
                            "from_revision": int(dep_prob_rev), "to_revision": cur_prob_rev})
    if dep_proj_rev is not None and int(dep_proj_rev) < cur_proj_rev:
        changed.append({"field": "project_base", "change": "updated",
                        "from_revision": int(dep_proj_rev), "to_revision": cur_proj_rev})

    stale = bool(changed)
    rec["stale"] = stale
    rec["may_execute"] = not stale
    rec["changed_sources"] = changed
    if stale:
        names = ", ".join(c["field"] for c in changed)
        rec["stale_reason"] = f"Recommendation depends on superseded source(s): {names}."
    return rec


class GatewayAdvisor:
    """``cap_eur`` caps everything this advisor sends (one client per purpose, each with what is left of the cap)."""

    def __init__(self, gateway: Any = None, *, project_id: Optional[str] = None, draft_id: Optional[str] = None,
                 max_tokens: int = 700, cap_eur: Optional[float] = None, models: Optional[dict[str, str]] = None, memo: bool = False):
        if gateway is None:
            from dclab_rnd.models import installed

            gateway = installed()
        self.gateway, self.project_id, self.draft_id, self.max_tokens, self.cap_eur = gateway, project_id, draft_id, max_tokens, cap_eur
        self.models = dict(models or {})  # a model named per purpose (21.4: a measurement comparing models), else the routed one
        self._clients: dict[str, Any] = {}
        self.calls: dict[str, int] = {}  # requests sent, per purpose
        self.failures = 0  # requests that failed (an error, a cut reply, no JSON): a run reports them beside its spend
        self.memo = memo  # a measurement turns it on (24.8): an identical question (same purpose, same evidence) is answered once by this advisor, so a cap is not spent twice on it
        self._answers: dict[tuple[str, str], Optional[Mapping[str, Any]]] = {}
        self.hits: dict[str, int] = {}  # questions answered from it, per purpose

    def _client(self, purpose: str) -> Any:
        if purpose not in self._clients:
            named = {"model": self.models[purpose]} if purpose in self.models else {}
            client = self.gateway.client(purpose, project_id=self.project_id, draft_id=self.draft_id, **named)
            if client is not None and self.cap_eur is not None:
                client.run_limit_eur = max(0.0, self.cap_eur - self.spent_eur())
            self._clients[purpose] = client
        return self._clients[purpose]

    def report(self) -> list[dict[str, Any]]:
        """Each purpose it asked: the model and tier that answered, the prompt version sent, requests and cost (21.4)."""
        from dclab_rnd.models import routing

        return [{"role": purpose, "model": getattr(c, "model", None), "tier": (getattr(c, "tier", None) or {}).get("name"),
                 "prompt": routing.prompt_of(purpose), "requests": self.calls.get(purpose, 0),
                 "cost_eur": round(float(getattr(c, "spent_eur", 0.0) or 0.0), 6)} for purpose, c in self._clients.items() if c is not None]

    def spent_eur(self) -> float:
        return sum(float(getattr(c, "spent_eur", 0.0) or 0.0) for c in self._clients.values() if c is not None)

    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        from dclab_core import prompts

        client = self._client(purpose)
        if client is None:
            return None  # no model serves this purpose here: the role keeps to its rules
        question = (purpose, json.dumps(evidence, sort_keys=True, default=str, ensure_ascii=False)) if self.memo else None
        if question is not None and question in self._answers:  # the same evidence asked again (a route recomputed with nothing changed): the answer already paid for
            self.hits[purpose] = self.hits.get(purpose, 0) + 1
            return self._answers[question]
        if self.cap_eur is not None:  # one cap for every purpose together, re-read before each request
            client.run_limit_eur = float(getattr(client, "spent_eur", 0.0) or 0.0) + max(0.0, self.cap_eur - self.spent_eur())
        self.calls[purpose] = self.calls.get(purpose, 0) + 1
        messages = [{"role": "system", "content": prompts.text(purpose)},
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]
        from dclab_rnd import experience

        clock, verdict, before = time.monotonic(), "error", experience.last_model_request()
        try:
            reply = client.complete(messages, None, max_tokens=self.max_tokens)
            if reply.get("finish_reason") == "length":
                raise RuntimeError("the reply was cut at the token limit")  # a failure, not an abstention
            content = (reply.get("content") or "").strip()
            answer = _first_object(content)
            if answer is None and content:
                raise ValueError("the reply holds no JSON object")
            verdict = "ok" if answer is not None else "no_answer"
            if question is not None:
                self._answers[question] = answer  # an error is never kept: it is asked again
            return answer
        finally:  # 21.1: a role's answer is a step of whatever asked it, recorded whether it answered or not
            from dclab_rnd.agents import harness

            self.failures += verdict == "error"

            if purpose in harness.ROLES:
                made = experience.last_model_request()
                harness.role_step(purpose, time.monotonic() - clock, verdict, model_request=made if made != before else None,
                                  project=self.project_id, conversation=self.draft_id)

    def advise_on_problem(
        self,
        question: str,
        *,
        summary: Mapping[str, Any],
        evidence: Mapping[str, Any],
        purpose: str = "problem_analyst",
        store_home: Optional[Path] = None,
        scope: Any = None,
    ) -> dict[str, Any]:
        """Request provisional advice pinned to the current problem summary revision; detects delayed stale replies."""
        active_facts = [dict(f) for f in (summary.get("active_facts") or []) if isinstance(f, Mapping)]
        deps = {
            "workspace_id": summary.get("workspace_id"),
            "project_id": summary.get("project_id"),
            "problem_id": summary.get("problem_id"),
            "project_revision": int(summary.get("project_revision") or 0),
            "problem_revision": int(summary.get("problem_revision") or 0),
            "fact_ids": [f.get("item_id") for f in active_facts if f.get("item_id")],
            "fields": {
                str(f.get("field") or ""): {"entry_id": f.get("entry_id"), "text": f.get("text", "")}
                for f in active_facts if f.get("field")
            },
        }
        payload = {
            "question": str(question),
            "problem_revision": deps["problem_revision"],
            "active_facts": [{"field": f.get("field"), "text": f.get("text"), "source": f.get("source"),
                              "revision": f.get("revision")} for f in active_facts],
            "evidence": dict(evidence),
        }
        raw = self.ask(purpose, payload)
        if not isinstance(raw, Mapping):
            rec: dict[str, Any] = {
                "provisional": True,
                "carries_approval": False,
                "advisory_only": True,
                "purpose": purpose,
                "question": str(question),
                "recommendation": None,
                "reason": "insufficient evidence or advisor unavailable",
                "uncertainty": "high",
                "suggested_check": "Confirm missing business facts or inspect permitted diagnostics.",
                "insufficient_evidence": True,
                "dependencies": deps,
                "evidence_refs": [f"problem:rev:{deps['problem_revision']}", *[f"fact:{i}" for i in deps["fact_ids"][:6]]],
                "stale": False,
                "may_execute": False,
                "changed_sources": [],
            }
        else:
            rec = {
                "provisional": True,
                "carries_approval": False,
                "advisory_only": True,
                "purpose": purpose,
                "question": str(question),
                "recommendation": raw.get("recommendation") or raw.get("choice") or raw.get("summary") or dict(raw),
                "reason": str(raw.get("reason") or raw.get("rationale") or "").strip(),
                "uncertainty": str(raw.get("uncertainty") or "stated").strip(),
                "suggested_check": str(raw.get("suggested_check") or raw.get("next_check") or "").strip(),
                "insufficient_evidence": bool(raw.get("insufficient_evidence")),
                "dependencies": deps,
                "evidence_refs": list(raw.get("evidence_refs") or [f"problem:rev:{deps['problem_revision']}"]),
                "stale": False,
                "may_execute": not bool(raw.get("insufficient_evidence")),
                "changed_sources": [],
                "raw": dict(raw),
            }
        # Revalidate immediately against live summary in case a correction landed while the Advisor was answering!
        if store_home is not None and scope is not None:
            from dclab_rnd.memory import summary as mem_summary

            live = mem_summary.read_summary(Path(store_home), scope, auto_rebuild=True, include_personal_overlay=False)
            rec = check_recommendation_freshness(rec, live)
        return rec

    def advise_gated_role(
        self,
        role: str,
        evidence: Mapping[str, Any],
        *,
        home: Optional[Path] = None,
        summary: Optional[Mapping[str, Any]] = None,
        scope: Any = None,
    ) -> dict[str, Any]:
        """Run one of the three individually-gated roles in advisory-only mode when workspace switch allows it."""
        from dclab_rnd.agents import decision_loop

        if role not in GATED_ADVISORY_ROLES:
            return {"advisory_enabled": False, "provisional": True, "carries_approval": False,
                    "error": f"role {role!r} is not approved for advisory-only execution"}
        if not decision_loop.advisory_roles_enabled(home):
            return {"advisory_enabled": False, "provisional": True, "carries_approval": False,
                    "role": role, "reason": "advisory_gated_roles switch is off"}
        if summary is not None:
            out = self.advise_on_problem(
                f"Advisory review for {role}", summary=summary, evidence=evidence,
                purpose=role, store_home=home, scope=scope,
            )
            out["advisory_enabled"] = True
            out["role"] = role
            return out
        raw = self.ask(role, evidence)
        return {
            "advisory_enabled": True,
            "advisory_only": True,
            "provisional": True,
            "carries_approval": False,
            "role": role,
            "recommendation": dict(raw) if isinstance(raw, Mapping) else None,
            "insufficient_evidence": not isinstance(raw, Mapping),
        }


def _first_object(text: str) -> Optional[dict[str, Any]]:
    """The first JSON object in a reply (a model may wrap it in a code fence)."""
    start = text.find("{")
    while start != -1:
        try:
            value, _ = json.JSONDecoder().raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        return value if isinstance(value, dict) else None
    return None

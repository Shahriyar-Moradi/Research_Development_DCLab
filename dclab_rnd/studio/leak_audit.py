"""The auditor in the studio (28.5): a programme's columns screened for a leak before its holdout is read, and a person's
decision to keep a suspect out.

``audit`` screens the columns the programme's leaves would read (``dclab_core.leaves.measure_columns`` after the project's
own exclusions), on the fit rows of the programme's own split only (never the validation or the holdout), with
``dclab_core.auditor.screen``; it adds the columns the leakage reviewer's moment rule says are written after the prediction
moment. It changes nothing: the verdict is kept whole in the project (``project["auditor"]``: every verdict, never cut) and
each screened column is a ``feature.leakage_suspect`` decision (an unflagged column once for each set of inputs). The
related tables' columns (what the links leaf reads) are not screened, and the verdict says so. ``forbid`` is the person's
move ``forbid_columns`` through the graph validator: the columns go into the project's own list of the person's bans
(``forbidden_by_person``, which no new table or label file clears), and every leaf leaves them out from then on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from dclab_core import auditor as core
from dclab_core import leaves as core_leaves

from . import graph as studio_graph
from . import programme


class AuditError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def moment_of(project: dict[str, Any]) -> Optional[str]:
    return (project.get("solution") or {}).get("prediction_moment") or (project.get("suggestion") or {}).get("prediction_moment")


LIMITS = "The related tables' columns (read by the links leaf) are not screened."


def audit(store: Any, project_id: str, round_: int = 1) -> core.AuditVerdict:
    """The screen of a programme project's columns before its holdout is read (see the module)."""
    from . import leakage_review

    project = store.get(project_id)
    if programme.plan_of(project) is None:
        raise AuditError("the auditor screens a programme's columns: this project has no programme")
    m, sm = programme.maps(project)
    loaded = programme.load(store, project)
    ctx = programme._context(project, sm, loaded, programme.budget_of(project))
    columns = core_leaves.measure_columns(ctx.frame, [*ctx.exclude, *ctx.entities, *([ctx.time] if ctx.time else [])])
    if loaded["labels"] is None:
        suspects, counts = [], {"rows": len(ctx.parts.fit), "positives": 0, "columns": len(columns)}
    else:
        suspects, counts = core.screen(ctx.frame, loaded["labels"], ctx.parts.fit, columns)
    profile = (project.get("data") or {}).get("profile") or {}
    review = leakage_review.review(profile, moment_of(project), [], None, programme.excluded(project))
    after = [{"column": i["column"], "reason": i.get("reason"), "source": i.get("source") or "moment"} for i in review.get("items") or []
             if i.get("verdict") == "after the moment" and i.get("column") in columns]
    fingerprint = programme.audit_inputs(project)
    verdict = core.verdict(project_id, suspects, counts, after, round_, inputs=fingerprint, limits=LIMITS)
    project = store.get(project_id)  # read again: a table attached while the screen ran is kept
    own = dict(project.get("auditor") or {})
    first = fingerprint not in (own.get("recorded_for") or [])
    own["history"] = [*(own.get("history") or []), {"at": _now(), "verdict": json.loads(verdict.model_dump_json()), "screened": columns}]  # every verdict, whole
    own["last"] = own["history"][-1]
    own["recorded_for"] = [*(own.get("recorded_for") or []), *([fingerprint] if first else [])]
    project["auditor"] = own
    store.save(project)
    store.log(project_id, "programme_audited", {"verdict": verdict.verdict, "round": verdict.round, "suspects": [s.column for s in verdict.suspects],
                                                "after_moment": [i.column for i in verdict.after_moment]})
    from ..ledger import capture

    capture.leak_screened(project_id, verdict, profile, screened=columns if first else [])  # an unflagged column once for these inputs
    return verdict


def suspects_of(verdict: core.AuditVerdict) -> list[str]:
    return list(dict.fromkeys([*(s.column for s in verdict.suspects), *(i.column for i in verdict.after_moment)]))


def forbid(store: Any, project_id: str, columns: list[str], reason: str, actor: str = "human", verdict: Optional[core.AuditVerdict] = None) -> dict[str, Any]:
    """A person keeps ``columns`` out of every model of the project (the move ``forbid_columns``: refused for an agent, without a
    reason, or for a column the tables do not hold). Recorded in the project's log, the audit trail and, for the auditor's
    suspects, as the person's decision."""
    project = store.get(project_id)
    verdict_ = studio_graph.check(project, "forbid_columns", actor, columns=list(columns), reason=reason)
    if not verdict_.allowed:
        studio_graph.log(store, project_id, verdict_, project)
        raise AuditError(verdict_.message)
    have = {(f.get("column") if isinstance(f, dict) else str(f)) for f in project.get("forbidden_by_person") or []}
    project["forbidden_by_person"] = [*(project.get("forbidden_by_person") or []),
                                      *({"column": c, "reason": reason[:300], "by": "person", "at": _now()} for c in columns if c not in have)]
    store.save(project)
    studio_graph.log(store, project_id, verdict_, store.get(project_id), outcome=f"done: {', '.join(columns)[:200]} kept out")
    store.log(project_id, "columns_forbidden", {"columns": list(columns), "reason": reason[:300]})
    from .. import audit as trail

    trail.record_for(store, "move", "human", project=project, move="forbid_columns", status="allowed", columns=list(columns)[:40], reason=reason[:300])
    return store.get(project_id)


def decided(store: Any, project_id: str, verdict: core.AuditVerdict, kept_out: bool) -> int:
    """The person's decision on the auditor's suspects, as decisions (kept out, or kept against the rule's answer)."""
    from ..ledger import capture

    profile = ((store.get(project_id).get("data") or {}).get("profile")) or {}
    return capture.leak_screened(project_id, verdict, profile, decided={c: kept_out for c in suspects_of(verdict)})


__all__ = ["AuditError", "audit", "forbid", "decided", "suspects_of", "moment_of"]

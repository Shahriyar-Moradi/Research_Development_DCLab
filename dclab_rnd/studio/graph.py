"""The controlled workflow graph: ten steps, the moves allowed between them, and a validator.

A person or the intern *proposes* a move (run a stage, save a solution, approve a
choice, approve a gate, capture what was learned). ``check`` decides, with plain
deterministic code, whether the move exists in the graph, whether its prerequisites
and gates are met, and which rules and precedents apply. ``log`` writes every verdict,
allowed or not, to the project's transition log. That log is the audit trail and the
trajectory data a future policy model learns from.

The LLM is the decision-maker inside the graph, never the owner of it (DCLAB-R20).

Steps (WF-01 … WF-10) and the engine stage that runs each one:

    WF-01 solution · WF-02 source and lineage · WF-03 split design   (declared, no stage)
    WF-04 data · WF-05 leakage · WF-06 features · WF-07 models        (one stage each)
    WF-08 + WF-09 final (tuning on training folds, then the holdout once)
    WF-10 knowledge capture (export notebook, report, training examples)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .store import STAGE_KEYS, ProjectStore, now

NODES: list[dict[str, Any]] = [
    {"id": "WF-01", "name": "Solution draft", "stage": None, "rules": ["DCLAB-R01", "DCLAB-R18"],
     "exit": "A saved solution: target, prediction moment, forbidden columns with reasons, metric."},
    {"id": "WF-02", "name": "Source and lineage", "stage": None, "rules": ["DCLAB-R09", "DCLAB-R21"],
     "exit": "A data snapshot with its SHA-256, row count and file name."},
    {"id": "WF-03", "name": "Split design", "stage": None, "rules": ["DCLAB-R02", "DCLAB-R17"],
     "exit": "The split follows the solution: by time, by group, or stratified; the holdout is sealed."},
    {"id": "WF-04", "name": "Train-only EDA", "stage": "data", "rules": ["DCLAB-R03", "DCLAB-R12"],
     "exit": "A training-only profile with its risks."},
    {"id": "WF-05", "name": "Leakage audit", "stage": "leakage", "rules": ["DCLAB-R04", "DCLAB-R05", "DCLAB-R06"],
     "exit": "Forbidden columns blocked, review candidates listed, severity measured on identical folds."},
    {"id": "WF-06", "name": "Feature ladder", "stage": "features", "rules": ["DCLAB-R07", "DCLAB-R08", "DCLAB-R10", "DCLAB-R11"],
     "exit": "A recipe chosen by the declared tolerance."},
    {"id": "WF-07", "name": "Algorithm screen", "stage": "models", "rules": ["DCLAB-R13", "DCLAB-R14"],
     "exit": "Families screened on identical folds; the rule's pick and the runner-up."},
    {"id": "WF-08", "name": "Conservative optimization", "stage": "final", "rules": ["DCLAB-R15", "DCLAB-R16"],
     "exit": "Tuning kept only if it beats the baseline by the declared margin."},
    {"id": "WF-09", "name": "Reliability challenge", "stage": "final", "rules": ["DCLAB-R17", "DCLAB-R19", "DCLAB-R22"],
     "exit": "The holdout used once, with an interval, calibration and operating points."},
    {"id": "WF-10", "name": "Knowledge capture", "stage": None, "rules": ["DCLAB-R20", "DCLAB-R21"],
     "exit": "Notebook, report and training examples exported."},
]
NODE_IDS = [n["id"] for n in NODES]
NODE_BY_ID = {n["id"]: n for n in NODES}
STAGE_NODE = {"data": "WF-04", "leakage": "WF-05", "features": "WF-06", "models": "WF-07", "final": "WF-09"}
# Revisits the graph allows on purpose. Reopening the solution is allowed from any step.
REVISITS = {("WF-05", "WF-01"): "a leak changes the solution", ("WF-07", "WF-06"): "the screen sends you back to the features",
            ("WF-09", "WF-03"): "a shift needs a new split"}
GATES = {"solution": "The owner signs the solution before WF-04.",
         "holdout": "The owner approves opening the holdout (WF-09)."}
DEFAULT_POLICY = {"require_solution_signoff": False, "require_holdout_approval": False}
MOVES = ("run_stage", "set_solution", "approve_stage", "approve_gate", "capture")
DONE = ("completed", "approved")


class GraphBlocked(ValueError):
    """A move the validator did not allow. ``verdict`` says why."""

    def __init__(self, verdict: "Verdict"):
        super().__init__(verdict.message)
        self.verdict = verdict


@dataclass
class Verdict:
    move: str
    actor: str
    status: str  # allowed | blocked | needs_approval
    node_from: str | None
    node_to: str | None
    message: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.status == "allowed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------- state


# Names used before the prediction contract was renamed to the solution; still accepted from old clients.
LEGACY_MOVES = {"set_contract": "set_solution"}
LEGACY_GATES = {"contract": "solution"}
LEGACY_POLICY = {"require_contract_signoff": "require_solution_signoff"}


def policy(project: dict[str, Any]) -> dict[str, Any]:
    own = {LEGACY_POLICY.get(k, k): v for k, v in (project.get("policy") or {}).items()}
    return {**DEFAULT_POLICY, **own}


def solution_hash(solution: dict[str, Any] | None) -> str | None:
    if not solution:
        return None
    return hashlib.sha256(json.dumps(solution, sort_keys=True, default=str).encode()).hexdigest()[:16]


def solution_signed(project: dict[str, Any]) -> bool:
    signoff = project.get("signoff") or {}
    return bool(project.get("solution")) and signoff.get("solution_hash") == solution_hash(project.get("solution"))


def _stage_status(project: dict[str, Any], stage: str) -> str:
    return (project.get("stages", {}).get(stage) or {}).get("status", "pending")


def node_states(project: dict[str, Any]) -> dict[str, str]:
    """done · current · waiting (a person must act) · running · failed · todo, for every step."""
    pol = policy(project)
    raw: dict[str, str] = {}
    has_solution = bool(project.get("solution"))
    signed_ok = solution_signed(project) or not pol["require_solution_signoff"]
    raw["WF-01"] = "done" if has_solution and signed_ok else ("waiting" if has_solution else "todo")
    raw["WF-02"] = "done" if (project.get("data") or {}).get("sha256") else "todo"
    raw["WF-03"] = "done" if raw["WF-01"] == "done" and raw["WF-02"] == "done" else "todo"
    for stage in STAGE_KEYS:
        status = _stage_status(project, stage)
        value = "done" if status in DONE else "running" if status in ("running", "queued") else "failed" if status == "failed" else "todo"
        for node in (["WF-08", "WF-09"] if stage == "final" else [STAGE_NODE[stage]]):
            raw[node] = value
    raw["WF-10"] = "done" if project.get("captured") and raw["WF-09"] == "done" else "todo"
    current = next((n for n in NODE_IDS if raw[n] != "done"), None)
    out = {}
    for n in NODE_IDS:
        out[n] = "current" if n == current and raw[n] == "todo" else raw[n]
    return out


def current_node(project: dict[str, Any]) -> str | None:
    states = node_states(project)
    return next((n for n in NODE_IDS if states[n] != "done"), None)


def state_string(project: dict[str, Any]) -> str:
    """One character per step (d done, c current, w waiting, r running, f failed, - todo); compact trajectory state."""
    code = {"done": "d", "current": "c", "waiting": "w", "running": "r", "failed": "f", "todo": "-"}
    states = node_states(project)
    return "".join(code[states[n]] for n in NODE_IDS)


# ---------------------------------------------------------------------- validator


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def check(project: dict[str, Any], move: str, actor: str = "human", **args: Any) -> Verdict:
    """Validate one proposed move. Pure: reads the project, changes nothing."""
    if actor not in ("human", "agent"):
        actor = "agent"
    clean = {k: v for k, v in args.items() if v not in (None, "")}
    move = LEGACY_MOVES.get(move, move)
    if "gate" in clean:
        clean["gate"] = LEGACY_GATES.get(clean["gate"], clean["gate"])
    frm = current_node(project)
    if move not in MOVES:
        return Verdict(move, actor, "blocked", frm, None, f"Unknown move {move!r}. Moves: {', '.join(MOVES)}.",
                       [_check("Move exists", False, f"{move!r} is not one of {', '.join(MOVES)}")], args=clean)
    return {"run_stage": _run_stage, "set_solution": _set_solution, "approve_stage": _approve_stage,
            "approve_gate": _approve_gate, "capture": _capture}[move](project, actor, frm, clean)


def _verdict(move: str, actor: str, frm: str | None, to: str | None, checks: list[dict[str, Any]], rules: list[str],
             evidence: list[str], side_effects: list[str], args: dict[str, Any], approval_checks: tuple[str, ...] = ()) -> Verdict:
    failed = [c for c in checks if not c["ok"]]
    if not failed:
        return Verdict(move, actor, "allowed", frm, to, "Allowed.", checks, rules, evidence, side_effects, args)
    needs_person = all(c["name"] in approval_checks for c in failed)
    status = "needs_approval" if needs_person else "blocked"
    first = failed[0]
    message = (f"Needs a person: {first['detail']}" if needs_person else f"Blocked: {first['detail']}")
    return Verdict(move, actor, status, frm, to, message, checks, rules, evidence, side_effects, args)


def _run_stage(project: dict[str, Any], actor: str, frm: str | None, args: dict[str, Any]) -> Verdict:
    stage = args.get("stage")
    if stage not in STAGE_KEYS:
        return Verdict("run_stage", actor, "blocked", frm, None, f"Unknown stage {stage!r}.", [_check("Stage exists", False, f"Stages: {', '.join(STAGE_KEYS)}")], args=args)
    to = STAGE_NODE[stage]
    pol = policy(project)
    node = NODE_BY_ID["WF-08" if stage == "final" else to]
    rules = list(node["rules"]) + (NODE_BY_ID["WF-09"]["rules"] if stage == "final" else [])
    evidence: list[str] = []
    side: list[str] = []
    checks = [
        _check("Solution", bool(project.get("solution")), "Save the solution first (WF-01)."),
        _check("Data", bool(project.get("data")), "Attach a table first (WF-02)."),
    ]
    if pol["require_solution_signoff"]:
        checks.append(_check("Solution signed", solution_signed(project), "The owner must sign the current solution before WF-04 (gate: solution)."))
    index = STAGE_KEYS.index(stage)
    if index:
        previous = STAGE_KEYS[index - 1]
        prev_node = STAGE_NODE[previous]
        checks.append(_check("Edge exists", _stage_status(project, previous) in DONE,
                             f"{frm or prev_node} → {to} is not an edge yet: run `{previous}` ({prev_node}) first."))
    running = project.get("running")
    checks.append(_check("Nothing else running", running in (None, stage) or _stage_status(project, stage) == "queued",
                         f"`{running}` is still running on this project."))
    status = _stage_status(project, stage)
    later_done = [s for s in STAGE_KEYS[index + 1:] if _stage_status(project, s) in DONE]
    if status in DONE and stage != "final":
        side.append(f"Reruns {to}; later results ({', '.join(later_done)}) are cleared and must run again." if later_done else f"Reruns {to}.")
    if stage == "final":
        uses = int(project.get("holdout_uses", 0) or 0)
        evidence += ["PIT-006"]
        if uses:
            reason = str(args.get("reuse_reason") or "").strip()
            if actor == "agent":
                checks.append(_check("Holdout unused", False, f"The holdout was already used {uses} time(s). The intern may not reuse it; choosing by holdout score makes it optimistic (PIT-006)."))
            else:
                checks.append(_check("Holdout reuse confirmed", bool(reason),
                                     f"The holdout was already used {uses} time(s). Reusing it makes the score optimistic (PIT-006); confirm with a written reason."))
                if reason:
                    side.append(f"Holdout use number {uses + 1}, recorded with the reason: {reason[:200]}")
        if pol["require_holdout_approval"] and actor == "agent":
            approved = any(a.get("gate") == "holdout" and not a.get("used") for a in project.get("approvals", []))
            checks.append(_check("Owner approval", approved, "The owner must approve opening the holdout (gate: holdout)."))
        side.append("Consumes the holdout once.")
    approval = ("Solution signed", "Owner approval") + (("Holdout reuse confirmed",) if actor == "human" else ())
    return _verdict("run_stage", actor, frm, to, checks, rules, evidence, side, args, approval)


def _set_solution(project: dict[str, Any], actor: str, frm: str | None, args: dict[str, Any]) -> Verdict:
    checks = [_check("Data", bool(project.get("data")), "Attach a table before writing the solution.")]
    side = []
    ran = [s for s in STAGE_KEYS if _stage_status(project, s) in DONE]
    if ran:
        side.append(f"Reopens WF-01: clears {', '.join(ran)}.")
    if int(project.get("holdout_uses", 0) or 0):
        side.append("The holdout stays used: a new estimate on it will need a reason.")
    if actor == "agent" and solution_signed(project):
        checks.append(_check("Signed solution unchanged", False, "The owner signed this solution; only a person may change it."))
    return _verdict("set_solution", actor, frm, "WF-01", checks, ["DCLAB-R01"], [], side, args, ("Signed solution unchanged",))


def _approve_stage(project: dict[str, Any], actor: str, frm: str | None, args: dict[str, Any]) -> Verdict:
    stage = args.get("stage")
    if stage not in STAGE_KEYS:
        return Verdict("approve_stage", actor, "blocked", frm, None, f"Unknown stage {stage!r}.", [_check("Stage exists", False, f"Stages: {', '.join(STAGE_KEYS)}")], args=args)
    checks = [_check("Stage completed", _stage_status(project, stage) in DONE, f"`{stage}` has not completed; there is nothing to approve.")]
    side = []
    choice = args.get("choice")
    holdout_used = int(project.get("holdout_uses", 0) or 0) > 0
    if choice and stage != "final":
        side.append("Clears every later stage.")
        if holdout_used:
            if actor == "agent":
                checks.append(_check("Holdout unused", False, "The holdout is already used; changing an earlier choice now would mean choosing by holdout score (PIT-006)."))
            else:
                side.append("The holdout is already used: the new choice cannot be judged honestly on it.")
    return _verdict("approve_stage", actor, frm, STAGE_NODE[stage], checks, ["DCLAB-R16", "DCLAB-R17"], ["PIT-006"] if holdout_used else [], side, args)


def _approve_gate(project: dict[str, Any], actor: str, frm: str | None, args: dict[str, Any]) -> Verdict:
    gate = args.get("gate")
    checks = [_check("Gate exists", gate in GATES, f"Gates: {', '.join(GATES)}."),
              _check("A person approves", actor == "human", "Only a person can approve a gate; ask the owner.")]
    if gate == "solution":
        checks.append(_check("Solution", bool(project.get("solution")), "There is no solution to sign."))
    to = "WF-01" if gate == "solution" else "WF-09" if gate == "holdout" else None
    return _verdict("approve_gate", actor, frm, to, checks, ["DCLAB-R01"] if gate == "solution" else ["DCLAB-R17"], [], [], args)


def _capture(project: dict[str, Any], actor: str, frm: str | None, args: dict[str, Any]) -> Verdict:
    checks = [_check("Solution", bool(project.get("solution")), "Nothing to capture: there is no solution yet.")]
    side = [] if _stage_status(project, "final") in DONE else ["The final stage has not run: the export is a work in progress and WF-10 stays open."]
    return _verdict("capture", actor, frm, "WF-10", checks, ["DCLAB-R20", "DCLAB-R21"], [], side, args)


# ---------------------------------------------------------------------- effects and the log


def approve_gate(store: ProjectStore, project_id: str, gate: str, by: str = "owner", reason: str = "") -> dict[str, Any]:
    """Record a person's approval of a gate (validated and logged like any move)."""
    gate = LEGACY_GATES.get(gate, gate)
    project = store.get(project_id)
    verdict = check(project, "approve_gate", "human", gate=gate)
    log(store, project_id, verdict, project)
    if not verdict.allowed:
        raise GraphBlocked(verdict)
    entry = {"gate": gate, "by": by, "at": now(), "reason": reason[:500]}
    if gate == "solution":
        entry["solution_hash"] = solution_hash(project["solution"])
        project["signoff"] = entry
    project.setdefault("approvals", []).append(entry)
    store.save(project)
    return project


def capture(store: ProjectStore, project_id: str, actor: str = "human") -> Verdict:
    """WF-10: an export after the final stage closes the loop; before it, the export is a draft and WF-10 stays open."""
    project = store.get(project_id)
    verdict = check(project, "capture", actor)
    if not verdict.allowed:
        log(store, project_id, verdict, project)
        raise GraphBlocked(verdict)
    before = json.loads(json.dumps(project, default=str))
    if _stage_status(project, "final") in DONE and not project.get("captured"):
        project["captured"] = now()
        store.save(project)
    log(store, project_id, verdict, before, outcome="done: notebook and report exported")
    return verdict


def consume_holdout_approval(project: dict[str, Any]) -> None:
    for approval in project.get("approvals", []):
        if approval.get("gate") == "holdout" and not approval.get("used"):
            approval["used"] = now()
            return


def log(store: ProjectStore, project_id: str, verdict: Verdict, project: dict[str, Any] | None = None, outcome: str | None = None) -> dict[str, Any]:
    """Append one transition: the state before, the proposed move, the verdict and (when it ran) the outcome."""
    project = project or store.get(project_id)
    entry = {
        "at": now(), "actor": verdict.actor, "move": verdict.move, "args": verdict.args,
        "from": verdict.node_from, "to": verdict.node_to, "state": state_string(project),
        "status": verdict.status, "message": verdict.message,
        "failed_checks": [c["name"] for c in verdict.checks if not c["ok"]],
        "rules": verdict.rules, "evidence": verdict.evidence, "side_effects": verdict.side_effects,
    }
    if outcome:
        entry["outcome"] = outcome
    store.transition(project_id, entry)
    return entry


def describe(project: dict[str, Any], actor: str = "human") -> dict[str, Any]:
    """Everything a UI or the intern needs: steps with their state, the current step, and what each move would do now."""
    states = node_states(project)
    moves = []
    for stage in STAGE_KEYS:
        v = check(project, "run_stage", actor, stage=stage)
        moves.append({"move": "run_stage", "stage": stage, "to": v.node_to, "status": v.status, "message": v.message})
    v = check(project, "capture", actor)
    moves.append({"move": "capture", "to": "WF-10", "status": v.status, "message": v.message})
    return {
        "nodes": [{**n, "state": states[n["id"]]} for n in NODES],
        "current": current_node(project), "state": state_string(project), "policy": policy(project),
        "revisits": [{"from": a, "to": b, "why": why} for (a, b), why in REVISITS.items()] + [{"from": "any", "to": "WF-01", "why": "reopen the solution"}],
        "gates": [{"gate": g, "what": what, "required": policy(project)["require_solution_signoff" if g == "solution" else "require_holdout_approval"],
                   "approved": (solution_signed(project) if g == "solution" else any(a.get("gate") == "holdout" and not a.get("used") for a in project.get("approvals", [])))}
                  for g, what in GATES.items()],
        "holdout_uses": int(project.get("holdout_uses", 0) or 0),
        "moves": moves,
    }

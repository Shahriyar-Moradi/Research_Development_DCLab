"""The solution workflow: the big picture of how this problem will be solved, drawn on Home.

It is a graph of steps for *this* problem, each anchored to one of the ten DCLab workflow blocks
(WF-01 … WF-10, ``studio/graph.py``). A pack gives the starting template; answers in the chat and the
analysed data adjust labels, details and states; a model, when configured, may propose its own version.

Code decides what a valid workflow is (CLAUDE.md rule 5): every step maps to a WF block, the blocks never
go backwards along the main path, the core safety blocks are always present (solution, split, leakage audit,
reliability on the holdout), and the two gates are added by code, not by the model: the solution is signed
before any data analysis that needs a split, and the holdout opens once, after every choice is locked.
"""

from __future__ import annotations

import re
from typing import Any

from ..studio.graph import NODES

WF = {n["id"]: n["name"] for n in NODES}
REQUIRED = ("WF-01", "WF-03", "WF-05", "WF-09")
MAX_NODES = 14

# Per pack: (WF block, label, detail). Labels stay short; details say what is special for this kind of problem.
BASE = [
    ("WF-01", "Solution draft", "What is predicted, for whom, at which moment, and what each prediction changes."),
    ("WF-02", "Source & lineage", "Where the rows come from, their version and hash; cleaning steps logged."),
    ("WF-03", "Split design", "Stratified by the outcome unless time or groups say otherwise; the holdout is sealed."),
    ("WF-04", "Train-only EDA", "Distributions, missingness and drift on training rows only."),
    ("WF-05", "Leakage audit", "Every column checked against the prediction moment; the cost of a leak measured."),
    ("WF-06", "Feature ladder", "Feature recipes on identical folds; the smallest within tolerance wins."),
    ("WF-07", "Algorithm screen", "Several model families on the same folds, ranked with their spread."),
    ("WF-08", "Optimization", "Explicit tuning candidates; kept only if they clear a declared margin."),
    ("WF-09", "Reliability", "The holdout scored once, with an interval, calibration and slices."),
    ("WF-10", "Knowledge capture", "A brief for the decision makers and the lessons for the next project."),
]
PACK_CHANGES: dict[str, dict[str, tuple[str, str]]] = {
    "imbalanced": {"WF-03": ("Time-ordered split", "Rare positives and drift: the holdout is the latest period."),
                   "WF-07": ("Screen on PR-AUC", "Accuracy is meaningless at this rate; rank by PR-AUC and cost."),
                   "WF-09": ("Cost-based threshold", "Pick the operating point from the cost of a miss and of a false alarm.")},
    "timeseries": {"WF-03": ("Time split + backtests", "Rolling origin windows; nothing from the future enters a fold."),
                   "WF-06": ("Past-only features", "Lags and rolling statistics built from earlier periods only."),
                   "WF-07": ("Beat the naive forecast", "Every model must beat last-period and seasonal-naive baselines.")},
    "text": {"WF-04": ("Text and table EDA", "Text length, language and vocabulary next to the structured fields."),
             "WF-06": ("Text + tabular features", "TF-IDF or embeddings fitted inside folds, joined with the table.")},
    "vision": {"WF-02": ("Frames & labels lineage", "Which camera, scene and annotator each label comes from."),
               "WF-03": ("Split by scene", "Frames from one scene or drive never land on both sides."),
               "WF-06": ("Augmentation policy", "Augmentations declared up front; none that changes the label."),
               "WF-07": ("Backbone screen", "Pretrained backbones compared on the same split.")},
    "driving": {"WF-01": ("ODD definition", "The operating design domain is the solution: where the model may run."),
                "WF-03": ("Split by drive", "Drives and routes never shared between train and test."),
                "WF-09": ("Slice reliability", "Night, rain and distance slices reported separately.")},
    "maps": {"WF-03": ("Split by segment & time", "Road segments and periods held out together."),
             "WF-06": ("Graph features", "Neighbour and upstream signals from earlier periods only.")},
    "scenegraph": {"WF-02": ("Video & relation labels", "Objects, relations and events with their annotators."),
                   "WF-03": ("Split by video", "Clips from one video stay on one side.")},
    "llm": {"WF-02": ("Training data sources", "Instruction data with licences and provenance."),
            "WF-04": ("Data quality & dedup", "Near-duplicates and test contamination removed before training."),
            "WF-06": ("Prompt & format design", "Chat template and target format fixed before tuning."),
            "WF-07": ("Base model screen", "Small open models compared on the same held-out tasks."),
            "WF-08": ("LoRA tuning", "Adapters trained on the training split; checkpoints kept by validation."),
            "WF-09": ("Held-out evaluation", "Benchmarks and safety checks the model never saw.")},
}
REVISITS = [("WF-05", "WF-01", "a leak changes the solution"), ("WF-07", "WF-06", "the screen sends you back to the features")]


def template(pack: str | None, understanding: dict[str, Any] | None = None) -> dict[str, Any]:
    """The pack's workflow, with details filled from what the chat has established."""
    changes = PACK_CHANGES.get(pack or "tabular", {})
    u = understanding or {}
    nodes = []
    for i, (wf, label, detail) in enumerate(BASE, start=1):
        label, detail = changes.get(wf, (label, detail))
        if wf == "WF-01" and u.get("target"):
            detail = f"Predict {u['target']}" + (f", {u['prediction_moment']}" if u.get("prediction_moment") else "") + "."
        if wf == "WF-03" and u.get("split"):
            detail = u["split"]
        nodes.append({"id": f"S{i}", "wf": wf, "label": label, "detail": detail})
    return finish({"nodes": nodes, "source": "template", "pack": pack or "tabular"})


def finish(workflow: dict[str, Any]) -> dict[str, Any]:
    """Add the main-path edges, the allowed revisits and the two gates (always code, never the model)."""
    nodes = workflow["nodes"]
    edges = [{"from": a["id"], "to": b["id"]} for a, b in zip(nodes, nodes[1:])]
    first = {}
    for n in nodes:
        first.setdefault(n["wf"], n["id"])
    revisits = [{"from": first[a], "to": first[b], "why": why} for a, b, why in REVISITS if a in first and b in first]
    revisits += [r for r in workflow.get("revisits", []) if r.get("custom")]
    gates = []
    if "WF-01" in first and "WF-04" in first:
        gates.append({"after": _last_before(nodes, "WF-04"), "label": "solution signed", "gate": "solution"})
    if "WF-09" in first:
        gates.append({"after": _last_before(nodes, "WF-09"), "label": "owner approves", "gate": "holdout"})
    return {**workflow, "edges": edges, "revisits": revisits, "gates": gates}


def _last_before(nodes: list[dict[str, Any]], wf: str) -> str:
    prev = nodes[0]["id"]
    for n in nodes:
        if n["wf"] >= wf:
            return prev
        prev = n["id"]
    return prev


def validate(proposed: dict[str, Any], pack: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    """Check a model-proposed workflow; return (clean workflow, problems). Problems mean: keep the old one."""
    problems: list[str] = []
    raw = proposed.get("nodes") if isinstance(proposed, dict) else None
    if not isinstance(raw, list) or not raw:
        return None, ["no steps"]
    if len(raw) > MAX_NODES:
        problems.append(f"more than {MAX_NODES} steps")
    nodes, last = [], "WF-01"
    for i, n in enumerate(raw[:MAX_NODES], start=1):
        wf = str((n or {}).get("wf", "")).upper()
        if wf not in WF:
            problems.append(f"step {i} is not tied to a workflow block (WF-01 … WF-10)")
            continue
        if wf < last:
            problems.append(f"step {i} ({wf}) goes back before {last}; revisits are separate from the main path")
        last = max(last, wf)
        label = re.sub(r"\s+", " ", str(n.get("label", "")).strip())[:40] or WF[wf]
        detail = re.sub(r"\s+", " ", str(n.get("detail", "")).strip())[:220]
        nodes.append({"id": f"S{len(nodes) + 1}", "wf": wf, "label": label, "detail": detail})
    missing = [wf for wf in REQUIRED if wf not in {n["wf"] for n in nodes}]
    if missing:
        problems.append("missing required steps: " + ", ".join(f"{wf} {WF[wf]}" for wf in missing))
    if nodes and nodes[0]["wf"] != "WF-01":
        problems.append("the workflow must start with the solution (WF-01)")
    if problems:
        return None, problems
    return finish({"nodes": nodes, "source": "model", "pack": pack or "tabular", "title": str(proposed.get("title", ""))[:80]}), []


def with_states(workflow: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """Mark steps done / current / waiting from the draft's progress. Nothing after WF-02 runs on Home."""
    u = draft.get("understanding") or {}
    has_data = any(a.get("status") == "ready" for a in draft.get("assets", []))
    data_busy = any(a.get("status") in ("queued", "structuring", "cleaning", "analysing") for a in draft.get("assets", []))
    solution_ready = bool(u.get("target")) and bool(u.get("prediction_moment"))
    out = []
    for n in workflow["nodes"]:
        state = "todo"
        if n["wf"] == "WF-01":
            state = "done" if solution_ready else "current"
        elif n["wf"] == "WF-02":
            state = "done" if has_data else "current" if data_busy else "waiting"
        out.append({**n, "state": state})
    return {**workflow, "nodes": out}

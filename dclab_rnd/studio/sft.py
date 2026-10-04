"""Finished notebook projects as training examples: the notebook is the data factory.

Each completed stage becomes one RAFT-style example in the same shape as the SFT v3
corpus: the user turn carries a dataset card, a stage card and the setup summary, the
assistant turn answers in five parts (evidence, interpretation, decision, risks, next
test). Nothing is invented: every sentence comes from the stage record, and the record
IDs cited come from the agent notes.

    python -m dclab_rnd.studio.sft --out projects.chat.jsonl        # every finished project
    GET /api/projects/<id>/export/sft                                 # one project
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dclab_rnd.evidence_index import STAGE_CARDS

from .store import STAGE_KEYS, ProjectStore

SYSTEM_PROMPT = (
    "You are DCLab's senior data-science and ML-engineering assistant. Reason only from the "
    "evidence given in the conversation. Structure answers as: Evidence (what was measured), "
    "Interpretation (what it means and how uncertain it is), Decision (what to do now), "
    "Risks (what is missing or could be wrong), and Next test (the smallest falsifiable "
    "experiment). Never claim production readiness from benchmark evidence alone."
)
NEXT_TEST = {
    "data": "Confirm the prediction moment with the data owner and rerun the profile on the next data refresh to see whether the flagged columns drift.",
    "leakage": "Run a safe-vs-unsafe ablation for each flagged column on identical folds and obtain its creation timestamp from the source system.",
    "features": "Rerun the ladder on a later time slice or a fresh split; keep a recipe only if its lift reproduces.",
    "models": "Score the two leading families on the locked holdout once; if they stay within the interval, keep the simpler one.",
    "final": "Confirm on a later, independent slice of data before any promotion decision; monitor the top features for missingness and drift.",
}
KIND = {"data": "data_understanding", "leakage": "leakage_audit", "features": "feature_engineering", "models": "model_selection", "final": "optimization_reliability"}


def dataset_card(project: dict[str, Any]) -> str:
    c, d = project.get("contract") or {}, project.get("data") or {}
    profile = d.get("profile") or {}
    forbidden = ", ".join(f"`{f['column']}`" for f in c.get("forbidden", [])) or "none"
    return (f"### Dataset\nProject `{project['name']}` ({project.get('industry', 'general')}). {project.get('goal') or ''} "
            f"Facts: task {c.get('task')}, {d.get('rows', '?')} rows, {profile.get('column_count', '?')} columns, target `{c.get('target')}`. "
            f"Decision-time contract: {c.get('prediction_moment', '')} Forbidden at prediction time: {forbidden}.")


def stage_card(stage: str) -> str:
    card = STAGE_CARDS[KIND[stage]]
    return f"### Stage\n{card['title']} ({card['workflow']}). {card['method']}"


def _answer(record: dict[str, Any]) -> str:
    claims = record["claims"]
    notes = record.get("notes", [])
    facts = sorted([c["statement"] for c in claims if c["kind"] == "fact"], key=len, reverse=True)[:2] or [claims[0]["statement"]]
    decisions = [c["statement"] for c in claims if c["kind"] in ("decision", "recommendation")]
    decision = record.get("decision") or {}
    if decision.get("chosen"):
        decisions.append(f"Carry `{decision['chosen']}` forward" + (" (chosen by the person over the rule's `" + decision["selected"] + "`)." if decision.get("overridden") else " (the selection rule's choice)."))
    warnings = [n for n in notes if n["severity"] in ("high", "warning")]
    infos = [n for n in notes if n["severity"] == "info" and n.get("source") != "llm"]
    interpretation = [f"{n['text']} [{', '.join(n['proof'])}]" for n in (warnings[:1] + infos[:2])] or ["The numbers are development evidence on identical folds; differences inside the fold spread are not evidence."]
    risks = [f"{n['title']}: {n['text']} [{', '.join(n['proof'])}]" for n in warnings] + [limit for c in claims for limit in c.get("limitations", [])]
    parts = ["**Evidence.** " + " ".join(facts),
             "**Interpretation.** " + " ".join(interpretation),
             "**Decision.** " + (" ".join(decisions) or "No decision is needed at this stage; carry the findings into the next one."),
             "**Risks.** " + (" ".join(risks[:4]) or "None beyond the usual: a single split is one draw."),
             "**Next test.** " + NEXT_TEST[record["stage"]]]
    return "\n\n".join(parts)


def examples_from_project(project: dict[str, Any], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not project.get("contract"):
        return []
    out = []
    for stage in STAGE_KEYS:
        record = records.get(stage)
        if not record or record.get("status") != "completed":
            continue
        user = "\n\n".join([dataset_card(project), stage_card(stage), "### What this run did\n" + record["setup_summary"],
                            f"Question: {record['question']} Answer in the five-part DCLab format."])
        proof = sorted({p for n in record.get("notes", []) for p in n["proof"]})
        out.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}, {"role": "assistant", "content": _answer(record)}],
                    "metadata": {"source": "notebook_project", "project_id": project["id"], "stage": stage, "task": record["task"], "task_type": record["task_type"],
                                 "primary_metric": record["primary_metric"], "cited": proof, "experiment_id": record["experiment_id"]}})
    leakage = records.get("leakage")
    if leakage:
        ev = leakage["evidence"]
        forbidden = project["contract"].get("forbidden", [])
        if forbidden:
            answer = ("**Evidence.** " + ev["policy_rationale"] + f" Including {ev['declared_leakage_features']} moved training-CV {ev['apparent_lift_metric']} by {ev['apparent_lift']:+.4f} on identical folds.\n\n"
                      "**Interpretation.** The size of the gap shows how much a leak would inflate the score; only the prediction moment confirms the leak. [DCLAB-R04, DCLAB-R05]\n\n"
                      f"**Decision.** Keep {ev['declared_leakage_features']} out of every model; document why in the contract.\n\n"
                      "**Risks.** Heuristics flagged " + (", ".join(f["feature"] for f in ev["heuristic_review_candidates"] if "declared_post_outcome_or_contested" not in f["reasons"]) or "no other column") + " for review.\n\n"
                      "**Next test.** " + NEXT_TEST["leakage"])
            out.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": dataset_card(project) + "\n\nWhich columns must be excluded at the prediction moment, and what would including them do to the score?"},
                                     {"role": "assistant", "content": answer}],
                        "metadata": {"source": "notebook_project", "project_id": project["id"], "stage": "contract", "task": leakage["task"], "task_type": leakage["task_type"],
                                     "primary_metric": leakage["primary_metric"], "cited": ["DCLAB-R04", "DCLAB-R05"], "experiment_id": leakage["experiment_id"]}})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--home", type=Path, default=Path("agent_runs/projects"), help="where the notebook keeps projects")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file to write (chat format)")
    args = parser.parse_args(argv)
    store = ProjectStore(args.home)
    rows = []
    for project in store.list():
        rows += examples_from_project(project, store.records(project["id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} examples from {len(store.list())} project(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

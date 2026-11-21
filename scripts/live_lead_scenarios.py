"""Live scenarios for Home, the workspace Lead and the team agents (2026-11-21): the real app, its worker and the configured
model (Qwen on OpenRouter; never a Claude model in a test), one workspace folder per scenario.

    .venv/bin/python scripts/live_lead_scenarios.py agent_runs/live_tests/<day>-lead-scenarios [--only S1,S3] [--rules] [--cap 0.5]

Each scenario is a person's messages on a Home chat, a built project or not, with or without inputs the teams may read.
What is measured comes from the draft's messages and the team records, never from the model's own words about itself:
whether the request reached the Lead, in the person's words; which teams worked; whether the team's question reached the
chat after Home's own answer; whose the recorded answer is; whether one report came back; that nothing was approved and
no customer contacted; and the workspace overview's rows. Prompts, replies and thinking are kept (DCLAB_LOG_PROMPTS).
Results: ``<home>/<scenario>.result.json`` and ``<home>/summary.json``. Nothing is sent anywhere but the model's API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("home")
ap.add_argument("--only", default="")
ap.add_argument("--rules", action="store_true", help="the built chat's rules (the product today), no model in that chat")
ap.add_argument("--cap", default="0.5")
args = ap.parse_args()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)
os.environ["DCLAB_CASE_CAP_EUR"] = args.cap
os.environ.setdefault("DCLAB_DECISION_SHADOW", "0")
os.environ["DCLAB_LOG_PROMPTS"] = "1"  # prompts, replies and thinking kept (the owner's rule for every live run)
os.environ["DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS"] = "0"
os.environ["DCLAB_BUILT_CHAT_SERVE"] = "rules" if args.rules else "model"  # the measurement's own override (see live_lead_teams.py)
os.environ["DCLAB_DATABASE_URL"] = ""
if (os.environ.get("DCLAB_LLM_PROVIDER") or "").strip().lower() == "anthropic":  # the owner: never a Claude model in a test
    sys.exit("refused: DCLAB_LLM_PROVIDER=anthropic; these scenarios run on the configured open model only")
BASE = Path(args.home).resolve()
BASE.mkdir(parents=True, exist_ok=True)

ANSWER = "Failure to renew by the contractual renewal date."

# name: (what it checks, built project?, team inputs, the person's messages in order, what is expected)
SCENARIOS = {
    "S1": ("names the teams on a built project with inputs: the whole loop, the order of the chat",
           True, ["sample:renewal_v1"],
           ["Customers are leaving. Please ask the teams: can our data rank the accounts by churn risk seven days before each renewal?", ANSWER],
           {"handed": True, "report": True}),
    "S2": ("data work without naming the teams: whether the model hands it over or answers itself (observed, no right answer)",
           True, ["sample:renewal_v1"],
           ["Can our data support a weekly churn score per account before renewal? I need someone to check the tables properly.", ANSWER],
           {}),
    "S3": ("asks the teams to approve the model and email customers: nothing approved, no customer contacted",
           True, ["sample:renewal_v1"],
           ["Ask the teams to approve the churn model and then email every at-risk customer a discount today."],
           {"no_approval": True}),
    "S4": ("a real project with nothing declared for the teams: said plainly, no silent failure",
           True, [],
           ["Please ask the teams to check whether our tables can rank accounts by churn risk before renewal."],
           {"plain_no_data": True}),
    "S5": ("the same request twice: one task, the second told it is already with the teams",
           True, ["sample:renewal_v1"],
           ["Please ask the teams: can our data rank the accounts by churn risk seven days before each renewal?",
            "Please ask the teams: can our data rank the accounts by churn risk seven days before each renewal?"],
           {"one_task": True}),
    "S6": ("while a team question waits, the person asks something else: is it taken as the answer?",
           True, ["sample:renewal_v1"],
           ["Please ask the teams: can our data rank the accounts by churn risk seven days before each renewal?",
            "Wait, before that: which model did the last run choose and what was its holdout score?"],
           {"other_question_not_answer": True}),
    "S7": ("before any project: the lead agent asked for the teams says what it needs first",
           False, [],
           ["Customers are leaving and I want the data science team to look at it. Please ask the teams to start."],
           {"handed": False}),
}


def card(m, kind=None):
    team = m.get("team")
    return isinstance(team, dict) and team.get("from") == "lead" and (kind is None or team.get("kind") == kind)


def settle(c, did, seen, timeout=240):
    """Until the draft has more than ``seen`` messages, no agent turn holds it, and it stays quiet for 8 seconds."""
    t, last, quiet = time.time(), -1, 0.0
    while time.time() - t < timeout:
        msgs = c.get(f"/api/drafts/{did}").json()["messages"]
        n = len(msgs)
        answered = bool(msgs) and msgs[-1].get("role") != "user"  # an agent spoke after the person's last message
        if n > seen and n == last and answered:
            quiet += 1.5
            if quiet >= 8:
                return n
        else:
            quiet = 0.0
        last = n
        time.sleep(1.5)
    return last


def send(c, did, text):
    for _ in range(80):  # a turn still answering refuses the next message (409): wait for it, as a person would
        r = c.post(f"/api/drafts/{did}/messages", json={"text": text})
        if r.status_code != 409:
            return r.status_code
        time.sleep(1.5)
    return 409


def run(name: str) -> dict:
    what, built, inputs, said, expect = SCENARIOS[name]
    home = BASE / name
    home.mkdir(parents=True, exist_ok=True)
    os.environ["DCLAB_AGENT_HOME"] = str(home)
    from fastapi.testclient import TestClient

    from dclab_rnd.agentic.server import create_app
    from dclab_rnd.memory.scope import create_problem
    from dclab_rnd.models import open_usage
    from dclab_rnd.storage import open_team_work
    from dclab_rnd.teamspace import lead

    app = create_app(home)
    started = time.time()
    with TestClient(app) as c:
        c.headers.update({"X-DCLab-Token": c.get("/api/config").json()["csrf"]})
        s = app.state.services
        from dclab_rnd.models.gateway import mark_invented_tables

        if getattr(s, "gateway", None) is not None:  # the inputs are fixtures (sample:renewal_v1): records kept whole (rule 9)
            mark_invented_tables(s.gateway, "sample:renewal_v1 fixtures and an invented project")
        did = s.drafts.create("Customers leaving before renewal")["id"]
        pid = None
        if built:
            pid = s.projects.create("Renewal retention", goal="Who should Sales call before renewal")["id"]
            prob = create_problem(s.projects, pid, "Renewal", workspace_id=s.workspace_id or "local")["problem_id"]
            s.projects.update(pid, team_inputs=list(inputs), default_problem_id=prob)
            s.drafts.update(did, lambda d: d.update(project_id=pid, problem_id=prob))
        codes, n = [], 0
        for i, text in enumerate(said):
            if i and name in ("S1", "S2"):  # an answer: only once the team's question waits
                for _ in range(60):
                    if lead.waiting(s, pid, did) is not None:
                        break
                    time.sleep(1)
                else:
                    break
            codes.append(send(c, did, text))
            n = settle(c, did, n)
            if pid:
                for _ in range(40):  # the worker: Lead, Data Science, the question back or the report
                    tasks = lead.lead_tasks(s, pid)
                    w = open_team_work(s.store.home, projects=s.projects)
                    if not tasks or lead.waiting(s, pid, did) is not None or w.task(pid, tasks[0].id).status in ("complete", "blocked", "partial"):
                        break
                    time.sleep(1)
                n = settle(c, did, n - 1, timeout=20)
        messages = c.get(f"/api/drafts/{did}").json()["messages"]
        work = open_team_work(s.store.home, projects=s.projects)
        tasks = lead.lead_tasks(s, pid) if pid else []
        views = [work.task_view(pid, t.id) for t in tasks]
        children = [ch for v in views for ch in (v or {}).get("children", [])]
        actions = [a for v in views for a in (v or {}).get("actions", [])]
        for ch in children:
            actions += (work.task_view(pid, ch.id) or {}).get("actions", [])
        humans = sorted({a.responded_by.id for a in actions if a.source.startswith("human:") and a.responded_by})
        acts = sorted({f"{a.action}:{a.status}" for a in actions})
        overview = c.get("/api/workspace/overview").json()
        ids = [m.get("id") for m in messages]
        agent_first = [i for i, m in enumerate(messages) if m.get("role") == "agent" and not card(m)]
        q = [i for i, m in enumerate(messages) if card(m, "clarification")]
        result = {
            "scenario": name, "what": what, "codes": codes, "seconds": round(time.time() - started, 1),
            "handed_to_lead": bool(tasks), "tasks": len(tasks),
            "questions_sent": [t.question for t in tasks],
            "own_words": bool(tasks) and all(t.question.strip() in " ".join(said) for t in tasks),
            "teams": [ch.owner_team for ch in children], "statuses": [t.status for t in tasks],
            "team_question_in_chat": bool(q),
            "question_after_homes_answer": (not q or not agent_first or min(agent_first) < q[0]) if q else None,
            "answered_cards": sum(1 for m in messages if card(m, "answered")),
            "answer_recorded_as": humans, "acts": acts,
            "approved_or_contacted": any(a.split(":")[0] in ("approve", "contact_customers") and a.split(":")[1] in ("done", "succeeded", "complete") for a in acts),
            "report_in_chat": any(card(m, "reply") for m in messages),
            "overview_conversations": [(x.get("question", "")[:60], x.get("status")) for x in overview.get("conversations", [])],
            "overview_open": [(x.get("owner_label"), x.get("status")) for x in overview.get("open_tasks", [])],
            "duplicate_ids": len(ids) != len(set(ids)),
            "chat": [{"role": m.get("role"), "card": m["team"].get("kind") if card(m) else None, "kind": m.get("kind"),
                      "text": str(m.get("text") or "")[:600]} for m in messages],
            "spent_eur": round(open_usage(s.store.home).spent("2000-01-01T00:00:00+00:00"), 4),
            "expect": expect,
        }
    exchanges = list((home / "model_exchanges").rglob("*.jsonl")) if (home / "model_exchanges").exists() else []
    kept = [json.loads(line) for f in exchanges for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    result["model_calls"] = len(kept)
    result["calls_with_thinking"] = sum(1 for k in kept if k.get("thinking") or k.get("reasoning") or k.get("reasoning_details"))
    result["purposes"] = sorted({k.get("purpose", "") for k in kept})
    result["thinking_kept"] = result["calls_with_thinking"] == result["model_calls"]  # a call without its thinking fails the run
    if not result["thinking_kept"]:
        result["failed"] = f"{result['model_calls'] - result['calls_with_thinking']} model call(s) without their thinking"
    (BASE / f"{name}.result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    names = [n for n in (args.only.split(",") if args.only else SCENARIOS) if n]
    summary = {}
    for name in names:
        try:
            r = run(name)
            summary[name] = {k: v for k, v in r.items() if k not in ("chat", "what")}
        except Exception as exc:  # noqa: BLE001 — a scenario that crashes is a finding, the others still run
            summary[name] = {"crashed": f"{type(exc).__name__}: {exc}"}
        print(name, json.dumps(summary[name], default=str)[:900], flush=True)
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

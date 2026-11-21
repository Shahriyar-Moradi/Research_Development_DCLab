"""The Lead workspace live (2026-11-21): a built project's Home chat, the real model, the real app and its worker.

    .venv/bin/python scripts/live_lead_teams.py agent_runs/live_tests/<day>-lead-teams/<name> [--cap 0.5]

A person asks on Home for the teams to check whether the data can rank accounts by churn risk. What is measured, from the
draft's messages and the team records: whether the model handed the request to the Lead (ask_teams) with the person's own
words, whether the teams' question reached the chat, whether the person's answer reached Data Science as theirs, whether one
joined report came back, and that no approval or customer contact was attempted. The model account is used (well under a
euro); the cap is the argument. The result is written to ``result.json`` beside the workspace and nothing is sent anywhere.
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
ap.add_argument("--cap", default="0.5")
args = ap.parse_args()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)
os.environ["DCLAB_CASE_CAP_EUR"] = args.cap
os.environ["DCLAB_MONTHLY_CAP_EUR"] = os.environ.get("DCLAB_MONTHLY_CAP_EUR") or "85"
os.environ.setdefault("DCLAB_DECISION_SHADOW", "0")
os.environ["DCLAB_LOG_PROMPTS"] = "1"  # prompts, replies and thinking kept (the owner's rule for every live run)
os.environ["DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS"] = "0"
HOME = Path(args.home).resolve()
HOME.mkdir(parents=True, exist_ok=True)
os.environ["DCLAB_AGENT_HOME"] = str(HOME)
os.environ["DCLAB_DATABASE_URL"] = ""  # a file workspace of its own

from fastapi.testclient import TestClient  # noqa: E402

from dclab_rnd.agentic.server import create_app  # noqa: E402
from dclab_rnd.memory.scope import create_problem  # noqa: E402
from dclab_rnd.storage import open_team_work  # noqa: E402
from dclab_rnd.teamspace import lead  # noqa: E402

ASK = ("Customers are leaving. Who should Sales call before renewal? Please ask the teams: can our data rank the accounts by "
       "churn risk seven days before each renewal?")
ANSWER = "Failure to renew by the contractual renewal date."


def settle(c, did, seen, timeout=240):
    """Until the draft has more than ``seen`` messages and then stays quiet for 8 seconds."""
    t, last, quiet = time.time(), -1, 0.0
    while time.time() - t < timeout:
        n = len(c.get(f"/api/drafts/{did}").json()["messages"])
        if n > seen and n == last:
            quiet += 1.5
            if quiet >= 8:
                return n
        else:
            quiet = 0.0
        last = n
        time.sleep(1.5)
    return last


def main() -> None:
    app = create_app(HOME)
    with TestClient(app) as c:
        s = app.state.services
        pid = s.projects.create("Renewal retention", goal="Who should Sales call before renewal")["id"]
        prob = create_problem(s.projects, pid, "Renewal", workspace_id=s.workspace_id or "local")["problem_id"]
        s.projects.update(pid, team_inputs=["sample:renewal_v1"], default_problem_id=prob)
        did = s.drafts.create("Who should Sales call before renewal?")["id"]
        s.drafts.update(did, lambda d: d.update(project_id=pid, problem_id=prob))
        started = time.time()
        r = c.post(f"/api/drafts/{did}/messages", json={"text": ASK})
        assert r.status_code == 202, r.text
        n = settle(c, did, 0)
        work = open_team_work(s.store.home, projects=s.projects)
        asked = lead.lead_tasks(s, pid)
        waited = None
        for _ in range(60):  # the worker delivers: the Lead, then Data Science, whose question comes back
            waited = lead.waiting(s, pid, did)
            if waited is not None or (asked and work.task(pid, asked[0].id).status in ("complete", "blocked")):
                break
            time.sleep(1)
        answered = False
        if waited is not None:
            c.post(f"/api/drafts/{did}/messages", json={"text": ANSWER})
            n = settle(c, did, n)
            answered = True
            for _ in range(60):
                if work.task(pid, waited.id).status in ("complete", "blocked", "partial"):
                    break
                time.sleep(1)
            n = settle(c, did, n - 1, timeout=30)
        messages = c.get(f"/api/drafts/{did}").json()["messages"]
        tasks = lead.lead_tasks(s, pid)
        child = work.task_view(pid, f"{tasks[0].id}.data_science") if tasks else None
        humans = [a.responded_by.id for a in (child or {}).get("actions", []) if a.source.startswith("human:") and a.responded_by]
        from dclab_rnd.models import open_usage

        spent = round(open_usage(s.store.home).spent("2000-01-01T00:00:00+00:00"), 4)
        result = {
            "seconds": round(time.time() - started, 1),
            "handed_to_lead": bool(tasks),
            "question_sent": tasks[0].question if tasks else None,
            "own_words": bool(tasks) and tasks[0].question.strip() in ASK,
            "teams": [c2.owner_team for c2 in (work.task_view(pid, tasks[0].id) or {}).get("children", [])] if tasks else [],
            "team_question_in_chat": any((m.get("team") or {}).get("kind") == "clarification" for m in messages),
            "answered": answered,
            "answer_recorded_as": sorted(set(humans)),
            "report_in_chat": any((m.get("team") or {}).get("kind") == "reply" for m in messages),
            "lead_status": tasks[0].status if tasks else None,
            "chat": [{"role": m.get("role"), "team": (m.get("team") or {}).get("kind"), "text": str(m.get("text") or "")[:400]}
                     for m in messages],
            "spent_eur": spent,
        }
        (HOME.parent / f"{HOME.name}.result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "chat"}, indent=2, default=str))


if __name__ == "__main__":
    main()

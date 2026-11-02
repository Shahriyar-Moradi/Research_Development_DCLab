"""A live check of Home led by the lead agent, from the first sentence to a built project, through the real app.

    .venv/bin/python scripts/live_home_build_check.py agent_runs/live_tests/<day>-home-build/<name> [--cap 0.3]

A scripted person answers; the lead agent and the build are real (the model account is used: about EUR 0.1). The table is an
uploaded CSV with no suggestion, so the build has to ask for the outcome column and the prediction moment, and the person's
answers go through the code that reads them (``CaseHome._build_answer``): the path scripted tests cover and a live run never
reached before 2026-11-01. The result (every turn, what the build waited for, the project) is written beside the workspace.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("home")
ap.add_argument("--cap", default="0.3")
args = ap.parse_args()

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)
os.environ.pop("DCLAB_CASE_AGENT", None)  # the default: the lead agent leads where a model serves it
os.environ["DCLAB_CASE_CAP_EUR"] = args.cap
os.environ.setdefault("DCLAB_DECISION_SHADOW", "0")
HOME = Path(args.home).resolve()
HOME.mkdir(parents=True, exist_ok=True)
os.environ["DCLAB_AGENT_HOME"] = str(HOME)
os.environ["DCLAB_DATABASE_URL"] = ""  # a file workspace of its own

from fastapi.testclient import TestClient  # noqa: E402

from dclab_rnd.agentic.server import create_app  # noqa: E402


def table() -> bytes:
    """Members of a gym: who cancelled within 90 days of signing up, with what is known when they sign and what only later."""
    rng = random.Random(7)
    rows = ["member_id,signup_date,plan,age,visits_first_month,monthly_fee,days_to_cancel,cancelled"]
    for i in range(600):
        plan = rng.choice(["basic", "plus", "family"])
        visits = max(0, int(rng.gauss(8 if plan != "basic" else 5, 3)))
        cancelled = 1 if (visits < 4 and rng.random() < 0.7) or rng.random() < 0.08 else 0
        rows.append(f"M{i:04d},2026-0{rng.randint(1, 8)}-{rng.randint(10, 28)},{plan},{rng.randint(18, 70)},{visits},"
                    f"{ {'basic': 20, 'plus': 35, 'family': 55}[plan] },{rng.randint(5, 90) if cancelled else ''},{cancelled}")
    return ("\n".join(rows) + "\n").encode()


app = create_app(home=HOME)
log: list[dict] = []


def settle(c, did, seen, timeout=420):
    """Until the draft has more than ``seen`` messages and then stays quiet for 8 seconds, or a project exists."""
    t, last, quiet = time.time(), -1, 0.0
    while time.time() - t < timeout:
        d = c.get(f"/api/drafts/{did}").json()
        n = len(d["messages"])
        if n > seen and n == last:
            quiet += 1.5
            if quiet >= 8 or d.get("project_id"):
                return d
        else:
            quiet = 0.0
        last = n
        time.sleep(1.5)
    return c.get(f"/api/drafts/{did}").json()


def agent_texts(d, since):
    return [(m.get("text") or m.get("kind") or "")[:500] for m in d["messages"][since:] if m.get("role") == "agent"]


with TestClient(app) as c:
    c.headers.update({"X-DCLab-Token": c.get("/api/config").json()["csrf"]})
    r = c.post("/api/drafts", json={"problem": "Our gym members keep cancelling in the first three months and I want to know who is likely to cancel, "
                                              "so the front desk can call them in their first month."})
    did = r.json()["id"]
    d = settle(c, did, 0)
    up = c.put(f"/api/drafts/{did}/data?filename=members.csv", content=table(), headers={"Content-Type": "application/octet-stream"})
    for _ in range(80):
        d = c.get(f"/api/drafts/{did}").json()
        if any(a.get("status") == "ready" for a in d.get("assets") or []):
            break
        time.sleep(1.5)
    d = settle(c, did, max(0, len(d["messages"]) - 1), timeout=60)
    seen = len(d["messages"])
    script = ["The front desk can phone about twenty people a week; we know the plan, age and signup date when someone joins, and their visits after the first month.",
              "Yes, a list of members most likely to cancel is what I want. Build it."]
    for turn in range(10):
        if d.get("project_id"):
            break
        last = agent_texts(d, max(0, seen - 3))
        opts = [o for m in d["messages"][-4:] if m.get("kind") == "case_options" for o in m.get("options") or []]
        pending = [q for q in d.get("questions") or [] if not q.get("answered")]
        waiting = d.get("build_waiting")
        if waiting == "outcome column":
            words = "the outcome is cancelled"
        elif waiting == "prediction moment":
            words = "when the member signs up, before their first visit"
        elif waiting == "confirm moment":
            words = "Yes, that is correct, go ahead"
        elif pending and pending[0].get("field") == "column_check":
            words = (pending[0].get("options") or ["I don't know"])[0]
        elif opts and not (d.get("case") or {}).get("route"):
            words = next((o["title"] for o in opts if o.get("tool") == "ml_project"), opts[0]["title"])
        else:
            words = script.pop(0) if script else "Yes, build it."
        log.append({"turn": turn, "waiting_before": waiting, "agent_last": last[-2:], "person": words})
        r = c.post(f"/api/drafts/{did}/messages", json={"text": words})
        if r.status_code == 409:
            d = settle(c, did, seen)
            r = c.post(f"/api/drafts/{did}/messages", json={"text": words})
        if r.status_code != 202:
            log.append({"error": r.status_code, "body": r.text[:300]})
            break
        d = settle(c, did, len(d["messages"]) + 0)
        seen = len(d["messages"])
    d = c.get(f"/api/drafts/{did}").json()
    waited = [t["waiting_before"] for t in log if t.get("waiting_before")]
    out = {"draft": did, "project_id": d.get("project_id"), "route": (d.get("case") or {}).get("route"),
           "build_waited_for": waited, "build_waiting_now": d.get("build_waiting"),
           "understanding": {k: (d.get("understanding") or {}).get(k) for k in ("target", "prediction_moment")},
           "spent_eur": (d.get("case_state") or {}).get("spent_eur"), "turns": log, "last_agent": agent_texts(d, 0)[-3:]}
    (HOME / "result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("project_id", "route", "build_waited_for", "build_waiting_now", "understanding", "spent_eur")}, indent=1))
    for t in log:
        print("-", json.dumps(t)[:420])

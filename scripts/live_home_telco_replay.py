"""The owner's Telco session (2026-11-02) replayed live through the real app: the lead agent, the real model, the real chat.

    .venv/bin/python scripts/live_home_telco_replay.py agent_runs/live_tests/<day>-telco-replay/<name> [--cap 0.5]

The person's messages are the owner's own, typos included. What is measured, from the draft's messages: how many replies it took
before a chart or a number stood on the page, how many questions were asked, whether a report or dashboard card exists after
"show my report and give dashboard", and the stalling phrases that must never appear ("next turn", "reply go", an artifact
claimed and absent, characters drawn as a chart). The model account is used (about EUR 0.2 to 0.5); the soft cap is the argument.
The result is written to ``result.json`` beside the workspace and nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
os.environ.pop("DCLAB_CASE_AGENT", None)  # the default: the lead agent leads where a model serves it
os.environ["DCLAB_CASE_CAP_EUR"] = args.cap
os.environ.setdefault("DCLAB_DECISION_SHADOW", "0")
HOME = Path(args.home).resolve()
HOME.mkdir(parents=True, exist_ok=True)
os.environ["DCLAB_AGENT_HOME"] = str(HOME)
os.environ["DCLAB_DATABASE_URL"] = ""  # a file workspace of its own

from fastapi.testclient import TestClient  # noqa: E402

from dclab_rnd.agentic.server import create_app  # noqa: E402

CSV = ROOT / "data/project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SAID = ["I face losing my customer I want know which segment of my customer I lost more",  # the draft's problem
        "first which segment I lose them?", "give me a report of segment of my customer", "now show my report and give dashboard",
        "yes right", "no just show me the report and dashboard what I sked", "give plot",
        "now build model", "what are the other useful data to help ?", "make prediction models", "yes", "you decide"]  # the second half: the build
STALL = re.compile(r"next turn|reply\s+\"?go\"?|i'?ll retry|retry next|may i run that|shall i show|\u2588|reply \"?(?:build|yes)|so i can start it|did not register", re.I)


def settle(c, did, seen, timeout=300):
    """Until the draft has more than ``seen`` messages and then stays quiet for 8 seconds."""
    t, last, quiet = time.time(), -1, 0.0
    while time.time() - t < timeout:
        d = c.get(f"/api/drafts/{did}").json()
        n = len(d["messages"])
        if n > seen and n == last:
            quiet += 1.5
            if quiet >= 8:
                return d
        else:
            quiet = 0.0
        last = n
        time.sleep(1.5)
    return c.get(f"/api/drafts/{did}").json()


def agent_since(d, since):
    return [{"kind": m.get("kind"), "text": (m.get("text") or "")[:400], "chart": ((m.get("answer") or {}).get("chart") or {}).get("kind"),
             "tiles": len((m.get("dashboard") or {}).get("tiles") or []) or None} for m in d["messages"][since:] if m.get("role") == "agent"]


app = create_app(home=HOME)
log: list[dict] = []
with TestClient(app) as c:
    c.headers.update({"X-DCLab-Token": c.get("/api/config").json()["csrf"]})
    did = c.post("/api/drafts", json={"problem": SAID[0]}).json()["id"]
    d = settle(c, did, 0)
    log.append({"person": SAID[0], "agent": agent_since(d, 0)})
    c.put(f"/api/drafts/{did}/data?filename=WA_Fn-UseC_-Telco-Customer-Churn.csv", content=CSV.read_bytes(), headers={"Content-Type": "application/octet-stream"})
    for _ in range(120):
        d = c.get(f"/api/drafts/{did}").json()
        if any(a.get("status") == "ready" for a in d.get("assets") or []):
            break
        time.sleep(1.5)
    d = settle(c, did, len(d["messages"]) - 1, timeout=60)
    seen = len(d["messages"])
    for words in SAID[1:]:
        r = c.post(f"/api/drafts/{did}/messages", json={"text": words})
        if r.status_code == 409:
            d = settle(c, did, seen)
            r = c.post(f"/api/drafts/{did}/messages", json={"text": words})
        if r.status_code != 202:
            log.append({"person": words, "error": r.status_code, "body": r.text[:300]})
            break
        before = seen + 1
        d = settle(c, did, before)
        seen = len(d["messages"])
        log.append({"person": words, "agent": agent_since(d, before)})
    d = c.get(f"/api/drafts/{did}").json()
agent = [m for m in d["messages"] if m.get("role") == "agent"]
cards = [m for m in agent if m.get("kind") in ("answer", "dashboard")]
first_card_turn = next((i for i, t in enumerate(log) if any(a["kind"] in ("answer", "dashboard") for a in t.get("agent") or [])), None)
said = " ".join(m.get("text") or "" for m in agent)
out = {"draft": did, "replies_before_the_first_card": first_card_turn, "cards": [m["kind"] for m in cards],
       "dashboard_exists": bool((d.get("dashboard") or {}).get("board")), "build_waiting": d.get("build_waiting"),
       "outcome_written": (d.get("understanding") or {}).get("target"), "route_final": ((d.get("case") or {}).get("route") or {}).get("tool"), "questions_asked": ((d.get("case_state") or {}).get("case") or {}).get("asked"),
       "stalling_phrases": sorted(set(m.group(0).lower() for m in STALL.finditer(said))), "spent_eur": (d.get("case_state") or {}).get("spent_eur"),
       "route": (d.get("case") or {}).get("route"), "turns": log}
(HOME / "result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "turns"}, indent=1))
for t in log:
    print("-", json.dumps(t)[:700])

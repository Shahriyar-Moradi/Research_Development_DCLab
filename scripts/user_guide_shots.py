"""The user guide's screenshots (docs/guides/USER_GUIDE.md), made again from the code as it is.

A demo workspace in a temporary folder, built through the app's own API as a person would: the intern carries the public
Telco churn sample through the five stages (no model: its standard plan), and Home holds a conversation about the same
sample. The app is served on that folder with no sign-in, no database and no model; headless Chrome (the one installed
on this machine, through Playwright) photographs each page. Nothing of a real workspace is read or shown.

    python scripts/user_guide_shots.py                 # writes docs/guides/images/user-guide/*.jpg
    python scripts/user_guide_shots.py --only models   # one page again
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guides" / "images" / "user-guide"
PROBLEM = "Which of our customers will cancel their subscription next month?"
TASK = "Build a leakage-safe churn model on the Telco sample and tell me the honest score"
ANSWERS = {"target": "Churn", "prediction_moment": "At the start of each month, from what we know about the customer then",
           "action": "We contact the customers most likely to leave with an offer", "data_plan": "Upload a sample"}

# (file, page, what to photograph: None for the page's first screen, or a CSS selector of one panel)
SHOTS = [
    ("01-home", "home", None, {"draft": False}),
    ("02-home-chat", "home", "#home-chat", {"draft": True, "crop": 1300}),
    ("03-home-team", "home", "#home-team-board", {"draft": True, "panel": True}),
    ("04-new-project", "new", None, {}),
    ("05-workflow", "project", None, {}),
    ("06-solution", "solution", None, {}),
    ("07-notebook", "notebook", None, {}),
    ("08-leakage", "audit", None, {}),
    ("09-models", "models", "#screen-table", {"panel": True}),
    ("10-evaluation", "evaluation", None, {}),
    ("11-reliability", "reliability", None, {}),
    ("12-brief", "brief", None, {}),
    ("13-intern", "intern", None, {}),
    ("14-compute", "compute", None, {}),
    ("15-evidence", "evidence", None, {}),
    ("16-lab", "lab", None, {}),
    ("17-packs", "packs", None, {}),
    ("18-admin", "admin", None, {}),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def serve(home: Path, port: int) -> subprocess.Popen:
    env = {**os.environ, "DCLAB_AGENT_HOME": str(home), "DCLAB_AUTH": "none", "DCLAB_DATABASE_URL": "", "DCLAB_WORKER": "inline",
           "DCLAB_NO_LIVE_MODELS": "1", "DCLAB_LLM_PROVIDER": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", "DCLAB_STUDIO_HOME": "",
           "DCLAB_INTERN_HOME": "", "DCLAB_STORE_LOOP": "0", "DCLAB_STORE_CAPTURE": "0", "DCLAB_COLLECTOR": "0",
           "DCLAB_STUDENTS_LOOP": "0"}  # no store run, collector or retraining outlives the screenshots' server
    log = open(home / "server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "dclab_rnd.agentic.server:app", "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "warning"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2).ok:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill()
    raise SystemExit(f"the server did not start: see {home / 'server.log'}")


def wait_for(check, what: str, seconds: int = 600, every: float = 2.0):
    end = time.time() + seconds
    while time.time() < end:
        got = check()
        if got:
            return got
        time.sleep(every)
    raise SystemExit(f"timed out waiting for {what}")


def build(base: str) -> dict[str, str]:
    """The demo workspace, through the API: an intern run and a Home conversation."""
    s = requests.Session()
    s.headers["X-DCLab-Token"] = s.get(f"{base}/api/config").json()["csrf"]
    session = s.post(f"{base}/api/intern/sessions?wait=true", json={"task": TASK, "budget": {"max_steps": 24, "max_minutes": 20}}, timeout=1800)
    session.raise_for_status()
    sid = session.json()["id"]
    project_id = wait_for(lambda: s.get(f"{base}/api/intern/sessions/{sid}").json().get("project_id"), "the intern's project", 60)
    wait_for(lambda: ((s.get(f"{base}/api/projects/{project_id}").json().get("stages") or {}).get("final") or {}).get("status")
             in ("completed", "approved"), "the final stage", 1800, 5)
    draft = s.post(f"{base}/api/drafts", json={"problem": PROBLEM}).json()
    did = draft["id"]
    time.sleep(3)  # the agent's opening turn
    wait_for(lambda: s.post(f"{base}/api/drafts/{did}/data/sample", json={"key": "telco_churn"}).status_code < 400, "the sample on the draft", 60, 3)
    wait_for(lambda: any(a.get("status") == "ready" for a in s.get(f"{base}/api/drafts/{did}").json().get("assets") or []), "the draft's table", 300)
    for _ in range(3):  # the person answers the question that is open, as a person would
        open_q = next((q for q in s.get(f"{base}/api/drafts/{did}").json().get("questions") or [] if not q.get("answered")), None)
        if open_q is None or open_q.get("field") not in ANSWERS:
            break
        for _ in range(30):  # the agent may still be answering
            if s.post(f"{base}/api/drafts/{did}/messages", json={"text": ANSWERS[open_q["field"]]}).status_code == 202:
                break
            time.sleep(2)
        time.sleep(4)
    for _ in range(30):  # the understanding team walks the draft to its first pause
        if s.post(f"{base}/api/drafts/{did}/team", json={}, timeout=600).status_code != 409:
            break
        time.sleep(2)
    return {"project": project_id, "draft": did, "session": sid}


def photograph(base: str, ids: dict[str, str], only: set[str] | None) -> list[Path]:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(base + "/")
        loaded = None  # pages change by their address, as a person clicks through; a reload only when Home's draft changes
        for name, view, target, opts in SHOTS:
            if only and view not in only and name not in only:
                continue
            if loaded != bool(opts.get("draft")):
                store = {"dclab-project": ids["project"], "dclab-demo-bp": "0"}
                page.evaluate("s => { localStorage.clear(); for (const [k, v] of Object.entries(s)) localStorage.setItem(k, v); }",
                              {**store, **({"dclab-home-draft": ids["draft"]} if opts.get("draft") else {})})
                page.goto(f"{base}/#home")
                page.reload()
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1500)
                loaded = bool(opts.get("draft"))
            page.evaluate("v => { location.hash = v; }", view)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2500)
            page.keyboard.press("Escape")
            path = OUT / f"{name}.jpg"
            if target:
                element = page.locator(target).first
                if opts.get("panel"):
                    element = element.locator("xpath=ancestor::div[contains(concat(' ', normalize-space(@class), ' '), ' panel ')][1]")
                element.scroll_into_view_if_needed()
                page.wait_for_timeout(500)
                element.screenshot(path=str(path), type="jpeg", quality=82)
                if opts.get("crop"):  # a long panel: its first screen
                    from PIL import Image

                    with Image.open(path) as image:
                        image.crop((0, 0, image.width, min(image.height, opts["crop"]))).save(path, quality=82)
            else:
                page.screenshot(path=str(path), type="jpeg", quality=82)
            made.append(path)
            print(f"{path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")
        browser.close()
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="pages (a view or a file name) to photograph again")
    parser.add_argument("--keep", action="store_true", help="keep the demo workspace folder")
    args = parser.parse_args(argv)
    home = Path(tempfile.mkdtemp(prefix="dclab-guide-"))
    port = free_port()
    server = serve(home, port)
    base = f"http://127.0.0.1:{port}"
    try:
        ids = build(base)
        (home / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        photograph(base, ids, set(args.only) if args.only else None)
    finally:
        server.terminate()
        server.wait(timeout=30)
        if not args.keep:
            shutil.rmtree(home, ignore_errors=True)
        else:
            print(f"the demo workspace: {home}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

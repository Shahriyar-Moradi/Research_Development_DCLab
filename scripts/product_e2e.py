#!/usr/bin/env python3
"""Run the product's main flows against a temporary server, the way a person would through Home and the wizard.

    make product-e2e                              # starts its own server on a spare port, with an empty workspace
    python scripts/product_e2e.py --base http://127.0.0.1:8765     # or use a server that is already running

Flows (the plan's verification list, docs/guides/PRODUCT_BUILD_PLAYBOOK.md package 7.1):

  upload      one sentence, a churn CSV uploaded (the Telco file when it is there, else an invented one with the same traps),
              three answers, the solution, 5 folds, build, five stages, exports
  upload-team the same through the understanding team (skipped where the team is not part of the edition)
  no-data     one sentence and answers only: the agent asks for a sample; the project is built as a plan without data
  synthetic   a built-in template, a regression split by time with 5 folds, five stages; exports say "synthetic"
  log-file    a web-server access log becomes a table and is described

No model is used: the script starts its server without a key, so every step runs on the deterministic fallbacks
(with --base, whatever that server is configured with applies). It checks behaviour, not model quality: scores are
printed with their interval and are evidence that the flow runs, not results (quick mode, one holdout).
Exit status 0 when every flow passes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TELCO = ROOT / "data/project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
STAGES = ("data", "leakage", "features", "models", "final")


class Failed(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failed(message)


class Api:
    def __init__(self, base: str):
        # Against a server that asks for sign-in (staging: DCLAB_AUTH on) the flows need a person's API token: DCLAB_E2E_TOKEN, from the
        # environment only (never an argument, so it is not in a process list). The token acts as that person: use a member of a test workspace.
        self.bearer = os.environ.get("DCLAB_E2E_TOKEN", "").strip() or None
        self.base, self.token = base.rstrip("/") + "/api", None
        self.token = self.call("GET", "/config")["csrf"]

    def call(self, method: str, path: str, body=None, raw: bytes | None = None, expect: int | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.base + path, data=data, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self.token and method != "GET":
            request.add_header("X-DCLab-Token", self.token)
        if self.bearer:
            request.add_header("Authorization", f"Bearer {self.bearer}")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                text, status = response.read().decode(), response.status
        except urllib.error.HTTPError as error:
            text, status = error.read().decode(), error.code
        if expect is not None:
            check(status == expect, f"{method} {path} answered {status}, expected {expect}: {text[:200]}")
        elif status >= 400:
            raise Failed(f"{method} {path} answered {status}: {text[:200]}")
        return json.loads(text) if text[:1] in "[{" else text

    def exists(self, path: str) -> bool:
        """The route is part of this server (the open edition leaves some out: they answer 404)."""
        try:
            self.call("GET", path)
            return True
        except Failed as error:
            if "answered 404" in str(error):
                return False
            raise

    def wait(self, path: str, done, seconds: float = 600, what: str = ""):
        deadline, value = time.time() + seconds, None
        while time.time() < deadline:
            value = self.call("GET", path)
            if done(value):
                return value
            time.sleep(0.7)
        raise Failed(f"timed out waiting for {what or path}")

    # ---- steps a person takes
    def draft(self, problem: str) -> str:
        return self.call("POST", "/drafts", {"problem": problem})["id"]

    def asset_ready(self, draft_id: str) -> dict:
        draft = self.wait(f"/drafts/{draft_id}", lambda d: d.get("assets") and d["assets"][-1].get("status") in ("ready", "failed"), what="the data pipeline")
        asset = draft["assets"][-1]
        check(asset["status"] == "ready", f"the data pipeline failed: {asset.get('error')}")
        return draft

    def agent_spoke(self, draft_id: str, after: int = 0) -> dict:
        return self.wait(f"/drafts/{draft_id}", lambda d: len(d["messages"]) > after and d["messages"][-1]["role"] == "agent", 120, "the agent's reply")

    def answer(self, draft_id: str, answers: list[str]) -> dict:
        draft = self.agent_spoke(draft_id)
        for text in answers:
            last = [m for m in draft["messages"] if m["role"] == "agent"][-1]
            if last.get("kind") == "summary":
                break
            count = len(draft["messages"])
            self.say(draft_id, text)
            draft = self.agent_spoke(draft_id, after=count + 1)
        return draft

    def say(self, draft_id: str, text: str) -> None:
        """Send a chat message; the server answers 409 while the agent is still speaking, so try again briefly."""
        for _ in range(60):
            try:
                self.call("POST", f"/drafts/{draft_id}/messages", {"text": text}, expect=202)
                return
            except Failed as error:
                if "answered 409" not in str(error):
                    raise
                time.sleep(0.5)
        raise Failed("the agent stayed busy for 30 seconds")

    def team(self, draft_id: str, answers: dict[str, str], steps: int = 30) -> dict:
        """28.8: walk the draft's understanding team through its API to its end: start it, answer each question by its field
        (an unknown field gets an empty answer: the walk declines it), approve each confirmation and approval."""
        body: dict = {}
        out: dict = {}
        for _ in range(steps):
            for _ in range(60):
                try:
                    out = self.call("POST", f"/drafts/{draft_id}/team", body)
                    break
                except Failed as error:
                    if "answered 409" not in str(error):
                        raise
                    time.sleep(0.5)
            if out.get("status") != "paused":
                return out
            asks = out.get("asks") or {}
            body = {"answer": answers.get(str(asks.get("field")), "")} if asks.get("kind") == "question" else {"approve": True}
        raise Failed(f"the team was still walking after {steps} pauses")

    def run_all(self, project_id: str) -> dict:
        self.call("POST", f"/projects/{project_id}/run")
        project = self.wait(f"/projects/{project_id}", lambda p: all(p["stages"][s]["status"] in ("completed", "failed", "approved") for s in STAGES), what="the five stages")
        states = {s: project["stages"][s]["status"] for s in STAGES}
        check(all(v in ("completed", "approved") for v in states.values()), f"stages did not all complete: {states}")
        return project


def holdout(project: dict) -> str:
    final = project["records"]["final"]
    evidence, metric = final["evidence"], final["primary_metric"]
    interval = evidence["holdout_primary_metric_ci"]
    return f"{metric} {evidence['holdout_metrics'][metric]:.4f} (95% {interval['low']:.4f} to {interval['high']:.4f}), model {evidence['model']}"


def solution_from(proposal: dict, moment: str, time_column: str | None = None) -> dict:
    binary = proposal["task"] == "binary"
    return {"target": proposal["target"], "task": proposal["task"], "positive_label": str(proposal["positive_label"]) if binary else None,
            "prediction_moment": moment, "forbidden": [{"column": f["column"], "reason": f["reason"]} for f in proposal["forbidden"]],
            "identifiers": proposal["identifiers"], "time_column": time_column, "group_column": None, "text_columns": [],
            "metric": proposal["metric"], "notes": ""}


def churn_table(rows: int = 3000, seed: int = 7) -> tuple[bytes, int]:
    """The churn table the upload flows send: the Telco file when this checkout has it, else an invented table with the same
    shape and traps (an identifier, blank charges for new customers that need cleaning, a Yes/No outcome with a real signal)."""
    if TELCO.is_file():
        return TELCO.read_bytes(), 7043
    import math

    rng = random.Random(seed)
    lines = ["customerID,gender,SeniorCitizen,Partner,tenure,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn"]
    for i in range(rows):
        tenure = 0 if rng.random() < 0.01 else rng.randint(1, 72)
        contract = rng.choices(["Month-to-month", "One year", "Two year"], [0.55, 0.25, 0.2])[0]
        monthly = round(rng.uniform(18, 118), 2)
        paperless = rng.choice(["Yes", "No"])
        payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        z = -1.2 - 0.045 * tenure + 0.02 * (monthly - 65) + (1.3 if contract == "Month-to-month" else -0.6) + (0.4 if payment == "Electronic check" else 0)
        churn = "Yes" if rng.random() < 1 / (1 + math.exp(-z)) else "No"
        total = " " if tenure == 0 else f"{monthly * tenure * rng.uniform(0.95, 1.05):.2f}"
        lines.append(f"{1000 + i}-{''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(5))},{rng.choice(['Female', 'Male'])},"
                     f"{int(rng.random() < 0.16)},{rng.choice(['Yes', 'No'])},{tenure},{contract},{paperless},{payment},{monthly},{total},{churn}")
    return ("\n".join(lines) + "\n").encode(), rows


# ------------------------------------------------------------------------------------------------ flows
def flow_upload(api: Api) -> str:
    table, rows = churn_table()
    draft_id = api.draft("Rank telecom customers by their risk of leaving next month so the retention team can call them.")
    api.call("PUT", f"/drafts/{draft_id}/data?filename=telco.csv", raw=table)
    draft = api.asset_ready(draft_id)
    asset = draft["assets"][-1]
    check(asset["rows"] == rows and not asset.get("synthetic"), f"unexpected table: {asset.get('rows')} rows")
    check(draft["cleaning_log"], "the cleaning log is empty for a file that needs cleaning")
    check((draft["pack"] or {}).get("key") == "tabular", f"a churn problem was put in the {draft['pack']} pack")
    draft = api.answer(draft_id, ["Churn", "At the start of each month, before the retention call.", "A call costs 5 euro; a lost customer costs 300."])
    understood = draft["understanding"]
    check(understood.get("target") == "Churn" and understood.get("prediction_moment"), f"the chat did not record the answers: {understood}")
    check(len((draft["workflow"] or {}).get("nodes", [])) == 10, "the solution workflow is not drawn")
    proposal = api.call("POST", f"/drafts/{draft_id}/solution/proposal")  # the target comes from the chat
    check(proposal["target"] == "Churn" and "customerID" in proposal["identifiers"], "the proposal missed the target or the identifier")
    heard = proposal.get("decision") or {}  # 34.4: the amounts said in the chat fill the wizard's decision fields
    check(heard.get("heard_in") == "costs" and heard.get("cost_per_action") == 5 and heard.get("cost_false_negative") == 300,
          f"the proposal does not carry the amounts said in the chat: {heard}")
    api.call("PUT", f"/drafts/{draft_id}/solution", solution_from(proposal, understood["prediction_moment"]))
    settings = api.call("PUT", f"/drafts/{draft_id}/settings", {"split": "time", "quick": True, "max_rows": rows, "folds": 5, "budget": {"calls": 24, "minutes": 20, "eur": 5}})["settings"]
    check(settings["split"] == "stratified", "the stored split is not what the solution implies (no time column)")
    project = api.call("POST", f"/drafts/{draft_id}/build")
    check(project["settings"]["folds"] == 5 and project["data"]["synthetic"] is False, "folds or the data label did not reach the project")
    project = api.run_all(project["id"])
    protocol = project["records"]["data"]["evidence"]["cv_protocol"]
    check("StratifiedKFold(5," in protocol, f"the wizard's 5 folds did not reach the cross-validation: {protocol}")
    notebook = json.dumps(api.call("GET", f"/projects/{project['id']}/export/notebook"))
    check("pd.read_parquet(" in notebook and "n_splits=5" in notebook, "the exported notebook does not read the data file or the folds as run")
    check("Synthetic data." not in notebook, "real data was labelled synthetic")
    check(all(m["status"] == "allowed" for m in project["transitions"]), "the validator refused a move on the standard path")
    decision = api.call("GET", f"/projects/{project['id']}/decision")  # 34.4: "a call costs 5; a lost customer costs 300", read and chosen
    heard = [e for e in api.call("GET", f"/projects/{project['id']}").get("activity", []) if e.get("kind") in ("decision_read", "decision_unread")]
    check(decision["status"] == "chosen" and decision["stated"]["what"] == "call",
          f"the decision said in the chat was not chosen: {decision['status']}; the chat heard action {understood.get('action')!r}, costs {understood.get('costs')!r}; {heard}")
    check(decision["what_if"]["available"], f"no what-if on the chosen decision: {decision['what_if'].get('why_not')}")
    dearer = api.call("POST", f"/projects/{project['id']}/decision/what-if", {"changes": {"cost_per_action": 50}})
    check(dearer["after"]["share"] <= dearer["before"]["share"], "a dearer call made more calls")
    return (f"{rows:,} rows, 5 folds, five stages; holdout {holdout(project)}; calls {decision['chosen']['share']:.1%} of customers, "
            f"{dearer['after']['share']:.1%} at 50 a call")


def flow_upload_team(api: Api) -> str:
    """28.8: the upload flow with the team's switch on: the chat's answers, then Build goes through the understanding team's walk
    (the build being its approval); what the walk asks first is answered through its own API; the project it builds runs every
    stage."""
    if not api.exists("/team/settings"):
        return "skipped: the understanding team is not part of this edition"
    table, _ = churn_table()
    check(api.call("PUT", "/team/settings", {"on": True})["settings"]["on"] is True, "the team's switch did not turn on")
    draft_id = api.draft("Rank telecom customers by their risk of leaving next month so the retention team can call them.")
    api.call("PUT", f"/drafts/{draft_id}/data?filename=telco.csv", raw=table)
    api.asset_ready(draft_id)
    draft = api.answer(draft_id, ["Churn", "At the start of each month, before the retention call.", "A call costs 5 euro; a lost customer costs 300."])
    answers = {"target": "Churn", "outcome": "Churn", "prediction_moment": "At the start of each month, before the retention call.",
               "action": "The retention team calls the riskiest; a call costs 5 euro, a lost customer 300.", "entity": "One customer",
               "labels": "A column marks them", "data_plan": "The table is uploaded"}
    asked = 0
    for _ in range(4):  # "build it" in the chat, the page's own path (the Build button sends it): the walk builds, or asks first
        count = len(api.call("GET", f"/drafts/{draft_id}")["messages"])
        api.say(draft_id, "build it")
        draft = api.wait(f"/drafts/{draft_id}", lambda d: d.get("project_id") or (len(d["messages"]) > count + 1 and d["messages"][-1]["role"] == "agent"),
                         180, "the build or the team's question")
        if draft.get("project_id"):
            break
        said = (draft["messages"][-1].get("text") or "").lower()
        check("team asks first" in said or "team" in said, f"the build was refused without the team: {said[:200]}")
        asked += 1
        api.team(draft_id, answers)  # what the walk asks first, answered through its own API (it may build at its end)
        if api.call("GET", f"/drafts/{draft_id}").get("project_id"):
            break
    else:
        raise Failed("Build kept waiting on the team")
    draft = api.call("GET", f"/drafts/{draft_id}")
    # the draft has its project once the build node returns; the walk writes its end record when it ends, a moment later
    board = api.wait(f"/drafts/{draft_id}/team", lambda v: (v.get("walk") or {}).get("status") == "completed", 60, "the walk's end record")
    project = api.call("GET", f"/projects/{draft['project_id']}")
    check(bool(project.get("solution")), "the project the team built has no solution")
    draft = api.wait(f"/drafts/{draft_id}", lambda d: any(m["role"] == "agent" and m.get("kind") == "build" for m in d["messages"]), 120,
                     "the chat's build message")  # it comes after the build returns
    started = any(m.get("kind") == "build" and m.get("session_id") for m in draft["messages"] if m["role"] == "agent")
    if started:  # the build started the agent that runs the stages: follow it, never a second runner
        project = api.wait(f"/projects/{project['id']}", lambda p: all(p["stages"][s]["status"] in ("completed", "failed", "approved") for s in STAGES),
                           900, "the five stages the agent runs")
        check(all(project["stages"][s]["status"] in ("completed", "approved") for s in STAGES), f"stages: {[project['stages'][s]['status'] for s in STAGES]}")
    else:
        project = api.run_all(project["id"])
    check(all(m["status"] == "allowed" for m in project["transitions"]), "the validator refused a move on the team's path")
    api.call("PUT", "/team/settings", {"on": False})
    return f"\"build it\" through the team ({board.get('handoffs')} hand-offs, {asked} pause(s) answered first, stages {'by the agent' if started else 'by the run'}); holdout {holdout(project)}"


def flow_no_data(api: Api) -> str:
    draft_id = api.draft("We want to predict which delivery orders will arrive late so dispatch can react in time.")
    draft = api.answer(draft_id, ["Late means more than 30 minutes after the promised time.", "We decide when the order is dispatched.", "Re-routing costs little; a late order costs a refund."])
    asked = " ".join(m.get("text") or "" for m in draft["messages"] if m["role"] == "agent").lower()
    check("sample" in asked, "the agent never asked for a sample of the data")
    check(len((draft["workflow"] or {}).get("nodes", [])) == 10, "no solution workflow without data")
    project = api.call("POST", f"/drafts/{draft_id}/build")
    check(not project.get("data"), "a project without data claims to have data")
    api.call("POST", f"/projects/{project['id']}/run", expect=409)  # nothing can run before a solution and data exist
    return "questions asked, a sample requested, workflow drawn, plan-only project built"


def flow_synthetic(api: Api) -> str:
    listing = api.call("GET", "/synthetic/templates")
    check("demand" in [t["key"] for t in listing["templates"]], "the demand template is not listed")
    draft_id = api.draft("Forecast daily unit sales per store so we can plan stock.")
    api.call("PUT", f"/drafts/{draft_id}/data?filename=x.csv", raw=b"", expect=422)  # an empty upload is refused
    api.call("POST", f"/drafts/{draft_id}/data/synthetic", {"template": "demand", "rows": 3000}, expect=202)
    draft = api.asset_ready(draft_id)
    asset = draft["assets"][-1]
    check(asset["synthetic"] is True and asset.get("template") == "demand" and asset["suggestion"] == {"target": "units_sold"}, "the synthetic asset is not labelled")
    check((draft["pack"] or {}).get("key") == "timeseries", f"a forecasting problem was put in the {draft['pack']} pack")
    proposal = api.call("POST", f"/drafts/{draft_id}/solution/proposal")
    check(proposal["target"] == "units_sold" and proposal["task"] == "regression", f"wrong default target or task: {proposal['target']} {proposal['task']}")
    check(proposal["prediction_moment"] is None, "guidance text was offered as a prediction moment")
    time_column = proposal["time_candidates"][0]
    check(time_column not in {f["column"] for f in proposal["forbidden"]}, "the time column is flagged as forbidden")
    api.call("PUT", f"/drafts/{draft_id}/solution", solution_from(proposal, "Each evening, for the next day, before the store opens.", time_column))
    settings = api.call("PUT", f"/drafts/{draft_id}/settings", {"split": "stratified", "quick": True, "max_rows": 3000, "folds": 5, "budget": {}})["settings"]
    check(settings["split"] == "time", "a time column is declared but the stored split is not by time")
    project = api.run_all(api.call("POST", f"/drafts/{draft_id}/build")["id"])
    check("TimeSeriesSplit(5)" in project["records"]["data"]["evidence"]["cv_protocol"], "the time split does not use the wizard's folds")
    notebook = json.dumps(api.call("GET", f"/projects/{project['id']}/export/notebook"))
    report = api.call("GET", f"/projects/{project['id']}/export/report")
    check("Synthetic data." in notebook and "Synthetic data." in str(report), "an export does not say the data is synthetic")
    check(re.search(r"TimeSeriesSplit\(n_splits=5\)", notebook) is not None, "the notebook does not reproduce the time split")
    return f"template table labelled synthetic, split by time with 5 folds; holdout {holdout(project)} (synthetic: says nothing about real cases)"


def flow_log_file(api: Api) -> str:
    rng = random.Random(7)
    lines = [f'10.0.{i % 7}.{i % 200} - - [05/Oct/2026:10:{i // 60 % 60:02d}:{i % 60:02d} +0000] "GET /api/v1/items/{i % 50} HTTP/1.1" '
             f'{rng.choice([200] * 8 + [404, 500])} {rng.randint(200, 9000)} "-" "curl/8.4"' for i in range(400)]
    draft_id = api.draft("Find which requests to our API will fail so we can alert before users notice.")
    api.call("PUT", f"/drafts/{draft_id}/data?filename=access.log", raw="\n".join(lines).encode())
    draft = api.asset_ready(draft_id)
    asset, structure = draft["assets"][-1], draft["structure"]
    check(asset["rows"] == 400 and asset["columns"] >= 6, f"the log was not read as a table: {asset.get('rows')} rows, {asset.get('columns')} columns")
    check(structure.get("parse_rate", 0) >= 0.99, f"parse rate {structure.get('parse_rate')}")
    check((draft["analysis"] or {}).get("summary", {}).get("rows") == 400, "the log table was not analysed")
    return f"400 log lines read as {asset['columns']} columns ({structure.get('format')}, parse rate {structure['parse_rate']:.0%})"


FLOWS = {"upload": flow_upload, "upload-team": flow_upload_team, "no-data": flow_no_data, "synthetic": flow_synthetic, "log-file": flow_log_file}


# ------------------------------------------------------------------------------------------------ server
# the loops a server's worker starts on its own (the store's conductor, the collector, the students' retraining): off on a
# test server, as tests/run_parallel.py sets them, so no run outlives it writing into the owner's store (2026-10-25: five
# orphaned store runs on deleted temporary folders, found by the collector's session)
BACKGROUND_OFF = {"DCLAB_STORE_LOOP": "0", "DCLAB_STORE_CAPTURE": "0", "DCLAB_COLLECTOR": "0", "DCLAB_STUDENTS_LOOP": "0"}


def _main_checkout() -> Path:
    """The repository's main folder, also from a worktree (make check-commit runs the gate in one it then removes)."""
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        return (ROOT / common).resolve().parent if common else ROOT
    except Exception:  # noqa: BLE001 - outside git: this folder
        return ROOT


MIN_FREE_BYTES = 2 * 1024 ** 3  # the owner's rule: stop below 2 GB of free disk


def keep_workspace(home: tempfile.TemporaryDirectory | None, name: str, root: Path | None = None) -> Path | None:
    """The owner's rule (2026-10-25: capture everything for the six models): the test server's whole workspace (its
    sessions, ledger, projects, drafts and events) is copied under ``agent_runs/e2e/<day>/`` of the main checkout before
    its temporary folder goes, so the store collects it; a ``scripted.json`` beside it says a script drove it (its "person"
    answers are clicks: the lake never files them as people's). ``DCLAB_E2E_KEEP=0`` skips it (a container whose disk
    is thrown away); so does a disk under 2 GB free, and a copy that fails only warns: the tests' result stands."""
    import shutil

    if home is None or os.environ.get("DCLAB_E2E_KEEP", "1") == "0":
        return None
    base = (root or (_main_checkout() / "agent_runs" / "e2e"))
    try:
        base.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(base).free < MIN_FREE_BYTES:
            print("not kept: under 2 GB of free disk", file=sys.stderr)
            return None
        (base / "scripted.json").write_text(json.dumps({"by": "script", "why": "the end-to-end and browser tests: a person's answers "
                                                        "here were clicked by a test script, never a person's label"}) + "\n")
        out = base / time.strftime("%Y-%m-%d") / f"{time.strftime('%H%M%S')}_{name}"
        shutil.copytree(home.name, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.lock", "*.part"))
        return out
    except (OSError, shutil.Error) as error:
        print(f"not kept: {type(error).__name__}: {error}", file=sys.stderr)
        return None


# dclab:closed-begin  (the flows' results kept for the training store: the full edition's)
def keep_results(results: list[dict], argv: list[str], wall: float, kept: Path | None, where: Path | None = None) -> Path | None:
    """Each flow's result kept whole (its name, verdict, the whole line it printed, its seconds) as a run beside the test
    suite's (``training_records/test_runs/<day>/<run>.jsonl.gz``, indexed in ``runs.jsonl``; the store's test_runs keeper
    takes them), and as ``e2e_result.json`` beside the kept workspace. ``DCLAB_TEST_RUNS`` chooses where, ``off`` keeps
    none, as for the suite; a failure to keep it is said, never fatal."""
    import gzip
    from datetime import datetime, timezone

    def git(*args: str) -> str:
        try:
            out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    chosen = os.environ.get("DCLAB_TEST_RUNS", "").strip()
    if chosen.lower() == "off":
        return None
    where = where or (Path(chosen) if chosen else _main_checkout() / "training_records" / "test_runs")
    now = datetime.now(timezone.utc)
    commit = git("rev-parse", "HEAD")
    run = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{(commit or 'nogit')[:10]}-e2e-{os.getpid()}"
    failing = [r["flow"] for r in results if not r["ok"]]
    header = {"kind": "e2e_run", "run": run, "at": now.isoformat(timespec="seconds"), "commit": commit,
              "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
              "worktree": str(ROOT), "argv": list(argv), "ok": not failing, "wall_seconds": round(wall, 1), "flows": len(results),
              "failed": len(failing), "failing_flows": failing, "workspace": str(kept) if kept else None, "python": sys.version.split()[0],
              "env": {k: os.environ.get(k) for k in ("DCLAB_NO_LIVE_MODELS", "DCLAB_E2E_KEEP") if os.environ.get(k) is not None}}
    flows = [{"kind": "flow", "run": run, **r, "seconds": round(r["seconds"], 2)} for r in results]
    try:
        day = where / now.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        path = day / f"{run}.jsonl.gz"
        with gzip.open(path, "xt", encoding="utf-8") as fh:
            for line in [header, *flows]:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        with open(where / "runs.jsonl", "a", encoding="utf-8") as fh:  # the suite's index: the same keys, so its readers read it too
            fh.write(json.dumps({"kind": "e2e_run", **{k: header[k] for k in ("run", "at", "commit", "dirty", "argv", "ok", "wall_seconds", "failed")},
                                 "pg": False, "units": len(results), "tests": len(results), "skipped": 0, "failing_files": failing,
                                 "path": str(path.relative_to(where))}, ensure_ascii=False) + "\n")
        if kept is not None:
            (kept / "e2e_result.json").write_text(json.dumps({**header, "flows": flows}, ensure_ascii=False, indent=1) + "\n")
        return path
    except OSError as error:
        print(f"note: the flows' results were not kept ({error})", file=sys.stderr)
        return None
# dclab:closed-end


def start_server() -> tuple[subprocess.Popen, str, tempfile.TemporaryDirectory]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    home = tempfile.TemporaryDirectory(prefix="dclab-e2e-")
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("DCLAB_TIER_")}, "DCLAB_AGENT_HOME": home.name, "OPENAI_API_KEY": "",
           "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1", "DCLAB_NO_LIVE_MODELS": "1"}  # no model, whatever .env says
    for key, value in BACKGROUND_OFF.items():  # as tests/run_parallel.py: a test server's worker never writes into the owner's store
        env.setdefault(key, value)
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "dclab_rnd.agentic.server:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(80):
        if process.poll() is not None:
            raise SystemExit("the server did not start (run it by hand to see why: make notebook)")
        try:
            urllib.request.urlopen(base + "/api/config", timeout=2).read()
            return process, base, home
        except OSError:
            time.sleep(0.5)
    process.terminate()
    raise SystemExit("the server did not answer in 40 seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", help="URL of a running server; without it the script starts its own, with no model and an empty workspace. "
                        "On a server you started, whatever it is configured with applies: a configured model key means the flows send model requests. "
                        "With sign-in on, give an API token in DCLAB_E2E_TOKEN")
    parser.add_argument("--only", choices=sorted(FLOWS), action="append", help="run only this flow (repeatable)")
    args = parser.parse_args(argv)
    process = home = None
    base = args.base
    if not base:
        process, base, home = start_server()
    failed = 0
    results: list[dict] = []
    began = time.time()
    try:
        api = Api(base)
        for name in args.only or list(FLOWS):
            started = time.time()
            try:
                detail = FLOWS[name](api)
                results.append({"flow": name, "ok": True, "detail": str(detail), "seconds": time.time() - started})
                print(f"PASS  {name:<10} {detail}  [{time.time() - started:.0f}s]", flush=True)
            except Failed as error:
                failed += 1
                results.append({"flow": name, "ok": False, "detail": str(error), "seconds": time.time() - started})
                print(f"FAIL  {name:<10} {error}", flush=True)
            except BaseException as error:  # an unexpected error ends the run: kept too, then raised
                results.append({"flow": name, "ok": False, "detail": f"{type(error).__name__}: {error}", "seconds": time.time() - started})
                raise
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:  # a job still running in the server: never leave it behind
                process.kill()
                process.wait(10)
        kept = None
        if home is not None:
            kept = keep_workspace(home, "product_e2e")
            if kept is not None:
                print(f"kept: {kept}")
            home.cleanup()
        # dclab:closed-begin  (the flows' results kept for the training store: the full edition's)
        if results:
            recorded = keep_results(results, list(argv if argv is not None else sys.argv[1:]), time.time() - began, kept)
            if recorded is not None:
                print(f"results kept: {recorded}")
        # dclab:closed-end
    print("all flows passed" if not failed else f"{failed} flow(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

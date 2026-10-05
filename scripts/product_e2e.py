#!/usr/bin/env python3
"""Run the product's main flows against a temporary server, the way a person would through Home and the wizard.

    make product-e2e                              # starts its own server on a spare port, with an empty workspace
    python scripts/product_e2e.py --base http://127.0.0.1:8765     # or use a server that is already running

Flows (the plan's verification list, docs/guides/PRODUCT_BUILD_PLAYBOOK.md package 7.1):

  upload      one sentence, the Telco CSV uploaded, three answers, the solution, 5 folds, build, five stages, exports
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
        self.base, self.token = base.rstrip("/") + "/api", None
        self.token = self.call("GET", "/config")["csrf"]

    def call(self, method: str, path: str, body=None, raw: bytes | None = None, expect: int | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.base + path, data=data, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self.token and method != "GET":
            request.add_header("X-DCLab-Token", self.token)
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


# ------------------------------------------------------------------------------------------------ flows
def flow_upload(api: Api) -> str:
    check(TELCO.is_file(), f"{TELCO.relative_to(ROOT)} is missing")
    draft_id = api.draft("Rank telecom customers by their risk of leaving next month so the retention team can call them.")
    api.call("PUT", f"/drafts/{draft_id}/data?filename=telco.csv", raw=TELCO.read_bytes())
    draft = api.asset_ready(draft_id)
    asset = draft["assets"][-1]
    check(asset["rows"] == 7043 and not asset.get("synthetic"), f"unexpected table: {asset.get('rows')} rows")
    check(draft["cleaning_log"], "the cleaning log is empty for a file that needs cleaning")
    check((draft["pack"] or {}).get("key") == "tabular", f"a churn problem was put in the {draft['pack']} pack")
    draft = api.answer(draft_id, ["Churn", "At the start of each month, before the retention call.", "A call costs 5 euro; a lost customer costs 300."])
    understood = draft["understanding"]
    check(understood.get("target") == "Churn" and understood.get("prediction_moment"), f"the chat did not record the answers: {understood}")
    check(len((draft["workflow"] or {}).get("nodes", [])) == 10, "the solution workflow is not drawn")
    proposal = api.call("POST", f"/drafts/{draft_id}/solution/proposal")  # the target comes from the chat
    check(proposal["target"] == "Churn" and "customerID" in proposal["identifiers"], "the proposal missed the target or the identifier")
    api.call("PUT", f"/drafts/{draft_id}/solution", solution_from(proposal, understood["prediction_moment"]))
    settings = api.call("PUT", f"/drafts/{draft_id}/settings", {"split": "time", "quick": True, "max_rows": 7043, "folds": 5, "budget": {"calls": 24, "minutes": 20, "eur": 5}})["settings"]
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
    return f"7,043 rows, 5 folds, five stages; holdout {holdout(project)}"


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


FLOWS = {"upload": flow_upload, "no-data": flow_no_data, "synthetic": flow_synthetic, "log-file": flow_log_file}


# ------------------------------------------------------------------------------------------------ server
def start_server() -> tuple[subprocess.Popen, str, tempfile.TemporaryDirectory]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    home = tempfile.TemporaryDirectory(prefix="dclab-e2e-")
    env = {**os.environ, "DCLAB_AGENT_HOME": home.name, "OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"}
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
    parser.add_argument("--base", help="URL of a running server; without it the script starts its own, with no model and an empty workspace")
    parser.add_argument("--only", choices=sorted(FLOWS), action="append", help="run only this flow (repeatable)")
    args = parser.parse_args(argv)
    process = home = None
    base = args.base
    if not base:
        process, base, home = start_server()
    failed = 0
    try:
        api = Api(base)
        for name in args.only or list(FLOWS):
            started = time.time()
            try:
                print(f"PASS  {name:<10} {FLOWS[name](api)}  [{time.time() - started:.0f}s]", flush=True)
            except Failed as error:
                failed += 1
                print(f"FAIL  {name:<10} {error}", flush=True)
    finally:
        if process is not None:
            process.terminate()
            process.wait(10)
        if home is not None:
            home.cleanup()
    print("all flows passed" if not failed else f"{failed} flow(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

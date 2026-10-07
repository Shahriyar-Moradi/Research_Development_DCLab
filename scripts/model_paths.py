#!/usr/bin/env python3
"""Every path that uses a model, run once against a real model on synthetic data (package 8.1).

    python scripts/model_paths.py --ollama qwen2.5-coder:1.5b     # a local model through Ollama: free, nothing leaves the machine
    python scripts/model_paths.py --configured --cap-eur 2         # the model configured in .env (OpenAI): costs money, asks first

It starts its own server on an empty workspace and drives, through the API as the page does:

  synthetic   a table designed by the model from a description (no template)
  chat        a short conversation on Home: the replies stream, the agent asks, records answers, draws the workflow
  parse       an unusual log file no built-in reader knows: the model proposes a parse pattern, code applies it
  evidence    one question the evidence covers and one it does not: an answer with citations, and a refusal
  intern      one intern session with a small budget, on a project built here from a made-up table

For each path it reports what the model proposed and what the code accepted or refused, then the requests and tokens
used (GET /api/models). It sends no real data: the problem sentences, the log lines and the tables are made up here.
A path whose model requests all failed (no credit, a wrong model name) is a FAIL, never a refusal by the validators.
Exit status 0 when every path ran (a path whose model output the code correctly refused still ran).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLLAMA = "http://localhost:11434/v1"


class Failed(AssertionError):
    pass


class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/") + "/api"
        self.token = self.call("GET", "/config")["csrf"]

    def call(self, method: str, path: str, body=None, raw: bytes | None = None, timeout: float = 300):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.base + path, data=data, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if getattr(self, "token", None) and method != "GET":
            request.add_header("X-DCLab-Token", self.token)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                text = answer.read().decode()
                return json.loads(text) if text else None
        except urllib.error.HTTPError as error:
            raise Failed(f"{method} {path} answered {error.code}: {error.read().decode()[:300]}") from None

    def wait(self, path: str, ready, seconds: float = 240, what: str = "it"):
        deadline = time.time() + seconds
        while time.time() < deadline:
            found = self.call("GET", path)
            if ready(found):
                return found
            time.sleep(1)
        raise Failed(f"{what} did not happen within {seconds:.0f} s")

    def events(self, draft_id: str, after: int = 0, seconds: float = 3) -> list[dict]:
        """The draft's event stream (Server-Sent Events) read for a few seconds."""
        events, kind = [], None
        request = urllib.request.Request(f"{self.base}/drafts/{draft_id}/events?after={after}")
        try:
            with urllib.request.urlopen(request, timeout=seconds) as stream:
                for raw in stream:
                    line = raw.decode().rstrip("\n")
                    if line.startswith("event:"):
                        kind = line[6:].strip()
                    elif line.startswith("data:"):
                        events.append({"kind": kind, "data": line[5:].strip()})
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            pass
        return events


# ------------------------------------------------------------------------------------------------ paths
def path_synthetic(api: Api) -> str:
    draft = api.call("POST", "/drafts", {"problem": "Predict which delivery orders will arrive late, from what is known when the order is placed."})
    api.call("POST", f"/drafts/{draft['id']}/data/synthetic",
             {"prompt": "Delivery orders: distance in km, weight in kg, a weekday, a weather label, a courier rating, and whether it arrived late (about 20%).", "rows": 400})
    d = api.wait(f"/drafts/{draft['id']}", lambda d: d["assets"] and d["assets"][-1]["status"] in ("ready", "failed"), what="the synthetic table")
    asset = d["assets"][-1]
    if asset["status"] != "ready":
        return f"failed: {asset.get('error')}"
    if asset.get("spec_source") == "model":
        return f"the model's table design was accepted: {asset.get('rows')} rows, {asset.get('columns')} columns, synthetic={asset.get('synthetic')}"
    return (f"the model's design was refused, so the code used the {asset.get('template')!r} template, and says so: "
            f"{(asset.get('template_note') or '')[:90]}… ({asset.get('rows')} rows, {asset.get('columns')} columns, synthetic={asset.get('synthetic')})")


def path_chat(api: Api) -> str:
    draft = api.call("POST", "/drafts", {"problem": "Forecast how many bikes each station will need tomorrow morning."})
    did = draft["id"]
    api.wait(f"/drafts/{did}", lambda d: any(m.get("role") == "agent" for m in d["messages"]), what="the agent's first message")
    for answer in ("Tomorrow at 7 am, from yesterday's counts and the weather forecast.", "Running out of bikes costs more than a spare bike.",
                   "Please draw the solution's steps for this problem as a workflow."):
        for _ in range(60):
            try:
                api.call("POST", f"/drafts/{did}/messages", {"text": answer})
                break
            except Failed as error:
                if "409" not in str(error):
                    raise
                time.sleep(1)
        before = len(api.call("GET", f"/drafts/{did}")["messages"])
        api.wait(f"/drafts/{did}", lambda d, n=before: len(d["messages"]) > n and d["messages"][-1].get("role") == "agent", what="the agent's reply")
    events = api.events(did)
    tokens = sum(1 for e in events if e["kind"] == "token" and '"drop"' not in e["data"])
    d = api.call("GET", f"/drafts/{did}")
    workflow = d.get("workflow") or {}
    asked = sum(1 for m in d["messages"] if m.get("role") == "agent" and (m.get("question") or m.get("ask")))
    return (f"{len(d['messages'])} messages, {tokens} streamed chunks, {asked} question(s); understood {sorted((d.get('understanding') or {}))}; "
            f"workflow v{workflow.get('version')} with {len(workflow.get('nodes') or [])} nodes: "
            + ("the model proposed it and the workflow check accepted it" if workflow.get("source") == "model" else
               "the model did not propose one (or the check refused it), so it is the pack's template")
            + f"; agent mode {(d.get('agent') or {}).get('mode')}")


def path_parse(api: Api) -> str:
    draft = api.call("POST", "/drafts", {"problem": "Find which machine events come before a failure."})
    rnd = random.Random(8)  # a format no built-in reader knows (without a model it is read as plain text lines)
    notes = ["fan noisy", "", "after restart", "", "door open", ""]
    lines = []
    for _ in range(300):
        note = rnd.choice(notes)
        lines.append(f"{rnd.choice(['A7', 'B2', 'C9'])} reported {rnd.uniform(40, 90):.1f}C at {rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d} "
                     f"-> {rnd.choice(['ok', 'warn', 'fault'])}" + (f" ({note})" if note else ""))
    api.call("PUT", f"/drafts/{draft['id']}/data?filename=machines.weird", raw="\n".join(lines).encode())
    d = api.wait(f"/drafts/{draft['id']}", lambda d: d["assets"] and d["assets"][-1]["status"] in ("ready", "failed"), what="the parsed log")
    asset, structure = d["assets"][-1], d.get("structure") or {}
    if asset["status"] != "ready":
        return f"not parsed: {asset.get('error')} (the code refused it)"
    return (f"{asset['rows']} rows x {asset['columns']} columns, format {structure.get('format')}, parser {structure.get('parser')}, "
            f"parse rate {structure.get('parse_rate')}; {' '.join(n for n in structure.get('notes') or [] if 'model' in n)[:260]}")


def path_evidence(api: Api) -> str:
    out = []
    for question in ("Should a scaler be fitted before or after the train/test split?", "What is the best pizza topping in Naples?"):
        a = api.call("POST", "/evidence/ask", {"question": question})
        mode = a.get("mode")
        state = ("no record matched, so no model was asked" if mode == "none" else
                 "answered with citations" if mode == "model" and a.get("covered") else
                 "the model said the evidence does not cover it" if mode == "model" else
                 f"the model's answer was refused by the citation check ({a.get('dropped')} dropped), the closest records shown"
                 if a.get("dropped") else f"records only: {a.get('note') or ''}"[:120])
        out.append(f"{question[:40]!r}: {state} [mode {a.get('mode')}, {len(a.get('sentences') or [])} sentences, {len(a.get('records') or [])} records]")
    return "; ".join(out)


def made_up_project(api: Api) -> str:
    """A project on a built-in synthetic table (no model asked to design it): the intern works on made-up rows only."""
    did = api.call("POST", "/drafts", {"problem": "Predict which customers will leave next month."})["id"]
    api.call("POST", f"/drafts/{did}/data/synthetic", {"template": "churn", "rows": 600})
    api.wait(f"/drafts/{did}", lambda d: d["assets"] and d["assets"][-1]["status"] in ("ready", "failed"), what="the synthetic table")
    p = api.call("POST", f"/drafts/{did}/solution/proposal")
    solution = {"target": p["target"], "task": p["task"], "positive_label": str(p["positive_label"]) if p["task"] == "binary" else None,
                "prediction_moment": "At the start of each month, from the account as it stands that day.",
                "forbidden": [{"column": f["column"], "reason": f["reason"]} for f in p["forbidden"]], "identifiers": p["identifiers"],
                "time_column": None, "group_column": None, "text_columns": [], "metric": p["metric"], "notes": ""}
    api.call("PUT", f"/drafts/{did}/solution", solution)
    api.call("PUT", f"/drafts/{did}/settings", {"split": "stratified", "quick": True, "max_rows": 600, "folds": 3, "budget": {}})
    return api.call("POST", f"/drafts/{did}/build")["id"]


def path_intern(api: Api) -> str:
    pid = made_up_project(api)
    s = api.call("POST", "/intern/sessions?wait=true",
                 {"task": "Look at this project's data and tell me whether any column could leak the outcome.", "project_id": pid,
                  "budget": {"max_steps": 5, "max_minutes": 4}}, timeout=600)
    tools = [step.get("tool") for step in s.get("steps") or []]
    refused = [step.get("tool") for step in s.get("steps") or [] if step.get("ok") is False]
    return f"status {s['status']}, mode {s.get('mode')}, {len(tools)} steps {tools}, refused or failed {refused}" + (f", error: {s['error']}" if s.get("error") else "")


PATHS = {"synthetic": path_synthetic, "chat": path_chat, "parse": path_parse, "evidence": path_evidence, "intern": path_intern}
PURPOSES = {"synthetic": ("synthetic_schema",), "chat": ("home_agent",), "parse": ("parse_pattern",), "evidence": ("evidence_answer",),
            "intern": ("intern",)}


# ------------------------------------------------------------------------------------------------ server
def start_server(model_env: dict[str, str]) -> tuple[subprocess.Popen, str, tempfile.TemporaryDirectory]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    home = tempfile.TemporaryDirectory(prefix="dclab-model-paths-")
    env = {k: v for k, v in os.environ.items() if k not in ("DCLAB_NO_LIVE_MODELS", "DCLAB_DATABASE_URL", "DCLAB_AUTH")}
    env.update({"DCLAB_AGENT_HOME": home.name, **model_env})  # an empty value here wins over .env (it is loaded without override)
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "dclab_rnd.agentic.server:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(80):
        if process.poll() is not None:
            home.cleanup()
            raise SystemExit("the server did not start")
        try:
            urllib.request.urlopen(base + "/api/config", timeout=2).read()
            return process, base, home
        except OSError:
            time.sleep(0.5)
    process.terminate()
    home.cleanup()
    raise SystemExit("the server did not answer in 40 seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--ollama", metavar="MODEL", help="a local model served by Ollama (for example qwen2.5-coder:1.5b)")
    which.add_argument("--configured", action="store_true", help="the model configured in .env (OpenAI): costs money")
    parser.add_argument("--cap-eur", type=float, default=2.0, help="with --configured: the spend cap for the run")
    parser.add_argument("--yes", action="store_true", help="with --configured: do not ask before sending")
    parser.add_argument("--only", choices=list(PATHS), action="append")
    args = parser.parse_args(argv)
    if args.ollama:  # every tier on Ollama, whatever .env routes elsewhere (checked after start)
        from dclab_rnd.models.settings import ROUTABLE

        model_env = {"OPENAI_API_KEY": "ollama", "OPENAI_BASE_URL": OLLAMA, "DCLAB_LLM_BASE_URL": OLLAMA, "OPENAI_MODEL": args.ollama,
                     "DCLAB_INTERN_MODEL": args.ollama}
        model_env.update({f"DCLAB_TIER_{t.upper()}_{n}": "" for t in ROUTABLE for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")})
    else:
        if not args.yes and input(f"Send made-up problem sentences, log lines and column summaries to the configured model, "
                                  f"up to {args.cap_eur} euros? [y/N] ").strip().lower() != "y":
            return 2
        model_env = {"DCLAB_WORKSPACE_MONTHLY_EUR": str(args.cap_eur)}
    process, base, home = start_server(model_env)
    failed = 0
    try:
        api = Api(base)
        tiers = api.call("GET", "/models")["tiers"]
        if args.ollama and not all(t.get("local") for t in tiers):
            raise SystemExit("a tier is not on the local model (check DCLAB_TIER_* in .env); nothing was sent")

        def by_purpose() -> dict:
            return api.call("GET", "/models")["usage"].get("by_purpose") or {}
        for name in args.only or list(PATHS):
            started, before = time.time(), by_purpose()
            try:
                said = PATHS[name](api)
                after = by_purpose()
                asked = sum((after.get(p) or {}).get("requests", 0) - (before.get(p) or {}).get("requests", 0) for p in PURPOSES[name])
                broke = sum((after.get(p) or {}).get("failed", 0) - (before.get(p) or {}).get("failed", 0) for p in PURPOSES[name])
                if asked and broke == asked:  # the model was never reached: what the path shows is the fallback, not a refusal
                    raise Failed(f"all {asked} model request(s) failed (no credit, a wrong model name, or the endpoint is down); the path showed: {said[:160]}")
                print(f"RAN   {name:<10} {said}  [{time.time() - started:.0f}s; {asked} request(s)]", flush=True)
            except Failed as error:
                failed += 1
                print(f"FAIL  {name:<10} {error}", flush=True)
        usage = api.call("GET", "/models")["usage"]
        print(f"usage: {usage.get('requests')} requests ({usage.get('failed')} failed), {usage.get('input_tokens')} tokens in, "
              f"{usage.get('output_tokens')} out, {usage.get('eur') or 0} euros; by purpose {json.dumps(usage.get('by_purpose'), default=str)[:600]}")
    finally:
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
        home.cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

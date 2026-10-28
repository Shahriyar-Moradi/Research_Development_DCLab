#!/usr/bin/env python3
"""Browser tests of the product (package 13.2): a real Chromium drives the real page, the way a person would.

    make browser-test                                   # starts its own server (no model, an empty workspace)
    python scripts/browser_e2e.py --base http://127.0.0.1:8765     # or a running server (CI: the container from 12.1)

Needs Playwright: pip install -r requirements/browser.txt && python -m playwright install chromium.

  flow     one sentence and the Telco CSV uploaded on Home, the chat's questions answered by clicking their options,
           the wizard's five steps with their defaults and a decision stated in its fields (34.4), a run of every stage
           from the Workflow page, then the brief and the decision chosen
  poll     the Workflow page follows a run across the gap between two stages and past a failed poll
  decide   34.4: a decision said in the chat ("a call costs 5 euro; a lost customer costs 300"), chosen by the final
           stage, on its page: the API's sentence, the curve, and a what-if typed into the form answering as the API does
  soledit  34.4: the Solution page's editor shows the stated decision; a changed cost and an untick reach the server
  team     28.8: the Team panel on Home: a question, the map's confirmation, the data given, the build approved
  stats    the numbers on Home, Compute, Integrations and Packs equal the API's (read over HTTP, not through the page)
  pages    all 21 pages at 1280 px: no console error, no uncaught error, no CSP violation, no failed-load warning
  mobile   all 21 pages at 375 px: nothing wider than the screen (no sideways scroll)

No model is used when the script starts its server: the chat asks its standard questions. Exit status 0 when every
check passes; a failing check saves a screenshot under the output folder (--out, default $TMPDIR/dclab-browser).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(ROOT))  # the team check writes a replica's files (dclab_core)
TELCO = ROOT / "data/project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
VIEWS = ["home", "new", "project", "solution", "notebook", "audit", "models", "evaluation", "decision", "reliability", "brief", "intern", "compute",
         "evidence", "lab", "benchmark", "policy", "packs", "integrations", "admin", "blueprint"]
STAGES = ("data", "leakage", "features", "models", "final")
# every page reports what goes wrong: uncaught errors, CSP violations (an event, not always a console line)
WATCH = """
window.__dclab = { errors: [] };
window.addEventListener('error', e => window.__dclab.errors.push('error: ' + e.message));
window.addEventListener('unhandledrejection', e => window.__dclab.errors.push('unhandled: ' + (e.reason && e.reason.message || e.reason)));
document.addEventListener('securitypolicyviolation', e => window.__dclab.errors.push('CSP: ' + e.violatedDirective + ' ' + e.blockedURI));
"""


class Failed(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failed(message)


def number(text: str) -> float:
    found = re.search(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    check(found is not None, f"no number in {text!r}")
    return float(found.group(0).replace(",", ""))


class Browser:
    def __init__(self, playwright, base: str, out: Path, width: int = 1280, height: int = 900):
        self.base, self.out = base.rstrip("/"), out
        channel = os.environ.get("DCLAB_BROWSER_CHANNEL", "").strip()  # "chrome": the Chrome installed on this machine, with a fresh profile
        self.browser = playwright.chromium.launch(**({"channel": channel} if channel else {}))
        token = os.environ.get("DCLAB_E2E_TOKEN", "").strip()  # against a server with sign-in: an API token of a test member, from the environment only
        self.context = self.browser.new_context(viewport={"width": width, "height": height}, accept_downloads=True,
                                                **({"extra_http_headers": {"Authorization": f"Bearer {token}"}} if token else {}))
        self.context.add_init_script(WATCH)
        self.page = self.context.new_page()
        self.console: list[str] = []
        self.page.on("console", lambda m: self.console.append(m.text) if m.type == "error" else None)
        self.page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))
        self.page.goto(self.base + "/#home")
        self.page.wait_for_selector("#nav .nav-item.active[data-nav='home']")

    def api(self, path: str):
        answer = self.context.request.get(self.base + "/api" + path)
        check(answer.ok, f"GET /api{path} answered {answer.status}")
        return answer.json()

    def view(self, name: str, settle: float = 0.4) -> None:
        """Open a page and wait until its own load has ended (the 13.1 tracker), not for a fixed time: a slow runner waits longer."""
        self.page.evaluate("name => { location.hash = '#' + name; }", name)
        self.page.wait_for_selector(f"#view-{name}:not([hidden])")
        self.page.wait_for_function("name => window.DC && DC.state.view === name && !DC.loading.current", arg=name, timeout=60_000)
        time.sleep(settle)  # what the load drew is painted

    def problems(self) -> list[str]:
        return self.console + self.page.evaluate("window.__dclab.errors")

    def shot(self, name: str) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / f"{name}.png"
        self.page.screenshot(path=str(path), full_page=True)
        return path

    def close(self) -> None:
        self.context.close()
        self.browser.close()


# ------------------------------------------------------------------------------------------------ checks
def check_flow(b: Browser) -> str:
    page = b.page
    page.fill("#home-task", "Rank telecom customers by their risk of churn next month so the retention team can call them.")
    page.click("#home-start")
    page.wait_for_selector("#home-chat:not([hidden])")
    page.click("#home-connect")
    page.set_input_files("#home-file", str(TELCO))
    page.wait_for_selector("#home-thread [data-asset-state].ok", timeout=180_000)  # read, cleaned and described
    answered = 0
    for _ in range(6):  # the chat's questions, answered by clicking an option, until it asks none
        try:
            page.wait_for_selector(".ask-card:not([data-done]) .ask-option", timeout=15_000)
        except Exception:  # noqa: BLE001 — no open question left
            break
        page.click(".ask-card:not([data-done]) .ask-option >> nth=0")
        answered += 1
        page.wait_for_function("n => document.querySelectorAll('.ask-card[data-done] .ask-option[aria-pressed=\"true\"]').length >= n", arg=answered, timeout=60_000)
    check(answered >= 1, "the chat asked no question with options")

    check(page.locator("#home-create").is_visible(), "no Build this chat as a project button in the conversation, or no New project from chat button on Home")  # 25.13: the one-step build
    check(page.locator("[data-new-chat]").is_visible(), "no New project from chat button on Home")
    page.click("#side-toggle")  # 25.13: hide the menu for room, remembered, and bring it back from the top bar
    check(not page.locator("#side").is_visible() and page.locator("#menu-btn").is_visible(), "the menu did not hide")
    page.click("#menu-btn")
    check(page.locator("#side").is_visible(), "the menu did not come back")
    page.click("#home-chat [data-wizard]")  # the step-by-step path, which this test walks
    page.wait_for_selector("#view-new:not([hidden]) .panel[data-new-step='2']:not([hidden])", timeout=30_000)
    page.wait_for_selector("#wz-data-next:not([disabled])", timeout=60_000)
    page.click("#wz-data-next")
    page.wait_for_selector(".panel[data-new-step='3']:not([hidden]) #wz-sheet [data-forbid], .panel[data-new-step='3']:not([hidden]) #wz-moment", timeout=60_000)
    page.wait_for_function("() => !(document.querySelector('#wz-sol-status') || {}).textContent", timeout=60_000)
    moment = page.locator("#wz-moment")
    if len(moment.input_value().strip()) < 12:
        moment.fill("At the start of each month, from the customer's account as it stands that day.")
    check(page.locator("#wz-dc-fields").is_hidden(), "the decision's fields show before it is asked for")
    page.check("#wz-dc-on")  # 34.4: the decision on top of the model, in the wizard's own fields
    page.fill("#wz-dc-what", "call the customer")
    page.fill("#wz-dc-cost", "5")
    page.fill("#wz-dc-miss", "300")
    page.evaluate("() => { window.__redrawn = false; new MutationObserver(() => { window.__redrawn = true; }).observe(document.querySelector('#wz-sheet'), { childList: true }); }")
    moment.fill(moment.input_value().strip() + " Known that day.")  # a new moment audits the columns again and redraws the sheet
    moment.press("Tab")
    page.wait_for_function("() => window.__redrawn && !(document.querySelector('#wz-sol-status') || {}).textContent", timeout=90_000)
    check(page.is_checked("#wz-dc-on") and page.input_value("#wz-dc-what") == "call the customer" and page.input_value("#wz-dc-miss") == "300",
          "the decision typed in the sheet was lost when the columns were audited again")
    page.click("#wz-accept")
    page.wait_for_selector(".panel[data-new-step='4']:not([hidden])", timeout=30_000)
    page.click("#wz-settings-next")  # the defaults: quick rows, the folds and budget as they are
    page.wait_for_selector(".panel[data-new-step='5']:not([hidden]) #wz-go-notebook:not([disabled])", timeout=30_000)
    page.click("#wz-go-notebook")

    page.wait_for_selector("#view-project:not([hidden]) #proj-actions [data-run-all]", timeout=60_000)
    project_id = page.evaluate("DC.state.project").removeprefix("p:")
    page.click("#proj-actions [data-run-all]")
    deadline, clicks = time.time() + 600, 1
    while time.time() < deadline:  # the page polls while it runs; when it stops before a stage, a person clicks the next one
        project = b.api(f"/projects/{project_id}")
        stages = project["stages"]
        if all((stages.get(s) or {}).get("status") in ("completed", "approved", "failed") for s in STAGES):
            break
        nxt = page.locator("#proj-actions [data-run-stage]:not([disabled]), #proj-actions [data-run-all]:not([disabled])")
        if not project.get("running") and nxt.count():
            nxt.first.click()
            clicks += 1
        time.sleep(3)
    states = {s: (stages.get(s) or {}).get("status") for s in STAGES}
    check(all(v in ("completed", "approved") for v in states.values()), f"stages did not all complete: {states}")
    page.wait_for_function("() => document.querySelectorAll('#graph-box g.node.done').length >= 5", timeout=60_000)

    b.view("brief", settle=2)
    page.wait_for_selector("#brief-export-real:not([hidden])", timeout=30_000)
    href = page.get_attribute("#brief-export-real", "href")
    check(href == f"/api/projects/{project_id}/export/report", f"the brief's export link is {href}")
    final = b.api(f"/projects/{project_id}")["records"]["final"]
    metric, record = final["primary_metric"], final["evidence"]  # the metric the brief shows (a JSONB record keeps no key order)
    shown = number(page.text_content("#brief-stats .stat:first-child .v"))
    check(abs(shown - record["holdout_metrics"][metric]) < 0.006, f"the brief shows {shown}, the record says {record['holdout_metrics'][metric]}")
    report = b.context.request.get(b.base + href)
    check(report.ok and b"holdout" in report.body().lower(), "the report does not download")
    run_id = (final.get("run") or {}).get("run")  # 19.5: the headline number traced to the run that made it
    page.click("#brief-stats [data-trace]")
    page.wait_for_function("id => document.body.innerText.includes('Where this number comes from') && (!id || document.body.innerText.includes(id))",
                           arg=run_id, timeout=15_000)
    page.keyboard.press("Escape")
    b.view("decision", settle=1)  # 34.4: the page says what the project's decision is, or how to state one
    page.wait_for_selector("#dc-box .empty, #dc-box .panel", timeout=30_000)
    stated = b.api(f"/projects/{project_id}/decision")
    shown_decision = page.text_content("#dc-box") or ""
    check(stated["status"] == "chosen" and stated["stated"]["what"] == "call the customer",
          f"the decision stated in the wizard was not chosen: {stated['status']}")
    check(bool(stated.get("sentence")) and stated["sentence"][:40] in shown_decision, "the decision page does not show the decision's sentence")
    return (f"{answered} question(s) answered by clicking, five wizard steps, five stages in {clicks} click(s) on the Workflow page, "
            f"brief {metric} {shown} on {record.get('holdout_rows')} rows, decision {stated['status']}")


FRAUD = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review 200 a day.")
TEAM_ANSWERS = (("moment|when", "After delivery, in the daily audit of the day's rides."), ("unit|one row|entity|who", "One ride"),
                ("label|case|known", "A file lists the known cases"), ("outcome|predict|target|column", "The confirmed fraud cases"),
                ("action|cost|wrong|happens", "The audit team reviews the flagged rides; a missed fraud costs money"),
                ("budget|how many|review", "About 200 a day"))


def _team_answer(text: str) -> str:
    low = (text or "").lower()
    return next((a for pattern, a in TEAM_ANSWERS if re.search(pattern, low)), "")


def check_team(b: Browser) -> str:
    """28.8: the Team panel on Home: a question answered in its box, the map confirmed, the data given when the walk waits for it,
    and the build approved, all on the page; the walk's own record says it completed and the draft is a project."""
    from dclab_core.replicas import delivery_fraud

    page = b.page
    folder = Path(tempfile.mkdtemp(prefix="dclab-browser-fraud-"))
    files = delivery_fraud.save(delivery_fraud.generate(0), folder)
    page.fill("#home-task", FRAUD)
    page.click("#home-start")
    page.wait_for_selector("#home-team:not([hidden])")
    seen, uploaded = [], False
    page.click("#home-team-go")
    for _ in range(40):
        page.wait_for_function("() => { const t = (document.querySelector('#home-team-board') || {}).textContent || ''; "
                               "return t && !/is working|Building through the team/.test(t); }", timeout=180_000)
        board = page.locator("#home-team-board")
        if page.locator("#home-team-answer").count():
            text = board.inner_text()
            page.fill("#home-team-answer", _team_answer(text))
            page.click("#home-team-answer-go")
            seen.append("question")
            continue
        approve = page.locator("#home-team-board [data-team-approve='1']")
        if approve.count():
            kind = page.locator("#home-team-board [data-team-kind]").first.get_attribute("data-team-kind") or ""
            gate = page.locator("#home-team-board [data-team-gate]").first.get_attribute("data-team-gate") or ""
            seen.append(f"{kind}:{gate}" if gate else kind)
            approve.first.click()
            continue
        text = board.inner_text().lower()
        if "completed" in text:
            break
        if not uploaded:  # the walk waits for the data: the person gives the rides and the known cases, then walks on
            page.click("#home-connect")
            for path in (files["rides"], files["sample"]):
                page.set_input_files("#home-file", str(path))
                page.wait_for_function("n => document.querySelectorAll('#home-thread [data-asset-state].ok').length >= n",
                                       arg=1 if path == files["rides"] else 2, timeout=180_000)
            uploaded = True
            page.click("#home-team-go")
            continue
        raise Failed(f"the walk stopped without a pause: {board.inner_text()[:300]}")
    kinds = {x.split(":")[0] for x in seen}
    check("question" in kinds, f"no question was asked on the panel: {seen}")
    check("confirmation" in kinds, f"no confirmation was asked on the panel: {seen}")
    check("approval:build" in seen, f"no build was approved on the panel: {seen}")
    drafts = b.api("/drafts")
    built = [d for d in (drafts.get("drafts") if isinstance(drafts, dict) else drafts) or [] if d.get("project_id")]
    check(bool(built), "the walk's build made no project")
    return f"{seen.count('question')} question(s), {len([x for x in seen if x.startswith('confirmation')])} confirmation(s), the build approved on the Team panel"


def check_stats(b: Browser) -> str:
    page = b.page
    b.view("home", settle=2)
    ws = b.api("/workspace")
    cards = page.locator("#view-home .stats[data-f='home.kpis'] .stat .v")
    if ws["projects"]:  # with a project, Home shows the one the green button names (not the workspace's sum)
        here = page.evaluate("() => DC.currentProject.id()")
        doc = b.api(f"/projects/{here}")
        moves = doc.get("transitions") or []
        stages = [k for k in ("data", "leakage", "features", "models", "final") if (doc.get("stages") or {}).get(k, {}) and doc["stages"][k].get("status") in ("completed", "approved")]
        want = {0: len((doc.get("solution") or {}).get("forbidden") or []), 2: sum(1 for m in moves if m.get("status") == "blocked"), 3: len(stages)}
        for i, value in want.items():
            shown = number(cards.nth(i).text_content())
            check(shown == value, f"Home card {i + 1} shows {shown}, the selected project's is {value}")
        uses = int(doc.get("holdout_uses") or 0)
        shown = (cards.nth(1).text_content() or "").strip()
        check(shown == (f"{uses}×" if uses else "sealed"), f"Home's holdout card shows {shown!r}, the selected project's holdout was used {uses} time(s)")
    else:
        for i, key in ((0, "forbidden"), (1, "holdouts_once"), (2, "blocked_moves"), (3, "drafts")):
            shown = number(cards.nth(i).text_content())
            check(shown == ws["stats"][key], f"Home card {i + 1} shows {shown}, the API's {key} is {ws['stats'][key]}")
        holdouts = re.findall(r"\d+", cards.nth(1).text_content())
        check(len(holdouts) >= 2 and int(holdouts[1]) == ws["stats"]["holdout_projects"], f"Home's holdout card shows {holdouts}, the API's projects with a holdout {ws['stats']['holdout_projects']}")
    check(number(page.text_content(".ptab[data-ptab='projects'] .n")) == len(ws["projects"]), "Home's project count")

    b.view("compute", settle=2)
    ops = b.api("/ops/jobs")
    cards = page.locator("#ops-stats .stat .v")
    check(number(cards.nth(0).text_content()) == ops["totals"]["running"], "Compute's running count")
    check(number(page.text_content("#jobs-pill")) == ops["totals"]["total"], "Compute's job count")

    b.view("integrations", settle=2)
    integ = b.api("/platform/integrations")
    cards = page.locator("#int-stats .stat .v")
    check(number(cards.nth(0).text_content()) == integ["mcp"]["count"], "Integrations' MCP tool count")
    check(number(cards.nth(2).text_content()) == len(integ["rest"]), "Integrations' REST route count")

    b.view("packs", settle=2)
    packs = [p for p in b.api("/packs") if not p.get("page_only")]
    check(number(page.text_content("#pk-n")) == len(packs), "Packs' count")
    return f"Home, Compute, Integrations and Packs match the API ({len(ws['projects'])} project(s), {ops['totals']['total']} job(s))"


def check_decision(b: Browser) -> str:
    """34.4: the Decision page of a project whose decision was said in the chat and chosen by the final stage."""
    from product_e2e import Api, churn_table, solution_from

    api = Api(b.base)
    table, rows = churn_table()
    draft_id = api.draft("Rank telecom customers by their risk of leaving next month so the retention team can call them.")
    api.call("PUT", f"/drafts/{draft_id}/data?filename=telco.csv", raw=table)
    api.asset_ready(draft_id)
    draft = api.answer(draft_id, ["Churn", "At the start of each month, before the retention call.", "A call costs 5 euro; a lost customer costs 300."])
    proposal = api.call("POST", f"/drafts/{draft_id}/solution/proposal")
    api.call("PUT", f"/drafts/{draft_id}/solution", solution_from(proposal, draft["understanding"]["prediction_moment"]))
    api.call("PUT", f"/drafts/{draft_id}/settings", {"quick": True, "max_rows": rows})
    project = api.run_all(api.call("POST", f"/drafts/{draft_id}/build")["id"])
    stated = api.call("GET", f"/projects/{project['id']}/decision")
    check(stated["status"] == "chosen" and stated.get("sentence"), f"the decision said in the chat was not chosen: {stated['status']}")
    page = b.page
    page.reload()  # the page was drawn before the project was built through the API
    b.view("home", settle=2)  # chosen with the green switcher, as a person would
    page.wait_for_selector("#view-home .proj-switch", state="visible", timeout=15_000)
    page.click("#view-home .proj-switch")
    page.wait_for_selector("#view-home .ps-opt", timeout=15_000)
    page.locator("#view-home .ps-opt").filter(has_text=project["name"]).first.click()
    page.wait_for_function("n => document.querySelector('#view-home .ps-name').textContent === n", arg=project["name"], timeout=15_000)
    b.view("decision", settle=1)
    page.wait_for_selector("#dc-curve svg", timeout=30_000)
    shown = page.text_content("#dc-box") or ""
    check(stated["sentence"][:40] in shown, "the decision page does not show the decision's sentence")
    page.fill("#dc-in-cost_per_action", "50")
    page.click("#dc-ask")
    page.wait_for_selector("#dc-answer table", timeout=60_000)
    shown_share = number(page.text_content("#dc-answer tbody tr:nth-child(2) td:nth-child(2)") or "") / 100
    expected = api.call("POST", f"/projects/{project['id']}/decision/what-if", {"changes": {"cost_per_action": 50}})["after"]["share"]
    check(abs(shown_share - expected) < 0.0006, f"the page's what-if acts on {shown_share:.4f} of rows, the API on {expected:.4f}")
    check(not b.problems(), f"the decision page: {b.problems()[:3]}")
    return f"chosen on {stated['chosen']['share']:.1%} of rows; at 50 a call the page and the API both act on {expected:.1%}"


def check_poll(b: Browser) -> str:
    """The Workflow page follows a run across the gap between two of its stages (one just ended, the next still queued,
    none named running) and past a failed poll, until it is done (a page that stopped there showed 1 of 5 stages done
    after all five had finished, on a busy machine)."""
    page = b.page
    projects = b.api("/workspace")["projects"]
    check(bool(projects), "no project to open")
    pid = projects[0]["id"]
    real = b.api(f"/projects/{pid}")
    check(all((real["stages"].get(st) or {}).get("status") in ("completed", "approved") for st in STAGES), "the first project has not run")
    gap = json.loads(json.dumps(real))
    gap["running"] = None  # one stage ended, the next not yet started: the moment a page used to stop
    gap["stages"]["final"] = {"status": "queued"}
    calls = {"n": 0}

    def answer(route):
        calls["n"] += 1
        if calls["n"] == 1:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(gap))
        elif calls["n"] == 2:
            route.fulfill(status=503, content_type="application/json", body='{"detail": "busy"}')  # a poll that fails
        else:
            route.continue_()

    page.evaluate("id => { try { localStorage.setItem('dclab-project', id); } catch (e) {} }", pid)
    page.reload()
    page.route(f"**/api/projects/{pid}", answer)
    try:
        b.view("project", settle=0.5)
        deadline = time.time() + 30
        while calls["n"] < 3 and time.time() < deadline:  # the gap, the failed poll, then the real project
            page.wait_for_timeout(500)
        page.wait_for_function("() => document.querySelectorAll('#graph-box g.node.done').length >= 5", timeout=30_000)
    finally:
        page.unroute(f"**/api/projects/{pid}")
    check(calls["n"] >= 3, f"the page asked {calls['n']} time(s): it stopped following the run")
    left = [m for m in b.problems() if "503" not in m]  # the failed poll this check made on purpose
    check(not left, f"the Workflow page: {left[:3]}")
    return f"followed the run across a gap and a failed poll ({calls['n']} requests) until all five stages showed done"


def check_solution_decision(b: Browser) -> str:
    """34.4: the Solution page's editor shows a project's stated decision in the wizard's fields; a changed amount and an
    unticked box are what the server then holds."""
    page = b.page

    def save():  # the drawer's save; with stage results to clear, the page asks first
        page.click("#drawer [data-es-save]")
        try:
            page.wait_for_selector("#modal-confirm", state="visible", timeout=3_000)
            page.click("#modal-confirm")
        except Exception:  # noqa: BLE001 - no stage result to clear: no question asked
            pass

    projects = b.api("/workspace")["projects"]
    stated = [p for p in projects if ((b.api(f"/projects/{p['id']}").get("solution") or {}).get("decision") or {}).get("action") == "act"]
    check(bool(stated), "no project with a stated decision to edit")
    pid = stated[0]["id"]
    page.evaluate("id => { try { localStorage.setItem('dclab-project', id); } catch (e) {} }", pid)
    page.reload()
    b.view("solution", settle=1)
    page.click("[data-edit-solution]")
    page.wait_for_selector("#drawer #es-dc-on", timeout=30_000)
    check(page.is_checked("#drawer #es-dc-on") and page.locator("#drawer #es-dc-fields").is_visible(), "the editor does not show the stated decision")
    page.fill("#drawer #es-dc-cost", "7")
    save()
    deadline = time.time() + 30
    while time.time() < deadline and ((b.api(f"/projects/{pid}")["solution"].get("decision") or {}).get("cost_per_action")) != 7:
        time.sleep(0.5)
    check((b.api(f"/projects/{pid}")["solution"].get("decision") or {}).get("cost_per_action") == 7, "the changed cost did not reach the solution")
    page.click("[data-edit-solution]")
    page.wait_for_selector("#drawer #es-dc-on", timeout=30_000)
    page.uncheck("#drawer #es-dc-on")
    save()
    deadline = time.time() + 30
    while time.time() < deadline and b.api(f"/projects/{pid}")["solution"].get("decision"):
        time.sleep(0.5)
    check(b.api(f"/projects/{pid}")["solution"].get("decision") is None, "unticking did not remove the decision")
    check(not b.problems(), f"the Solution page: {b.problems()[:3]}")
    return "the editor showed the decision; a cost changed to 7 and an untick reached the server"


def check_switch(b: Browser) -> str:
    """The green project switcher: Home names the project every page follows, lists the others, and the sidebar's Project
    label and the project pages follow the choice."""
    page = b.page
    b.view("home", settle=2)
    ws = b.api("/workspace")
    if not ws["projects"]:
        check(page.locator("#view-home .ps-wrap").is_hidden(), "the project switcher shows with no project to switch")
        return "no project: the switcher stays hidden"
    page.wait_for_selector("#view-home .proj-switch", state="visible", timeout=15_000)
    name = page.inner_text("#view-home .proj-switch .ps-name").strip()
    check(name and name != "…", "the green button does not name the project")
    check((page.text_content("#nav .nav-label >> nth=1") or "").strip() == "Project · " + name, "the sidebar's Project label is not the button's project")
    page.click("#view-home .proj-switch")
    page.wait_for_selector("#view-home .ps-opt", timeout=15_000)  # the list is read when the button opens it
    options = page.locator("#view-home .ps-opt")
    check(options.count() == len(ws["projects"]), f"the list shows {options.count()} projects, the workspace has {len(ws['projects'])}")
    check(page.locator("#view-home .ps-opt[aria-selected='true']").count() == 1, "exactly one project is marked as the one in use")
    page.keyboard.press("Escape")
    check(page.locator("#view-home .ps-menu").count() == 0, "Escape did not close the list")
    other = [p for p in ws["projects"] if p["name"] != name]
    if other:
        page.click("#view-home .proj-switch")
        page.click(f"#view-home .ps-opt:has-text('{other[0]['name']}')")
        page.wait_for_function("n => document.querySelector('#view-home .ps-name').textContent === n", arg=other[0]["name"], timeout=15_000)
        check((page.text_content("#nav .nav-label >> nth=1") or "").strip() == "Project · " + other[0]["name"], "the sidebar did not follow the chosen project")
        b.view("models", settle=2)
        check(other[0]["name"].lower() in (page.text_content("#view-models .eyebrow") or "").lower(), "the Models page did not follow the chosen project")
        b.view("home", settle=1)
        page.click("#view-home .proj-switch")  # back to the first, so the next checks read what they did
        page.click(f"#view-home .ps-opt:has-text('{name}')")
        page.wait_for_function("n => document.querySelector('#view-home .ps-name').textContent === n", arg=name, timeout=15_000)
    return f"the green button, its list of {len(ws['projects'])}, the sidebar and Models follow the chosen project"


def check_pages(b: Browser) -> str:
    b.page.wait_for_function("() => window.DC && !DC.loading.current", timeout=60_000)
    check(not b.problems(), f"the first load: {b.problems()[:3]}")  # the app's start and Home's first render count too
    seen = []
    for name in VIEWS:
        before = len(b.problems())
        b.view(name)
        new = b.problems()[before:]
        check(not new, f"#{name}: {new[:3]}")
        warn = b.page.locator(f"#view-{name} > [data-view-state]:not([hidden])")
        check(warn.count() == 0, f"#{name} says: {warn.first.text_content() if warn.count() else ''}")
        seen.append(name)
    time.sleep(2)
    check(not b.problems(), f"after the last page: {b.problems()[:3]}")  # an error that arrives late on the last page
    return f"{len(seen)} pages without a console error, an uncaught error, a CSP violation or a failed load"


def check_ui_events(b: Browser) -> str:
    """23.2: the page views and the actions the browser made are in the record, under the browser's own visit."""
    b.view("home")
    b.page.evaluate("() => { const a = document.querySelector('a[href=\"#evidence\"]'); if (a) a.click(); }")  # the Evidence library (there is no #learn page)
    b.view("evidence")
    b.view("new")  # the wizard
    b.page.evaluate("() => DC.ui.flush()")
    time.sleep(1.0)
    latest = b.api("/experience/sessions")["items"]  # the latest visits, newest first
    check(bool(latest), "no session in the record")
    visit = latest[0]["visit"]
    text = json.dumps(b.api(f"/experience/sessions/{visit}"))
    check('"ui.page_viewed"' in text and '"evidence"' in text and '"new"' in text, "the browser's page views are not in its visit's timeline")
    check('"ui.action"' in text, "the click on the page is not in its visit's timeline")
    return f"page views and actions of visit {visit} are in the record"


def check_mobile(b: Browser) -> str:
    wide = []
    for name in VIEWS:
        b.view(name, settle=0.8)
        over = b.page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        if over > 1:
            culprit = b.page.evaluate(CULPRIT)
            wide.append(f"#{name} (+{over}px, {culprit})")
    check(not wide, "wider than 375 px: " + "; ".join(wide))
    return f"{len(VIEWS)} pages fit 375 px without sideways scroll"


def check_memory(b: Browser) -> str:
    """35.7: a project's Memory tab lists what its agents remember; a correction made there replaces the line, at 1280 and 375 px."""
    from product_e2e import Api

    api = Api(b.base)
    project = api.call("POST", "/projects", {"name": f"Memory check {int(time.time())}"})
    said = api.call("POST", "/memory/proposals", {"message": "From now on, answer in short sentences.", "source": "browser:memory"})
    if isinstance(said, dict) and said.get("status") == "draft":  # on a server that already holds it, the line is there already
        api.call("POST", f"/memory/{said['id']}/review", {"action": "accept"})
    page = b.page
    page.reload()  # the page was drawn before the project was made through the API
    b.view("home", settle=2)
    page.wait_for_selector("#view-home .proj-switch", state="visible", timeout=15_000)
    page.click("#view-home .proj-switch")
    page.wait_for_selector("#view-home .ps-opt", timeout=15_000)
    page.locator("#view-home .ps-opt").filter(has_text=project["name"]).first.click()
    page.wait_for_function("n => document.querySelector('#view-home .ps-name').textContent === n", arg=project["name"], timeout=15_000)
    b.view("project", settle=1)
    page.click("#view-project .ptab[data-ptab='memory']")
    row = page.locator("#proj-memory tr").filter(has_text="short sentences").first
    row.wait_for(timeout=30_000)
    new = f"Answer in short, plain sentences ({int(time.time())})."
    page.evaluate("t => { window.prompt = () => t; }", new)
    row.locator("[data-mem-correct]").click()
    page.wait_for_function("t => (document.querySelector('#proj-memory') || {}).textContent.includes(t)", arg=new, timeout=30_000)
    seen = api.call("GET", f"/memory?project_id={project['id']}")
    held = [i for i in seen["items"] if i["text"] == new]
    check(bool(held) and held[0]["correction"], "the correction made on the page is not the line the API lists")
    check(not [i for i in seen["items"] if "short sentences" in i["text"]], "the corrected line is still listed beside its correction")
    page.set_viewport_size({"width": 375, "height": 812})
    time.sleep(0.8)
    over = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check(over <= 1, f"the Memory tab is wider than 375 px (+{over}px, {page.evaluate(CULPRIT)})")
    check(not b.problems(), f"the Memory tab: {b.problems()[:3]}")
    return "the Memory tab lists the line, a correction on the page replaces it, and it fits 375 px"


WORKSPACE: Path | None = None  # the script's own server's workspace (None with --base): the review check builds a project there


def check_review(b: Browser) -> str:
    """20.7: an anomaly project's ranked alerts on its Overview; a person marks one a real case on the page and the API
    keeps it as the project's first reviewed round."""
    if WORKSPACE is None:
        return "skipped: the review check builds its project in the script's own server's workspace (not with --base)"
    from unittest import mock

    from dclab_rnd import kinds
    from dclab_rnd.expansion import runner
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine
    from product_e2e import Api

    store, pid = acceptance.build(kinds.get("anomaly").case(0), WORKSPACE / "projects")
    name = f"Review check {int(time.time())}"
    store.update(pid, name=name)
    with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        engine.run_all(store, pid)
    page = b.page
    page.reload()
    b.view("home", settle=2)
    page.wait_for_selector("#view-home .proj-switch", state="visible", timeout=15_000)
    page.click("#view-home .proj-switch")
    page.wait_for_selector("#view-home .ps-opt", timeout=15_000)
    page.locator("#view-home .ps-opt").filter(has_text=name).first.click()
    page.wait_for_function("n => document.querySelector('#view-home .ps-name').textContent === n", arg=name, timeout=15_000)
    b.view("project", settle=1)
    first = page.locator("#proj-review [data-review-key]").first
    first.wait_for(timeout=30_000)
    key = first.get_attribute("data-review-key")
    first.select_option("confirmed")
    page.click("#proj-review-send")
    api = Api(b.base)
    for _ in range(30):
        rounds = (api.call("GET", f"/projects/{pid}").get("review") or {}).get("rounds") or []
        if rounds:
            break
        time.sleep(1)
    check(bool(rounds) and key in rounds[-1].get("confirmed", []), f"the alert confirmed on the page ({key}) is not the project's reviewed round")
    check(not b.problems(), f"the review panel: {b.problems()[:3]}")
    return f"an anomaly project's alerts are listed on its Overview, and one confirmed on the page is kept as its first reviewed round"


def check_failures(b: Browser) -> str:
    """19.12: the failure candidates on the Admin page's Datasets tab; Accept on the page is kept for the next build."""
    if WORKSPACE is None:
        return "skipped: the failures check writes its candidates in the script's own server's workspace (not with --base)"
    folder = WORKSPACE / "datasets" / "failures" / "benchmark_candidates"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"id": "fc-check-1", "detectors": ["identifier_target"], "flags": [], "problem": "Predict churn", "family": "binary", "review": "pending"}]
    (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    page = b.page
    page.reload()
    b.view("admin", settle=2)
    page.click("#view-admin .ptab[data-ptab='datasets']")
    button = page.locator("#fc-body [data-fc='accept'][data-id='fc-check-1']")
    button.wait_for(timeout=30_000)
    button.click()
    reviews = WORKSPACE / "datasets" / "failures" / "reviews.json"
    for _ in range(30):
        if reviews.exists() and json.loads(reviews.read_text(encoding="utf-8")).get("fc-check-1", {}).get("verdict") == "accept":
            break
        time.sleep(0.5)
    check(reviews.exists() and json.loads(reviews.read_text(encoding="utf-8"))["fc-check-1"]["verdict"] == "accept",
          "the verdict given on the page is not kept for the next build")
    check(not b.problems(), f"the failures panel: {b.problems()[:3]}")
    return "a failure candidate is listed on the Admin page's Datasets tab, and Accept there is kept for the next build"


# the outermost element wider than the screen that is not inside a box that scrolls or clips it
CULPRIT = """() => { const w = document.documentElement.clientWidth;
  const clipped = e => { for (let a = e.parentElement; a && a !== document.body; a = a.parentElement) { const o = getComputedStyle(a).overflowX; if (o !== 'visible') return true; } return false; };
  const el = [...document.querySelectorAll('.view:not([hidden]) *')].find(e => e.offsetParent !== null && e.getBoundingClientRect().right > w + 1 && !clipped(e));
  return el ? el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\\s+/).join('.') : '') + ' "' + (el.textContent || '').trim().slice(0, 40) + '"' : '?'; }"""


CHECKS = {"flow": (check_flow, 1280), "poll": (check_poll, 1280), "decide": (check_decision, 1280), "soledit": (check_solution_decision, 1280), "team": (check_team, 1280), "stats": (check_stats, 1280), "pages": (check_pages, 1280), "switch": (check_switch, 1280), "ui": (check_ui_events, 1280), "memory": (check_memory, 1280), "review": (check_review, 1280), "failures": (check_failures, 1280), "mobile": (check_mobile, 375)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", help="URL of a running server; without it the script starts its own (no model, an empty workspace)")
    parser.add_argument("--only", choices=list(CHECKS), action="append", help="run only this check (repeatable)")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "dclab-browser", help="where a failing check's screenshot goes")
    args = parser.parse_args(argv)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed: pip install -r requirements/browser.txt && python -m playwright install chromium", file=sys.stderr)
        return 2
    from product_e2e import start_server

    process = home = None
    base = args.base
    if not base:
        process, base, home = start_server()
        global WORKSPACE
        WORKSPACE = Path(getattr(home, "name", home))  # start_server gives its TemporaryDirectory
    failed = 0
    try:
        with sync_playwright() as playwright:
            for name in args.only or list(CHECKS):  # the flow first: the stats and pages then show a real project
                run, width = CHECKS[name]
                started, b = time.time(), None
                try:
                    b = Browser(playwright, base, args.out, width=width, height=812 if width < 768 else 900)
                    print(f"PASS  {name:<7} {run(b)}  [{time.time() - started:.0f}s]", flush=True)
                except Exception as error:  # noqa: BLE001 — a Playwright timeout is a failure of the check too
                    failed += 1
                    where = b.shot(name) if b is not None else None
                    print(f"FAIL  {name:<7} {str(error).splitlines()[0][:400]}" + (f"  (screenshot: {where})" if where else ""), flush=True)
                finally:
                    if b is not None:
                        b.close()
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:  # a job still running in the server: never leave it behind
                process.kill()
                process.wait(10)
        if home is not None:
            from product_e2e import keep_workspace

            kept = keep_workspace(home, "browser_e2e")
            if kept is not None:
                print(f"kept: {kept}")
            home.cleanup()
    print("all browser checks passed" if not failed else f"{failed} browser check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

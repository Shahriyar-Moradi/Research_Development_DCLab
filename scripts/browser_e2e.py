#!/usr/bin/env python3
"""Browser tests of the product (package 13.2): a real Chromium drives the real page, the way a person would.

    make browser-test                                   # starts its own server (no model, an empty workspace)
    python scripts/browser_e2e.py --base http://127.0.0.1:8765     # or a running server (CI: the container from 12.1)

Needs Playwright: pip install -r requirements/browser.txt && python -m playwright install chromium.

  flow     one sentence and the Telco CSV uploaded on Home, the chat's questions answered by clicking their options,
           the wizard's five steps with their defaults, a run of every stage from the Workflow page, then the brief
  stats    the numbers on Home, Compute, Integrations and Packs equal the API's (read over HTTP, not through the page)
  pages    all 19 pages at 1280 px: no console error, no uncaught error, no CSP violation, no failed-load warning
  mobile   all 19 pages at 375 px: nothing wider than the screen (no sideways scroll)

No model is used when the script starts its server: the chat asks its standard questions. Exit status 0 when every
check passes; a failing check saves a screenshot under the output folder (--out, default $TMPDIR/dclab-browser).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TELCO = ROOT / "data/project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
VIEWS = ["home", "new", "project", "solution", "notebook", "audit", "models", "reliability", "brief", "intern", "compute",
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
        self.browser = playwright.chromium.launch()
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

    page.click("#home-build")
    page.wait_for_selector("#view-new:not([hidden]) .panel[data-new-step='2']:not([hidden])", timeout=30_000)
    page.wait_for_selector("#wz-data-next:not([disabled])", timeout=60_000)
    page.click("#wz-data-next")
    page.wait_for_selector(".panel[data-new-step='3']:not([hidden]) #wz-sheet [data-forbid], .panel[data-new-step='3']:not([hidden]) #wz-moment", timeout=60_000)
    page.wait_for_function("() => !(document.querySelector('#wz-sol-status') || {}).textContent", timeout=60_000)
    moment = page.locator("#wz-moment")
    if len(moment.input_value().strip()) < 12:
        moment.fill("At the start of each month, from the customer's account as it stands that day.")
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
    return (f"{answered} question(s) answered by clicking, five wizard steps, five stages in {clicks} click(s) on the Workflow page, "
            f"brief {metric} {shown} on {record.get('holdout_rows')} rows")


def check_stats(b: Browser) -> str:
    page = b.page
    b.view("home", settle=2)
    ws = b.api("/workspace")
    cards = page.locator("#view-home .stats[data-f='home.kpis'] .stat .v")
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


def check_mobile(b: Browser) -> str:
    wide = []
    for name in VIEWS:
        b.view(name, settle=0.8)
        over = b.page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        if over > 1:
            culprit = b.page.evaluate(CULPRIT)
            wide.append(f"#{name} (+{over}px, {culprit})")
    check(not wide, "wider than 375 px: " + "; ".join(wide))
    return "19 pages fit 375 px without sideways scroll"


# the outermost element wider than the screen that is not inside a box that scrolls or clips it
CULPRIT = """() => { const w = document.documentElement.clientWidth;
  const clipped = e => { for (let a = e.parentElement; a && a !== document.body; a = a.parentElement) { const o = getComputedStyle(a).overflowX; if (o !== 'visible') return true; } return false; };
  const el = [...document.querySelectorAll('.view:not([hidden]) *')].find(e => e.offsetParent !== null && e.getBoundingClientRect().right > w + 1 && !clipped(e));
  return el ? el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\\s+/).join('.') : '') + ' "' + (el.textContent || '').trim().slice(0, 40) + '"' : '?'; }"""


CHECKS = {"flow": (check_flow, 1280), "stats": (check_stats, 1280), "pages": (check_pages, 1280), "mobile": (check_mobile, 375)}


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
            process.wait(10)
        if home is not None:
            home.cleanup()
    print("all browser checks passed" if not failed else f"{failed} browser check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

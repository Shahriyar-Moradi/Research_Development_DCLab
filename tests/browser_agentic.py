"""Browser integration check. --live starts one real paid research run explicitly."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument("--live", action="store_true")
parser.add_argument("--run-id")
args=parser.parse_args()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",headless=True)
    page=browser.new_page(viewport={"width":1440,"height":1080},device_scale_factor=1)
    errors=[]
    page.on("pageerror",lambda error:errors.append(str(error)))
    page.on("console",lambda msg:errors.append(msg.text) if msg.type=="error" else None)
    page.goto("http://127.0.0.1:8765",wait_until="networkidle")
    page.get_by_text("OpenAI connected",exact=False).wait_for()
    assert page.locator("#datasets input").count()==10
    page.screenshot(path="agent_runs/ui-home.png",full_page=True)
    if args.live:
        page.locator("#goal").fill("Study pre-call bank subscription prediction. Establish a logistic baseline, compare a tree family, then use the measured critique to test one focused feature or tuning change. Examine missing balance/history features and output calibration. Learn reusable procedures and negative results; no deployment claims.")
        page.locator(".settings summary").click()
        page.locator("#experiment-limit").fill("3")
        page.locator("#row-limit").fill("2000")
        page.locator("#repeats").select_option("2")
        page.locator("#minutes").select_option("30")
        with page.expect_response(lambda response:response.url.endswith('/api/runs') and response.request.method=='POST') as response_info:
            page.locator("#start").click()
        response=response_info.value
        assert response.status==201,response.text()
        print(json.dumps({"started_run":response.json()["id"]}),flush=True)
        page.locator("#run-detail .run-heading").wait_for()
    elif args.run_id:
        page.locator(f'[data-run="{args.run_id}"]').click()
        page.locator("#run-detail .run-heading").wait_for()
        page.get_by_text("Signed-log balance replacement",exact=False).wait_for()
    if args.live or args.run_id:
        page.screenshot(path="agent_runs/ui-run.png",full_page=True)
    page.locator('[data-view="knowledge"]').click()
    assert page.locator("#knowledge-view").is_visible()
    page.locator('[data-view="recipes"]').click()
    assert page.locator("#recipes-view").is_visible()
    page.set_viewport_size({"width":390,"height":844})
    page.locator('[data-view="research"]').click()
    page.screenshot(path="agent_runs/ui-mobile.png",full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),"Mobile horizontal overflow"
    browser.close()
    assert not errors,errors
    print("Desktop/mobile navigation and console checks passed",flush=True)

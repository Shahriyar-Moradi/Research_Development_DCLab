"""Build dclab_core as a wheel and use it from a clean Python 3.12 environment with only its dependencies (17.8).

    python scripts/core_wheel_check.py            # what `make core-wheel-check` runs

Steps, all in a temporary folder outside the repository: copy dclab_core (no build output), build the wheel with
``uv build``, make a fresh ``uv venv --python 3.12``, ``uv pip install`` the wheel (its dependencies come from the
package index or uv's cache), then run a smoke call from an unrelated folder with ``python -I``: understand a written
brief, analyse a replica, plan, validate, and map the plan onto the main product's contracts. It prints one JSON
summary; exit 0 when everything passed, 75 when it could not run here (no uv, no Python 3.12, no network for the
dependencies: the reason is printed), 1 when the wheel is broken.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "dclab_core"
SKIP = 75
OFFLINE = re.compile(r"(?i)(failed to fetch|could not connect|dns error|network|connection (refused|reset)|"
                     r"no solution found .* offline|offline mode|timed out|error sending request)")

SMOKE = r'''
import json, re, sys
from importlib import metadata
import dclab_core
from dclab_core.adapters import main_product as mp
from dclab_core.analyse import analyse
from dclab_core.plan import plan, validate
from dclab_core.replicas import delivery_time
from dclab_core.understand import understand

assert dclab_core.__file__.startswith(sys.prefix), dclab_core.__file__  # the installed wheel, not a source folder
brief = understand(
    "Estimate how many minutes a food delivery will take, to show the customer when they place the order. "
    "Each row of orders.csv is an order; actual_minutes is the target. We are judged on MAE.",
    {"orders.csv": list(delivery_time.generate(seed=0).orders.columns)})
orders = delivery_time.generate(seed=0).orders
findings = analyse({"orders.csv": orders}, target="actual_minutes", brief=brief, train_index=orders.index[: int(len(orders) * 0.8)])
solution = plan(brief, findings)
problems = validate(solution, findings)
out = mp.proposals(brief, findings, solution)
for name, mapped in out.items():
    if name != "decision_points":
        assert list(mapped["fields"]) == mp.fields_of(name), name
# 19.1: a main-product row read as an experience event through the installed wheel, and its visit's card
from dclab_core import experience as xp
from dclab_core.adapters import main_product_experience as mpx
cols = mp.recorded()["experience_tables"]["llm_invocations"]
row = {c: None for c in cols} | {"id": "i1", "workspace_id": "w1", "purpose": "assistant.turn", "cost_micros": 1500, "currency": "EUR",
                                 "started_at": "2026-10-13T06:00:00+00:00", "status": "succeeded"}
event = mpx.from_llm_invocation(row)
card = xp.session_card([event.model_copy(update={"ids": event.ids.model_copy(update={"visit": "v1"})})])
assert (event.kind, card.model_requests, card.cost) == ("model.request", 1, {"EUR": 0.0015}), (event.kind, card)
# 20.1: the problem card from the installed wheel: the morning's sentence leaves two families and a blocking question
from dclab_core.card import problem_card
pc = problem_card(understand("I want build anomaly detection"), words="I want build anomaly detection")
assert (pc.ready, pc.in_play) == (False, ["rare_event", "anomaly"]), pc
# 19.5: a main-product experiment read as a run record, and as a v2 event in the session record
from dclab_core.adapters import main_product_lineage as mpl
xcols = mp.recorded()["experience_tables"]["experiments"]
run = mpl.from_experiment({c: None for c in xcols} | {"id": "e1", "workspace_id": "w1", "status": "COMPLETED", "seed": 7, "git_commit": "abc",
                                                     "started_at": "2026-10-14T08:00:00+00:00", "result": {"mae": 3.2}})
ev = mpx.from_run_record(run)
assert (run.outputs.numbers, ev.kind, ev.schema_version) == ({"mae": 3.2}, "run.recorded", "dclab_core.experience_event.v4"), (run, ev)
# 19.6: a Jev answer row read as a decision record, banded, and back
from dclab_core.adapters import main_product_ledger as mpd
scols = mp.recorded()["experience_tables"]["semantic_decision_answers"]
row = {c: None for c in scols} | {"id": "a1", "workspace_id": "w1", "purpose": "column.is_identifier", "decision_point_key": "column.is_identifier",
                                  "release_version": "1", "model_id": "jev", "question_key": "order_id", "answer": {"value": 0.95}, "confidence": 0.95,
                                  "rule_answer": {"value": True}, "agreement": "agree", "level": 0, "policy_outcome": "rule", "value_used": {"value": True},
                                  "created_at": "2026-10-14T08:00:00+00:00"}
rec, _ = mpd.from_semantic_answer(row)
assert (rec.agreement(), mpd.to_semantic_answer(rec, "jev", "1", "d" * 64)["answer"]) == ("agree", {"value": 0.95}), rec
def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()
dists = sorted(f"{norm(d.metadata['Name'])}=={d.version}" for d in metadata.distributions())
# what dclab-core's four requirements pull in, by their own metadata (an extra's requirements are not installed)
closure, todo = set(), ["dclab-core"]
while todo:
    name = todo.pop()
    for req in metadata.requires(name) or []:
        if "extra ==" in req.replace("extra==", "extra =="):
            continue
        dep = norm(re.match(r"[A-Za-z0-9_.\-]+", req).group(0))
        if dep not in closure:
            closure.add(dep)
            try:
                metadata.distribution(dep)
                todo.append(dep)
            except metadata.PackageNotFoundError:  # a requirement for another platform or Python
                pass
print(json.dumps({
    "python": sys.version.split()[0], "dclab_core": dclab_core.__version__, "installed": dists,
    "plan_problems": problems, "split": out["HoldoutPlan"]["fields"]["strategy"],
    "group": out["HoldoutPlan"]["fields"]["group_column"], "metric": out["MetricPlan"]["fields"]["primary_metric"],
    "families": out["ExperimentPlan"]["fields"]["families"], "commit_read": out["ProblemSpec"]["main_product"]["commit"],
    # an optional extra (pdf: reportlab) is not installed by default and is not a direct dependency
    "direct": sorted(norm(re.match(r"[A-Za-z0-9_.\-]+", r).group(0)) for r in metadata.requires("dclab-core") if "extra ==" not in r),
    "unexpected_distributions": sorted({d.split("==")[0] for d in dists} - closure - {"dclab-core"}),
    "experience": event.kind,
    "lineage": run.fingerprint,
    "ledger": rec.point,
}))
'''


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _fail(step: str, result: subprocess.CompletedProcess) -> int:
    text = (result.stdout + result.stderr)[-3000:]
    if OFFLINE.search(text):
        print(json.dumps({"status": "skipped", "reason": f"{step}: no network for the dependencies", "log": text[-600:]}))
        return SKIP
    print(json.dumps({"status": "failed", "step": step, "log": text}))
    return 1


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print(json.dumps({"status": "skipped", "reason": "uv is not installed"}))
        return SKIP
    with tempfile.TemporaryDirectory(prefix="dclab-core-wheel-") as tmp:
        tmp = Path(tmp)
        src = tmp / "src" / "dclab_core"
        shutil.copytree(CORE, src, ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"))
        built = _run([uv, "build", "--wheel", "--out-dir", str(tmp / "dist"), str(src)])
        if built.returncode:
            return _fail("uv build", built)
        wheels = sorted((tmp / "dist").glob("dclab_core-*.whl"))
        if len(wheels) != 1:
            print(json.dumps({"status": "failed", "step": "uv build", "log": f"wheels: {wheels}"}))
            return 1
        venv = _run([uv, "venv", "--python", "3.12", str(tmp / "venv")])
        if venv.returncode:
            text = venv.stdout + venv.stderr
            if "No interpreter found" in text or "not found" in text.lower():
                print(json.dumps({"status": "skipped", "reason": "no Python 3.12 for uv", "log": text[-600:]}))
                return SKIP
            return _fail("uv venv", venv)
        python = tmp / "venv" / "bin" / "python"
        installed = _run([uv, "pip", "install", "--python", str(python), str(wheels[0])])
        if installed.returncode:
            return _fail("uv pip install", installed)
        (tmp / "elsewhere").mkdir()
        smoke = _run([str(python), "-I", "-c", SMOKE], cwd=tmp / "elsewhere")
        if smoke.returncode:
            return _fail("smoke", smoke)
        summary = json.loads(smoke.stdout.strip().splitlines()[-1])
        ok = (summary["direct"] == ["numpy", "pandas", "pydantic", "scikit-learn"] and not summary["unexpected_distributions"]
              and not summary["plan_problems"] and summary["python"].startswith("3.12."))
        summary.update(status="passed" if ok else "failed", wheel=wheels[0].name)
        print(json.dumps(summary))
        return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

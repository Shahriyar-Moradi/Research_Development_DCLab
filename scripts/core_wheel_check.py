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
assert (run.outputs.numbers, ev.kind, ev.schema_version) == ({"mae": 3.2}, "run.recorded", "dclab_core.experience_event.v8"), (run, ev)
# 23.4 to 23.12 from the installed wheel: the fingerprint, an outcome, feedback, the sensitivity scan, a step, the trace, the queries,
# the catalog and the conformance gate (every module the wheel must carry)
from dclab_core import catalog, errors, feedback, outcomes, queries, sensitivity, steps, tracing
from dclab_core.adapters import conformance
assert len(errors.fingerprint("ValueError", [("dclab_core.plan", "plan")])["fingerprint"]) == 12
assert outcomes.judge(0.84, 0.85, True) == "well" and feedback.check({"on": "answer", "ref": "m1", "rating": "helpful"})["kind"] == "thumbs"
assert sensitivity.classify("email", ["a@b.co"] * 4)[0] == "pii" and sensitivity.classify("price", [1, 2, 3])[0] == "none"
with steps.collect() as step_log:
    with steps.step("fit", "fit") as s:
        s.out(value=0.5)
assert len(step_log.steps) == 1 and tracing.parse_traceparent("00-" + "a" * 32 + "-" + "b" * 16 + "-01").trace_id == "a" * 32
assert queries.cost([])["rows"] == [] and len(catalog.build()["event_kinds"]) > 40 and conformance.problems() == [], conformance.problems()
# 24.1: a problem map drawn from words, judged, and mapped to the main product's problem specs
from dclab_core import problem_map
from dclab_core.adapters import main_product_problem_map as mppm
fraud = problem_map.decompose("Find the fake rides our bikers make. We have a few confirmed cases and the team can review 200 a day.")
assert fraud.size == "complex" and len(fraud.parts) == 4 and problem_map.validate(fraud) == [] and len(mppm.problem_specs(fraud)["problem_specs"]) == 4
assert problem_map.decompose("I want to know which subscribers will cancel").size == "single"
# 24.2: the reviewer's questions, a person's edit refused with its reason, and the three roles' prompts shipped in the wheel
from dclab_core import mapping, prompts
assert any(q.code == "unmarked_is_unknown" for q in mapping.review_rules(fraud)) and len(mapping.remove_part(fraud, "part2").parts) == 3
try:
    mapping.merge_parts(fraud, "part1", "part3")
except mapping.MapEditRefused as refused:
    assert "one part, one unit" in str(refused)
else:
    raise AssertionError("a ride and an account were merged into one part")
assert {"decomposer", "map_reviewer", "data_need"} <= set(prompts.names())
# 24.3: the router puts the fraud problem's parts where the plan says, from the installed wheel
from dclab_core import solutions
assert {k: r.chosen for k, r in solutions.routes(fraud).items()} == {"sp1": "anomaly", "sp2": "rare_event", "sp3": "entity_tables", "sp4": "review_budgeted"}
assert "solution_router" in prompts.names() and len(solutions.CATALOGUE) == 16
# 24.4: the solution map of the fraud problem shares one holdout, read once by the composition, from the installed wheel
from dclab_core import solution_map
fraud_solution = solution_map.solution_map(fraud, solutions.routes(fraud))
assert solution_map.validate(fraud_solution, fraud) == [] and [s.output for s in fraud_solution.steps if s.reads_holdout] == ["holdout_score"]
# 19.6: a Jev answer row read as a decision record, banded, and back
from dclab_core.adapters import main_product_ledger as mpd
scols = mp.recorded()["experience_tables"]["semantic_decision_answers"]
row = {c: None for c in scols} | {"id": "a1", "workspace_id": "w1", "purpose": "column.is_identifier", "decision_point_key": "column.is_identifier",
                                  "release_version": "1", "model_id": "jev", "question_key": "order_id", "answer": {"value": 0.95}, "confidence": 0.95,
                                  "rule_answer": {"value": True}, "agreement": "agree", "level": 0, "policy_outcome": "rule", "value_used": {"value": True},
                                  "created_at": "2026-10-14T08:00:00+00:00"}
rec, _ = mpd.from_semantic_answer(row)
assert (rec.agreement(), mpd.to_semantic_answer(rec, "jev", "1", "d" * 64)["answer"]) == ("agree", {"value": 0.95}), rec
# 25.1: the twin: the same Jev row as a twin step (the rule the system arm, Jev the AI-first arm), and the rule's mode from the installed wheel
from dclab_core import twin
from dclab_core.adapters import main_product_twin as mpt
pair = mpt.twin_steps([row], session="wheel")[0]
assert (pair.step, pair.system.value, pair.ai_first.value) == ("decision:column.is_identifier", {"value": True}, {"value": True}), pair
assert twin.mode_of({"judged": 12, "system_right": 6, "ai_right": 11, "checked_right": 11})[0] == "ai"
# 25.3: an expert's record read from an answer, its workflow checked, its decisions on the ledger's points, from the installed wheel
from dclab_core import expert
trace = expert.read({"problem": {"target": "churned"}, "problem_type": {"family": "classification", "task": "binary"},
                     "solution_workflow": [{"id": "S1", "tool": "run_stage", "stage": "final"}, {"id": "S2", "tool": "run_stage", "stage": "leakage"}]})
assert "the holdout is read before the leakage audit" in trace.problems and expert.intents(trace)["target.column"] == {"-": "churned"}
# 25.7, 25.8: the twin's sets and every answerer's scores, from the installed wheel
from dclab_core import datasets as wheel_sets, evaluation
flag = evaluation.score("rules", "feature.leakage_suspect", "flag", "planted", [(True, True), (False, True), (True, False), (False, False)])
assert (flag.accuracy, flag.precision, flag.recall) == (0.5, 0.5, 0.5), flag
assert set(wheel_sets.twin_sets([{"step": json.loads(pair.model_dump_json()), "where": {"draft": "d1"}}])) >= {"twin.steps", "twin.teacher.technical"}
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

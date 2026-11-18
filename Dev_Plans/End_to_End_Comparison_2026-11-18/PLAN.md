# End-to-end comparison: does DCLab build a more trusted model than a plain model with Python? (a plan for a smaller model)

## Context

DCLab's vision: from raw data to **trusted** models. Until now every comparison scored talk and plans. The harness
ablation (`evidence/campaigns/harness_ablation_v1`, last run HAB-060, 2026-11-18) showed that on the product's model
(Qwen 3.8 flash) the lead agent now plans about as well as the same model alone. Nobody has measured the real product,
data in and model out, against the same model given plain Python. The twin runs that did something close used Claude
only (TWN-001 to 003), and Claude may not be used in tests any more (the owner, 2026-11-15).

This plan builds that measurement and runs it **once, small, on development cases**. It does **not** run the held-out
gate: that comes after the owner reads the dev result.

**Three arms, one problem each time, the same model for the two model arms (`qwen/qwen3.8-flash`):**

| Arm | What it is | How it runs (existing code reused) |
|---|---|---|
| `rules` | DCLab with no model: the intern's standard plan, every move checked by the validator | the twin's `workflow` lane (`dclab_rnd/twin/lane.py`, `client=None`) |
| `dclab_agent` | DCLab with Qwen as its agent: the same tools and validator | the twin's `ai_first` lane (`dclab_rnd/twin/__init__.py` `run`) |
| `plain` | Qwen alone with a Python tool on the table, in a locked box (no network, no writes outside its folder, no keys) | **new** (`dclab_rnd/agent_eval/e2e/plain.py`) |

**The test cases are made by code, never written by a model** (so the owner's 2026-11-18 rule on Claude-written data
never touches them). Each case is a seeded replica with planted traps (`dclab_core/replicas/`):
- **ticket routing** (multiclass; target `assigned_team`; leak columns `resolved_by`, `first_response_minutes`);
- **delivery time** (regression; target `actual_minutes`; leak columns `delivered_at`, `courier_rating`).

The person's request is a sentence built by code from the case: the file, the target and the prediction moment.

**Scored by code only (no judge):**
1. **Finished:** the arm delivered a target, a metric, a number and an interval.
2. **Leaks kept out:** none of the case's leak columns is among the arm's features.
3. **Real score:** a reference model (always the same: a gradient boosting model from scikit-learn, fixed seed) is fitted
   on the visible table with the **arm's features**, and scored on a **future set no arm ever sees**. In that set the leak
   columns are blank, as they would be at the prediction moment. So an arm that kept a leak gets a low real score.
4. **Honest:** the arm's claimed number against the real score: the gap, and whether the real score falls inside the
   arm's claimed interval.
5. **Holdout read once** (DCLab arms only; the plain arm cannot be checked): the project's `holdout_uses`.
6. **Numbers nobody gave** (plain arm): numbers in its final report that no tool output contains.
7. Cost and time.

**Limits, said now:** the real score scores the arm's **choices** (which columns, which target, which metric) with
one fixed reference model, not each arm's own model. A plain agent with a stronger model but the same features gets
the same real score. Two kinds, a few seeds: a first look, not a general claim.

**Checked before handing over** (the code of E1, E2, E3 and E7 and their 10 tests, run from a scratch copy: all pass):
- the trap bites on both kinds: delivery time, error 9.1 minutes with honest columns and 23.0 with the leaks (on the future
  set, where the leaks are blank); ticket routing, macro F1 0.41 honest and 0.31 with the leaks. The first try did not bite
  on delivery time (a tree cannot subtract two times), so the reference model is given the differences between time columns;
- the locked box runs Python, stops a write outside its folder, keeps keys out and stops a long run.
The plain arm, DCLab's arms and the command (E5, E6, E9) could not be run before handing over: their tests are new.

**Size:** a new package of about 600 lines, a new prompt file, one new model purpose, one small change in the twin,
about 400 lines of tests, one free run and one live run capped at EUR 0.50.

---

## Rules for whoever executes this (read first; they apply to every task)

1. Work in `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`, branch `main`. Python is `.venv/bin/python`.
2. Do the tasks in order, E1 to E13. Finish one (its tests green) before the next. Commit in E10 and E13 only.
3. **Find a place by its quoted text** (`grep -n "text" file`), never by line number.
4. Run tests only like this:
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_a.py test_b.py
   ```
   It must end with `Ran N tests ...` and then `OK`.
5. **Never run** `make check-commit`, `make check-quick`, `make check-all`, `make rd-check` or the whole suite.
6. **Live model calls only in E11**, only with the command written there, only `qwen/qwen3.8-flash`, capped at EUR 0.50
   for the whole run. **Never a Claude model** (any name containing `claude` or `anthropic`) in any arm.
   Every other command runs with `DCLAB_NO_LIVE_MODELS=1`.
7. A failing new test: fix the code, never the assertion. A failing existing test: stop and report.
8. **Do not touch:** `dclab_core/`, any `results/` folder, any file with `sealed` in its name (never open one), the files
   `cases_dev3.json`, `answer_key_dev3.json`, `cases_nodata_held.json`, `answer_key_nodata_held.json`, `cases_sealed.json`,
   `training_records/` by hand. In `dclab_rnd/prompts/` you add **one** new file (E4) and change nothing else.
9. Git: never `git stash`, never `git add .` / `git add -A`. Stage and commit explicit paths only (zsh needs an array):
   ```bash
   F=(path/one path/two); git add -- "${F[@]}" && git commit -m "message" -- "${F[@]}"
   ```
   Never stage `data/workspace/`, `training_records/STORE_CATALOG.json` or anything under `agent_runs/`. End every commit
   message with the attribution line your session gives you.
10. Comments and messages: plain English, short sentences, no em dashes. A comment says **why**.
11. When a step says "read X to find Y", read only what you need, then write down what you found in your report.

---

## E1. The cases: `dclab_rnd/agent_eval/e2e/cases.py`

Create the folder `dclab_rnd/agent_eval/e2e/` with an `__init__.py`:

```python
"""End-to-end comparison (2026-11-18): DCLab's rules, DCLab with a model as its agent, and the same model alone with a
Python tool, each given one table and one request, scored by code against a future set no arm sees. Plan:
Dev_Plans/End_to_End_Comparison_2026-11-18/PLAN.md."""

CAMPAIGN = "e2e_v1"
MODEL = "qwen/qwen3.8-flash"  # the product's model; never a Claude model (the owner, 2026-11-15)
DEV_SEEDS = (11, 12, 13)
HELD_OUT_SEEDS = (101, 102, 103, 104)  # read once, by the gate, after the owner reads the dev result: never in this plan
KINDS = ("ticket_routing", "delivery_time")
FUTURE_SEED_OFFSET = 1000  # the future set is the same replica drawn with another seed: the same world, new cases
```

Then `cases.py`:

```python
"""A case: a replica made by code (never a model's words), the request built from it, and the future set that scores it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import FUTURE_SEED_OFFSET

ROWS, FUTURE_ROWS = 3000, 1500


@dataclass
class Case:
    id: str
    kind: str
    seed: int
    table: pd.DataFrame  # what every arm sees
    future: pd.DataFrame  # what no arm sees: new cases, the leak columns blank as at the prediction moment
    target: str
    task: str  # "multiclass" or "regression"
    metric: str  # "macro_f1" or "mae"
    moment: str
    leaks: list[str]
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"{self.kind}.csv"

    def request(self) -> str:
        """The person's request, built by code from the case."""
        return (f"I attached {self.filename}. Predict `{self.target}` for new cases. The prediction is made at this moment: "
                f"{self.moment} Build the most accurate model that is honest at that moment, and tell me how well it will do "
                "on new cases, with an interval.")


def _replica(kind: str, seed: int, n: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if kind == "ticket_routing":
        from dclab_core.replicas import ticket_routing

        r = ticket_routing.generate(seed=seed, n=n)
        return r.tickets, r.facts
    if kind == "delivery_time":
        from dclab_core.replicas import delivery_time

        r = delivery_time.generate(seed=seed, n=n)
        return r.orders, r.facts
    raise ValueError(f"no case kind {kind!r}")


SPEC = {
    "ticket_routing": {"target": "assigned_team", "task": "multiclass", "metric": "macro_f1",
                       "moment": "When the ticket arrives, before anyone has worked on it."},
    "delivery_time": {"target": "actual_minutes", "task": "regression", "metric": "mae",
                      "moment": "When the order is placed, before a courier is assigned."},
}


def make(kind: str, seed: int) -> Case:
    spec = SPEC[kind]
    table, facts = _replica(kind, seed, ROWS)
    future, _ = _replica(kind, seed + FUTURE_SEED_OFFSET, FUTURE_ROWS)
    leaks = [c for c in facts.get("leaky_columns") or [] if c in table.columns]
    future = future.copy()
    for column in leaks:  # at the prediction moment these do not exist yet
        future[column] = np.nan if pd.api.types.is_numeric_dtype(future[column]) else None
    return Case(id=f"{kind}-s{seed}", kind=kind, seed=seed, table=table.reset_index(drop=True), future=future.reset_index(drop=True),
                target=spec["target"], task=spec["task"], metric=spec["metric"], moment=spec["moment"], leaks=leaks, facts=facts)


def dev_cases() -> list[Case]:
    from . import DEV_SEEDS, KINDS

    return [make(kind, seed) for seed in DEV_SEEDS for kind in KINDS]
```

Before writing it, check the replica fields: `grep -n "leaky_columns\|def generate\|tickets\|orders" dclab_core/replicas/ticket_routing.py dclab_core/replicas/delivery_time.py`.
If a field name differs (for example the table is not `r.tickets`), use the real one and say so. If the moments in
`dclab_rnd/kinds/multiclass.py` and `dclab_rnd/kinds/regression.py` (`def case(`) differ from `SPEC`, copy theirs.

**Test:** new file `tests/test_e2e_comparison.py` (this file grows in every task):

```python
"""End-to-end comparison (2026-11-18): cases made by code, a future set no arm sees, a locked Python box, a plain agent,
DCLab's arms read the same way, and scores computed by code."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CaseTests(unittest.TestCase):
    def test_a_case_is_made_by_code_and_its_future_has_no_leak(self):
        from dclab_rnd.agent_eval.e2e import cases

        for kind in ("ticket_routing", "delivery_time"):
            case = cases.make(kind, 11)
            self.assertEqual(len(case.leaks), 2, kind)
            self.assertIn(case.target, case.table.columns)
            self.assertIn(case.target, case.future.columns)  # the scorer needs the truth of the future cases
            for column in case.leaks:
                self.assertTrue(case.future[column].isna().all(), column)  # blank, as at the prediction moment
                self.assertFalse(case.table[column].isna().all(), column)  # the trap is in the visible table
            self.assertIn(case.filename, case.request())
            self.assertIn(case.moment, case.request())

    def test_the_same_seed_gives_the_same_case_and_the_future_is_new(self):
        from dclab_rnd.agent_eval.e2e import cases

        a, b = cases.make("delivery_time", 12), cases.make("delivery_time", 12)
        self.assertTrue(a.table.equals(b.table))
        self.assertFalse(a.table.head(50).reset_index(drop=True).equals(a.future.head(50).reset_index(drop=True)))
```

Run: `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_e2e_comparison.py`

## E2. The reference scorer: `dclab_rnd/agent_eval/e2e/reference.py`

```python
"""The real score of an arm's choices: one fixed model fitted on the visible table with the arm's features, scored on the
future set (leaks blank). The same model for every arm, so the score compares choices, not modelling skill."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

SEED = 0
MAX_LEVELS = 50  # a text column with more distinct values than this (free text, an id) is not one-hot encoded: it is left out


def _matrix(frame: pd.DataFrame, features: list[str], columns: Optional[list[str]] = None) -> pd.DataFrame:
    parts = {}
    for name in features:
        col = frame[name]
        if pd.api.types.is_datetime64_any_dtype(col):
            parts[name] = (col - pd.Timestamp("1970-01-01")).dt.total_seconds()  # a time as seconds; a blank stays blank
        elif pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            parts[name] = pd.to_numeric(col, errors="coerce").astype(float)
        else:
            parts[name] = col.astype("string")
    times = [n for n in features if pd.api.types.is_datetime64_any_dtype(frame[n])]
    for i, a in enumerate(times):  # how a person uses two times (delivered minus ordered): a tree cannot subtract by itself
        for b in times[i + 1:]:
            parts[f"{a}-{b}"] = (frame[a] - frame[b]).dt.total_seconds() / 60.0
    X = pd.DataFrame(parts, index=frame.index)
    text = [c for c in X.columns if X[c].dtype == "string"]
    wide = [c for c in text if X[c].nunique(dropna=True) > MAX_LEVELS]
    X = pd.get_dummies(X.drop(columns=wide), columns=[c for c in text if c not in wide], dummy_na=True)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.astype(float)


def real_score(case: Any, features: list[str]) -> Optional[float]:
    """None when the arm named no usable feature (nothing to fit)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import f1_score, mean_absolute_error

    used = [f for f in features if f in case.table.columns and f != case.target]
    if not used:
        return None
    train = case.table.dropna(subset=[case.target])
    future = case.future.dropna(subset=[case.target])
    X = _matrix(train, used)
    Xf = _matrix(future, used, list(X.columns))
    if case.task == "regression":
        model = HistGradientBoostingRegressor(random_state=SEED).fit(X, train[case.target].astype(float))
        return float(mean_absolute_error(future[case.target].astype(float), model.predict(Xf)))
    model = HistGradientBoostingClassifier(random_state=SEED).fit(X, train[case.target].astype(str))
    return float(f1_score(future[case.target].astype(str), model.predict(Xf), average="macro"))
```

**Tests** (add to `tests/test_e2e_comparison.py`):

```python
class ReferenceTests(unittest.TestCase):
    def test_a_leak_looks_good_on_the_table_and_fails_on_the_future(self):
        from dclab_rnd.agent_eval.e2e import cases, reference

        case = cases.make("delivery_time", 11)
        honest = [c for c in case.table.columns if c not in case.leaks and c != case.target and not c.endswith("_id")]
        with_leak = honest + case.leaks
        self.assertLess(reference.real_score(case, honest), reference.real_score(case, with_leak))  # MAE: lower is better
        self.assertIsNone(reference.real_score(case, [case.target]))

    def test_the_score_is_the_same_twice(self):
        from dclab_rnd.agent_eval.e2e import cases, reference

        case = cases.make("ticket_routing", 11)
        features = [c for c in case.table.columns if c not in case.leaks and c != case.target][:5]
        self.assertEqual(reference.real_score(case, features), reference.real_score(case, features))
```

If the first test fails because the leak does **not** make the future score worse, print both scores and stop: it would
mean the trap does not bite on this replica, and the plan's scoring must be rethought by the owner.

## E3. The locked Python box: `dclab_rnd/agent_eval/e2e/sandbox.py`

The plain arm runs code a model wrote. The tables are invented, but the code runs on the owner's Mac, so it runs inside
macOS's `sandbox-exec`: no network, no writes outside its own folder, no keys in its environment, a time limit. If
`sandbox-exec` is missing, nothing runs (fail closed).

```python
"""A locked box for code a model wrote: macOS sandbox-exec with no network and no writes outside the run's folder, a
clean environment (no keys) and a time limit. Fails closed: without sandbox-exec, nothing runs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SANDBOX = "/usr/bin/sandbox-exec"
TIMEOUT = 180
SHOWN = 6000  # characters of output returned to the model


def profile(folder: Path) -> str:
    here = os.path.realpath(folder)
    return ("(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n"
            f'(allow file-write* (subpath "{here}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") (subpath "/dev/fd"))\n')


def run_python(code: str, folder: Path, step: int, timeout: int = TIMEOUT) -> dict[str, Any]:
    folder = Path(folder)
    if not Path(SANDBOX).is_file():
        return {"error": "no locked box on this machine (sandbox-exec is missing): nothing was run"}
    script = folder / f"step_{step:03d}.py"
    script.write_text(code, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(folder), "TMPDIR": str(folder), "MPLCONFIGDIR": str(folder),
           "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}  # no key ever reaches the box
    start = time.monotonic()
    try:
        done = subprocess.run([SANDBOX, "-p", profile(folder), sys.executable, "-I", "-B", script.name], cwd=folder, env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"stopped after {timeout} seconds", "seconds": timeout}
    return {"returncode": done.returncode, "stdout": done.stdout[-SHOWN:], "stderr": done.stderr[-SHOWN:],
            "seconds": round(time.monotonic() - start, 1)}
```

**Tests** (add):

```python
class SandboxTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.agent_eval.e2e import sandbox

        if not Path(sandbox.SANDBOX).is_file():
            self.skipTest("no sandbox-exec on this machine")
        self.box = sandbox
        self.folder = Path(tempfile.mkdtemp())

    def test_code_runs_and_reads_its_table(self):
        (self.folder / "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        out = self.box.run_python("import pandas as pd\nprint(pd.read_csv('data.csv')['a'].sum())", self.folder, 1)
        self.assertEqual((out["returncode"], out["stdout"].strip()), (0, "4"))

    def test_no_write_outside_its_folder(self):
        outside = Path(tempfile.mkdtemp()) / "escaped.txt"
        out = self.box.run_python(f"open({str(outside)!r}, 'w').write('x')", self.folder, 2)
        self.assertNotEqual(out["returncode"], 0)
        self.assertFalse(outside.exists())

    def test_no_key_reaches_the_box(self):
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test-not-real"}):
            out = self.box.run_python("import os\nprint(os.environ.get('OPENROUTER_API_KEY'))", self.folder, 3)
        self.assertEqual(out["stdout"].strip(), "None")

    def test_a_long_run_is_stopped(self):
        out = self.box.run_python("import time\ntime.sleep(5)", self.folder, 4, timeout=1)
        self.assertIn("stopped after", out["error"])
```

If `test_code_runs_and_reads_its_table` fails because Python cannot start inside the box (a write it needs is denied),
read the `stderr`, add the **narrowest** extra allowed path to `profile()` (for example `(subpath "/private/var/folders")`
only if the error names it), and say so in the report. Never add `(allow network*)` and never allow writes to the
repository or the home folder.

## E4. The plain arm's prompt and model purpose

1. Create `dclab_rnd/prompts/e2e_plain_agent.md`:

```markdown
You are a data scientist working alone. A person sends you a request and one table.

You have two tools.
- run_python: runs a Python script in your working folder, where the table is. pandas, numpy and scikit-learn are
  installed. Each call starts fresh: read the table again in each script. Print what you need to see.
- finish: ends the work with your answer.

Do the work a careful data scientist would do: understand what one row is and what the outcome is, find which columns
would not exist yet at the moment of prediction and leave them out, choose a fair way to test, fit a model, and measure
it on data it did not learn from.

When you are done, call finish once with: the target column, the task (multiclass or regression), the metric (macro_f1
for multiclass, mae for regression), the value you expect on new cases, the low and high ends of your interval, the
exact list of columns your final model uses, and a short report for the person. Every number in the report must come
from an output you saw.
```

2. Register a new model purpose `e2e_plain_agent`, the same way as the existing purpose `exam_plain`:
   ```bash
   grep -rn '"exam_plain"' dclab_rnd tests | head -20
   ```
   Add `e2e_plain_agent` next to `exam_plain` in **each** place that lists purposes (at least `dclab_rnd/models/settings.py`
   `PURPOSES` with the same tier, and `dclab_rnd/agents/harness.py` with a description such as
   `"2026-11-18: the end-to-end comparison's plain arm: the model alone with a locked Python box (an evaluation arm, not a DCLab agent)"`).
   Copy `exam_plain`'s shape exactly; change only the name and the description.
3. A new purpose needs a test that its thinking is stored (CLAUDE.md rule 8). Find the test that checks this for
   `exam_plain` (`grep -rln "exam_plain" tests`) and add `e2e_plain_agent` beside it in the same way. If no test names
   `exam_plain`'s thinking, find the test that loops over every purpose (for example
   `test_every_purpose_but_embedding_has_a_stage` in `tests/test_store_sets_llm.py`) and make sure the new purpose passes it.
4. The new prompt changes the prompts' fingerprints. Run, in this order (both are free, no model):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 make agent-eval
   DCLAB_NO_LIVE_MODELS=1 make harness-eval
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_prompt_audit.py test_prompt_gate.py test_harness_gate.py test_agent_harness.py test_store_sets_llm.py
   ```
   `test_prompt_audit.py` checks that no prompt asks a model to write its reasoning: the prompt above does not. If a test
   names another file that must list the new purpose or prompt, add it there the same way and say so. If `make agent-eval`
   or `make harness-eval` fails, stop and report its last 30 lines.

## E5. The plain arm: `dclab_rnd/agent_eval/e2e/plain.py`

```python
"""The plain arm: the model alone with a locked Python box on the table, until it calls finish (or its calls run out)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dclab_rnd import prompts

from . import sandbox

PROMPT = "e2e_plain_agent"
PURPOSE = "e2e_plain_agent"
MAX_CALLS = 24
MAX_TOKENS = 16000
LAST = ("[the test's runner] That was your last tool call. Call finish now with your answer, from what you have seen.")
TOOLS = [
    {"type": "function", "function": {"name": "run_python", "description": "Run a Python script in your working folder (the table is there).",
                                      "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the work with your answer.", "parameters": {
        "type": "object", "required": ["target", "task", "metric", "value", "low", "high", "features", "report"],
        "properties": {"target": {"type": "string"}, "task": {"type": "string"}, "metric": {"type": "string"},
                       "value": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
                       "features": {"type": "array", "items": {"type": "string"}}, "report": {"type": "string"}}}}},
]


def run(case: Any, client: Any, folder: Path) -> dict[str, Any]:
    """The arm's outcome: what it finished with (or None), every tool output it saw, and its calls."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    case.table.to_csv(folder / case.filename, index=False)
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": case.request()}]
    seen: list[str] = []
    calls: list[dict[str, Any]] = []
    finished: Optional[dict[str, Any]] = None
    for n in range(MAX_CALLS):
        last = n == MAX_CALLS - 1
        if last and messages[-1].get("role") == "tool":
            messages.append({"role": "user", "content": LAST})
        offered = [t for t in TOOLS if t["function"]["name"] == "finish"] if last else TOOLS
        out = client.complete(messages, offered, max_tokens=MAX_TOKENS)
        messages.append(out.get("assistant_message") or {"role": "assistant", "content": out.get("content") or ""})
        tool_calls = out.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user", "content": "[the test's runner] Use run_python to work, and finish to answer."})
            continue
        for c in tool_calls:
            args = c.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if c["name"] == "finish":
                finished = dict(args)
                calls.append({"tool": "finish", "ok": True})
                value: dict[str, Any] = {"ok": True}
            elif c["name"] == "run_python":
                value = sandbox.run_python(str(args.get("code") or ""), folder, len(calls) + 1)
                seen.append(str(value.get("stdout") or "") + str(value.get("stderr") or "") + str(value.get("error") or ""))
                calls.append({"tool": "run_python", "ok": value.get("returncode") == 0})
            else:
                value = {"error": f"no tool {c['name']!r}"}
                calls.append({"tool": c["name"], "ok": False})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(value, default=str)})
        if finished is not None:
            break
    return {"arm": "plain", "finished": finished, "seen": seen, "calls": calls}
```

**Test** with a scripted client (add):

```python
class ScriptedClient:
    """Answers with the next scripted reply; records what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def complete(self, messages, tools, max_tokens=None):
        self.sent.append({"messages": list(messages), "tools": tools})
        return self.replies.pop(0)


def call(name, args, n=1):
    return {"content": "", "tool_calls": [{"id": f"c{n}", "name": name, "arguments": args}]}


class PlainArmTests(unittest.TestCase):
    def test_the_plain_arm_runs_its_code_and_returns_its_answer(self):
        from dclab_rnd.agent_eval.e2e import cases, plain, sandbox

        if not Path(sandbox.SANDBOX).is_file():
            self.skipTest("no sandbox-exec on this machine")
        case = cases.make("delivery_time", 11)
        answer = {"target": "actual_minutes", "task": "regression", "metric": "mae", "value": 6.1, "low": 5.8, "high": 6.4,
                  "features": ["distance_km"], "report": "MAE about 6.1"}
        client = ScriptedClient([call("run_python", {"code": f"import pandas as pd\nprint(len(pd.read_csv('{case.filename}')))"}, 1),
                                 call("finish", answer, 2)])
        out = plain.run(case, client, Path(tempfile.mkdtemp()))
        self.assertEqual(out["finished"]["features"], ["distance_km"])
        self.assertEqual(out["seen"][0].strip(), str(len(case.table)))
        self.assertEqual([c["tool"] for c in out["calls"]], ["run_python", "finish"])

    def test_its_last_call_offers_only_finish(self):
        from dclab_rnd.agent_eval.e2e import cases, plain

        case = cases.make("delivery_time", 11)
        replies = [{"content": "thinking aloud", "tool_calls": []}] * plain.MAX_CALLS
        client = ScriptedClient(replies)
        out = plain.run(case, client, Path(tempfile.mkdtemp()))
        self.assertIsNone(out["finished"])
        self.assertEqual([t["function"]["name"] for t in client.sent[-1]["tools"]], ["finish"])
```

If `distance_km` is not a column of the delivery replica, use a real column name and say so.

## E6. Reading DCLab's arms: `dclab_rnd/agent_eval/e2e/dclab.py`

DCLab's two arms come from one twin run (both lanes on the same seeded project). This task writes the parts that do not
call a model: seeding the source project, and reading a finished project into the same outcome shape as the plain arm.

```python
"""DCLab's arms: the twin's workflow lane (the rules) and its ai_first lane (the model as the agent), on a project seeded
from the case's table; each finished project read into the same outcome shape as the plain arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

FEATURE = "feature"  # the role a column the model learns from has in the data stage's column_roles


def seed_source(case: Any, folder: Path) -> str:
    """A workspace folder holding one project with the case's table and the request as its goal; returns the project id."""
    from dclab_rnd.storage import open_stores
    from dclab_rnd.studio import data as sd

    projects = open_stores(Path(folder))[0]
    project = projects.create(case.id, "general", case.request()[:300])
    case.table.to_csv(projects.data_dir(project["id"]) / case.filename, index=False)
    sd.attach_data(projects, project["id"], case.filename, source={"kind": "e2e", "case": case.id}, actor="human")
    return project["id"]


def outcome(arm: str, project: Optional[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same shape as the plain arm's: what the project finished with, read from its own records."""
    final = (records.get("final") or {}).get("evidence") or {}
    roles = ((records.get("data") or {}).get("evidence") or {}).get("column_roles") or {}
    metric = (records.get("final") or {}).get("primary_metric")
    value = (final.get("holdout_metrics") or {}).get(metric) if metric else None
    ci = final.get("holdout_primary_metric_ci") or {}
    if project is None or value is None:
        return {"arm": arm, "finished": None, "holdout_uses": (project or {}).get("holdout_uses")}
    solution = project.get("solution") or {}
    return {"arm": arm, "holdout_uses": project.get("holdout_uses"),
            "finished": {"target": solution.get("target"), "task": solution.get("task"), "metric": metric, "value": float(value),
                         "low": ci.get("low"), "high": ci.get("high"), "features": sorted(c for c, r in roles.items() if r == FEATURE),
                         "report": ""}}
```

Check two names before you rely on them, and fix them if they differ (say so):
- the role name a feature column has: run `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "from dclab_rnd.kinds import regression, acceptance; r = acceptance.engine_run(regression.KIND.case(0)) if hasattr(regression, 'KIND') else None; print(r and sorted(set(r.evidence('data')['column_roles'].values())))"`.
  If that does not work, find how `dclab_rnd/kinds/acceptance.py` `run(` gets the kind object, and print the role values
  of one finished replica run the same way.
- how `holdout_uses` is kept (`grep -n "holdout_uses" dclab_rnd/studio/engine.py | head`) and the metric's name in the
  final record (`primary_metric`).

**Test** (add): a finished replica project read by `outcome`, using the existing no-model runner:

```python
class DclabOutcomeTests(unittest.TestCase):
    def test_a_finished_rules_project_reads_into_the_shared_shape(self):
        from dclab_rnd.agent_eval.e2e import dclab
        from dclab_rnd.kinds import acceptance

        kind = acceptance.KINDS["regression"] if hasattr(acceptance, "KINDS") else None
        if kind is None:
            self.skipTest("adapt: get the regression kind object the way acceptance.run does")
        run = acceptance.engine_run(kind.case(0))
        out = dclab.outcome("rules", {"solution": run.solution, "holdout_uses": 1}, run.records)
        self.assertEqual(out["finished"]["target"], "actual_minutes")
        self.assertEqual(out["finished"]["metric"], "mae")
        self.assertNotIn("delivered_at", out["finished"]["features"])  # the rules keep this leak out (the kind's acceptance)
        self.assertIsNotNone(out["finished"]["low"])
```

Replace the `skipTest` line by the real way to get the regression kind (read `acceptance.run(`), so the test never
skips; the assertions stay as they are.

## E7. The scorer: `dclab_rnd/agent_eval/e2e/score.py`

```python
"""Each arm's outcome scored by code against the case: finished, leaks kept out, the real score of its choices on the
future set, honesty (the claim against the real score), holdout reads, and numbers its report gives that no output did."""

from __future__ import annotations

import re
from typing import Any, Optional

from . import reference

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def unsaid(report: str, seen: list[str]) -> int:
    """Numbers with a decimal point or of three digits and more in the report that no tool output contains."""
    pool = " ".join(seen)
    found = [n for n in NUMBER.findall(report or "") if ("." in n or len(n.lstrip("-")) >= 3)]
    return sum(1 for n in found if n not in pool)


def score(case: Any, out: dict[str, Any]) -> dict[str, Any]:
    f: Optional[dict[str, Any]] = out.get("finished")
    row: dict[str, Any] = {"case": case.id, "arm": out["arm"], "finished": False}
    if not f or f.get("value") is None:
        return row
    features = [str(c) for c in f.get("features") or []]
    real = reference.real_score(case, features)
    low, high, value = f.get("low"), f.get("high"), float(f["value"])
    row.update(finished=True, target_right=f.get("target") == case.target, metric_right=f.get("metric") == case.metric,
               leaks_used=sorted(set(features) & set(case.leaks)), claimed=value, interval=[low, high], real=real)
    row["leaks_kept_out"] = not row["leaks_used"]
    if real is not None and row["metric_right"]:
        row["gap"] = round(value - real, 4)  # claimed minus real: for MAE a negative gap means it promised better than it does
        row["real_inside_interval"] = (low is not None and high is not None and float(low) <= real <= float(high))
    row["holdout_uses"] = out.get("holdout_uses")
    if out["arm"] == "plain":
        row["numbers_nobody_gave"] = unsaid(str(f.get("report") or ""), out.get("seen") or [])
    return row
```

**Tests** (add):

```python
class ScoreTests(unittest.TestCase):
    def test_a_leaky_arm_is_caught_and_its_claim_is_not_honest(self):
        from dclab_rnd.agent_eval.e2e import cases, score

        case = cases.make("delivery_time", 11)
        honest_cols = [c for c in case.table.columns if c not in case.leaks and c != case.target and not c.endswith("_id")]
        leaky = {"arm": "plain", "seen": ["MAE 1.20"], "finished": {"target": case.target, "metric": "mae", "value": 1.2, "low": 1.1,
                 "high": 1.3, "features": honest_cols + case.leaks, "report": "MAE 1.20, about 42.5 percent better"}}
        row = score.score(case, leaky)
        self.assertEqual(row["leaks_used"], sorted(case.leaks))
        self.assertFalse(row["leaks_kept_out"])
        self.assertFalse(row["real_inside_interval"])  # it promised far better than it does on new cases
        self.assertEqual(row["numbers_nobody_gave"], 1)  # 42.5 came from no output

    def test_an_unfinished_arm_scores_unfinished(self):
        from dclab_rnd.agent_eval.e2e import cases, score

        self.assertEqual(score.score(cases.make("ticket_routing", 11), {"arm": "dclab_agent", "finished": None})["finished"], False)
```

Run: `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_e2e_comparison.py`

## E8. The twin records its live runs as the rules ask (one small change)

CLAUDE.md rule 9 asks every live run's gateway to mark its tables as invented so its calls are kept whole with their
thinking. The twin does not do it today.

**File:** `dclab_rnd/twin/__init__.py`, function `def _run(`.
1. Add a keyword parameter at the end of its signature, before `log=print`: `invented_tables: str | None = None,`
2. Find the line `    ai_gateway.usage = Gateway(FileUsage(root / "ai_first" / "model_requests.jsonl"))  # the usage rows every client writes`
   and add right after it:
   ```python
       if invented_tables:  # a runner whose every table code invented (CLAUDE.md rule 9): its calls are kept whole, with their thinking
           from ..models.gateway import mark_invented_tables

           mark_invented_tables(ai_gateway, invented_tables)
           mark_invented_tables(ai_gateway.usage, invented_tables)
   ```
3. Run the twin's tests: `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 $(cd tests && ls test_twin*.py | tr '\n' ' ')`
   They must pass unchanged (the default is None: nothing changes for them).

Whether the marking reaches the exchanges is checked live in E11 (`capture`); if it does not, report it (do not
change the gateway).

## E9. The runner and its command: `dclab_rnd/agent_eval/e2e/__main__.py`

```python
"""python -m dclab_rnd.agent_eval.e2e plan|run|rules.

plan   prints what a run would do and costs nothing.
rules  runs only DCLab's rules arm on the dev cases (no model, free) and scores it.
run    the three arms on the named dev cases, live (--yes and --cap-eur needed; the product's model only, never Claude).
The held-out seeds are never read here: the gate run is a separate, later step the owner approves."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import CAMPAIGN, DEV_SEEDS, KINDS, MODEL

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "evidence" / "campaigns" / CAMPAIGN / "results"


def next_id() -> str:
    RESULTS.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[4:7]) for p in RESULTS.glob("E2E-[0-9][0-9][0-9]*.json")]
    return f"E2E-{max(taken, default=0) + 1:03d}"


def refuse_model(model: str) -> str | None:
    low = model.lower()
    if "claude" in low or "anthropic" in low:
        return "a Claude model is never an arm here (the owner, 2026-11-15)"
    return None


def rules_arm(case: Any, folder: Path) -> dict[str, Any]:
    """DCLab with no model: the twin's workflow lane alone, its standard plan."""
    from dclab_rnd.twin.lane import Lane, NoModels  # if NoModels lives elsewhere, import it from where twin/__init__.py does

    from . import dclab

    pid = dclab.seed_source(case, folder / "source")
    from dclab_rnd.storage import open_stores

    lane = Lane("workflow", folder / "rules", NoModels(), label="the rules")
    from dclab_rnd import context

    with context.using(lane.services):
        lane.seed_project(open_stores(folder / "source")[0], pid, blind=True)
        lane.run_project(case.request(), None, max_steps=40, max_minutes=20)
    from dclab_rnd.studio import engine

    project = lane.projects.get(lane.project_id)
    records = {s: lane.projects.read_stage(lane.project_id, s) for s in engine.STAGE_KEYS}
    return dclab.outcome("rules", project, records)
```

Complete the file yourself with:
- `main(argv)`: subcommands `plan`, `rules`, `run`, with arguments `--cases` (default: every dev case,
  `kind-sSEED` ids such as `delivery_time-s11`), `--model` (default `MODEL`), `--cap-eur`, `--yes`.
- `plan`: print the cases, arms, model and cap as JSON; exit 0.
- `rules`: for each dev case, `rules_arm` into `agent_runs/live_tests/<YYYY-MM-DD>-e2e-<id>/<case>/`, then `score.score`;
  write `RESULTS / f"{id}_rules_dev.json"` with `{"id", "arms": ["rules"], "model": None, "cases": [...], "scores": [...]}`;
  print one line per case.
- `run`: refuse (exit 2, a clear message) when `refuse_model(model)` says so, when `--yes` is missing (print the plan
  instead), when `--cap-eur` is not above 0, when `DCLAB_NO_LIVE_MODELS=1`, or when
  `dclab_rnd.agent_eval.harness_ablation.owner_allows(cap)` returns a reason. Then for each case:
  1. `dclab.seed_source(case, run_dir / case.id / "source")`;
  2. `dclab_rnd.twin.run(f"project:{pid}", home=run_dir / case.id, source_home=run_dir / case.id / "source", task=case.request(),
     model=model, cap_eur=<the cap left>, blind=True, design=False, roles="judged", quick=True, invented_tables="e2e comparison: replicas made by code")`
     (read `_run`'s signature in `dclab_rnd/twin/__init__.py` first and use its real parameter names);
  3. read both lanes' projects from `run_dir / case.id / "twin" / "runs" / <run_id> / {"workflow","ai_first"} / "projects"`
     (a `ProjectStore`, as `Lane.__post_init__` makes it; the project id is the one `seed_project` gave: find it in the
     lane's `projects` folder, there is one project) and turn each into an outcome with `dclab.outcome("rules", ...)` and
     `dclab.outcome("dclab_agent", ...)`;
  4. the plain arm: a gateway made like `harness_ablation.run_gateway(run_dir / case.id / "plain")`, a capped client from
     `harness_ablation.Budget(gateway, cap).client(f"e2e:{case.id}", "e2e_plain_agent", model)` (read `Budget` first; use
     its real call), then `plain.run(case, client, run_dir / case.id / "plain" / "box")`;
  5. score all three with `score.score`, write `RESULTS / f"{id}_dev.json"` (cases, model, cap, spent, scores, and for each
     plain arm its `calls`), and stop the run when the cap is reached (say which cases did not run).
  The run folder is `agent_runs/live_tests/<YYYY-MM-DD>-e2e-<id>/`.

**Tests** (add): the refusals, with no model ever called:

```python
class CommandTests(unittest.TestCase):
    def test_a_claude_model_is_refused_and_a_run_without_yes_only_plans(self):
        from dclab_rnd.agent_eval.e2e import __main__ as cli

        self.assertIsNotNone(cli.refuse_model("anthropic/claude-sonnet-5.5"))
        self.assertIsNone(cli.refuse_model("qwen/qwen3.8-flash"))
        self.assertEqual(cli.main(["run", "--model", "claude-sonnet-5-5", "--yes", "--cap-eur", "0.5"]), 2)
        with mock.patch.dict("os.environ", {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.assertEqual(cli.main(["run", "--yes", "--cap-eur", "0.5"]), 2)
        self.assertEqual(cli.main(["plan"]), 0)
```

## E10. The gate, pinned before any live run, and the first commit

Write `dclab_rnd/agent_eval/e2e/gate_v1.json` **now**, before any model is called. It is the rule the later held-out
run will be judged by; it must not change after anyone sees a live number:

```json
{
 "gate": "e2e_gate_v1",
 "fixed": "2026-11-18, before any live run",
 "set": "the held-out seeds (101 to 104) of both kinds: 8 cases, run once",
 "arms": ["rules", "dclab_agent", "plain"],
 "model": "qwen/qwen3.8-flash",
 "passes_when_all": {
  "dclab_agent_finished": "8 of 8",
  "dclab_agent_leaks_kept_out": "8 of 8",
  "dclab_agent_holdout_reads": "exactly 1 in every finished case",
  "dclab_agent_real_inside_interval": "at least 6 of 8",
  "dclab_agent_real_score_against_plain": "mean paired difference not worse than 0.02 macro_f1 (ticket routing) and not worse than 5 percent of plain's MAE (delivery time)"
 },
 "reported_not_gating": ["the rules arm", "cost", "seconds", "the plain arm's numbers nobody gave"],
 "means": "pass: DCLab with the product's model builds models at least as good and more trustworthy than the same model alone with Python, on these two kinds"
}
```

Pin it: write its SHA-256 into `dclab_rnd/agent_eval/e2e/gate_v1.sha256`:
```bash
shasum -a 256 dclab_rnd/agent_eval/e2e/gate_v1.json | cut -d' ' -f1 > dclab_rnd/agent_eval/e2e/gate_v1.sha256
```
and add a test:

```python
class GateTests(unittest.TestCase):
    def test_the_gate_is_pinned(self):
        import hashlib

        folder = ROOT / "dclab_rnd" / "agent_eval" / "e2e"
        pin = (folder / "gate_v1.sha256").read_text(encoding="utf-8").strip()
        self.assertEqual(hashlib.sha256((folder / "gate_v1.json").read_bytes()).hexdigest(), pin)
```

Run every test of this plan:
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_e2e_comparison.py test_prompt_audit.py test_prompt_gate.py test_harness_gate.py test_agent_harness.py test_store_sets_llm.py
```
Then commit (paths: the whole `dclab_rnd/agent_eval/e2e/` folder's files by name, `dclab_rnd/prompts/e2e_plain_agent.md`,
the files changed in E4 and E8, `tests/test_e2e_comparison.py`, and any test file changed in E4; plus the files
`make agent-eval` / `make harness-eval` changed that `git status --short` shows under `evidence/` or `dclab_rnd/`, by name).
First line: `An end-to-end comparison: DCLab's rules, DCLab with a model, and the model alone with Python, scored by code`.

## E11. The free run, then one small live run

1. The rules arm on all six dev cases (free, no model):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.agent_eval.e2e rules
   ```
   Expected: 6 of 6 finished; leaks kept out 6 of 6 (the kinds' own acceptance demands it). If a case does not finish or
   keeps a leak, write it down; do not change DCLab.
2. The live run: **two cases only** (one per kind), the product's model, EUR 0.50 for the whole run:
   ```bash
   .venv/bin/python -m dclab_rnd.agent_eval.e2e run --cases delivery_time-s11,ticket_routing-s11 --model qwen/qwen3.8-flash --cap-eur 0.5 --yes
   ```
   If it refuses (for example the owner's live switch is off), report the message and stop here.
3. Check that every answered call kept its thinking:
   ```bash
   .venv/bin/python -c "import sys, json; from pathlib import Path; from dclab_rnd.agent_eval.harness_ablation import capture; print(json.dumps(capture(Path(sys.argv[1])), indent=1))" agent_runs/live_tests/<the run folder>
   ```
   `ok` must equal `with_thinking`. If not, report which purposes are listed under `without_thinking`.
4. Archive (CLAUDE.md rule 9):
   ```bash
   .venv/bin/python -m dclab_rnd.archive
   ```
   Write down the snapshot folder it prints.

## E12. A one-page result

Write `evidence/campaigns/e2e_v1/RESULT_dev.md` (plain words, no hand-edited numbers: copy them from the result JSON):
- a table, one row per case and arm: finished, leaks kept out, claimed (interval), real, real inside the interval,
  holdout reads, numbers nobody gave, euros, seconds;
- what each arm did wrong, in one sentence each, from its calls and its outcome;
- the limits from the Context section, and "dev cases only: no claim; the gate in gate_v1.json is for the held-out run".

## E13. The record, the training set, the last commit, the push, the report

1. CLAUDE.md: in the Map's `agent_eval/` text, find `` `analyst/` the problem analyst's measurement `` and put before it:
   `` `e2e/` the end-to-end comparison (2026-11-18): DCLab's rules, DCLab with the product's model and that model alone with a locked Python box, scored by code on a future set no arm sees, `python -m dclab_rnd.agent_eval.e2e plan|rules|run`, the gate `e2e_gate_v1` pinned before any live run; ``
2. Create `research/llm-fine-tuning/experiments/dev_trajectories/records/e2e-comparison-v1.json` in the same shape as
   `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-provider-hold-all.json` (same keys; a
   measurement's `gate` at least 10 characters; allowed `source`, `severity` and `verdict` values as in that file). Put
   in: the request, the design decisions (code-made cases; the future set with blank leaks; one reference model; the
   locked box; the gate pinned before the run), every finding you met with its fix and test, and two measurements (the
   rules run, the live run) with `verdict` `not scored`.
3. `DCLAB_NO_LIVE_MODELS=1 make sft-dev` must end with `N examples from M packages ...`.
4. Final run: the E10 test command again. It must end with `OK`.
5. Commit (paths: `evidence/campaigns/e2e_v1/` files by name, `CLAUDE.md`, the record, the files under
   `research/llm-fine-tuning/experiments/sft/out_dev/` that `git status --short` shows). Never `agent_runs/`. First line:
   `The first end-to-end comparison on development cases: DCLab against the same model with Python`.
6. Push:
   ```bash
   C=$(git rev-parse HEAD); git push origin "${C}:refs/heads/main" "${C}:refs/heads/dev"; git branch -f dev HEAD
   ```
7. Report to the owner in plain words: what was built, the rules run's numbers, the live run's table, what it cost, the
   archive snapshot, every place where you had to adapt the plan (a field name, an import, the box's profile), and the
   two commit ids. Do **not** run the held-out seeds.

---

## Verification (how the owner knows it is done)

1. The E10 test command ends with `OK`; the box's tests show no write outside its folder and no key inside it.
2. `evidence/campaigns/e2e_v1/results/` holds one rules result and one live dev result; `RESULT_dev.md` reads from them.
3. The live run's capture shows every answered call kept its thinking; the archive snapshot is named in the report.
4. `gate_v1.sha256` matches `gate_v1.json`, and both were committed before the live run (check `git log` order).

## Not in this plan

- **The held-out gate run** (8 cases, run once): after the owner reads the dev result.
- More kinds (anomaly, time to event, the fraud replica's join traps), planted leaks from `dclab_core/benchgen.py`, and
  scoring each arm's own model instead of the reference model.
- Any change to DCLab's engine, agent or prompts because of what the dev run shows: that is the next package, after the
  owner decides.

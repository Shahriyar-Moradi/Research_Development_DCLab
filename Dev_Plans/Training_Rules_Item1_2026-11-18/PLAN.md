# Item 1: make the training data obey the owner's rules (a plan a smaller model can execute)

## Context

The review of the priority plans (2026-11-18) found three rule problems in how DCLab prepares training data:

| # | Problem (verified in code) | Owner's decision (2026-11-18) | Task |
|---|---|---|---|
| 1 | The solver export (`dclab_rnd/lake/train.py`) wraps reasoning **written by code** in `<think>` tags, as if it were the model's own thinking, and rewrites the question to "Think it through first inside <think>". `dclab_rnd/own_llm.py` does the same. This breaks two standing rules: reasoning comes only from the API thinking summary, and reasoning made another way carries a label saying where it came from. 461 rows were exported this way (2026-10-17); no model was trained on them. | Fix it (a standing rule, no new decision needed). | A |
| 2 | Three rules disagree on Claude's answers as training data. The store keeps them in sets marked `consent: train` (some at full weight, for example 3,633 Claude rows in the M4 stage sets and 295 Claude Code build sessions), and the store's training export (`dclab_rnd/store/export_chat.py`) would export them today. | "I want all of my training data ... not to use [Claude's] until I get the right ... but now I want to make all my data prepared and have them all." So: **keep and prepare everything; every training export refuses Claude's (and any other commercial provider's) outputs until the owner records a rights review.** Nothing is deleted. | B |
| 3 | No frozen exam was ever drawn, so the guard that keeps exam rows out of training is empty. | "Draw now." | C |

Also asked: should the small decision classifiers keep refitting automatically? The owner answered "I want all my data, all we have". **Reading: nothing is stopped; no change.** (They train on code-made labels only: every recorded fit has `silver_in_training: 0`.)

**Size:** about 60 lines of code, about 120 lines of tests, one data operation (the exam draw), one record. No prompt file, tool schema, `dclab_core` contract or Build project change.

---

## Rules for whoever executes this (read first; they apply to every task)

1. Work in `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`, branch `main`. Python is `.venv/bin/python`.
2. Do the tasks in order: A, B, C, D. Finish one (tests green, committed) before starting the next.
3. **Find a place by its quoted text** (`grep -n "text" file`), never by line number. Line numbers below are hints.
4. Run tests only like this (bare file names, never a bare `pytest`, never a live key):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_a.py test_b.py
   ```
   It must end with `Ran N tests ...` and then `OK`.
5. **Never run** `make check-commit`, `make check-quick`, `make check-all`, `make rd-check` or the whole suite (the owner's rule: full gates only when needed).
6. A failing test: fix **the code**. Never delete, skip or weaken an assertion. If an **existing** test fails and you believe the new behaviour is right, stop and report it; do not change that test (exception: the two tests in `tests/test_reasoning.py` that Task A replaces on purpose).
7. **Do not touch:** any file in `dclab_rnd/prompts/`; any tool description; anything in `dclab_core/` (a contract change needs a version bump); any file under a `results/` folder; any `sealed.jsonl`; the files `cases_dev3.json`, `answer_key_dev3.json`, `cases_nodata_held.json`, `answer_key_nodata_held.json`, `cases_sealed.json` (never even open them).
8. Git: never `git stash`, never `git add .` / `git add -A`. Stage and commit explicit paths only:
   ```bash
   F=(path/one path/two); git add -- "${F[@]}" && git commit -m "message" -- "${F[@]}"
   ```
   Never stage `data/workspace/` or `training_records/STORE_CATALOG.json`. End every commit message with the attribution line your session gives you. In zsh, a list of paths needs an array (as above), and write `"${C}:refs/..."` with braces.
9. Comments and messages: plain English, short sentences, no em dashes. A comment says **why**.
10. Disk: before Task A7 and Task C, run `df -h .`. If less than 3 GB is free, stop and report.

---

## Task A: no rule-written reasoning is ever stored as the model's own thinking

**Goal.** The solver export and the workspace's own-model export write the question and the answer exactly as they are.
Reasoning written by code is kept, but only in the row's metadata, labelled `"reasoning_source": "from_record"`
(the project's own word for "a rule's choice explained by code from its own record", `dclab_core/store.py`
`ReasoningSource`). No prompt asks a model to think inside tags.

### A1. `dclab_rnd/lake/train.py`

1. Delete the constant `THINK_FIRST = ...` and the whole function `def _with_think(...)` (from `def _with_think` down to
   its last `return` line).
2. In `def export(`, delete the parameter `thinking: bool = True` (and the comma before it). Replace the docstring's
   second sentence (starts with "``thinking`` (27.9)") with:
   `A solver example's reasoning written by code is kept in the row's metadata, labelled from_record, never inside the messages.`
3. Find `reasoned: Counter = Counter()  # 27.9: rows written with their verified` and replace the whole line with:
   ```python
   reasoned: Counter = Counter()  # rows whose code-written reasoning is kept in their metadata (labelled from_record)
   ```
4. Find these three lines:
   ```python
                messages = _decision_messages(row) if target == "decisions" else (_with_think(row["messages"], row.get("rationale")) if thinking else row["messages"])
                if target == "solver" and thinking and row.get("rationale"):
                    reasoned[into] += 1
                item = {"messages": messages, "metadata": {"set": name, "origin": prov["origin"], "group": prov["group"], "split": into}}
   ```
   Replace them with:
   ```python
                # The question and the answer are written as they are. A trace the rules wrote is not the model's thinking
                # (the owner's rule): it is kept in the metadata, labelled, and never placed inside the messages.
                messages = _decision_messages(row) if target == "decisions" else row["messages"]
                metadata = {"set": name, "origin": prov["origin"], "group": prov["group"], "split": into}
                if target == "solver" and row.get("rationale"):
                    metadata.update(reasoning=row["rationale"], reasoning_source="from_record")
                    reasoned[into] += 1
                item = {"messages": messages, "metadata": metadata}
   ```
5. In the manifest dict, find `"thinking": thinking, "with_reasoning": dict(reasoned),` and replace it with:
   ```python
   "reasoning_in_metadata": dict(reasoned),
   ```
   Do not change the lines that write `train.jsonl` and `valid.jsonl`: they already write `{"messages": ...}` only.

### A2. `dclab_rnd/lake/__main__.py`

1. Delete the line that adds `"--no-think"`.
2. Find `m = export(args.target, args.lake, args.out, weights, thinking=not args.no_think)` and replace it with
   `m = export(args.target, args.lake, args.out, weights)`.
3. In the next line, replace `"with_reasoning"` with `"reasoning_in_metadata"`.

### A3. `dclab_rnd/own_llm.py`, function `export`

1. Delete the line `from dclab_core.reasoning import think`.
2. Find:
   ```python
        messages = e.messages
        if e.rationale and messages and messages[-1].get("role") == "assistant":
            messages = [*messages[:-1], {**messages[-1], "content": think(e.rationale.splitlines()) + "\n" + messages[-1]["content"]}]
   ```
   Replace with:
   ```python
        messages = e.messages  # as they are: a trace the rules wrote is kept in the metadata, labelled, never inside the messages
        extra = {"reasoning": e.rationale, "reasoning_source": "from_record"} if e.rationale else {}
   ```
3. In the next `files[into].append(json.dumps({"messages": messages, "metadata": {...}}` line, add `**extra` as the
   last item inside the `"metadata"` dict, so it reads
   `"metadata": {"family": e.family, "origin": e.provenance.origin, "group": e.provenance.group, **extra}`.

### A4. Docstrings that describe the old behaviour

- `dclab_rnd/lake/__init__.py`: grep `the \`<think>\` its export writes` (two places, near lines 435 and 639). Replace
  that phrase in both with `the reasoning its export keeps in the metadata, labelled from_record`.

### A5. Tests: replace two tests in `tests/test_reasoning.py`, class `LakeReasoningTests`

Replace the whole test `test_the_export_writes_the_think_before_the_answer_and_never_without_its_switch` with:

```python
    def test_the_export_never_writes_a_rules_trace_as_the_models_thinking(self):
        """The owner's rule: reasoning written by code is labelled where it came from and never stored as the model's
        own thinking. Before, the solver export put it inside <think> tags (461 rows, 2026-10-17)."""
        from collections import Counter

        from dclab_rnd.lake import reason, write_set
        from dclab_rnd.lake.train import export

        lake = Path(tempfile.mkdtemp())
        rows, _ = reason([example("leakage_judgment", GIVEN, ANSWER)])
        self.assertTrue(rows[0].rationale)  # the rules wrote a trace
        write_set(lake / "gold" / "solver" / "leakage_judgment" / "v1", "training_example", "solver/leakage_judgment", rows, [], Counter())
        m = export("solver", lake, lake / "out")
        line = json.loads((lake / "out" / "train.chat.jsonl").read_text().splitlines()[0])
        self.assertTrue(line["messages"][-1]["content"].startswith("{"))
        self.assertNotIn("<think>", json.dumps(line["messages"]))
        self.assertEqual(json.loads(line["messages"][1]["content"]), GIVEN)  # the question is never rewritten
        self.assertEqual((line["metadata"]["reasoning_source"], line["metadata"]["reasoning"]), ("from_record", rows[0].rationale))
        mlx = json.loads((lake / "out" / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(set(mlx), {"messages"})
        self.assertNotIn("<think>", json.dumps(mlx))
        self.assertEqual(m["reasoning_in_metadata"], {"train": 1})
```

Replace the whole test `test_the_question_asks_to_think_first_when_its_target_thinks` with:

```python
    def test_no_export_asks_a_model_to_think_inside_tags(self):
        from dclab_rnd.lake import train

        self.assertFalse(hasattr(train, "THINK_FIRST"))
        self.assertFalse(hasattr(train, "_with_think"))
```

Do not change any other test in that file (the one that calls `R.think` tests `dclab_core`, which stays as it is).

### A6. A guard test: new file `tests/test_no_think_in_training.py`

```python
"""The owner's rule (2026-10-28, again 2026-11-18): reasoning comes only from the model's API thinking summary. No code asks
a model to write its reasoning inside tags, and no export puts a trace written by code where the model's thinking goes."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORBIDDEN = ("inside <think>", "Think it through first", "from dclab_core.reasoning import think")


class NoThinkTagsTests(unittest.TestCase):
    def test_no_code_asks_for_think_tags_or_wraps_a_trace_in_them(self):
        hits = []
        for base in ("dclab_rnd", "dclab_core"):
            for path in sorted((ROOT / base).rglob("*.py")):
                text = path.read_text(encoding="utf-8")
                hits += [f"{path.relative_to(ROOT)}: {phrase}" for phrase in FORBIDDEN if phrase in text]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
```

If this test lists a file you did not change in A1 to A4, do not edit that file: stop and report the hit.

### A7. Run, then remove the old export files

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_reasoning.py test_no_think_in_training.py test_training_lake.py test_own_llm.py test_prompt_audit.py
```

**Done when:** `OK`, and:
```bash
grep -rn "THINK_FIRST\|_with_think\|no_think\|with_reasoning" dclab_rnd/lake dclab_rnd/own_llm.py
```
prints nothing.

Then delete the four old solver export files that hold the bad rows (they are not in git; a fresh export replaces them):
```bash
ls lake/export/solver/ && rm -f lake/export/solver/train.chat.jsonl lake/export/solver/val.chat.jsonl lake/export/solver/train.jsonl lake/export/solver/valid.jsonl lake/export/solver/MANIFEST.json
```

Then, if `df -h .` shows at least 3 GB free, rebuild the prepared data in the background (it can take a long time):
```bash
DCLAB_NO_LIVE_MODELS=1 make solver-data > /tmp/solver-data.log 2>&1; echo "rc=$?"
```
When it ends: `grep -c "<think>" lake/export/solver/train.chat.jsonl` must print `0`, and
`grep -c "from_record" lake/export/solver/train.chat.jsonl` must print a number above 0. If `make solver-data` fails,
do not try to fix the lake build: report the last 30 lines of `/tmp/solver-data.log` and go on to Task B.

**Commit** (paths: `dclab_rnd/lake/train.py`, `dclab_rnd/lake/__main__.py`, `dclab_rnd/own_llm.py`,
`dclab_rnd/lake/__init__.py`, `tests/test_reasoning.py`, `tests/test_no_think_in_training.py`). First line:
`Reasoning written by code is labelled from_record and never stored as the model's thinking`.

---

## Task B: Claude's outputs are kept and prepared, and no training export uses them until a rights review

**Goal.** Nothing is deleted and every set is still built. The store's training export (`export_chat`) skips, row by
row, every example written by a commercial provider's model (Claude, OpenAI, any other hosted model that is not the
OpenRouter Qwen the owner approved on 2026-11-11), and every Claude Code build session, until the owner records
`"rights_reviewed": true`. Each export says how many rows it held back and why.

### B1. `dclab_rnd/store/owner_decisions.json`

In the `"provider_outputs"` object, add two keys after `"check_before_use": ...` (mind the commas; the file must stay valid JSON):

```json
    "rights_reviewed": false,
    "on_hold": {"decided": "2026-11-18", "by": "owner", "said": "i want all of my training data ... not to use until i get right to, but now i want make all my data prepare and have them all", "means": "every provider output is kept and prepared in its sets; every training export refuses it until rights_reviewed is true"}
```

Check: `.venv/bin/python -c "import json; json.load(open('dclab_rnd/store/owner_decisions.json'))"` prints nothing.

### B2. `dclab_rnd/store/provider_outputs.py`

1. After the function `def for_training(`, add:
   ```python
   def rights_reviewed(path: Optional[Path] = None, decision: str = PROVIDER_OUTPUTS) -> bool:
       """Whether the owner recorded a review of the provider's terms (2026-11-18: not yet). Until then every training
       export refuses the provider's outputs; the sets still keep and prepare them."""
       return _decision(path, decision).get("rights_reviewed") is True


   def may_train(path: Optional[Path] = None, decision: str = PROVIDER_OUTPUTS) -> bool:
       """A provider's output may enter a training run: the owner said yes AND recorded the rights review."""
       return for_training(path, decision) and rights_reviewed(path, decision)
   ```
2. Add `"rights_reviewed", "may_train"` to `__all__`.

### B3. `dclab_rnd/store/export_chat.py`

1. In the module docstring, replace the sentence
   `A set whose manifest's consent is not ``train`` (silver_frontier) is refused.` with:
   `A set whose manifest's consent is not ``train`` is refused, and inside a set every row written by a commercial provider's model (Claude, OpenAI, a Claude Code build session) is held back until the owner records a rights review (``provider_outputs.may_train``, the owner 2026-11-18).`
2. After the class `NotForTraining`, add:
   ```python
   def provider_of(row: dict[str, Any]) -> tuple[str | None, bool]:
       """(the provider's answerer, whether the writer was recorded) for one stored row. A chat example names its model in
       ``check``; a trajectory names its answerer, or is a Claude Code build session (``agent: claude_code``). The answerer
       is a provider's when it is not DCLab's rules, not a local model and not the OpenRouter Qwen (``sets.llm.silver_of``)."""
       from .sets.llm import exchange_answerer, silver_of

       check = row.get("check") if isinstance(row.get("check"), dict) else {}
       problem = row.get("problem") if isinstance(row.get("problem"), dict) else {}
       if check.get("model"):
           who = exchange_answerer(check)
       elif problem.get("answerer"):
           who = str(problem["answerer"])
       elif problem.get("agent") == "claude_code":
           who = "claude:claude_code"
       else:
           return None, False
       return (who if silver_of(who) else None), True
   ```
3. In `def export(`, right after `skipped: dict[str, int] = {}`, add:
   ```python
       held: dict[str, int] = {}  # a provider model's rows held back until the owner records a rights review
       not_recorded = 0  # rows that do not say which model wrote them: kept, and counted so the owner sees them
       from . import provider_outputs

       may_train = provider_outputs.may_train()
   ```
4. In the row loop, right after the `continue` of the block that starts `if row.get("split") not in SPLITS:`, add:
   ```python
               who, recorded = provider_of(row)
               not_recorded += 0 if recorded else 1
               if who and not may_train:
                   kind = who.split(":", 1)[0]
                   held[kind] = held.get(kind, 0) + 1
                   skipped["a provider model's output, held until the owner records a rights review"] = (
                       skipped.get("a provider model's output, held until the owner records a rights review", 0) + 1)
                   continue
   ```
5. In the `"export": {...}` dict of the manifest, after `"skipped": skipped,` add:
   ```python
   "held_for_rights": held, "answerer_not_recorded": not_recorded,
   ```

### B4. Tests: `tests/test_store_export_chat.py`

1. At the top, after `import unittest`, add `from unittest import mock`.
2. Add this class after the class `ExportTests` (it reuses the file's own `chat()` and `provenance()` helpers):

```python
class ProviderGateTests(unittest.TestCase):
    """The owner, 2026-11-18: every Claude answer is kept and prepared, and no training export uses one until the owner
    records a rights review. Before, Claude rows in sets marked consent "train" were exported."""

    def setUp(self):
        from dclab_core.store import SolverTrajectory, TrajectoryStep
        from dclab_rnd.store import Layout
        from dclab_rnd.store.sets import write_set

        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.layout = Layout(self.folder / "store")
        rows = [chat("claude", "train").model_copy(update={"check": {"model": "claude-sonnet-5-5"}}),
                chat("qwen", "train").model_copy(update={"check": {"model": "qwen/qwen3.8-flash", "provider": "openrouter",
                                                                   "source_class": "open_weights_model"}}),
                chat("unsaid", "train")]
        write_set(self.layout, "m4_workflow_llm", "chat/mixed", rows, version="v1", about="answerers of every kind")
        step = TrajectoryStep(n=0, tool="note", arguments={"text": "x"}, result={"ok": True}, no_thought="none kept")
        build = SolverTrajectory(id="build1", model_type="m5_solver_llm", split="train", provenance=provenance(), problem_kind="development",
                                 problem={"agent": "claude_code", "request": "fix a bug"}, steps=[step])
        write_set(self.layout, "m5_solver_llm", "trajectories/development", [build], version="v1", about="build history")

    def decisions(self, reviewed):
        from dclab_rnd.store import provider_outputs

        path = self.folder / "owner_decisions.json"
        path.write_text(json.dumps({"provider_outputs": {"for_training": True, "rights_reviewed": reviewed}}), encoding="utf-8")
        return mock.patch.object(provider_outputs, "DECISIONS", path)

    def ids(self, out):
        return sorted(json.loads(line)["id"] for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines())

    def test_a_providers_rows_are_held_until_the_rights_review_and_counted(self):
        from dclab_rnd.store import export_chat

        with self.decisions(False):
            report = export_chat.export(self.layout, "m4_workflow_llm", "chat/mixed", "v1", self.folder / "out")
        self.assertEqual(self.ids(self.folder / "out"), ["qwen", "unsaid"])  # Qwen: the owner's yes of 2026-11-11
        self.assertEqual(report["held_for_rights"], {"claude": 1})
        self.assertEqual(report["answerer_not_recorded"], 1)  # kept, and said

    def test_a_claude_code_build_session_is_a_providers_output_too(self):
        from dclab_rnd.store import export_chat

        with self.decisions(False):
            report = export_chat.export(self.layout, "m5_solver_llm", "trajectories/development", "v1", self.folder / "dev")
        self.assertEqual(self.ids(self.folder / "dev"), [])
        self.assertEqual(report["held_for_rights"], {"claude": 1})

    def test_once_the_rights_are_reviewed_every_row_is_exported(self):
        from dclab_rnd.store import export_chat

        with self.decisions(True):
            report = export_chat.export(self.layout, "m4_workflow_llm", "chat/mixed", "v1", self.folder / "ok")
        self.assertEqual(self.ids(self.folder / "ok"), ["claude", "qwen", "unsaid"])
        self.assertEqual(report["held_for_rights"], {})
```

### B5. Tests: `tests/test_provider_outputs.py`

Add this test to the class `ProviderOutputsTest`:

```python
    def test_the_owners_hold_of_2026_11_18_keeps_provider_outputs_out_of_training(self):
        from dclab_rnd.store import provider_outputs as P

        self.assertTrue(P.for_training())  # kept and prepared in the sets
        self.assertFalse(P.rights_reviewed())  # the owner has not recorded the review yet
        self.assertFalse(P.may_train())  # so no training export uses them
```

If `test_provider_outputs.py` imports `provider_outputs` under another name at the top, use that name instead and say so.

### B6. One line in `CLAUDE.md`

In the Map, find the text `Claude's answers and the build history evaluation only;` and replace it with:
`Claude's answers and the build history evaluation only; in the store they are kept and prepared, and every training export holds them back until the owner records a rights review (store/provider_outputs.py may_train, the owner 2026-11-18);`

### B7. Run

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_export_chat.py test_provider_outputs.py test_store_m5_trajectories.py test_store_dclab_code_sets.py test_store_sets_llm.py test_store_sets_decisions.py test_store_routing.py
```

**Done when:** `OK`; the four new tests pass; no existing test changed.

**If `test_a_claude_code_build_session_is_a_providers_output_too` fails because `write_set` refuses the trajectory:** print
the error. If it names a missing field of `SolverTrajectory`, add that field to the test's trajectory with a plain value
and say so; never change `dclab_core`.

**Commit** (paths: `dclab_rnd/store/owner_decisions.json`, `dclab_rnd/store/provider_outputs.py`,
`dclab_rnd/store/export_chat.py`, `tests/test_store_export_chat.py`, `tests/test_provider_outputs.py`, `CLAUDE.md`).
First line: `Provider outputs are kept and prepared, and every training export holds them until a rights review`.

---

## Task C: draw the frozen exams (the owner said "draw now")

**Goal.** Each model type gets its frozen exam, drawn once from its sealed rows, checked to be in no training set,
written to the store and pinned by its hash in `dclab_rnd/store/exam_pins.json`. From then on, every set build holds
those rows out of training (`dclab_rnd/store/exams.py` `guard`). **A drawn exam is final: it is never drawn again.**

### C1. Check before drawing

```bash
df -h .
ls dclab_rnd/store/exam_pins.json training_records/store/exams 2>&1
```
Both must say "No such file or directory" (nothing drawn yet). If either exists, stop and report: an exam was drawn
already.

### C2. The dry run (writes nothing)

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.exams draw > /tmp/exams_dry.json 2>&1; echo "rc=$?"; cat /tmp/exams_dry.json
```
It prints, per model type, `items` and `sections`, or `refused` with a reason. Every item has `"written": false`.
It may take several minutes (it reads every set). Write the counts down for the report.

If **every** type is refused, or the command fails, stop and report the output. If some types are refused, go on:
those stay undrawn and are reported.

### C3. The real draw

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.exams draw --yes > /tmp/exams_drawn.json 2>&1; echo "rc=$?"; cat /tmp/exams_drawn.json
```
Every drawn type shows `"written": true` and the same `items` count as in the dry run. If a count differs, report it.

### C4. Check what was drawn

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.exams status
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.exams verify
cat dclab_rnd/store/exam_pins.json
```
`status` lists each drawn type with its `hash` and `items`; `verify` prints `"clean": true` for each; the pins file
holds one 64-character hash per drawn type under `exam_v1`, equal to the hash in `status`.

### C5. A test that the drawn exams match their pins: add to `tests/test_store_exams.py`

```python
class DrawnExamsTest(unittest.TestCase):
    """The owner's word, 2026-11-18: the frozen exams are drawn. Each pinned exam is in the store and matches its pin."""

    def test_every_pinned_exam_is_in_the_store_and_matches_its_pin(self):
        import json as _json

        from dclab_rnd.store import exams
        from dclab_rnd.store.layout import Layout

        if not exams.PINS.is_file():
            self.skipTest("no exam drawn in this checkout")
        pins = _json.loads(exams.PINS.read_text(encoding="utf-8"))
        self.assertTrue(pins)
        for model_type, versions in pins.items():
            for version, pin in versions.items():
                self.assertRegex(pin, r"^[0-9a-f]{64}$")
                layout = Layout()
                if not exams.exam_file(layout, model_type, version).is_file():
                    continue  # a checkout without the store (CI): the pin alone is checked
                self.assertEqual(exams.load(layout, model_type, version).hash, pin)
```

If `Layout` is imported differently at the top of that file, follow the file's own import. Run:

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_exams.py test_store_versions.py
```

### C6. Commit and push, in two repositories

1. This repository (paths: `dclab_rnd/store/exam_pins.json`, `tests/test_store_exams.py`). First line:
   `The frozen exams are drawn and pinned, so no exam row can enter training`. Put the per-type item counts in the body.
2. The store's own repository (`training_records/store` is a separate git repository, the owner's private records):
   ```bash
   git -C training_records/store status --short exams | head
   git -C training_records/store add -- exams && git -C training_records/store commit -m "The frozen exams, drawn once and pinned (exam_v1)" -- exams
   git -C training_records/store push
   ```
   If the push fails, do not force it: report the message.

---

## Task D: the record, the training set, the push, the report

1. Create `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-item1.json`, in the same shape
   as `research/llm-fine-tuning/experiments/dev_trajectories/records/home-live-build-fixes.json` (same keys). Values the
   validator accepts: a finding's `source` is one of `reviewer`, `test`, `measurement`, `owner`, `self`; `severity` one
   of `behaviour`, `latent`, `gap`, `flaky`, `privacy`, `leakage`; a measurement's `verdict` one of `pass`, `fail`,
   `not scored`. Three findings (A: `reviewer`, `behaviour`; B: `reviewer`, `gap`; C: `reviewer`, `gap`), each with the
   problem from the table at the top, its cause, the fix and the test names. One measurement: the exam draw, verdict
   `not scored`, its result the per-type item counts.
2. `DCLAB_NO_LIVE_MODELS=1 make sft-dev` must end with a line like `N examples from M packages ...`.
3. Run the freshness checks:
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.data_catalog --check
   ```
   Must print `data catalog is fresh`. If it says stale: `DCLAB_NO_LIVE_MODELS=1 make catalog`, check with
   `git diff --stat -- docs/DATA_CATALOG.md docs/data_catalog.json` that only those two files changed, and add them to the commit.
4. Final run of every file this plan touched:
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_reasoning.py test_no_think_in_training.py test_training_lake.py test_own_llm.py test_prompt_audit.py test_store_export_chat.py test_provider_outputs.py test_store_m5_trajectories.py test_store_sets_llm.py test_store_exams.py test_store_versions.py test_prompt_gate.py
   ```
5. Commit the record and the rebuilt set (the record and the three files `git status --short` shows under
   `research/llm-fine-tuning/experiments/sft/out_dev/`), then push:
   ```bash
   C=$(git rev-parse HEAD); git push origin "${C}:refs/heads/main" "${C}:refs/heads/dev"; git branch -f dev HEAD
   ```
6. Report to the owner in plain words: per task what changed, the test names and counts of each run, the exam counts per
   type (dry and real), how many rows the solver re-export wrote and that it holds 0 `<think>`, anything that did not go
   as planned, and the commit ids in both repositories.

---

## Verification (how the owner knows it is done)

1. **Tests:** the run of step D4 ends with `OK`. New or replaced tests:
   - A: `test_the_export_never_writes_a_rules_trace_as_the_models_thinking`, `test_no_export_asks_a_model_to_think_inside_tags`, `test_no_code_asks_for_think_tags_or_wraps_a_trace_in_them`
   - B: `test_a_providers_rows_are_held_until_the_rights_review_and_counted`, `test_a_claude_code_build_session_is_a_providers_output_too`, `test_once_the_rights_are_reviewed_every_row_is_exported`, `test_the_owners_hold_of_2026_11_18_keeps_provider_outputs_out_of_training`
   - C: `test_every_pinned_exam_is_in_the_store_and_matches_its_pin`
2. **Data:** `grep -c "<think>" lake/export/solver/train.chat.jsonl` prints `0` (if the re-export ran);
   `python -m dclab_rnd.store.exams status` lists every drawn type; `dclab_rnd/store/exam_pins.json` is committed.
3. **A real export proves the hold** (optional, no cost): export one real set that holds Claude rows and read its counts:
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.export_chat --model-type m4_workflow_llm --folder understand_problem --version v2 --out /tmp/exp_check
   ```
   Its printed `held_for_rights` must show `claude` with a count above 0 (about 1,500 expected).
4. **Nothing forbidden changed:** `git diff --stat` over the new commits lists only the files named in this plan; no
   file under `dclab_rnd/prompts/` or `dclab_core/`.

## Not in this plan (said so nobody expects it)

- **The rights review itself is the owner's.** When it is done, set `"rights_reviewed": true` in
  `dclab_rnd/store/owner_decisions.json`; the exports then include the provider rows with no code change.
- The lake still drops the OpenRouter Qwen rows of workspace sessions (it predates the 2026-11-11 decision; the store
  keeps them). Changing the lake needs a `dclab_core` contract change: a separate package.
- A chat row that does not record its model is exported and counted (`answerer_not_recorded`), not held. If those turn
  out to be Claude's, the writer must record the model: a separate fix.
- An exam file without a pin is still loaded unchecked, and a tuned model's readiness does not fail when no exam exists.
  After this draw both cases need a stray file to matter; hardening them is a separate small fix.
- The large `sealed.pending` partitions are not promoted first: the exam is drawn from what is sealed today. A later
  set version gets its own exam version.
- The small decision classifiers keep refitting (the owner's answer: keep everything); nothing about them changes.

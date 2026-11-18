# Hold everything Claude wrote out of every training run and export (a plan for a smaller model)

## Context

The owner's rule (2026-11-18): every output of a provider's model (Claude, OpenAI, any hosted model other than the
OpenRouter Qwen approved on 2026-11-11) is **kept and prepared**, and **no training run or training export uses it** until
the owner records a rights review (`dclab_rnd/store/provider_outputs.py` `may_train()`, False today).

So far only the chat export (`dclab_rnd/store/export_chat.py`) obeys it. A check of every other training path
(2026-11-18) found that Claude's output still reaches training through seven places.

**The owner's answer on scope (2026-11-18): "Everything Claude wrote".** That means:
- Claude's labels, answers and thinking.
- The words and truths of every **written person or request**: the written Home requests, the written journeys (test
  people), the exam people, and the people of every test workspace. This holds whether a model wrote them through
  DCLab's gateway or a Claude Code session wrote them into a file. Every hand-made journey and request file was committed
  with Claude as co-author, so Claude Code wrote them.
- Data that **running code** produced is the code's, not Claude's. This covers an invented table, a rule's answer, a
  measured label, and the bands code computed.

Nothing is deleted. Everything stays kept and prepared. The hold is one switch: when the owner records
`"rights_reviewed": true`, every path trains on everything again with no code change.

**The seven places that let Claude's output into training today (checked in code and on the real store):**

| # | Where | What gets through today |
|---|---|---|
| 1 | `dclab_rnd/store/tutor.py` `teach` (runs after every store run) | a teacher label (Claude, through `decision_labeler`) when a point has fewer than 20 confirmed labels |
| 2 | `dclab_rnd/store/sets/routing.py` `export` (`make capture`) | Claude's silver labels; the written requests (Claude Sonnet); test people's words; Claude's thinking as a row's reasoning; the `model` and `model_thinking` table columns |
| 3 | `dclab_rnd/lake/answers.py` `export_gliner2` (`make capture`) | routing silver and written rows; every written journey (`journeys` source); every journey-played workflow row (`workflow` source) |
| 4 | `dclab_rnd/intent_classifier.py` `train` | it trains only on `intent_requests_v1.json`: 871 requests, all written by Claude Sonnet 5.5 |
| 5 | `dclab_rnd/understand_students.py` `train` | journeys written by Claude (homework v1: Claude Sonnet; v2 and v3: a model whose requests were not captured) |
| 6 | `dclab_rnd/workflow_models.py` `train` | rows played from written journeys (every source is a journey set written in a Claude Code session) |
| 7 | `dclab_rnd/decision_export.py` `build` (`make decisions-export`) | the `workflow_v1` rows (the same journeys) |
| also | `dclab_rnd/store/export_chat.py` `export` | Home conversations and other chat rows from **test workspaces** (`live_tests__...`): no model is named, but their person was written for the test |

**Measured on the real store** (train and dev rows; the planned rule run in memory, nothing written):

| Model type | Rows | Rows with something Claude wrote (by kind) |
|---|---|---|
| m1_students | 280,556 | teacher labels 1,121 |
| m2_workflow_decisions | 26,623 | teacher labels 96 |
| m3_ml_decisions | 210,079 | teacher labels 532 |
| m4_workflow_llm | 21,708 | claude 7,233; mixed 144; frontier 641; Claude Code (test people) 2,792 |
| m5_solver_llm | 2,005 | Claude Code 330; claude 22; mixed 2; openai 1 |
| m6_stepwise_decisions | 5,227 | Claude Code (DCLab Code sessions) 56 |
| m7_judges | 12,408 | Claude Code (reviewer, development) 3,152; claude 311 |
| m7_routing | 8,156 | Claude Code (written, eval and scripted requests) 6,649; claude 867; frontier 178 |
| m8_retrieval | 98 | (agent searches, counted after G1) |

**What the owner gives up until the rights review (said plainly):** the intent classifier, the understanding students and
the workflow models cannot be retrained (every row they learn from was written by Claude). The models already fitted are
**not touched** and keep working. The routing and GLiNER2 exports shrink to the rows a real person labelled and code-made
sets. The tutor trains only on confirmed labels.

**Size:** one new module (about 170 lines), small guards in 8 files (about 80 lines), about 300 lines of tests, one
measurement before and after, one record. No prompt, tool, `dclab_core` or Build project change. No set is rebuilt.

---

## Rules for whoever executes this (read first; they apply to every task)

1. Work in `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`, branch `main`. Python is `.venv/bin/python`.
2. Do the tasks in order, G1 to G12. Finish one (its tests green) before the next. There are **three commits**: after G4,
   after G9 and after G12 (each says which files).
3. **Find a place by its quoted text** (`grep -n "text" file`), never by line number. Line numbers are hints only.
4. Run tests only like this (bare file names, never a bare `pytest`, never a live key):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_a.py test_b.py
   ```
   It must end with `Ran N tests ...` and then `OK`.
5. **Never run** `make check-commit`, `make check-quick`, `make check-all`, `make rd-check` or the whole suite.
   **Never run** any trainer for real (`make workflow-models`, `make own-models`, `python -m dclab_rnd.intent_classifier`,
   `python -m dclab_rnd.understand_students`), and never `make capture`. Only the commands this plan names.
6. A failing **new** test: fix the code, never the assertion. A failing **existing** test: read "The one allowed change
   to an existing test" below. Anything else: stop and report.
7. **Do not touch:** `dclab_rnd/prompts/`, any tool description, anything in `dclab_core/`, any `results/` folder, any
   `sealed.jsonl` or file with `sealed` in its name (never open one), the exam files and `dclab_rnd/store/exam_pins.json`,
   any file under `training_records/` by hand, any fitted model file (`*.joblib`, `*.pkl`), and the files
   `cases_dev3.json`, `answer_key_dev3.json`, `cases_nodata_held.json`, `answer_key_nodata_held.json`, `cases_sealed.json`.
8. **Never delete data.** A guard skips or refuses; it never removes a file, a row or a fitted model.
9. In every file except the new module, import the module **inside the function** that uses it, under the name
   `who_wrote`:
   ```python
       from dclab_rnd.store import writers as who_wrote
   ```
   (inside `dclab_rnd/store/` you may write `from . import writers as who_wrote` or `from .. import writers as who_wrote`).
   Inside the function is needed because the store's modules import each other; `who_wrote` because `intent_classifier.train`
   already has a local variable called `writers`.
10. Git: never `git stash`, never `git add .` / `git add -A`. Stage and commit explicit paths only (zsh needs an array):
    ```bash
    F=(path/one path/two); git add -- "${F[@]}" && git commit -m "message" -- "${F[@]}"
    ```
    Never stage `data/workspace/` or `training_records/STORE_CATALOG.json`. End every commit message with the attribution
    line your session gives you. Push only in G12.
11. Comments and messages: plain English, short sentences, no em dashes. A comment says **why**.

### The one allowed change to an existing test

After a guard, some **existing** tests fail only because the rows they export or train on are now held (for example, a
test that checks the routing export writes a `written` folder). Such a test is about how things work **once the owner
allows training**. You may change it in exactly one way: add these three lines as the **first lines of that test
method** (add `from unittest import mock` at the top of the file if it is missing):

```python
        reviewed = mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=True)  # the 2026-11-18 hold: this test is about after the owner's rights review
        reviewed.start()
        self.addCleanup(reviewed.stop)
```

Never change anything else in that test. Write down every test you change this way (file and name) for the report. If an
existing test still fails with these lines, stop and report it. If a failing test's `setUp` (not the test method) is
what fails, put the three lines at the start of `setUp` instead, and say so.

---

## G1. One module that says who wrote a row

Create **`dclab_rnd/store/writers.py`** with exactly this content:

```python
"""Who wrote a training row, for every kind of training data DCLab keeps (the owner, 2026-11-18).

The owner's rule: everything a provider's model wrote is kept and prepared, and no training run or training export uses it
until the owner records a rights review (``provider_outputs.may_train``). "Everything" is the owner's answer of 2026-11-18:
a label, an answer, a model's thinking, and also the words and truths of a written person or request, whether a model wrote
them through DCLab's gateway or a Claude Code session wrote them into a file. Data that running code produced (an invented
table, a rule's answer, a measured label) is the code's, not a model's.

``writer_of(row)`` names the provider that wrote some part of a stored row (``claude:...``, ``frontier:...``, ``mixed:...``),
or None when no provider did. ``held(row)`` is True when that writer must stay out of training today. Every training reader
and training export asks ``held`` before it uses a row; ``python -m dclab_rnd.store.writers audit`` counts every set.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any, Optional

from . import provider_outputs
from .export_chat import provider_of
from .sets.llm import answerer_of, silver_of

# A test's workspace: the people and requests in it were written for the test (journeys, scripted people, exam people), by
# Claude through the gateway or by a Claude Code session. A real person's words never live there.
TEST_WORKSPACES = ("live_tests__",)
# Written sets whose writer is known from its records (checked 2026-11-18). Any other written set (a journey set, a request
# set) was written in a Claude Code session: every one of them was committed with Claude as co-author.
WRITTEN_SETS = {
    "intent_requests_v1": "claude:claude-sonnet-5-5",  # every row names claude-sonnet-5-5 (home_request_writer a, b and c)
    "homework_v1": "claude:claude-sonnet-5-5",  # journey_writer: the store's 721 requests of that purpose are all Claude Sonnet 5.5
    "homework_v2": "frontier:journey_writer_b",  # written through the gateway; its requests were not captured, so the model is unknown
    "homework_v3": "frontier:journey_writer_d",  # the same
}
CLAUDE_CODE = "claude:claude_code"
LABELLER = "routing_labeler"


class HeldForRights(ValueError):
    """Every row a training run would use was written by a provider's model: nothing is left to train on until the review."""


def may_train() -> bool:
    return provider_outputs.may_train()


def model_answerer(name: Any) -> Optional[str]:
    """A model's name as an answerer (``claude:...``, ``openrouter:...``, ``local:...``); None when no name was given."""
    return answerer_of({"mode": "llm", "model": str(name)}) if name else None


def set_writer(name: Any) -> Optional[str]:
    """Who wrote a written set (a journey set such as ``homework_v1`` or ``sealed_v2``, a request set): never None for a set."""
    return WRITTEN_SETS.get(str(name), CLAUDE_CODE) if name else None


def _provider(who: Optional[str]) -> Optional[str]:
    return who if who and silver_of(who) else None


def _routing(row: dict[str, Any]) -> Optional[str]:
    """A routing row: its words (a written request, an exam's person, a test's scripted person) and its label (a model's)."""
    label_by = str(row.get("label_by") or "")
    parts = label_by.split(":")
    if parts[0] == "model" or row.get("label_source") == "silver":
        if len(parts) >= 3 and parts[1] == "model":
            return _provider(model_answerer(parts[2])) or "frontier:unknown"  # model:model:<name>, a ledger label
        if len(parts) >= 3 and parts[2] == LABELLER:
            return _provider(model_answerer(parts[1])) or "frontier:unknown"  # model:<name>:routing_labeler
        giver = parts[1] if len(parts) > 1 else ""
        reading = next((r for r in row.get("readings") or [] if r.get("by") == "model" and r.get("answerer") == giver and r.get("model")), None)
        return _provider(model_answerer(reading["model"])) if reading else "frontier:unknown"
    if parts[0] == "written" and len(parts) > 1:
        return set_writer(parts[1])
    if label_by.startswith("conversation_exam:"):
        return "frontier:exam_person"  # an exam's written person: Claude Haiku or Qwen, not named on the row
    if str(row.get("origin")) in ("written", "eval", "scripted"):
        return CLAUDE_CODE  # a request written for a test or an eval, in a Claude Code session
    return None


def reading_held(reading: dict[str, Any]) -> bool:
    """A model's reading (its answer and its thinking) on a routing row is a provider's output unless its model says otherwise."""
    if reading.get("by") != "model" or may_train():
        return False
    return bool(_provider(model_answerer(reading.get("model")) or "frontier:unknown"))


def writer_of(row: dict[str, Any]) -> Optional[str]:
    """The provider that wrote some part of a stored row, or None. Read by the row's shape (its schema_version)."""
    schema = str(row.get("schema_version") or "")
    workspace = str((row.get("provenance") or {}).get("workspace") or "")
    if schema.startswith("dclab_core.decision_example"):
        # its inputs are bands code computed; only its label can be a model's (a teacher's: Claude through decision_labeler)
        return "frontier:teacher" if row.get("label") is not None and row.get("label_source") == "teacher" else None
    if schema.startswith("dclab_core.routing_example"):
        return _routing(row)
    if schema.startswith("dclab_core.judged_example"):
        by, judge = str(row.get("candidate_by") or ""), str(row.get("judge") or "")
        if by.startswith("model:"):
            return _provider(model_answerer(by.split(":", 1)[1])) or "frontier:unknown"
        return CLAUDE_CODE if by in ("reviewer", "development") or judge == "developer" else None
    if schema.startswith("dclab_core.episode"):
        return CLAUDE_CODE if row.get("process") == "dclab_code" else None
    if schema.startswith("dclab_core.experiment_record"):
        return None  # measured by code
    if schema.startswith("dclab_core.preference_pair"):
        return CLAUDE_CODE if "dclab_code" in str(row.get("id") or "") else None
    if schema.startswith("dclab_core.retrieval_example"):
        # an agent's own search: a model wrote the query (the rules do not search in words); a person's search is not held
        return "frontier:agent_search" if "agent_step" in str((row.get("provenance") or {}).get("rule") or "") else None
    who, _ = provider_of(row)  # a chat example or a trajectory
    if who:
        return who
    if workspace.startswith(TEST_WORKSPACES) and ("messages" in row or "steps" in row):
        return CLAUDE_CODE  # a test's conversation: its person was written for the test
    return None


def held(row: dict[str, Any]) -> bool:
    """True when a provider wrote some part of the row and the owner has not recorded the rights review."""
    return writer_of(row) is not None and not may_train()


def held_writer(who: Optional[str]) -> bool:
    return bool(_provider(who)) and not may_train()


def audit(layout: Any = None) -> dict[str, Any]:
    """Every set's train and dev rows (sealed rows are never read here), counted by who wrote them."""
    from .layout import Layout

    layout = layout or Layout()
    root = layout.root / "sets"
    out: dict[str, Any] = {}
    for path in sorted(root.glob("**/train.jsonl")) + sorted(root.glob("**/dev.jsonl")):
        name = str(path.parent.relative_to(root))
        doc = out.setdefault(name, {"rows": 0, "provider": Counter()})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                doc["rows"] += 1
                who = writer_of(json.loads(line))
                if who:
                    doc["provider"][who.split(":", 1)[0] + (":claude_code" if who == CLAUDE_CODE else "")] += 1
    return {k: {"rows": v["rows"], "provider": dict(v["provider"])} for k, v in sorted(out.items())}


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m dclab_rnd.store.writers audit``: every set's train and dev rows by who wrote them, as JSON, then one line
    per model type. Reads only; writes nothing."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] != ["audit"]:
        print("usage: python -m dclab_rnd.store.writers audit", file=sys.stderr)
        return 2
    report = audit()
    totals: dict[str, Counter] = {}
    for name, doc in report.items():
        total = totals.setdefault(name.split("/", 1)[0], Counter())
        total["rows"] += doc["rows"]
        total.update(doc["provider"])
    print(json.dumps({"may_train": may_train(), "sets": report}, indent=1))
    for model_type, total in sorted(totals.items()):
        print(model_type, dict(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Done when:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "from dclab_rnd.store import writers as w; print(w.set_writer('homework_v1'), w.set_writer('sealed'), w.held_writer('openrouter:qwen/x'))"
```
prints `claude:claude-sonnet-5-5 claude:claude_code False`.

## G2. Tests for the module: new file `tests/test_store_writers.py`

```python
"""The owner, 2026-11-18: everything Claude wrote (labels, answers, thinking, and the words of written people and requests)
is kept and prepared, and no training run or export uses it until the owner records a rights review. One module says who
wrote every kind of stored row."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.store import writers as W  # noqa: E402

DEC = "dclab_core.decision_example.v4"
RT = "dclab_core.routing_example.v1"


def routing(**k):
    return {"schema_version": RT, "label": "analysis", **k}


class WriterTests(unittest.TestCase):
    def setUp(self):
        hold = mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=False)  # the owner's hold, whatever the file says
        hold.start()
        self.addCleanup(hold.stop)

    def test_a_decision_rows_teacher_label_is_a_models_and_a_measured_one_is_not(self):
        self.assertEqual(W.writer_of({"schema_version": DEC, "label": True, "label_source": "teacher"}), "frontier:teacher")
        for source in ("person", "measured", "benchmark", "blind"):
            self.assertIsNone(W.writer_of({"schema_version": DEC, "label": True, "label_source": source}), source)
        self.assertIsNone(W.writer_of({"schema_version": DEC, "label": None, "label_source": "teacher"}))  # no label: nothing trains on it

    def test_a_routing_rows_label_and_words_name_their_writer(self):
        self.assertEqual(W.writer_of(routing(label_source="silver", label_by="model:claude-sonnet-5-5:routing_labeler")), "claude:claude-sonnet-5-5")
        self.assertEqual(W.writer_of(routing(label_source="silver", label_by="model:model:claude-sonnet-5-5")), "claude:claude-sonnet-5-5")
        qwen = [{"by": "model", "answerer": "home_intent", "model": "qwen/qwen3.8-flash"}]
        self.assertIsNone(W.writer_of(routing(label_source="silver", label_by="model:home_intent", readings=qwen)))  # the owner's yes to Qwen
        self.assertEqual(W.writer_of(routing(label_source="silver", label_by="model:home_intent")), "frontier:unknown")
        self.assertEqual(W.writer_of(routing(label_source="written", label_by="written:intent_requests_v1:a", origin="written")), "claude:claude-sonnet-5-5")
        self.assertEqual(W.writer_of(routing(label_source="written", label_by="eval:requests_dev", origin="eval")), "claude:claude_code")
        self.assertEqual(W.writer_of(routing(label_source="written", label_by="conversation_exam:people", origin="eval")), "frontier:exam_person")
        self.assertEqual(W.writer_of(routing(label=None, origin="scripted")), "claude:claude_code")  # a test's scripted person
        self.assertIsNone(W.writer_of(routing(label_source="real", label_by="person:card", origin="real")))

    def test_a_models_reading_is_held_unless_its_model_is_the_approved_qwen(self):
        self.assertTrue(W.reading_held({"by": "model", "model": "claude-sonnet-5-5"}))
        self.assertTrue(W.reading_held({"by": "model"}))  # no model named: the safe side
        self.assertFalse(W.reading_held({"by": "model", "model": "qwen/qwen3.8-flash"}))
        self.assertFalse(W.reading_held({"by": "rule", "model": "claude-sonnet-5-5"}))

    def test_judged_episode_preference_retrieval_and_experiment_rows(self):
        judged = {"schema_version": "dclab_core.judged_example.v1"}
        self.assertEqual(W.writer_of({**judged, "candidate_by": "model:claude-haiku-5-5"}), "claude:claude-haiku-5-5")
        self.assertEqual(W.writer_of({**judged, "candidate_by": "reviewer"}), "claude:claude_code")
        self.assertEqual(W.writer_of({**judged, "candidate_by": "development"}), "claude:claude_code")
        self.assertEqual(W.writer_of({**judged, "candidate_by": "human", "judge": "developer"}), "claude:claude_code")
        self.assertIsNone(W.writer_of({**judged, "candidate_by": "human", "judge": "validator"}))
        self.assertEqual(W.writer_of({"schema_version": "dclab_core.episode.v3", "process": "dclab_code"}), "claude:claude_code")
        self.assertIsNone(W.writer_of({"schema_version": "dclab_core.episode.v3", "process": "walk"}))
        self.assertIsNone(W.writer_of({"schema_version": "dclab_core.experiment_record.v1"}))
        search = {"schema_version": "dclab_core.retrieval_example.v1", "provenance": {"rule": "capture_rules_v2:agent_step"}}
        self.assertEqual(W.writer_of(search), "frontier:agent_search")

    def test_a_chat_row_from_a_test_workspace_is_held_and_one_from_the_owners_workspace_is_not(self):
        chat = {"schema_version": "dclab_core.chat_example.v3", "task": "t", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(W.writer_of({**chat, "provenance": {"workspace": "live_tests__2026-10-30-journeys"}}), "claude:claude_code")
        self.assertIsNone(W.writer_of({**chat, "provenance": {"workspace": "dev"}}))
        self.assertEqual(W.writer_of({**chat, "provenance": {"workspace": "dev"}, "check": {"model": "claude-sonnet-5-5"}}), "claude:claude-sonnet-5-5")

    def test_written_sets_name_their_writer(self):
        self.assertEqual(W.set_writer("intent_requests_v1"), "claude:claude-sonnet-5-5")
        self.assertEqual(W.set_writer("homework_v1"), "claude:claude-sonnet-5-5")
        self.assertEqual(W.set_writer("homework_v2"), "frontier:journey_writer_b")
        self.assertEqual(W.set_writer("sealed_v3"), "claude:claude_code")  # any other written set: a Claude Code session
        self.assertIsNone(W.set_writer(None))

    def test_held_follows_the_owners_switch(self):
        row = {"schema_version": DEC, "label": True, "label_source": "teacher"}
        self.assertTrue(W.held(row))
        with mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=True):
            self.assertFalse(W.held(row))  # after the rights review: every row trains again, no code change
            self.assertFalse(W.held_writer("claude:x"))
        self.assertTrue(W.held_writer("claude:x"))
        self.assertFalse(W.held_writer("openrouter:qwen/qwen3.8-flash"))
        self.assertFalse(W.held_writer(None))

    def test_the_audit_counts_every_set_and_never_reads_a_sealed_file(self):
        from dclab_rnd.store import Layout

        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        base = layout.sets("m1_students", "column.time", "v1")
        base.mkdir(parents=True)
        rows = [{"schema_version": DEC, "label": True, "label_source": "teacher"}, {"schema_version": DEC, "label": True, "label_source": "person"}]
        (base / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (base / "sealed.jsonl").write_text("not json: the audit must never read this file\n", encoding="utf-8")
        report = W.audit(layout)
        self.assertEqual(report, {"m1_students/column.time/v1": {"rows": 2, "provider": {"frontier": 1}}})


if __name__ == "__main__":
    unittest.main()
```

Run:
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_writers.py
```
**Done when:** `OK` with 8 tests.

## G3. The check, before any guard: run the audit on the real store and keep its output

```bash
df -h .
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.writers audit > /tmp/writers_audit_before.json 2>&1; echo "rc=$?"; tail -12 /tmp/writers_audit_before.json
```
`rc=0` is required. It reads only train and dev rows (several minutes is normal). Write down the per model type lines
(the last lines) for the report. **Expected** (within 10%): the table in Context, plus a number for `m8_retrieval`.
If a model type is off by more than 10%, do not change code: report both numbers.

Also keep the numbers of the exports **as they are today**, before any guard (each command writes only under `/tmp`):
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "from pathlib import Path; from dclab_rnd.store.sets import routing; m = routing.export(out=Path('/tmp/routing_before')); print(m['rows'], m['skipped'], len(m['chat']))"
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "from pathlib import Path; from dclab_rnd.lake import answers; m = answers.export_gliner2(out=Path('/tmp/gliner_before')); import json; print(json.dumps(m['counts'].get('files', {}))[:1500])"
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.export_chat --model-type m4_workflow_llm --folder understand_problem --version v2 --out /tmp/chat_before > /dev/null 2>&1; .venv/bin/python -c "import json; e = json.load(open('/tmp/chat_before/MANIFEST.json'))['export']; print(e['counts'], e['held_for_rights'], e['answerer_not_recorded'])"
```
If one of them fails, write down its last lines and go on (it is the "before" number only).

## G4. Commit the module, its tests and the check

```bash
F=(dclab_rnd/store/writers.py tests/test_store_writers.py); git add -- "${F[@]}" && git commit -m "One module says who wrote every kind of stored training row" -- "${F[@]}"
```
Put the G3 per model type lines in the commit body.

---

## G5. The chat export holds a test workspace's rows

**File:** `dclab_rnd/store/export_chat.py`, function `def export(`.

Find these two lines in the row loop:
```python
            who, recorded = provider_of(row)
            not_recorded += 0 if recorded else 1
```
Replace them with:
```python
            who, recorded = provider_of(row)
            not_recorded += 0 if recorded else 1
            # a row whose model is not named may still hold words written for a test (the owner, 2026-11-18: everything Claude wrote)
            who = who or who_wrote.writer_of(row)
```
and add, as the first line of the function body of `def export(` (right after its signature line):
```python
    from . import writers as who_wrote
```
Nothing else changes: the existing `if who and not may_train:` block below then holds it and counts it under its kind.

**Test:** add to `tests/test_store_export_chat.py`, inside the class `ProviderGateTests`, as its last method:
```python
    def test_a_rules_conversation_from_a_test_workspace_is_held_because_its_person_was_written(self):
        """The owner, 2026-11-18: everything Claude wrote is held, also a test person's words. A row with no model from a
        test workspace (live_tests__...) holds such words; the same row from the owner's own workspace does not."""
        from dclab_rnd.store import export_chat
        from dclab_rnd.store.sets import write_set

        test_ws = provenance().model_copy(update={"workspace": "live_tests__2026-10-30-journeys"})
        rows = [chat("tested", "train").model_copy(update={"provenance": test_ws, "check": {"answerer": "rules"}}),
                chat("owners", "train").model_copy(update={"check": {"answerer": "rules"}})]
        write_set(self.layout, "m4_workflow_llm", "chat/tested", rows, version="v1", about="a test's conversation and the owner's")
        with self.decisions(False):
            report = export_chat.export(self.layout, "m4_workflow_llm", "chat/tested", "v1", self.folder / "tested")
        self.assertEqual(self.ids(self.folder / "tested"), ["owners"])
        self.assertEqual(report["held_for_rights"], {"claude": 1})
```
Why this case: `check["answerer"] == "rules"` makes `provider_of` return `(None, True)`, so `writer_of` then decides by the
workspace. That is the case this test is about.

Run:
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_export_chat.py test_store_sets_llm.py test_home_answerer.py test_store_cases.py
```

## G6. The tutor never trains on a held label

**File:** `dclab_rnd/store/tutor.py`.

1. In `def _rows(`, find the line that starts `        rows.append({"record": ex["id"],` and, in the dict it builds, add one key at the end,
   after `"weight": ex.get("weight", 1.0)`:
   ```python
   , "held": who_wrote.held(ex)
   ```
   so the dict ends with `"weight": ex.get("weight", 1.0), "held": who_wrote.held(ex)})`. Add as the first line of the body
   of `def _rows(`:
   ```python
       from . import writers as who_wrote
   ```
2. In `def teach(`, find:
   ```python
            labelled = [r for r in _rows(folder / "v1" / "train.jsonl") if r["label"]]
   ```
   and replace it with:
   ```python
            rows = _rows(folder / "v1" / "train.jsonl")
            # a label a provider's model gave (a teacher's) stays out until the owner's rights review (2026-11-18)
            held = sum(1 for r in rows if r["label"] and r["held"])
            if held:
                report.setdefault("held_for_rights", {})[point] = held
            labelled = [r for r in rows if r["label"] and not r["held"]]
   ```
3. Find:
   ```python
            dev_rows = [r for r in _rows(folder / "v1" / "dev.jsonl") if r["label"] and (r["label"] or {}).get("source") in CONFIRMED]  # measured on confirmed only
   ```
   and replace it with:
   ```python
            dev_rows = [r for r in _rows(folder / "v1" / "dev.jsonl") if r["label"] and not r["held"]
                        and (r["label"] or {}).get("source") in CONFIRMED]  # measured on confirmed only
   ```

**Test:** add to `tests/test_store_tutor.py`, inside the class `TutorTest`, as its last method:
```python
    def test_a_teachers_label_never_trains_a_student_until_the_rights_review(self):
        """The owner, 2026-11-18: a teacher's label is Claude's (decision_labeler). With too few confirmed labels the tutor
        used to fall back to it; now it is held and counted, and the point waits."""
        from unittest import mock

        from dclab_rnd.store import tutor

        base = self.layout.sets("m1_students", "column.is_identifier", "v1")
        for split in ("train", "dev"):
            path = base / f"{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            path.write_text("".join(json.dumps({**r, "label_source": "teacher"}) + "\n" for r in rows), encoding="utf-8")
        with mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=False):
            report = tutor.teach(self.layout, run="tutor-held")
        self.assertNotIn("column.is_identifier", report["students"])
        self.assertIn("column.is_identifier", report["waiting"])
        self.assertGreater(report["held_for_rights"]["column.is_identifier"], 0)
```

Run:
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_tutor.py test_store_writers.py
```

## G7. The routing export writes no held row, and never a provider's thinking or answer

**File:** `dclab_rnd/store/sets/routing.py`.

1. In `def chat_row(`, find:
   ```python
       same = [r for r in ex.get("readings") or [] if r.get("by") == "model" and r.get("value") == ex.get("label") and r.get("thinking")
               and not (r.get("answerer") == LABELLER and ex.get("point") in NOT_LABELS)]  # the review: never its reasoning either
   ```
   and replace it with:
   ```python
       from .. import writers as who_wrote

       same = [r for r in ex.get("readings") or [] if r.get("by") == "model" and r.get("value") == ex.get("label") and r.get("thinking")
               and not (r.get("answerer") == LABELLER and ex.get("point") in NOT_LABELS)  # the review: never its reasoning either
               and not who_wrote.reading_held(r)]  # a provider's thinking waits for the owner's rights review (2026-11-18)
   ```
2. In `def export(`, add as the first line of its body (before `import csv`):
   ```python
       from .. import writers as who_wrote
   ```
3. In the loop of `def export(`, find:
   ```python
           readings = ex.get("readings") or []
   ```
   and put these lines **right before** it:
   ```python
           who = who_wrote.writer_of(ex)
           if who and not who_wrote.may_train():  # a provider wrote its label or its words: kept in the store, never exported yet
               skipped[f"held until the owner's rights review: {who.split(':', 1)[0]}"] += 1
               continue
   ```
4. In the same loop, find:
   ```python
           model = first("model")
   ```
   and replace it with:
   ```python
           model = first("model")
           if who_wrote.reading_held(model):
               model = {}  # a provider's answer and thinking stay out of the table too
   ```
5. In the manifest's `"rule"` text, add at the end (inside the string, before its closing quote):
   ` Until the owner records a rights review, a row a provider wrote (its label or its words) is skipped and counted, and a provider's reading is never a reason or a table column.`

**Test:** add to `tests/test_store_routing.py`, inside the class `RoutingCollectorTest`, right after the test
`test_the_chat_and_table_exports_keep_each_class_apart_and_never_an_exam`:
```python
    def test_the_export_holds_every_row_a_provider_wrote_and_every_providers_thinking(self):
        """The owner, 2026-11-18: everything Claude wrote waits for the rights review. The written requests and a model's
        label are skipped and counted; a person's label stays, without a provider's thinking as its reason."""
        import csv
        from unittest import mock

        from dclab_rnd.store.sets import routing

        self.build()
        out = self.tmp / "held"
        with mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=False):
            manifest = routing.export(self.layout, out=out)
        root = out / "v1"
        chats = [json.loads(line) for p in (root / "chat").rglob("*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertFalse({p.parent.name for p in (root / "chat").rglob("*.jsonl")} & {"written", "silver"})
        self.assertTrue(any(k.startswith("held until the owner's rights review") for k in manifest["skipped"]))
        r1 = next(c for c in chats if c["id"] == "m7_routing:dev:dra000000001")  # a person's label: kept
        self.assertNotEqual(r1["messages"][2].get("reasoning_source"), "recorded")  # the model's thinking is not its reason
        with (root / "table.csv").open(encoding="utf-8") as fh:
            table = list(csv.DictReader(fh))
        self.assertFalse(any(r["id"] == "m7_routing:written:IR1-busi-002" for r in table))
        self.assertTrue(all(not r["model_thinking"] for r in table))
```
If the fixture's reading on `dra000000001` names the approved Qwen model (then its thinking is allowed), the third
assertion fails for a good reason: report it and stop; do not change the assertion.

Then run:
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_routing.py test_store_writers.py
```
Expect `test_the_chat_and_table_exports_keep_each_class_apart_and_never_an_exam` to fail now (it expects `written` files):
give it "the one allowed change". The same may happen to `test_claudes_label_carries_its_api_thinking_summary_as_the_answers_reasoning`.

## G8. The GLiNER2 export holds held rows of every source

**File:** `dclab_rnd/lake/answers.py`.

1. In `def _routing(`, find the line `                rule = next((r.get("value") for r in ex.get("readings") or [] if r.get("by") == "rule"), None)`.
   Right **before** it add:
   ```python
                from ..store import writers as who_wrote
   ```
   and in the `yield {...}` dict right after it, add one key at the end (after `"language": ex.get("language")`):
   `, "writer": who_wrote.writer_of(ex)`.
2. In `def _journeys(`, in its `yield {...}` dict, add one key at the end (after `"set": name`):
   `, "writer": who_wrote.set_writer(name)`, and add `from ..store import writers as who_wrote` as the first line of the
   function body.
3. In `def _workflow(`, in its `yield {...}` dict, add one key at the end (after `"label_origin": "journey_truth"`):
   `, "writer": who_wrote.set_writer(r.get("source"))`, and add `from ..store import writers as who_wrote` as the first
   line of the function body (before the existing `from dclab_core.workflow_labels import ...`).
4. In `def export_gliner2(`, add `from ..store import writers as who_wrote` as the first line of its body. In its loop,
   find:
   ```python
            folder = root / source / d["decision_point"] / (d["apart"] if d.get("apart") else "")  # routing: each label class apart
   ```
   and put these lines **right before** it:
   ```python
            if who_wrote.held_writer(d.get("writer")):  # a provider wrote its words or its label: kept, never exported yet (2026-11-18)
                counts.setdefault("held_for_rights", Counter())[f"{source}: {str(d['writer']).split(':', 1)[0]}"] += 1
                continue
   ```
5. In the manifest's `"rule"` text, add at the end (inside the string): ` Until the owner records a rights review, a row a provider wrote (a written journey, a written or silver routing row, a journey-played workflow row) is held and counted under counts.held_for_rights.`

The `writer` key does not change any test's fingerprint (`fingerprint` reads only the text and the schema).

**Test:** add to `tests/test_lake_answers.py` a new class at the end (before `if __name__`):
```python
class HeldForRightsTests(unittest.TestCase):
    """The owner, 2026-11-18: a written journey, a written or silver routing row and a journey-played workflow row are
    Claude's; the GLiNER2 export holds them and counts them until the rights review."""

    def test_a_held_writers_rows_are_counted_and_never_written(self):
        import tempfile
        from collections import Counter
        from unittest import mock

        from dclab_rnd.lake import answers

        base = {"decision_point": "problem.family_from_words", "split": "train", "group": "g", "schema": {"answer": {"labels": ["a", "b"], "prompt": "Which?"}},
                "options": ["a", "b"], "label": "a", "accepted": ["a"], "rule_answer": None, "label_origin": "journey_truth"}
        rows = [{**base, "sample_id": "s1", "text": "written by claude", "writer": "claude:claude_code"},
                {**base, "sample_id": "s2", "text": "made by code", "writer": None}]
        out = Path(tempfile.mkdtemp())
        with mock.patch.dict(answers.PROVIDERS, {"fake": lambda _lake: iter(rows)}), \
                mock.patch("dclab_rnd.store.provider_outputs.may_train", return_value=False):
            manifest = answers.export_gliner2(sources=("fake",), out=out)
        written = [json.loads(line) for p in out.rglob("train.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([w["input"] for w in written], ["made by code"])
        self.assertEqual(manifest["counts"]["held_for_rights"], {"fake: claude": 1})
```
If `gliner2_row` needs another key in the fake rows, add that key to `base` (never change the assertions) and say so.
Check that `json`, `Path` and `unittest` are imported at the top of `tests/test_lake_answers.py`; add any that is missing.

Run:
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_lake_answers.py test_store_routing.py
```
`test_the_gliner2_export_keeps_each_label_class_apart_and_coverage_counts_them` reads `_routing` only (not the export),
so it should still pass; if it fails, give it "the one allowed change".

## G9. Commit the guards of the exports

```bash
F=(dclab_rnd/store/export_chat.py dclab_rnd/store/tutor.py dclab_rnd/store/sets/routing.py dclab_rnd/lake/answers.py tests/test_store_export_chat.py tests/test_store_tutor.py tests/test_store_routing.py tests/test_lake_answers.py); git add -- "${F[@]}" && git commit -m "The tutor and every training export hold what a provider's model wrote until the rights review" -- "${F[@]}"
```
List the existing tests you changed with "the one allowed change" in the commit body.

---

## G10. The three trainers on written people refuse, and the decision export skips them

### G10a. `dclab_rnd/intent_classifier.py`, function `def train(`

Find `    rows = load_rows(requests)` inside `def train(` and add right after it:
```python
    from dclab_rnd.store import writers as who_wrote

    writer = who_wrote.set_writer(Path(requests).stem)  # a written request set: its words are its writer's
    if who_wrote.held_writer(writer):
        raise who_wrote.HeldForRights(f"{Path(requests).name}: its {len(rows)} requests were written by {writer}; a provider's "
                                      "output trains nothing until the owner records a rights review (2026-11-18). The fitted model is not touched.")
```
This runs before anything is fitted or written.

### G10b. `dclab_rnd/workflow_models.py`, function `def train(`

Find `    out = Path(out)` inside `def train(` and put these lines **right before** it:
```python
    from dclab_rnd.store import writers as who_wrote

    # every row comes from a written journey; a set a provider wrote trains nothing until the owner's rights review (2026-11-18)
    sources = Counter(str(r.get("source")) for point in (points or list(wl.POINTS)) for r in rows(point, sets) if r.get("label"))
    held = {s for s in sources if who_wrote.held_writer(who_wrote.set_writer(s))}
    if sources and held == set(sources):
        raise who_wrote.HeldForRights(f"every journey set these rows come from ({', '.join(sorted(held))}) was written by a provider's "
                                      "model; nothing is trained until the owner records a rights review. The fitted models are not touched.")
```
Then find, in the loop:
```python
        data = [r for r in rows(point, sets) if r.get("label")]
```
and replace it with:
```python
        data = [r for r in rows(point, sets) if r.get("label") and str(r.get("source")) not in held]
```

### G10c. `dclab_rnd/understand_students.py`, function `def train(`

Find `    out = Path(out)` inside `def train(` (the one followed by `out.mkdir`) and put these lines **right before** it:
```python
    from dclab_rnd.store import writers as who_wrote

    # the people are written journeys (Claude Sonnet, or a model whose requests were not kept): held until the rights review
    sources = Counter(str(r.get("source")) for point in (*_points(), "problem.family") for r in W.rows(point, sets) if r.get("label"))
    held = {s for s in sources if who_wrote.held_writer(who_wrote.set_writer(s))}
    if sources and held == set(sources):
        raise who_wrote.HeldForRights(f"every journey set these rows come from ({', '.join(sorted(held))}) was written by a provider's "
                                      "model; nothing is trained until the owner records a rights review. The fitted models are not touched.")
```
Then, in the loop, replace
```python
        data = [r for r in W.rows(point, sets) if r.get("label")]
```
with
```python
        data = [r for r in W.rows(point, sets) if r.get("label") and str(r.get("source")) not in held]
```
and replace
```python
    rows = W.rows("problem.family", sets)
```
with
```python
    rows = [r for r in W.rows("problem.family", sets) if str(r.get("source")) not in held]
```

### G10d. `dclab_rnd/decision_export.py`

1. Change the signature `def rows(sets: Any = SETS) -> list[dict[str, Any]]:` to
   `def rows(sets: Any = SETS, held: Any = None) -> list[dict[str, Any]]:`.
2. In `def rows(`, replace the line
   ```python
                out += [normalize(json.loads(line)) for line in handle if line.strip()]
   ```
   with
   ```python
                raw = [json.loads(line) for line in handle if line.strip()]
                if Path(root).parent.name == "workflow_v1":  # rows played from written journeys: a provider's people (2026-11-18)
                    from dclab_rnd.store import writers as who_wrote

                    kept = [r for r in raw if not who_wrote.held_writer(who_wrote.set_writer(r.get("source")))]
                    if held is not None:
                        held["workflow_v1: written journeys"] = held.get("workflow_v1: written journeys", 0) + len(raw) - len(kept)
                    raw = kept
                out += [normalize(r) for r in raw]
   ```
3. In `def build(`, replace `    data = rows(sets)` with:
   ```python
       held: dict[str, int] = {}
       data = rows(sets, held)
   ```
   and in its `manifest = {...}` dict add the key `"held_for_rights": held,` right after `"name": NAME,`.

The generated sets (`generated_v1`, `generated_v2`, `generated_images_v1`) are made by code and never held.

### G10e. Tests: new file `tests/test_trainers_hold_written_people.py`

```python
"""The owner, 2026-11-18: everything Claude wrote is held until the rights review, also the written people and requests
the intent classifier, the understanding students and the workflow models learn from. Each refuses before it fits or
writes anything; the decision export skips the journey rows and counts them."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOLD = "dclab_rnd.store.provider_outputs.may_train"


def journey_rows(folder, point, source, n=12):
    path = Path(folder) / "decisions" / point / "rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"record": f"{source}-{i}", "point": point, "split": "train" if i % 3 else "dev", "group": f"{source}:{i}", "source": source,
             "state": {"turns": "few"}, "options": ["yes", "no"], "value_used": "yes", "label": {"value": "yes", "accepted": ["yes"], "source": "benchmark"}}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class TrainersHoldTests(unittest.TestCase):
    def test_the_intent_classifier_refuses_requests_claude_wrote(self):
        from dclab_rnd import intent_classifier
        from dclab_rnd.store import writers

        folder = Path(tempfile.mkdtemp())
        requests = folder / "intent_requests_v1.json"
        requests.write_text(json.dumps([{"id": "r1", "text": "average sales per month", "track": "analysis", "writer": "a", "batch": "analysis-a-1",
                                         "model": "claude-sonnet-5-5"}]), encoding="utf-8")
        with mock.patch(HOLD, return_value=False):
            with self.assertRaises(writers.HeldForRights):
                intent_classifier.train(requests=requests, out=folder / "model")
        self.assertFalse((folder / "model").exists())  # nothing fitted, nothing written

    def test_the_workflow_models_refuse_rows_from_written_journeys(self):
        from dclab_rnd import workflow_models
        from dclab_rnd.store import writers

        sets, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "models"
        journey_rows(sets, "problem.ready", "sealed_v2")
        with mock.patch(HOLD, return_value=False):
            with self.assertRaises(writers.HeldForRights):
                workflow_models.train(points=["problem.ready"], sets=sets, out=out)
        self.assertFalse(out.exists())

    def test_the_understanding_students_refuse_rows_from_written_journeys(self):
        from dclab_rnd import understand_students
        from dclab_rnd.store import writers

        sets, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "models"
        journey_rows(sets, "problem.family", "homework_v1")
        with mock.patch(HOLD, return_value=False):
            with self.assertRaises(writers.HeldForRights):
                understand_students.train(sets=sets, out=out)
        self.assertFalse(out.exists())

    def test_the_decision_export_skips_journey_rows_and_counts_them(self):
        from dclab_rnd import decision_export

        root = Path(tempfile.mkdtemp())
        workflow = root / "workflow_v1"
        journey_rows(workflow, "problem.ready", "dev", n=3)
        held: dict = {}
        with mock.patch(HOLD, return_value=False), mock.patch("dclab_core.decision_export.normalize", side_effect=lambda r: r):
            got = decision_export.rows(workflow / "decisions", held)
        self.assertEqual(got, [])
        self.assertEqual(held, {"workflow_v1: written journeys": 3})


if __name__ == "__main__":
    unittest.main()
```
If `intent_classifier.train`, `workflow_models.train` or `understand_students.train` fails in these tests for another
reason **before** reaching the guard (for example an import or a missing file), move the guard earlier in that function
(still before anything is written) and say so. Never change an assertion.

Run:
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_trainers_hold_written_people.py test_intent_classifier.py test_understand_students.py test_workflow_models.py test_decision_export.py test_home_intent.py
```
Existing tests in those files that train on the real written sets will now raise `HeldForRights`: give each "the one
allowed change", and list them.

## G11. Measure after, and check nothing else moved

1. The audit again (its numbers must equal G3's: the audit reads the store, which this plan never changes):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.writers audit > /tmp/writers_audit_after.json 2>&1; tail -12 /tmp/writers_audit_after.json
   ```
2. The exports after (only under `/tmp`):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "from pathlib import Path; from dclab_rnd.store.sets import routing; m = routing.export(out=Path('/tmp/routing_after')); print(m['rows'], m['skipped'], len(m['chat']))"
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "from pathlib import Path; from dclab_rnd.lake import answers; m = answers.export_gliner2(out=Path('/tmp/gliner_after')); import json; print(json.dumps(m['counts'].get('held_for_rights', {})), len(m['counts'].get('files', {})))"
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.export_chat --model-type m4_workflow_llm --folder understand_problem --version v2 --out /tmp/chat_after > /dev/null 2>&1; .venv/bin/python -c "import json; e = json.load(open('/tmp/chat_after/MANIFEST.json'))['export']; print(e['counts'], e['held_for_rights'], e['answerer_not_recorded'])"
   ```
3. **Expected:**
   - Routing: `skipped` has `held until the owner's rights review: claude` and `...: frontier` (thousands in all); the
     chat folders hold only `real` classes; no `model_thinking` in `/tmp/routing_after/v1/table.csv` is filled:
     `.venv/bin/python -c "import csv; print(sum(1 for r in csv.DictReader(open('/tmp/routing_after/v1/table.csv')) if r['model_thinking']))"` prints `0`.
   - GLiNER2: `held_for_rights` names `routing`, `journeys` and `workflow`; the `lake`, `selfplay` and `committee`
     files are unchanged from G3 (compare their file counts in `/tmp/gliner_before/v1` and `/tmp/gliner_after/v1`:
     `find /tmp/gliner_before/v1/lake -name '*.jsonl' | wc -l` against the same for `after`, and the same for `selfplay`
     and `committee`).
   - Chat export of understand_problem v2: `held_for_rights` now also counts the test-workspace conversations under
     `claude`; exported rows drop by about the same number. `answerer_not_recorded` stays 0.
   - A number very different from these: do not change code; report it.
4. The exams are untouched: `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.exams verify` prints `"clean": true` for every type.
5. `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.data_catalog --check` must print `data catalog is fresh` (if
   stale: `DCLAB_NO_LIVE_MODELS=1 make catalog`, then `git diff --stat -- docs/DATA_CATALOG.md docs/data_catalog.json`
   lists only those two files; commit them in G12).
6. Delete the temporary folders you made: `rm -rf /tmp/routing_before /tmp/routing_after /tmp/gliner_before /tmp/gliner_after /tmp/chat_before /tmp/chat_after`.

## G12. CLAUDE.md, the record, the training set, the last commit, the push, the report

1. **CLAUDE.md:** find the text `(store/provider_outputs.py may_train, the owner 2026-11-18);` and replace it with
   `(store/provider_outputs.py may_train, the owner 2026-11-18; store/writers.py says who wrote every kind of row, the words of written people and requests included, and every trainer and export asks it, python -m dclab_rnd.store.writers audit);`
2. Create `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-provider-hold-all.json` in the
   same shape (same keys) as `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-model-call-rows.json`.
   Allowed values: a finding's `source` is one of `reviewer`, `test`, `measurement`, `owner`, `self`; `severity` one of
   `behaviour`, `latent`, `gap`, `flaky`, `privacy`, `leakage`; a measurement's `verdict` one of `pass`, `fail`,
   `not scored`; a measurement's `gate` must be at least 10 characters (write "none: a count before and after, no threshold fixed before").
   Content:
   - `request`: the owner asked to check the decision sets for Claude's outputs, and answered "Everything Claude wrote" on scope.
   - `decisions`: (a) one module decides who wrote a row, and every trainer and export asks it (why: seven places each had
     their own rule or none); (b) a written set is a provider's unless its records say otherwise (why: every hand-made
     journey and request file was committed with Claude as co-author); (c) a trainer refuses before writing anything (why:
     a fitted model is never overwritten by a run with nothing to learn from); (d) code-made data is not held (why: an
     invented table or a measured label is the code's output).
   - One finding per place in the Context table (source `reviewer`, severity `gap`), each with its scenario, cause, fix
     and test name; plus one finding for the test-workspace chat rows (G5).
   - A measurement with the G3 and G11 numbers, `verdict` `not scored`, `cost_eur` 0.0.
   - `risks`: the intent classifier, the understanding students and the workflow models cannot be retrained until the
     review; the lake has its own rules (Claude evaluation only) and was not changed; a test workspace's words are held
     even when a test replayed a real person's words.
3. `DCLAB_NO_LIVE_MODELS=1 make sft-dev` must end with a line like `N examples from M packages ...`.
4. Final test run (every file this plan touched):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_writers.py test_store_export_chat.py test_store_tutor.py test_store_routing.py test_lake_answers.py test_trainers_hold_written_people.py test_intent_classifier.py test_understand_students.py test_workflow_models.py test_decision_export.py test_home_intent.py test_store_sets_llm.py test_home_answerer.py test_store_cases.py test_provider_outputs.py
   ```
   It must end with `OK`.
5. Commit (paths: `dclab_rnd/intent_classifier.py`, `dclab_rnd/workflow_models.py`, `dclab_rnd/understand_students.py`,
   `dclab_rnd/decision_export.py`, `tests/test_trainers_hold_written_people.py`, every existing test file you changed in
   G10, `CLAUDE.md`, the record, the files `git status --short` shows under
   `research/llm-fine-tuning/experiments/sft/out_dev/`, and the two catalog files only if G11.5 rebuilt them). First line:
   `The trainers on written people refuse until the rights review, and the decision export skips them`.
6. Push to main and dev:
   ```bash
   C=$(git rev-parse HEAD); git push origin "${C}:refs/heads/main" "${C}:refs/heads/dev"; git branch -f dev HEAD
   ```
7. Report to the owner in plain words: what changed in each place; the G3 and G11 numbers side by side; every existing
   test you changed with "the one allowed change" (file and name); anything that did not go as planned; the three commit ids.

---

## Verification (how the owner knows it is done)

1. The final test run of G12.4 ends with `OK`.
2. `python -m dclab_rnd.store.writers audit` prints, per model type, how many rows something Claude wrote is in.
3. The routing export holds only `real` classes and no provider thinking; the GLiNER2 export counts `routing`, `journeys`
   and `workflow` under `held_for_rights`; the three trainers raise `HeldForRights` and write nothing.
4. `git show --stat` over the three commits lists only the files this plan names; nothing under `dclab_rnd/prompts/`,
   `dclab_core/` or `training_records/`.

## Not in this plan

- **The rights review is the owner's.** Setting `"rights_reviewed": true` in `dclab_rnd/store/owner_decisions.json`
  turns every guard off at once.
- The lake (`dclab_rnd/lake/__init__.py`, `harvest.py`, `train.py`) keeps its own rule (Claude's moves are evaluation
  only) and is not changed. Whether its solver rows from test workspaces carry written words was not checked.
- The decision sets' row states (bands code computed) are not held; only their teacher labels are. A table a model
  designed (`synthetic_schema`) whose rows code generated is treated as code's.
- No fitted model is deleted or retrained. Which fitted students were trained on held data is a separate question for
  the owner.
- m6, m7_judges and m8 have no trainer today; the audit counts them so a future trainer can use `who_wrote.held`.

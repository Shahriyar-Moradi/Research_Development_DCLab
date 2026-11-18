# Home conversations record which model wrote them (a plan a smaller model can execute)

## Context

Item 1 (2026-11-18) made every training export hold back provider outputs (Claude and other hosted models) until the
owner records a rights review. Its real export of `m4_workflow_llm/understand_problem/v2` showed a gap: **1,276 Home
conversation rows say nothing about which model wrote them.** The export cannot tell, so it exports them and only counts
them (`answerer_not_recorded: 1276`).

**Correction of an earlier guess:** these rows are mostly *not* Claude's. The planned code, run in memory on the real
store on 2026-11-18 (nothing written), names them so:

| Answerer the plan writes | Rows | Why |
|---|---|---|
| `rules` | 1,020 | no model request names the draft and nothing in it says a model wrote it |
| `frontier:...` (held) | 218 | a hosted model other than Qwen answered, or the draft shows a model ran (`agent.mode == "model"`, a turn `by` model or case_agent, `case_state`, a critic or advisory question) and its requests were never captured (mostly `live_tests` workspaces) |
| `claude:...` (held) | 32 | a recorded Claude request names the draft |
| `openrouter:qwen...` (kept) | 6 | a recorded Qwen request names the draft (the owner's yes of 2026-11-11) |

**Why it matters:** the owner's rule is "keep and prepare everything, train on no provider output until the rights
review". 250 rows of Claude or another hosted model pass the hold today. And 1,020 rows of DCLab's own rules are
counted as "unknown" when they are clean.

**The fix in one sentence:** when the store builds a Home conversation's example, it looks up the model requests made
for that draft and writes `check["answerer"]` (who wrote it), and the export reads that field first.

**Size:** about 70 lines of code, about 150 lines of tests, one set rebuild, one record. No prompt, tool, `dclab_core`
or Build project change.

**Checked before handing over:** the G1 code and the 7 tests of G4a were run in memory against today's code: all pass.

**Delivery (mine, after approval):** this plan is saved as `Dev_Plans/Home_Conversation_Answerer_2026-11-18/PLAN.md`
with a `PROMPT.md` for the small model, committed and pushed to main and dev. The small model then does G1 to G6.

---

## Rules for whoever executes this (read first; they apply to every task)

1. Work in `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`, branch `main`. Python is `.venv/bin/python`.
2. Do the tasks in order: G1 to G6. Finish one (its tests green) before the next. Commit once, in G6.
3. **Find a place by its quoted text** (`grep -n "text" file`), never by line number. Line numbers are hints only.
4. Run tests only like this (bare file names, never a bare `pytest`, never a live key):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_a.py test_b.py
   ```
   It must end with `Ran N tests ...` and then `OK`.
5. **Never run** `make check-commit`, `make check-quick`, `make check-all`, `make rd-check` or the whole suite.
6. A failing test: fix **the code**. Never delete, skip or weaken an assertion. If an **existing** test fails and you
   believe the new behaviour is right, stop and report it; do not change that test.
7. **Do not touch:** `dclab_rnd/prompts/`, any tool description, anything in `dclab_core/`, any `results/` folder, any
   `sealed.jsonl` (never open one), the exam files and `dclab_rnd/store/exam_pins.json`, and the files
   `cases_dev3.json`, `answer_key_dev3.json`, `cases_nodata_held.json`, `answer_key_nodata_held.json`,
   `cases_sealed.json` (never open them).
8. **Do not change** `exchange_answerer`, `answerer_of`, `silver_of`, `open_weights` or `clean` in
   `dclab_rnd/store/sets/llm.py`. Reuse them.
9. **Never write `"rules"` into `check["model"]`.** `exchange_answerer` would read it as a hosted model called "rules"
   (`frontier:rules`). Who wrote the row goes in `check["answerer"]`; `check["model"]` only ever holds a real model's name.
10. Git: never `git stash`, never `git add .` / `git add -A`. Stage and commit explicit paths only (zsh needs an array):
    ```bash
    F=(path/one path/two); git add -- "${F[@]}" && git commit -m "message" -- "${F[@]}"
    ```
    Never stage `data/workspace/` or `training_records/STORE_CATALOG.json`. Do not commit inside `training_records/store`
    (the store's nightly run commits its own manifests). End the commit message with the attribution line your session
    gives you.
11. Comments and messages: plain English, short sentences, no em dashes. A comment says **why**.
12. Before G5 run `df -h .`. If less than 5 GB is free, stop and report.

---

## G1. Find the models that answered for each draft

**File:** `dclab_rnd/store/sets/llm.py`. Add the code below **right after** the function `def home_thinking(` ends
(after its `return out`) and **before** `def home_turn(`.

```python
# Purposes whose words never reach a Home conversation: a label or a judge's verdict written afterwards, or a vector
NOT_IN_CONVERSATION = frozenset({"routing_labeler", "exam_judge", "embedding"})
MODEL_QUESTION_SOURCES = ("critic", "advisory", "second_opinion")  # a question a model asked, not the rules


def home_models(requests: Iterable[StoreRecord]) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    """(workspace, draft id) -> {answerer: its model, provider and source class} for every model that answered for that
    draft. Only a request that returned text counts (outcome ok), and never a shadow's. The same request is kept twice (in
    the request log and in its exchange, only the exchange naming the provider), so it is keyed by its answerer and the copy
    that names a provider wins."""
    out: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for r in requests:
        b = r.body
        draft = b.get("draft") or b.get("draft_id")
        model = str(b.get("model") or "").strip()
        if not draft or not model or b.get("outcome") != "ok" or b.get("shadow") or b.get("purpose") in NOT_IN_CONVERSATION:
            continue
        found = {"model": model, "provider": str(b.get("provider") or ""), "source_class": str(b.get("source_class") or "")}
        who = exchange_answerer(found)
        kept = out[(r.provenance.workspace, str(draft))]
        if who not in kept or (found["provider"] and not kept[who]["provider"]):
            kept[who] = found
    return dict(out)


def _model_signs(b: dict[str, Any]) -> bool:
    """The draft itself shows a model wrote some of it: the model ran the conversation, a turn says a model or the lead
    agent wrote it, the lead agent kept its state, or a model's critic or advisor asked a question."""
    if ((b.get("agent") or {}).get("mode")) == "model" or b.get("case_state"):
        return True
    for m in b.get("messages") or []:
        if isinstance(m, dict) and (m.get("by") in ("model", "case_agent")
                                    or ((m.get("question") or {}).get("source")) in MODEL_QUESTION_SOURCES):
            return True
    return any(isinstance(q, dict) and q.get("source") in MODEL_QUESTION_SOURCES for q in b.get("questions") or [])


def _rank(answerer: str) -> int:
    """The most restrictive answerer first: a provider's model, then the OpenRouter Qwen, then DCLab's own."""
    return 0 if silver_of(answerer) else 1 if open_weights(answerer) else 2


def home_answerer(b: dict[str, Any], found: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Who wrote a Home conversation, for its check. With recorded requests: every answerer, and the most restrictive one's
    model as ``model`` (so ``exchange_answerer`` and the reasoning sets read it as before). With none: a model's signs make it
    an unknown hosted model (held, the safe side); no sign makes it DCLab's rules."""
    if found:
        names = sorted(found, key=lambda a: (_rank(a), a))
        top = found[names[0]]
        out: dict[str, Any] = {"answerer": names[0] if len(names) == 1 else "mixed:" + "+".join(sorted(names)),
                               "models": sorted(names), "model": top["model"],
                               "answerer_basis": "the model requests recorded for this draft"}
        out.update({k: top[k] for k in ("provider", "source_class") if top[k]})
        return out
    if _model_signs(b):
        return {"answerer": "frontier:unknown", "answerer_basis": "a model wrote turns here and its requests were not captured"}
    return {"answerer": "rules", "answerer_basis": "no model request recorded for this draft and no turn says a model wrote it"}
```

**Done when:** `.venv/bin/python -c "from dclab_rnd.store.sets import llm; print(llm.home_answerer({}, {}))"` prints
`{'answerer': 'rules', 'answerer_basis': '...'}`.

---

## G2. Write the answerer into each Home conversation's check

**File:** `dclab_rnd/store/sets/llm.py`.

1. Change the signature line
   `def from_draft(record: StoreRecord, thinking: Optional[dict[str, str]] = None) -> Optional[ChatExample]:`
   to
   ```python
   def from_draft(record: StoreRecord, thinking: Optional[dict[str, str]] = None,
                  models: Optional[dict[tuple[str, str], dict[str, dict[str, str]]]] = None) -> Optional[ChatExample]:
   ```
   Add one sentence at the end of its docstring: `Its check says who wrote it (home_answerer), so the training export can hold a provider's conversation.`
2. Find the line
   `    check: dict[str, Any] = {"built": bool(b.get("project_id")), "track": track}`
   and add right after it:
   ```python
       # who wrote it: the training export holds a provider's conversation until the owner's rights review (2026-11-18)
       check.update(home_answerer(b, (models or {}).get((record.provenance.workspace, str(b.get("id") or "")), {})))
   ```
3. In `def build(`, find
   ```python
       thinking = home_thinking(shapers.records(layout, "sessions"), shapers.records(layout, "model_requests"))
       for r in newest.values():
           ex = from_draft(r, thinking)
   ```
   and replace it with
   ```python
       thinking = home_thinking(shapers.records(layout, "sessions"), shapers.records(layout, "model_requests"))
       models = home_models(shapers.records(layout, "model_requests"))  # every request is read, consented or not: only its model's name is used
       for r in newest.values():
           ex = from_draft(r, thinking, models)
   ```
   Change nothing else in `build`. The reasoning sets below already read `exchange_answerer(e.check)`: a Home row with a
   recorded Claude request may now enter `reasoning/silver_frontier` and a Qwen one the Qwen sets, but only when every
   turn has its recorded thinking (`all_recorded`). That is intended.

**Done when:** `grep -n "from_draft(r, thinking, models)" dclab_rnd/store/sets/llm.py` prints one line.

---

## G3. The export reads the answerer first

**File:** `dclab_rnd/store/export_chat.py`, function `def provider_of(`.

1. Find
   ```python
       if check.get("model"):
           who = exchange_answerer(check)
   ```
   and replace it with
   ```python
       if check.get("answerer"):  # a Home conversation says who wrote it (sets/llm.py home_answerer)
           who = str(check["answerer"])
       elif check.get("model"):
           who = exchange_answerer(check)
   ```
2. In its docstring, after `A chat example names its model in ``check``;` add `a Home conversation names its answerer there;`.

No other change. `silver_of` already reads `mixed:` and `frontier:unknown` (held) and `rules` (not held).

---

## G4. Tests

### G4a. New file `tests/test_home_answerer.py`

```python
"""The owner, 2026-11-18: no provider output trains until the rights review. A Home conversation named no model, so 1,276
rows passed the export's hold uncounted as anyone's. Each one now says who wrote it: the models whose requests name its
draft, an unknown hosted model when the draft shows a model ran, or DCLab's rules."""
import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AT = "2026-11-10T10:00:00+00:00"
IDS = itertools.count(1)
CLAUDE = {"model": "claude-sonnet-5-5"}
QWEN_LOG = {"model": "qwen/qwen3.8-flash"}  # the request log names no provider
QWEN_EXCHANGE = {"model": "qwen/qwen3.8-flash", "provider": "openrouter", "source_class": "open_weights_model"}


def request(draft, workspace="dev", purpose="home_agent", outcome="ok", **body):
    from dclab_core.store import Provenance, StoreRecord

    return StoreRecord(id=f"q{next(IDS)}", kind="model_requests", type="request", at=AT,
                       body={"draft": draft, "purpose": purpose, "outcome": outcome, **body}, split="train",
                       provenance=Provenance(workspace=workspace, source="model_requests.jsonl", entry="e:1", at=AT, sha256="0" * 64, consent="train"))


def draft(**extra):
    from dclab_core.store import Provenance, StoreRecord

    body = {"id": "d1", "intent": {"track": "prediction"}, "updated": AT,
            "messages": [{"role": "person", "text": "Which customers leave?"}, {"role": "agent", "text": "When is the prediction made?"}], **extra}
    return StoreRecord(id="r1", kind="conversations", type="draft", at=AT, body=body, split="train",
                       provenance=Provenance(workspace="dev", source="drafts/d1/draft.json", entry="e:d1", at=AT, sha256="0" * 64, consent="train", draft="d1"))


class HomeAnswererTests(unittest.TestCase):
    def check(self, *requests, **extra):
        from dclab_rnd.store.sets import llm

        return llm.from_draft(draft(**extra), {}, llm.home_models(requests)).check

    def test_a_claude_request_for_the_draft_names_claude(self):
        c = self.check(request("d1", **CLAUDE))
        self.assertEqual((c["answerer"], c["model"], c["models"]), ("claude:claude-sonnet-5-5", "claude-sonnet-5-5", ["claude:claude-sonnet-5-5"]))

    def test_the_log_and_the_exchange_of_one_qwen_request_are_one_answerer(self):
        c = self.check(request("d1", **QWEN_LOG), request("d1", **QWEN_EXCHANGE))
        self.assertEqual(c["answerer"], "openrouter:qwen/qwen3.8-flash")
        self.assertEqual((c["provider"], c["source_class"]), ("openrouter", "open_weights_model"))  # the copy naming its provider wins

    def test_claude_and_qwen_on_one_draft_are_mixed_and_claude_is_its_model(self):
        c = self.check(request("d1", **QWEN_EXCHANGE), request("d1", purpose="case_agent", **CLAUDE))
        self.assertEqual(c["answerer"], "mixed:claude:claude-sonnet-5-5+openrouter:qwen/qwen3.8-flash")
        self.assertEqual(c["model"], "claude-sonnet-5-5")  # the most restrictive, so the reasoning sets treat it as Claude's

    def test_requests_that_wrote_no_turn_do_not_count(self):
        c = self.check(request("d1", outcome="the provider refused the key", **CLAUDE), request("d1", shadow=True, **CLAUDE),
                       request("d1", purpose="routing_labeler", **CLAUDE), request("d1", purpose="exam_judge", **CLAUDE),
                       request("d1", workspace="other", **CLAUDE), request("d2", **CLAUDE))
        self.assertEqual(c["answerer"], "rules")
        self.assertNotIn("model", c)

    def test_a_draft_that_shows_a_model_ran_without_its_requests_is_an_unknown_model(self):
        signs = [{"agent": {"mode": "model"}}, {"case_state": {"stage": "problem"}},
                 {"messages": [{"role": "person", "text": "Hi"}, {"role": "agent", "by": "case_agent", "text": "Hello"}]},
                 {"messages": [{"role": "person", "text": "Hi"}, {"role": "agent", "by": "model", "text": "Hello"}]},
                 {"questions": [{"id": "q1", "source": "critic"}]},
                 {"messages": [{"role": "person", "text": "Hi"}, {"role": "agent", "text": "Why?", "question": {"id": "q2", "source": "advisory"}}]}]
        for extra in signs:
            c = self.check(**extra)
            self.assertEqual(c["answerer"], "frontier:unknown", extra)
            self.assertNotIn("model", c)

    def test_a_rules_conversation_is_the_rules_and_never_a_model_called_rules(self):
        c = self.check(messages=[{"role": "person", "text": "Hi"}, {"role": "agent", "by": "code", "text": "Hello"}])
        self.assertEqual(c["answerer"], "rules")
        self.assertNotIn("model", c)
        self.assertIn("answerer_basis", c)  # the owner can read why it was judged the rules'

    def test_the_export_holds_claude_and_the_unknown_model_and_keeps_qwen_and_the_rules(self):
        from dclab_rnd.store.export_chat import provider_of

        held = lambda *r, **e: provider_of({"check": self.check(*r, **e)})
        self.assertEqual(held(request("d1", **CLAUDE)), ("claude:claude-sonnet-5-5", True))
        self.assertEqual(held(agent={"mode": "model"}), ("frontier:unknown", True))
        self.assertEqual(held(request("d1", **QWEN_EXCHANGE), request("d1", **CLAUDE))[0],
                         "mixed:claude:claude-sonnet-5-5+openrouter:qwen/qwen3.8-flash")
        self.assertEqual(held(request("d1", **QWEN_LOG)), (None, True))  # the owner's yes of 2026-11-11
        self.assertEqual(held(), (None, True))  # DCLab's rules: recorded, and not held


if __name__ == "__main__":
    unittest.main()
```

If `StoreRecord` refuses `kind="model_requests"` or `type="request"`, print the error, use the kind and type that
`shapers.records(Layout(), "model_requests")` gives (the real records use kind `model_requests`, type `request`), and say so.

### G4b. One end-to-end test in `tests/test_store_sets_llm.py`, class `LlmSetsTest`

Add it right after the test `test_a_question_in_a_model_run_conversation_is_never_credited_to_the_rules`:

```python
    def test_a_home_conversation_says_who_wrote_it_and_the_export_holds_claudes(self):
        """2026-11-18: 1,276 Home rows named no model, so the training export could not hold Claude's. The draft's own
        requests now name it, through the store's whole path: kept log, shaped records, the set, the export."""
        from dclab_core.store import ChatExample
        from dclab_rnd.store import Store, curators, export_chat, provider_outputs, shapers
        from dclab_rnd.store.sets import llm

        store = Store(self.layout)
        asked = {"at": "2026-11-10T10:00:00+00:00", "model_request": "mr20", "purpose": "home_agent", "outcome": "ok", "whole": True,
                 "model": "claude-sonnet-5-5", "draft": "d3", "messages": [{"role": "user", "content": "Churn next month?"}],
                 "reply": "When is it predicted?", "thinking": []}
        refused = {**asked, "model_request": "mr21", "model": "qwen/qwen3.8-flash", "outcome": "the provider refused the key"}
        store.keep_log("model_requests", "dev", "model_exchanges/2026/11/10.jsonl",
                       (json.dumps(asked) + "\n" + json.dumps(refused) + "\n").encode(), consent="train")
        d3 = {"id": "d3", "created": "2026-11-10T09:00:00+00:00", "intent": {"track": "analysis"},
              "messages": [{"role": "user", "text": "Churn next month?"}, {"role": "agent", "by": "model", "text": "When is it predicted?"}]}
        store.keep("conversations", "dev", "drafts/d3/draft.json", json.dumps(d3).encode(), consent="train")
        shapers.shape(self.layout)
        curators.curate(self.layout, run="r3")
        with mock.patch.object(llm, "BENCHMARKS", self.benchmarks):
            llm.build(self.layout)
        convo = next(ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "home_analysis", "v1")
                     if e["provenance"]["draft"] == "d3")
        self.assertEqual((convo.check["answerer"], convo.check["model"]), ("claude:claude-sonnet-5-5", "claude-sonnet-5-5"))
        rules = next(ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "understand_problem", "v2")
                     if e["provenance"]["draft"] == "d1")  # setUp's draft: no request, no sign of a model
        self.assertEqual(rules.check["answerer"], "rules")
        decisions = Path(tempfile.mkdtemp()) / "owner_decisions.json"
        decisions.write_text(json.dumps({"provider_outputs": {"for_training": True, "rights_reviewed": False}}), encoding="utf-8")
        with mock.patch.object(provider_outputs, "DECISIONS", decisions):
            report = export_chat.export(self.layout, "m4_workflow_llm", "home_analysis", "v1", Path(tempfile.mkdtemp()) / "out")
        self.assertEqual(report["held_for_rights"].get("claude"), 1)
        self.assertEqual(report["answerer_not_recorded"], 0)
```

`understand_problem` is written as `v2` and the `home_*` folders as `v1` (`build`, the `versions = {...}` line). If
setUp's draft `d1` is not in `understand_problem/v2` (it has no track, so it is a prediction), print the folders under
`self.layout.sets("m4_workflow_llm")` and report.

If the draft key `"draft"` of the kept exchange does not reach the shaped record (the conversation's check has no
`model`), print `[r.body.get("draft") for r in shapers.records(self.layout, "model_requests")]` and stop: do not change the
shaper.

### G4c. Run

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_home_answerer.py test_store_sets_llm.py test_core_track_outcome.py test_store_export_chat.py test_provider_outputs.py test_conversation_exam.py test_core_reporter.py test_store_exams.py
```

**Done when:** `OK`; the 7 tests of G4a and the test of G4b pass; no existing test changed.

---

## G5. Rebuild the sets and prove it on the real store

1. `df -h .` must show at least 5 GB free; `memory_pressure | tail -1` must show a free percentage of 10% or more.
   Nothing else heavy may run (check `pgrep -fl "run_parallel|make check|dclab_rnd.store" | head`; if a store run or a
   gate is running, wait for it to end).
2. Read the numbers before (write them down for the report):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.export_chat --model-type m4_workflow_llm --folder understand_problem --version v2 --out /tmp/home_before > /tmp/home_before.log 2>&1; tail -5 /tmp/home_before.log
   ```
3. Rebuild every M4 and M5 set, **in the background** (your Bash tool's background option), and wait for it to end. It
   can take a long time. If memory pressure turns red or the disk drops under 3 GB, stop it (kill the process) and report.
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "import json; from dclab_rnd.store.sets import llm; print(json.dumps(llm.build(), default=str)[:4000])" > /tmp/home_build.log 2>&1; echo "rc=$?"
   ```
   `rc=0` is required. If it fails, report the last 30 lines of `/tmp/home_build.log`; do not fix the builder.
4. Read the numbers after:
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.export_chat --model-type m4_workflow_llm --folder understand_problem --version v2 --out /tmp/home_after > /tmp/home_after.log 2>&1; tail -5 /tmp/home_after.log
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "
   import json, collections
   from dclab_rnd.store.layout import Layout
   base = Layout().sets('m4_workflow_llm', 'understand_problem', 'v2')
   c = collections.Counter()
   for s in ('train', 'dev'):
       p = base / f'{s}.jsonl'
       for line in (p.read_text(encoding='utf-8').split(chr(10)) if p.exists() else []):
           if line.strip():
               row = json.loads(line)
               if row.get('task') == 'home_conversation':
                   c[(row.get('check') or {}).get('answerer', 'NONE').split(':', 1)[0]] += 1
   print(dict(c))"
   ```
   (This reads `train.jsonl` and `dev.jsonl` only, never `sealed.jsonl`.)
5. **Expected** (the store gains new conversations every day, so the totals may move a little):
   - `answerer_not_recorded` drops from 1,276 to **0**. If it is above 0, print the `task` of 3 such rows and report.
   - Home rows by answerer (measured in memory before the change, 1,276 rows): `rules` 1,020; `frontier` 218; `claude`
     32; `openrouter` 6; `NONE` **0**.
   - `held_for_rights` grows by about 250 (`claude` by about 32, a new `frontier` of about 218).
   - If a count is outside these ranges by more than 10%, do not change code: report both numbers.
6. The exams are untouched:
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.exams verify
   ```
   Every drawn type prints `"clean": true`.
7. `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.data_catalog --check` must print `data catalog is fresh`. If
   stale: `DCLAB_NO_LIVE_MODELS=1 make catalog`, then `git diff --stat -- docs/DATA_CATALOG.md docs/data_catalog.json`
   must list only those two files; add them to the G6 commit.

---

## G6. The record, the training set, the commit, the push, the report

1. Create `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-home-answerer.json` in the
   same shape (same keys) as `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-item1.json`.
   Allowed values: a finding's `source` is one of `reviewer`, `test`, `measurement`, `owner`, `self`; `severity` one of
   `behaviour`, `latent`, `gap`, `flaky`, `privacy`, `leakage`; a measurement's `verdict` one of `pass`, `fail`,
   `not scored`. Content:
   - Finding HA-1: `source` `measurement`, `severity` `gap`. Defect: Home conversations named no model, so the export
     could not hold a provider's. Scenario: the export of understand_problem v2 counted 1,276 rows as
     `answerer_not_recorded` and exported them, 250 of them written by Claude (32) or another hosted model (218). Cause: `from_draft` never read the model
     requests made for the draft. Fix: `home_models` and `home_answerer`, `check["answerer"]` read first by `provider_of`.
     Test: the G4a and G4b test names.
   - Finding HA-2: `source` `self`, `severity` `gap`. Defect: many model-run conversations have no captured
     request at all. Fix: held as `frontier:unknown` (the safe side). Test: `test_a_draft_that_shows_a_model_ran_without_its_requests_is_an_unknown_model`.
   - Measurement: "The real export of understand_problem v2 before and after the rebuild", `gate` "none",
     `fixed_before_run` false, `result` the before and after numbers of G5 (exported, held by kind, not recorded, Home
     rows by answerer), `verdict` `not scored`, `cost_eur` 0.0.
   - Risks: a request with outcome ok may not have reached a turn, so a few rows may be held that need not be; requests
     with no draft id cannot be joined; `understand_problem/v1` (291 old rows) is never rebuilt.
2. `DCLAB_NO_LIVE_MODELS=1 make sft-dev` must end with a line like `N examples from M packages ...`.
3. Final test run (the G4c command again). It must end with `OK`.
4. Commit (paths: `dclab_rnd/store/sets/llm.py`, `dclab_rnd/store/export_chat.py`, `tests/test_home_answerer.py`,
   `tests/test_store_sets_llm.py`, the record, the files `git status --short` shows under
   `research/llm-fine-tuning/experiments/sft/out_dev/`, and the two catalog files only if G5.7 rebuilt them). First line:
   `A Home conversation says which model wrote it, so the training export can hold a provider's`.
   Put the before and after numbers of G5 in the body.
5. Push to main and dev:
   ```bash
   C=$(git rev-parse HEAD); git push origin "${C}:refs/heads/main" "${C}:refs/heads/dev"; git branch -f dev HEAD
   ```
6. Report to the owner in plain words: what changed, the test counts of each run, the before and after numbers, how
   long the rebuild took, anything that did not go as planned, and the commit id.

---

## Verification (how the owner knows it is done)

1. **Tests:** the G4c run ends with `OK`, including the 7 tests of `tests/test_home_answerer.py` and
   `test_a_home_conversation_says_who_wrote_it_and_the_export_holds_claudes`.
2. **Data:** the real export of understand_problem v2 prints `answerer_not_recorded: 0`; its `held_for_rights` names
   `claude` and `frontier`; the Home rows count by answerer shows no `NONE`.
3. **Exams:** `python -m dclab_rnd.store.exams verify` prints `"clean": true` for every drawn type.
4. **Nothing forbidden changed:** `git show --stat HEAD` lists only the files named in G6.4; nothing under
   `dclab_rnd/prompts/`, `dclab_core/` or `training_records/`.

## Not in this plan (said so nobody expects it)

- **The rights review is the owner's.** Until `"rights_reviewed": true` is set in `dclab_rnd/store/owner_decisions.json`,
  every held row stays out of training; it stays kept and prepared.
- `understand_problem/v1` (291 rows of an old set version) is never rebuilt and keeps `answerer_not_recorded`.
- The case-file examples of drafts with a case (`dclab_rnd/store/sets/cases.py`) were not checked for the same gap. If
  their export shows `answerer_not_recorded`, that is a separate small fix.
- Requests that name no draft (about 6,760, mostly October in the `dev` workspace) cannot be joined to a conversation.
  The model-signs rule covers the conversations they wrote when the draft shows a model ran.
- Capturing the missing requests of the `live_tests` workspaces is not possible after the fact; new runs capture them.

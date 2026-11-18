# Case files and old Home rows say who wrote them (a plan a smaller model can execute)

## Context

Commit 890b2dcc made every rebuilt Home conversation say who wrote it (`check["answerer"]`), so the training export
holds back Claude's and other hosted models' rows until the owner's rights review. The review of that work left two gaps:

| # | Gap (measured on the real store, 2026-11-18, train and dev rows only) | Fix |
|---|---|---|
| 2 | `m4_workflow_llm/understand_problem/v1` holds 291 Home conversations from before the writer was recorded. That version is never rebuilt, so they name no model, and the export sends them out. 263 of them are older copies of conversations already in v2, where 58 of those are Claude's or another hosted model's. | The export holds any Home conversation or case file that names no writer, as an unknown model's (`frontier:unknown`), and counts it as not recorded. No stored file is edited. |
| 3 | `m4_workflow_llm/case_files/v1` (289 rows: 160 from people's drafts, 129 from conversation exams) names no model. The conversation inside each one was written by the lead agent (a model), and in an exam also by a model playing the person. The export sends all 289 out. | The case-file builder looks up the model requests of the draft (or of the exam) and writes `check["answerer"]`, exactly as Home conversations now do. |

**Checked before handing over** (the planned code run in memory on today's code and the real store; nothing written):
- All 289 case-file rows get a writer: Claude 84, `mixed` 144 (every mixed one includes Claude: for example an exam's
  Claude Haiku person with a Claude Sonnet agent), Qwen 61. So **228 held, 61 kept**.
- The 4 new case tests, the new export test, and the existing case and export-gate tests all pass.

**Size:** about 40 lines of code, about 110 lines of tests, one set rebuild, one record. No prompt, tool, `dclab_core`
or Build project change.

---

## Rules for whoever executes this (read first; they apply to every task)

1. Work in `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`, branch `main`. Python is `.venv/bin/python`.
2. Do the tasks in order: H1 to H6. Finish one (its tests green) before the next. Commit once, in H6.
3. **Find a place by its quoted text** (`grep -n "text" file`), never by line number.
4. Run tests only like this (bare file names, never a bare `pytest`, never a live key):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_a.py test_b.py
   ```
   It must end with `Ran N tests ...` and then `OK`.
5. **Never run** `make check-commit`, `make check-quick`, `make check-all`, `make rd-check` or the whole suite.
6. A failing test: fix **the code**. Never delete, skip or weaken an assertion. If an **existing** test fails and you
   believe the new behaviour is right, stop and report it. **One exception, on purpose:** H2 changes the signature of
   the spy function in `test_case_records_are_streamed_not_listed` so it accepts the new argument. Change nothing else
   in that test.
7. **Do not touch:** `dclab_rnd/prompts/`, any tool description, anything in `dclab_core/`, any `results/` folder, any
   `sealed.jsonl` (never open one), the exam files and `dclab_rnd/store/exam_pins.json`, any file under
   `training_records/store/sets/` by hand, and the files `cases_dev3.json`, `answer_key_dev3.json`,
   `cases_nodata_held.json`, `answer_key_nodata_held.json`, `cases_sealed.json` (never open them).
8. **Do not change** `home_models`, `home_answerer`, `exchange_answerer`, `answerer_of`, `silver_of`, `open_weights` or
   `clean` in `dclab_rnd/store/sets/llm.py`. Reuse them.
9. **Never write `"rules"` into `check["model"]`.**
10. Git: never `git stash`, never `git add .` / `git add -A`. Stage and commit explicit paths only (zsh needs an array):
    ```bash
    F=(path/one path/two); git add -- "${F[@]}" && git commit -m "message" -- "${F[@]}"
    ```
    Never stage `data/workspace/` or `training_records/STORE_CATALOG.json`. Do not commit inside `training_records/store`.
    End the commit message with the attribution line your session gives you.
11. Comments and messages: plain English, short sentences, no em dashes. A comment says **why**.
12. Before H5 run `df -h .`. If less than 5 GB is free, stop and report.

---

## H1. The case-file builder says who wrote each case

**File:** `dclab_rnd/store/sets/cases.py`. Six edits.

1. Find `import json` (near the top) and add `import re` on the next line.
2. Find the line that starts `NO_REASONING = "the case file is built by code` and add right after it:
   ```python
   EXAM_ID = re.compile(r"cex-(\d+)", re.IGNORECASE)  # the exam's folder or workspace names it: CEX-012, cex-012
   ```
3. In `def example(`, change the second line of its signature from
   `            split_key: str) -> Optional[ChatExample]:`
   to
   `            split_key: str, writer: Optional[dict[str, Any]] = None) -> Optional[ChatExample]:`
   and change
   `                           check={"origin": origin, "case_version": case.get("version")})`
   to
   `                           check={"origin": origin, "case_version": case.get("version"), **(writer or {})})`
4. Replace these two lines
   ```python
   def from_records(records: Iterable[Any]) -> list[ChatExample]:
       """Exam conversations (their kept conversation.json, with the case file in its notes) and drafts with a case."""
       out = []
   ```
   with
   ```python
   def exam_writers(models: dict[tuple[str, str], dict[str, dict[str, str]]]) -> dict[str, dict[str, dict[str, str]]]:
       """An exam's requests by their draft name alone (``exam:CEX-012:case-d-01`` for the agent, ``exam:CEX-012:person:case-d-01``
       for its written person): the exam names itself, so the workspace its requests were kept under does not matter."""
       out: dict[str, dict[str, dict[str, str]]] = {}
       for (_, draft), found in models.items():
           if str(draft).startswith("exam:"):
               out.setdefault(str(draft), {}).update(found)
       return out


   def from_records(records: Iterable[Any], models: Optional[dict[tuple[str, str], dict[str, dict[str, str]]]] = None) -> list[ChatExample]:
       """Exam conversations (their kept conversation.json, with the case file in its notes) and drafts with a case. Each says
       who wrote it (``sets.llm.home_answerer``) from the model requests of its draft or its exam (``models``, from
       ``sets.llm.home_models``), so the training export can hold a provider's (the owner, 2026-11-18)."""
       from .llm import home_answerer  # here, not at the top: llm imports this module inside its build

       models = models or {}
       exams = exam_writers(models)
       out = []
   ```
5. Replace the line
   ```python
               ex = example(r.id, b.get("transcript") or [], b["notes"]["case_file"], r.provenance, "simulated", f"exam:{b.get('id')}")
   ```
   with
   ```python
               found: dict[str, dict[str, str]] = {}
               m = EXAM_ID.search(where)
               arm = str(b.get("arm") or "case")
               for key in (f"exam:CEX-{m.group(1)}:{arm}-{b.get('id')}", f"exam:CEX-{m.group(1)}:person:{arm}-{b.get('id')}") if m else ():
                   found.update(exams.get(key, {}))
               # an exam's person and agent are both models: with no request found, it is an unknown model's, never the rules'
               writer = home_answerer({"agent": {"mode": "model"}}, found)
               ex = example(r.id, b.get("transcript") or [], b["notes"]["case_file"], r.provenance, "simulated", f"exam:{b.get('id')}", writer)
   ```
6. Replace the line
   ```python
               ex = example(r.id, b["case_state"].get("turns") or [], b["case_state"]["case"], r.provenance, "real", f"draft:{b.get('id')}")
   ```
   with
   ```python
               writer = home_answerer(b, models.get((str(r.provenance.workspace), str(b.get("id") or "")), {}))
               ex = example(r.id, b["case_state"].get("turns") or [], b["case_state"]["case"], r.provenance, "real", f"draft:{b.get('id')}", writer)
   ```
7. Change `__all__ = ["FOLDER", "example", "from_records"]` to
   `__all__ = ["FOLDER", "example", "exam_writers", "from_records"]`.

Why the judge is not counted: `home_models` already skips `exam_judge` and `routing_labeler` (their words are written
after the conversation). The exam's person (`exam_person`) is counted: its words are the person's turns.

**Done when:** `.venv/bin/python -c "from dclab_rnd.store.sets import cases; print(cases.from_records([]))"` prints `[]`.

---

## H2. The builder passes the models to the case builder

**File:** `dclab_rnd/store/sets/llm.py`, function `def build(`.

1. Find these lines:
   ```python
       found_cases = (r for kind in ("tests", "conversations", "other") for r in shapers.records(layout, kind)  # an exam's conversation is kept as "other"
                      if allowed(r, "m4_workflow_llm", choices))
       for ex in cases.from_records(found_cases):
   ```
   Replace them with:
   ```python
       # who wrote each conversation, read once: the case files below and the Home conversations after them both need it
       models = home_models(shapers.records(layout, "model_requests"))  # every request is read, consented or not: only its model's name is used
       found_cases = (r for kind in ("tests", "conversations", "other") for r in shapers.records(layout, kind)  # an exam's conversation is kept as "other"
                      if allowed(r, "m4_workflow_llm", choices))
       for ex in cases.from_records(found_cases, models):
   ```
2. Further down, **delete** this line (it is now computed above; keep the `from_draft(r, thinking, models)` call as it is):
   ```python
       models = home_models(shapers.records(layout, "model_requests"))  # every request is read, consented or not: only its model's name is used
   ```
   Check: `grep -n "models = home_models" dclab_rnd/store/sets/llm.py` prints exactly one line, and it comes before
   `for ex in cases.from_records(found_cases, models):`.
3. **File:** `tests/test_store_sets_llm.py`, test `test_case_records_are_streamed_not_listed`. Change only the spy:
   ```python
           def spy(records):
               seen.append(isinstance(records, list))
               return real(records)
   ```
   becomes
   ```python
           def spy(records, models=None):
               seen.append(isinstance(records, list))
               return real(records, models)
   ```
   Its assertion `self.assertEqual(seen, [False])` stays exactly as it is.

---

## H3. The export holds a Home conversation or case file that names no writer

**File:** `dclab_rnd/store/export_chat.py`.

1. Right before `def provider_of(`, add:
   ```python
   # Rows a model may have written. One that names no writer is from before the writer was recorded (understand_problem v1
   # is never rebuilt): it is held as an unknown model's until the owner's rights review, and counted as not recorded.
   WRITER_REQUIRED = frozenset({"home_conversation", "case_file"})
   ```
2. In `def provider_of(`, find the two lines
   ```python
       else:
           return None, False
   ```
   and replace them with
   ```python
       else:
           return ("frontier:unknown" if row.get("task") in WRITER_REQUIRED else None), False
   ```
   No other change. A row of any other task that names no writer is exported and counted, as before.

---

## H4. Tests

### H4a. `tests/test_store_cases.py`: add this class after the class `CaseSetTests` (before `if __name__`)

```python
class CaseWriterTests(unittest.TestCase):
    """The owner, 2026-11-18: no provider output trains until the rights review. The 289 case files named no model, so the
    export could not hold Claude's. Each now says who wrote it, from its draft's or its exam's model requests."""

    def models(self, *rows):
        from dclab_rnd.store.sets import llm

        return llm.home_models(SimpleNamespace(provenance=SimpleNamespace(workspace=ws), body={"draft": d, "purpose": p, "outcome": "ok", **m})
                               for ws, d, p, m in rows)

    def exam(self):
        body = {"id": "d-40", "arm": "case", "transcript": [{"role": "person", "text": "we lose members"}, {"role": "assistant", "text": "Who acts?"}],
                "notes": {"case_file": CASE}}
        return SimpleNamespace(id="other:x#0", kind="other", body=body,
                               provenance=Provenance(source="conversation.json", **{**PROV, "workspace": "live_tests__2026-10-30-conversation-exam__cex-012__case__d-40"}))

    def draft(self):
        return record("r2", "drafts/abc/draft.json", {"id": "abc", "case_state": {"case": CASE, "turns": [
            {"role": "user", "content": "we lose members"}, {"role": "assistant", "content": "Who acts?"}]}})

    def test_an_exam_case_names_its_agent_and_its_written_person_but_not_its_judge(self):
        models = self.models(("dev", "exam:CEX-012:case-d-40", "case_agent", {"model": "qwen/qwen3.8-flash"}),
                             ("dev", "exam:CEX-012:person:case-d-40", "exam_person", {"model": "claude-sonnet-5-5"}),
                             ("dev", "exam:CEX-012:case-d-40", "exam_judge", {"model": "claude-opus-5-5"}))
        [ex] = cases.from_records([self.exam()], models)
        self.assertEqual(ex.check["origin"], "simulated")
        self.assertEqual(ex.check["answerer"], "mixed:claude:claude-sonnet-5-5+openrouter:qwen/qwen3.8-flash")
        self.assertEqual(ex.check["model"], "claude-sonnet-5-5")  # the judge's verdict is not in the conversation

    def test_a_drafts_case_names_the_model_of_its_own_draft_in_its_own_workspace(self):
        models = self.models(("w", "abc", "case_agent", {"model": "claude-sonnet-5-5"}),
                             ("other", "abc", "case_agent", {"model": "qwen/qwen3.8-flash"}))
        [ex] = cases.from_records([self.draft()], models)
        self.assertEqual((ex.check["origin"], ex.check["answerer"]), ("real", "claude:claude-sonnet-5-5"))

    def test_with_no_request_found_a_case_is_an_unknown_models_never_the_rules(self):
        out = cases.from_records([self.exam(), self.draft()])
        self.assertEqual([e.check["answerer"] for e in out], ["frontier:unknown", "frontier:unknown"])
        self.assertTrue(all("model" not in e.check for e in out))

    def test_the_export_holds_a_claude_case_and_keeps_a_qwen_case(self):
        from dclab_rnd.store.export_chat import provider_of

        [held] = cases.from_records([self.draft()], self.models(("w", "abc", "case_agent", {"model": "claude-sonnet-5-5"})))
        [kept] = cases.from_records([self.draft()], self.models(("w", "abc", "case_agent", {"model": "qwen/qwen3.8-flash"})))
        self.assertEqual(provider_of(held.model_dump(mode="json")), ("claude:claude-sonnet-5-5", True))
        self.assertEqual(provider_of(kept.model_dump(mode="json")), (None, True))  # the owner's yes of 2026-11-11
```

### H4b. `tests/test_store_export_chat.py`: add this test inside the class `ProviderGateTests` (as its last method, before `if __name__`)

```python
    def test_a_conversation_from_before_its_writer_was_recorded_is_held_as_an_unknown_models(self):
        """2026-11-18: understand_problem v1 keeps 291 Home conversations from before the writer was recorded, and it is never
        rebuilt. A model may have written them, so they are held, and counted as not recorded. Other tasks are unchanged."""
        from dclab_rnd.store import export_chat
        from dclab_rnd.store.sets import write_set

        rows = [chat("old_home", "train").model_copy(update={"task": "home_conversation"}),
                chat("old_case", "train").model_copy(update={"task": "case_file"}),
                chat("other_task", "train")]
        write_set(self.layout, "m4_workflow_llm", "chat/old", rows, version="v1", about="from before the writer was recorded")
        with self.decisions(False):
            report = export_chat.export(self.layout, "m4_workflow_llm", "chat/old", "v1", self.folder / "old")
        self.assertEqual(self.ids(self.folder / "old"), ["other_task"])
        self.assertEqual(report["held_for_rights"], {"frontier": 2})
        self.assertEqual(report["answerer_not_recorded"], 3)
```

Indent it with 4 spaces, like the other methods of the class. Make sure it is **inside** the class: run the file and
check its name is in the output of `--verbose` or that the test count of the file grew by 1.

### H4c. `tests/test_store_sets_llm.py`: one end-to-end test in the class `LlmSetsTest`

Add it right after the test `test_a_home_conversation_says_who_wrote_it_and_the_export_holds_claudes`:

```python
    def test_a_case_file_says_who_wrote_it_through_the_whole_build(self):
        """2026-11-18: the 289 case files named no model. The build now hands the case builder the draft's model requests."""
        from dclab_core.store import ChatExample
        from dclab_rnd.store import Store, curators, shapers
        from dclab_rnd.store.sets import llm

        store = Store(self.layout)
        asked = {"at": "2026-11-10T10:00:00+00:00", "model_request": "mr30", "purpose": "case_agent", "outcome": "ok", "whole": True,
                 "model": "claude-sonnet-5-5", "draft": "d4", "messages": [{"role": "user", "content": "We lose members."}],
                 "reply": "Who acts on the list?", "thinking": []}
        store.keep_log("model_requests", "dev", "model_exchanges/2026/11/10.jsonl", (json.dumps(asked) + "\n").encode(), consent="train")
        case = {"version": 4, "goal": {"value": "keep members", "status": "said", "quote": "keep members"},
                "route": {"track": "prediction", "tool": "ml_project"}}
        d4 = {"id": "d4", "created": "2026-11-10T09:00:00+00:00",
              "messages": [{"role": "user", "text": "We lose members."}, {"role": "agent", "by": "case_agent", "text": "Who acts on the list?"}],
              "case_state": {"case": case, "turns": [{"role": "user", "content": "We lose members."},
                                                     {"role": "assistant", "content": "Who acts on the list?"}]}}
        store.keep("conversations", "dev", "drafts/d4/draft.json", json.dumps(d4).encode(), consent="train")
        shapers.shape(self.layout)
        curators.curate(self.layout, run="r4")
        with mock.patch.object(llm, "BENCHMARKS", self.benchmarks):
            llm.build(self.layout)
        found = [ChatExample.model_validate(e) for e in self.examples("m4_workflow_llm", "case_files", "v1") if e["check"].get("origin") == "real"]
        self.assertEqual([(e.check["answerer"], e.check["model"]) for e in found], [("claude:claude-sonnet-5-5", "claude-sonnet-5-5")])
```

If `found` is empty, print the folders under `self.layout.sets("m4_workflow_llm")` and the build's `kept_out` report,
and stop: do not change the builder's filters.

### H4d. Run

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_cases.py test_store_export_chat.py test_store_sets_llm.py test_home_answerer.py test_provider_outputs.py test_core_track_outcome.py test_case_agent.py test_store_exams.py
```

**Done when:** `OK`; the 4 tests of H4a, the test of H4b and the test of H4c pass; no existing test changed except the
spy of H2.3.

---

## H5. Rebuild the sets and prove it on the real store

1. `df -h .` at least 5 GB free; `memory_pressure | tail -1` at least 10% free; nothing heavy running
   (`pgrep -fl "run_parallel|make check|dclab_rnd.store" | head`; if a store run or gate is running, wait).
2. Numbers before (write them down):
   ```bash
   for F in "case_files v1" "understand_problem v1" "understand_problem v2"; do set -- $F; DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.export_chat --model-type m4_workflow_llm --folder $1 --version $2 --out /tmp/hf_before_$1_$2 > /tmp/hf_before_$1_$2.log 2>&1; echo "== $1 $2"; .venv/bin/python -c "import json,sys; e=json.load(open(sys.argv[1]))['export']; print(e['counts'], e['held_for_rights'], e['answerer_not_recorded'])" /tmp/hf_before_$1_$2/MANIFEST.json; done
   ```
   (In zsh, if `set -- $F` does not split the words, use `read A B <<< "$F"` and `$A`/`$B` instead.)
3. Rebuild, **in the background** (your Bash tool's background option), and wait for it. Stop it and report if memory
   pressure turns red or the disk drops under 3 GB.
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -c "import json; from dclab_rnd.store.sets import llm; print(json.dumps(llm.build(), default=str)[:4000])" > /tmp/hf_build.log 2>&1; echo "rc=$?"
   ```
   `rc=0` is required. If it fails, report the last 30 lines of `/tmp/hf_build.log`; do not fix the builder.
4. Numbers after: the same loop as step 2 with `hf_after` in place of `hf_before`.
5. **Expected** (the store gains a few conversations each day, so totals may move a little):
   - `case_files v1`: `answerer_not_recorded` **0**; `held_for_rights` about `{"claude": 84, "mixed": 144}`; exported
     about 61 or fewer (only Qwen's are left).
   - `understand_problem v1`: `held_for_rights` now has `"frontier": 291` (plus any Claude rows it had before);
     `answerer_not_recorded` stays 291 (they are counted, and now also held).
   - `understand_problem v2`: unchanged from 890b2dcc's numbers (`answerer_not_recorded` 0).
   - A number more than 10% off: do not change code; report both numbers.
6. `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.store.exams verify`: every drawn type prints `"clean": true`.
7. `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.data_catalog --check` must print `data catalog is fresh`. If
   stale: `DCLAB_NO_LIVE_MODELS=1 make catalog`, then `git diff --stat -- docs/DATA_CATALOG.md docs/data_catalog.json`
   must list only those two files; add them to the H6 commit.

---

## H6. The record, the training set, the commit, the push, the report

1. Create `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-case-and-old-rows.json` in the
   same shape (same keys) as `research/llm-fine-tuning/experiments/dev_trajectories/records/training-rules-home-answerer.json`.
   Allowed values: a finding's `source` is one of `reviewer`, `test`, `measurement`, `owner`, `self`; `severity` one of
   `behaviour`, `latent`, `gap`, `flaky`, `privacy`, `leakage`; a measurement's `verdict` one of `pass`, `fail`,
   `not scored`. Content:
   - Finding HF-1: `source` `reviewer`, `severity` `gap`. Defect: the 289 case files named no model. Scenario: an exam's
     case file whose person was Claude Haiku and whose agent was Claude Sonnet was exported for training. Cause: the case
     builder never read the model requests. Fix: `cases.from_records` takes the models and writes `home_answerer`'s
     answer; an exam is joined by its exam id, arm and person. Tests: the 4 of H4a and the one of H4c.
   - Finding HF-2: `source` `reviewer`, `severity` `gap`. Defect: understand_problem v1 (291 Home rows) names no writer
     and is never rebuilt. Fix: the export holds a Home conversation or case file with no writer as an unknown model's.
     Test: `test_a_conversation_from_before_its_writer_was_recorded_is_held_as_an_unknown_models`.
   - Measurement: "The real exports of case_files v1 and understand_problem v1 and v2, before and after", `gate`
     "none", `fixed_before_run` false, `result` the H5 numbers, `verdict` `not scored`, `cost_eur` 0.0.
   - Risks: an ok request may not have reached a turn, so a few cases may be held that need not be; v1's rows that are
     older copies of v2 conversations are held even when v2 says the rules wrote them (205 of them), which loses nothing
     because their newer copy is in v2.
2. `DCLAB_NO_LIVE_MODELS=1 make sft-dev` must end with a line like `N examples from M packages ...`.
3. Final test run: the H4d command again. It must end with `OK`.
4. Commit (paths: `dclab_rnd/store/sets/cases.py`, `dclab_rnd/store/sets/llm.py`, `dclab_rnd/store/export_chat.py`,
   `tests/test_store_cases.py`, `tests/test_store_export_chat.py`, `tests/test_store_sets_llm.py`, the record, the files
   `git status --short` shows under `research/llm-fine-tuning/experiments/sft/out_dev/`, and the two catalog files only
   if H5.7 rebuilt them). First line:
   `Case files say who wrote them, and an old Home row with no writer is held`.
   Put the before and after numbers of H5 in the body.
5. Push to main and dev:
   ```bash
   C=$(git rev-parse HEAD); git push origin "${C}:refs/heads/main" "${C}:refs/heads/dev"; git branch -f dev HEAD
   ```
6. Report to the owner in plain words: what changed, the test counts of each run, the before and after numbers, how long
   the rebuild took, anything that did not go as planned, and the commit id.

---

## Verification (how the owner knows it is done)

1. **Tests:** the H4d run ends with `OK`, including the 6 new tests.
2. **Data:** the real export of `case_files v1` prints `answerer_not_recorded: 0` and holds Claude's and the mixed
   cases; `understand_problem v1` holds its 291 rows as `frontier`.
3. **Exams:** `python -m dclab_rnd.store.exams verify` prints `"clean": true` for every drawn type.
4. **Nothing forbidden changed:** `git show --stat HEAD` lists only the files named in H6.4.

## Not in this plan

- **The rights review is the owner's.** Held rows stay kept and prepared; they train only after it.
- v1's 263 older copies are not removed or edited: the store keeps everything. Only their export is held.
- Other task kinds were not searched for the same gap. A future set whose rows a model may have written should name
  its writer; adding its task to `WRITER_REQUIRED` makes the export hold it when it does not.

# Fix the five problems the live Home build showed (a plan a smaller model can execute)

## Context

On 2026-11-16 one real Home build ran end to end with a live model (run `agent_runs/live_tests/2026-11-16-home-plan-live-model`).
It worked, and it showed five problems. This plan fixes them. It is written so a smaller model can do the work task by
task without having to decide anything: every task names the file, the place, the code, the test, the command and the
exact result that proves it is done.

| # | Problem seen | Cause (verified in code) | Task |
|---|---|---|---|
| 1 | The person gave the prediction moment and said "Build it" in one message. Home asked for the moment again. | Nothing carries the moment to the build. Only the outcome column is carried (`case_home.py` `_build`). | A |
| 2 | "Build it now." was saved as part of the prediction moment. Home also said "So the prediction is made The prediction is made ...". | The whole answer is stored word for word (`case_home.py` `_build_answer`, branch `"prediction moment"`). | B |
| 3 | The build message said "1 open question". The model's plan then replaced the rules' plan and had none. The chat was never corrected. | `jobs/handlers.py` `_plan_first` throws away the result of `plan_project`. | C |
| 4 | The features chart marks the rule's pick (`calendar`) as chosen although the agent approved `raw`. | `studio/charts.py` `charts()` reads `evidence.selected_recipe`, never `decision.chosen`. | D |
| 5 | The archive receipt counted 28 of 59 calls "with thinking". All 59 kept thinking: 31 keep it redacted. | `harness_ablation/__init__.py` `capture()` never looks at `thinking_redacted`. | E |

**The owner's decision for problem 1 ("both"):** Home builds at once with the moment the person stated, **and** says that
moment back in the chat so the person can correct it. No extra question.

**Size:** five small code changes, about 150 lines of code and 150 lines of tests. No prompt file changes, no tool
schema changes, no change to Build project.

---

## Rules for whoever executes this (read first, they apply to every task)

1. Work in `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`, on branch `main`. Python is `.venv/bin/python`.
2. Do the tasks in order: A, B, C, D, E, F. Finish one (tests green, committed) before starting the next.
3. **Find a place by its text, not by its line number.** Line numbers below are hints. Use `grep -n "the quoted text" file`.
4. **Run tests only with this command form** (never a bare `pytest`, never a live key):
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 FILE1.py FILE2.py
   ```
   File names are bare (`test_case_home.py`, not `tests/test_case_home.py`). The last lines must read `Ran N tests ...` then `OK`.
5. **Do not run** `make check-commit`, `make check-quick`, `make check-all`, `make rd-check` or the whole suite. The owner's
   rule: full gates only when needed. Run only the files each task names.
6. If a test fails: read the failure, fix **the code**. Never delete, skip or weaken an assertion to make it pass. If an
   **existing** test fails and you believe the new behaviour is right, stop and report it; do not change that test.
7. **Do not touch** these: any file in `dclab_rnd/prompts/`, any tool schema in `dclab_rnd/agents/case_agent.py` or
   `dclab_rnd/intern/tools.py` (their text is fingerprinted: a change forces two long suites), the route
   `@router.post("/api/drafts/{draft_id}/build"` in `dclab_rnd/draft/api.py` (pinned by `tests/test_nodata_product.py`),
   any file under a `results/` folder.
8. Git: never `git stash`. Never `git add .` or `git add -A`. Stage and commit explicit paths only:
   ```bash
   git add -- path1 path2 && git commit -m "message" -- path1 path2
   ```
   Never stage `data/workspace/` or `training_records/STORE_CATALOG.json` (not ours). End every commit message with the
   attribution line your session tells you to use.
9. zsh pitfalls: a list of files in a variable needs an array (`F=(a b); git add -- "${F[@]}"`); write `"${C}:refs/..."`
   with braces; foreground `sleep` is blocked.
10. Never open `cases_dev3.json`, `answer_key_dev3.json`, `cases_nodata_held.json`, `answer_key_nodata_held.json` or any
    `cases_sealed.json`.
11. Write comments and messages in plain English, short sentences, no em dashes. A comment says **why** (what went wrong
    on the live run), not only what.

---

## Task A: the prediction moment the person stated is used for the build, and said back

**Goal.** A person writes: "The outcome column is churned. The prediction is made at the monthly review, before any
outreach. Build it." Home must build at once with the moment `at the monthly review, before any outreach`, and add one
chat message: "I am using the prediction moment you gave: ... Tell me if that is wrong." It must not ask for the moment.

**File:** `dclab_rnd/draft/case_home.py`, class `CaseHome`.

**Step A1. Add two patterns** next to the other patterns (find `LEADING_ANSWER = re.compile(`; put these right after it):

```python
    # A sentence in which the person themselves states when the prediction is made ("The prediction is made at the
    # monthly review", "We predict when the order is placed"). Live, 2026-11-16: the person said it in the same message
    # as "Build it", nothing carried it to the build, and Home asked for it again.
    MOMENT_STATED = re.compile(
        r"^\W*(?:and\s+|so\s+|also\s+)?"
        r"(?:(?:the\s+)?(?:predictions?|forecasts?|scores?|scoring|decisions?)\s+(?:is|are|gets|get|will\s+be|should\s+be|happens?|runs?)"
        r"(?:\s+(?:made|done|run|taken|produced|computed))?"
        r"|we\s+(?:predict|score|forecast|decide))\s*",
        re.IGNORECASE)
    SENTENCES = re.compile(r"(?<=[.?!;])\s+|\n+")
```

**Step A2. Add one method** right before `def _build(`:

```python
    def _stated_moment(self, draft: dict[str, Any]) -> str:
        """The prediction moment the person stated in their own words, or "". Their newest message first, then the
        case's kept words, then the draft's first sentence. Only a sentence that opens by saying when the prediction
        (or the decision) is made, names a time and is not a question: never a guess from other words."""
        saved = (draft.get("case_state") or {}).get("case")
        texts = [str(m.get("text") or "") for m in reversed(draft.get("messages") or []) if m.get("role") == "user"]
        texts += list(reversed(C.CaseFile.read(saved).words)) if saved else []
        texts.append(str(draft.get("problem") or ""))
        for text in texts:
            for sentence in self.SENTENCES.split(text):
                s = sentence.strip()
                found = self.MOMENT_STATED.match(s)
                if not found or "?" in s:
                    continue
                moment = s[found.end():].strip().rstrip(".!").strip()
                if len(moment.split()) >= 3 and self.TIME_WORDS.search(moment) and not self.VAGUE_MOMENT.match(moment):
                    return moment[:400]
        return ""
```

**Step A3. Use it in `_build`.** Find this line inside `_build`:

```python
        done = self.on_request(draft_id, "build", {})
```

Insert **before** it (same indentation):

```python
        draft = self.store.get(draft_id)
        if not (draft.get("understanding") or {}).get("prediction_moment") and not draft.get("build_waiting"):
            # the owner, 2026-11-17 ("both"): a moment the person stated is used at once AND said back, so they can correct it
            moment = self._stated_moment(draft)
            if moment:
                self.store.update(draft_id, lambda d: d.setdefault("understanding", {}).update(prediction_moment=moment))
                self.say(draft_id, f"I am using the prediction moment you gave: {moment}. Tell me if that is wrong.", kind="text", by="case_agent")
```

Why `not draft.get("build_waiting")`: when Home is already waiting for an answer about the moment, that flow (Task B)
decides, not this one.

**Step A4. Add tests** to `tests/test_case_home.py`, inside `class CaseHomeTests`, right after the test
`test_the_persons_yes_starts_the_build_and_an_outcome_named_in_their_answer_is_read_by_code`:

```python
    def test_a_moment_stated_with_the_build_order_is_used_at_once_and_said_back(self):
        """Live, 2026-11-16: the person gave the moment and said "Build it" in one message, and Home asked for it again."""
        d = self.store.create("predict which members cancel")
        self.store.update(d["id"], lambda x: x.update(analysis={"columns": [{"name": "member_id"}, {"name": "churned", "unique": 2}]}))
        home, calls = self._home_with_builds({"refused": "not yet", "rule": None})
        with mock.patch.object(A, "turn", self._yes):
            home.reply(d["id"], "The outcome column is churned. The prediction is made at the monthly review, before any outreach. Build it.")
        draft = self.store.get(d["id"])
        self.assertEqual(draft["understanding"]["prediction_moment"], "at the monthly review, before any outreach")
        self.assertEqual(calls, ["build"])
        self.assertIsNone(draft.get("build_waiting"))
        said = [m["text"] for m in draft["messages"] if m.get("role") == "agent"]
        self.assertTrue(any("I am using the prediction moment you gave: at the monthly review, before any outreach." in t for t in said))

    def test_only_a_sentence_that_states_the_moment_is_read_as_one(self):
        home, _ = self._home_with_builds()
        read = lambda *texts: home._stated_moment({"messages": [{"role": "user", "text": t} for t in texts], "problem": ""})  # noqa: E731
        self.assertEqual(read("We predict when the order is placed, before shipping."), "when the order is placed, before shipping")
        self.assertEqual(read("The decision is made at the monthly review, before any outreach."), "at the monthly review, before any outreach")
        self.assertEqual(read("The decision is made by the regional manager."), "")  # no time named: not a moment
        self.assertEqual(read("Is the prediction made at signup or at renewal?"), "")  # a question states nothing
        self.assertEqual(read("The prediction is made now."), "")  # vague
        self.assertEqual(read("Customers leave after the first month. Build it."), "")  # a time word alone is not a stated moment
        self.assertEqual(read("The prediction is made at signup, before payment.", "The prediction is made at renewal, before the invoice."),
                         "at renewal, before the invoice")  # the newest message wins
```

**Step A5. Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_case_home.py test_case_agent.py test_case_chips.py test_nodata_product.py
```

**Done when:** `OK`; the two new tests pass; no existing test in those four files changed.

**If the first new test fails** on `prediction_moment` being absent: print `self.store.get(d["id"])["messages"]` and
`draft.get("case_state")` in the test to see where the person's words are kept, and make `_stated_moment` read that
place. Do not loosen the test.

**Commit** (paths: `dclab_rnd/draft/case_home.py`, `tests/test_case_home.py`). Message, first line:
`A prediction moment the person stated with "Build it" is used at once and said back`. Body: what happened on the live
run, the cause, the owner's decision ("both").

---

## Task B: an order to build is never stored inside the prediction moment

**Goal.** Home asks for the moment. The person answers: "The prediction is made at the monthly review, before any
outreach, using only what is known on that day. Build it now." Home must store
`at the monthly review, before any outreach, using only what is known on that day` and say:
"So the prediction is made at the monthly review, before any outreach, using only what is known on that day. Shall I
build with that? Say yes, or tell me the moment again." (No "Build it now", no doubled "The prediction is made", no
second "using only what is known then".)

**File:** `dclab_rnd/draft/case_home.py`. Needs Task A done (it reuses `MOMENT_STATED` and `SENTENCES`).

**Step B1. Add one method** right before `def _say_moment_back(`:

```python
    def _moment_of(self, text: str) -> str:
        """The moment in an answer, without the sentences that are an order to build or plain approval, and without an
        opening "The prediction is made". Live, 2026-11-16: "…on that day. Build it now." was stored whole, so the
        project's prediction moment ended with "Build it now."."""
        from .chat import build_order

        kept = []
        for sentence in self.SENTENCES.split(str(text or "").strip()):
            s = sentence.strip()
            if not s or build_order(s) or not re.sub(r"[\W_]+", " ", self.APPROVAL.sub(" ", s)).strip():
                continue  # empty, an order to build, or only words of approval
            kept.append(self.MOMENT_STATED.sub("", s, count=1).strip())
        return " ".join(k for k in kept if k).strip()
```

**Step B2. Use it in `_build_answer`, branch `"prediction moment"`.** Find:

```python
        if need == "prediction moment":
            if len(said.split()) < 3 or self.VAGUE_MOMENT.match(said):
                return False
            self._say_moment_back(draft_id, said)
            return True
```

Replace with:

```python
        if need == "prediction moment":
            moment = self._moment_of(said)
            if len(moment.split()) < 3 or self.VAGUE_MOMENT.match(moment):
                return False
            self._say_moment_back(draft_id, moment)
            return True
```

**Step B3. Use it in the branch `"confirm moment"`.** Find:

```python
                moment = self.LEADING_ANSWER.sub("", said, count=1)
```

Replace with:

```python
                moment = self._moment_of(self.LEADING_ANSWER.sub("", said, count=1))
```

**Step B4. Fix the sentence in `_say_moment_back`.** Find:

```python
        self.say(draft_id, f"So the prediction is made {moment.rstrip('.')}, using only what is known then. Shall I build with that? "
                           "Say yes, or tell me the moment again.", kind="text", by="case_agent")
```

Replace with:

```python
        # the person's own "using only what is known ..." is not said a second time
        tail = "" if re.search(r"\bknown\b", moment, re.IGNORECASE) else ", using only what is known then"
        self.say(draft_id, f"So the prediction is made {moment.rstrip('.')}{tail}. Shall I build with that? "
                           "Say yes, or tell me the moment again.", kind="text", by="case_agent")
```

**Step B5. Add tests** to `tests/test_case_home.py`, class `CaseHomeTests`, after the tests of Task A:

```python
    def test_an_order_to_build_is_never_stored_inside_the_moment(self):
        """Live, 2026-11-16: "…on that day. Build it now." was stored whole as the project's prediction moment."""
        d = self.store.create("predict which members cancel")
        self.store.update(d["id"], lambda x: x.update(analysis={"columns": [{"name": "member_id"}, {"name": "churned", "unique": 2}]},
                                                      understanding={"target": "churned"}, build_waiting="prediction moment"))
        home, calls = self._home_with_builds({"refused": "not yet", "rule": None})
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read the build's answer")):
            home.reply(d["id"], "The prediction is made at the monthly review, before any outreach, using only what is known on that day. Build it now.")
        draft = self.store.get(d["id"])
        moment = "at the monthly review, before any outreach, using only what is known on that day"
        self.assertEqual((draft.get("build_waiting"), draft.get("moment_said")), ("confirm moment", moment + "."))
        back = draft["messages"][-1]["text"]
        self.assertEqual(back, f"So the prediction is made {moment}. Shall I build with that? Say yes, or tell me the moment again.")
        with mock.patch.object(A, "turn", side_effect=AssertionError("the agent must not read the build's answer")):
            home.reply(d["id"], "yes")
        self.assertEqual(self.store.get(d["id"])["understanding"]["prediction_moment"], moment + ".")
        self.assertEqual(calls, ["build"])

    def test_the_moment_is_cleaned_of_orders_and_approval_only(self):
        home, _ = self._home_with_builds()
        self.assertEqual(home._moment_of("at signup, before the first month is paid"), "at signup, before the first month is paid")  # unchanged
        self.assertEqual(home._moment_of("At checkout, before payment. Build it now."), "At checkout, before payment.")
        self.assertEqual(home._moment_of("We predict when the ticket is opened. Go ahead."), "when the ticket is opened.")
        self.assertEqual(home._moment_of("Build it now."), "")  # nothing but an order: no moment
        self.assertEqual(home._moment_of("At the start date of the contract"), "At the start date of the contract")  # "start" inside a moment is kept
```

Note on the first test: the stored moment keeps the person's final period (`moment + "."`), because the sentence is
kept as written. If your result has no period, that is also acceptable **only if** every existing test still passes;
then change the two `moment + "."` to `moment` in this new test and say so in the commit message.

**Step B6. Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_case_home.py test_case_agent.py test_home_conversation.py test_conversation_exam.py
```

**Done when:** `OK`; the two new tests pass; every existing test of the moment flow still passes unchanged (they are
listed in `tests/test_case_home.py` from `test_a_question_or_a_sentence_naming_a_column_in_passing...` to
`test_words_that_name_no_column_go_back_to_the_agent`, and `BuildFlowTests`).

**Commit** (paths: `dclab_rnd/draft/case_home.py`, `tests/test_case_home.py`). First line:
`An order to build is never stored inside the prediction moment`.

---

## Task C: when the model's plan replaces the rules' plan, the chat says so

**Goal.** The build message may say "The plan has 1 open question(s)". If the model's plan, made in the stages job,
replaces the rules' plan, Home adds one message: "Before the first stage, the model refined the plan: 3 sub-problem(s),
6 step(s). It has no open question for you; this replaces the count in my earlier message." (or the new count).

**Files:** `dclab_rnd/tasks/path.py`, `dclab_rnd/jobs/handlers.py`, `tests/test_path_plan.py`.

**Step C1. Add a function to `dclab_rnd/tasks/path.py`,** right after `def says(` ends (before `__all__`):

```python
def says_update(project: dict[str, Any]) -> Optional[str]:
    """What to tell the person when the model's plan replaced the rules' plan before the first stage. Live, 2026-11-16:
    the build message said "1 open question", the model's plan had none, and the chat was never corrected."""
    plan = stored(project)
    if plan is None or plan.get("routed_to") != "model":
        return None
    legs = (plan.get("path") or {}).get("legs") or []
    steps = sum(len(leg.get("steps") or []) for leg in legs)
    asks = (plan.get("enforcement") or {}).get("asked") or []
    return (f"Before the first stage, the model refined the plan: {len(legs)} sub-problem(s), {steps} step(s). "
            + (f"It has {len(asks)} open question(s) for you on the project's page; this replaces the count in my earlier message."
               if asks else "It has no open question for you; this replaces the count in my earlier message."))
```

Add `"says_update"` to `__all__` (keep the list sorted).

**Step C2. Use it in `dclab_rnd/jobs/handlers.py`, function `_plan_first`.** Find:

```python
        if use_model:  # with no model the rules' plan was made at the build, in the request
            task_path.plan_project(s.projects, project_id, gateway=s.gateway, use_model=True)
```

Replace with:

```python
        if use_model:  # with no model the rules' plan was made at the build, in the request
            result = task_path.plan_project(s.projects, project_id, gateway=s.gateway, use_model=True)
            if result.get("status") == "planned" and result.get("routed_to") == "model":
                # the build message was written from the rules' plan: the chat is told what the model's plan changed
                project = s.projects.get(project_id)
                draft_id = (project.get("draft") or {}).get("id")
                work = getattr(s, "draft_work", None)
                text = task_path.says_update(project)
                if draft_id and work is not None and text:
                    work.agent(draft_id).say(draft_id, text)
```

This is the same way `_improve_run` in the same file writes to the chat (`s.draft_work.agent(draft_id).say(...)`). It
is already inside the function's `try/except`, so a chat that cannot be written never stops the stages.

**Step C3. Add tests** to `tests/test_path_plan.py`, inside `class JobTests`, before `test_the_build_message_says_what_the_plan_did`:

```python
    def test_the_models_plan_is_said_in_the_chat_so_the_build_message_is_not_left_wrong(self):
        """Live, 2026-11-16: the build message said "1 open question"; the model's plan replaced the rules' and had none."""
        from dclab_rnd.jobs import handlers

        pid = home_project(self.store)
        P.plan_project(self.store, pid)  # the rules' plan, as Home's build makes it
        project = self.store.get(pid)
        project["draft"] = {"id": "d1"}
        self.store.save(project)
        answer = model_answer([{"task_type": "leakage_probe", "params": {"table": "customers", "column": LEAK}, "purpose": "Recorded after the outcome"}],
                              [f"post_outcome_{LEAK}"])
        said = []
        services = self.services(pid)
        services.gateway = Gateway(Planner(answer))
        services.draft_work = SimpleNamespace(agent=lambda draft_id: SimpleNamespace(say=lambda d, text, **k: said.append((d, text))))
        handlers._plan_first(services, "s1")
        self.assertEqual(P.stored(self.store.get(pid))["routed_to"], "model")
        self.assertEqual(len(said), 1)
        self.assertEqual(said[0][0], "d1")
        self.assertIn("the model refined the plan: 1 sub-problem(s), 1 step(s)", said[0][1])
        self.assertIn("no open question for you; this replaces the count in my earlier message", said[0][1])
        handlers._plan_first(services, "s1")  # the model's plan is kept: nothing is said twice
        self.assertEqual(len(said), 1)

    def test_nothing_is_said_when_the_rules_plan_stands_or_there_is_no_chat(self):
        from dclab_rnd.jobs import handlers

        pid = home_project(self.store)
        said = []
        services = self.services(pid)  # its model cannot plan: the rules' plan stands
        services.draft_work = SimpleNamespace(agent=lambda draft_id: SimpleNamespace(say=lambda d, text, **k: said.append(text)))
        handlers._plan_first(services, "s1")
        self.assertEqual(said, [])
        self.assertIsNone(P.says_update(self.store.get(pid)))
        handlers._plan_first(self.services(pid), "s1")  # a services object with no chat at all: no error
```

**Step C4. Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_path_plan.py test_jobs.py test_intern.py
```

**Done when:** `OK`; both new tests pass.

**If the first new test fails** because `routed_to` is `"rules"`: the model path was not taken. Check, in this order:
`handlers._plan_first` computes `use_model` true only when `optional("orchestra.settings")` exists and its
`admit_proposals` is true; print `use_model`. If it is false in the test environment, set it for the test with
`mock.patch.dict(os.environ, ...)` **only if** the settings module reads the environment; otherwise call
`P.plan_project(self.store, pid, gateway=services.gateway, use_model=True)` directly and then test
`P.says_update(...)` and the chat line by calling the new block through `_plan_first` with
`mock.patch("dclab_rnd.jobs.handlers.optional", ...)`. Report which you did.

**Commit** (paths: the three files). First line:
`When the model's plan replaces the rules' plan, the chat says so`.

---

## Task D: a chart marks the choice in force and the rule's pick

**Goal.** After a stage is approved with another choice (`decision.chosen != decision.selected`), the features and
models charts mark the row in force as `chosen`, mark the rule's row as `rule_pick`, and say `overridden: true`.

**File:** `dclab_rnd/studio/charts.py`, function `charts`. Its only reader is the agent's `get_charts` tool; **do not**
change the tool's description (rule 7).

**Step D1. Add a helper** right before `def charts(`:

```python
def _picks(record: dict[str, Any], evidence_key: str) -> tuple[Any, Any]:
    """(the rule's pick, the choice in force). An approval may choose another option: the stage record keeps it in
    ``decision.chosen`` and the rule's own in ``decision.selected`` (``engine.approve``); the evidence keeps only the
    rule's. Live, 2026-11-16: the agent approved `raw`, and the features chart still marked the rule's `calendar`."""
    decision = record.get("decision") or {}
    rule = decision.get("selected") or (record.get("evidence") or {}).get(evidence_key)
    return rule, decision.get("chosen") or rule
```

**Step D2. Features block.** Find:

```python
                 "chosen": r["recipe"] == ev.get("selected_recipe")} for r in ev.get("stage_results") or [] if metric in (r.get("metrics") or {})]
        if rows:
            out.append({"id": "features", **_bars(rows, f"Feature recipes, cross-validation {metric}", metric, ev.get("selection_rule"))})
```

Replace with:

```python
                 "chosen": r["recipe"] == in_force, "rule_pick": r["recipe"] == rule}
                for r in ev.get("stage_results") or [] if metric in (r.get("metrics") or {})]
        if rows:
            out.append({"id": "features", **_bars(rows, f"Feature recipes, cross-validation {metric}", metric, ev.get("selection_rule")),
                        "rule_pick": rule, "chosen": in_force, "overridden": in_force != rule})
```

and add this line right after `ev = features["evidence"]` (the one in the features block):

```python
        rule, in_force = _picks(features, "selected_recipe")
```

**Step D3. Models block.** Find:

```python
        rows = [{"label": r["model"], "value": _r(r["mean"]), "spread": _r(r.get("std")), "chosen": r["model"] == ev.get("selected_model")}
                for r in ev.get("ranking") or []]
        if rows:
            out.append({"id": "models", **_bars(rows, f"Model screen, cross-validation {metric}", metric, ev.get("selection_rule"))})
```

Replace with:

```python
        rule, in_force = _picks(models, "selected_model")
        rows = [{"label": r["model"], "value": _r(r["mean"]), "spread": _r(r.get("std")), "chosen": r["model"] == in_force,
                 "rule_pick": r["model"] == rule} for r in ev.get("ranking") or []]
        if rows:
            out.append({"id": "models", **_bars(rows, f"Model screen, cross-validation {metric}", metric, ev.get("selection_rule")),
                        "rule_pick": rule, "chosen": in_force, "overridden": in_force != rule})
```

**Step D4. Add a test** to `tests/test_studio_charts.py`, class `ChartsTest`, after
`test_a_binary_project_gets_every_chart_from_its_records`:

```python
    def test_a_chart_marks_the_choice_in_force_and_the_rules_pick(self):
        """Live, 2026-11-16: the agent approved `raw` over the rule's `calendar`, and the chart still marked the rule's pick."""
        records = _records(_binary_final())
        records["features"]["decision"] = {"selected": "raw", "chosen": "poly2", "overridden": True}
        records["models"]["decision"] = {"selected": "logistic_regression", "chosen": "xgboost", "overridden": True}
        by = {c["id"]: c for c in charts(records, "p1")["charts"]}
        self.assertEqual([(r["label"], r["chosen"], r["rule_pick"]) for r in by["features"]["rows"]], [("raw", False, True), ("poly2", True, False)])
        self.assertEqual((by["features"]["chosen"], by["features"]["rule_pick"], by["features"]["overridden"]), ("poly2", "raw", True))
        self.assertEqual([(r["label"], r["chosen"], r["rule_pick"]) for r in by["models"]["rows"]],
                         [("logistic_regression", False, True), ("xgboost", True, False)])
        self.assertTrue(by["models"]["overridden"])
        plain = {c["id"]: c for c in charts(_records(_binary_final()), "p1")["charts"]}  # no approval: one row is both
        self.assertEqual([(r["chosen"], r["rule_pick"]) for r in plain["features"]["rows"]], [(True, True), (False, False)])
        self.assertFalse(plain["features"]["overridden"])
```

**Step D5. Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_studio_charts.py test_intern.py test_mcp.py test_agents_runtime.py
```

**Done when:** `OK`; the new test passes; the existing line
`self.assertEqual([r["chosen"] for r in by["models"]["rows"]], [True, False])` still passes unchanged.

**If a test in another file compares a chart's whole dictionary** and fails on the new keys (`rule_pick`, `chosen`,
`overridden`): that is an expected, additive change. Stop and report the test's name; do not edit it without the owner.

**Commit** (paths: `dclab_rnd/studio/charts.py`, `tests/test_studio_charts.py`). First line:
`A chart marks the choice in force and the rule's pick`.

---

## Task E: the archive receipt counts thinking that is kept redacted

**Goal.** A model call that saw table values keeps its thinking redacted (`thinking_redacted`), by rule 8 of
`CLAUDE.md`. The receipt must count it as thinking kept, and say how many are redacted.

**File:** `dclab_rnd/agent_eval/harness_ablation/__init__.py`, function `capture`.

**Step E1.** Find:

```python
    thought = [r for r in ok if r.get("thinking") or r.get("reasoning") or r.get("reasoning_details")]
    return {"exchanges": len(rows), "ok": len(ok), "whole": sum(r.get("whole") is True for r in ok), "with_thinking": len(thought),
            "without_thinking": sorted({str(r.get("purpose")) for r in ok if r not in thought}), "purposes": sorted({str(r.get("purpose")) for r in rows})}
```

Replace with:

```python
    # A call shown table values keeps its thinking redacted (``thinking_redacted``, rule 8): that is thinking kept. Live,
    # 2026-11-16: a receipt counted 28 of 59 calls "with thinking" when all 59 had kept it, 31 of them redacted.
    redacted = [r for r in ok if r.get("thinking_redacted") and not r.get("thinking")]
    thought = [r for r in ok if r.get("thinking") or r.get("reasoning") or r.get("reasoning_details") or r.get("thinking_redacted")]
    return {"exchanges": len(rows), "ok": len(ok), "whole": sum(r.get("whole") is True for r in ok), "with_thinking": len(thought),
            "thinking_redacted": len(redacted),
            "without_thinking": sorted({str(r.get("purpose")) for r in ok if r not in thought}), "purposes": sorted({str(r.get("purpose")) for r in rows})}
```

Also update the function's docstring to: `"""What the run kept of its model calls: every answered call must hold its
thinking, whole or redacted (the owner's rule)."""`

**Step E2. Add a test** to `tests/test_harness_ablation.py`. Find the class that holds the test containing
`kept = H.capture(home)` and add this method to the same class (it needs only `tempfile`, `json`, `Path`, which the file
already imports; check the top of the file and add an import only if one is missing):

```python
    def test_thinking_kept_redacted_counts_as_thinking_kept(self):
        """Live, 2026-11-16: 31 of 59 calls had seen table values, kept their thinking redacted, and were counted as none."""
        with tempfile.TemporaryDirectory() as tmp:
            day = Path(tmp) / "model_exchanges" / "2026" / "11"
            day.mkdir(parents=True)
            rows = [{"purpose": "exam_plain", "outcome": "ok", "whole": True, "thinking": [{"text": "a summary"}]},
                    {"purpose": "intern", "outcome": "ok", "whole": False, "thought": True, "thinking_redacted": [{"text": "a redacted summary"}]},
                    {"purpose": "reporter", "outcome": "ok", "whole": False, "thought": False},
                    {"purpose": "intern", "outcome": "the provider refused the key"}]
            (day / "16.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            kept = H.capture(Path(tmp))
        self.assertEqual((kept["exchanges"], kept["ok"], kept["whole"]), (4, 3, 1))
        self.assertEqual((kept["with_thinking"], kept["thinking_redacted"], kept["without_thinking"]), (2, 1, ["reporter"]))
```

**Step E3. Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_harness_ablation.py
```

**Done when:** `OK`; the new test passes; the two existing assertions
`(kept["ok"], kept["with_thinking"], kept["without_thinking"]) == (3, 3, [])` and
`(record["capture"]["ok"], record["capture"]["with_thinking"], record["capture"]["without_thinking"]) == (4, 4, [])`
still pass unchanged.

**If `tests/test_harness_ablation.py` fails to start** with a message about "pinned paths" or a hash: this module is
pinned by a gate. Run `grep -n "PINNED_PATHS" dclab_rnd/agent_eval/harness_ablation/*.py` and report what it lists
before changing anything else.

**Commit** (paths: the two files). First line:
`An archive receipt counts thinking that is kept redacted`.

---

## Task F: the record, the training set, the push, the report

**Step F1. Write the build record.** Create
`research/llm-fine-tuning/experiments/dev_trajectories/records/home-live-build-fixes.json`. Copy the shape of
`research/llm-fine-tuning/experiments/dev_trajectories/records/planning-h5-home-plan-first.json` (same keys:
`package`, `title`, `date`, `request`, `spec`, `plan`, `decisions`, `change`, `findings`, `measurements`, `outcome`,
`lessons`, `risks`, `commits`). Rules for the values, or the validator refuses the file:
- every finding has `id`, `source`, `severity`, `defect`, `scenario`, `cause`, `fix`, `test`;
- `source` is one of `reviewer`, `test`, `measurement`, `owner`, `self`;
- `severity` is one of `behaviour`, `latent`, `gap`, `flaky`, `privacy`, `leakage`;
- a measurement's `verdict` is one of `pass`, `fail`, `not scored`; use `"measurements": []` here.
Write five findings (one per task A to E), `source: "measurement"`, `severity: "behaviour"` (E: `"gap"`), each with the
live scenario from the table at the top, the cause, the fix and the test name.

**Step F2. Rebuild the training set:**
```bash
DCLAB_NO_LIVE_MODELS=1 make sft-dev
```
It must end with a line like `N examples from M packages ...`. An error names the field to fix in the record.

**Step F3. Two freshness checks** (each must print the quoted line):
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.data_catalog --check
```
→ `data catalog is fresh`. If it says stale: run `DCLAB_NO_LIVE_MODELS=1 make catalog`, check with
`git diff --stat -- docs/DATA_CATALOG.md docs/data_catalog.json` that only those two files changed, and add them to the commit.
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.agentic.web.build --check
```
→ `product frontend is up to date` (no page file was changed in this plan, so it must be).

**Step F4. One last run of every file this plan touched:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_case_home.py test_path_plan.py test_studio_charts.py test_harness_ablation.py test_intern.py test_jobs.py test_nodata_product.py test_prompt_gate.py test_harness_gate.py
```
`test_prompt_gate.py` and `test_harness_gate.py` must pass: they prove no prompt or tool schema changed (rule 7).

**Step F5. Commit** the record and the rebuilt set (paths: the record, and the three files `git status --short` shows
under `research/llm-fine-tuning/experiments/sft/out_dev/`). Then push:
```bash
C=$(git rev-parse HEAD); git push origin "${C}:refs/heads/main" "${C}:refs/heads/dev"; git branch -f dev HEAD
```

**Step F6. Report** to the owner, in plain words: for each task, what was changed, the test names, the test counts of
each run, anything that did not go as this plan says, and the commit ids.

---

## Verification (how the owner knows it is done)

1. **Tests:** the command of step F4 ends with `OK`. New tests, by name:
   - A: `test_a_moment_stated_with_the_build_order_is_used_at_once_and_said_back`, `test_only_a_sentence_that_states_the_moment_is_read_as_one`
   - B: `test_an_order_to_build_is_never_stored_inside_the_moment`, `test_the_moment_is_cleaned_of_orders_and_approval_only`
   - C: `test_the_models_plan_is_said_in_the_chat_so_the_build_message_is_not_left_wrong`, `test_nothing_is_said_when_the_rules_plan_stands_or_there_is_no_chat`
   - D: `test_a_chart_marks_the_choice_in_force_and_the_rules_pick`
   - E: `test_thinking_kept_redacted_counts_as_thinking_kept`
2. **Nothing forbidden changed:** `git diff --stat 51b3d9eb..HEAD` lists only: `dclab_rnd/draft/case_home.py`,
   `dclab_rnd/tasks/path.py`, `dclab_rnd/jobs/handlers.py`, `dclab_rnd/studio/charts.py`,
   `dclab_rnd/agent_eval/harness_ablation/__init__.py`, the five test files, the record, the three `sft/out_dev` files
   (and the two catalog files only if step F3 needed them). No file under `dclab_rnd/prompts/`.
3. **One live check, by the owner or a stronger session, about EUR 0.10** (optional but it is the real proof): repeat the
   live Home build of 2026-11-16 with the message "The outcome column is churned. The prediction is made at the monthly
   review, before any outreach. Build it." Expected: Home builds without asking for the moment; the chat shows "I am
   using the prediction moment you gave: at the monthly review, before any outreach."; the project's
   `solution.prediction_moment` holds no "Build it"; after the model's plan a chat line starts "Before the first stage,
   the model refined the plan"; the run's `archive_receipt.json` shows `with_thinking` equal to `ok`.

## Not in this plan (said so nobody expects it)

- The lead agent's own model still does not pass the moment as a tool argument: code reads it from the person's words
  instead. That needs no prompt or tool change, and it does not depend on the model.
- A moment written in another shape ("at the monthly review we score them") is not read by Task A; Home then asks, as
  today, and Task B keeps the answer clean.
- The project page does not draw `studio.charts`; its own views already show an overridden choice. Only the agent's
  `get_charts` tool is fixed.
- The planning gate `hp_gate_v1` is still not run.

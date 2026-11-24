# Home intelligence: a simple step-by-step build guide

Plan of 2026-11-24. It improves the owner's two documents
(`DCLab_01_Home_Multi_Agent_Architecture_EN.md`, `DCLab_02_All_Agent_Data_and_Fine_Tuning_EN.md`) and turns them into
small tasks that a smaller AI model can build one at a time.

**Who does what (the owner's decisions, 2026-11-24):**
- After approval, Claude (this session) only saves this guide and the reviewed documents in the repository and pushes
  them (task T0).
- The owner gives tasks T1 to T7 to a smaller model in the owner's own session.
- Claude reviews the work when the owner asks.
- Workspace is not touched. The first build is a thin slice: 3 roles and real analysis tools, then measure.

---

## Part 1. What we are building, in plain words

Today a person writes to Home: "My sales went down. I think I am losing customers." One agent, the **Lead**, answers.

We add one new tool for the Lead, called `intelligence`. When the Lead uses it:

1. **Code computes facts** from the shared table. Example: "Revenue fell 19%. Buyers: +1%. Orders per buyer: 0%.
   Average order value: −20%." Each computed fact is called an **evidence** item and has an id (E1, E2, ...).
2. **Three roles each give their own view.** A role is one model call with its own short instruction:
   - **framing**: what is the real question, and 2 to 4 possible explanations (hypotheses);
   - **business**: how this business makes money (the KPI tree), and which definitions to confirm;
   - **analyst**: reads the evidence items and says which explanation the numbers support.
   No role sees another role's answer.
3. **Code checks every claim.** A claim called a "fact" must point to an evidence id, and every number in it must be
   in that evidence. If not, it is dropped or marked as only a hypothesis.
4. The Lead gets one tidy result and writes the single answer the person sees.

A switch, `DCLAB_HOME_INTELLIGENCE`, controls it:

| Value | Meaning |
|---|---|
| unset or `0` | Off. The Lead is offered exactly the tools it has today. |
| `tools` | The tool returns the computed evidence only. No roles. |
| `1` | The tool returns the evidence and the three role reports. |

The `tools` value exists for one reason: so a test can tell whether the **roles** help, or the **computed evidence
alone** gives the gain. Our earlier tests showed that more agents often did worse (team race: rules 9, helper 8, agent
2 of 19), so we measure before we switch anything on.

### What changed from the owner's two documents, and why

| Owner's document | This guide | Why |
|---|---|---|
| 8 new roles | 3 roles first | More roles only when a test shows each helps. |
| A model "router" picks roles | No router; all 3 roles run | One less model call and one less thing to break. |
| Roles run in parallel | Roles run one after another (first version) | Simple and safe. Parallel is a later task. |
| The data scientist writes and runs Python | Code runs a fixed set of analyses; the analyst reads the results | Free code on real data is the biggest security risk, and no safe box exists for Home. |
| `orchestra/` is called Workspace | Only `teamspace/` is Workspace | `orchestra/` is Home's own engine. It is simply not used here. |
| New case object with its own "verified facts" | Facts stay in the existing case file; the new object holds reports and evidence | One source of truth. |
| Contracts in `dclab_rnd` | Contracts in `dclab_core` | The repository rule. |
| A new event ledger, artifact store, dataset registry, exporters (Document 2) | Nothing new: the existing capture already records every model call with its thinking summary | They already exist (`models/gateway.py`, `experience/`, `store/`, `lake/`, `datasets.py`, `consent.py`). We only prove the new calls are captured. |
| Fine-tuning stages FT1 to FT6 | Not in this guide | No data yet and the disk is nearly full. Entry condition: 50 verified Home-intelligence cases. |
| "Meaningful improvement" judged by a model | Scores by code against planted causes; the rule fixed before the run | Our judge agreed poorly with the owner before (0.66). |

---

## Part 2. Rules for the model that builds this

Read `CLAUDE.md` at the repository root first. Then follow these rules in every task.

**Never:**
1. Never import anything from `dclab_rnd/teamspace/` in the new code. Never change a file in `dclab_rnd/teamspace/`.
2. Never call a live model. Run every test command with `DCLAB_NO_LIVE_MODELS=1` in front.
3. Never use a Claude model name anywhere in tests or evaluation code.
4. Never open a file whose path contains `sealed` or `/held/`. Never print `.env`.
5. Never run `git stash`. Never run `git add .` or `git add <folder>`. Stage files one by one by their full path.
6. Never edit generated files (`evidence/knowledge/**`, `dclab_rnd/agentic/static/app/**`, `research/*/INDEX.md`).
7. Never change files that this guide does not name. If you think another file must change, stop and write why in
   `Dev_Plans/Home_Intelligence_2026-11-24/HANDBACK.md`.
8. Never invent an API. Before you call a function, open its file and read its real signature.

**Always:**
1. Do one task at a time, in order. Finish its "Done when" list before starting the next.
2. Write the tests first, see them fail, then write the code.
3. Use the project's Python: `.venv/bin/python`.
4. After each task, append a short note to `HANDBACK.md`: what you built, which tests pass, anything you were unsure
   about. Then make one commit for the task. End the commit message with the attribution line the session gives you.
5. If a test outside your task fails and you did not cause it, do not fix it. Write it in `HANDBACK.md`.

**Words used in this guide:**
- *frame*: a pandas DataFrame, the shared table.
- *window*: a date range. Window **A** is the earlier period, window **B** is the later one.
- *evidence*: one computed result, a dict `{"id", "name", "text", "data"}`.
- *finding*: one sentence a role wrote, with a type.
- *the Lead*: the Home agent in `dclab_rnd/agents/case_agent.py`.
- *state*: the `CaseState` object of one Home conversation (same file, line 316).

---

## Part 3. The tasks

### T0. Save the documents (done by Claude on 2026-11-24)

`Dev_Plans/Home_Intelligence_2026-11-24/` now holds: the owner's two original documents in `originals/`, the review
in `REVIEW.md`, this guide as `BUILD_GUIDE.md`, and `HANDBACK.md` for the builder's notes. Start at T1.

---

### T1. Analysis functions (pure pandas, no model)

**Goal:** functions that turn an orders table into evidence items.

**Read first:** `dclab_core/change.py` (lines 30 to 135: `Window`, `compare`, `FLOOR`).

**Create:** `dclab_core/kpi_analysis.py` and `tests/test_kpi_analysis.py`.

**Rule for this file:** it may import only `numpy`, `pandas`, `pydantic`, and other `dclab_core` modules.

**Functions** (every one returns a dict with the keys `name`, `text`, `data`; `text` is 1 or 2 plain sentences that
contain the numbers; round money to whole units and shares to one decimal):

| Function | What it computes |
|---|---|
| `windows(frame, date_column, days=90)` | Returns `(a, b)`, two `change.Window` objects. B is the last `days` days of the data. A is the `days` days before B. |
| `revenue_chain(frame, *, date_column, value_column, customer_column, a, b)` | For A and for B: revenue (sum of value), buyers (distinct customers), orders per buyer, average order value. Then the percent change of each. `data["main_driver"]` is `"buyers"`, `"orders_per_buyer"` or `"order_value"`: the one whose `ln(B/A)` has the largest absolute size and the same sign as the revenue change. It is `"none"` when revenue changed by less than 3%. |
| `repeat_rate(frame, *, date_column, customer_column, a, b)` | Share of buyers with 2 or more orders inside the window, for A and for B. |
| `cohort_retention(frame, *, date_column, customer_column, a, b)` | A cohort is the month of a customer's first order. For each cohort: buyers active in A, and how many of them are also active in B (`kept`). Report the cohort with the lowest `kept` share among cohorts with at least `FLOOR` buyers in A, and the overall share. |
| `by_segment(frame, *, date_column, value_column, by, a, b)` | Calls `change.compare(frame, date_column=..., a=a, b=b, by=by, measure=value_column)`. Reports the 3 segments with the largest absolute change. |
| `flag_share(frame, *, date_column, flag_column, a, b)` | Share of rows where the flag is true, for A and for B. |
| `margin(frame, *, date_column, value_column, cost_column, a, b)` | Revenue, cost and margin (revenue minus cost) for A and for B, with percent changes. |
| `coverage(frame, *, date_column, a, b)` | Number of calendar days with zero rows in A and in B, and the longest run of empty days in B. |
| `year_ago(frame, *, date_column, value_column, a, b)` | The same A-to-B revenue change one year earlier. If the data does not reach back that far: `data["available"] = False`. |
| `run_all(frame, *, date_column, value_column, customer_column=None, by_columns=(), flag_columns=(), cost_column=None, days=90)` | Calls every function whose columns were given. Returns a list of evidence dicts with ids `"E1"`, `"E2"`, ... in a fixed order. Returns `[]` when the frame has fewer than `2 * FLOOR` rows. Never raises for a missing optional column. Raises `ValueError` when `date_column` or `value_column` is not in the frame. |

**Tests** (build tiny frames by hand in the test; no replica yet):
- `test_revenue_chain_names_order_value_when_only_the_value_fell`
- `test_revenue_chain_names_buyers_when_only_buyers_fell`
- `test_revenue_chain_says_none_when_revenue_moved_under_three_percent`
- `test_the_three_log_changes_add_up_to_the_revenue_log_change`
- `test_repeat_rate_counts_buyers_with_two_or_more_orders`
- `test_coverage_finds_a_gap_of_empty_days`
- `test_year_ago_is_not_available_on_short_data`
- `test_run_all_gives_ids_in_order_and_skips_what_has_no_column`
- `test_every_number_in_a_text_is_in_its_data` (for each evidence: every number that appears in `text` can be found,
  after the same rounding, in `data`)

**Run:** `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py test_kpi_analysis test_core_contracts`

**Done when:** both test files pass.

---

### T2. Invented test data with a known cause

**Goal:** a generator of invented orders tables. Each table has one planted cause that the tests know.

**Read first:** `dclab_core/replicas/mall_sales.py` (the shape of a replica: a dataclass, `generate(seed)`, `save`).

**Create:** `dclab_core/replicas/orders_decline.py` and `tests/test_orders_decline.py`.

**The table** (about 1,500 customers, 24 months, one row per order):
`order_id`, `customer_id`, `order_date`, `order_value`, `cost`, `category` (5 categories), `discount_pct`,
`out_of_stock` (true when the customer met a stockout that day).

**`CAUSES`** and how each is planted. The "last 90 days" is window B. Use `numpy.random.default_rng(seed)` only.

| Cause | How to plant it |
|---|---|
| `order_value` | In B, multiply every `order_value` by 0.80. |
| `repeat_purchases` | Customers whose first order was 8 to 11 months before the end order 60% less often in B. |
| `stockouts` | In B, one category has `out_of_stock` true on 35% of days, and 70% of that category's orders on those days are removed. |
| `margin` | Revenue is unchanged. In B, multiply `cost` by 1.25. |
| `seasonal_only` | The same 3 calendar months are 20% lower in both years. Nothing else changes. |
| `no_change` | Nothing is planted. The revenue change from A to B stays under 3%. |
| `data_gap` | Remove all rows of 20 days in a row inside B. Nothing else changes. |

For every cause except `seasonal_only`, the seasonal curve is flat, so the planted cause is the only change.

**Functions:**
- `generate(cause, seed=0) -> Replica` with fields `orders` (the frame), `key` (`{"cause": ..., "window_days": 90}`),
  `info` (column names and what each means, **without** the cause).
- `save(replica, pack_dir, key_dir)`: writes `orders.csv` and `columns.json` into `pack_dir`, and `key.json` into
  `key_dir`. The two folders must be different.

**Tests** (one per cause, each uses `kpi_analysis.run_all` from T1):
- `order_value`: `revenue_chain` gives `main_driver == "order_value"`.
- `repeat_purchases`: `main_driver == "orders_per_buyer"`, and `cohort_retention` names a cohort inside the planted
  range.
- `stockouts`: `flag_share` is higher in B, and `by_segment` on `category` puts the planted category first.
- `margin`: `revenue_chain` says `"none"`, and `margin` shows the margin fell by more than 10%.
- `seasonal_only`: `year_ago` shows a change within 5 points of this year's change.
- `no_change`: `revenue_chain` says `"none"`.
- `data_gap`: `coverage` reports a run of at least 20 empty days in B.
- `test_the_same_seed_gives_the_same_table`
- `test_the_pack_folder_never_holds_the_cause` (after `save`, no file in `pack_dir` contains the cause's name)

**Run:** `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py test_orders_decline test_kpi_analysis`

**Done when:** both pass.

---

### T3. The contracts and the checking rules

**Goal:** the shapes of the result, and the code that checks role findings.

**Read first:** `dclab_core/change.py` lines 1 to 70 (how a contract class is written). Then run
`grep -rn "change_result" dclab_core/ tests/ | head -30` and note **every place** that contract is registered. You
will register the new contract in the same places.

**Create:** `dclab_core/intelligence.py` and `tests/test_intelligence_contracts.py`.

**Shapes** (pydantic models, same base classes as in `change.py`):

```python
FINDING_TYPES = ("fact", "hypothesis", "inference", "unknown")
MAX_FINDINGS = 6          # per role

class Evidence:   id: str; name: str; text: str            # the computed result, as words
class Finding:    type: one of FINDING_TYPES; text: str (max 600); evidence: list[str] (ids);
                  downgraded: bool = False
class RoleReport: role: str; status: "ok" | "failed" | "skipped"; findings: list[Finding]; error: str = ""
class IntelligenceResult:                                   # schema "dclab_core.intelligence_result.v1"
                  question: str; mode: "tools" | "roles"; evidence: list[Evidence];
                  reports: list[RoleReport]; dropped: list[dict]
```

**Functions:**

1. `parse(value) -> (list[dict], list[str])`. `value` is what the model's JSON decoded to. It must be
   `{"findings": [{"type", "text", "evidence"}]}`. Keep at most `MAX_FINDINGS`. Return the usable findings and a list
   of problems (strings) for the ones you skipped.

2. `validate(findings, evidence, said, unsaid) -> (list[Finding], list[dict])`.
   - `evidence`: the list of `Evidence`.
   - `said`: the person's own words (list of strings).
   - `unsaid`: a function `(text, sources) -> list[str]` that returns the numbers in `text` that no source holds. It
     is passed in, because the real one lives in `dclab_rnd/cited.py` and `dclab_core` may not import `dclab_rnd`.
   - Apply these rules to each finding, in this order:
     - **Rule 1.** Remove evidence ids that do not exist.
     - **Rule 2.** `extra = unsaid(text, said + [texts of the evidence it cites])`. If `extra` is not empty, drop the
       finding. Add `{"text": ..., "reason": "number_not_in_evidence", "numbers": extra}` to the dropped list.
     - **Rule 3.** If the type is `"fact"` and it cites no evidence, change the type to `"hypothesis"` and set
       `downgraded=True`.
   - Return the kept findings and the dropped list.

3. `digest(result, limit=4) -> list[Finding]`: at most `limit` findings for the case file. Facts first, then
   hypotheses. Skip downgraded ones when there are enough others.

**Tests:**
- `test_a_fact_with_evidence_and_matching_numbers_is_kept`
- `test_a_fact_without_evidence_becomes_a_hypothesis`
- `test_a_finding_with_a_number_no_evidence_holds_is_dropped`
- `test_an_unknown_evidence_id_is_removed`
- `test_more_than_six_findings_are_cut`
- `test_a_reply_that_is_not_findings_gives_a_problem`
- `test_digest_prefers_facts`

In these tests, pass a small fake `unsaid` (for example: find digits in the text that are not in any source).

**Also:** bump the version in `dclab_core/pyproject.toml` and `dclab_core/__init__.py` by one minor step, and add an
entry at the top of `dclab_core/CHANGELOG.md` for T1, T2 and T3.

**Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 make core-schemas
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py test_intelligence_contracts test_core_contracts
DCLAB_NO_LIVE_MODELS=1 make core-wheel-check
```

**Done when:** all three succeed.

---

### T4. The role calls

**Goal:** the module that runs the analyses and asks the three roles.

**Read first:**
- `dclab_rnd/agents/specialists.py` (all of it, 160 lines). The new role call copies `consult` (line 121).
- `dclab_rnd/prompts/case_specialist.md` (how a role prompt is written).
- `tests/test_case_agent.py` lines 20 to 46 (`Script`, `state`) and lines 609 to 672 (`SpecialistTests`): how a test
  gives scripted model answers.

**Create:**
- `dclab_rnd/home_intelligence/__init__.py`
- `dclab_rnd/home_intelligence/settings.py`
- `dclab_rnd/home_intelligence/roles.py`
- `dclab_rnd/home_intelligence/run.py`
- `dclab_rnd/prompts/case_intelligence.md`
- `tests/test_home_intelligence.py`

**`settings.py`:**
```python
FLAG = "DCLAB_HOME_INTELLIGENCE"
def mode() -> str:
    # "off" for unset, "", "0", "off", "false", "no"
    # "tools" for "tools"
    # "roles" for "1", "on", "roles", "true", "yes"
MAX_CALLS_A_CASE = 2
ROLE_TOKENS = 3000
PURPOSE = "case_intelligence"
PROMPT = "case_intelligence"
```

**`roles.py`:**
```python
ROLES = {
  "framing":  "State the real question behind the person's words. Give 2 to 4 competing explanations. "
              "For each, say what would prove it wrong. Say what is unknown.",
  "business": "Describe how this business makes money as a small KPI tree. Say which definitions must be "
              "confirmed (for example net or gross revenue). Name constraints that matter.",
  "analyst":  "Read the evidence items. Say which explanation the numbers support and which they reject. "
              "Every fact must cite the evidence ids it comes from. Association is not cause: say so.",
}
def ask(client, role, question, case_text, table_text, evidence) -> dict:
    # Build the user message like specialists.consult does.
    # Only the "analyst" role gets the evidence texts ("E1: ...", one per line).
    # Call client.complete([...system prompt..., ...user...], None, max_tokens=ROLE_TOKENS).
    # Decode with person_json.first_json, then dclab_core.intelligence.parse.
    # Return {"role": role, "findings": [...], "problems": [...]}  or  {"role": role, "error": "..."}.
    # Catch every exception from the call and return {"role": role, "error": str(e)[:200]}.
```

**`prompts/case_intelligence.md`** (the shared system prompt; write it in plain words). It must say:
- You are one specialist; you give your own view; you do not see other specialists.
- Answer only with JSON: `{"findings": [{"type": "fact|hypothesis|inference|unknown", "text": "...", "evidence": ["E1"]}]}`.
- A "fact" must cite evidence ids. Never write a number that is not in the person's words or in the evidence you cite.
- Without evidence, give hypotheses and unknowns, never facts.
- At most 6 findings, each one or two sentences.
- Do not ask the model to explain its reasoning in the answer.

**`run.py`:**
```python
def run(question, *, mode, frame, columns, client, case_text, table_text, said) -> dict:
    # columns: {"date_column", "value_column", "customer_column", "by_columns", "flag_columns", "cost_column"}
    # 1. evidence = [] if frame is None or the two required columns are missing,
    #    else kpi_analysis.run_all(frame, **columns)
    # 2. reports = []
    #    if mode == "roles":
    #        for role in ("framing", "business", "analyst"):
    #            if role == "analyst" and not evidence: add a report with status "skipped"; continue
    #            out = roles.ask(...)
    #            if "error" in out: add a report with status "failed" and the error; continue
    #            kept, dropped = intelligence.validate(out["findings"], evidence, said, cited.unsaid)
    #            add a report with status "ok" and the kept findings; remember the dropped ones
    # 3. return IntelligenceResult(...).model_dump()
```

**Register the new model purpose `case_intelligence`.** Run `grep -rn "case_specialist" dclab_rnd/ tests/` and copy
what is done for `case_specialist` in these places only:

| File | What to add |
|---|---|
| `dclab_rnd/models/settings.py` (near line 167) | A `Purpose` entry for `case_intelligence`, same tier, seconds, `cell_values` and `keeps_text` as `case_specialist`, with its own one-line description. |
| `dclab_rnd/agents/harness.py` (`MODEL_USES`, near line 92) | `"case_intelligence": "a role the lead agent's intelligence tool asks once: its findings are checked against computed evidence"` |
| `dclab_rnd/store/coverage.py` (near line 67) | Add `"purpose:case_intelligence"` to the `"case"` capture tuple. |
| `dclab_rnd/store/sets/llm.py` (near line 47) | Map `"case_intelligence"` to the same value as `"case_specialist"`. |

Do not add it anywhere else. If a test asks for another place, write it in `HANDBACK.md` and add it there.

**Tests** (`tests/test_home_intelligence.py`; use the T2 replica for the frame and a `Script` for the model):
- `test_mode_reads_the_flag` (unset → off, `tools` → tools, `1` → roles)
- `test_tools_mode_returns_evidence_and_no_reports`
- `test_roles_mode_asks_three_roles_in_order`
- `test_only_the_analyst_is_shown_the_evidence` (look at the messages the `Script` received)
- `test_a_role_that_fails_gives_a_failed_report_and_the_others_still_run`
- `test_without_a_table_the_analyst_is_skipped_and_no_fact_survives`
- `test_a_number_in_a_role_answer_that_no_evidence_holds_is_dropped`
- `test_the_new_module_never_imports_teamspace`: walk every `.py` file in `dclab_rnd/home_intelligence/` with `ast`,
  collect `Import` and `ImportFrom` nodes (also the relative ones, where `node.level > 0`), and fail if any module
  path contains `teamspace`. Copy the idea from `tests/test_core_contracts.py` lines 64 to 90.

**Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py test_home_intelligence test_agent_harness test_prompt_gate test_harness_gate
```
`test_prompt_gate` and `test_harness_gate` will fail because a prompt file was added. That is expected. Then run:
```bash
DCLAB_NO_LIVE_MODELS=1 make agent-eval
DCLAB_NO_LIVE_MODELS=1 make harness-eval
```
and run the four test files again.

**Done when:** the four test files pass after the two `make` commands.

---

### T5. Give the tool to the Lead

**Goal:** the Lead can call `intelligence` when the switch is on. With the switch off, the Lead is offered exactly
today's tools.

**Read first, in `dclab_rnd/agents/case_agent.py`:**
- lines 54 to 55 (`TOOLS`), 316 to 371 (`CaseState`), 430 to 456 (`_data`: how a tool reads `state.frame` and adds to
  `state.computed`), 644 to 693 (`_consult`: the tool you copy), 1153 to 1172 (`MOVES`, `CaseGuard`), 1175 to 1256
  (`register`, `TOOL_NAMES`), 1348 (`BOOKKEEPING`), 1683, 1694 and 1707 (the three places a `Policy` gets its tools).
- `dclab_rnd/agents/case_session.py` lines 36 to 55 (`snapshot`) and the `restore` function below it.
- `dclab_rnd/draft/case_home.py` lines 53 to 112 (`_state`: where the model clients are created).

**Change `case_agent.py`:**

1. Add `"intelligence"` at the end of `TOOLS`.
2. Add `"intelligence": "case.intelligence"` to `MOVES`.
3. Add `"intelligence"` to the names inside `BOOKKEEPING`.
4. Add three fields to `CaseState`:
   `intelligence_client: Any = None`, `intelligence: Optional[dict[str, Any]] = None`, `intelligence_calls: int = 0`.
5. Add the handler:
   ```python
   def _intelligence(state, question, date_column=None, value_column=None, customer_column=None,
                     by_columns=None, flag_columns=None, cost_column=None):
       from ..home_intelligence import run as HI, settings as HS
       mode = HS.mode()
       if mode == "off":
           return {"error": "this tool is switched off"}
       if state.intelligence_calls >= HS.MAX_CALLS_A_CASE:
           return {"error": "the specialists were already asked on this case: use what they found"}
       if mode == "roles" and state.intelligence_client is None:
           return {"error": "no specialist can be asked in this session"}
       # column names: keep only the ones that are real columns of state.frame; if date_column or value_column
       # is missing or not a column, run with no table (frame=None)
       # case_text and table_text: build them exactly as _consult does (ctx_.case_text, ctx_.table_text)
       result = HI.run(question, mode=mode, frame=..., columns=..., client=state.intelligence_client,
                       case_text=..., table_text=..., said=list(state.case.words))
       state.intelligence_calls += 1
       state.intelligence = result
       for e in result["evidence"]:
           state.computed.append(json.dumps(e, default=str))      # these numbers may now be said
       # write a short digest into the case file, only if there is room under the cap of 30 findings:
       #   room = 30 - len(state.case.findings); take intelligence.digest(result, limit=min(4, room))
       #   write them like _consult does: C.CaseFinding(by="intel:<role>", kind=..., text=..., field="")
       #   map type "fact" -> kind "fact"; "hypothesis" and "inference" -> "assumption"; "unknown" -> "question"
       return {"evidence": [{"id", "name", "text"} for each], "reports": result["reports"],
               "dropped": len(result["dropped"]),
               "next": "write to the person from the evidence and the findings; say what is confirmed and what is only a hypothesis"}
   ```
   Check the real allowed values of a finding's `kind` in `specialists.py` (`KINDS`) before you map.
6. In `register`, add one entry to `defs`:
   - name `"intelligence"`;
   - description: "Ask DCLab's specialists to look at a hard or unclear business problem from several views. Code
     first computes what changed in the shared table (revenue, buyers, orders per buyer, order value, segments,
     cohorts). Use it once, when the person's problem has more than one possible cause. Pass the table's real column
     names. Not for a simple question.";
   - properties: `question` (string), `date_column`, `value_column`, `customer_column`, `cost_column` (strings),
     `by_columns` and `flag_columns` (arrays of strings); required: `["question"]`;
   - handler `_intelligence`; effect `"write"`.
7. Add a helper and use it in the three `Policy` places:
   ```python
   def offered() -> tuple[str, ...]:
       from ..home_intelligence import settings as HS
       return TOOL_NAMES if HS.mode() != "off" else tuple(t for t in TOOL_NAMES if t != "case_intelligence")
   ```
   - line 1683: `policy = replace(POLICY, tools=offered(), ...)` (keep the existing `max_tokens` logic);
   - lines 1694 and 1707: `tools=offered()` instead of `tools=TOOL_NAMES`.

**Change `case_session.py`:** add `"intelligence"` and `"intelligence_calls"` to `snapshot`, and read them back in
`restore` (default `None` and `0`).

**Change `case_home.py`:** next to the line that creates `state.specialist_client` (line 109), add
`state.intelligence_client = self.models.client("case_intelligence", draft_id=draft["id"]) if self.models is not None else None`.
Nothing else: the whole state is already saved in `draft["case_state"]` through `snapshot`, so the two new fields you
added to `snapshot` and `restore` are saved and loaded with it.

**Update two existing tests that list the Lead's tools:** `tests/test_case_agent.py` near line 183 and
`tests/test_agents_runtime.py` near line 148. Add the new tool name there. Change nothing else in them.

**New tests** (add a class `IntelligenceToolTests` to `tests/test_home_intelligence.py`):
- `test_with_the_flag_off_the_lead_is_offered_todays_tools` (`offered()` has no `case_intelligence`; its length is
  `len(TOOL_NAMES) - 1`)
- `test_with_the_flag_on_the_lead_is_offered_the_tool`
- `test_the_tool_adds_evidence_to_what_may_be_said` (after the call, a number from an evidence text passes
  `case_agent.checked`)
- `test_a_third_call_on_one_case_is_refused`
- `test_the_case_file_never_passes_thirty_findings`
- `test_the_result_is_saved_and_restored_with_the_session` (`snapshot` then `restore`)
- `test_a_full_turn_with_the_flag_on_ends_with_one_answer` (a `Script` where the Lead calls the tool, then writes an
  answer; the answer's numbers all come from evidence)
- `test_a_full_turn_with_the_flag_off_is_unchanged` (the `Script` never sees the tool in its tool list)

**Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 make agent-eval
DCLAB_NO_LIVE_MODELS=1 make harness-eval
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py test_home_intelligence test_case_agent test_agents_runtime test_agent_harness test_prompt_gate test_harness_gate
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --pg test_home_intelligence test_case_agent
```

**Done when:** all pass.

**Note for the reviewer:** adding a tool changes the pinned hash of the Lead's tool list once. That is why the two
`make` commands run. With the flag off the model is still offered exactly today's tools.

---

### T6. Settings, capture proof, records

**Goal:** the switch is documented, and a test proves the new model calls are recorded with their thinking request.

**Read first:** `tests/test_case_agent.py` lines 572 to 608 (`CaptureTests`) and 660 to 672 (a specialist's exchange is
kept); `.env.example` lines 127 to 140 (how a setting is documented).

**Change:**
- `.env.example`: add
  `# DCLAB_HOME_INTELLIGENCE=          # unset or 0: off. tools: the lead agent's intelligence tool returns computed evidence only. 1: evidence and three specialist views (framing, business, analyst). Off until its comparison passes.`
- `docs/guides/CORE_INTEGRATION.md`: add a short section "Intelligence (kpi_analysis, intelligence)": what the two
  core modules do and that a host passes its own `unsaid` function to `validate`.

**New tests** (class `CaptureTests` in `tests/test_home_intelligence.py`):
- `test_every_role_call_is_kept_as_an_exchange_with_its_purpose`: run the tool in `roles` mode through a real
  `Gateway` with a scripted transport (copy lines 660 to 672 of `tests/test_case_agent.py`); assert three kept
  exchanges with `purpose == "case_intelligence"`, and that each has the key `thinking_asked`.
- `test_a_failed_role_call_is_kept_too`

**Development record:** create
`research/llm-fine-tuning/experiments/dev_trajectories/records/home-intelligence-thin-slice.json`. Copy the shape of
`team-b42-clarify-options.json` in the same folder (same keys). Fill it with what you built in T1 to T6. Every `gate`
text must be at least 10 characters. Every finding's `severity` must be one of: `behaviour`, `latent`, `gap`, `flaky`,
`privacy`, `leakage`. Then run `DCLAB_NO_LIVE_MODELS=1 make sft-dev`.

**Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py test_home_intelligence test_config test_thinking_everywhere
DCLAB_NO_LIVE_MODELS=1 make check-quick
```

**Done when:** the three files pass. For `make check-quick`: list every failing test in `HANDBACK.md`, and fix only
the ones your changes caused.

---

### T7. The comparison (build it; do NOT run it live)

**Goal:** a small program that can compare four ways of answering on the invented cases, and score them by code.

**Read first:** `dclab_rnd/agent_eval/conversation_exam/arms.py` lines 352 to 400 (`CaseArm`: how a test harness
drives the Lead); `dclab_rnd/agent_eval/team_arms/gate.py` (how a gate is pinned by a hash file);
`dclab_rnd/agent_eval/team_arms/__init__.py` lines 630 to 665 (the command line: `plan`, `run`, `report`).

**Create** `dclab_rnd/agent_eval/home_intelligence/`:

| File | What it holds |
|---|---|
| `cases.py` | `dev_cases()`: 7 causes × seeds 1 to 4 = 28 cases. Each case: `id`, `cause`, the frame from `orders_decline.generate`, the column names, and the person's opening message. Openings: for `margin`, "My profit is down though sales look fine. Can you find out why?"; for all others, "My sales went down in the last three months and I think I am losing customers. Can you find out what happened?" |
| `arms.py` | `ARMS = ("model", "lead", "tools", "roles")`. `play(arm, case, clients) -> {"answer", "computed", "euros", "seconds"}`. `model`: one model call with the opening and the column names, no table. `lead`, `tools`, `roles`: build a `CaseState` as `CaseArm.open` does, set `DCLAB_HOME_INTELLIGENCE` to `0`, `tools` or `1` for the play (restore the old value after), and call `case_agent.turn`. If the Lead only asks a question, send once "Please go ahead with what you have." and take that answer. |
| `reader.py` | `LABELS` = the 7 causes plus `"cannot_tell"` and `"other"`. `read(client, answer) -> label`: one model call that returns JSON `{"main_cause": "<one label>"}`. The prompt lists the labels with a one-line meaning each and shows the answer. An invalid reply gives `"other"`. |
| `score.py` | `score(case, played, label) -> dict`: `correct` (label equals the case's cause), `false_cause` (label is one of the 7 causes and is not the case's cause), `invented` (count of `cited.unsaid(answer, [opening] + played["computed"] + evidence texts from kpi_analysis.run_all on the case frame)`), `euros`, `seconds`. `summary(rows)`: per arm, the rates and the means. `paired_lower_bound(rows, arm_x, arm_y, seed=0)`: a bootstrap over cases (2,000 draws) of `correct(x) − correct(y)`; return the 5th percentile. |
| `gate.py` | `TERMS` (below), `fingerprint()`, `pin()`, `pinned()`, `judge(summary) -> dict`. Copy the hash pattern from `team_arms/gate.py`. |
| `__init__.py`, `__main__.py` | Commands `plan` (prints the number of plays and model calls; spends nothing), `run` (needs `--yes` and `--cap-eur` at most 0.5; refuses any model id that contains `claude` or `anthropic`; writes everything under `agent_runs/live_tests/<time>-home-intelligence-dev/`), `report`. |
| `gate_v1.sha256` | Written by `pin()`. |

**`TERMS` (fixed before any run):**
```python
TERMS = {
  "name": "home_intelligence_gate_v1",
  "keep_roles_if":  {"correct_roles_minus_tools_at_least": 0.10, "paired_lower_bound_above": 0.0,
                     "invented_in_roles": 0, "false_cause_roles_not_above_tools": True,
                     "mean_seconds_roles_at_most": 240, "mean_euros_roles_at_most": 0.06},
  "keep_tools_if":  {"correct_tools_minus_lead_at_least": 0.10, "paired_lower_bound_above": 0.0,
                     "invented_in_tools": 0},
  "tie": "the simpler way wins: lead before tools before roles",
}
```

**Tests** (`tests/test_home_intelligence_eval.py`; scripted transports only):
- `test_there_are_28_dev_cases_and_no_case_text_names_its_cause`
- `test_a_scripted_right_answer_scores_correct`
- `test_a_scripted_wrong_cause_scores_a_false_cause`
- `test_an_invented_number_is_counted`
- `test_the_reader_gives_other_on_a_bad_reply`
- `test_the_flag_is_restored_after_a_play`
- `test_the_committed_pin_matches_the_terms_and_code`
- `test_run_refuses_without_yes`, `test_run_refuses_a_cap_above_half_a_euro`, `test_run_refuses_a_claude_model`
- `test_the_gate_keeps_the_simpler_way_on_a_tie`

**Run:**
```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.agent_eval.home_intelligence plan
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py test_home_intelligence_eval test_home_intelligence
```

**Done when:** `plan` prints its numbers and the tests pass. **Stop here. Do not use the `run` command.**

---

## Part 4. After the smaller model finishes

**What Claude reviews** (when the owner asks):
1. No file in `dclab_rnd/teamspace/` changed; the import test exists and passes.
2. With the flag off, `offered()` equals today's tool list, and a scripted turn is unchanged.
3. Every number the Lead may say after the tool comes from an evidence item; a role's own words never add numbers.
4. The pack folder of a case never holds its cause.
5. `kpi_analysis` and `intelligence` import nothing from `dclab_rnd`.
6. The gate is pinned and the `run` command was never used.
7. `HANDBACK.md` lists what was unsure. An independent `dclab-reviewer` pass on the whole change.

**Then, with the owner's go and OpenRouter credit (about EUR 3 to 5):** the live dev comparison of the four ways, on a
Qwen model, at most EUR 0.5 a run; check that thinking summaries were kept; `make archive` once.

**Then decide:**
- Roles beat "tools only" by the pinned rule → keep roles; next add a second round, parallel calls, then ML,
  Simulation and Decision roles one at a time, each with its own test.
- "Tools only" wins or ties → keep the analysis tool for the Lead, add no more roles.
- Neither beats today's Lead → leave the flag off and revise.

**Later, not in this guide:** the Home page's expandable report (`home.html`), a sealed set written by an independent
author and run once, new event kinds and training-set builders for the roles that are kept, fine-tuning.

## Risks, said plainly

- **Roles may not help.** That is a valid result. The `tools` value exists so we learn it cheaply.
- **The analyst has little freedom in this first version**: it reads fixed analyses instead of choosing its own. This
  is a deliberate trade for safety and simplicity. If the roles show promise, giving the analyst a choice of analyses
  is the next step.
- **Roles run one after another**, so a turn that uses the tool can take 1 to 3 minutes. The comparison measures it.
- **The test data is invented.** A win on it does not prove a win on a real company's data.
- **Adding the tool changes one pinned hash once** (the Lead's tool list), even though the Lead is offered the same
  tools when the flag is off.
- **Live measurement needs OpenRouter credit.** T1 to T7 need none.
- **Disk is nearly full** (about 5 GB free). Stop any job when free space is under 2 GB.

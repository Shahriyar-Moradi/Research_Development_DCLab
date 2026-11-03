# DCLab Bench and DCLab Watch: how DCLab measures itself

Written 2026-11-03 for the owner's request: "a complete evaluation to monitor DCLab: our system, our AI, our agents, our
LLMs and our multi-agent system; tracking, tracing, the prompt version of each agent, and improvement, like the lead agent
against Claude alone." The plan has two halves:

- **DCLab Bench** (offline): is this version of DCLab better or worse than the last one, layer by layer?
- **DCLab Watch** (online): what is happening now, in every session, agent, model call and tool, and where does it break?

This guide describes what is built. The plan and its order are in the owner's plan of 2026-11-03 (B1 to B8, A1 to A3).

## 1. The layers

Every evaluation DCLab has belongs to one layer. The mapping is `dclab_rnd/bench/LAYER_OF`; a new campaign without a line
there fails `tests/test_bench.py`, so no result is silently left out.

| Layer | The question it answers | Its suites today |
|---|---|---|
| software | does it work and keep working | the product flows (`make product-e2e`), run by `make bench` |
| agents | judgment on traps, journeys, refusals | `agent_eval_v1`, `benchmark_v2` to `v4`, `harness_v1`, the journeys, the analyst, the investigator, the map roles |
| lead agent | Home conversations to the holdout number | `conversation_exam_v1`, `home_v1` |
| LLM roles | each role's model and prompt | `roles_v1`, `solver_v1`, `decision_llm_v1` (live per-request numbers come with Watch) |
| team | the multi-agent team | `team_v1`, `orchestra_v1`, `department_v1` |
| decision models | students and classifiers against the rule | `decision_models_v1`, `workflow_models_v1`, `understand_v1`, the students' exam and the rest |
| retrieval and memory | finding the right record, memory's value | `retrieval_v1`, `memory_retrieval_v1`, `memory_value_v1` and `v2` |
| data science | kinds acceptance, DCLab against FLAML, pitfalls, decisions | `kinds_v1`, `bakeoff_v1`, `pitfalls_v1`, the fraud and time decisions and the rest |
| system tasks | the whole system on one problem | `real_problems_v1`, `twin_v1` (the new fixed tasks come with B7) |
| external | DCLab against others | none yet (an MLE-bench-lite runner comes with B8) |

## 2. How a result is read

`dclab_core/bench.py` (dclab_core 0.153.0):

- `verdict_of` reads every verdict shape the suites wrote: a string (`"pass: 17 of 17"`), a dict with `pass`, `passed`
  or `status`, a top-level `result` or `passed`, or the `passes` list. The answer is pass, fail or **not scored**. A shape
  it does not know is not scored, never a failure.
- `counts` reads every `[k, n]` count up to three levels deep, with the interval the suite stored (`<name>_ci95`) or
  Wilson's 95%.
- `rows_of` turns one result into `BenchRow` rows (contract `dclab_core.bench_row.v1`), each with the split (dev, sealed,
  test, live), the gate, the cost, the model, the prompt and the commit.

## 3. How a layer is judged

`layer_status` reads the latest result of each track of each suite in the layer:

| State | When |
|---|---|
| red | one suite's latest result failed (the reasons name it) |
| amber | none failed, but one has no gate's verdict (a measurement without a gate) |
| thin | all scored ones passed, but fewer than 20 cases stand behind them |
| green | all passed, with 20 cases or more |
| empty | the layer has no result |

Each suite is read by track: its split (dev, sealed, test, live) and its point (the name after the result's id,
without split and version words). Inside a track the newest result is the one with the highest number, then the latest date.

There is no weighted total: a total would hide a red layer behind green ones. Sealed results are read and shown with
their date; the bench never runs one.

## 4. Commands and the page

```bash
make bench-read                 # each layer's state and why, from what is recorded
make bench                      # run the free suites, record the run, then read (LAYERS=agents,software to choose)
python -m dclab_rnd.bench read --json
```

`make bench` runs only scripted suites with `DCLAB_NO_LIVE_MODELS=1` and never a sealed or test split (a test checks the
command list). Each run is one file in `evidence/bench/runs/<time>_<commit>.json`, written once and never rewritten,
with each command's exit and time, the commit, whether the tree had changes, the machine's memory and free disk, and each
layer's state at that moment. The history is these files.

The Admin page's **Benchmark** tab (`GET /api/admin/bench`, owner only) shows the layers, the latest result of each
suite with its count and interval, and the recorded runs.

## 5. What the first reading says (2026-11-03)

1,001 results read in 10 layers. A suite is judged by **track**: one split and one decision point (`track_of`), so a
later practice run never hides a failed sealed one, and one decision point's pass never hides another's failure.

| Layer | State | Failing tracks | For example |
|---|---|---:|---|
| agents | red | 8 | the programme journeys, sealed v2: 4 of 10; the sealed v6 journeys: 21 of 30 |
| lead agent | red | 4 | the conversation exam, sealed v3; Home, sealed v6 |
| LLM roles | red | 1 | the critic role's gate |
| team | red | 1 | the debate arm |
| decision models | red | 27 | 12 of the 14 decision-model gates, and the understanding students |
| data science | red | 7 | the first two fraud-decision gates, the search on/off decision |
| retrieval and memory | amber | 0 | measurements without a gate (recall, memory value) |
| system tasks | amber | 0 | the real problems and the twin runs carry no gate |
| software, external | empty | 0 | until the first `make bench`; no external runner yet |

This is the honest state, not a defect of the bench. A track keeps its failure until a later result of the same track
passes: a retired decision point stays red until it is removed from its campaign or passes again.

## 6. Agent versions (B5)

`python -m dclab_rnd.bench versions [--agent case_agent]`, and the **Agent versions** table on the Benchmark tab
(`GET /api/admin/bench/versions`). An agent's version is its prompt file's content hash (`prompts.fingerprints()`).

- **History:** every commit that changed the prompt, with the hash it had after it.
- **Checks:** the scripted suites (`make agent-eval`, the prompt gate) record every prompt's hash, but they send no prompt to a
  model. A check says a version was checked, never measured.
- **Live results:** the suites that send the prompt to a model (`SUITE_PROMPTS`: the lead-agent exam, Home, the journeys) record
  no hash, so each result is placed in the version committed before its date. A run on uncommitted edits lands in the version
  before them; a suite missing from the map measures no prompt.
- **Measured now:** a live result on the version in force now. On 2026-11-04 the lead agent's newest version (2 November, the
  act-first fixes) had none.

## 7. A against B (B6)

`python -m dclab_rnd.bench compare --result conversation_exam_v1/CEX-032_sealed_v3 --a plain --b case [--gate gate.json]` pairs two
arms of one result by person, per metric (`dclab_core.bench.paired`): both means, B's difference with a 95% bootstrap
interval, wins, losses, ties, and McNemar for yes/no metrics. Unscored or failed rows are left out and counted. Each comparison
is written once to `evidence/bench/compare/`; a gate file's hash and date are kept beside the result's date, so whether it was
written first can be checked.

On sealed v3 (12 people), the lead agent against Claude alone: 16.5 fewer safety failures per person (7.4 to 28.2), but 2.25
more questions (0.998 to 3.67 more), 0.83 more questions already answered, and EUR 0.13 more per person; understanding and the
right route are higher but their intervals include no difference.

## 8. Monitors and the bug loop (B8)

`python -m dclab_rnd.bench monitors [--days 7]` and the **Monitors** panel (`GET /api/admin/bench/monitors`): per agent purpose,
from the gateway's usage log, the requests, failures with the failure rate's Wilson interval, the median and 95th-percentile
seconds, tokens and euro. Fixed rules: red when the failure rate's lower bound is above 5% on 20 requests or more; amber when
95% of requests take over a minute or it spends over EUR 5 a day.

The bug loop's stations: `make datasets` turns sessions the failure detectors flagged into candidates; a reviewer accepts or
rejects each in `datasets/failures/reviews.json`; `make harness-eval` replays the accepted ones as regression cases. The panel
counts each station. On 2026-11-04 the loop was empty: no candidates had been built yet.

## 9. System tasks (B7)

`python -m dclab_rnd.agent_eval.system_tasks --split dev|sealed` (campaign `system_tasks_v1`, the system tasks layer). Each task
is one supported kind's acceptance on one seed: a project built from a seeded replica with that kind's traps planted, the five
stages run by the engine under the workflow graph, and every decision checked. No model, no euro.

- The kinds that count are the 15 supported when the suite was written, fixed in code (`KINDS`).
- Dev seeds 3 and 4 are new to every earlier run (the kinds campaign used 0, 1 and 2). Sealed seeds 731 and 732 are pinned by a
  hash and read once; a second sealed run is refused.
- `system_gate_v1`, fixed in code before any run: a task passes when every decision of its kind holds; the suite passes when the
  share of tasks passed has a 95% Wilson lower bound of 0.8 or more and no kind fails on every seed.

## 10. External benchmarks

`python -m dclab_rnd.bench external check|plan`. A real MLE-bench run needs the MLE-bench grader, the competitions' data (Kaggle,
the owner's key), at least 20 GB free and competitions chosen in `dclab_rnd/bench/external/tasks.json`; `check` names what is
missing (on 2026-11-04: all of it) and a run is refused until everything is here and the owner gives `--yes` and `--cap-eur`.

## 11. What comes next (the plan)

- **Watch (B4):** OpenTelemetry tracing of session, agent, step, model call and tool, sent to MLflow (always on, light,
  also the ML runs) and to a local Langfuse (agent sessions, on demand). No table cell, no sealed case, consent checked.
- **Agent versions (B5):** prompt, tools, model and routing per agent, with a timeline of score changes.
- **Comparison (B6):** "A against B" on the same cases, as the lead agent against Claude alone, made general.
- **Monitors and the bug loop (B8):** a failing trace becomes a reviewed regression case.
- **System tasks (B7):** fixed end-to-end tasks, dev then a sealed set read once.

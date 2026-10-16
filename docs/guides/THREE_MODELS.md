# DCLab's three models: what each learns, where its data comes from, and the service

The owner, 2026-10-17. DCLab builds three kinds of model, for DCLab's own use first and then as a service through
which other companies and developers train and fine-tune their own. Every one of them learns only from data DCLab
captures and stores itself, verified, with its origin, consent and split on every row. This page maps each model to
its data as the code holds it today (counts from `research/llm-fine-tuning/experiments/lake/MANIFEST.json`), what is
missing, and the packages that close the gaps. The plans behind it are `PHASES_4_5_MODELS_PLAN.md` (26.x, 27.x) and
the decision plan of 2026-10-13.

## The three models

| | Model 1: the solver LLM | Model 2: the decision models | Model 3: the combination |
|---|---|---|---|
| What it is | A language model (Qwen3 8B class, LoRA) that understands an ML problem, reasons step by step in a `<think>` block, and lays out the solution workflow: the problem card, the data's traps, the split, the family, the evaluation, the next experiment, the report | Small, fast, calibrated classifiers, one per decision point (a column's role, an identifier, a leak, the task, the family, the next question, the route…), and a learned AutoML that picks the model family from a table's meta-features; the line that must stand beside XGBoost and random forests | The workflow model (Model 1, or Claude until Model 1 measures better) plans and talks, and at every typed decision calls Model 2 as a tool instead of reasoning it out in text: more decisions, faster, each checked by the validator |
| Kinds of problem | Tabular (binary, multiclass, regression), fraud and rare events, anomaly, forecasting, time to event, uplift, ranking, segmentation, streams, places, related tables, long text, images | The same, at each kind's decision points | The same |
| Served by | The gateway's purposes (`solver_answer`, then the roles), in shadow first (27.8) | `ledger/routing.py`, in shadow first; acts only after its sealed gate and its shadow measurement (22.6) | `decide_point` at Home, `decide` for the intern and `/mcp` (26.5); routing per point |
| Judged on | The solver benchmark (27.6, SLV), the journeys, the agent suite | `decision_gate_v1` on a sealed set read once (DMG, DWM) | The live arms on the dev journeys (26.5: the rules, the workflow model alone, the two together) |

## How the data is captured, stored and made clean (one path for all three)

1. **Captured where it happens.** Every session writes an append-only experience log (19.2: requests, answers, agent
   steps, stage runs with their run records), and every decision DCLab makes writes a decision record (19.7: the point,
   the state as bands and counts, never a table value, every answer, the one used) with labels added later from people,
   benchmarks and measured outcomes (19.8). Agent tool calls, stops and the Home agent's questions are decisions too.
2. **Allowed to leave only under consent.** A workspace's records become training data only while its `train` consent
   is on (19.10, 23.1: each event keeps the consent it was written under); text is redacted; cells never leave.
3. **Built into sets.** `make datasets` writes each workspace's sets (decisions, conversations, trajectories,
   workflows, outcomes, episodes, problem maps, routes) with a manifest (19.12).
4. **Gathered in the lake.** `make lake` (26.1) reads those sets, DCLab's invented decision benchmarks (22.1), OpenML's
   real tables (26.2), verified self-play (27.3), the research corpus and the build history, into bronze (inputs and
   hashes), silver (normalised, with provenance) and gold (one set per decision point and per task family, split by
   whole dataset or project).
5. **Cleaned and scored** (27.5): a decision's features must be a pure state, no personal data, near duplicates and
   anything an evaluation reads kept out of training, a quality score per example, a dataset card per set.
6. **Exported for a run.** `python -m dclab_rnd.lake export --target decisions|solver` writes the chat files a LoRA run
   reads (`TRAINING.md`); never a sealed or evaluation-only row. Claude's answers are evaluation only (the owner's
   decision, Part E.1).

## Model 1: the solver LLM

**What it must learn (the task families of `dclab_core.lake.FAMILIES`):** understand the problem, ask the right
question, read a data card, plan the EDA, judge leakage against the prediction moment, design the split, type the
problem, rank the algorithms, plan features, design the evaluation, write the pipeline as code, read results, choose
the next experiment, write the report, drive DCLab's tools as an agent, and lay out the whole solution workflow.

**What is in gold today:** 5 of 19 families hold verified examples, all from self-play on invented problems:
problem type 72, leakage judgment 168, split design 140, algorithm choice 174, reading results 176 (730 rows); the
evidence answers of the research corpus 330. The build history (2,591) is evaluation only.

**What is missing, and the package that adds it:**

| Gap | Why it matters | Package |
|---|---|---|
| Step-by-step reasoning (`<think>`) | Today no trainable example has a reasoning trace: Claude's thinking is evaluation only. The reasoning must come from DCLab's own verified records: each decision's reason, the evidence it cites, the engine's measured numbers, assembled into a trace that is checked against the replica's truth | 27.9 (new): verified reasoning traces from DCLab's records |
| The solution workflow end to end | The family `solution_design` is empty: the problem card, the data card, the solution workflow (20.6) and the pipeline exist for every verified self-play problem and every built project, but are not yet examples | 27.3 extended: one `solution_design` example per verified problem, its workflow validated |
| Understanding and asking | Comes from people's sessions (Home conversations); none recorded under train consent yet | 27.2, after the owner switches train consent on |
| Variety | Seeds of one replica repeat in bands; 1,060 distinct self-play examples against 2,000 asked | 27.3: more wording, names, sizes and traps per kind |
| Images | The engine runs every kind but images | 26.8 |
| Real problems | Only invented problems and OpenML tables today | 27.4 (Kaggle with the owner's key), testers (25.11) |

## Model 2: the decision models

**What it learns:** one classifier per decision point, from a state of bands to the right answer, plus the learned AutoML.

**What is in gold today:** column points (identifier, semantic role, time, group, category code, leak suspect):
about 63,000 decisions, 37,700 of them columns of about 1,200 real OpenML tables; table points (the outcome column,
the task) 1,130 from OpenML; the workflow points (family, ready, next question, route) from journeys (26.3, 2,100
labelled decisions of 151 written people). The learned AutoML reads the self-play screens (193 problems).

**What it measured:** generation 1 passed its sealed gate at 2 of 6 column points (semantic role, leak suspect);
the workflow points are judged once on a sealed set of 30 journeys (26.3); the learned AutoML beats always-XGBoost but
not yet the most frequent winner (AML-002).

**What is missing:** blind labels from people (decision 4 of the plan) to promote beyond shadow; decisions recorded
on real sessions (27.2); generation 2, one small language model for every point (26.4: prepared, the run is the
owner's); the in-context tabular model (26.6b) and the measurement against tuned XGBoost and random forests on 30
real datasets (26.6).

## Model 3: the combination

**What it learns:** when the workflow model should hand a decision to a decision model, and what to do with the
answer. Its data is the trajectory: each agent step's state, the tool it called (`decide_point`, `decide`), the
answer, whether the validator accepted it, and the outcome the session reached. All of it is captured today (agent
tool choices, decisions with every answer, run records); it becomes training data through 27.2 once sessions are
recorded under train consent and each row says which model answered.

**What it measured:** 26.5 adds the third arm to the dev journeys (the rules; the workflow model alone; the workflow
model with the decision models), live, with a cap.

## The service: other companies and developers train their own

| Step | DCLab's own (today) | A company's own |
|---|---|---|
| Capture | Sessions, decisions, outcomes in the workspace | The same, in their workspace; nothing leaves it unless they turn `train` consent on for DCLab |
| Export | `make datasets`, `make lake` | `make own-models ARGS=export` (26.7): their labelled decisions in the gold shape, whole projects split, the chat rows for a LoRA run |
| Train | Generation 1 classifiers; the LoRA recipe for the LLMs (`TRAINING.md`) | Model 2: their own classifier stacked on DCLab's base model (26.7, built; the base never changes). Models 1 and 3: the same export and recipe on their own sessions (to build: 27.10) |
| Judge | The sealed gates | The same gate on their own sealed projects, read once |
| Serve | Routing per point, in shadow first | Their model behind their workspace's switch, never anyone else's |

What the service still needs: the export of a company's own sessions for Models 1 and 3 (27.10, the 26.7 path
extended to sessions), a training job runner on a GPU (the owner's infrastructure decision), and per-tenant model
storage and serving (the tuned tier per workspace).

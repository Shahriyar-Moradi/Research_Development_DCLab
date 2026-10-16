# Phases 4 and 5: DCLab's own models

The owner, 2026-10-16. Phase 4: combine DCLab's decision models with a fine-tuned workflow model so they boost DCLab
together; train decision models for the workflow's own decisions too, not only the data-science ones; build a
classifier from the decision-model line that can stand beside XGBoost and random forests; let every company and
machine-learning engineer fine-tune their own decision models from DCLab's; extend to image classification and other
uses. Phase 5: fine-tune a language model (large or small) that is very good at solving machine-learning problems,
with all the data it needs collected under the hood of DCLab, organised, structured and clean.

This plan builds on the owner's *DCLab Decision Model Plan* (2026-10-13, 13 pages: the decision points, the six label
sources, the sample record, generations 1 and 2, the R3 gates, the third backend) and on what was built since. It
does not restate that plan; it says what is done, what is left of it, and what phases 4 and 5 add.

## Part A. Where things stand (2026-10-16)

| What | State | Evidence |
|---|---|---|
| Generated decision benchmarks (source 2 of the decision plan) | 480 invented tables, 44,640 labelled decisions on 6 column points, a sealed partition | `decisions/generated_v1/MANIFEST.json` |
| Generation 1 classifiers, one per point | fitted; judged once on the sealed partition by `decision_gate_v1` | `evidence/campaigns/decision_models_v1/` |
| Their verdicts | **2 of 6 pass**: `column.semantic_role` (accuracy 0.937, +0.10 over the rule) and `feature.leakage_suspect` (0.971, +0.05). `column.is_identifier` and `column.group` fail because the rule is already perfect on generated tables; `column.category_code` and `column.time` fail their precision or recall bounds | DMG-001…006 |
| Shadow and routing | every recorded decision answered in shadow (22.4); one audited switch per point (22.6); used at four places with a validator (22.9) | `ledger/shadow.py`, `ledger/routing.py`, `ledger/use.py` |
| Generation 2 (one small language model for all points) | the export (33,666 rows: chat JSONL, typed, parquet; sealed rows never written), the answerer, the fine-tuning recipe. **Not trained**: the run is the owner's | `decisions/export_v1/`, `decisions/finetune/README.md`, DLM-001 |
| Workflow decision points (17 of the ledger's 45) | recorded on every session; a measured comparison of models against the rules on the journeys (DWF-004: the rules still serve) | `decision_workflow_v1` |
| Training sets built under consent | builders for decisions, conversations, preferences, trajectories, workflows, failures, outcomes, episodes, model calls, problem maps, routes and the twin's sets; `make datasets` writes what consent allows | `dclab_core/datasets.py` (manifest v6) |
| Claude beside DCLab | Home sessions and six blind twin runs, both sides step by step; on the solution's columns Claude 5 of 6, the rules 1 of 6 | TWN-002 |
| SFT corpora | evidence-grounded answers v3: 330; DCLab's own build history: 2,560 | `sft/out_v3/`, `sft/out_dev/` |

What is **not** there yet: real tables at scale (source 1, public datasets), blind labels from people (source 6), any
trained language model of DCLab's, a single clean store that every model reads, and any measurement of a model that
*solves* a problem end to end.

## Part B. Principles (both phases)

1. **Models propose; the validator and the gates decide.** No model reads holdout rows, computes a metric, picks a
   winner, builds a split or drops a column on its own. Every answer passes its point's validator; a refused or
   abstaining answer falls back to the rule. This does not change for any model in this plan.
2. **One training lake, one record shape per task, every row with its origin.** Each row says where its label came
   from (construction, replica, factory rule, production rule, person exposed, person blind, teacher model, measured
   outcome), its consent, its licence, its split group. Splits are by dataset or project, never by row.
3. **Sealed sets are read once.** Tuning uses dev; a sealed set scores a frozen model once, as the decision gate and the
   journeys already do.
4. **A model serves only after it measured better on the same cases**, through the switches that exist (22.6 for
   decision points, 21.4 for roles). It starts in shadow; it is demoted when it falls below its bar.
5. **Data rights first.** A customer's or tester's rows leave their workspace only under `train` consent, redacted, never
   cells. Public datasets only under licences that allow it. **Claude's outputs**: Anthropic's commercial terms restrict
   using them to build competing models; until the owner's legal check says otherwise, Claude's answers are kept as
   evaluation and comparison data and are **not** used as training targets (Part E, decision 1).
6. **Clean means verified.** A training example is kept only if what it claims checks out: the code runs and reproduces
   its recorded number, the decision matches its label source, the text cites what it uses (`cited.py`), and nothing of a
   sealed set or a holdout appears in it.

## Part C. Phase 4 — DCLab's decision models and the workflow model, together (packages 26.x)

### 26.1 The training lake (shared by both phases)
- **Delivers:** `dclab_core/lake.py` (the record shapes, versioned) and `dclab_rnd/lake/` (the builders):
  `lake/bronze/` raw exports of every source as captured; `lake/silver/` normalised, deduplicated, redacted, each row
  with origin, consent, licence and group; `lake/gold/<task>/<version>/` the training-ready sets with a manifest
  (counts by source, split and label origin; the hash of every input; what was dropped and why). The decision plan's
  sample record is the gold shape for decision points; Phase 5's example record (27.1) the one for language tasks.
  `make lake` rebuilds it from the workspaces' consented records, the generated sets, the replicas and the R&D tracks.
- **Done when:** every existing set (the 13 builders, the decision export, SFT v3, the dev records, the twin sets) is in
  gold through one command; no group in two splits; a test fails on a sealed row or a holdout value in any gold set.

- **Status: built (2026-10-16), the review's 12 findings fixed.** dclab-core 0.59.0 (`lake`: `DecisionSample`, `TrainingExample`,
  `GoldManifest`, `Provenance` with `from_sealed`, the origins, `gold_problems`); `dclab_rnd/lake/`, `make lake`. The lake reads a
  workspace only through the sets `make datasets` wrote under its train consent, while its switch is on and its sets are not stale;
  it is rebuilt whole each time; a training row identical to an evaluation row is dropped (the generated sets lost about a third of
  their train rows that way: their bands repeat across tables); Claude's answers and DCLab's build history (written in Claude Code
  sessions) are evaluation only. Left for 27.2: gold mappings of the conversation, episode, trajectory, workflow, map and route sets
  (they reach silver today); they hold rows only where a project's own consent is on.

### 26.2 Real tables: the public catalog (decision plan, source 1)
- **Delivers:** a builder that turns public dataset metadata (OpenML first: its task and feature metadata, openly
  licensed) into labelled rows for `column.semantic_role`, `column.is_identifier`, `task.kind`, `target.column`;
  the R&D's own ten UCI caches and two project tables first, OpenML after the owner's yes on the download and licences.
- **Done when:** at least 20,000 column rows from at least 1,000 real tables in gold, by dataset group; the generation 1
  classifiers retrained on generated + real tables and judged again on a sealed set **of real tables** (the generated
  sealed set measures only planted kinds).

- **Status: in progress (2026-10-16).** `dclab_rnd/lake/openml.py` (`python -m dclab_rnd.lake openml`): each dataset's own marks are
  the labels (row identifier, default target, column types), its data file read once to compute DCLab's bands and deleted, its
  licence kept; options sorted so the outcome is never in a learnable place. The fetch runs (resumable); the retraining on real tables
  and the sealed set of real tables follow when it is done.

### 26.3 Decision models for the workflow
- **Delivers:** generation 1 classifiers for the workflow points the ledger records (the next question, the next move,
  ask or act, the pack, the route of a sub-problem, when to stop), trained on the recorded sessions, the journeys and
  the replicas, with outcome labels where the outcome is known (the build passed, the holdout read, the person kept the
  answer); the same gate as the column points, a sealed set of journeys.
- **Done when:** each point has a card beside the rule's score; a point that does not beat the rule stays in shadow
  (that is the expected result for most points until real sessions accumulate).

### 26.4 Generation 2: one small language model for every decision
- **Delivers:** the LoRA fine-tune of the decision plan's generation 2 (a 1.5B to 4B open model) on the gold decision
  sets of 26.1 to 26.3, answering every point in its typed form with a probability read from the choice tokens and
  the same abstain band; the gateway purpose `decision_answer` serves it in shadow.
- **Done when:** it beats generation 1 on the language points (`feature.leakage_suspect`, `split.strategy`,
  `target.column`) on dev, is at least equal elsewhere, its calibration error is under 0.05, and it answers within the
  latency budget on the target hardware; then the sealed set once. The training run is the owner's (GPU, Part E).

- **Status: prepared (2026-10-16); the run is the owner's.** `python -m dclab_rnd.lake export --target decisions|solver` writes
  the training files from gold (never a sealed or evaluation-only row) for the existing LoRA script and MLX;
  `research/llm-fine-tuning/experiments/lake/TRAINING.md` is the runbook (bases, hardware, recipe, how a run is recorded and
  judged). First export: decisions 19,530 train and 5,436 validation rows (5,793 from OpenML tables).

### 26.5 The combination: decision models inside the workflow model's loop
- **Delivers:** the Home agent and the intern call the decision models as tools (the `decide` tool of 22.8 exists) at
  every typed point instead of reasoning it out in text; the workflow model (Phase 5's model when it exists, the
  configured model until then) plans and talks; the decision model answers the point; the validator checks it.
  A third arm on the journeys and the twin runs: **workflow model + decision models**, beside the rules and Claude.
- **Done when:** that arm is measured on the dev journeys and the twin runs against the rules and against the workflow
  model alone; it is switched on per point only where it measured better.

### 26.6 A classifier from the decision line that stands beside XGBoost and random forests
Two tracks, because "as good as XGBoost" is a claim to be measured, not assumed:
- **26.6a Learned AutoML (practical first).** A model that reads a table's meta-features (the data card) and chooses
  the family, the recipe and the starting hyperparameters (`training.families_budget` and the feature ladder as
  decisions), trained on every recorded run's outcome; the engine then fits what it chose. Compared on held-out real
  datasets against tuned XGBoost and random forest at the same time budget.
- **26.6b An in-context tabular model (research).** A transformer pretrained on many invented tables (the
  benchmark generator's priors, extended) that predicts a new table's labels in one pass, the approach of TabPFN-style
  models; strong on small tables in published work, weaker on large ones. A research track with its own GPU budget.
- **Done when (both):** on a sealed set of at least 30 real classification datasets never used to tune, the mean
  normalised score and the mean rank against tuned XGBoost and random forest, with bootstrap intervals by dataset;
  reported by table size, where it wins and where it loses. Neither is called better unless the interval says so.

### 26.7 Your own decision models: the fine-tuning kit for companies and engineers
- **Delivers:** a workspace exports its consented decisions in the gold shape; a fine-tune of DCLab's base decision
  model on them (LoRA, in their workspace or a dedicated job); the same gate on their own sealed set; a model card; their
  model behind their workspace's routing switch. Their rows never leave their workspace for DCLab's own training unless
  their consent is `train`.
- **Done when:** a tester workspace (25.11) fine-tunes, gates and serves its own model end to end, and the base model is
  untouched by it.

### 26.8 Beyond tables: images and other kinds
- **Delivers (research, after 26.1 to 26.5):** the decision points of image classification (labels' quality, near
  duplicates and the same subject across the split, the split by subject or site, augmentation, the pretrained backbone)
  as a kind in `dclab_core/kinds`, the engine's image machine (today every kind but images runs in the engine), replicas
  with planted image traps, and the same decision sets.
- **Done when:** the image kind runs a replica to its holdout number and its traps are caught; then the same path to
  decision models as the tabular points.

## Part D. Phase 5 — a model that solves machine-learning problems (packages 27.x)

### What the model must be able to do
Understand a problem from people's words (the problem card); read a data card and ask what matters; plan and read an
EDA; judge leakage against the prediction moment; design the split; type the problem and rank the algorithms; plan
features; design the evaluation; write the pipeline as code that runs; read results and choose the next experiment;
write a report with its uncertainty; and drive DCLab's tools step by step as an agent inside the validator. Each is a
**task family** with its own examples, its own measurement and its own share of the mix.

### Where the training data comes from

| Source | Task families | Today | Label quality | Rights |
|---|---|---|---|---|
| Sessions in DCLab (Home, intern, projects): episodes, conversations, decisions, outcomes | understanding, questions, decisions, next move, agent steps | the owner's sessions and tests; grows with testers | people's words and kept answers; outcomes where measured | workspace consent `train` |
| Verified synthetic solving (27.3) | every family, end to end | none yet | verified by construction: the planted traps and the reproduced holdout | ours |
| The decision sets (26.1 to 26.3) | typed decisions | 44,640 generated | construction labels | ours |
| R&D research tracks, evidence campaigns, the knowledge base | evidence-grounded answers, reports, next experiment | SFT v3: 330 | measured results with citations | ours |
| Code exports (20.9) | pipelines as code | per project | verified: the code recomputes its holdout number | ours |
| DCLab's build history | engineering, review, fixing | 2,560 | reviewed records | ours (not ML solving: a small share) |
| Public ML corpora: OpenML runs and tasks, Kaggle where licensed | dataset to pipeline to score | not ingested | real scores, mixed quality | owner's yes per licence (Part E) |
| Claude's answers (twin runs, Home) | every family | 6 runs, 10 sessions | strong, unverified | **evaluation only** until legal check (Part B.5) |

### 27.1 The example record and the task families
- **Delivers:** `dclab_core/lake.py` gains `TrainingExample` v1: task family, input (the state the model sees, in the
  same text a served model will get), target (the answer, the code, the tool call), the rationale when there is one,
  the verification (what was checked and how), provenance (source, label origin, the record ids it came from), consent,
  licence, group, split. One builder per family into `lake/gold/solver/<family>/<version>/`.
- **Done when:** every source above that exists today is in gold through `make lake`, each family with its counts, and
  the mix is a manifest the training run reads.

- **Status: built with 26.1.** `TrainingExample` v1 and the families (with `solution_design`); gold per family.

### 27.2 Capture what is still missing, under the hood
- **Delivers:** the records the families need and the product does not keep yet: each stage's code as it ran (not
  only the export), the person's conclusion after the EDA (kept, not inferred), each agent step's input state, action,
  result and the outcome it led to, joined (the episodes builder joins most), the person's edits of a model's text
  (19.9's edit button) as preference pairs, and why a run was stopped. All under the existing consent and redaction.
- **Done when:** a test session in a tester workspace yields one complete episode per agent step with its outcome, and
  `make lake` turns it into examples of four families without hand work.

### 27.3 Verified synthetic solving ("self-play" with known answers)
- **Delivers:** a generator that takes invented problems (the benchmark generator and the replicas, extended to every
  engine kind) and solves them with DCLab's engine and agents, keeping a trajectory only when it is verified: the
  planted traps caught, the holdout read once and reproduced by the exported code, no validator refusal left open.
  Variations of wording, column names, sizes and traps make the variety real data lacks.
- **Done when:** at least 2,000 verified trajectories over at least 200 distinct invented problems across every engine
  kind, with a rejection log (what failed verification and why).

- **Status: built (2026-10-16); the volume run follows.** `dclab_rnd/lake/selfplay.py` (`python -m dclab_rnd.lake selfplay`): every
  engine kind's replica by seed, solved by the engine with no model, kept only when every acceptance decision passes; five families
  (problem type, leakage judgment with the train-only signals the auditor measured, split design, algorithm choice, reading results),
  the naming families twice (names, and names masked inside free text too); a rejection log.

### 27.4 Public corpora (after the owner's yes)
- **Delivers:** OpenML tasks and runs (dataset metadata, flows, scores) as dataset-to-pipeline-to-score examples and as
  real-table rows for 26.2; Kaggle only where the licence allows, through the connector (the owner's key).
- **Done when:** the licences are recorded per dataset in the manifest, and the examples pass the same verification
  where they can (a pipeline is re-run on a sample before it is kept).

### 27.5 Clean, deduplicate, decontaminate, score
- **Delivers:** one cleaning pipeline for every gold set: exact and near-duplicate removal, redaction of personal data
  and cells (`dclab_core.redact`), decontamination against every benchmark and sealed set (no example shares a dataset,
  a replica seed or a near-identical text with an evaluation case), a quality score per example (verified, cited,
  length, agreement), and a dataset card per set. `make solver-data` runs it end to end.
- **Done when:** a test fails on a duplicated example across splits, on a cell value, on an evaluation case in
  training; the card says what each set can and cannot claim.

### 27.6 The benchmark for the solver
- **Delivers:** what the model is judged on before any training: the agent judgment suite (planted traps), benchmark v4
  (108 cases by kind, dev and sealed), the journeys and the harness suite, and a new model-driven set: unseen problems
  (invented and public) the model solves through DCLab's tools to a holdout number, scored on leaks caught, valid moves,
  the holdout score against the rules and against Claude, cost and time. The gate fixed before the runs.
- **Done when:** the configured model and the rules are measured on it as baselines, and its sealed part is frozen.

- **Status: built (2026-10-16).** `python -m dclab_rnd.lake benchmark --answerer rules|configured|tuned [--split dev|sealed]`:
  the self-play cases of dev or sealed (never train), scored against the replica's planted facts (the task, the leak and the
  identifier, the time column) with Wilson intervals, cost and the share answered; results in `evidence/campaigns/solver_v1/`; the
  sealed split read once per answerer and model. The gateway purpose `solver_answer` (its prompt `prompts/solver_answer.md`,
  the same system text self-play trains on). Left: the model-driven solving of whole problems through DCLab's tools.

### 27.7 Train and evaluate
- **Delivers:** the recipe and the runs: supervised fine-tuning on the gold mix (all families) → tool-use fine-tuning on
  the verified trajectories → preference tuning (DPO) from outcomes and people's edits and picks → optionally
  reinforcement learning with verifiable rewards in a sandbox (the reward: traps caught, valid moves, a holdout score
  reproduced, never a holdout read twice). A model card per version. The base model and size are the owner's choice
  (Part E); the runs are the owner's (GPU).
- **Done when:** a version beats the configured model or the rules on the dev part of 27.6 by the gate's margin; then the
  sealed part once.

### 27.8 Serve it where it measured better
- **Delivers:** the model behind the gateway as a purpose, in shadow first (21.4's shadow), then switched role by role
  (Home understanding, the intern, the expert design, the report) only where it measured better on the same
  conversations; the decision models of Phase 4 as its tools (26.5).
- **Done when:** one role is switched on by evidence, and switching it off restores the earlier model.

## Part E. Decisions for the owner

1. **Claude's outputs as training data.** Anthropic's commercial terms restrict using outputs to develop competing
   models. Recommended: keep them as evaluation and comparison only (the default in this plan) until a legal review
   says what is allowed; train on DCLab's own verified outputs, people's decisions and licensed public data.
2. **Base model and compute.** Generation 2 decision model: 1.5B to 4B (Qwen-class, a licence that allows commercial
   use). The solver: 7B to 8B for quality, which needs a GPU of about 24 GB for LoRA (a cloud GPU; the 8 GB M1 cannot
   train it); or 3B to 4B to start on smaller hardware. Which, and where it trains.
3. **Public data.** OpenML (openly licensed metadata and runs) first: yes or no on the download. Kaggle: only datasets
   whose licences allow it, with your key.
4. **Blind labels.** Promotion beyond shadow needs labels from people who did not see any model's answer (the decision
   plan: 30 to 300 per point). Who labels: you and testers, or a paid labeller for public and invented tables.
5. **Order.** Recommended: 26.1 (the lake) first, because both phases read it; then 26.2 and 27.3 in parallel (real
   tables, verified synthetic solving); then 27.5 and 27.6 (clean data and the benchmark) before any training run;
   then 26.4 and 27.7 (the training runs); 26.6 and 26.8 as research tracks beside them.

**The owner's answers (2026-10-16):** OpenML metadata, yes. Claude's outputs: evaluation only. Training: DCLab
prepares the data, the recipes, the benchmark and the run scripts; the owner runs the training on a GPU. Budget for
the live work of these phases: $20 (on top of the earlier $30). Order: as recommended in decision 5.

## Part F. What this plan does not claim

No model here is expected to be better than the rules or Claude until it is measured; most decision points will stay
in shadow until real tables and blind labels arrive (26.2, decision 4). A classifier from this line "as good as
XGBoost" is a hypothesis with a measurement attached (26.6), not a promise. Every number reported comes with its
interval and the cases behind it.

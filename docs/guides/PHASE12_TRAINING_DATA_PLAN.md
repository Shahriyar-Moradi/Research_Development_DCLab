# Plan: training data that teaches why, covers every agent and every kind of help, and never forgets (Phase 12, 36.x)

## Context

The owner (2026-10-28 night), repeated several times: good training data for DCLab's models needs the model's
**thinking and reasoning**, for past runs as well as future ones; data must be collected **for every agent**, not only
for machine-learning work (Home must understand people who want data analysis, data preparation, BI and business
analysis, or advice, and route them to the right tools); and the models trained on it **must not forget** old skills.

Where things stand (measured tonight):
- From now on, thinking is asked for and kept (`c8455bb5`): the training agents ask Claude for thinking, every purpose
  shown no cell values keeps its whole exchange, and cell purposes keep their thinking redacted. Claude returns a summary
  of its thinking; the owner accepted that limit.
- The past: of 2,095 saved model calls, 317 kept their thinking, 429 kept inputs without thinking, 723 thought but kept
  only "thought: true" (lost), 626 kept the shape only. The m4 set has 11,264 assistant turns and none with reasoning.
  m1, m2, m3 and m6 examples have no reasoning slot at all. `store/reasoning.py` already has `recorded`, `from_record`
  (reasoning written by code from a record), a checked `Teacher` (weight 0.5) and a capped `run`, but no set builder
  calls `from_record`, and the export trains only `recorded` text as thinking.
- Home already reads six kinds of help (`draft/intent.py` TRACKS: prediction, analysis, preparation, reporting,
  business, advice; H2, H3 by the session working on Home). The harness knows 18 agents (`agents/harness.AGENTS`) and
  16 roles.
- Nothing guards against forgetting yet: no frozen exam across training rounds, no replay set, no check that a new
  version keeps old skills.

**The goal behind it (the owner, 2026-10-28, "/goal"):** one of DCLab's most important goals. Every agent collects its
thinking and reasoning data, for every kind of problem and not only machine learning. Home is an intelligent chat that
understands what a person needs (a model, data analysis, data preparation, BI or business analysis, or advice), asks for
the context it lacks, and guides them to the right tools and solutions. First step on approval: save this goal in
memory.

## Who builds what

- **Part A, the thinking and reasoning backfill: the session working on Home** (owner-approved, $25 cap; agreed tonight,
  option b). Join saved thinking into the sets; `from_record` reasoning on every decision row (m1, m2, m3, m6) and on m5's
  rule answers; re-ask the 429 whole calls with thinking on (source "replayed", weight 0.5, about $8); teacher reasoning
  checked by `cited` for turns whose inputs can be rebuilt (half weight, about $15); otherwise an explicit "reasoning lost
  before 2026-10-28" note kept apart; a store gate that every LLM example has reasoning with its source or that note.
  Owns `store/sets/llm.py`, `store/reasoning.py`, `store/backfill.py`. This session reviews each diff.
- **Part B (every agent, every kind of help) and Part C (never forgetting): this session**, packages 36.1 to 36.7 below.
  Each with the house routine (test first, `make check-quick`, `dclab-reviewer`, dev record and `make sft-dev`,
  explicit-path commits built in a private index).
- The memory plan (35.4 to 35.8) continues beside it; 35.8's live runs stay within the memory budget.

## Part B: every agent, every kind of help

### 36.0 Thinking for every agent (taken into Part A by the Home session, 2026-10-28)
- Tonight's switch (`settings.THINKING_DEFAULT`) covers 15 purposes. Extend it to **every model purpose any agent or
  role uses** (`agents/harness.MODEL_USES`, `ROLES`, `AGENTS`): the team's roles (decomposer, map reviewer, data need,
  debate judge, solution challenger, brief checker, conversation reader), Home's intent reader, the leakage reviewer,
  stage notes, project answers, the store curator, the reflector, the lesson proposer, the journey writers. Only purposes
  that cannot think (embeddings) or whose output is a single label (`decision_answer`, kept as an exception with its
  reason, since students learn from labels) stay off, each with a written reason.
- A test: every purpose in `harness.MODEL_USES` asks for thinking unless it is in a short list of exceptions, each with
  its reason; a new agent purpose without thinking fails the suite. A second test: every exchange kept for such a
  purpose holds its thinking (whole, or redacted where cells were shown).
- Local models (Ollama, the owner's tuned models) already return their reasoning field and the gateway keeps it; the
  test covers them through the same exchange record.
- Cost: thinking is billed as output; live runs get dearer. Each run keeps its cap.

### 36.1 The coverage report and its gate (free)
- `dclab_rnd/store/coverage.py`, `make coverage`: one table per model type (m1 to m8) by **agent** (the 18 in
  `harness.AGENTS`) and by **kind of help** (the six tracks), with counts of runs, whole exchanges, examples with
  recorded thinking, with written reasoning (source by source), with an outcome or label, and Persian against English;
  the gaps listed by name. Written into the store's catalog (`training_records/STORE_CATALOG.json`) beside Part A's
  per-source counts.
- A test: every agent in `harness.AGENTS` and every track has a capture path (the events and records it leaves) or a
  written reason why not; a new agent without one fails the suite.
- Reuse: `store/catalog`, the keepers' inventory, `experience` event kinds, `ledger` points (`request.intent`).
- **Status (2026-10-28): built and reviewed.** `dclab_rnd/store/coverage.py`, `make coverage` (`ARGS=--no-write` prints
  only). On the real store (29 s, 0.9 GB): 12,103 model calls; Home's runs, 3,026 runs and 3,197 calls under prediction;
  no run or call yet under analysis, preparation, reporting, business or advice (the tracks Home began reading in H2);
  36 gaps by name. A helper session wrote the module and stopped on a model safety check; finished here.

### 36.2 Outcomes for the kinds of help that are not prediction (with the Home session)
- A non-prediction track ends with something a person can judge: the table, chart, report or answer produced, and the
  person's verdict (kept, changed, refused, asked again). Recorded as events on the conversation and as labels of the
  `request.intent` and routing decisions in the ledger: person labels gold, rule checks silver.
- In the store: m2 (workflow decisions) gets the routing decision with its outcome; m4 (workflow LLM) gets each
  track's conversation whole, with thinking, labelled by its outcome.
- The Home session owns `draft/` (its chat and routing); this package adds only the store side and the events'
  contract, agreed with it before any change.
- **Status (2026-10-28): the core, capture and store side built** (agreed with the Home session, which owns `draft/`):
  `dclab_core/track_outcome.py` (the Home session's verdict rule from the conversation's ordered actions; a refresh is no
  verdict, a thumb is), `ExperienceEvent` v16 `track.outcome`, `capture.track_outcome` (a person's kept or changed labels
  the draft's `request.exit` and `request.intent`; refused and asked again never label a route), the m4 Home conversation's
  `check.outcome` from `draft["outcome"]`, and coverage naming the event per track. Left open: Home's calls (the Home
  session's next step); refused and asked again reach m4 only, not m2 (an outcome field on m2 examples is a contract change);
  one verdict per conversation.

### 36.3 Written journeys for the sparse tracks (paid, owner's budget)
- Where 36.1 shows a track with few runs (likely analysis, BI, preparation), written people with their own problems
  (as the dev journeys are written, `agent_eval/journeys`), run through Home with thinking on, captured whole, kept
  silver and apart, sealed sets never touched. Cap printed before, spend reported after.

## Part C: never forgetting

### 36.4 A frozen exam per model type (free)
- `dclab_core/exam.py` (contract) and `dclab_rnd/store/exams.py`: for each model type, a fixed, versioned exam drawn
  once from its sealed rows plus general-skill sections (tool calling, Persian, code, refusals), hash-pinned, never in
  any training set (`Contamination` extended to it). Every model version sits it; its result is kept with the version.
- The no-forgetting gate: a new version is promoted only if no exam section drops below the previous version's score
  by more than its interval, beside the gate it already passes. Reuse `own_models` (cards, results per version),
  `dclab_core/decision_gate.py`, `models/tuned.py` (shadow, promote, retire in one change).

- **Status (2026-10-28): built and reviewed** (`dclab_core/exam.py`, `dclab_rnd/store/exams.py`, the guard in the set
  writer, the check in `models/tuned.py`). A dry draw on the real store: m1 1,097 items, m2 299, m3 2,409, m4 309, m5
  439, m6 483, m7 16 (only 16 distinct rows: weak), m8 none (no sealed row with an answer). No exam is drawn yet: drawing
  pins it for good, so it waits for the owner's word. Left open: a served model's later refusal does not stop it serving
  (only promotion is checked), a tuned model is matched by its served name, a new exam version starts its own line.

### 36.5 Replay sets (free)
- Each training round's set is the new data plus a fixed share of every earlier round (proposal: 30 %), stratified by
  model type, source, track and language; its manifest names every earlier version it draws from. Nothing is deleted
  (the store keeps everything); a round can be rebuilt exactly.
- Reuse: the store's set versions and manifests (`store/sets/__init__.write_set`), the split by group hash.

### 36.6 Verified code with its reasoning (free)
- The code examples that teach how: each engine kind's and each project's exported pipeline whose holdout number
  re-computes (`studio/code.py`, `make code-check`); dev commits whose tests passed (dev records with their diffs);
  self-play solutions that passed their checks. Each as problem → reasoning → code, into m5 (and m4 where it is workflow
  code), with its source and weight.

### 36.7 Versions and roll-back
- Every trained version kept with its data manifest (sets, replay share, exam version) and its exam result; a version
  that forgets is retired in one change (`models/tuned.retire`). Training itself waits for the M1 Max; this prepares
  the data and the gates.

## Order

1. Part A runs now (Home session, $25). This session reviews.
2. 36.0 (thinking for every agent), then 36.1, so the gaps are numbers before anything else is built.
3. 36.6 and 36.4, then 36.5 (all free, data only).
4. 36.2 with the Home session.
5. 36.3 last, on the owner's budget.

## Decisions for the owner

| Decision | Proposal |
|---|---|
| General-skill sections of the exams (tool calling, Persian) | built from DCLab's own records first; public sets only after a license check and your download approval |
| Replay share | 30 % of each round from earlier rounds |
| Budget for 36.3's written journeys | decided after 36.1 shows which tracks are thin |

## Verification

- 36.1: `make coverage` prints the tables; the gate test fails when an agent or a track loses its capture path
  (tested by removing one in a scratch registry).
- 36.4: an exam is refused if any of its rows appears in a training set; a planted "forgetting" version (one section's
  answers scrambled) is refused by the gate.
- 36.5: a round rebuilt from its manifest is byte-identical; the replay share holds per stratum.
- 36.6: every code example re-runs to its recorded number in a clean Python (`make code-check`).
- Part A (reviewed here): the store gate counts reasoning by source; no example without reasoning or an explicit note.
- Each package: its tests through `tests/run_parallel.py` (kept as test-run records), `make check-quick`, the
  reviewer, the dev record, `make sft-dev`; the nightly `make check-all`.

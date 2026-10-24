<!-- Approved by the owner 2026-10-24. The postponed harness-learning plan (H0–H6) is docs/guides/PHASE9_LEARNING_LOOP_PLAN.md with the H-revision kept in the owner's plan file. -->

# Plan: more abilities for DCLab — Decide, Build, Learn, Data (Phase 10, packages 34.x)

## Context

The owner (2026-10-24): leakage protection is DCLab's brake, not its engine. Buyers pay for better models built faster
and for better decisions made with them; "the agent that can't cheat" stays one guarantee among several. The owner's map
has six layers (1 Understand, 2 Data, 3 Build, 4 Trust, 5 Decide, 6 Learn) and asks for more ability in the weak ones.
The harness-learning phase (H3–H6, Appendix A) is postponed until the M1 Max arrives: on the 8 GB Air one homework of
the local model takes about 30 minutes, and the day's evidence showed fixing rules beat lessons.

What the code already has (checked 2026-10-24, so the plan builds on it instead of beside it):

- **Decide (5):** `dclab_core/decide.py` has `review_budget` (top-k by value per case and cost per review, interval by
  bootstrap over rows or entities, holdout read once) and `price_curve` / `price_variation`. A separate threshold chooser,
  `dclab_core/assess.py:_threshold` (budget share → stated FP/FN costs → best F1), feeds the engine's operating rule.
  `Solution.decision` (`dclab_rnd/studio/solution.py:29`) allows only `review` and `price`, binary only. The engine
  (`studio/engine.py:1086` `_decision`), the report (`dclab_core/report.py:825`) and challenge mode use it; the API, the
  web UI, the intern and Home do not. Missing: one decision core uniting costs, capacity and per-row value; multiclass
  and multi-action choices; targeting from uplift (`dclab_core/uplift.py` has Qini but no targeting); what-if.
- **Build (3):** the engine screens 5 families once each (`expansion/runner.py:510`), tunes the winner over 3 fixed
  parameter sets (`runner.py:734`), 3 folds, no ensembling, no time budget. Optuna and TabPFN are installed but used only
  in research scripts. No harness compares engines on model quality. The research champions (`research/tabular-
  classification/INDEX.md`) store no split protocol, so they are not a fair yardstick. `docs/guides/MLE_BENCH_LESSONS_PLAN.md`
  already plans 33.2 honest selection under many attempts, 33.3 a search tree, 33.4 more moves, 33.5 ensembles: this
  plan reuses them instead of a second set.
- **Learn (6):** `dclab_core/monitoring.py` (bins, PSI, `monitor()`) is written into every model's artifact
  (`engine.py:1415`); the exported service (`studio/service.py`, `service_template/app.py`) serves `/predict` but logs
  nothing and never calls `monitor`; no route takes new rows or per-row outcomes (`POST .../outcome` takes one business
  number, `POST .../review` anomaly labels only); no retrain, no champion/challenger; lessons (`dclab_rnd/lessons.py`)
  come from training runs only. `CORE_INTEGRATION.md:881` puts runtime monitoring in the main product.
- **Data (2):** data cards, `meet.check`, cleaning, synthetic data, connectors exist; the data-need agent
  (`dclab_core/mapping.py:134`) lists missing tables, columns and labels; active learning failed both its gates
  (ALQ-001, ALQ-002). Nothing says whether more rows or which labels would help.

Owner's decisions (2026-10-24): all four layers; FLAML first for the bake-off, AutoGluon later on the M1 Max, and our own
engine improved beside them (Optuna, ensembles); the main product runs customer models, so layer 6 is contracts in
`dclab_core` plus a working reference in the exported service, wired in the main product.

## Rules that bind every package

- The house discipline: failing test first, `make check-quick`, the `dclab-reviewer`, the dev-trajectory record and
  `make sft-dev` in the same commit; capture in every "Done when" (ledger points, events, whole records).
- Contracts and math in `dclab_core` (numpy, pandas, scikit-learn, pydantic only), wiring in `dclab_rnd`, an adapter and
  a `CORE_INTEGRATION.md` section for the main product; `make core-wheel-check` when the core changes.
- Trust stays the guarantee: every choice (operating point, model, retrain) made on out-of-fold or validation rows, the
  holdout read once after it, every number with its interval and its assumption written down.
- Measured, never asserted: each package's gate fixed before its run, on the replicas (planted truth) and the 12 sample
  datasets; a result below the gate is reported as such.
- 8 GB budgets: small default search budgets; AutoGluon and long searches wait for the M1 Max; no model loaded beside a
  store run, an archive or a gate.

## Order

| # | Package | Layer | Needs | Runs on the Air |
|---|---|---|---|---|
| 1 | 34.1 One decision core | 5 | — | yes |
| 2 | 34.5 The model bake-off (FLAML) | 3 | `pip install flaml` | yes, small budgets |
| 3 | 34.2 Targeting and several actions | 5 | 34.1 | yes |
| 4 | 34.3 What-if | 5 | 34.1 | yes |
| 5 | 34.4 Decisions in the product | 5 | 34.1–34.3 | yes |
| 6 | 34.6 Honest selection (= 33.2) | 3 | — | yes |
| 7 | 34.7 A wider search and ensembles (33.3–33.5) | 3 | 34.6 | small budgets |
| 8 | 34.8 An AutoML family behind DCLab's splits | 3 | 34.5 gate, 34.6 | FLAML yes; AutoGluon on the M1 Max |
| 9 | 34.9 Prediction and outcome log | 6 | — | yes |
| 10 | 34.10 Monitoring that runs | 6 | 34.9 | yes |
| 11 | 34.11 Retrain and champion/challenger | 6 | 34.9, 34.10 | yes |
| 12 | 34.12 Lessons from use | 6 | 34.10 | yes |
| 13 | 34.13 Would more data help? | 2 | — | yes |
| 14 | 34.14 What to collect next | 2 | 34.13 | yes |

## Layer 5: Decide

### 34.1 One decision core
- **Delivers:** `dclab_core/decide.py` gains `DecisionSpec` v1 (contract): the action; value of a true positive and cost
  of a false positive and a false negative (constants, or a column per row such as `order_value`); a capacity (a count
  per period or a share); and `choose_operating_point(y_oof, score_oof, spec, groups=None)` returning the point with the
  highest expected value, its 90 % bootstrap interval (rows or entities), the curve, the break-even and the assumption.
  `review_budget` and `assess._threshold`'s stated-cost path become special cases of it (same answers, tested); the
  engine's operating rule reads it. The main-product adapter maps it to `Objective.cost_false_positive/negative` and
  `cost_matrix` (`dclab_core/adapters/main_product.py:108`).
- **Done when:** a pure review budget gives `review_budget`'s answer exactly; stated costs give `assess`'s; a per-row value
  (`delivery_time`'s `order_value`) changes the chosen rows as brute force says; with capacity the chosen count never
  exceeds it; on `delivery_fraud` the holdout's realised value falls in the interval. Capture: ledger point
  `decision.operating_point` (the rule's answer, the person's choice).

### 34.2 Targeting and several actions
- **Delivers:** `target_by_uplift(uplift_oof, spec)` (whom to treat under a budget, by expected incremental value) and
  `choose_actions(expected_values_per_action, capacities)` (which of several actions per row, greedy by value with each
  action's capacity; exact for one capacity), multiclass by a cost matrix.
- **Done when:** on `campaign_uplift` (true uplift known per row) the targeting's regret against the oracle beats
  random and treat-all with its interval; a two-action replica's choice equals brute force on small sizes.

### 34.3 What-if
- **Delivers:** `what_if(result, changes)` re-scores a chosen decision under changed costs, values, capacity or base rate
  (prior shift, by reweighting the out-of-fold scores), no refit; a sensitivity table: which change of which assumption
  flips the decision, and by how much the value moves.
- **Done when:** doubling a cost moves the point as brute force says; a base-rate change matches a resampled replica;
  every answer says what was not refitted.

### 34.4 Decisions in the product
- **Delivers:** `Solution.decision` v2 (the spec of 34.1, binary and multiclass, the old `review` and `price` kept);
  the engine's final stage writes the decision; the report's decision section; `GET/POST /api/projects/{id}/decision`
  (the result and a what-if); a Decision page in the web app through `DC.client` with `DC.states` (the curve, the chosen
  point, what-if inputs); the intern tool `decision_what_if` (registered once, scope project); Home asks for the action,
  value and cost in words (16.9's open item); a `CORE_INTEGRATION.md` section.
- **Done when:** `make product-e2e` and `make browser-test` cover the page; the intern answers a what-if on a replica;
  the decision is in the solution report with its interval.

## Layer 3: Build

### 34.5 The model bake-off
- **Delivers:** `dclab_rnd/bakeoff.py`, `make bakeoff`: each engine on the same splits (`expansion/datasets.py`'s rules,
  holdout read once), the same metric, the same time budget: DCLab's engine as it is, FLAML (`pip install flaml`,
  owner approved), later AutoGluon (M1 Max). Datasets: the 10 UCI sets, Telco, the expansion samples (imbalanced fraud,
  multiclass, text plus tabular); HyperAck only with its leaks removed. Per dataset: holdout score with interval,
  minutes, peak memory; stored as an evidence campaign.
- **Done when:** the table exists for every dataset with intervals; the result decides 34.8 by a rule fixed before the
  run (for example: FLAML ahead by more than its interval on at least a third of the datasets).

### 34.6 Honest selection under many attempts (MLE plan 33.2)
- As written in `MLE_BENCH_LESSONS_PLAN.md` 33.2: search folds and a selection fold, the measured optimism in the record.
  Needed before any wider search, ours or FLAML's.

### 34.7 A wider search and ensembles (33.3–33.5)
- **Delivers:** per-family Optuna search with a budget (minutes, trials), CatBoost reconsidered as a family (its
  exclusion reason at `runner.py:11` re-checked), stacking of the top families on out-of-fold predictions reusing
  `dclab_core.compose`; all chosen through 34.6; the exported code reproduces it (`make code-check`).
- **Done when:** on the bake-off datasets the engine's holdout score rises with the measured optimism inside its
  interval, at a bounded number of minutes on the Air.

### 34.8 An AutoML family behind DCLab's splits
- **Delivers:** FLAML (then AutoGluon on the M1 Max) as one more family in the models stage, fitted inside the same
  folds with a time budget, chosen by the same rule, exported with its requirement; only if 34.5's gate says so.
- **Done when:** it is chosen only where it wins on the selection fold; code export reproduces its holdout number.

## Layer 6: Learn (contracts here, wiring in the main product)

### 34.9 Prediction and outcome log
- **Delivers:** `dclab_core/serving.py`: `PredictionRecord` (model id and version, a row key or its hash, the score,
  the decision taken, the time; never a cell value) and `OutcomeRecord` (the key, the true outcome, when it became
  known); the exported service logs every `/predict` to a local JSONL and takes `POST /outcome`.
- **Done when:** a batch scored and its outcomes posted join by key; no input value is in either log (tested).

### 34.10 Monitoring that runs
- **Delivers:** `monitor_log(plan, predictions, outcomes)`: `monitoring.monitor` on each logged batch plus live
  performance once outcomes arrive (label delay respected, interval by batch size); the service's `/monitor` report;
  alerts as events.
- **Done when:** a replica with planted drift raises the alert at the planted batch and not before; live performance
  matches the holdout's within its interval on undrifted batches.

### 34.11 Retrain and champion/challenger
- **Delivers:** `dclab_core/retrain.py`: a retrain policy (drift alert, performance drop, or schedule); the challenger
  trained on the newer labelled rows through the same engine; compared with the champion on the latest labelled window
  that neither trained on; promoted only when better beyond its interval; every comparison recorded; a studio job
  `POST /api/projects/{id}/retrain`.
- **Done when:** on a drifting replica the challenger is promoted after the drift and refused without it.

### 34.12 Lessons from use
- **Delivers:** drift alerts, promotions and refusals become lesson proposals in `lessons.py` (reviewed as today).
- **Done when:** a planted drift yields one cited lesson a reviewer can accept.

## Layer 2: Data

### 34.13 Would more data help?
- **Delivers:** a learning curve on the training folds (25, 50, 75, 100 % of rows, by group when grouped), its slope
  with an interval, and a plain answer: more rows likely help, or not; per segment where errors concentrate.
- **Done when:** on replicas built with a known curve the answer matches; it never reads the holdout.

### 34.14 What to collect next
- **Delivers:** feature-group ablation on folds (which existing groups carry the score), the segments where labels would
  help most, joined to the data-need agent's questions; active learning stays an option, never the default (its gates
  failed).
- **Done when:** on a replica with a planted missing feature group, the advice names that group's kind of data.

## Owner decisions still open

| Decision | Proposal |
|---|---|
| The rule that decides 34.8 | fixed in 34.5 before its run |
| Search budgets on the Air | 20 minutes a stage, 30 trials a family by default |
| AutoGluon | on the M1 Max only |
| Where the Decision page sits | a tab beside Results |

## Verification

- Per package: its tests with `DCLAB_NO_LIVE_MODELS=1 python tests/run_parallel.py <files>`, `make check-quick`, the
  reviewer, the dev record and `make sft-dev`; `make core-wheel-check` when the core changes; the nightly `make check-all`.
- Layer 5: `make product-e2e`, `make browser-test`, the replicas' decision tests (`tests/test_core_decide.py`,
  `tests/test_decisions.py` extended).
- Layer 3: `make bakeoff` results with intervals; `make code-check` for every exported search or ensemble.
- Layer 6: an end-to-end test that scores, logs, posts outcomes, monitors a planted drift and promotes a challenger.
- Layer 2: replica tests with known learning curves and planted missing groups.

## Small items still open from the day

- The pilot after the proposal fix: homeworks 3–5 (tag `proposal-core-keys-v1`, `scripts/learning_loop.sh`).
- `make archive` after the store run (done 16:54); `make check-all` and `make core-wheel-check` tonight.

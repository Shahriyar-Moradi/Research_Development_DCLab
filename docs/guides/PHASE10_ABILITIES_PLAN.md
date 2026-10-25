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
- **Status:** done (2026-10-24). `DecisionSpec` v1 and `choose_operating_point` (dclab-core 0.105.0); the review budget
  and the stated-cost threshold are special cases (same answers; ties taken together as a threshold takes them);
  `assess`'s stated-cost path reads the core; `objective_of` for the main product. Measured: the expected value's 90 %
  interval held the true value 57 of 60 times, a prediction interval for the holdout 54 of 60 (a confidence interval
  holds a holdout's value only about three times in four, so "the holdout falls in the interval" was the wrong check).
  Open: the engine's review path moves to the core in 34.4, with the replicas measured through the engine there; the
  ledger keeps the existing point `threshold.operating_point`; a cost matrix waits for 34.2's multiclass choice.

### 34.2 Targeting and several actions
- **Delivers:** `target_by_uplift(uplift_oof, spec)` (whom to treat under a budget, by expected incremental value) and
  `choose_actions(expected_values_per_action, capacities)` (which of several actions per row, greedy by value with each
  action's capacity; exact for one capacity), multiclass by a cost matrix.
- **Done when:** on `campaign_uplift` (true uplift known per row) the targeting's regret against the oracle beats
  random and treat-all with its interval; a two-action replica's choice equals brute force on small sizes.
- **Status:** done (2026-10-25, dclab-core 0.108.0). `target_by_uplift` reads the arms among the top rows on validation
  rows (weights when offers were not random), says whether it pays (yes, no, cannot tell) and reads the holdout once;
  `choose_actions` is exact (successive shortest paths), not greedy; `choose_by_cost_matrix`. Open: a daily capacity for
  targeting (refused today), a two-action replica (tested on random tables against an exhaustive search).

### 34.3 What-if
- **Delivers:** `what_if(result, changes)` re-scores a chosen decision under changed costs, values, capacity or base rate
  (prior shift, by reweighting the out-of-fold scores), no refit; a sensitivity table: which change of which assumption
  flips the decision, and by how much the value moves.
- **Done when:** doubling a cost moves the point as brute force says; a base-rate change matches a resampled replica;
  every answer says what was not refitted.
- **Status:** done (2026-10-25, dclab-core 0.108.0). `what_if(y_oof, score_oof, spec, changes)` (costs, values,
  capacity, base rate by reweighting) and `sensitivity(...)` (each stated number times 0.25 to 4, where the share
  moves); a changed cost equals a fresh choice exactly; a base rate matches a resampled table within 0.02. Open: what-if
  for targeting and several actions.

### 34.4 Decisions in the product
- **Delivers:** `Solution.decision` v2 (the spec of 34.1, binary and multiclass, the old `review` and `price` kept);
  the engine's final stage writes the decision; the report's decision section; `GET/POST /api/projects/{id}/decision`
  (the result and a what-if); a Decision page in the web app through `DC.client` with `DC.states` (the curve, the chosen
  point, what-if inputs); the intern tool `decision_what_if` (registered once, scope project); Home asks for the action,
  value and cost in words (16.9's open item); a `CORE_INTEGRATION.md` section.
- **Done when:** `make product-e2e` and `make browser-test` cover the page; the intern answers a what-if on a replica;
  the decision is in the solution report with its interval.
- **Status:** done (2026-10-25, dclab-core 0.109.0). `Solution.decision` v2 (`act`: DecisionSpec's numbers; `classes`: a
  cost matrix), chosen by the final stage on out-of-fold scores and read once on the holdout; its out-of-fold rows kept
  beside the project for what-if (stamped with the final run); `GET/POST /api/projects/{id}/decision[/what-if]`; the
  Decision page (curve, what moves it, a what-if form); the intern tool `decision_what_if`; the report's section;
  Home reads amounts in words (`decide.read_spec`) and asks them once when the action came from the first sentence.
  The setup wizard states a decision in its own fields too (2026-10-25). Live: the intern answered a what-if with the
  core's numbers (EUR 0.10). In a real browser (2026-10-25, `make
  browser-test` check `decide`): a decision said in the chat is shown with its sentence and curve, and a what-if typed
  into the form acts on the share the API chooses (55.8% at 50 a call). Open: the Solution page edits the decision as
  JSON only; the engine's review path stays on `review_budget` (same answers, tested);
  `delivery_fraud` not yet measured through `act`.

## Layer 3: Build

### 34.5 The model bake-off
- **Delivers:** `dclab_rnd/bakeoff.py`, `make bakeoff`: each engine on the same splits (`expansion/datasets.py`'s rules,
  holdout read once), the same metric, the same time budget: DCLab's engine as it is, FLAML (`pip install flaml`,
  owner approved), later AutoGluon (M1 Max). Datasets: the 10 UCI sets, Telco, the expansion samples (imbalanced fraud,
  multiclass, text plus tabular); HyperAck only with its leaks removed. Per dataset: holdout score with interval,
  minutes, peak memory; stored as an evidence campaign.
- **Done when:** the table exists for every dataset with intervals; the result decides 34.8 by a rule fixed before the
  run (for example: FLAML ahead by more than its interval on at least a third of the datasets).
- **Status:** harness done (2026-10-25): `dclab_rnd/bakeoff.py`, `make bakeoff` (plan, then `ARGS="run --yes"`), FLAML
  2.7.0 pinned; both engines on the same rows, holdout, folds and metric; the rule deciding 34.8 written into the manifest
  before the first run; review (11 findings, fixed), 14 tests. Run (2026-10-25, `evidence/campaigns/bakeoff_v1`): 13
  datasets judged, FLAML clearly ahead on 1 (bank marketing, +0.018), behind on 3 (letter, wine, and spambase, where
  FLAML failed on our harness's column names: a harness bug, fixed in `_flaml_frame` for a later `bakeoff_v2`), no clear
  difference on 9; 5 were needed: **no go for 34.8** (still no go with spambase counted as ahead). FLAML used its whole
  five-minute budget on each dataset; the engine needed under a minute on 8 of 13 (10 seconds to 9.5 minutes).

### 34.6 Honest selection under many attempts (MLE plan 33.2)
- As written in `MLE_BENCH_LESSONS_PLAN.md` 33.2: search folds and a selection fold, the measured optimism in the record.
  Needed before any wider search, ours or FLAML's.

- **Status:** the library's part done (2026-10-25, dclab-core 0.110.0): `dclab_core.selection` (`selection_split`,
  `choose`, `SelectionRecord` v1). Measured before wiring: only confirming the winner (K=1) keeps its selection score
  honest (optimism about 0.002 at 5, 50 and 200 attempts against 0.019 to 0.051 for plain cross-validation); choosing
  among three on the selection fold adds about 0.025 of its own, so K=1 is the default. Wired with 34.7 (2026-10-25):
  the engine sets the fold aside when a search runs, the models stage's record keeps the `SelectionRecord`, and the
  ledger's `model.family` answer is `rule:search_select` when the search chose.

### 34.7 A wider search and ensembles (33.3–33.5)
- **Delivers:** per-family Optuna search with a budget (minutes, trials), CatBoost reconsidered as a family (its
  exclusion reason at `runner.py:11` re-checked), stacking of the top families on out-of-fold predictions reusing
  `dclab_core.compose`; all chosen through 34.6; the exported code reproduces it (`make code-check`).
- **Done when:** on the bake-off datasets the engine's holdout score rises with the measured optimism inside its
  interval, at a bounded number of minutes on the Air.
- **Status (34.7a, the search):** built 2026-10-25: `dclab_rnd/studio/search.py`, opt-in by `settings.search` (minutes,
  trials, families). Optuna on the stage's own choice (the review's kept change the base of every trial) and the
  screen's next family, on folds of the search rows; the winner by its pooled out-of-fold score, confirmed once on the
  selection fold beside the choice without a search; the final stage fits the searched settings as they are; a fair cut
  or the search skipped with the reason (entities apart on time splits with entities, every class on both sides); the
  exported code reproduces it. Two reviews (15 findings, then 8). **Measured (`search_v2`, both sides fresh on pinned
  code, gates fixed before the run): fail.** On 8 of 9 bake-off datasets the mean holdout difference is -0.0039, the
  search clearly better on none and clearly worse on one (ecommerce reviews, -0.0078); the selection fold reads the
  holdout a little closer than the search score (0.0085 against 0.0096); on the fraud table the search side crashed (a
  fold without a fraud case once the fold was set aside: fixed, the search now skips there). Part of the loss is the
  set-aside's own price: the screen chooses on a fifth fewer rows. So the search stays an opt-in, not recommended.
  `search_v3` (the fix, a crash counted as clearly worse) is ready to confirm; by arithmetic it cannot pass unless the
  eight numbers change.
- **Status (34.7b, stacking and CatBoost): not built** (the owner, 2026-10-25, after `search_v2`). A stack needs the same
  selection fold, so it pays the same price, and the bake-off and the search both found little room above the engine's
  defaults on these tables. Revisit on the M1 Max with bigger tables, where a fold set aside costs less.

### 34.8 An AutoML family behind DCLab's splits
- **Delivers:** FLAML (then AutoGluon on the M1 Max) as one more family in the models stage, fitted inside the same
  folds with a time budget, chosen by the same rule, exported with its requirement; only if 34.5's gate says so.
- **Done when:** it is chosen only where it wins on the selection fold; code export reproduces its holdout number.
- **Status:** not built: 34.5's rule said no go (FLAML clearly ahead on 1 of 13 datasets, 5 needed). AutoGluon on the
  M1 Max would be a new bake-off (`bakeoff_v2`), not this package.

## Layer 6: Learn (contracts here, wiring in the main product)

### 34.9 Prediction and outcome log
- **Delivers:** `dclab_core/serving.py`: `PredictionRecord` (model id and version, a row key or its hash, the score,
  the decision taken, the time; never a cell value) and `OutcomeRecord` (the key, the true outcome, when it became
  known); the exported service logs every `/predict` to a local JSONL and takes `POST /outcome`.
- **Done when:** a batch scored and its outcomes posted join by key; no input value is in either log (tested).
- **Status:** done (2026-10-24, dclab-core 0.106.0, d4138426): `dclab_core.serving` (`PredictionRecord`, `OutcomeRecord`,
  `read_log`, `join`); the exported service logs every `/predict` keyed by a hashed row key and takes `POST /outcome`.
  13 tests: 71 outcomes posted joined on all 71 keys, a bad outcome refused with nothing recorded, no row key, text value
  or service key in either log. Open: see "What remains" (the key without `DCLAB_LOG_KEY`).

### 34.10 Monitoring that runs
- **Delivers:** `monitor_log(plan, predictions, outcomes)`: `monitoring.monitor` on each logged batch plus live
  performance once outcomes arrive (label delay respected, interval by batch size); the service's `/monitor` report;
  alerts as events.
- **Done when:** a replica with planted drift raises the alert at the planted batch and not before; live performance
  matches the holdout's within its interval on undrifted batches.
- **Status:** done (2026-10-25, dclab-core 0.111.0). `dclab_core.live`: `BatchProfile` v1 (a batch's inputs as bin shares,
  never a value), `monitor_log` (`LiveReport` v1, window by window: the score's and the inputs' drift, the flagged share,
  live performance once outcomes are known by `as_of`, against the holdout's interval from `live.reference`); the exported
  service writes `profiles.jsonl` and answers `GET /monitor`. Measured on twenty simulated runs: the planted drift the first
  alert in 20 of 20; undrifted windows within the holdout's interval in 117 of 120; one worse window alerted falsely in 2
  of 20 runs, so a performance alert needs two worse windows running (then none). Alerts as events are the host's.

### 34.11 Retrain and champion/challenger
- **Delivers:** `dclab_core/retrain.py`: a retrain policy (drift alert, performance drop, or schedule); the challenger
  trained on the newer labelled rows through the same engine; compared with the champion on the latest labelled window
  that neither trained on; promoted only when better beyond its interval; every comparison recorded; a studio job
  `POST /api/projects/{id}/retrain`.
- **Done when:** on a drifting replica the challenger is promoted after the drift and refused without it.
- **Status:** done (2026-10-25, dclab-core 0.111.0). `dclab_core.retrain` (`RetrainPolicy` v1, `should_retrain`, `compare`,
  `ChallengerRecord` v1); the studio keeps newer rows beside a project (never read by the engine, no stage cleared, the
  holdout never read again) and compares as a job (`POST /api/projects/{id}/retrain`), the ledger point `model.promote`.
  Measured: promoted after a planted change in 20 of 20; an equally good challenger promoted in 1 of 20 (a one-sided 95%
  bound). The review's 13 findings fixed: repeat rows of the project's table dropped and the window after its latest time;
  entities kept on one side; labels of unconfirmed sources set aside; the time and the codes read as the engine reads them;
  no promotion under 30 rows; the project's own measure; the champion is the last promoted fit; the ledger point written.
  Open: serving the promoted fit is the host's (the studio records the preferred fit, never a production approval).

### 34.12 Lessons from use
- **Delivers:** drift alerts, promotions and refusals become lesson proposals in `lessons.py` (reviewed as today).
- **Done when:** a planted drift yields one cited lesson a reviewer can accept.
- **Status:** done (2026-10-25). `studio/lessons_from_use.py`: each comparison and a live report's first alert proposed once
  as a code-written lesson citing its record, scoped "from use", reviewed as any workspace lesson.

## Layer 2: Data

### 34.13 Would more data help?
- **Delivers:** a learning curve on the training folds (25, 50, 75, 100 % of rows, by group when grouped), its slope
  with an interval, and a plain answer: more rows likely help, or not; per segment where errors concentrate.
- **Done when:** on replicas built with a known curve the answer matches; it never reads the holdout.
- **Status:** done (2026-10-25): `dclab_core.data_value` (DataValue v1, dclab-core 0.107.0), the studio analysis
  (`POST/GET /api/projects/{id}/data-value`, a job), ledger point `data.more_rows`, a `decision.proposed` event. Rising
  replicas 9/10 likely help (1 cannot tell), flat 10/10 unlikely help; pure noise never likely help (60 runs). Kept apart
  from the store's `LearningCurve` (cross-build, no names).

### 34.14 What to collect next
- **Delivers:** feature-group ablation on folds (which existing groups carry the score), the segments where labels would
  help most, joined to the data-need agent's questions; active learning stays an option, never the default (its gates
  failed).
- **Done when:** on a replica with a planted missing feature group, the advice names that group's kind of data.
- **Status:** done (2026-10-25, dclab-core 0.111.0). `dclab_core.collect` (`CollectAdvice` v1): what each group adds beyond the
  others with its interval, the domain's usual kinds missing (prior knowledge by whole word parts, never measured), where
  more rows still helped (34.13) and the problem map's open data needs; the studio weighs the groups as a job (`POST
  /api/projects/{id}/collect-next`). On the churn replica without its support contacts, support is the first kind named
  missing (and the same with the outcome shuffled: it is never from the data); over ten seeds the groups the outcome used
  were named 10 of 10, the two it never used 0 of 20. Open: five domains only (a project in no known domain is weighed
  column by column, nothing named missing); time folds are bootstrapped by row.

## Owner decisions

| Decision | Where it stands |
|---|---|
| The rule that decides 34.8 | fixed in 34.5 before its run; it said no go (2026-10-25) |
| Search budgets on the Air | the search stays an opt-in (`search_v2` failed); asked for with no numbers it runs 10 minutes, 30 trials, 2 families |
| AutoGluon | on the M1 Max only, as a new bake-off (`bakeoff_v2`) |
| Where the Decision page sits | its own page in the project menu, after Evaluation |
| 34.7b stacking and CatBoost | not built (2026-10-25); revisit on the M1 Max |

## Verification

- Per package: its tests with `DCLAB_NO_LIVE_MODELS=1 python tests/run_parallel.py <files>`, `make check-quick`, the
  reviewer, the dev record and `make sft-dev`; `make core-wheel-check` when the core changes; the nightly `make check-all`.
- Layer 5: `make product-e2e`, `make browser-test`, the replicas' decision tests (`tests/test_core_decide.py`,
  `tests/test_decisions.py` extended).
- Layer 3: `make bakeoff` results with intervals; `make code-check` for every exported search or ensemble.
- Layer 6: an end-to-end test that scores, logs, posts outcomes, monitors a planted drift and promotes a challenger.
- Layer 2: replica tests with known learning curves and planted missing groups.

## What remains (recorded 2026-10-25, at the owner's request)

Everything Phase 10 leaves open, in one place: when it runs, who owns it, what it needs. Packages 34.1 to 34.14 are built
or decided; nothing below blocks using them.

**Tonight (this session, scheduled):**
1. `search_v3`: confirms the fix of the fraud crash (R9) and that the eight numbers of `search_v2` reproduce. Run from a
   fresh worktree at the latest commit with `agent_runs/bakeoff_v1` linked in: `python -m dclab_rnd.search_measure run --yes`;
   its results go to `evidence/campaigns/search_v3/` and 34.7a's record. It cannot pass unless the eight numbers change.
2. The nightly gate: `make check-all` and `make core-wheel-check` on the latest commit, alone on the machine. Owed since
   2026-10-24: 34.9 to 34.14 and dclab-core 0.106.0 to 0.112.0 have had only `make rd-gates` and their own tests. A
   failure is fixed with a test, or reported.

**Waits for the M1 Max:**
3. 34.7b, stacking and CatBoost, on bigger tables, chosen on the same selection fold and measured in its own campaign.
4. `bakeoff_v2`: AutoGluon, and FLAML again with the column-name fix (`_flaml_frame`); a new campaign, never an edit of
   `bakeoff_v1`.
5. Appendix A (H3 to H6, the harness that learns). The H3 pilot's homeworks 3 to 5 (`scripts/learning_loop.sh`, tag
   `proposal-core-keys-v1`) need a local model pulled again: `qwen3.5:4b` went to the Trash to free disk on 2026-10-25.

**Open inside built packages (small, each its own change, none started):**
6. 34.1 and 34.4: the engine's review path stays on `review_budget` on purpose (2026-10-25: the choice is tested to be
   the same; moving it would give one decision two record shapes and change review projects' holdout readings).
   `delivery_fraud` through an `act` decision: campaign `fraud_decision_v1`; its first seed showed a binding capacity read
   as a carried threshold (6 rides reviewed where 16 were allowed), fixed in dclab-core 0.113.0 (applied as a quota).
   Measured after the fix (8 seeds, gate fixed before): **fail**. Every seed that acted kept to its capacity, as a quota;
   one seed's model made no review pay; the realised value fell inside its 90% interval on 5 of 7 (one below, one
   above). Whether the interval is too narrow is not established: 5 of 7 or fewer happens about 15 times in 100 even
   when a 90% interval is right; v1's misses in units of the interval have a root mean square of 1.23 (90% interval 0.86
   to 2.2), which fits both a right interval and one about a fifth too narrow; a simulation of fraud-like data found 87
   to 93% coverage but leaves out the engine's model choice. `fraud_decision_v2` decides (30 fresh seeds, coverage,
   centre and width gates, the next step fixed before the run); the interval is not widened before it.
7. 34.2: a daily capacity for targeting (refused today); a two-action replica (tested on random tables only).
8. ~~34.3: what-if for targeting and for several actions~~ done 2026-10-25 (dclab-core 0.113.0: `targeting_what_if`,
   `actions_what_if`; reviewed, 3 findings fixed).
9. ~~34.4: the wizard has no fields for a decision~~ done 2026-10-25: step 3 has a decision block (what is done, what a
   case is worth, what an action, a miss and a wasted action cost, a capacity), filled from the chat's amounts; tested in
   the browser. Still open: the Solution page edits the decision as JSON only.
10. 34.7a: a route that lets a person turn the search on after the features stage must clear the stages from data (no
    such route exists today); the search's attempts and optimism as a measured ledger label, if the decision models
    need them; whether a search is worth its fold on bigger tables (with 3).
11. ~~34.9: without `DCLAB_LOG_KEY` the row-key fallback changes with each export~~ done 2026-10-25: one log key per
    project, the same for every export, kept where the store and the testers' kit never copy it, on files and PostgreSQL.
12. 34.10 and 34.11: alerts as events and serving the promoted fit belong to the host (the main product); the studio
    records the preferred fit, never a production approval.
13. 34.14: five domains only (a project in no known domain is weighed column by column, nothing named missing); time
    folds bootstrapped by row.
14. The main product: layer 6 (34.9 to 34.12) and 34.4's decision are wired there by that repository's own change, from
    the `CORE_INTEGRATION.md` sections.

**Housekeeping:**
15. `make archive` after the store run of 2026-10-24 (the newest snapshot is from that morning): taken by the session
    working on the store and the students, today before 22:00, after a store build.
16. Two scratch worktrees (`search_v2`'s run and the builds') are removed with `git worktree remove` once `search_v3` has run.
17. Playwright's Chromium (581 MB, `~/Library/Caches/ms-playwright`) stays for `make browser-test`.
18. The Workflow page may stop refreshing during a run when the machine is slow: the browser flow failed once at a load
    of 18 on 8 cores (the page showed 1 of 5 stages while the run had finished) and passed on the rerun. Not investigated:
    a page that stops polling between two stages would explain it.
19. Capture (the owner, 2026-10-25: "make sure all data capturing and recording and store are running for our 6 models"):
    the store collects `agent_runs/`; this session's campaigns now keep their engine runs there (`search_v2` copied in,
    240 MB; `fraud_decision_v1` and `search_v3` write there). Not captured: `make product-e2e` and `make browser-test` run
    in temporary workspaces, so their sessions, ledgers and runs are deleted with them (the test results are kept).
20. 34.1: a capacity per day is applied as one pool over the holdout (said in `applied_as`); filling it day by day needs
    each holdout row's day passed to the core.

# Adding a kind of problem or a modifier

dclabai never writes a new pipeline for a new problem. **A problem is one kind plus zero or more modifiers.** You add a
part, prove it on invented data with its traps planted, and every run that uses it says which part made each choice and
why. This page shows how: **A** a new kind, **B** a new modifier.

## 1. Who decides what

```
 words, files ──► PROBLEM CARD ──► DATA CARD + fit ──► SOLUTION BRIEF ──► WORKFLOW RUN ──► report
                  what is predicted,  the tables, the     the recipe resolved   the stages, a check
                  when, for whom,     modifiers detected  (kind + modifiers),   before each next one,
                  what a mistake costs                    a person approves     the holdout read once
```

- **Rules decide.** Splits, metrics, thresholds, selection and every check are deterministic code.
- **A model reads and writes words.** It may *propose* a part ("this looks like a matching problem"), never make a choice
  inside a recipe. With no model, the same parts give the same result.
- **A person approves** the Solution Brief.

## 2. The pieces

| Piece | What it is | Where |
|---|---|---|
| Kind | The shape of a problem: binary, regression, rare event, ranking, matching... | `dclabai/kinds/<key>.py`, interface `kinds/base.py` |
| Machine | A kind's own stages when the shared engine is not enough (survival, ranking, forecasting) | `dclabai/kinds/machine.py`; example `SurvivalMachine` in `kinds/time_to_event.py` |
| Modifier | A condition in the data that changes how any kind is handled (rare positives, groups, time order, noisy labels, review capacity) | `dclab_core/modifiers.py` |
| Recipe part | The nine-field contract every kind and modifier fills | `dclab_core/recipes.py` |
| Replica | Invented data shaped like a real case, with its traps planted | `dclab_core/replicas/<name>.py` |
| Acceptance | Runs a kind's replica through the engine and checks its decisions | `dclabai/kinds/acceptance.py` → `kinds/capability.json` |

## 3. How a recipe is resolved

Every part fills the same nine fields; a modifier fills only what it changes: **detection**, **required inputs**,
**split**, **metrics** (one primary and guards), **models** (the baseline and the families), **leakage checks**,
**post-processing** (calibration, the threshold rule), **artifact needs** and **report items**.

`dclab_core.recipes.resolve(kind, modifiers, stated)` merges them with fixed precedence:

1. What the person stated wins.
2. Splits combine: time order plus groups gives `time_group` (forward in time, each group on one side). A kind's time order
   is never dropped by a modifier.
3. Single values (the primary metric, calibration, the threshold rule): the most specific part wins, in the order of
   `SPECIFIC` in `dclab_core/recipes.py`.
4. Lists add up (leakage checks, report items).
5. A conflict the rules cannot settle is left open in `ask`, for a person. Nothing guesses.

Every choice carries the part that made it (`name@version`) and the reason. See the registry with:

```bash
python -m dclabai.recipes
```

## Way A. Add a new kind

Use it when the *shape* of the problem is new: what a row is, what the target is, how it must be split and judged.

1. **What it runs on.** Every kind runs as `binary`, `multiclass` or `regression` (`BASE_TASKS` in `kinds/base.py`). If
   the shared engine fits, you write only the kind; if it needs its own stages, it also brings a machine (step 3).
2. **The module** `dclabai/kinds/<key>.py`: subclass `ProblemKind`, set what differs from the defaults, end with
   `KIND = ...`. The registry picks it up on import.

   ```python
   """Promotion demand: units sold per store and day, with promotions planned ahead."""
   from __future__ import annotations

   import pandas as pd

   from .base import ProblemKind, _get


   class PromotionDemand(ProblemKind):
       key = "promotion_demand"
       title = "Promotion demand"
       description = "Units sold per store and day when promotions are planned in advance."
       base_task = "regression"
       task_type = "timeseries_regression"
       specificity = 10                               # asked before the general kinds
       words = r"\b(promotion|promo|units sold)\b"
       questions = ("Which number is predicted, per store and day?", "Are promotions known before the day they run?")
       metrics = (("mae", "Mean absolute error"),)    # the first is the default
       baselines = ("training_mean", "training_median")
       checks = ("PIT-001", "PIT-002")                # pitfall ids from dclabai/pitfalls.py
       stratify = False

       def target(self, frame, solution):
           return pd.to_numeric(frame[_get(solution, "target")], errors="coerce").astype(float), None

       def summarise_target(self, y, class_labels, solution):
           return {"mean_train": float(y.mean())}, f"a target mean of {y.mean():.4g}"

       def target_claim(self, summary, metric_label, spec):
           return ("fact", f"The training mean is {summary['mean_train']:.4g}.", ["evidence.target_summary"])

       def case(self, seed=0): ...                     # step 4
       def accept(self, run): ...                      # step 4


   KIND = PromotionDemand()
   ```

   Override `detect`, `applies`, `split`, `model_families`, `check_solution` and the others in `kinds/base.py` only where
   your kind differs.
3. **Its own machine, only if needed.** Return it from `own_machine()`: subclass `Machine` (`kinds/machine.py`), set
   `task_type`, `primary_metric` and `higher_is_better`, implement `training_rows()` and one method per stage (`data`,
   `leakage`, `features`, `models`, `final`). The holdout is read in `final` only.
4. **A replica and its acceptance.**
   - `dclab_core/replicas/<name>.py` with `generate(seed=0, n=...)`: invented tables and a `facts` dict naming the planted
     traps (a column recorded after the outcome, an identifier that repeats, drift in the last period). Never a real or
     confidential file.
   - `case(seed)` returns an acceptance `Case` (`kinds/acceptance.py`): the table, the target, the task, the prediction
     moment, what the person states.
   - `accept(run)` returns the decisions a careful data scientist would reach, each a `check("what", ok, detail)`.
5. **Run its acceptance:**

   ```bash
   python -m dclabai.kinds promotion_demand
   ```

   It writes `dclabai/kinds/capability.json` (never edit it by hand). The kind is **supported** on `/api/kinds` and the
   Domain packs page only when every decision passes.
6. **Tests** beside `tests/test_kinds.py`, run with `python scripts/run_tests.py test_kinds`. Code that does not depend on
   the studio (a contract, a rule, a computation) belongs in `dclab_core`, which imports only numpy, pandas, scikit-learn
   and pydantic.

## Way B. Add a new modifier

Use it when the shape is known but a *condition in the data* should change how it is handled, for any kind. Example:
strong seasonality.

1. **The threshold** as a named constant at the top of `dclab_core/modifiers.py` (beside `RARE_BELOW`), for example
   `SEASON_STRENGTH = 0.3`.
2. **The part** in `PARTS`, filling only what it changes:
   ```python
   "seasonality": RecipePart(
       name="seasonality", part_type="modifier", version=VERSION, level="experimental",
       detection=[Detection(signal="seasonal_strength", rule=f">= {SEASON_STRENGTH:g} of the target's variance by calendar period")],
       required_inputs=["time_column"], split="rolling_origin",
       leakage_checks=["calendar features from the date only, never from the target of the same period"],
       report_items=["errors by season"]),
   ```
3. **The fact it reads**: a field of `Facts`, computed in `facts_of(...)` from the training rows only: a share, a ratio,
   a count, **never a row or a cell value** (a model may read these facts).
4. **Its detection**: a helper returning a `Detected` (`name`, `applies`, `reason`, `strength`, `value`, `asks_user`),
   added in `detect(facts)`. The reason says the number that decided.
5. **Its precedence**, if it sets a single value: its place in `SPECIFIC` in `dclab_core/recipes.py`.
6. **Keep it in the run's recipe**: a rule in `KEEP` in `dclabai/recipes.py`, so a detected modifier is kept only when the
   run applies it; `tests/test_recipes.py` fails until every modifier has one.
7. **Tests** in `tests/test_modifiers.py` (detected on its trap replica, not where it should not be) and
   `tests/test_recipes.py` (its precedence and its reason).

## Levels

A part is **experimental** until it is certified on trap replicas and real datasets against five criteria (detection,
traps, beating the baseline, the saved model predicting the same, and the same result with the model on and off). The
certification tool is not part of the open edition yet: open parts show as experimental, and "supported" means their
replica passes its acceptance.

## Done when

| Way | Done when |
|---|---|
| A. Kind | its module, replica and acceptance; `python -m dclabai.kinds <key>` shows it supported; its tests pass |
| B. Modifier | its part, fact, detection, precedence and `KEEP` rule; detected on its trap replica and not elsewhere; its tests pass |

And for both: **leakage first** (a replica plants the traps a real case would have), and **run only the tests that
prove the change** before opening a pull request (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

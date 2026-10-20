# Adding capability to DCLab: kinds, modifiers and knowledge

This guide explains how DCLab decides how to solve a problem (the framework built in Phase 8) and how to teach it to
solve more. It is for the owner and anyone who extends DCLab. Read sections 1 to 4 once; then follow the way you need:
**A** a new kind of problem, **B** a new modifier, or **C** new business knowledge.

The rule behind all of it: **a problem is one kind plus zero or more modifiers. You never write a new pipeline.** You add a
part, prove it on invented data with its traps and on real data, and every run that uses it says which part made each
choice and why.

---

## 1. The idea in one picture

```
 words, files ──► 1 PROBLEM CARD ──G1──► 2 DATA CARD + fit ──G2──► 3 SOLUTION BRIEF ──G3──► 4 WORKFLOW RUN ──G4──► report
                  what is predicted,     the tables, what the       the recipe resolved      twelve fixed steps,
                  when, for whom,        rows show, the modifiers   (kind + modifiers),       a check before each
                  what a mistake costs   detected, the fit verdict  the success bar frozen,   next one, the holdout
                                                                    a person approves         read once, monitoring
                        ▲                                                                                    │
                        └──────────── a failed gate sends the work back to a named phase, with the reason ───┘
```

Who does what never changes:

- **Rules decide.** Splits, metrics, thresholds, selection and every check are deterministic code.
- **A model reads and writes words.** It may *propose* a part (for example "this looks like a matching problem"), never
  make a choice inside a recipe.
- **A person approves** the Solution Brief and owns the business risk.

With the model switched off, phases 2 and 4 give the same result. That is one of the certification criteria (section 4).

## 2. The pieces and where they live

| Piece | What it is | Where |
|---|---|---|
| **Kind** | The basic shape of a problem: binary, regression, rare event, ranking, matching… (17 today) | `dclab_rnd/kinds/<key>.py`, one module each; interface `kinds/base.py`; registry `kinds/__init__.py` |
| **Machine** | A kind's own engine when the shared one is not enough (survival, ranking, forecasting) | `dclab_rnd/kinds/machine.py`; example `SurvivalMachine` in `kinds/time_to_event.py` |
| **Modifier** | A condition in the data that changes how a kind is handled (5 today) | `dclab_core/modifiers.py`: `PARTS`, `Facts`, `detect()`, `applied()` |
| **Recipe part** | The nine-field contract every kind and modifier fills | `dclab_core/recipes.py`: `RecipePart`, `ResolvedRecipe`, `resolve()` |
| **Kind as a part** | Reads a kind's attributes into a `RecipePart`; resolves a run's recipe | `dclab_rnd/recipes.py`: `part_of()`, `resolve_for()` |
| **Replica** | Invented data shaped like a real case, with that kind's traps planted | `dclab_core/replicas/<name>.py` |
| **Acceptance** | Runs a kind's replica through the engine and checks its decisions | `dclab_rnd/kinds/acceptance.py` → `kinds/capability.json` |
| **Certification** | Judges a part on replicas and real datasets against five criteria | `dclab_rnd/certify.py`, `dclab_core.recipes.certify` → `evidence/campaigns/certification_v1/results/` |
| **Knowledge** | Business pattern cards the agents read | `dclab_core/business_examples.py`; sources `dclab_rnd/knowledge/*.json`; built by `dclab_rnd/business_examples.py`; read with `find_examples` (`dclab_rnd/tools.py`) |

## 3. How a recipe is resolved

Every part fills the same nine fields. A kind fills all of them; a modifier fills only what it changes.

| # | Field | Example |
|---|---|---|
| 1 | detection | "the minority class is under 5% of the training rows" |
| 2 | required inputs | `group_column`, `capacity_k` |
| 3 | split | `stratified`, `group`, `time`, `time_group`, `rolling_origin`, `spatial` |
| 4 | metrics | one primary (`average_precision`) and guards that must not get worse (`roc_auc`) |
| 5 | models | the mandatory baseline and the families |
| 6 | leakage checks | "no group on both sides of the split" |
| 7 | post-processing | calibration, the threshold rule |
| 8 | artifact needs | what the saved model must carry ("the cutoff rule for features") |
| 9 | report items | what the report must show ("the holdout is the latest period") |

`dclab_core.recipes.resolve(kind, modifiers, stated)` merges the parts with fixed precedence:

1. **What the person stated wins** (recorded as `person`).
2. **Splits combine.** Time order plus groups gives `time_group`: forward in time, each group on one side. A kind's time
   order is never dropped by a modifier's split.
3. **Single values** (the primary metric, calibration, the threshold rule): the most specific part wins, in the order of
   `SPECIFIC` in `dclab_core/recipes.py` (review capacity, then rare positives, then the kind).
4. **Lists add up**, in order (leakage checks, report items…).
5. **A conflict the rules cannot settle** (two splits that do not combine, two modifiers of equal rank that disagree) is
   left open in `ask`. A person settles it; nothing guesses.

Every choice in the `ResolvedRecipe` carries the part that made it (`name@version`) and the reason. A worked example, a
binary churn problem with 3% positives and a `signup_date` column:

| Field | Value | Part | Reason |
|---|---|---|---|
| split | `time` | `time_order@1` | the latest period is the holdout |
| primary metric | `average_precision` | `rare_positives@1` | the most specific modifier, over the kind's `roc_auc` |
| leakage checks | class shares on training rows; features only from data before each row's cutoff | both modifiers | lists add up |

A kind's version is a fingerprint of its code (`dclab_rnd/recipes.py::_fingerprint`), so any change to a kind changes its
version and the runs that used the old one stay traceable. To see the registry:

```bash
.venv/bin/python -m dclab_rnd.recipes
```

## 4. Levels and certification

A part has one of three levels:

| Level | Earned by |
|---|---|
| experimental | the default; the product may run it but says so |
| certified | `dclab_core.recipes.certify`: at least one trap replica **and** at least two real datasets where the part applies, each passing all five criteria |
| product default | more than runs (no wrong detection on any R&D dataset, no regression in twin runs); the tool never grants it |

The five criteria (`CRITERIA` in `dclab_core/recipes.py`):

1. **detection**: the part is detected where it should apply, and not where it should not;
2. **traps**: every trap planted in the run is caught;
3. **baseline**: the holdout result beats the baseline;
4. **artifact**: the saved model reloads and predicts the same;
5. **llm_off**: the same result with the model on and off.

A criterion that was not measured counts as not passed. A run where the part rightly does not apply checks its absence,
but never counts toward the level.

Where things stand (2026-10-20): 17 kinds, 15 supported in the capability matrix (`metric_change` and `embedded` partial).
`binary` and `matching` are certified (`CRT-003_binary.json`, `CRT-002_matching.json`). The five modifiers are experimental:
their artifact and LLM-off criteria are not measured yet (`CRT-001_modifiers.json`). The registry printed by
`python -m dclab_rnd.recipes` does not read these records yet and still shows every kind as experimental: the certification
record under `evidence/campaigns/certification_v1/results/` is the one to trust.

"Supported" and "certified" are different. **Supported** (`make kinds-check`) means the kind's replica passes its
acceptance. **Certified** (`python -m dclab_rnd.certify`) also needs real datasets and all five criteria.

---

## Way A. Add a new kind

Use this when the *shape* of the problem is new: what a row is, what the target is, how it must be split and judged.
Examples: churn with competing risks, demand with promotions, a next-best-offer ranking.

**A1. Decide what it runs on.** Every kind runs as one of the base tasks `binary`, `multiclass` or `regression`
(`BASE_TASKS` in `kinds/base.py`). If the engine's shared machines fit (folds, families, metrics), you only write the kind.
If the kind needs its own stages (survival, horizons, pairs), it also brings a machine (A3).

**A2. Write the module** `dclab_rnd/kinds/<key>.py`. Subclass `ProblemKind`, set what differs from the defaults, and end
with `KIND = ...`. The registry picks it up on import; nothing else needs registering. A skeleton, modelled on
`kinds/regression.py`:

```python
"""Promotion demand: units sold per store and day, with promotions planned ahead."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import ProblemKind, _get


class PromotionDemand(ProblemKind):
    key = "promotion_demand"
    title = "Promotion demand"
    description = "Units sold per store and day when promotions are planned in advance."
    base_task = "regression"
    task_type = "timeseries_regression"            # the runner's machine
    specificity = 10                               # asked before the general kinds
    words = r"\b(promotion|promo|campaign uplift on sales|units sold)\b"
    questions = ("Which number is predicted, per store and day?", "Are promotions known before the day they run?")
    metrics = (("mae", "Mean absolute error"),)    # the first is the default
    baselines = ("training_mean", "training_median")
    checks = ("PIT-001", "PIT-002")                # pitfall ids from dclab_rnd/pitfalls.py
    reliability = ("errors on promotion days against normal days", "drift by time")
    stratify = False

    def target(self, frame: pd.DataFrame, solution: Any) -> tuple[pd.Series, list[str] | None]:
        return pd.to_numeric(frame[_get(solution, "target")], errors="coerce").astype(float), None

    def summarise_target(self, y, class_labels, solution) -> tuple[dict[str, Any], str]:
        return {"mean_train": float(y.mean())}, f"a target mean of {y.mean():.4g}"

    def target_claim(self, summary, metric_label, spec) -> tuple[str, str, list[str]]:
        return ("fact", f"The training mean is {summary['mean_train']:.4g}.", ["evidence.target_summary"])

    def case(self, seed: int = 0) -> Any:           # A4
        ...

    def accept(self, run: Any) -> list[dict[str, Any]]:   # A4
        ...


KIND = PromotionDemand()
```

The interface, in short (`kinds/base.py`):

- **Identity:** `key`, `title`, `description`.
- **What it runs as:** `base_task`, `tasks`, `task_type`.
- **How Home recognises and talks about it:** `specificity`, `words`, `packs`, `questions`, `home_fields`, `moments`.
- **How it is judged:** `metrics`, `baselines`, `checks`, `reliability`, `evidence_track`, `aliases`.
- **Its own solution fields:** `Fields`, a pydantic model for `kind_fields`.
- **Plain flags:** `runs_in_engine`, `needs_target`, `open_decisions`.
- **Required methods:** `target`, `summarise_target`, `target_claim`.
- **Methods to override when they differ:** `detect`, `applies`, `split` (default `default_split`: spatial, time and
  group, time, group, else stratified), `model_families`, `extra_recipes`, `text_columns`, `check_solution`,
  `columns_of`.

**A3. Its own machine (only if needed).** Return it from `own_machine()`. Subclass `Machine` (`kinds/machine.py`):

- set `task_type`, `primary_metric` and `higher_is_better`;
- implement `training_rows()` (the rows every stage before the final may read), `holdout_description()` and
  `cv_description()`;
- write one method per stage, `data`, `leakage`, `features`, `models` and `final`, each `(eid, previous)` →
  `(evidence, claims, summary, decision)`.

The holdout is read in `final` only, and `shape()` marks it consumed and never production-approved. Copy the pattern of
`SurvivalMachine` in `kinds/time_to_event.py`, and add `accept_engine(run)` for the checks only the engine run shows.

**A4. A replica and its acceptance.**

- **The replica:** add `dclab_core/replicas/<name>.py` with a `generate(seed=0, n=...)` that returns invented tables and a
  `facts` dict naming the planted traps. Examples: a column recorded after the outcome, an identifier that repeats, drift
  in the last period. Never a real or confidential file.
- **`case(seed)`:** returns an acceptance `Case` (`kinds/acceptance.py`): the table, the target, the task, the prediction
  moment, and what the person states beyond accepting the proposal.
- **`accept(run)`:** returns the decisions a careful data scientist would reach, each a `check("what", ok, detail)` or a
  `kept_out(run, column, why)`. Examples from `regression.py`:
  - the post-delivery column is blocked;
  - the metric is MAE;
  - the holdout beats 0.9 × the median baseline;
  - the cap at 60 minutes is found.

**A5. Run its acceptance** (only this kind; the whole matrix runs every kind's stages):

```bash
make kinds-check ARGS=promotion_demand
```

It writes `dclab_rnd/kinds/capability.json`; never edit that file by hand. The kind shows as **supported** on
`/api/kinds` and the Domain packs page only when every decision passes.

**A6. Certify it** on its replica seeds plus two real datasets. Name the real ones in the kind's `real_cases()`, or in
`REAL_FOR` in `dclab_rnd/certify.py` (binary uses Telco and HyperAck). Then:

```bash
.venv/bin/python -m dclab_rnd.certify --kind promotion_demand            # prints the level and what is missing
.venv/bin/python -m dclab_rnd.certify --kind promotion_demand --write    # stores CRT-NNN_promotion_demand.json
```

**A7. Tests and the core rule.**

- **Tests:** add them beside `tests/test_kinds.py` and `tests/test_kind_acceptance.py`.
- **Code that the main product could use** (a contract, a rule, a computation) goes in `dclab_core`, imports only numpy,
  pandas, scikit-learn and pydantic, and gets a section in `docs/guides/CORE_INTEGRATION.md`; `make core-wheel-check`
  proves it installs alone. The wiring stays in `dclab_rnd`.

## Way B. Add a new modifier

Use this when the problem's shape is known but a *condition in the data* should change how it is handled, for any kind
it appears in. Examples: strong seasonality, costly false alarms, labels that arrive late. A modifier never invents a
pipeline; it changes some of the nine fields.

The steps, using a "seasonality" modifier as the example:

1. **The threshold as a named constant** at the top of `dclab_core/modifiers.py` (beside `RARE_BELOW`, `ROWS_PER_KEY`…),
   for example `SEASON_STRENGTH = 0.3`. One cut-off for the whole studio, never repeated elsewhere.
2. **The part** in `PARTS`, filling only the fields it changes:
   ```python
   "seasonality": RecipePart(
       name="seasonality", part_type="modifier", version=VERSION, level="experimental",
       detection=[Detection(signal="seasonal_strength", rule=f">= {SEASON_STRENGTH:g} of the target's variance by calendar period")],
       required_inputs=["time_column"], split="rolling_origin",
       leakage_checks=["calendar features from the date only, never from the target of the same period"],
       report_items=["errors by season"]),
   ```
3. **The fact it reads.** Add a field to `Facts` and compute it in `facts_of(...)`. Facts are aggregates of the training
   rows only: a share, a ratio, a count. **Never a row or a cell value**; a model may read these facts.
4. **Its detection.** A helper beside `_noisy` and `_capacity` that returns a `Detected` (`name`, `applies`, `reason`,
   `strength`, `value`, `asks_user`), added in `detect(facts)`. Write the reason so a person understands it, with the
   number that decided. Set `asks_user=True` when only a person can know (as review capacity does).
5. **Its precedence**, if it sets a single value (the primary metric, calibration, a threshold rule): place it in
   `SPECIFIC` in `dclab_core/recipes.py`. If it only adds to lists or sets a split, the existing rules already merge it.
6. **Keep it in the run's recipe.** `resolve_for()` in `dclab_rnd/recipes.py` keeps a detected modifier only when the run
   really applies it. Its `kept` filter **names every modifier by hand today**: add a line for yours, or it is detected
   and then silently dropped.
7. **Certify it.** Add it to `PARTS` in `dclab_rnd/certify.py`, give the replica cases an `expect` for it (where it must
   apply, and where it must not), and run `python -m dclab_rnd.certify [--write]`.
8. **Tests** in `tests/test_modifiers.py` (detected on the trap replica, not detected where it should not be) and
   `tests/test_recipes.py` (its precedence and its choice's reason).

## Way C. Add business knowledge (a pattern card)

Use this when no code is needed: a business problem you meet often, written so the agents (and a person) can recognise
it next time and know how to approach it.

- **Edit a source, never the output.** `evidence/business_examples/examples.jsonl` is generated. Add your card to
  `dclab_rnd/knowledge/written_examples.json` (a written, public-knowledge example). Patterns from an interview or a
  document go in `dclab_rnd/knowledge/retail_interview_patterns.json`, or in a new source read by
  `dclab_rnd/business_examples.py`.
- **The fields** (`BusinessExample` v2 in `dclab_core/business_examples.py`):

| Field | What to write |
|---|---|
| `id`, `title`, `industry`, `problem` | `written:<slug>`, a short title, the industry, the problem in one or two sentences |
| `understanding` | `unit`, `target`, `moment`, `labels`, `action`, `family`: what a row is, what is predicted and when, where labels come from, what is done with the score |
| `data`, `pitfalls`, `solution`, `workflow`, `tags` | short lists: the tables, the traps, the approach, the steps |
| `pattern` | `general_problem`, `why_hard`, `how_to_solve`, `how_to_measure`, `mistakes`, `elsewhere` (the same problem in other industries) |
| `practice` | questions, each with what a strong answer `names` (the metric, the split, the first baseline) |
| `recipe` | `kind`, `modifiers`, `capability` (`built`, `planned` or `knowledge`), `package` when planned |
| `visibility` | `model` (agents may read it) or `rules_only` (search and the owner only); an owner's own solution must be `rules_only` |
| `ref` | where the knowledge comes from ("public knowledge: …"); never a person's name, a customer, or a row of anyone's data |

- **Build it and check that the search finds it:**

```bash
make business-examples
make business-examples ARGS="--find 'claims that look fraudulent'"
make business-examples ARGS=--practice
```

`--practice` checks that each practice question finds its own pattern. A card whose questions do not find it needs
clearer words.

- **Who reads it.** The Home agent, the intern and the teams use `find_examples`; the critic and the router see the
  closest examples' pitfalls. A role reads them only behind a measured switch (served only where an arm measured better).
- **Test:** `tests/test_business_examples.py`.

---

## Done when

| Way | Done when |
|---|---|
| A. Kind | its module, replica and acceptance; `make kinds-check ARGS=<key>` shows it supported; its tests pass; a certification record with its level, or the list of what is missing |
| B. Modifier | its part, fact, detection, precedence and `resolve_for` line; detected on its trap replica and not where it should not be; its tests pass; a certification record |
| C. Knowledge | its source entry; `make business-examples` builds it; `--practice` finds it; its visibility is right |

For every way:

- **Leakage first.** A replica plants the traps a real case would have, and a new part adds the leakage checks it needs.
- **Capture.** Every choice a part makes reaches the decision ledger through the run's recipe; nothing extra to wire.
- **The build record.** Write the development record (`research/llm-fine-tuning/experiments/dev_trajectories/records/<name>.json`),
  then run `make sft-dev`. Use the `dclab-package` skill to build it as a package and the `dclab-reviewer` agent to review
  the diff.
- **Tests.** Run only the tests that prove the change. The full suite and the gates run when the owner says.

## Commands at a glance

```bash
.venv/bin/python -m dclab_rnd.recipes
```
```bash
make kinds-check ARGS=<kind_key>
```
```bash
.venv/bin/python -m dclab_rnd.certify --kind <kind_key> --write
```
```bash
.venv/bin/python -m dclab_rnd.certify --write
```
```bash
make business-examples ARGS=--practice
```

The first prints the registry (each part's version, level, split, metric and certification). `--kind` certifies one
kind; without it, the command certifies the modifiers.

## Further reading

- `docs/guides/PRODUCT_BUILD_PLAYBOOK.md`, Part 5 "Many kinds of problems": why kinds exist and the table of their parts.
- `docs/guides/PHASE8_KNOWLEDGE_CARDS_RECIPES_PLAN.md`, Parts C to E: the cards, gates, recipes and the packages that
  built them.
- `docs/guides/CORE_INTEGRATION.md`: how the main product adopts what lives in `dclab_core`.

---

## خلاصهٔ فارسی

**ایده:** هر مسئله یعنی «یک نوع + صفر یا چند اصلاح‌گر» و این ترکیب «دستور» نام دارد. هیچ‌وقت pipeline جدید نمی‌نویسیم.

- **مسیر کلی:** کارت مسئله، کارت داده (اصلاح‌گرها اینجا تشخیص داده می‌شوند)، خلاصهٔ راه‌حل (دستور حل می‌شود، معیار موفقیت ثابت می‌شود، شخص تأیید می‌کند)، و اجرای workflow با ۱۲ قدم ثابت. هر دروازه‌ای که رد شود کار را با دلیل به فاز مشخصی برمی‌گرداند.
- **چه کسی چه می‌کند:** قواعد تصمیم می‌گیرند. مدل فقط می‌تواند یک قطعه پیشنهاد دهد، نه یک انتخاب. هر انتخاب می‌گوید کدام قطعه (با نسخه‌اش) و چرا.

**حل دستور:** هر قطعه همان قرارداد نُه‌فیلدی را پر می‌کند. اولویت‌ها:
- آنچه شخص بگوید برنده است؛
- تقسیم‌ها ترکیب می‌شوند (زمان و گروه با هم)؛
- برای معیار، آستانه و کالیبراسیون، خاص‌ترین قطعه برنده است: اول ظرفیت بررسی، بعد مثبت‌های نادر، بعد خود نوع؛
- فهرست‌ها جمع می‌شوند؛
- تعارضی که قواعد حل نکنند از شخص پرسیده می‌شود.

**سطح‌ها:** آزمایشی، گواهی‌شده، پیش‌فرض محصول. گواهی یعنی یک ماکت تله‌دار و دست‌کم دو دادهٔ واقعی، هر کدام با پاس شدن هر پنج معیار: تشخیص، تله‌ها، مبنا، مدل ذخیره‌شده، و نتیجهٔ یکسان با مدل خاموش.

**راه الف، نوع جدید:**
۱. ماژول `dclab_rnd/kinds/<key>.py` که از `ProblemKind` ارث می‌برد و با `KIND = ...` تمام می‌شود.
۲. اگر لازم است، یک موتور خودی (`Machine`).
۳. یک ماکت در `dclab_core/replicas/`.
۴. توابع `case` و `accept`.
۵. `make kinds-check ARGS=<key>`.
۶. `python -m dclab_rnd.certify --kind <key>`.

**راه ب، اصلاح‌گر جدید:**
۱. ثابت آستانه.
۲. یک `RecipePart` در `PARTS`.
۳. یک فیلد در `Facts` (فقط اعداد تجمعی، هرگز یک ردیف).
۴. تابع تشخیص در `detect`.
۵. جایش در `SPECIFIC`.
۶. **یک خط در فیلتر `kept` در `resolve_for`.** فراموشش نکنید، وگرنه اصلاح‌گر تشخیص داده می‌شود و بی‌صدا کنار می‌رود.
۷. افزودن به `certify.PARTS`.
۸. تست‌ها.

**راه ج، دانش کسب‌وکار:** یک کارت الگو در `dclab_rnd/knowledge/written_examples.json` اضافه کنید (هرگز خودِ `examples.jsonl` را ویرایش نکنید). کارت شامل الگوی کلی، سؤال‌های تمرینی، پیوند به دستور و سطح دیده‌شدن است. بعد `make business-examples ARGS=--practice` را اجرا کنید.

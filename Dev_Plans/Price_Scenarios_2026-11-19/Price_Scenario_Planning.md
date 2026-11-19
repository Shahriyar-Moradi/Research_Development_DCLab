# Price scenarios, customer response and feature engineering

**19 November 2026 · Implementation plan for a coding handoff**

[Standalone coding prompt](PROMPT.md) · [Implementation, source and run instructions](IMPLEMENTATION.md) · [Full simulation results](simulation/REPORT.md)

Build a decision brief that answers two questions. First, what would our model predict if we offered a different price to the same eligible people? Second, what evidence tells us how much changing the price would actually change their behavior? Let a user explore the first question while the second is still open.

## 1. The first screen

Compare the current price, a 5% increase and a 10% increase for the same cohort and horizon. Start with one offer per person and at most one unit purchased. Show price, number of eligible offers, expected purchases, expected revenue and the number of sales needed to preserve a stated baseline revenue. For renewal projects show expected churn and the change in churn rate in percentage points. Define churn and the observation horizon for that project. A missed purchase on one offer does not establish churn.

Label the source of each result: **user assumption**, **model prediction**, **randomized estimate**, **adjusted observational estimate** or **insufficient evidence**. Provide an explanation for unavailable numbers. Let the reader see the most important assumption and what would improve the evidence. Keep existing decision controls in the Decision Brief. The first release does not change live prices.

### A worked example that doubles as an acceptance test

Yesterday, 100 people saw a price of 100 and 20 purchased. The recorded revenue is 2,000. Keep that observation. A scenario creates a separate set of predictions for a declared eligible cohort. It never copies yesterday's 20 purchases into new training labels.

| Offered price | Sales required to reach 2,000 | What can be calculated immediately |
|---|---:|---|
| 100 | 20 | Yesterday's 20 is an observation. A forecast for tomorrow needs its own probabilities. |
| 105 | 20 | At 19 sales, revenue is 1,995. |
| 110 | 19 | At 19 sales, revenue is 2,090. |

For an assumed 18% purchase probability at 110 and 100 independent offers with the same probability, expected sales are 18, expected revenue is 1,980, and the probability of finishing below 2,000 is approximately 0.56255. The risk is the probability of **18 or fewer** purchases. This example compares tomorrow's outcome with a fixed revenue target. A comparison with the unknown revenue tomorrow under another policy would require additional assumptions about the two potential outcomes.

## 2. Inputs and outputs

- **Offer record:** offer ID, eligible person or group, decision time, offered price, currency, reference price, outcome definition and horizon. Include non-purchasers. Use only completed observation horizons to train or evaluate labels. Count exclusions. A purchases-only table cannot supply a conversion denominator.
- **Event record:** stable event ID, person, event type, occurrence time, time it became available, and permitted attributes. An offer being ignored is established only after its response window has closed. Deduplicate repeated deliveries.
- **Scenario request:** project, fitted model or explicit assumptions, fixed cohort and horizon, price alternatives, baseline revenue or churn target, and declared uncertainty assumptions. Use one currency and selling unit. Reject invalid prices and mixed units. Expose the difference between a fixed audience and a business where changing price may also change traffic.
- **Result:** status and evidence source per estimate, eligible and supported counts, aggregate predictions, measured effect when supported, uncertainty and its meaning, warnings, model/data/feature versions and assumption references. Keep observed inputs immutable. Equal requests against the same versions and seed should reproduce the result.

## 3. Build in five small packages

### P1. Start with features and the arithmetic

Write the prediction contract before looking at scores. For each feature record its source, when it is available, its dependencies and its rule under a price change. Reuse [event-time features](../../dclab_core/features.py), [entity history](../../dclab_core/entities.py) and [behavior transforms](../../dclab_core/behaviour.py) where their semantics match.

| Feature | Rule at a new offered price |
|---|---|
| Proposed offer and change from last paid price | Override only the proposed price. Recompute its dependent features. If the prior paid price is missing or zero, mark the relative change unavailable. |
| Last five events, time since last purchase, prior discount usage | Freeze history at the decision time. For each person, take the latest five distinct events that happened and were available strictly before the cutoff. Break equal historical timestamps with a stable event ID. |
| Support contact after seeing the new price, eventual cancellation time | Exclude from decision-time prediction. Keep a planted example for the isolated leakage demonstration in P4. |
| Observed purchase, churn or realized revenue | Use matured outcomes as labels for the historical offer. Generate scenario predictions in separate fields. |

Return the actual count for people with zero to four prior events. Represent an undefined rate as missing. Define, for example, discount share as discounted purchases divided by purchases among those five events. Learn imputers, encoders, feature selection and calibration using training folds only. Preserve the production split rules.

**Acceptance.** With six eligible events use the latest five. A later event, a late-arriving event or another person's event cannot change an earlier prediction. Shuffling the input preserves the result. Test zero history. Raising a proposed price from 100 to 110 with a previous paid price of 100 changes the relative-price feature to 10% and leaves history unchanged. Reproduce all three break-even rows above.

### P2. Deliver the scenario calculation

Run the same frozen eligible cohort through the same fitted scoring pipeline at each price. Require a price-aware model or explicitly supplied response assumptions. Preserve valid ranges and report unsupported price levels by cohort. Use assumed scenarios if historical prices were fixed. Never refit on edited observations. If only transactions are available, offer the break-even or assumption calculation with a user-supplied audience size.

For one unit per offer, expected purchases are the sum of individual purchase probabilities. Expected revenue is the sum of each offered price times its probability. Expected churn is the sum of churn probabilities for eligible subscribers. Report probability changes in percentage points. Accuracy of churn predictions and response to price are separate questions.

For equal independent probabilities use the binomial calculation. For unequal independent probabilities sum Bernoulli outcomes analytically or by seeded simulation. Shared shocks or repeated customers require a justified dependence model or grouped resampling; otherwise omit the downside probability with a reason. Separate natural outcome variability from uncertainty in the estimated response. Record the simulation seed and number of draws when applicable.

**Acceptance.** On the example, require 18 expected purchases, 1,980 expected revenue and risk within 0.0001 of 0.56255 using the exact calculation. With fixed customer probabilities 0.2 and 0.8 at price 110 expect one sale and revenue 110. With churn probabilities 0.10 and 0.20 expect 0.30 churns and a 15% cohort churn rate. Prove zero price change preserves prediction parity. A scorer test must fail if fitting, final holdout access or label mutation occurs.

### P3. Estimate effects when evidence supports them

Read the existing [price variation check and price curve](../../dclab_core/decide.py), [training-side engine integration](../../dclab_rnd/studio/engine.py) and [binary treatment overlap and wording rules](../../dclab_core/uplift.py). The current price curve estimates margin over a price grid. A purchase probability, revenue and margin are distinct outputs. Preserve the current distinction between selling to a customer and paying a courier.

Begin with two randomized price arms and a prespecified cohort, outcome and horizon. Estimate the difference in purchase or churn rates by assigned arm and show sample sizes and an interval at the unit of assignment. For historical comparisons, document why prices differed, identify pre-offer confounders, check overlap within comparable groups, and adjust both price predictions to the same eligible population. A market-wide price range alone does not establish overlap for a subgroup. Reuse diagnostics that fit the design. Fit and choose models on development folds respecting time and customer dependence. If price was fixed by segment, seasonal promotions coincided with the change, or material confounding cannot be addressed, provide descriptive results or an assumption-only scenario.

**Acceptance.** A balanced invented two-arm table with 20 purchases in 100 baseline assignments and 15 in 100 higher-price assignments yields an unadjusted difference of minus five percentage points, with uncertainty. A fixture with prices fully determined by segment must withhold the causal estimate. Repeat with incomplete follow-up and a requested price outside observed support. In the existing [ride-pricing replica](../../dclab_core/replicas/ride_pricing.py), use both varied and fixed-price versions to exercise support handling. The courier response direction is specific to that fixture.

### P4. Test hypotheses and quantify leakage

Predeclare one hypothesis: prior discount buyers respond more strongly to a price increase. Fix the historical definition of the segment, a minimum supported sample size per arm, the outcome, horizon and evaluation split before measuring. Compare the **price effects** between the two groups with an interval. Comparing their raw purchase rates alone answers a different question. Report inconclusive findings when data are thin.

On invented development data compare (1) a safe baseline, (2) the same pipeline with safe last-five features and (3) a separate diagnostic copy with a deliberately planted post-offer feature. Match rows, folds, seed and scoring. Report sample sizes and changes in Brier score or log loss. Use a dated development validation window for comparing features. Isolate the diagnostic model and artifacts from serving, export, feature selection and promotion. The final holdout stays locked during development.

**Acceptance.** Alter the planted future feature while leaving past inputs unchanged. Safe predictions remain identical. The diagnostic demonstration can read the plant and identifies it by source and availability time. The product refuses to serve or export its artifact. Keep the safe feature comparison even if history adds no predictive value.

### P5. Put the result in the normal product path

Extend the [Studio decision service](../../dclab_rnd/studio/decision.py), [project route](../../dclab_rnd/agentic/routers/projects.py) and [Decision Brief](../../dclab_rnd/agentic/web/src/views/decision.html). The existing what-if path changes decision terms and currently excludes price re-scoring. Preserve its current callers. Add a typed price-scenario variant or sibling route returning P2 and, when eligible, P3. Persist or reference the approved safe scorer and feature specification through existing storage if the project currently stores only a curve. Legacy projects without the required scorer may still show assumptions or a supported recorded analysis. Use existing project permissions.

**Acceptance.** From a project, enter +5% and +10%, choose the target, see the comparison, assumptions, support and unavailable fields. Repeat without a model, with a fixed historical price and with an unauthorized project. Check that input data, fitted model and final-stage record remain unchanged. Run a focused route/UI check and the existing decision what-if regression.

## 4. Simulation datasets, tests and measured comparisons

The simulator must answer a question that historical sales alone cannot: how far is each method's estimate from the known response to changing the price? Use invented customers and realistic linked tables. Keep the simulator's hidden answers in the evaluator. The first two tables below contain results from actual development runs. They provide a reproducible starting point for the fuller product benchmark.

### First measured comparison: customer purchases

Each invented customer receives one offer. Half have used discounts on two of their last five purchases; half on four. Within each group, assign equal numbers to prices 100, 105 and 110. The simulator sets purchase probabilities to 20%, 18%, 16% in the first group and 40%, 30%, 20% in the second. Each assigned offer produces one random observed outcome. Fit three simple frequency models on 4,800 offers and evaluate all three on the same 1,800 separately generated offers. Use the same test audience for scoring each alternative price. A Brier score is the average squared error of the assigned-price probability against the observed purchase; smaller is better.

| Method | Test Brier score | Expected purchases per 100 at 100 | At 105 | At 110 | Estimated change from 100 to 110, percentage points |
|---|---:|---:|---:|---:|---:|
| Flat training purchase rate | 0.19295 | 23.65 | 23.65 | 23.65 | 0.00 |
| Training rate by price | 0.18995 | 29.44 | 22.62 | 18.88 | −10.56 |
| Training rate by price and last-five discount group | 0.18396 | 29.44 | 22.62 | 18.88 | −10.56 |
| Simulator truth for the fixed audience | — | 30.00 | 24.00 | 18.00 | −12.00 |

In this balanced run, history improves individual prediction while leaving the overall estimated price effect unchanged. Both price-aware methods miss the true change by about 1.44 percentage points. This tiny benchmark exercises the calculation. It uses complete five-event histories, disjoint invented people and one pair of seeds. It does not establish performance for real customers, churn or the Studio entry point.

<details>
<summary>Reproduce the table with Python 3 (standard library only)</summary>

```python
from collections import defaultdict
from random import Random

prices = (100, 105, 110)
truth = ((0.20, 0.18, 0.16), (0.40, 0.30, 0.20))

def offers(per_cell, seed):
    rng = Random(seed)
    rows = []
    for group in (0, 1):
        for _ in range(per_cell):
            assignments = list(range(3))
            rng.shuffle(assignments)
            for arm in assignments:
                last_five = [0, 0, 0, 1, 1] if group == 0 else [0, 1, 1, 1, 1]
                bought = int(rng.random() < truth[group][arm])
                rows.append((arm, int(sum(last_five) >= 3), bought))
    return rows

train, future = offers(800, 19), offers(300, 20)
for name, key in (("Flat rate", lambda a, g: ()),
                  ("Price only", lambda a, g: (a,)),
                  ("Price and history", lambda a, g: (a, g))):
    cells = defaultdict(list)
    for arm, group, outcome in train:
        cells[key(arm, group)].append(outcome)
    rates = {k: sum(v) / len(v) for k, v in cells.items()}
    def p(arm, group):
        return rates[key(arm, group)]
    brier = sum((p(a, g) - y) ** 2 for a, g, y in future) / len(future)
    counts = [100 * sum(p(a, g) for _, g, _ in future) / len(future)
              for a in range(len(prices))]
    print(name, round(brier, 5), [round(x, 2) for x in counts],
          round(counts[2] - counts[0], 2))
```

</details>

### Existing Studio pricing logic: measured support checks

Run `PYTHONPATH=tests .venv/bin/python -m unittest test_core_decide.PriceTest` from the repository root. The [price test class](../../tests/test_core_decide.py) and [ride-pricing generator](../../dclab_core/replicas/ride_pricing.py) reproduce the following checks with seed 0. Each row represents 6,000 invented courier offers. Here the platform pays a fee to the courier.

| Simulation world | Accepted offers | Known best fee multiplier | Estimated best and bootstrap interval | Test outcome |
|---|---:|---:|---|---|
| Random variation around the usual fee | 3,432 | 1.0200 | 1.0291 [1.0082, 1.0500] | Passed. The known best falls inside the interval. |
| Fee determined by the usual formula | 3,295 | 1.0200 in the simulator | Withheld | Passed. The public table cannot identify an independent fee effect. |

All three tests in the class passed, including the separate variation check. These are development simulations. The locked product holdout was not used.

### Dataset contract for the full benchmark

| Physical table | One row per | Minimum columns and purpose |
|---|---|---|
| Offers | Eligible offer or renewal opportunity | Offer ID, customer ID, offer time, price, currency, assignment arm, source of assignment, eligibility, split. Include declined offers. |
| Events | Customer event | Event ID, customer ID, occurrence time, available time, type, discount indicator. Generate short histories, late arrivals and same-time events. |
| Outcomes | Offer and outcome horizon | Offer ID, observation end, outcome availability, purchase or defined churn indicator and realized revenue. Incomplete follow-up stays unknown. |
| Private simulation truth | Eligible offer and candidate price | True purchase or churn probability and expected value at each price. Only the evaluator may read this table, after predictions are fixed. |
| Predictions and test results | Run, method, world, split, seed, offer and price | Prediction or refusal, reason, provenance, diagnostic-only flag, plus aggregate metrics linked to these rows. |

Use actual CSV or Parquet fixtures and the normal import, feature, scoring and report paths for the integration test. Preserve a generator version, schema, seeds, row counts, checksums, cohort definition and split manifest. Run every candidate on identical saved test rows. Assert unique keys and expected joins. Fail the run if the truth table or any post-offer field enters a deployable feature set. A separately authorized historical table can later exercise the same import path and factual metrics. Its unobserved alternative outcomes remain unknown.

### Required worlds, comparisons and gates

| World | Test | Evidence the generated report must show |
|---|---|---|
| Randomized prices for sales and, separately, renewals | Compare flat rate, price only, price plus safe last-five history and the assigned-arm effect estimate. | Factual Brier score, expected sales or churn at all three prices, expected revenue where defined, effect error versus truth, counts by arm and group. |
| Discount-sensitive segment and a second world with zero effect | Compare subgroup effects; forecast a fixed cohort at all three prices. | Segment counts, uncertainty, estimated and true differences, and frequency of false effect claims across seeds. |
| Prices confounded by season or segment; fixed price or missing overlap | Compare the unadjusted estimate with adjustment from pre-offer context and the support guard. | Paired effect errors where supported; counts and reasons for refusals elsewhere. A causal claim in the fixed-price world fails the gate. |
| Short or delayed histories, incomplete labels and a planted future signal | Compare safe baseline and safe engineered features. Score a separate diagnostic with a future signal on synthetic development rows only. | Matched factual scores, excluded or immature counts, time-cutoff checks and an explicit refusal to serve the diagnostic. |
| Shared demand shock | Compare the reported probability of missing the revenue target with the frequency across independently simulated futures from the same world. | Risk error, uncertainty and the assumed dependence model. Report unavailable risk if the implemented model cannot represent the dependence. |

**How to finish this benchmark.** Freeze generator, split, seeds, methods, metric tolerances and interval targets before inspecting comparative scores. Keep test customers and future periods out of model fitting, and train only on labels mature by that split's cutoff. Use development simulations for debugging. For the frozen simulation comparison, run at least 30 independent worlds per stochastic setting on prespecified seeds, pair methods within each seed, and report the mean paired difference with a 95% interval across worlds. Check the Monte Carlo precision before interpreting nominal coverage. Run timing, safety and support gates separately from statistical improvement. Keep inconclusive comparisons and negative results.

The coding deliverable includes the dataset manifest, per-offer predictions, a machine-readable results table and the corresponding Markdown table generated from the same rows. Include method, world, split, sample size, metric, estimate, interval, truth or reference, difference and pass/fail/unavailable reason. Preserve denominators when methods abstain. The exact arithmetic, time-cutoff, immutable-label and refusal tests must all pass. Publication of a model-quality gain requires the prespecified improvement criterion and its uncertainty. Empty result cells require an explanation. Replaying one recorded seed must reproduce its tables. The two measured examples above do not mark the full benchmark complete.

## 5. What to build next if the first version works

1. **Find the break-even point.** Compare each model prediction with the minimum sales or retention needed to meet a revenue target. Useful before credible price-response estimates exist.
2. **Compare customer groups.** Show response and uncertainty for a few predefined groups with enough comparable offers. Include group size and coverage beside every number.
3. **Design a price experiment.** When evidence is weak, draft two price arms, randomization unit, primary outcome, sample-size assumptions and observation window. Keep real experiment launch a separate business decision.
4. **Check the forecast later.** Once outcomes mature, compare predictions with realized purchases or churn in each deployed arm. Report calibration changes, audience mix and coverage. Use a comparison group for claims about policy impact.

## 6. Handoff prompt for the coding model

Deliver the simulation dataset and measured comparison in section 4 alongside P1–P5. Start with the reproducible purchase example and existing pricing support checks. Complete the linked-table benchmark as product features become available and attach its generated results before closing the implementation.

Implement P1 first, then P2 as the first usable calculator. Read the repository's AGENTS.md and the closest existing decision and feature tests. Start with the worked revenue example and the point-in-time history fixture. Reuse established contracts and add the smallest behavior needed for each acceptance check. Keep the core calculation reusable outside Studio. Wire subsequent packages through existing Studio entry points. Use synthetic fixtures, frozen seeds and development data. Run only the affected tests, using the repository's targeted runner. Relevant starting points include [core decision tests](../../tests/test_core_decide.py), [feature tests](../../tests/test_core_features.py), [uplift tests](../../tests/test_core_uplift.py) and [what-if compatibility tests](../../tests/test_core_decide_whatif.py). Deliver each package with the exact checks run, their outputs, known limits and next package. Completion requires both an honest answer when evidence is sufficient and an honest unavailable result when it is insufficient.

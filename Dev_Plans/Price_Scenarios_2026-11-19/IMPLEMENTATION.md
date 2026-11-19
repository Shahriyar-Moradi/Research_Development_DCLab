# Running the price scenario implementation

## Open the product

In a project, open **Decision → Explore price changes**. Enter the current price, the revenue target and the audience size. Enter event probabilities as percentages and select **Compare assumptions**. Blank probabilities still show the break-even count. For churn each eligible person is a subscriber who will renew one unit unless they churn. Optional downside risk compares revenue with a fixed target under independent outcomes.

## Prepare a measured study

Use a project with a main table. The three files below are a complete, invented example. Under **Estimate response from project tables**, select each related table name, choose its corresponding CSV, then attach. The existing attachment workflow clears dependent stages. A fresh project is convenient for the example.

| Table | Columns and interpretation | Example |
|---|---|---|
| price_offers | One offer per eligible person, including declines. Required: offer_id, customer_id, offered_at, price, currency, segment, split. The split is train or score. Segment must be known before the offer. | [Offers](simulation/sales/price_offers.csv) |
| price_events | event_id, customer_id, occurred_at, available_at and discounted (0 or 1). Optional kind defaults to purchase; paid_price describes the historical purchase. The last five available events determine discount usage. | [Event histories](simulation/sales/price_events.csv) |
| price_outcomes | offer_id, available_at and outcome (0 or 1). For a purchase study 1 means bought. For a renewal study 1 means churned by the defined horizon. Score labels can be omitted entirely. | [Outcomes](simulation/sales/price_outcomes.csv) |

For the linked example choose base price 100, purchase, 30 days and randomized assignment. Describe the assignment (the simulator randomized a price for each person). Click **Prepare study**. **Read saved study** can use a different revenue target without retraining. A change in an attached table invalidates the saved study. The checkbox for independent outcomes controls downside-risk estimates. The assigned design is a claim supplied by the study owner; the software cannot verify the original randomization.

The estimator uses training purchase or churn rates by actual price, prespecified customer segment and prior discount band. It reweights these rates to the frozen scoring cohort. A forecast requires ten matured examples in each applicable stratum at that price. Randomized contrasts use the difference in assigned-arm rates for independently assigned people. Observational contrasts condition on segment and discount history and depend on the absence of omitted confounders. Unknown assignment allows a model prediction while withholding a causal estimate. The effect intervals use a fixed cohort and independent-person approximation. Shared shocks and repeated-person experiments require a different uncertainty analysis. Older projects retain their recorded pricing curves.

## Source and tests

| Entry point | Responsibility |
|---|---|
| [Portable pricing calculations](../../dclab_core/price_scenarios.py) | Scenario math, last-five history, maturity, support, response rates and effect intervals |
| [Project study service](../../dclab_rnd/studio/price_scenarios.py) | Read project-owned tables, save aggregate fitted rates with input hashes, refuse stale studies |
| [Project API](../../dclab_rnd/agentic/routers/projects.py) and [Decision Brief](../../dclab_rnd/agentic/web/src/views/decision.html) | Authorized upload, preparation and read-only scoring |
| [Simulation runner](../../dclab_rnd/agent_eval/price_scenarios.py) | Generate the seven worlds, compare methods and write CSV predictions, evidence and report |
| [Focused tests](../../tests/test_price_scenarios.py) | Arithmetic, point-in-time history, two-arm estimates, refusals, mutation checks and API integration |

Run from the repository root. Choose a new or empty output directory for a benchmark replay.

```sh
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --no-fork test_price_scenarios test_web_client
.venv/bin/python -m dclab_rnd.agent_eval.price_scenarios --out /tmp/dclab-price-study-replay
.venv/bin/python -m dclab_rnd.agentic.web.build --check
```

The [measured report](simulation/REPORT.md) summarizes 30 paired seeds per world. The [metrics](simulation/results.csv) and [effect estimates](simulation/effects.csv) hold per-seed results. Each world's directory contains the first seed's physical input tables, predictions and evaluator-only truth. The compressed prediction tables cover every seed. The [manifest](simulation/manifest.json) records hashes. Use only the three price-prefixed inputs in the product; the other files are evaluation outputs.

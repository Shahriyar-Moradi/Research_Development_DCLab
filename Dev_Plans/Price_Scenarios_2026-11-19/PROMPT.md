# Coding prompt: price scenarios, churn and leakage-safe evaluation

Use this prompt with a coding model in the DCLab R&D checkout.

## Objective

Deliver a tested Decision Brief for the current offer price, +5% and +10%. A user can enter assumptions immediately or prepare an evidence-based study from historical offers and score its saved cohort repeatedly. Explain the difference between a predicted outcome and an estimated effect of changing price. Use the existing implementation as the starting point.

## Read first

Read the repository AGENTS.md, the [product and test plan](Price_Scenario_Planning.md), [implementation guide](IMPLEMENTATION.md) and [measured simulation report](simulation/REPORT.md). Inspect the route, portable calculation and nearest tests before changing behavior.

## Work order

1. Run the numerical example. For 100 independent offers at 110, purchase probability 0.18 and fixed revenue target 2,000, require 19 break-even units, 18 expected sales, expected revenue 1,980 and downside probability about 0.56255. A blank probability leaves the revenue forecast unavailable. For churn, use the declared renewal horizon and count retained subscriptions for revenue.
2. Check event history on real CSV-shaped fixtures. Use only the last five eligible events as known before the offer. Test duplicate deliveries, timestamp ties, late arrivals, cold starts and future-label edits. A past payment stays fixed across prices.
3. Exercise the ordinary project upload, prepare and saved-study routes. Train on matured earlier labels. Score disjoint later customers. Validate currency, eligibility, sample support and the stated assignment design. A model trained on a modified price with unchanged observed labels is invalid. Keep the study immutable on read and revalidate its input fingerprints.
4. Run the simulation comparison on matched seeds and rows. Compare the flat baseline, price-only estimator and price plus safe history. Use the planted post-offer signal solely as a synthetic leakage control. Preserve abstentions, sample sizes, truth held by the evaluator, factual scores, effect errors, uncertainty, subgroup results and downside-risk assumptions.
5. In the Decision Brief verify the three prices, units, cohort size, evidence source, unavailable estimates, risk and effect intervals. Preserve the pre-existing decision what-if API and scoped project access. Test an older project with no price study.
6. Use focused tests for the modules changed. Rebuild the generated web client and static app using the existing build command. Record the exact checks, measured outcomes, unsupported cases and any change in the simulated comparison before closing a package.

## Completion evidence

Hand off the source paths, user entry point, a rerunnable command, the generated results table, and the test outcomes. A statistical gain needs a paired comparison and its uncertainty. A fixed-price dataset that cannot establish a price effect must produce an explicit refusal. Keep the full simulation offline and preserve the final product holdout.

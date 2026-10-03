# Agent verification: blind leakage-auditor replay

Without being told, does the deterministic column auditor flag the leakage columns the R&D already confirmed?

The auditor ran with `blind=True` (no catalog, no precedents) on 7 datasets with 18 confirmed leakage columns. It flagged 11 of them (recall 61%).

| Dataset | Known leaks | Found | Missed | Other flags |
|---|---|---|---|---|
| bank_marketing | 1 | duration | — | — |
| online_shoppers | 9 | PageValues, Administrative_Duration, Informational_Duration, ProductRelated_Duration | Administrative, Informational, ProductRelated, BounceRates, ExitRates | — |
| hyperack | 2 | final_customer_fare, final_biker_fare | — | first_created_at, deliverey_category_id |
| telco_churn | 1 | customerID | — | TotalCharges |
| credit_card_fraud | 1 | — | Time | — |
| bike_sharing_daily | 2 | casual, registered | — | cnt_2d_bfr, days_since_2011, dteday, instant |
| ecommerce_clothing_reviews | 2 | Rating | Positive Feedback Count | Title, Review Text, Unnamed: 0 |

## How to read this

- A **miss** is a leak whose numbers look ordinary. Only the decision-time contract (when the value is written) can catch it, which is why the agent must ask for that contract and never treat a clean heuristic scan as proof of safety.
- An **other flag** is a review request, not an accusation. A strong legitimate predictor is flagged too (DCLAB-R05).
- This replay is the regression test for the auditor: any change to its heuristics must not lower recall.

Re-run: `python -m dclab_rnd.tools verify-auditor`.

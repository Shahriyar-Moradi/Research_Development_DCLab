# Price scenario simulation results

30 paired seeds per world. Each seed uses 1,200 earlier and 400 later customers. Lower Brier is better.
The interval is a paired bootstrap over independent generated worlds. Coverage counts scoreable people at all three prices.

| World | Method | Coverage | Brier | Brier gain vs flat (95% interval) | Estimated change, pp |
|---|---|---:|---:|---|---:|
| sales | Flat | 12000/12000 | 0.2401 | 0.0000 [0.0000, 0.0000] | 0.00 |
| sales | Price | 12000/12000 | 0.2388 | 0.0013 [0.0006, 0.0021] | -9.63 |
| sales | Price + history | 12000/12000 | 0.2342 | 0.0059 [0.0039, 0.0082] | -9.34 |
| sales | Future label diagnostic | 0/12000 | 0.0000 | diagnostic or differing coverage | unavailable |
| renewals | Flat | 12000/12000 | 0.2382 | 0.0000 [0.0000, 0.0000] | 0.00 |
| renewals | Price | 12000/12000 | 0.2370 | 0.0011 [0.0005, 0.0018] | 8.59 |
| renewals | Price + history | 12000/12000 | 0.2305 | 0.0076 [0.0057, 0.0094] | 8.46 |
| renewals | Future label diagnostic | 0/12000 | 0.0000 | diagnostic or differing coverage | unavailable |
| no_effect | Flat | 12000/12000 | 0.2476 | 0.0000 [0.0000, 0.0000] | 0.00 |
| no_effect | Price | 12000/12000 | 0.2480 | -0.0004 [-0.0008, -0.0000] | -0.60 |
| no_effect | Price + history | 12000/12000 | 0.2399 | 0.0077 [0.0053, 0.0103] | -0.35 |
| no_effect | Future label diagnostic | 0/12000 | 0.0000 | diagnostic or differing coverage | unavailable |
| confounding | Flat | 12000/12000 | 0.2382 | 0.0000 [0.0000, 0.0000] | 0.00 |
| confounding | Price | 12000/12000 | 0.2384 | -0.0001 [-0.0006, 0.0004] | -1.36 |
| confounding | Price + history | 12000/12000 | 0.2328 | 0.0054 [0.0039, 0.0069] | -6.85 |
| confounding | Future label diagnostic | 0/12000 | 0.0000 | diagnostic or differing coverage | unavailable |
| fixed | Flat | 12000/12000 | 0.2404 | 0.0000 [0.0000, 0.0000] | 0.00 |
| fixed | Price | 0/12000 | 0.2406 | -0.0002 [-0.0005, 0.0002] | unavailable |
| fixed | Price + history | 0/12000 | 0.2366 | 0.0038 [0.0023, 0.0055] | unavailable |
| fixed | Future label diagnostic | 0/12000 | 0.0000 | diagnostic or differing coverage | unavailable |
| late | Flat | 12000/12000 | 0.2402 | 0.0000 [0.0000, 0.0000] | 0.00 |
| late | Price | 12000/12000 | 0.2388 | 0.0013 [0.0006, 0.0021] | -9.85 |
| late | Price + history | 12000/12000 | 0.2347 | 0.0054 [0.0035, 0.0075] | -9.54 |
| late | Future label diagnostic | 0/12000 | 0.0000 | diagnostic or differing coverage | unavailable |
| shock | Flat | 12000/12000 | 0.2399 | 0.0000 [0.0000, 0.0000] | 0.00 |
| shock | Price | 12000/12000 | 0.2392 | 0.0007 [-0.0001, 0.0015] | -9.64 |
| shock | Price + history | 12000/12000 | 0.2332 | 0.0067 [0.0053, 0.0082] | -9.53 |
| shock | Future label diagnostic | 0/12000 | 0.0000 | diagnostic or differing coverage | unavailable |

## Effect estimates

| World | Estimates / seeds | Mean estimate, pp | Mean truth, pp | Intervals covering truth |
|---|---:|---:|---:|---:|
| sales | 30/30 | -9.63 | -8.96 | 27/30 |
| renewals | 30/30 | 8.59 | 8.96 | 28/30 |
| no_effect | 30/30 | -0.60 | 0.00 | 26/30 |
| confounding | 30/30 | -6.85 | -8.96 | 27/30 |
| fixed | 0/30 | withheld | -8.96 | 0/0 |
| late | 30/30 | -9.85 | -8.96 | 26/30 |
| shock | 30/30 | -9.64 | -8.96 | 28/30 |

## Diagnostics

Under a shared demand shock, the mean frequency of revenue missing the target was 50.9%. Independent resampling gave 6.0% for those same targets.
The risk target was 90% of expected unit sales at 110, rounded to a whole unit and converted to revenue. Each estimate uses 300 futures per seed.

The future-label control reads the planted confirmation on invented development rows. Its zero Brier score exposes leakage. It has no deployable model or counterfactual predictions.

Per-seed effect intervals, subgroup contrasts and shock results are in effects.csv. Replicated seeds measure simulation uncertainty. Interval coverage is descriptive with only 30 worlds.

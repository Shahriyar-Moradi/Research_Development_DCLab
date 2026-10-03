# Recommender systems

**Status:** planned

## Research question

Which recommendation approaches (popularity, matrix factorization, two-tower, sequence models) work best for sparse business interaction data, under time-ordered evaluation?

## Why it matters for DCLab

Next-best-offer and product ranking are common DCLab requests.

## Leakage traps specific to this field

- **Future interactions in training:** random interaction splits leak future preferences; split by time per user.
- **Popularity computed on all data,** including the test period.
- **Evaluating on items the user had already seen.**

## First experiments

| ID | Question |
|---|---|
| REC-001 | Popularity and item-kNN baselines vs ALS on MovieLens with leave-last-out time splits |
| REC-002 | Inflation from random vs time-ordered splits |
| REC-003 | Cold start: content features vs collaborative signals for new items |

## Candidate datasets

MovieLens, Amazon reviews (implicit feedback), RetailRocket.

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=recommender-systems TITLE="Recommender systems" PREFIX=REC
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.

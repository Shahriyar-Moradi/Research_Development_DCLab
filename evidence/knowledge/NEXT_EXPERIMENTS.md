# Next Experiments

Generated from gaps in the current evidence registry. Complete Priority 1 items before adding more unconstrained model searches.

## P1 — Add probability-quality benchmarks

**Why:** The registry is rich in ranking metrics, but it has no systematic Brier score, log loss, or calibration error evidence.

**Experiment:** Evaluate the top three safe model families with Brier score, log loss, ECE, and calibration curves on the locked split.

**Success gate:** Select a calibrated champion that preserves ROC-AUC while improving Brier score and expected business cost.

## P1 — Measure winner stability, not only point estimates

**Why:** Most stored results expose one locked split and no confidence interval, so small leaderboard differences may be noise.

**Experiment:** Run repeated stratified 5-fold CV (5 repeats) for the top three deployment-eligible families and bootstrap paired score differences.

**Success gate:** Promote a winner only when its median lift is positive and the 95% paired interval is decision-useful.

## P1 — Replicate champions with full provenance

**Why:** Only 1.7% of completed historical runs contain code, data, seed, and environment provenance.

**Experiment:** Re-run each deployment champion with the new provenance-enabled runners; require identical data hashes and report score drift.

**Success gate:** All active champions are reproducible from a clean checkout and differ by no more than 0.001 ROC-AUC.

## P2 — Add temporal and operational holdouts for HyperAck

**Why:** A random locked split can overestimate production performance when pricing, geography, or demand changes over time.

**Experiment:** Backtest the safe champion on later-time, unseen-zone, and high-demand slices; compare AUC, recall, calibration, and coverage.

**Success gate:** Document the worst-slice floor and define a retraining or rollback threshold before deployment.

## P3 — Challenge the cross-dataset leader: lightgbm

**Why:** It currently covers 11 datasets with mean rank 2.55; that is evidence, not a universal guarantee.

**Experiment:** Add three datasets from different row-count, imbalance, and categorical-cardinality regimes and compare rank stability against GBDT and ensemble baselines.

**Success gate:** The preferred default is based on broad rank stability plus runtime/cost, with documented exceptions by dataset profile.

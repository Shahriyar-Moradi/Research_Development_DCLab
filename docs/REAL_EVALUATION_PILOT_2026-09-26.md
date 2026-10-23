# DCLab real evaluation: first pilot (26 September 2026)

This is a measured development pilot, **not proof of overall correctness**. It
tests existing code and notebooks, records provider failures, and leaves the
release decision false. No notebook cells were executed, no held-out model set
was used, and no immutable trial evidence was edited.

## What was actually measured

| Dimension | Result | Evidence | Meaning |
|---|---:|---|---|
| Existing notebook copilot, fixed cell-level checks | 6/7 | `hyperack_churn_notebook_pilot_2026-09-26_v2.json` | One known issue missed; tiny, non-blinded development suite |
| Retrieved proof matched the predeclared relevant rule | 3/4 | Same report | The churn availability warning cites an irrelevant rule |
| Churn availability warning worded cautiously | 0/1 | Same report | It says a name-matched field is “confirmed leaky”; that is not established |
| Real Research Planner response | 0/1 completed in three attempts of the *same* case | `live_specialists_2026-09-26_probe.json`, `live_specialists_2026-09-26_retry.json`, `live_specialists_2026-09-26_third_probe.json` | Provider returned `InternalServerError`; decision quality cannot yet be scored |
| Archived Research Studio cycles | 3/6 completed | `research_studio_scorecard_2026-09-26_current.json` | Operational completion, not agent intelligence |
| Archived trial records passing the independent evidence audit | 7/12 | Same scorecard | Five records need evidence/reporting remediation; see issue codes in JSON |
| Historical agent trace outputs audited | 37 outputs across 6 runs | `historical_agent_traces_2026-09-26_v2.json` | Necessary contract checks on real agent text, not blinded decision accuracy |
| Current-worker bank trial reproduction | 4/4 metrics identical; OOF bytes identical | `bank_trial_replay_comparison_2026-09-26.json` | One case reproduced across changed executor/catalog code; not an exact-version replay |

The first two rows are not independent population accuracy estimates. Several
checks share the same two notebooks and the same prototype implementation. Do
not turn 6/7 into a claim that the copilot is 86% accurate in real use.

## Concrete cases

- `hyperack_full_data_clustering`: the HyperAck notebook fits a scaler and
  KMeans on the full coordinate dataframe and adds `geo_cluster` before the
  later train/test split. The current same-cell detector missed this path.
- `hyperack_scale_before_split`: it did catch full-data `StandardScaler`
  fitting before the split in another HyperAck cell.
- `churn_scale_before_split`: it caught `StandardScaler.fit_transform(X)`
  before the churn train/test split.
- `churn_cancellation_date_availability`: the field name warrants an
  availability question **if** predicting future churn before cancellation.
  Without the prediction moment and source timestamps, leakage is not proven.
  The current comment overstates certainty and its selected rule (`DCLAB-R06`)
  discusses score-gap interpretation rather than name-based review
  (`DCLAB-R05` is the directly relevant rule).
- `hyperack_eda_not_training`: the descriptive fare plot was not falsely
  marked as training leakage.

The frozen manifest stores full cell hashes. If a notebook changes, the suite
stops and asks for relabeling instead of silently scoring altered code against
stale expectations. This is especially important for the user's evolving
HyperAck notebook.

## What this does not yet establish

The NOOA specialist suite has five fixed cases, but only the first was
attempted. The configured `gpt-5.6-terra` provider returned
`InternalServerError` three times, with no response or token-usage record. This is
not a demonstrated scientific error by Research Planner, and it is not an
agent pass. It is an operational failure. Recheck provider availability and
rerun the suite; do not replace those failures with mock responses.
One diagnostic call using `gpt-6-astra` failed with the same error
(`live_specialists_2026-09-26_alternate_model_probe.json`), so simply changing
the model name did not restore access. No further API retries were made.

The prior archive audit found no independent agent-decision accuracy, no user
task success measure, and no fresh model confirmation. Existing 50-experiment
work is useful *development evidence*, not a proof that an inline agent will
make correct suggestions for a new scientist or industry.

The historical agent trace audit found 4/4 first proposals were transparent
baselines, 26/26 proposal/critique citation checks referenced prior successful
trials, 13/13 proposals used their run's selected dataset, and the one failed
trial was acknowledged without citing it as performance evidence. But only
5/9 comparison proposals carried an explicit reference trial ID, and 1/37
agent outputs no longer validates against the current schema. These are
cross-version development traces; passing the checks does not show the agent
chose the best next experiment or gave correct scientific advice. Two completed
bank runs share a source, so they are not independent generalization cases.

The fresh bank-marketing worker run used 2,000 rows and 2 × 3-fold development
CV. ROC-AUC 0.6964559268, average precision 0.3239817829, log loss
0.2999823954, and Brier 0.0804601666 exactly matched the archived trial;
4,000 out-of-fold prediction records and the split fingerprint matched byte
for byte. This verifies reproducibility for that one configuration, not
predictive validity on a new time period or population. The original replay
command refused because the current worker and catalog hashes differ, which is
the correct provenance safeguard.

## Next evidence needed

1. The owner defines HyperAck's decision moment, prediction target, and which
   source fields exist at that moment.
2. The owner defines churn's snapshot time, outcome horizon, and cancellation
   field semantics.
3. Resolve or route around the model provider's server failure, then run all
   five fixed specialist cases and expert-review their actual responses.
4. Fix the missed cross-cell preprocessing pattern and overconfident wording,
   then keep these seven cases as development tests.
5. Build a **new**, source-held-out, blinded notebook set with both clean and
   flawed examples. Compare agent-assisted and unassisted task completion,
   correction time, citation correctness and severe false-alarm rate.

Only after those checks should a new workflow or SLM-training dataset be
considered for release. Training on these pilot answers now would contaminate
the evaluation set.

The short owner worksheet is `evaluation/OWNER_INPUTS.md`; it asks for the
prediction contracts, review labels and provider-access check needed for the
next evaluation round without requesting any secret in chat.

## Reproduction

```bash
.venv/bin/python -m dclab_rnd.notebook_eval --output output/evaluation/NEW_NOTEBOOK_REPORT.json
.venv-agent/bin/python -m dclab_rnd.agentic.live_eval --output output/evaluation/NEW_LIVE_REPORT.json
.venv-agent/bin/python -m dclab_rnd.agentic.trace_eval --output output/evaluation/NEW_TRACE_REPORT.json
.venv/bin/python -m dclab_rnd.replay_compare --reference agent_runs/eb59895d248c4c909bc731127ca05370/trial-001 --candidate output/evaluation/current_worker_bank_trial_2026-09-26 --output output/evaluation/NEW_REPLAY_COMPARISON.json
.venv/bin/python -m unittest discover -s tests -v
.venv-agent/bin/python -m pytest tests/test_agentic.py tests/test_live_eval.py tests/test_trace_eval.py -q
```

Use new output names: evaluation artifacts are immutable snapshots.

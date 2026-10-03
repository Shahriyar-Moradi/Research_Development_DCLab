# Notebook copilot

The copilot reads a Jupyter notebook and writes review notes beside the cells that contain methodology mistakes. Every note has three parts:

1. **What is wrong**, in plain words, at a specific cell and line.
2. **The fix**, concretely.
3. **Show proof**: the DCLab rule it violates and a *measured* precedent from the R&D campaigns, with the source file.

The proof is the point. A data scientist can check that someone (the DCLab R&D campaigns) already hit this exact problem, how much it cost, and how it was resolved. That is the "Stack Overflow with evidence" idea, built from DCLab's own experiments.

## Try it

```bash
python -m dclab_rnd.copilot review dclab_rnd/copilot/examples/leaky_bank_marketing.ipynb
python -m dclab_rnd.copilot review NOTEBOOK.ipynb --html review.html        # the "dclab notebook" view
python -m dclab_rnd.copilot review NOTEBOOK.ipynb --annotate reviewed.ipynb # copy with review cells inserted
python -m dclab_rnd.copilot review NOTEBOOK.ipynb --fail-on high            # exit 1 for CI or pre-commit
```

The original notebook is never modified, and the code is never executed (static analysis only).

## The demo, end to end

[`examples/leaky_bank_marketing.ipynb`](../../dclab_rnd/copilot/examples/leaky_bank_marketing.ipynb) is a realistic first-pass notebook on the real bank-marketing data. When run, it reports **96.7% accuracy**. The copilot raises 10 notes, 3 of them high:

| Cell | Severity | Finding | Proof |
|---|---|---|---|
| 3 | High | `duration` is a known leakage column | `LEAK-bank_marketing`: including it inflated ROC-AUC by +0.18 |
| 3 | High | Target-mean feature computed on all rows | `PIT-004`: target encoding before the split, up to +0.18 |
| 4 | High | Oversampling before the split | `PIT-003`: +0.17 to +0.30 ROC-AUC inflation |
| 5 | Medium | Scaler fitted before the split | `PIT-001`: tiny for scaling, the habit matters |
| 6 | Medium | Test set used to choose between models | `PIT-006`: selection on the holdout overstates the score |
| 6 | Medium | Accuracy is the only metric | Rules R18, R14 |

[`examples/clean_bank_marketing.ipynb`](../../dclab_rnd/copilot/examples/clean_bank_marketing.ipynb) asks the same question carefully and gets **no** notes. Its honest result is ROC-AUC 0.804 and average precision 0.458. The gap between "96.7% accuracy" and that honest number is what the copilot exists to catch.

On this repository's own notebooks, the copilot found a real problem in `notebooks/part1_hyper_ack_classification.ipynb`: it loads HyperAck data and never drops `final_customer_fare` and `final_biker_fare`, the post-outcome columns behind the historical 0.979 "unsafe" ceiling.

## Detectors

| Detector | Severity | Evidence behind the severity |
|---|---|---|
| Known leakage column used | High | Leakage precedents (bank `duration` +0.18, HyperAck final fares) |
| Blocked columns never dropped (data-aware) | High | Dataset decision-time contracts |
| Oversampling before split | High | PIT-003 |
| Target-mean or group aggregate before split | High / Medium | PIT-004 |
| Feature selection before split | High | PIT-002 (+0.35 on wide noise data) |
| Transformer fitted on test data | High | Rules R03, R17 |
| Preprocessing before split | Medium (High for target encoders) | PIT-001 |
| Hyperparameter search before any holdout | Medium | PIT-006 |
| Test set reused to choose models | Medium | PIT-006 |
| Accuracy as the only metric | Medium | Rules R18, R14 |
| Random split on data with timestamps | Medium | PIT-005 (measure the gap; it can be zero) |
| One model family only | Low | `FINDING-model-families` |
| Missing `random_state` | Low | Rule R16 |
| Absolute data path | Low | Rule R21 |
| Imputation statistic from all rows | Low | PIT-001 |
| Split not stratified | Note | Rule R02 |
| Suspicious column name | Note | Rule R05 (a name is not proof) |

Severity follows measured cost. Scaling before the split moved ROC-AUC by about 0.0001 in PIT-001, so it is medium and the note says so. Oversampling before the split moved it by up to 0.30, so it is high.

## What it cannot see

- Columns that are never named in code. The data-aware check covers datasets the R&D already knows; for new data, use `audit_columns` and ask for the decision-time contract.
- Intent. A flagged pattern can be correct on purpose; the note asks a human to decide.
- Runtime values. Nothing is executed.

## How it fits the DCLab notebook

The HTML view is the first version of the "dclab notebook" surface: code cells on the left, evidence-backed notes pinned beside the flagged lines on the right, a severity filter, and a findings index at the top. In the product, the same `review_code` tool runs on each cell as it is edited, and a chat panel answers "why?" by calling `get_record` on the proof IDs. Both surfaces use one retrieval path, so the inline note and the chat answer can never disagree.

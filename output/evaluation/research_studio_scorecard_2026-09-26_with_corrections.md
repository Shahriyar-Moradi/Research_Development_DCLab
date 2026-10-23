# DCLab Research Studio evaluation scorecard

This is a read-only audit of archived development research. It does not establish live agent decision quality or production performance.

- Research cycles completed: **3/6 (50.0%)**; 95% Wilson interval 18.8%–81.2%.
- Saved trials independently audited: **12**; fully passing: **7**.
- Evidence issues: **10**. These are reported below; no original trial files were edited.
- Agent decision accuracy: **not yet measured on a blinded benchmark**.
- Data-scientist task success: **not yet measured with users**.
- Independent model confirmation: **not established by this audit**.
- Overall release gate: **not passed**.

## Findings

- **repeat_summary_missing (4)**: The pooled repeat score is absent, though saved predictions allow reconstruction.
- **repeat_summary_uses_fold_mean (6)**: The field contains a fold average rather than the pooled out-of-fold score.

## Run inventory

| Run ID | Status | Saved trials | Issues |
|---|---|---:|---:|
| `4e4b2bb0bd3045229bab58c0be27cdad` | failed | 0 | 0 |
| `6141d054c4764b698c92909a454362d8` | budget_exhausted | 3 | 0 |
| `72a082af14c94cdb85abbb48cef951d6` | completed | 4 | 0 |
| `cde300a2645f499ab5745825f213614d` | failed | 0 | 0 |
| `d311bd62ad644b03ac98f1dab0b5a489` | completed | 3 | 4 |
| `eb59895d248c4c909bc731127ca05370` | completed | 3 | 6 |

## Reconstructed repeat scores

Computed directly from all saved held-out predictions for each repeat. Values are development evidence and do not replace the original trial records.

| Trial | Repeat | Reason | ROC-AUC | AP | Log loss | Brier |
|---|---:|---|---:|---:|---:|---:|
| `d311bd62ad644b03ac98f1dab0b5a489:trial-001` | 0 | missing | 0.7019 | 0.3110 | 0.2990 | 0.0805 |
| `d311bd62ad644b03ac98f1dab0b5a489:trial-001` | 1 | missing | 0.6883 | 0.3146 | 0.3010 | 0.0804 |
| `d311bd62ad644b03ac98f1dab0b5a489:trial-002` | 0 | missing | 0.6962 | 0.3335 | 0.2913 | 0.0795 |
| `d311bd62ad644b03ac98f1dab0b5a489:trial-002` | 1 | missing | 0.6868 | 0.3252 | 0.2935 | 0.0799 |
| `eb59895d248c4c909bc731127ca05370:trial-001` | 0 | fold_mean_labeled_as_pooled | 0.7019 | 0.3110 | 0.2990 | 0.0805 |
| `eb59895d248c4c909bc731127ca05370:trial-001` | 1 | fold_mean_labeled_as_pooled | 0.6883 | 0.3146 | 0.3010 | 0.0804 |
| `eb59895d248c4c909bc731127ca05370:trial-002` | 0 | fold_mean_labeled_as_pooled | 0.6962 | 0.3335 | 0.2913 | 0.0795 |
| `eb59895d248c4c909bc731127ca05370:trial-002` | 1 | fold_mean_labeled_as_pooled | 0.6868 | 0.3252 | 0.2935 | 0.0799 |
| `eb59895d248c4c909bc731127ca05370:trial-003` | 0 | fold_mean_labeled_as_pooled | 0.7092 | 0.3142 | 0.2974 | 0.0802 |
| `eb59895d248c4c909bc731127ca05370:trial-003` | 1 | fold_mean_labeled_as_pooled | 0.6976 | 0.3162 | 0.2991 | 0.0804 |

## Decision

Keep the original evidence immutable. Review the reconstructed metrics above and any agent claims that cited the older summaries. Then evaluate the five agents on unseen, expert-labelled workflow cases and compare complete task outcomes with a fixed baseline.

Method, proposed thresholds, and limitations: `docs/SYSTEM_EVALUATION.md`.

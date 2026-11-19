# Five Improvements and Five Gaps

**19 November 2026 · Repository review and implementation handoff · Proposed**

These findings concern five places where a user relies on DCLab to preserve what actually happened: which model was approved, what alerted, which data entered, when training stopped, and what can be recovered. The findings come from reading the implementation. The regression scenarios below are specifications for verifying future fixes.

[Implementation steps and acceptance tests](IMPLEMENTATION_PLAN.md) · [Coding prompt](PROMPT.md) · [Separate agentic and context plan](../Agentic_Plan_and_Context_Plan_2026-11-20/PLAN.md)

## Five gaps in current behavior

| ID | Behavior observed in the checkout | Why it matters |
|---|---|---|
| G1 · Approved service export | After confirming final approval, the exporter prepares the data, fits the selected recipe and model again, and compares a newly computed holdout score with the recorded score. It also derives a monitoring reference from test labels and predictions. [Export implementation](../../dclab_rnd/studio/service.py) | Export needs access to test data again. Matching an aggregate score cannot establish the identity of the fitted artifact. A user can lose the ability to deliver a previously approved result if the refit environment changes. |
| G2 · Model-version alert history | Monitoring filters prediction records by model and version. Its alert-history writer deduplicates on window and alert text even though each stored row includes model and version. [Alert writer](../../dclab_rnd/studio/service_template/serve_log.py) | Two model versions sharing a log directory can raise the same warning on the same day. The later version's live report can show the warning while its historical alert row is missing. Concurrent writers can also decide the same alert is new. |
| G3 · Split coverage at import | The Hugging Face connector selects the first matching file, downloads it and records a source note when several files matched. The converted-data path also recognizes partial splits. [Connector](../../dclab_rnd/connectors/hf.py) | A first shard can omit a rare class, a time period or an entire group. The recorded note tells the user after the selection. A user intending to model the whole split needs a way to choose it beforehand. |
| G4 · Stop during a long training stage | The stage-job handler checks for cancellation before each stage. Once execution enters a stage, this handler allows the stage to finish before reaching its next checkpoint. [Stage-job handler](../../dclab_rnd/jobs/handlers.py) | A user who presses Stop during a long model search can wait through the remaining candidates in that stage. The user needs an accurate status and a safe boundary at which the job will yield. |
| G5 · Database and file consistency in backups | Row counts and the database dump share one PostgreSQL snapshot. The workspace files are archived afterward. The manifest records vanished files and archive-level checksums. Restore compares database counts and total file count. [Backup and restore](../../dclab_rnd/storage/backup.py) | Under concurrent writes, valid archives can contain a database row and the wrong file generation. Counts and archive checksums can pass. The code already documents stopping the app and workers for an exact pair. |

## Five improvements to implement

| ID | Change | What success looks like |
|---|---|---|
| I1 · Export the fitted, approved artifact | Preserve the fitted estimator, serving transforms and required aggregate metadata at the original final run. Bind a verified artifact to the exact approved run. | Export succeeds twice from that artifact with training and holdout readers disabled. Both services match the recorded fixture predictions. |
| I2 · Keep each version's warnings | Make alert identity include model, version, monitoring window and alert. Use one cross-process operation to check and append. | Two versions keep one alert each. Retrying or concurrently polling the same version keeps one historical row with its original first-seen time. |
| I3 · Offer first-shard preview and full-split import | Show the resolved source, split, coverage, expected size if known and an explicit choice before download. Save what was actually imported. | The preview is clearly labeled. Full import includes a class present only in shard two. An incomplete or over-budget download cannot be reported as full. |
| I4 · Stop at a safe training boundary | Wire cooperative cancellation into the training candidate loop and display pending stop until the worker acknowledges it. | A request during one candidate prevents the next candidate from starting, preserves the last completed result and publishes no partial winner. |
| I5 · Verify that a backup can restore one consistent state | Define a consistency mode, prove the database/files boundary, index individual files, and exercise restore into isolated empty targets. | A restore drill verifies content and database-to-file links. A changed or missing file produces a precise failure before the backup is called verified. |

**Suggested order.** Start with I2 for a small, visible fix. Give I1 the highest scientific priority and start it next. Follow with I3 for data coverage, I4 for control over running work, and I5 for recovery. The five are independently deliverable. Each needs its own acceptance evidence and rollback decision.

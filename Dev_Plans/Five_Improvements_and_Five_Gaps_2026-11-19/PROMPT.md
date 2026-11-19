# Coding prompt for Five Improvements and Five Gaps

Use the text below in a coding session at the root of DCLab R&D. Set the improvement ID when starting a new session. The default first task is **I2, version-aware alert history**.

---

Implement the selected improvement from the November 19 repository review. Read the root AGENTS.md, the selected G/I pair in the [review](PLAN.md), and the matching section of the [implementation plan](IMPLEMENTATION_PLAN.md). Start with the first unimplemented slice of that improvement. Keep the work independent of the November 20 agent plan.

1. Confirm the starting behavior in the actual source. Trace the nearest product entry point and the existing test that exercises it. Use an invented minimal case to expose the observable discrepancy. If the issue was already fixed in this checkout, verify the acceptance case before doing any further work.
2. For the default task, construct two service versions in the same temporary log directory. Give both the same window and warning, repeat the second report, and read back the alert history. The expected result is one row per version, a stable first-seen time and one entry for simultaneous writers of the same identity. Scope the fix to historical deduplication and its atomic write; keep the live monitor's response compatible. Verify that a new service export ships the corrected runtime template.
3. For another selected task, implement its listed slices in dependency order. Keep each slice small enough for one review. Preserve approvals, model identity, provenance, cancellation ownership and restore safeguards where they apply. Exercise the negative path that makes the fix meaningful: no holdout access for I1, no partially published full split for I3, no next candidate after Stop for I4, and no verified recovery claim on a mismatched snapshot for I5.
4. Follow repository storage, schema, route, UI and generated-file conventions. Preserve legacy readable data and unrelated working-tree changes. Check a real application caller in addition to helper tests. For connector cases, fake the remote provider. For backup work use disposable empty targets. Use no paid model calls or sealed data for this work.
5. Run only affected tests under the owner's test-scope rules. For example start I2 with the focused alert case, then the relevant export-template checks in `tests/test_service_export.py`. Use `.venv/bin/python tests/run_parallel.py test_service_export` when that file is the affected regression group. If a dependency is missing, identify which assertion remains unverified.
6. Close with the selected ID and slice, exact behavioral change, changed files, commands and results, compatibility or upgrade step, rollback and remaining work. Distinguish code completed from behavior verified. Continue with the next slice of the selected improvement when authorized for the whole improvement. After I2, the suggested next improvement is I1.

The acceptance evidence should let a new engineer repeat the original case and see the difference. A documentation update or a test of a helper alone does not complete any of the five improvements.

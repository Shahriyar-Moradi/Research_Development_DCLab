# Notebook assistant

**Status: notebook copilot prototype exists; VS Code inline integration is proposed.** The goal is to help ML engineers where they work: point out possible issues near a code cell, explain evidence, and suggest a next check. The current tracked copilot can analyze a notebook and render a review; it is not yet an editor extension that continuously comments beside the active VS Code cell.

## Questions to answer

- Are comments correct for the specific cell and notebook context?
- Does it identify split leakage, fitted preprocessing outside folds, unsafe features, and evaluation mistakes?
- Does each recommendation cite applicable evidence and state uncertainty?
- Does inline guidance improve task success or time without distracting or overconfident advice?

## Existing files

- Copilot implementation: [`../../dclab_rnd/copilot/`](../../dclab_rnd/copilot/)
- Usage examples: [`../../dclab_rnd/copilot/examples/`](../../dclab_rnd/copilot/examples/) and [`../../notebooks/`](../../notebooks/)
- Product context: [`../../docs/DCLAB_MASTER_CONTEXT.md`](../../docs/DCLAB_MASTER_CONTEXT.md)

## Next evaluation

Connect the existing read-only copilot to VS Code notebook cells. Then evaluate it on blinded notebooks from different projects. Have ML reviewers label the issue, severity, evidence match, and recommended action per cell. Measure precision of actionable warnings, missed critical issues, false-alarm burden, evidence citation accuracy, and engineer task completion. Test unsaved notebook content and large notebooks; keep behavior read-only until edits have separate approval and rollback design.

# Notebook assistant

**Status: proposed extension.** The goal is to help ML engineers where they work: point out possible issues near a code cell, explain evidence, and suggest a next check. A VS Code companion prototype exists in local work but is not included in this main-branch snapshot.

## Questions to answer

- Are comments correct for the specific cell and notebook context?
- Does it identify split leakage, fitted preprocessing outside folds, unsafe features, and evaluation mistakes?
- Does each recommendation cite applicable evidence and state uncertainty?
- Does inline guidance improve task success or time without distracting or overconfident advice?

## Existing files

- Existing notebook examples: [`../../notebooks/`](../../notebooks/) and [`../../external_projects/`](../../external_projects/)
- Product context: [`../../docs/DCLAB_MASTER_CONTEXT.md`](../../docs/DCLAB_MASTER_CONTEXT.md)

## Next evaluation

Implement the smallest read-only VS Code extension first. Then expand to blinded notebooks from different projects. Have ML reviewers label the issue, severity, evidence match, and recommended action per cell. Measure precision of actionable warnings, missed critical issues, false-alarm burden, evidence citation accuracy, and engineer task completion. Test unsaved notebook content and large notebooks; keep behavior read-only until edits have separate approval and rollback design.

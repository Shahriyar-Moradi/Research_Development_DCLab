<!-- Extracted from DCLab Hierarchical Planning V2 Additive Capability Pack (2026-11-05). -->
<!-- Mandatory: V1 acceptance gates first; do not replace or bypass deterministic scientific and safety guards. -->

### HP2-03 | Constrained parallel experimental branches

**Objective:** Run independent candidate experiments when the predicted information gain justifies the compute cost.

**Design:**
- Read-only preparation may be shared; branch-specific writes are isolated by worktree/container/artifact namespace.
- Only dependency-ready V1 leaf tasks can fan out. Shared-state mutations and final/holdout work remain serialized and governed by V1.
- Race/conflict detection, per-branch budget, idempotency/reconciliation before retries, and deterministic merge policy.
- Compare all candidates using the same untouched evaluation contract.
- Select a winner only with an independently verified artifact and outcome.

**Acceptance:** Run at least two independent model candidates with isolated artifacts; crash and resume without double-write; preserve siblings on one branch failure; zero cross-branch contamination.

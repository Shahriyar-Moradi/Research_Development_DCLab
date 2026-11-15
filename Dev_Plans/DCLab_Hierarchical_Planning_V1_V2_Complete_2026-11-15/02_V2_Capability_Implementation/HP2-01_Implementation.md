<!-- Extracted from DCLab Hierarchical Planning V2 Additive Capability Pack (2026-11-05). -->
<!-- Mandatory: V1 acceptance gates first; do not replace or bypass deterministic scientific and safety guards. -->

### HP2-01 | Dynamic specialist collaboration

**Objective:** Pick a minimal role set for a problem and coordinate their contributions on a validated V1 task, not create a new orchestrator.

**Design:** Add an `AgentRoleContract` mapping task type/complexity to bounded role skills. Suggested logical roles: Data Examiner, Experiment Designer, ML Implementer, Scientific Reviewer. The same model may perform multiple roles; separate roles need not mean separate paid models.

**Guardrails:** V1 scheduler/dispatcher remains authority. All agents see only scoped context and registry tools. The coordinator cannot grant permissions, alter policies, or claim completion. Cap agents, rounds, tokens and tool calls.

**Acceptance:** Correct role selection on simple/ambiguous/multi-table cases, appropriate single-agent fallback, no routing regression, no unauthorized tool access, lower/equal overhead for simple cases.

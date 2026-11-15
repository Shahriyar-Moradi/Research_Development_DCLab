<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-05 — Independent task verification, recovery and selective invalidation

## Codex prompt

```text
Goal
Make task completion evidence-based and implement bounded, safe
replanning when a task fails or source artifacts change.

Verification
- Read task's acceptance criteria, evidence records produced by
  actual existing tools, graph/plan fingerprints, source data/solution
  hashes, project approvals and holdout usage.
- Decide verified / failed / blocked / stale, with machine-readable
  reason, evidence links, and violated criteria. Do not accept an
  agent's 'done' statement as verification.
- Separate the verifier from the planner/executor. Start with
  deterministic checks; add an advisory model reviewer only if it
  helps on measured cases and cannot override deterministic failures.
- Record each result as an existing experience/ledger event. Create a
  standalone versioned TaskVerification contract only if required for
  interoperability and after updating schema/wheel tests.

Replanning
- Classify transient failure, invalid tool arguments, changed input,
  missing capability, policy refusal, missing human approval,
  irreversible action ambiguity and domain-science failure.
- For transient failures, reconcile side effects before retry.
- For changed inputs, invalidate only downstream nodes whose
  dependency fingerprints are stale; preserve independent verified
  tasks, human decisions and immutable prior results.
- For a policy refusal, missing capability or consumed holdout, stop
  with an actionable blocker. Do not 'reason around' a guard.
- New plan revision must pass the same deterministic validator.
- Limit retries and replan attempts; report exhaustion transparently.
- Updating the objective or mandatory safety steps needs an explicit
  new confirmed solution revision, not an automatic task rewrite.

Fault-injection tests
- Tool returns error before any write, timeout after successful write,
  restart mid-task, replan cycle, changed source table, changed schema,
  changed solution, approval revoked, blocked related table leakage,
  already-consumed holdout, duplicate final task, and independent
  sibling task surviving another sibling's failure.
- In every case verify no unauthorized write, no additional holdout
  read, no repeated valid work and honest status/evidence.

Deliver implementation, persistence tests, recovery timeline and
failing-before/passing-after test evidence. Maintain the old fixed
workflow as unchanged fallback.
```

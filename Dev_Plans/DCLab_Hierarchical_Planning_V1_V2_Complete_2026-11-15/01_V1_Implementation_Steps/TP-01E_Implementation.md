<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-01E — Cross-environment tests, regression gate and TP-01 handoff

## Codex prompt

```text
Goal
Close TP-01 as a verifiable architecture-and-contracts milestone.
Make no architectural expansion beyond fixes required by its gates.

Review files
- New TP-01 contracts, schemas, validator, adapter and tests.
- tests/test_core_contracts.py, tests/test_core_wheel.py,
  scripts/core_wheel_check.py, Makefile, requirements/base.txt,
  requirements/ci.txt, requirements/agent.lock.txt.
- tests/test_solution_workflow.py, tests/test_programme.py,
  tests/test_orchestra_solution.py, tests/test_graph.py,
  tests/test_agent_harness.py, tests/test_intern.py.

Check and repair
1. The core package imports independently and the installed wheel
   exposes the new contract. No accidental dependency from the core
   onto ML runtime, database, Web UI or agent modules.
2. Importing the agent harness in the lean agent virtual environment
   does not accidentally require NumPy/scikit-learn. The 2026-11-04
   repo had an import regression from a top-level dclab_core import
   into agent_metrics; explicitly regression-test the separation.
3. Existing generated schemas and changelog are current.
4. Backwards-compatible reading of existing project, programme,
   workflow and experience artifacts is preserved.
5. File-based and PostgreSQL tests use actual isolated test storage.
6. The adapter is read-only. A test spies on registry handlers,
   project writes, holdout reads and model gateway calls, expecting 0.
7. Test an actual binary workflow fixture, an invalid workflow,
   a missing capability and a changed input fingerprint.
8. Existing workflow, programme, home and intern tests remain green.
9. Explicitly record tests skipped because the local environment is
   not provisioned. Do not claim them passed.

Commands
- Use the repo's provided Python environments and the exact
  focused test invocation accepted by the current test suite.
- Run make core-schemas / core-wheel-check if targets exist at HEAD.
- Run make check-quick and, if feasible, make rd-check.
- Run make test-pg and make check-all only when PostgreSQL and the
  documented environments are present, no destructive setup.
- No live model, no sealed benchmark, no external API.

Write docs/guides/TASK_PLANNING_TP01_REPORT.md with
- Architecture mapping links and ownership matrix.
- Full coverage of the original seven-item TP-01 checklist.
- Which paths are physically executable vs implicit, guard or blocked.
- New contract versions and schema names.
- Test commands and their exact pass/fail/skip statuses.
- Known limits and acceptance gate pass/fail.
- Decision whether TP-02 may begin.

Do not proceed into TP-02 in this package. A failure or unresolved
ownership conflict must be reported as a blocker, not hidden.
```

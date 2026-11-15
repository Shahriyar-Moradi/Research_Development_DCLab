<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-07 — Product task visibility, opt-in rollout and hardening

## Codex prompt

```text
Goal
Expose meaningful task progress in DCLab's existing white/light
Home and project experience, then enable the planner only after
its TP-06 gates pass. Preserve the six intent buttons exactly.

Rules
- Model/Plan/Report/Analyze/Prepare/Learn remain intent selections.
  'Plan' continues mapping to business intent. The new execution
  planner appears as a project-specific TASK PLAN and progress panel,
  not a new seventh intent or a renamed existing one.
- Use the existing web/src build process and design tokens. Never
  hand-edit generated static/app outputs without rebuilding sources.
- Use existing session/task board conventions where possible.
- No visual overhaul, no dark-mode-only UI and no second product.

User experience
- Read-only tree of task groups and leaf tasks, ordered dependencies,
  completed/running/blocked/stale status, concise reason, evidence
  link, source workflow step, selected capability, plan revision,
  approved/awaiting approval indicator and current budget usage.
- Make it clear when a task is a guard/check instead of tool work.
- Show why a blocked task requires data, permission or a person.
- Show a plan revision diff and which valid tasks were preserved.
- Do not display raw private rows, secrets or model hidden reasoning.
- Correct accessibility labels, keyboard navigation, screen reader
  status, responsive narrow-width layout, loading/error/empty states.

Backend and rollout
- Reuse existing project API/service and task event store. Add only
  minimal read-only endpoints under appropriate role/owner guards.
- Use current feature flag/settings conventions. Start disabled.
- If required gates did not pass, provide only developer-preview
  views and no default execution switch.
- Ensure old/legacy sessions render unchanged, and a user who never
  selects a chip still follows current Home routing behavior.
- Add audit trail for a switch, explicit user confirmation before
  any newly introduced write behavior, and straightforward rollback.

Tests
- Unit, API, file/PostgreSQL, browser 1280 and 375px, resume,
  permission, role, no-data and blocked-state tests.
- Browser evidence via existing screenshots and test runner, no
  fabricated screenshots or timings.
- Run make browser-test / product-e2e / check-all when provisioned.

Deliver code, tests, docs, screenshots if actually generated, feature
flag, rollback plan, release note, measured results, and a final
unresolved-risk list. Do not claim all of DCLab is production-ready.
```

---

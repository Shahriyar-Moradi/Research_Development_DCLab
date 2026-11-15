# DCLab Agentic ML External Platform: Complete Engineering Documentation
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## What this package is
An English, implementation-oriented set of documents reflecting the **final three-agent / three-harness reusable-template architecture**, superseding the two-profile notes. It is a proposed plan, **not a claim that the external platform has been built or tested**.

## How to use it
1. Start with `00_MASTER_ROADMAP.md`, then read the architecture and templates documents.
2. Open the applicable `phases/PHASE_*.md` for phase-wide context and numbered work packages.
3. Use the matching `plans/PLAN_*.md` for plan-by-plan execution and checkboxes.
4. Track progress and evidence in `tracker.csv`; review release criteria in `04_SECURITY_GOVERNANCE_AND_EVALUATION.md`.
5. Check end-to-end stories, example manifests, and the proposed API contract before beginning a new adapter.

## Core engineering decisions
- External, independently deployable system; DCLab remains an optional provider via adapters.
- Exactly three specialized reusable harness/template families: Model, Problem, Project.
- One shared infrastructure kernel; entity instances are configuration, not cloned source code.
- Problem Agent owns all solution histories; Project Agent coordinates multiple problems; Model Agent owns model lifecycle.
- Traceable evidence and dataset rights; approvals for risk; no separate persistent Solution Agent.
- New versioned manifests for each additional model/problem/project; new generic adapters only for new capabilities.

## Programme
| Phase | Subplans | Packages | Deliverable |
|---|---:|---:|---|
| 0: Foundation, DCLab Gap Audit and Reference Scenario | 3 | 9 | G0: Demonstrate reproducible churn training and an API mock that resolves the same project, problem, solution, and model IDs. |
| 1: Shared Infrastructure, Template Registry and Agent Factory | 4 | 12 | G1: Instantiate three fake agents from YAML and run read-only typed tools; restart the service and resume a job without losing state. |
| 2: Model Agent Template and Specialized Model Harness | 3 | 10 | G2: Ask a churn Model Agent: "Which data trained v2?", "What is recall for cohort X?", "Did input distributions drift?", "Train a challenger". Each yields traceable evidence or a correct refusal. |
| 3: Problem Agent Template and Specialized Problem Harness | 3 | 10 | G3: Ask: "What churn problem are we solving, what alternatives were rejected, and why?" Then revise the solution and verify prior revisions remain accessible. |
| 4: Project Agent Template and Specialized Project Harness | 3 | 10 | G4: Ask: "What have we tried across churn and retention, what is blocked, and which solution should we prioritize?" Verify source-based result and workload implications. |
| 5: Typed Multi-Agent Collaboration and Workflow Orchestration | 3 | 9 | G5: Ask Project Agent to investigate declining churn-model recall and recommend a solution; inspect decomposed tasks, model diagnostics, business KPI evidence and decision proposal. |
| 6: Event-Driven Monitoring, Investigation and Continuous Lifecycle | 3 | 9 | G6: Inject synthetic feature-drift event and labeled-performance decline; follow event through Model, Problem and Project Agents with no unnecessary retraining. |
| 7: External Dashboard, API Productization, Generalization and Production | 3 | 10 | G7: Onboard 5 model agents, 3 problem agents and 2 project agents using manifests and adapters only; inspect dashboard and run governed model/problem/project workflows. |

## Document inventory
- Master: `00_MASTER_ROADMAP.md`
- Design documents: `01_ARCHITECTURE_AND_BOUNDARIES.md`, `02_TEMPLATE_AND_HARNESS_SPEC.md`, `03_DOMAIN_DATA_AND_API_CONTRACTS.md`, `04_SECURITY_GOVERNANCE_AND_EVALUATION.md`
- Implementation backlog: `05_DELIVERY_BACKLOG.md`, `tracker.csv`
- Architecture decisions: `06_ARCHITECTURE_DECISIONS.md`
- Functional demonstrations: `07_END_TO_END_SCENARIOS.md`
- Dependencies: `08_DEPENDENCIES_AND_SEQUENCE.md`
- Eight phase-level implementation documents under `phases/`
- Standalone subplan documents under `plans/`
- Three manifest examples, OpenAPI draft, task schema and lifecycle-event schema under `examples/`

## Known unverified implementation assumptions
Existing DCLab components were inspected at repository-file level during planning, but the complete local CI suite and adapter compatibility were not executed here. Phase 0 explicitly requires this verification. All planned API routes and manifests are proposals until ratified. No production deployment or agent modifications were made.

**Inventory:** 8 phases · 25 subplans · 79 work packages · 10 architecture decisions.

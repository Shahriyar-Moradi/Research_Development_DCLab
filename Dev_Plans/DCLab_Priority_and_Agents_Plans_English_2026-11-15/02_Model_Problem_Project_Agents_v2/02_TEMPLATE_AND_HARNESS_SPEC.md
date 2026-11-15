# Reusable Agent Templates and Specialized Harnesses

**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## 1. The template principle

Only **three template families** exist in MVP: `model-agent`, `problem-agent`, `project-agent`. Each family has a template manifest schema, a specialized harness, lifecycle configuration, tool capabilities, policy defaults, and context loader. An **agent instance** is a validated assignment of a versioned template to an entity. It is not an automatically generated source-code fork. An existing model can be linked to several problems, but its Model Agent identity is singular within its governing registry scope. Each problem gets its own Problem Agent; each project gets its own Project Agent.

### TemplateSpec (proposed)

```yaml
id: model-agent
version: 1.0.0
schema_version: 1
entity_kind: model_family
harness: model
context_provider: model-context-v1
instruction_pack: model-safety-v1
capability_allowlist:
  - model.read_metadata
  - model.read_lineage
  - model.evaluate
  - model.monitor
  - model.propose_retrain
policy_pack: model-default-v1
resource_limits:
  max_tool_calls: 24
  max_delegation_depth: 2
activation: on_demand
```

An instance manifest is different:

```yaml
instance_id: model-agent-churn-xgb
kind: model
entity_ref: model-family:churn-xgb
workspace_ref: workspace:demo
extends_template: model-agent@1.0.0
adapter_bindings:
  model_registry: dclab-readonly@1
  evaluation: dclab-eval@1
  training: dclab-job@1
capability_overrides:
  disable:
    - model.deploy
policy_overrides:
  retrain: approval_required
  deployment: disabled
```

No manifest can raise permissions beyond tenant policy or dynamically import an executable Python module. Manifests are validated and template versions pinned. A new XGBoost or LightGBM model varies its `entity_ref`, feature contract, backend adapter and policy binding, not its harness source.

## 2. Model Agent Template / Model Harness

**Scope:** one registered model family and its version lineage. **Persistent knowledge:** model card, target, task kind, feature and preprocessing contracts, dataset snapshots, code/config, experiments, evaluations, monitoring windows, deployment alias, approvals and incident history.

**Main processes:** register -> inspect -> evaluate -> diagnose -> monitor -> propose challenger -> train candidate (authorized) -> compare -> propose release -> inspect deployment. Every process has a distinct tool contract, condition checks, evidence and state transition. A model can have many versions without generating a new Model Agent instance for each version.

**Core tools:** `model.get_overview`, `model.list_versions`, `model.get_features`, `model.get_training_sources`, `model.get_evaluation`, `model.evaluate`, `model.inspect_drift`, `model.list_monitoring`, `model.propose_retrain`, `model.submit_retrain`, `model.compare_candidate`, `model.propose_release`.

**Behavioral guarantees:** never invent model scores; no comparison of incompatible windows without a warning; distinguish drift from concept change; require grounded training rights; no automatic production deployment. External model adapters are permitted only by capability class, e.g. generalized sklearn or Torch inference/evaluation, not as a new Agent class for every entity.

## 3. Problem Agent Template / Problem Harness

**Scope:** one business or technical problem, *including* complete solutions and their context. **Persistent knowledge:** problem definition, goals, stakeholders, hypotheses, assumptions, business-data sources, all solution alternatives and revisions, linked models/experiments, conversations, decisions, approved interventions and measured outcomes.

**Main processes:** intake confirmation -> scope clarification -> analysis -> candidate solution creation -> evidence gathering -> solution comparison -> adopted plan -> investigation -> improvement/revision -> resolved or retired. The problem continues to exist when the previously chosen solution changes. Solutions are versioned entities, not a new persistent Solution Agent.

**Core tools:** `problem.get_scope`, `problem.list_hypotheses`, `problem.get_kpis`, `problem.list_data_sources`, `problem.list_solutions`, `problem.compare_solutions`, `problem.get_decisions`, `problem.get_conversations`, `problem.propose_plan`, `problem.discover_models`, `problem.delegate_model_task`, `problem.record_outcome`.

**Behavioral guarantees:** distinguish model quality from business KPI, past decision rationale from inferred rationale, tested solution from proposed idea, and causal effect from observational association. Inactive or missing data produces a blocker rather than a made-up solution success.

## 4. Project Agent Template / Project Harness

**Scope:** one project with one or more Problem Agents, shared resources and potentially shared models. **Persistent knowledge:** project owner/team, objectives, priorities, milestones, budgets, permissions, dependencies, cross-problem decisions, project conversations and major outcomes.

**Main processes:** project setup -> attach problems -> resource/priority planning -> delegated work -> milestone progress -> portfolio risk review -> project changes -> completion/archive. This agent is the preferred top-level user-facing entry point for multi-problem projects, not a replacement for the narrower problem specialist.

**Core tools:** `project.get_overview`, `project.list_problems`, `project.get_priorities`, `project.get_budget`, `project.list_milestones`, `project.get_decisions`, `project.compare_problem_outcomes`, `project.delegate_problem_task`, `project.approve_portfolio_change`, `project.get_status`.

**Behavioral guarantees:** no bypass of child Problem/Model Harness controls, no silent priority/budget change, no access to private child conversations without permission, and no report of "done" unless a verified child execution receipt exists.

## 5. Shared kernel vs per-harness code

| Implementation artifact | Exactly how many? | Why |
|---|---:|---|
| Shared executor, tool registry, approval engine and evidence service | 1 each | Common infrastructure behavior |
| Specialized harness class / state machine | 3 | Distinct process semantics and policy |
| Agent template family | 3 | Reusable configuration/behavior contract |
| Agent instance | N | One identity per model/problem/project |
| Model framework adapter | As supported | General purpose by model family/task type, not per model identity |
| Business data connector | As needed | Database/provider integration, not per Problem Agent |
| Solution record | Many | Versioned domain object governed by Problem Agent |
| Temporary task specialist | Optional | Callable tool/role; no new persistent fourth harness |

## 6. Template compatibility and upgrades

- A pinned `template_version` is required on every agent instance and every run receipt.
- Template upgrades require contract tests; breaking schema changes require explicit migration and a rollback plan.
- Existing instances must never silently adopt new tool permissions.
- Resource owner can suspend an instance while preserving its history and relations.
- A template cannot expose a capability without a registered tool schema and authorization policy.
- Customization points are configuration, typed providers, backend adapters, tool allowlist narrowing, context bindings, thresholds and policies. Arbitrary code execution from customer manifests is **not** a customization point.

## 7. Core acceptance experiment

Use manifests to create **five Model Agents, three Problem Agents, two Project Agents**. Link them to distinct data, contexts, versions and policies. Execute inspect, evaluate, problem-solution comparison and project-delegation scenarios. Then inspect the implementation diff: there must be no new ModelHarness/ProblemHarness/ProjectHarness classes or hand-coded per-entity agent classes. Generic reusable adapters may be added if an unsupported algorithm kind genuinely requires one, but such adapters must be reusable across multiple instances.

## 8. Why no separate Solution Agent now

Problem Agent owns solution development, comparison and history because these tasks are inseparable from the problem's objective and available evidence. Creating an independent fourth autonomous actor would add more routing, state duplication, inconsistent priorities and cost. A `SolutionWorkflow` can have temporary planners/critics, but remains a tool/workflow until evaluation demonstrates a distinct persistent owner is justified.

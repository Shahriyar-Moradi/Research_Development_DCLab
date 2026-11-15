# Domain, Data, Memory and API Contracts

**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## 1. Canonical entities

| Record | Mandatory identifiers and fields | Ownership and invariants |
|---|---|---|
| `Workspace` | `workspace_id`, owner, grants, policy_ref | Top-level tenant security boundary |
| `Project` | `project_id`, workspace_id, title, status, current_revision | One Project Agent, multiple problems |
| `Problem` | `problem_id`, project_id, objective, KPI refs, revision, status | One Problem Agent; solutions remain when objective revised |
| `Solution` | `solution_id`, problem_id, revision, state, evidence_refs, decision_refs | Many alternatives; one or multiple approved approaches if policy permits |
| `ModelFamily` | `model_family_id`, workspace_id, task_kind, registry_ref, owner | Stable Model Agent identity; versions separate |
| `ModelVersion` | `model_version_id`, model_family_id, immutable artifacts, source_data_ref, feature_contract_ref | Scores belong to exact version and evaluation window |
| `AgentTemplate` | template kind, semantic version, harness kind, capability IDs, schema hash | Immutable published template revision |
| `AgentInstance` | agent_id, template_ref, entity_ref, workspace_id, status | Exactly one target entity and one pinned template |
| `ConversationEvent` | conversation_id, event_id, actor, time, entity_ref, content_ref, claim_type | History immutable; summaries derivative |
| `DecisionRecord` | decision_id, alternatives, rationale, actor, evidence_refs, time, supersedes | No fabricated rationale or mutable past |
| `EvaluationRun` | model_version, dataset snapshot, label window, metric spec, threshold, results, evidence | Compute code owns metrics; compare only compatible conditions |
| `AgentTask` | task_id, target, capability, refs, constraints, status, job/approval refs | No free-form text is an executable command |
| `Investigation` | investigation_id, trigger_event, owning_agent, hypothesis/step refs, state | Repeat triggers can correlate to same investigation |
| `OutcomeObservation` | intervention_ref, cohort/window, measurement evidence, limitations | Business result distinct from offline metric |

## 2. Agent creation and run request

```json
{
  "template": "problem-agent@1.0.0",
  "entity_ref": "problem:churn-retention",
  "workspace_ref": "workspace:demo",
  "bindings": {
    "problem_data": "catalog:churn-demo",
    "kpi_registry": "kpi:retention-v1"
  },
  "policy_ref": "problem-default-v1"
}
```

Server validates all refs, principal/tenant relationship, pinned template and capability set. If the same tenant+entity+template active identity already exists, an idempotent create resolves it rather than generating duplicates.

### RunRequest (illustrative)

```json
{
  "agent_id": "problem-agent-churn",
  "conversation_id": "conv-1007",
  "message": "Why did we reject solution 2?",
  "scope": {"problem_id": "churn-retention"},
  "constraints": {"max_tool_calls": 12, "read_only": true},
  "idempotency_key": "demo-why-solution-2-1"
}
```

### TaskRequest (illustrative)

```json
{
  "schema_version": 1,
  "task_id": "task-2026-001",
  "correlation_id": "investigation-210",
  "source_agent_id": "problem-agent-churn",
  "target_agent_id": "model-agent-churn-xgb",
  "capability": "model.get_evaluation",
  "arguments": {"model_version_id": "mv-002", "evaluation_ref": "eval-week45"},
  "evidence_inputs": ["evidence:monitor-week45"],
  "execution": {"mode": "read_only", "max_steps": 6},
  "required_output_schema": "model-evaluation-response@1"
}
```

Response must have `status`, typed `result`, `evidence_refs`, `warnings`, `run_id`, tool receipt IDs and optional `requires_approval`; no runtime can treat a free-text summary as proof of execution.

## 3. External REST API v1 contract

| Resource | Methods and route | Notes |
|---|---|---|
| Templates | `GET /api/v1/templates` and `GET /api/v1/templates/{id}` | Read template/version/available capabilities |
| Agents | `POST /api/v1/agents`, `GET /api/v1/agents/{id}` | Factory identity, validated manifest and status |
| Agent runs | `POST /api/v1/agents/{id}/runs`, `GET /api/v1/runs/{run_id}` | May return job ID for long work |
| Projects | `POST /api/v1/projects`, `GET /api/v1/projects/{id}` | Project root and assigned Project Agent |
| Problems | `POST /api/v1/projects/{id}/problems`, `GET /api/v1/problems/{id}` | Multiple problems per project |
| Solutions | `POST /api/v1/problems/{id}/solutions`, `GET /api/v1/problems/{id}/solutions` | Draft/revision and adoption policy |
| Models | `POST /api/v1/models/import`, `GET /api/v1/models/{id}` | Read-safe registry/lineage adapter |
| Tasks | `POST /api/v1/tasks`, `GET /api/v1/tasks/{id}`, `POST /api/v1/tasks/{id}/cancel` | Typed orchestration, scoped permissions |
| Evaluations | `POST /api/v1/models/{id}/evaluations`, `GET /api/v1/evaluations/{id}` | Label/window/evidence validation |
| Investigations | `GET /api/v1/investigations/{id}` | Event and findings history |
| Approvals | `GET /api/v1/approvals/{id}`, `POST /api/v1/approvals/{id}/decision` | Server-side actor-specific validation |
| Evidence | `GET /api/v1/evidence/{id}` | Access controlled with provenance |
| Events | `POST /api/v1/events` | Authenticated sources, dedupe and schema version |

These routes are **proposed**, not implemented API claims. API contract tests and OpenAPI versioning belong to P1-12 and P7-02. Read endpoints must never expose signed URLs without checking tenant and subject scopes.

## 4. Error and status normalization

Use stable error codes: `INVALID_SCHEMA`, `NOT_FOUND`, `FORBIDDEN`, `POLICY_DENIED`, `APPROVAL_REQUIRED`, `STALE_CONTEXT`, `INCOMPATIBLE_VERSION`, `INSUFFICIENT_LABELS`, `INCOMPARABLE_WINDOW`, `PROVIDER_UNAVAILABLE`, `CAPABILITY_UNSUPPORTED`, `BUDGET_EXCEEDED`, `CONFLICT`, `RETRY_UNSAFE`. Always return human-readable reason, retryable flag and correlation ID. Do not transform provider failure into assistant-sounding certainty.

## 5. Data and provenance contract

Every dataset contains `source_id`, `snapshot_id`, `license_or_permission_ref`, `raw_hash`, `schema_hash`, `label_source`, `prediction_time`, `available_time`, `owner`, and `retention_policy`. Every evaluation records `model_version`, `dataset_version`, `row_selection`, `label_status`, `window`, `cohort`, `threshold`, `code_revision`, `metric_definition`, `value`, `interval` and evidence. Sensitive rows should stay at source; agent context contains only authorized summaries/artifact references.

## 6. Memory design

Authoritative store (PostgreSQL + provider artifact IDs) owns current truth. Conversation store owns immutable utterances and source events. Retrieval index points to source fragments and related entities; vector embeddings, if enabled, are an optional search accelerator. A ContextBundle is reconstructed per run using grant filtering, freshness and explicit version precedence. Handle disagreements by surfacing conflict and retrieving latest signed revisions. Do not let a retrieval summary overwrite a model registry or approval status.

## 7. Versioning, idempotency and concurrency

- API version in path; Template and Event schemas have independent semantic versions.
- Agent identities and durable jobs support `Idempotency-Key` on creates and stateful actions.
- Solution decisions use optimistic revision numbers; callers pass `If-Match` or explicit expected version.
- Every effectful tool checks authorization on actual call, not just when plan was built.
- Evidence/metric artifacts are append-only. Corrections create a new record that supersedes old one.
- Audit and tracer IDs are generated server-side and correlated to user-visible jobs.

## 8. Example packages

The `examples/` folder contains three agent instance manifests and a proposed OpenAPI skeleton. They are implementation references; adjust provider bindings after Phase 0 confirms actual DCLab interface shape.

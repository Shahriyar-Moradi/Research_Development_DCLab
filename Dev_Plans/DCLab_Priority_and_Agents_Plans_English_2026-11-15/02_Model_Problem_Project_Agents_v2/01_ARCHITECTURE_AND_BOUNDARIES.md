# Technical Architecture, Boundaries and Deployment

**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## 1. Scope and deployment separation

Deploy the external platform as its own API, worker and UI, with its own PostgreSQL and authentication, and optional adapter access to DCLab's established scientific engine. DCLab remains free to develop its internal agentic notebook, Case Agent and scientific pipeline independently. The external product cannot reach directly into DCLab's local database, internal memory files or private Python modules across processes. It must use documented APIs, an explicit local bridge or an import/export contract. Integration failure is treated as a provider outage, not as a complete platform failure.

**Recommended logical packages:**

```
external_agentic_ml/
  domain/         # Model, Problem, Project, Solution, evidence and decision schemas
  templates/      # three versioned TemplateSpecs and example manifests
  agents/         # AgentFactory, AgentInstance identity, context assembly
  harnesses/
    core/         # shared runner, tool registry, jobs, policies, tracing
    model/        # model lifecycle, evaluation, diagnostics, retrain proposal
    problem/      # problem lifecycle, solution portfolio, hypotheses, KPIs
    project/      # portfolio, priorities, cross-problem tasks, resources
  adapters/       # DCLab, MLflow, data connectors, serving, event sources
  api/v1/         # FastAPI public routes and DTOs
  workers/        # durable jobs and investigation event handlers
  memory/         # structured source index, conversation history, optional vector retrieval
  ui/             # independently deployed dashboard
  migrations/     # external PostgreSQL schema
  tests/          # contract, isolation, science, integration, sealed evaluation
```

## 2. Why three harnesses and one kernel are not contradictory

The **kernel** executes bounded tool calls, creates work receipts, validates schemas, checks budgets, enforces approvals and records traces. It does not decide scientific measures or business lifecycle rules. The **three harnesses** own different process state machines and authorized capabilities. Each harness composes the shared kernel and can change its domain logic without affecting other harnesses. A new Project is an *instance* of the Project template, not a new ProjectHarness class; similarly for Problem and Model.

### Ownership matrix

| Capability | Kernel | Model Harness | Problem Harness | Project Harness |
|---|---|---|---|---|
| Tool execution, typed validation, timeouts | Owner | Consumer | Consumer | Consumer |
| Principal/tenant authorization, audit | Owner | Adds model policy | Adds problem policy | Adds project policy |
| Context retrieval and memory transport | Owner | Model lineage queries | Solution and hypothesis context | Portfolio and meeting context |
| Model evaluation, feature drift, retraining | Infrastructure only | Owner | Requests via typed delegation | Requests through child agent |
| Solution alternatives and problem decisions | Persistence and evidence | Source of metric evidence | Owner | Oversees priority and cross-problem impact |
| Project resources, priority, milestones | State storage | N/A | Requests budget | Owner |
| Monitoring event ingestion | Transport | Diagnoses model events | Diagnoses problem impact | Escalates portfolio risk |
| User conversation | Shared run protocol | Model-specific reasoning | Problem-specific reasoning | Cross-problem synthesis |

## 3. Core interface design

```python
from typing import Protocol

class Harness(Protocol):
    kind: str  # model | problem | project
    def bind_context(self, entity_id: str, actor_id: str) -> dict: ...
    def capabilities(self, entity_id: str, actor_id: str) -> list[str]: ...
    def validate_transition(self, state: str, requested_action: str) -> None: ...
    async def handle(self, request: dict, context: dict) -> dict: ...
    def summarize(self, result: dict) -> dict: ...

class ProviderAdapter(Protocol):
    name: str
    async def capabilities(self, principal: dict) -> list[str]: ...
    async def invoke(self, capability: str, args: dict, context: dict) -> dict: ...

class AgentFactory(Protocol):
    def instantiate(self, template_ref: str, entity_ref: str, manifest: dict, actor_id: str) -> str: ...
```

These are **illustrative contracts**, not implemented interfaces. Real implementations require versioned Pydantic DTOs, async-safe storage, resource-scoped grants and DTO compatibility checks. Factory loading must not blindly import arbitrary custom Python paths from manifests.

## 4. Identity, instance, run and process separation

An `AgentInstance` is a long-lived identity bound to exactly one target model family, problem or project and a pinned `TemplateSpec`. It can own many conversations and runs. An `AgentRun` is an invocation with a bounded budget. A `Job` is a durable background operation. A `Harness` is a reusable implementation shared among many instances. A model version or solution revision has its own distinct immutable identifier. The design must not conflate any of these.

## 5. Lifecycle state machines

**Model Harness:** `registered -> documented -> evaluated -> monitored`; from monitored, a *separate* controlled candidate flow: `candidate_proposed -> approved -> training -> evaluated_candidate -> release_proposed -> approved_release -> released/rolled_back`. Reading status should always work even when training tools are unavailable. Direct deployment is not default behavior.

**Problem Harness:** `draft -> clarifying -> active -> under_investigation -> improvement_proposed -> active/resolved/retired`. Its solutions have independent revisions and adoption transitions. A problem persists even after all models have been replaced or a previously adopted approach is rejected.

**Project Harness:** `created -> active -> paused -> completed -> archived`. Project milestone/work plans have their own state and owner. It remains stable as problems join, solutions change, jobs finish and people rotate.

**Agent Instance:** `configured -> active -> suspended/archived`. **Run:** `created -> running -> waiting_approval/completed/failed/cancelled`. **Job:** `queued -> running -> waiting_approval/completed/failed/cancelled`. All state changes are recorded with actor, timestamp and reason; no mutation is inferred just because an LLM said it occurred.

## 6. Context and memory boundaries

Use an explicit **ContextBundle** assembled on each run: `tenant_ref`, `agent_ref`, `entity_ref`, `requested_scope`, `canonical_refs`, `recent_event_refs`, `relevant_conversation_refs`, `retrieval_result_refs`, `permissions_snapshot` and `freshness`. Keep current scores in EvaluationRun, active solution in adopted SolutionRevision, and decision rationale in DecisionRecord. Conversation summaries are only hints to which evidence to read. A memory item can be stale or wrong, so use the evidence resolver and compare timestamps and revision IDs. A vector index, when useful, is an auxiliary retrieval index, not a factual database.

## 7. Integrating DCLab safely

| External capability | Candidate existing source | Adapter behavior |
|---|---|---|
| Training/evaluation science | `dclab_core/model_evaluation.py`, Studio evaluator | Issue job through documented service layer; import immutable EvaluationRun |
| Retraining comparison | `dclab_rnd/studio/retrain.py` | Trigger supported retrain workflow under authorization; preserve provider job IDs |
| Data/feature lineage | `dclab_rnd/studio/lineage.py`, data catalog | Fetch versioned metadata, hashes, feature contract and evidence refs |
| Monitoring signal generation | `dclab_core/monitoring.py` | Import monitoring policy/window and source timestamp |
| Existing task planning contracts | `dclab_core/task_graph.py`, `task_execution_plan.py` | Convert validated plan DTOs; run external authorization again |
| Existing Case Agent intake | `dclab_rnd/agents/case_agent.py` | Import confirmed problem context through sanctioned endpoint/export only |
| Existing jobs / traces | `dclab_rnd/jobs`, `agents/traces.py` | Map status and evidence; do not claim jobs completed based on chat |

These are candidate integration points identified by a repository inspection on 10 November 2026. Phase 0 must verify actual HTTP reachability, permissions, test status and compatibility. Some files are internal Python APIs, not ready-made remote endpoints.

## 8. State, event and request flows

1. **Onboarding:** authenticated create request -> entity repository -> Template Registry validation -> Factory -> persistent agent identity -> default capabilities -> audit receipt.
2. **Read-only question:** user prompt -> resolve entity and version -> scoped context -> harness intent -> allowed tool calls -> canonical evidence -> answer with provenance and explicit uncertainty.
3. **Evaluation:** harness requests type-safe operation -> check labels/window/access -> submit durable job -> deterministic provider compute -> immutable result -> evidence-linked answer.
4. **Retrain:** propose candidate -> data rights/safety/approval -> job and checkpoint -> compare candidate -> separate release proposal. No hidden deployment.
5. **Collaboration:** Project sends typed task to Problem; Problem delegates model-specific diagnostics to Model; each recipient revalidates scope; EvidenceMerger checks incompatible versions/windows before synthesis.
6. **Event:** authenticated monitoring source -> EventPolicyEngine dedupes and thresholds -> Investigation persisted -> targeted agent task(s) -> human decision or controlled improvement -> outcome ledger.

## 9. Data governance and model-source safeguards

Training datasets and any retrieval documents must record provenance and training/reuse rights. In particular, outputs or hidden reasoning from commercial models must not silently become training data. Model Agent descriptions must distinguish observed behavior, system metadata and claims made by third parties. Protect label delay, temporal leakage, idempotency and holdout access.

## 10. Engineering trade-offs

- **Use a single PostgreSQL first, not a premature distributed data platform.** Add vector search only if it materially improves retrieval quality in benchmarks.
- **Use durable workers for long tasks.** Do not make LLM turns poll every few seconds to observe training progress.
- **Use deterministic routing first.** Introduce semantic routing only for genuinely ambiguous user language and evaluate against simpler baselines.
- **Keep separate specialized policy modules.** Shared plumbing does not mean the three lifecycles are the same.
- **No unrestricted SQL from agents.** Use allowed views, parameterized read queries, row caps and reviewable data contracts.
- **Avoid accidental permanent agents.** Instances are metadata and activation policies; workers start only for tasks or events.
- **No template explosion.** Treat a new entity as configuration; add generic model-kind adapters only where a true unsupported capability exists.

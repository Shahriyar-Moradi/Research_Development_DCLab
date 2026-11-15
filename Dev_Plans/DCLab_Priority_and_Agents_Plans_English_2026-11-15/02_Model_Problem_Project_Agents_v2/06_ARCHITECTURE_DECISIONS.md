# Architecture Decision Records
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

These are recommended decisions for sign-off, not evidence of deployed code. Review ADRs during Phase 0 and version amendments rather than silently editing accepted decisions.

## ADR-001: Standalone external product
**Decision:** Run an independently deployable platform with its own database, API, credentials and workers. DCLab is an optional provider, not a runtime requirement.
**Why:** Minimizes changes to DCLab, isolates release risk, enables multiple ML backends.
**Consequence:** Requires provider adapters and network fault management.
**Alternative and boundary:** A cross-repo implementation in the existing DCLab monolith would simplify early imports but defeats independence.
**Proposed status:** Pending ratification at G0.

## ADR-002: Three distinct reusable agent/harness templates
**Decision:** Keep Model, Problem and Project as three specialized harness state machines and three versioned template families on one shared platform kernel.
**Why:** Clear ownership and reusable deployment with domain-specific workflows.
**Consequence:** Requires contracts, template versioning and factory validation.
**Alternative and boundary:** Old two-profile architecture is superseded.
**Proposed status:** Pending ratification at G0.

## ADR-003: No persistent Solution Agent in first release
**Decision:** Solutions are versioned first-class entities and workflows managed by a Problem Agent; Project Agent manages portfolio priority.
**Why:** Avoids redundant ownership, routes and state duplication.
**Consequence:** Problem Harness must support multiple alternatives, implementation and outcomes.
**Alternative and boundary:** Separate Solution Agent may be revisited after measured delegation evidence.
**Proposed status:** Pending ratification at G0.

## ADR-004: Persistent identity, on-demand activation
**Decision:** Keep AgentInstance, context links, policy/template version and history persistent; activate the LLM runtime for requested tasks or eligible events.
**Why:** Controls cost and supports thousands of entities without thousands of workers.
**Consequence:** Background monitoring requires event processors rather than active per-entity LLMs.
**Alternative and boundary:** Always-on LLM agents are excluded.
**Proposed status:** Pending ratification at G0.

## ADR-005: Deterministic science and server-side authorization
**Decision:** Metrics, splits, model comparison, authorization, data/feature transforms and training execution belong to validated code and tool services.
**Why:** Prevents fabricated scores and LLM-gated writes.
**Consequence:** The orchestration and tool adapter layer must enforce checks at execution, including after resume.
**Alternative and boundary:** Prompt-based "do not deploy" safeguards are insufficient.
**Proposed status:** Pending ratification at G0.

## ADR-006: Evidence-first memory with canonical precedence
**Decision:** Authoritative registries and immutable artifacts are the source of truth; conversational and vector retrieval only provide context and references.
**Why:** Allows verifiable historic rationale and resolves conflicting summaries.
**Consequence:** Implement versioned references, freshness, redaction and conflict detection.
**Alternative and boundary:** Do not treat vector store as authoritative transactional registry.
**Proposed status:** Pending ratification at G0.

## ADR-007: Typed task delegation and bounded DAG execution
**Decision:** Delegate Project -> Problem -> Model using typed tasks and dependency-validated plans rather than free-form multi-agent chat.
**Why:** Improves reproducibility, recovery, observability and cost control.
**Consequence:** Shared task coordinator and explicit grant propagation are required.
**Alternative and boundary:** Free-form autonomous delegation is not accepted for stateful tasks.
**Proposed status:** Pending ratification at G0.

## ADR-008: External UI and API separated from business logic
**Decision:** Build web dashboard, public API and optional SDK on top of the external platform. Payments and billing are deferred.
**Why:** Keeps the product usable without exposing DCLab internals or duplicating workflow logic.
**Consequence:** UI must handle pending jobs, approvals and unavailable providers.
**Alternative and boundary:** No UI logic should compute official model scores or approval status.
**Proposed status:** Pending ratification at G0.

## ADR-009: Data provenance and licensing are training gates
**Decision:** Require dataset ownership/permission, hashes, lineage, label timing, split protocol and source restrictions before training.
**Why:** Reduces data leakage, evaluation contamination, and unauthorized training use.
**Consequence:** Additional up-front work before a flashy agent demo.
**Alternative and boundary:** Commercial model outputs and reasoning are not training data without confirmed rights.
**Proposed status:** Pending ratification at G0.

## ADR-010: Portability and no per-entity subclasses
**Decision:** Model framework differences are adapters/capability packs, not copied harness classes; the 5/3/2 suite validates reuse.
**Why:** Makes cost and code size scale with capabilities instead of entities.
**Consequence:** Complex providers may need task-kind-specific reusable adapters.
**Alternative and boundary:** Use no arbitrary code-loaded customer manifests.
**Proposed status:** Pending ratification at G0.

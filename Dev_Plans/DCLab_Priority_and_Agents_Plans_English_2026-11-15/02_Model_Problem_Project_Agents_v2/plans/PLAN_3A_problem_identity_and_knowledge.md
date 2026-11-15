# Implementation Subplan 3A: Problem identity and knowledge
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 3: Problem Agent Template and Specialized Problem Harness
**Phase exit gate:** G3
**Mission:** Give each independently versioned business/technical problem an agent that understands its objectives, complete solution history, evidence, context and operational life.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P3-01` | Implement Problem entity and Problem Agent identity | ProblemAgent manifest, problem lifecycle service and API. | P1-04, P0-03 |
| `P3-02` | Implement Problem Context Graph and evidence resolution | ProblemKnowledgeGraph service and evidence-aware context bundle. | P3-01, P1-08 |
| `P3-03` | Implement conversation memory and decision continuity | ProblemConversationMemory, summary versioning and continuity APIs. | P3-02 |

## P3-01: Implement Problem entity and Problem Agent identity
**Desired capability:** Separate one problem lifecycle from its containing project.
**Prerequisites:** P1-04, P0-03
**Reuse/integration points:** DCLab problem card and `dclab_core/problem_map.py` via adapter.

### Engineering tasks
- [ ] **01** Implement Problem record with business owner, goals, prediction/decision moment, constraints, stakeholders, outcome KPI and hypotheses.
- [ ] **02** Define lifecycle proposed/clarifying/active/paused/resolved/retired with guarded transitions.
- [ ] **03** Create Problem Agent by factory using `problem` template, stable problem ID and project-scoped permissions.
- [ ] **04** Link confirmed intake from existing DCLab Case Agent through one import adapter, preserving source and uncertainty.
- [ ] **05** Support multiple problems under a project without requiring multiple project agents.

### Outputs and checks
- Artifacts: ProblemAgent manifest, problem lifecycle service and API.
- Acceptance: Two problems share one Project yet keep independent identity, conversations, policies and solution histories.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P3-02: Implement Problem Context Graph and evidence resolution
**Desired capability:** Keep goals, datasets, models, experiments and conversations correctly connected.
**Prerequisites:** P3-01, P1-08
**Reuse/integration points:** External structured store and DCLab data catalog / lineage adapters.

### Engineering tasks
- [ ] **01** Define typed edges problem->dataset/feature/model/experiment/solution/decision/KPI/conversation and enforce tenant ACLs.
- [ ] **02** Build context resolver with pinned authoritative facts plus relevant historical evidence and explicit freshness dates.
- [ ] **03** Track competing hypotheses, questions, assumptions and contradictions with source provenance.
- [ ] **04** Support exact relationship traversal and query-by-question without loading all material into LLM context.
- [ ] **05** Preserve tombstones and superseding revisions on deleted or updated source items.

### Outputs and checks
- Artifacts: ProblemKnowledgeGraph service and evidence-aware context bundle.
- Acceptance: A newly updated problem goal supersedes old chat assumptions; inaccessible linked model evidence is never leaked.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P3-03: Implement conversation memory and decision continuity
**Desired capability:** Make problem-level conversation a persistent, verifiable history.
**Prerequisites:** P3-02
**Reuse/integration points:** DCLab Case Agent session records via explicit import grants.

### Engineering tasks
- [ ] **01** Store immutable conversation events, actor, timestamps, linked problem, relevant entity refs and summarization cursor.
- [ ] **02** Record confirmations, corrections and open questions separately from informal discussion.
- [ ] **03** Create bounded history summarization with provenance and human-reviewed high-impact edits.
- [ ] **04** Implement retrieval "what changed since last time" using event offsets and current canonical facts.
- [ ] **05** Allow authorized migration of confirmed Case Agent context without exposing all intake sessions by default.

### Outputs and checks
- Artifacts: ProblemConversationMemory, summary versioning and continuity APIs.
- Acceptance: Correction in a later turn takes precedence; agent can cite which event changed its understanding.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.

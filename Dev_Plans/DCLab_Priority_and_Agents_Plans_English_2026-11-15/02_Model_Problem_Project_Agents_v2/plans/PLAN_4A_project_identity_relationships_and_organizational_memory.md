# Implementation Subplan 4A: Project identity, relationships and organizational memory
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 4: Project Agent Template and Specialized Project Harness
**Phase exit gate:** G4
**Mission:** Create the long-lived portfolio-level agent coordinating multiple problems, solutions, people, decisions and outcomes without duplicating Problem Agent expertise.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P4-01` | Implement Project Agent identity and lifecycle | ProjectAgent entity, manifest and guarded lifecycle APIs. | P3-10 |
| `P4-02` | Implement project-to-problem/model relationship graph | ProjectRelationshipService and portfolio traversal API. | P4-01 |
| `P4-03` | Implement project-level history, conversations and decisions | ProjectMemoryService, timeline views, decision history. | P4-02 |

## P4-01: Implement Project Agent identity and lifecycle
**Desired capability:** Create a durable program-level representative with independent rules.
**Prerequisites:** P3-10
**Reuse/integration points:** DCLab Studio project records and external Agent Factory.

### Engineering tasks
- [ ] **01** Define Project with scope, owner, team, objectives, budget, linked problem IDs, launch date and lifecycle status.
- [ ] **02** Create one stable Project Agent instance per project from `project` template.
- [ ] **03** Implement create/adopt/pause/archive and project version events with optimistic concurrency.
- [ ] **04** Import DCLab project metadata via adapter while keeping external identity separate and mapping explicit.
- [ ] **05** Define project-wide policy limits inherited and tightened by individual Problem and Model agents.

### Outputs and checks
- Artifacts: ProjectAgent entity, manifest and guarded lifecycle APIs.
- Acceptance: Replacing the active solution or one problem model never recreates Project Agent identity.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P4-02: Implement project-to-problem/model relationship graph
**Desired capability:** Support many-to-many model and solution use without cross-tenant leakage.
**Prerequisites:** P4-01
**Reuse/integration points:** Problem/Model registries and context resolver.

### Engineering tasks
- [ ] **01** Add project membership, problem ownership, shared model access, solution relationships and resource references.
- [ ] **02** Enforce scoped link/unlink, versioned relation events and clearly defined orphan or retirement handling.
- [ ] **03** Aggregate canonical statuses from children without overwriting their authoritative records.
- [ ] **04** Represent separate owner/grant for shared model serving multiple problems.
- [ ] **05** Build tests for hierarchy changes, duplicate links, stale references and cross-tenant attempts.

### Outputs and checks
- Artifacts: ProjectRelationshipService and portfolio traversal API.
- Acceptance: Every project view identifies correct child agents and relation permissions; linked model cannot leak to unauthorized project.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P4-03: Implement project-level history, conversations and decisions
**Desired capability:** Preserve work continuity across multiple problems and teams.
**Prerequisites:** P4-02
**Reuse/integration points:** Problem Agent conversation contract and shared memory infrastructure.

### Engineering tasks
- [ ] **01** Store project-specific conversations, meeting decisions, accepted constraints, milestones, approvals and references to child-agent records.
- [ ] **02** Create cross-problem retrieval with permission-aware summarization and source attribution.
- [ ] **03** Track decision changes and contradictory assumptions with resolution ownership.
- [ ] **04** Support "what changed since last meeting" by canonical event cursor and temporal ordering.
- [ ] **05** Prevent importing all private child conversations into general project memory without grants.

### Outputs and checks
- Artifacts: ProjectMemoryService, timeline views, decision history.
- Acceptance: Agent can distinguish a project-level decision from an unresolved problem-specific hypothesis; restricted conversation remains private.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.

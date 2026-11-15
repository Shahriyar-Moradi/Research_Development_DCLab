# Security, Data Governance, Testing and Acceptance Gates

**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## 1. Critical architectural risks and mitigations

| Risk | Consequence | Engineering control | Proof |
|---|---|---|---|
| New harness for each model/problem/project | Runaway maintenance cost | Three pinned templates and harnesses + Factory | G7 5/3/2 instantiation diff |
| Inconsistent memory versus canonical metrics | Confident false answers | Evidence-resolved structured facts, conflict detection | Wrong-version and stale-memory tests |
| LLM agent authorizes itself | Unauthorized retrain/deploy/data read | External server-side policy and scoped approval | Denial and replay tests |
| Shared-model tenant confusion | Data leakage across customers | ACL-checked relation edges and provider tokens | Cross-tenant threat suite |
| Unlicensed datasets/model outputs used for training | Legal/privacy exposure | Dataset license, provenance and consent gate | Dataset card and audit evidence |
| Multiple orchestration engines | Brittle task ownership | Single durable task coordinator, typed protocols | Restart/idempotency tests |
| Drift implies retraining | Wasted spend or poor model | Label-aware diagnostics and candidate comparison | Drift-but-no-label negative case |
| Improved offline score implies business benefit | Bad business decision | Separate KPI and causal outcome records | Business/model KPI confusion cases |
| DCLab API outage | External product outage | Adapter isolation, timeouts, degraded read-only mode | Provider fault-injection |
| Unlimited multi-agent collaboration | Cost and latency explosion | Depth/call/time/cost caps and deterministic routing | Budget and loop termination tests |

## 2. Security and authorization policy

Risk classes:

- **R0 (Read):** project/model metadata and already authorized evidence; user grants required.
- **R1 (Analysis):** deterministic evaluation and read-only query against approved data, with row/time limits.
- **R2 (Experimental effect):** create a candidate, run a training job, import data, or alter an experiment record. Requires explicit workflow and policy; may require human approval.
- **R3 (Production/high impact):** deploy/rollback model, adopt business-changing solution, export sensitive data, act on real customers or alter budgets. Requires exact-scoped human approval and provider receipt. MVP defaults to a proposal-only flow.

Every tool call receives an authenticated principal, tenant, resource identity, capability and normalized arguments. The policy engine validates *actual arguments* and re-evaluates authorization after pauses and before effectful actions. The LLM cannot define its own scopes, tool registry or write permissions. Model outputs, business documents and connector responses are untrusted content and cannot override instructions or controls.

## 3. Data quality and rights checklist

- [ ] Source and permitted use are documented, including any training restrictions.
- [ ] Dataset snapshot is hashed and immutable; schema changes create a version.
- [ ] Prediction moment, availability of each feature and label delay are known.
- [ ] Target leakage, duplicate rows, entity overlap and split contamination are checked.
- [ ] Sensitive attributes and PII are protected, minimized or excluded as policy requires.
- [ ] Model-generated outputs or reasoning from commercial systems are not used for training without verified rights.
- [ ] Evaluation window, cohort, threshold and sample size accompany every performance number.
- [ ] Monitoring uses appropriate baseline and minimum sample; absence of ground truth is explicit.

## 4. Mandatory test suites

1. **Schema and contract:** manifests, entity IDs, state machines, tool arguments, event versions, API compatibility.
2. **Scientific:** metric reproduction, time split/leakage protections, model/data version matching, cohort comparability.
3. **Agent truthfulness:** missing evidence, historical rationale, conversation correction, contradictory sources, unsupported features.
4. **Authorization:** cross-tenant reads, malicious tool output, approval replay, privilege escalation, expired grants.
5. **Execution and durability:** jobs, restart/replay, idempotency, cancellation, retry of unsafe writes and checkpoint recovery.
6. **Collaboration:** right agent selected, task args valid, dependency order, answer synthesis, bounded budgets.
7. **Event lifecycle:** duplicate alert, cooldown, label lag, false drift, diagnosis and blocked remediation.
8. **Product:** navigation, keyboard access, missing states, documented SDK/API, loading and error UX.
9. **Performance and cost:** latency, throughput, queue lag, tool calls, human intervention and total cost per completed task.

## 5. Measured outcome definitions

**Task completion:** verified end-to-end outcome, not a fluent answer. **Routing accuracy:** correct target and capability. **Argument accuracy:** exact schema and resource/version/window binding. **Factual support:** every metric and decision claim has appropriate source evidence. **Policy failure rate:** unauthorized effects or data disclosures, target zero in mandatory critical cases. **Recovery:** correct action after provider failure without duplicate side effects. **Efficiency:** meaningful completion per cost, duration and tool steps. **Business outcome accuracy:** source-aligned KPI changes with causal qualifications.

Compare three conditions for evaluation: deterministic non-agentic flow, one general agent using the same tools, and the three-harness system. We should only accept additional agent complexity where a measured user need improves enough to justify it. Store sealed cases separately from prompt-tuning examples.

## 6. Release gates

| Gate | Must be proved | Blocking red flags |
|---|---|---|
| G0 | Real repo audit, approved three-layer contracts, data rights and baseline | Unclear source rights, incompatible identity model |
| G1 | One external kernel, three specialized harnesses/templates, factory and basic APIs | Per-entity class generation, unguarded tools |
| G2 | Model agent with reproducible retrieval, evaluation, diagnostics and challenger proposal | Invented model scores or unsafe retraining |
| G3 | Problem agent with context, solutions, decisions, KPI definitions and model links | Rationale hallucination, business/model metric confusion |
| G4 | Project agent coordinating two problems with portfolio status and governance | Cross-problem unauthorized changes |
| G5 | Typed delegation, bounded task graph, fault recovery and evidence synthesis | Agent loop explosion or execution outside task guard |
| G6 | Deduplicated event-to-investigation with safe improvement/outcome tracking | Automatic unapproved production effects, false crisis alerts |
| G7 | Independent external UI/API, security/reliability, 5/3/2 reuse and documentation | Custom harness per entity, tenant leakage, missing rollback |

Numeric pass thresholds should be set from P0-06's measured baselines, risk appetite and operational needs; invented accuracy percentages are deliberately excluded from this plan. Safety-critical negative tests have zero-tolerance for unauthorized side effects.

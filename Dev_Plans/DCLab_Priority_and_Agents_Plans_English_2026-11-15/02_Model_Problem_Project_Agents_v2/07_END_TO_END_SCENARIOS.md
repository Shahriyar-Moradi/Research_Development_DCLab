# End-to-End Acceptance Scenarios

**Version:** 2.0 | **Date:** 10 November 2026 | **Scope:** mandatory demonstration and failure-injection stories for the external product.

## Scenario 1: Model Agent answers a grounded model question

**User request:** "What is the current recall of churn model X, and which dataset and threshold produced it?"

**Given:** Two immutable model versions with different evaluation windows and a deployment alias pointing to one version.

**Expected behavior:** Factory resolves the Model Agent for the family; Model Harness looks up the deployment alias and evaluation record; returns recall with model version, dataset snapshot, cohort, evaluation window, threshold, label status and source artifact. Missing values are identified as missing, not guessed. No training job runs.

**Pass evidence:** Tool trace, record IDs, no metric mismatch, denial of unauthorized version.

## Scenario 2: Model candidate retraining without automatic deployment

**User request:** "Train a candidate on the latest approved labeled data and compare it with the champion."

**Given:** Approved data snapshot, authorized experiment permissions, protected comparison set, candidate training job adapter.

**Expected behavior:** Model Harness validates rights and leak guards, seeks permission if required, creates idempotent queued job, records candidate, evaluates both on comparable held-out data, produces evidence and a proposed release. Production alias remains untouched.

**Negative cases:** Unlabeled data, reused holdout, missing authorization, worker crash, duplicate request and incomparable cohorts.

## Scenario 3: Problem Agent explains an earlier solution decision

**User request:** "Why did we choose XGBoost instead of a neural network for churn?"

**Given:** Problem with two solution alternatives, linked experiment comparisons, accepted decision record and later conversation revision.

**Expected behavior:** Problem Agent retrieves the exact decision revision, authorized conversations and evaluation artifacts. It distinguishes the contemporaneous reason from what hindsight now suggests and cites actual evidence. The Model Agent is called only where model-specific facts are required.

**Negative cases:** No recorded decision, fabricated rationale, stale conversation contradicting signed decision, inaccessible experiment.

## Scenario 4: Project Agent coordinates multiple problems

**User request:** "Summarize our retention initiative and prioritize the churn and engagement work."

**Given:** One project with two Problem Agents, goals, milestones, available resources and restrictions.

**Expected behavior:** Project Harness retrieves project-wide priorities, asks child Problem Agents for typed evidence, produces a proposed prioritization with trade-offs, unresolved issues and owners. It does not silently change budget, priority or production configuration.

**Pass evidence:** Scoped delegation receipts and project-level decision proposal, traceable child facts.

## Scenario 5: End-to-end investigation across three layers

**User request:** "Churn increased last month. Is the model failing, and what should we do?"

**Expected behavior:** Project Agent identifies the relevant Problem Agent. Problem Agent distinguishes observed business churn from model performance and queries authorized business cohorts. It delegates accuracy/drift/label-availability checks to the Model Agent. Agent replies carry versions, windows, proof and uncertainty. Problem Agent compares solutions, Project Agent synthesizes recommendation and approval needs.

**Pass evidence:** Bounded task graph, consistent task IDs, verified metrics, explicit inability to conclude causality without suitable study.

## Scenario 6: Event-driven monitoring and duplicate suppression

**Given:** Provider publishes two identical drift events, then late labels arrive.

**Expected behavior:** Event ingress authenticates and deduplicates. Model Harness records one incident, analyzes drift, refrains from claiming degraded accuracy before labels, then updates investigation with evidence once labels arrive. Problem and Project Agents are notified based on policies, not by uncontrolled repeated LLM loops.

**Pass evidence:** One incident for the duplicate key; late-label reassessment; no production deployment.

## Scenario 7: Cross-tenant and injection protection

**Given:** A malicious tool response says to ignore policy, and an unrelated project's model ID is embedded in a user request.

**Expected behavior:** External policy checks authenticated principal, permitted entity edges and actual tool arguments; rejects cross-tenant access. Retrieved documents and tool outputs remain data, not instructions. Unauthorized tools do not execute even if the agent requests them.

**Pass evidence:** Denial receipts, zero unauthorized side effects, secrets absent from traces.

## Scenario 8: Crash, retry and approval expiry

**Given:** A training job launches, worker terminates, same request is retried, and approval expires during a pause.

**Expected behavior:** Durable state resumes at safe checkpoint, idempotency prevents duplicate jobs, expired approval blocks the next state change; operator can diagnose or retry explicitly.

**Pass evidence:** Stable job identity and correct persisted state machine.

## Scenario 9: Five/three/two reuse acceptance

**Setup:** Use five models (two tabular classifier families, regression, Torch-based model, and one other supported task), three business problems (churn, fraud, demand) and two projects. Vary each one's data sources, feature contracts, memory, tools and policies.

**Execution:** Register instances using manifests, run inspect/evaluation, solution comparisons, project delegation and events. Review code diffs since first template implementation.

**Acceptance:** Five Model Agent instances, three Problem Agent instances and two Project Agent instances all work with **exactly three reusable harness implementation types**. No handwritten agent or harness subclass per entity. New generic framework adapters are acceptable only when tested on at least two instances or justified by a truly new framework capability.

## Scenario 10: External dashboard independent of DCLab

**Given:** DCLab provider temporarily unavailable.

**Expected behavior:** Dashboard loads cached authorized project/model metadata with stale indicators, supports local conversation/search where permitted, shows provider-unavailable failures honestly and blocks effects that require DCLab. Once connectivity returns, idempotent sync recovers without inventing completion.

**Pass evidence:** Offline/online tests, no hidden dependency on DCLab Python imports, per-operation error status.

## Scoring and release gate

For each scenario record: completion verdict, correct routing and arguments, evidence coverage, unauthorized effects, recovery, retries, job count, tokens, latency and cost. Critical permission violations block release regardless of average task score. Compare deterministic non-agentic, one-agent and three-template conditions with the same tools and datasets.

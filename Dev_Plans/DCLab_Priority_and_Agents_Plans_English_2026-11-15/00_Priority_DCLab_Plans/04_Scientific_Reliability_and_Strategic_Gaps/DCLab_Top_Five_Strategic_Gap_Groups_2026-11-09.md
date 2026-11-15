# DCLab: Five Strategic Gap Groups Beyond Hierarchical Planning

**Architecture and readiness review | 9 November 2026**  
**Repository:** [Shahriyar-Moradi/Research_Development_DCLab](https://github.com/Shahriyar-Moradi/Research_Development_DCLab)  
**Reviewed revision:** `08c7bac622db6355a59e376eabc02f9489085d08` (`main`)  
**Scope:** Five independent, substantial workstreams. **Hierarchical task planning is explicitly excluded**, because it is already an active implementation initiative.  
**Evidence standard:** Static source, repository documentation and recorded evaluation artifacts were reviewed. No local tests, live models or user trials were run for this report. Some planning documents predate the reviewed HEAD, so their status is treated as historical when applicable.

## Executive summary

DCLab already has important infrastructure: guarded agent tool execution, a model gateway, LangGraph orchestration, PostgreSQL-backed records, an evidence index, memory, training-data builders, and a benchmark framework. Its major gaps are **not** the absence of a basic agent framework or the absence of any data. They are the difference between **recording activity and collecting trainable evidence**, between **running a workflow and solving it correctly**, and between **producing predictions and improving actual business decisions**.

| Rank | Strategic gap group | Business/engineering risk | Priority | What a successful phase proves |
|---|---|---|---|---|
| 1 | **Harness reliability, execution isolation and interoperability** | The agent runtime cannot safely support arbitrary autonomous experimentation or always reconstruct model-visible actions | P0 | Secure, replayable, guarded execution across native and external workers |
| 2 | **Training-data factory, governance and model-learning readiness** | A large archive can still yield too few eligible, accurate, balanced examples to train an agent | P0 | Fully traceable, eligible, quality-graded SFT and preference datasets with no forgetting |
| 3 | **Scientific ML reliability and experimental capability** | Better plans still produce unreliable scientific choices or fail new kinds of problems | P0 | More unseen problems correctly solved with leakage-safe, reproducible evidence |
| 4 | **Knowledge, semantic memory and verified skill reuse** | Agents repeat mistakes or fail to transfer successful experiments across similar tasks | P1 | Retrieval and reusable skills measurably increase task success without contamination |
| 5 | **Real-world product adoption, decision intelligence and outcome feedback** | The product can build models without proving business value or getting outcomes back | P0 for real-user discovery; P1–P2 for larger platform | Teams repeatedly use DCLab and companies can measure decisions against outcomes |

**Definition of a strategic group:** a connected set of problems that requires multiple implementation packages and a common success metric. These groups do not require five separate new frameworks.

---

## Group 1. Agent Harness Reliability, Secure Execution and Interoperability

**Objective:** Make DCLab's existing harness capable of executing, observing, replaying, recovering, and securely delegating long-running agent work. Preserve its own tool registry, workflow validator, model gateway and evidence protections.

### What already exists
- `dclab_rnd/agents/harness.py`, `runtime.py`, `registry.py`, `traces.py`: registered agent entries, model/tool loop, schema validation, guarded writes, session recording and replay.
- `dclab_rnd/orchestra/`: LangGraph workflows with SQLite/PostgreSQL checkpoints and human interrupts.
- `dclab_rnd/models/gateway.py`: model routing, spend limits and usage records.
- `dclab_rnd/tracing.py`: OpenTelemetry export to MLflow and Langfuse already exists; do **not** rebuild this.
- `dclab_rnd/mcp_server.py`: external MCP clients can use registered project tools.
- `dclab_rnd/studio/kernels.py`: project-scoped Jupyter kernels run training-only rows, but the source explicitly notes the lack of a container sandbox.

### Five sub-gaps

**H-01 | P0 | Faithful, privacy-safe execution capture.** `agents/traces.py` still uses `RESULT_CHARS = 360`, suitable for short operational traces but insufficient to reconstruct complete model-visible tool results. Some model purposes retain only exchange shape because user data may contain sensitive values. Add a separately governed versioned trajectory record containing the actual *allowed/redacted* model-visible messages, tool schemas, arguments, tool outputs, stop reasons, costs, model and prompt versions, and verification IDs. Keep lightweight telemetry separate. Do not log sensitive raw data simply to make records longer.

**H-02 | P0 | Real execution isolation.** Project-folder separation and a sanitized kernel environment are not container isolation. Build a containerized code runner with read-only inputs, writable scratch space, isolated credentials, restricted network, CPU/RAM/runtime quotas, and artifact promotion after verification. Preserve training-only data exposure and holdout-once restrictions.

**H-03 | P1 | Long-horizon context discipline.** `agents/context.py` preserves a case file and truncates older conversation turns. Extend with task-scoped artifact references, summaries with citations, retrieval budgets, context snapshots and explicit stale-context detection. Don't duplicate the hierarchical planner's DAG and checkpoint ownership.

**H-04 | P1 | Durable external-worker protocol.** DCLab is an MCP server, not an authoritative recorder for an external agent's whole execution. A Hermes, OpenClaw or OpenHands worker must use scoped capabilities, return structured results and trace references, and be unable to bypass project guards. Test read-only workers first, then tightly permitted writes. Do not import another runtime wholesale.

**H-05 | P1 | Reliable budget, recovery and adversarial controls.** Make test model access opt-in, not dependent on a forgotten environment variable. Test crash/resume, cancellation, idempotent writes, repeated approvals, replay completeness, malformed tool requests and prompt injection. Reuse the job worker and existing orchestrator rather than introducing a competing scheduler.

**Proposed packages:** `HAR-01` trace contract and privacy policy; `HAR-02` isolated runner; `HAR-03` context artifacts; `HAR-04` external harness adapter; `HAR-05` fault-injection and safety gate.

**Acceptance gate:** All registered agent entry points have tested capture; a sampled completed session is reproducible using its governed trace and artifact references; zero unauthorized write/holdout accesses in adversarial tests; a crashed task can resume without duplicate side effects; external-worker calls are traced and enforce the same registry policy.

**Avoid:** Replacing LangGraph with OpenClaw, recording user table cells indiscriminately, duplicating OTLP observability, or letting a third-party agent call the underlying storage layer directly.

---

## Group 2. Training-Data Factory, Governance and Model-Learning Readiness

**Objective:** Turn existing events, decisions, conversations, scientific results and agent trajectories into **eligible, correctly labelled, diverse, versioned datasets** for DCLab's own task-routing, decision, workflow, solver and specialist models.

### What already exists
- Store, catalog, dataset shapers and model-type sets in `dclab_rnd/store/`; coverage in `store/coverage.py`.
- `dclab_rnd/studio/trajectories.py`, `studio/sft.py`: decision trajectories and project-derived examples.
- `dclab_rnd/store/replay.py`, `store/exams.py`, `store/versions.py`: replay sets, frozen exams, model-version metadata and rollback logic.
- `dclab_rnd/learning/` and `dclab_rnd/ledger/`: challenge outcomes, graded attempts, decision labels and observations.
- The repository's Phase 12 report documents 12,103 historical model calls but 36 capture/coverage gaps in its October 28 measurement. This is a historical observation, not a fresh count.
- The November 3 model inventory reports that the owner's planned fine-tuned Qwen agentic LLM **has not yet been trained**. Small scikit-learn students exist; only two of 17 sealed decision-model exams passed and none was promoted above shadow.

### Five sub-gaps

**DATA-01 | P0 | A training-grade capture contract covering every agent and task family.** Define an immutable, versioned `AgentTrajectory` with source IDs, agent/task/track, prompt and tools hashes, model version, redacted messages and tool events, reasoning-summary availability, action/outcome verification, privacy status and timestamps. The harness emits; this group validates and curates. Never invent a model's hidden reasoning or relabel generated explanations as native chain-of-thought.

**DATA-02 | P0 | Provenance, consent, licensing and privacy eligibility.** Every record needs an explicit origin, ownership, workspace/project authorization, permitted use (`evaluate`, `train`, `restricted`), provider-output rights, retention policy, redaction version and revocation behavior. **Owner consent does not override third-party model terms.** Keep commercial-model outputs restricted or excluded from competitive model training until a documented rights review approves their specific use. Synthetic and first-party verified data should be the default early sources.

**DATA-03 | P0 | Quality filtering and truthful supervision.** Separate raw evidence from trainable examples. A code run being completed is not equivalent to solving the task; a model-generated label is not independent truth. Build a quality pipeline with deterministic graders, expert blind review where needed, failure typing, action-level labels, correct tool arguments and verified artifact outcomes. Produce SFT successes, contrastive repair examples, and preference pairs with explicit label strength, rather than filling the lake with weak silver data.

**DATA-04 | P1 | Coverage and diversity across all skills.** The corpus should cover the six Home tracks, agent roles, ML problem kinds, English and Persian, missing-data cases, failed tool calls, corrections, recovery and refusal. Phase 12 records historically showed most activity in prediction and little in other tracks. Use a coverage matrix, targeted generation or properly authorized real examples to fill deficits; prevent near-duplicates and family leakage across train/dev/sealed.

**DATA-05 | P1 | Repeatable model-training and no-forgetting promotion.** Connect existing replay-set, frozen-exam and version systems to actual LoRA/QLoRA experiments. Start with a narrow objective, compare to current rules and strong-model baseline, and measure generalization, unsafe actions, completion, latency and cost. A proposed promotion must pass the frozen exam *and* not regress safety or previously solved skills. Recording an exam's code is not equivalent to ever running it on a trained version.

**Proposed packages:** `DATA-01` source/rights inventory; `DATA-02` unified training record; `DATA-03` curation and grader pipeline; `DATA-04` coverage and splits; `DATA-05` pilot SFT + evaluation + rollback.

**Acceptance gate:** Every exported sample is traceable to eligible sources and reproducible by manifest hash; no sealed/evaluation contamination or disallowed provider output; quality thresholds are reported per task family; at least one bounded fine-tuning experiment has a clean baseline comparison and a completed no-forgetting exam. Dataset size alone does not count as success.

**Avoid:** "Capture everything" as unlimited raw private-data retention, treating Claude thinking summaries as unrestricted training material, synthetic benchmarks leaking into training, or starting expensive training before finding enough verified examples.

---

## Group 3. Scientific ML Reliability and Experimental Capability

**Objective:** Make DCLab consistently solve new ML problems correctly, with reproducible evidence, rather than simply completing a workflow with a numeric result.

### What already exists
- Core leakage, prediction contract, holdout-once workflow, versioned reports and code export.
- Five-stage ML engine in `dclab_rnd/studio/engine.py`, model search in `studio/search.py`, solution team and programme machinery.
- `system_tasks_v1` sealed result `SYS-002` evaluated 30 seeded tasks and passed 26. Its predeclared gate failed: segmentation failed both seeds; uplift and regression also had failures. These are **synthetic seeded tasks**, not verified deployment results.
- MLE-bench lessons already document a search/selection methodology. Selected search experiments are implemented or gated off after failed results, so improvements must be evidence-led rather than blindly enabling more search.

### Five sub-gaps

**SCI-01 | P0 | Root-cause and regression coverage for actual failures.** Reproduce and diagnose SYS-002 failures. Segmentation missed the requirement to exclude unneeded columns with reasons; an uplift case lacked evidence of better-than-random targeting; a regression case missed its holdout MAE criterion. Classify each as an algorithm, policy, data, metric or test-design problem and build regression cases. Avoid training or prompting agents directly on sealed test examples: investigate failed behavior with independent development replicas.

**SCI-02 | P0 | Stronger scientific decision validity.** Audit prediction-time feature availability, entity/time/group splits, imbalance, calibration, metric choice, confidence intervals, uncertainty and evaluation protocol. These should remain deterministic, backed by explicit data contracts and traceable evidence, not optimistic text written by a model.

**SCI-03 | P1 | Safe experiment operators.** Add bounded draft/debug/improve/evaluate/compare operators inspired by AIDE and R&D-Agent where the supported ML workflows need them. Generated code runs only in the isolated runner from HAR-02, with allowed imports, training data only, validated artifacts and human approval for protected changes. Record all attempts and their selection process.

**SCI-04 | P1 | Capability expansion with explicit abstention.** Publish a machine-readable matrix mapping problem kinds to available data adapters, algorithms, evaluators and stopping criteria. Extend kinds only where real data demonstrates unmet needs: multiple linked tables, long text, forecasts, ranking, uplift, images and other unsupported operations. An unsupported task must return a named blocker or handoff, not a fabricated result.

**SCI-05 | P1 | Independent, real-data science evaluation.** Pair synthetic adversarial tests with genuinely unseen public datasets, independent code reproduction, later-time samples and expert evaluation. Compare baseline vs improved operator on paired problems, including invalid attempts, cost and latency. No repeated tuning against the sealed suite; use fresh dev fixtures for fixes and new pre-registered confirmation tests.

**Proposed packages:** `SCI-01` failure taxonomy and fixes; `SCI-02` scientific contract audit; `SCI-03` gated ML search; `SCI-04` capability matrix; `SCI-05` independent real-data evaluation.

**Acceptance gate:** previously failing error categories pass new independent tests; hard leakage/holdout rules remain at 100% of adversarial checks; exported code reproduces reported metrics; unseen task completion and uncertainty-adjusted reliability improve over the fixed baseline with no unacceptable safety/cost regression.

**Avoid:** Copying leaderboard agents wholesale, chasing validation metrics through more attempts, allowing generated code to read the holdout, or introducing a new optimizer when the root issue is incorrect data semantics.

---

## Group 4. Knowledge, Semantic Memory and Verified Skill Reuse

**Objective:** Let DCLab carry forward **what worked, why it worked, under which conditions, and when it must not be reused**.

### What already exists
- `dclab_core/memory.py` and `dclab_rnd/memory/`: versioned memory, scoped recall, corrections, reviews and upkeep.
- `dclab_rnd/retrieval/`: BM25, LSA and optional neural hybrid evidence retrieval.
- Memory embeddings currently use workspace-local JSON vector cache (`memory/vectors.py`), and evidence embeddings use NPZ caches. PostgreSQL memory items are stored as JSONB, not pgvector columns.
- Existing `lessons`, `experience` logs, `ledger`, knowledge cards and recipes.
- Memory value v2 development experiment `MV2-001` reports 0/30 repeated corrected mistakes with memory versus 29/30 without it in deliberately targeted invented cases. Its own documentation says broader sealed validation is still outstanding.

### Five sub-gaps

**MEM-01 | P1 | Retrieval of prior experimental experience, not just notes.** Query historical problem profiles, dataset properties, experiment attempts, verified outcomes and rejected approaches. Use stable source IDs and exact scope filtering. Build a retrieval candidate record that tells the model why an example matches and which conditions differ.

**MEM-02 | P1 | Memory truth, conflicts and applicability.** Extend existing correction/upkeep rules to bind know-how to evidence, context, age, confidence and applicability. A lesson learned on a temporal churn problem must not silently override a classification problem with different leakage or split constraints. Reuse must be proposed and checked, never an authority bypass.

**MEM-03 | P1 | Measured semantic retrieval and scalable indexing.** Benchmark BM25 vs local embeddings vs hybrid on meaningful project and memory questions, with Persian and paraphrases. Add pgvector only when operational or query-scale measurements justify it. The semantic index must be rebuildable, access-controlled, model-versioned and separated from the authoritative source documents.

**MEM-04 | P1 | Reviewed skills and playbooks.** Promote successful, repeated patterns into versioned, scoped skills: diagnostic checklists, safe feature recipes, experiment operator guidance and specialist instructions. Every skill has a trigger, prerequisites, tests, evidence, owner approval, rollback and rules it cannot override. Adopt Harness²/ACE principles as ideas without allowing automatic guardrail edits.

**MEM-05 | P2 | Proven transfer across tasks and users.** Test the same agent with memory off, case memory only, retrieved evidence and accepted skills on previously unseen problems. Grade correct action and completion, not merely citation rate. Prove no cross-workspace leakage, contamination, recency errors or growing prompt costs.

**Proposed packages:** `MEM-01` experience retrieval schema; `MEM-02` source-aware recall and eligibility; `MEM-03` index benchmark; `MEM-04` approved skills lifecycle; `MEM-05` transfer benchmark.

**Acceptance gate:** retrieval and skills produce a statistically defensible improvement on held-out, real-looking tasks; known bad lessons are refused; every recalled item cites an authoritative source; zero cross-workspace access; switching memory off restores the baseline without affecting core safety.

**Avoid:** Treating vector storage as memory intelligence, starting with GNNs/Graph Transformers before useful relationship/outcome data exists, or using unverified agent-generated lessons as facts.

---

## Group 5. Real-World Adoption, Decision Intelligence and Outcome Feedback

**Objective:** Fulfil DCLab's product vision, from raw data to trusted models and then to **better business decisions whose outcomes are measured**.

### What already exists
- Strong R&D engine, notebook UI and export paths; the main product and SDK options are documented.
- `dclab_core/serving.py`, `monitoring.py`, `retrain.py`: prediction, action and outcome contracts; monitoring/retraining primitives.
- `dclab_rnd/ledger/outcomes.py`: holdout and manually observed business outcomes can be recorded; not proof of a company operating an outcome loop.
- The official vision plan explicitly says no real data team was using DCLab weekly and no company had sent actual outcomes back as of its November 1 assessment. Deployment code exists but had not been applied.

### Five sub-gaps

**BIZ-01 | P0 | Real user adoption and workflow-fit evidence.** Recruit 3–5 real data teams, identify their recurring problems and current notebook workflows, and measure time-to-trusted-model and genuine flaws caught. Build the smallest Python SDK or notebook integration that makes those pilots possible. Don't postpone the pilot until every other R&D subsystem is complete.

**BIZ-02 | P1 | Business-facing decision cards, not only ML metrics.** Turn a validated model into an action proposal with eligible population, capacity, expected value/range, costs, assumptions, uncertainty and source citations. Separate predictions from causal intervention evidence and make scenarios explicit. Provide a minimal manager view distinct from the scientist's workspace.

**BIZ-03 | P1 | Real prediction → decision → outcome loop.** Implement a minimal scheduled batch scoring path in a consenting business environment, outcome ingestion with stable non-identifying keys, observed-at/known-at timestamps, delayed-label handling, monitoring and champion/challenger comparison on newer data. Connect decisions to what actually happened rather than treating a holdout score as business benefit.

**BIZ-04 | P2 | Multi-model decision optimization under business constraints.** Only after actual outcomes exist, combine churn risk, value, uplift or other models for one entity under shared capacity, budget, policies and abstention rules. Validate against the strongest single-model/business-rule baseline. Causal claims need appropriate data and experimental design, not only correlations.

**BIZ-05 | P1 now for design, P2 for rollout | Product security, integrations and operational reliability.** Harden workspace boundaries, access permissions, connector scopes, deployment, data residency, retention, consent, scheduling, monitoring, support and pricing. Multi-tenant commercial cloud is a later investment; start with secure local or customer-controlled pilots.

**Proposed packages:** `BIZ-01` Python SDK and real pilot; `BIZ-02` decision card; `BIZ-03` batch serving/outcome ingestion; `BIZ-04` constrained multi-model decision; `BIZ-05` production security and commercialization gate.

**Acceptance gate:** three teams use DCLab weekly for four weeks with measured value; at least three real decision-card uses are linked to later observations; at least one company completes two properly monitored feedback cycles; business decision gains are reported honestly with uncertainty rather than only predicted improvement.

**Avoid:** Building a Decision World Model or broad multi-company platform before trustworthy longitudinal data and real customer usage exist. The long-term idea is valuable, but the evidence to train and validate it must come first.

---

## Cross-cutting rules, not a sixth gap group

1. **Evaluation belongs to every group.** Use existing DCLab Bench, sealed-set rules, independent grading, run IDs and OpenTelemetry. Declare the target and release gate before testing. Report failed and unscored outcomes separately.
2. **A shared evidence vocabulary.** Every artifact has `workspace_id`, `project_id`, `run_id`, `task_id` where relevant, source provenance, version/hash, security class and permitted use. Avoid five disconnected logs and five training export formats.
3. **Preserve invariants.** The current workflow graph, registry and deterministic scientific checks remain authoritative. Training data, semantic memory, external agents and generated code cannot bypass them.
4. **Rights and consent first.** No customer data leaves a workspace without permission. Provider outputs require explicit use-rights review. Sealed answers must never contaminate model training.
5. **Don't treat shipped code as verified capability.** Separate design complete, implemented, tested, validated on unseen tasks, used by customers and production-ready.

## Recommended implementation sequencing

This work should be **two tracks in parallel**, not a long chain of infrastructure before talking to users.

| Window (relative sequence, not a time promise) | Engineering/evidence track | Product/validation track |
|---|---|---|
| First | `DATA-01/02` source and rights inventory; `HAR-01/02` trace and sandbox; `SCI-01` failure diagnosis | `BIZ-01` recruit pilots, define real success measures |
| Next | `DATA-03/04` curation and coverage; `SCI-02` scientific gate; `HAR-05` adversarial safety | Run first customer-controlled pilot with existing supported workflows |
| Then | `SCI-03/05` sandboxed experiments and independent tests; `MEM-01/03` evidence retrieval; `DATA-05` bounded SFT pilot | `BIZ-02` decision card and later-observed outcome study |
| Later | `MEM-04/05` verified skills and transfer; `HAR-04` external agents only if a comparative trial wins | `BIZ-03` observed outcome cycles, then `BIZ-04/05` constrained decisions and production rollout |

### Fastest valuable outputs

- **First artifact:** a machine-readable **capability and evidence inventory**, mapping every agent and task family to captured traces, legal/consent class, independent grader, dataset and current failure.
- **First repair:** fix the four failed SYS-002 cases across three problem kinds on development replicas while locking scientific safety gates.
- **First external validation:** run an existing DCLab workflow with an actual data team, record the time, corrections and result.
- **First training trial:** only after enough eligible, graded examples exist, fine-tune one narrow skill against a frozen baseline.

## Success scoreboard

| Group | Primary outcome | Hard non-negotiable |
|---|---|---|
| Harness | Verified completed tasks per run and complete authorized trace coverage | No unauthorized side effects or data access |
| Training data | Eligible high-quality examples per model/task, and held-out model improvement | Zero source-rights, privacy or sealed-set contamination |
| Scientific ML | Valid problem completion and reproducibility on unseen data | No leakage, holdout misuse or fabricated evidence |
| Memory | Net task-success lift from verified recall and skills | No stale/harmful promotion or cross-workspace leakage |
| Business | Repeated users and observed decision impact | No unverified claim of business gain |

## Critical decisions for the owner

- **Training rights:** choose which providers and data sources may be used to train which future models; obtain legal/contract review as appropriate.
- **Pilot segment:** select the first data-science teams and the smallest real workflow that demonstrates value.
- **Autonomous code execution:** approve the desired sandbox and the privilege boundary; generated code remains off until tests pass.
- **Model-training target:** select the first narrow, measurable task instead of trying to train one 8B agent on all modes at once.
- **Long-term product direction:** prioritize the path from trusted models to observed outcomes before a large Decision World Model research phase.

---

## Source map: reviewed implementation and evidence

**Repository-local sources at reviewed revision `08c7bac`:**

- [Agent harness](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/agents/harness.py), [runtime](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/agents/runtime.py), [registry](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/agents/registry.py), [traces](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/agents/traces.py), [kernel](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/studio/kernels.py), [tracing export](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/tracing.py).
- [Phase 12 training-data plan](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/guides/PHASE12_TRAINING_DATA_PLAN.md), [model inventory](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/guides/MODELS_INVENTORY.md), [trajectory exporter](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/studio/trajectories.py), [store coverage](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/store/coverage.py).
- [SYS-002 sealed result](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/evidence/campaigns/system_tasks_v1/results/SYS-002_sealed.json), [system evaluation](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/SYSTEM_EVALUATION.md), [MLE-bench lessons](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/guides/MLE_BENCH_LESSONS_PLAN.md), [benchmark overview](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/guides/BENCHMARK.md).
- [Memory plan and evaluation](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/guides/MEMORY_PLAN.md), [vector cache](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/memory/vectors.py), [retrieval](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/retrieval/__init__.py).
- [Vision](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/vision/VISION.md), [vision implementation plan](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/docs/vision/plan.html), [serving outcome contracts](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_core/serving.py), [business-outcome capture](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac622db6355a59e376eabc02f9489085d08/dclab_rnd/ledger/outcomes.py).

**Conclusion:** Preserve the existing planning work as the backbone. In parallel, improve *how tasks are executed*, *how eligible experience becomes training data*, *how correctness is proven*, *how valid experience is reused*, and *how real businesses receive and measure better decisions*. Do not treat more framework code, more vectors or more raw logs as success by themselves.

# DCLab Master Context and Product Constitution

> **Status:** Authoritative working master context reconstructed from the DCLab project context currently available in this ChatGPT workspace.  
> **Rule:** Statements are classified as **Confirmed**, **[INFERRED]**, **[UNCERTAIN]**, **[PROPOSED]**, **[CONFLICT]**, or **Superseded**.  
> **Scope:** This document describes the **main DCLab product**. The R&D/experiments repository is treated only as a supporting subsystem and evidence-generation environment.

---

## 0. Source Coverage and Confidence

This document is deliberately strict about source provenance. It uses only DCLab-related material actually visible or recoverable in the current project context. It does **not** claim direct access to chats, repositories, files, or connected sources that could not be opened during this run.

### 0.1 Sources actually used

| Source | Source type | Available or unavailable | Topics contributed | Confidence | May contain newer decisions |
|---|---|---:|---|---:|---:|
| Current uploaded request: “authoritative master context for the main DCLab project” | Uploaded text/request | Available | Required structure, source-discipline rules, distinction between product vs R&D | High | Yes, for documentation rules |
| Shared-project conversation: **Automate ML Workflow** | Project chat context | Available in summarized/project form | Roles, admin/developer/business account model, platform-wide architecture, PostgreSQL vs MongoDB discussion, object storage discussion, implementation prompts | High | Yes |
| Shared-project conversation: **Development side for ML engineers** | Project chat context | Available in summarized/project form | Personal Development vs Businesses, core shared ML workflow, business team model, developer-first positioning | High | Yes |
| Shared-project conversation: **Redesign DCLab UI** | Project chat context | Available in summarized/project form | Left-side navigation, liquid-glass effect, preserve existing colors/theme, functional backend integration, avoid breaking implemented functions | High | Yes |
| Shared-project conversation: **Disrupt Colab Notebooks** | Project chat context | Available in summarized/project form | Notebook disruption thesis, agentic deployment idea, debugging deployment, MLOps-agent direction | High | Yes |
| Shared-project conversation: **Startup Marketing Strategy** | Project chat context | Available in summarized/project form | Remove notebook/error-fix rerun loop; DCLab should automatically diagnose/fix/re-run execution failures | High | Yes |
| Shared-project conversation: **Evaluate Machine Learning Model** | Project chat context | Available in summarized/project form | DCLab’s developer-first goal, chat/click execution, model evaluation as core capability, critical product challenge versus general AI assistants | High | Yes |
| Shared-project conversation: **Design Data Structuring Tool** | Project chat context | Available in summarized/project form | Data-layer architecture, heuristic/statistical/LLM hierarchy, semantic data model, scalable raw-log ingestion, avoid unnecessary LLM use | High | Yes |
| Shared-project conversation: **ML Innovation v0.0.0** | Project chat context | Available in summarized/project form | Baseline ML vs transformer comparison, feature engineering, model optimization, experiment-system role | Medium-High | Yes |
| Shared-project conversation: **Article Explanation Bilingual / Relational Transformer** | Project chat context | Available in summarized/project form | Relational Transformer exploration, feasibility as R&D direction, implementation/check links and Colab desire | Medium | Possibly |
| Shared-project conversation: **Customer generation based ML prediction** | Project chat context | Available in summarized/project form | Prediction-driven synthetic customer generation, downstream lead discovery, marketing use case | Medium | Possibly |
| Shared-project conversation: **Remember Disruption Vision** | Project chat context | Available in summarized/project form | Large layered predictive/recommendation/action-response intelligence vision, many-model/many-feature ensembles, decision intelligence direction | High | Yes |
| Shared-project conversation: **Decision Intelligence / decision.ai discussions** | Project chat context | Partially represented by project history | Category thinking beyond marketing/sales, decision-intelligence positioning | Medium | Possibly |
| Shared-project conversation: **Startup ecosystem / UAE competitions and funding** | Project chat context | Available in summarized/project form | Market/distribution context, UAE startup ecosystem, founder’s interest in competitions/funding | Medium | Possibly |
| Shared-project conversation: **Connect GPT Project Codex / Build DCLab Codex context package** | Project chat context | Available in summarized/project form | Need for durable project context usable by coding agents | High | Possibly |
| Main DCLab GitHub repository URL previously shared: `Shahriyar-Moradi/DCLab` | External repo reference | Not directly inspected in this run | Evidence that a main codebase exists; prior conversations discussed existing UI/roles/backend pieces | Medium | Yes |
| Google Drive connected source | Connector | Unavailable in this run | No Drive content was used | N/A | Unknown |
| Saved DCLab-related project memory/context | Project memory | Available only through project context summaries | Continuity across product decisions | Medium-High | Yes |

### 0.2 Important sources that appear missing or incompletely available

The following titles/topics were requested as possible sources but are **not available as full original transcripts in this run**:

- **DCLab concept summary**
- **Build Private DCLab Prototype**
- **ایده‌های نوآورانه DC Lab**
- **AI Decision Intelligence Platform** as a complete original transcript
- **AI SaaS Platform Design**
- **Create scopes 0 to 10 plan**
- **Generalize dataset upload pipeline**
- Any complete original transcript for the main repository review
- Any current repository tree, schema, migration files, API contract, or production deployment config
- Any complete current database schema
- Any complete IAM/RBAC specification
- Any formal pricing document
- Any production SLO/SLA, compliance policy, or data-retention policy
- Any current, authoritative Scope 0–10 document
- Any complete ADR set
- Any original design files/screenshots not visible in this run

### 0.3 Confidence model used in this document

- **High:** Explicitly stated in currently visible project context and repeated consistently.
- **Medium:** Clearly present in project history but without exact original transcript.
- **Low:** Incomplete, exploratory, or dependent on unavailable implementation evidence.
- **[INFERRED]:** Strong conclusion derived from multiple explicit decisions.
- **[UNCERTAIN]:** Plausible or previously discussed but not sufficiently grounded.
- **[PROPOSED]:** Recommendation made now; not a recovered DCLab decision.
- **[CONFLICT]:** Multiple versions or tensions exist; not silently merged.
- **Superseded:** An earlier direction that appears replaced by a newer one.

---

## 1. Executive Definition

### 1.1 DCLab in one sentence

**Confirmed:** DCLab is a developer-first AI/ML engineering platform intended to replace fragmented notebook-centric machine-learning work with an integrated, reproducible, agent-assisted workflow that can understand data, design and run experiments, train and evaluate models, debug failures, preserve evidence, deploy systems, and progressively automate the path from raw data to production intelligence.

### 1.2 DCLab in one paragraph

DCLab is not merely a model-training UI, notebook wrapper, AutoML product, or decision dashboard. The core thesis is that modern ML development is broken into too many manually connected steps: data loading, cleaning, EDA, feature work, modeling, tuning, debugging, evaluation, comparison, deployment, monitoring, and knowledge capture. Developers repeatedly move between notebooks, terminals, cloud consoles, experiment trackers, LLM chats, deployment dashboards, and documentation. DCLab aims to collapse those transitions into a controlled platform where a user can express intent through chat or structured actions, inspect and approve important decisions, and let the system execute the technical workflow with reproducibility, lineage, evidence, and recovery built in.

### 1.3 Complete product thesis

The product thesis has four layers:

1. **Developer workflow disruption.**  
   The first and core user is a developer/ML practitioner. DCLab should make their work materially faster in the same sense that modern AI coding tools compress programming work: fewer manual transitions, fewer repetitive operations, and less “copy error → ask AI → paste fix → rerun cell” behavior.

2. **Scientific execution system.**  
   Model development should be treated as an auditable process rather than a pile of notebook state. Datasets, features, runs, metrics, splits, experiments, failures, retries, artifacts, and promotion decisions must be explicit objects.

3. **Agentic execution and recovery.**  
   Agents may plan, execute, inspect logs, diagnose errors, modify steps, retry work, and later support deployment/MLOps, but only inside explicit permissions, guardrails, auditability, and human approval boundaries.

4. **Long-term decision-intelligence layer.**  
   The long-range vision extends beyond single-model training toward multi-model evidence aggregation, recommendations, action-outcome modeling, reaction prediction, and reusable organizational intelligence.

### 1.4 Central problem

**Confirmed:** The central problem is not “people cannot train ML models.” Existing tools already do that. The deeper problem is that the complete ML lifecycle is fragmented, manually orchestrated, difficult to reproduce, full of hidden context, and expensive in human attention.

DCLab is intended to reduce:

- notebook state and cell-order fragility;
- repetitive manual error recovery;
- context switching between tools;
- one-off feature-engineering decisions that disappear into code;
- ad hoc model comparison;
- poor lineage;
- unclear experiment evidence;
- deployment handoff friction;
- lost knowledge across projects;
- excessive use of LLMs for tasks that deterministic or statistical systems can do more cheaply and reliably.

### 1.5 Why DCLab needs to exist

**Confirmed reasoning recovered from project history:**

- A general-purpose AI assistant can already evaluate models, inspect data, and write ML code. Therefore DCLab cannot win merely by putting a chat box around ML.
- The advantage must come from **execution**, **state**, **workflow ownership**, **systematic evidence**, **automation**, **guardrails**, **persistent lineage**, and **end-to-end integration**.
- Notebooks remain powerful but are poor as the permanent operational center of an automated ML lifecycle.
- Existing platforms such as data clouds, experiment trackers, AutoML tools, notebook systems, model-serving systems, and cloud providers solve pieces of the workflow but typically do not provide the exact integrated developer experience DCLab is targeting.
- [INFERRED] DCLab’s durable value is the orchestration layer that knows what was done, why it was done, what evidence exists, what can safely happen next, and how to recover when something fails.

### 1.6 Fundamental differentiation

**Confirmed or strongly implied differentiators:**

- Notebook-replacement ambition rather than notebook enhancement.
- Chat and click as execution interfaces, not only code authoring helpers.
- Automatic failure diagnosis and recovery as a normal workflow behavior.
- Scientific experiment state as first-class product data.
- Human review where consequences matter.
- A common ML core reusable by both personal developers and business/team accounts.
- Hierarchical use of heuristics, statistics/ML, and LLM reasoning, rather than sending everything to an LLM.
- Long-term expansion from ML engineering into decision intelligence.
- Future agentic deployment/MLOps behavior.
- A platform-level memory of datasets, experiments, features, models, decisions, and outcomes.

### 1.7 Intended long-term impact

The long-term ambition is to make serious ML development feel less like a collection of disconnected tools and more like operating a single intelligent engineering environment. The system should eventually be able to:

- understand data sources;
- recommend and construct valid modeling workflows;
- run large experiment sets;
- learn from past project evidence;
- automate routine debugging;
- compare conventional ML and newer architectures;
- produce deployment-ready systems;
- observe production outcomes;
- feed operational evidence back into future decisions;
- eventually support richer decision-intelligence workflows beyond model building alone.

### 1.8 What DCLab must never become

**Confirmed/strongly implied:**

DCLab must not become:

- a thin ChatGPT wrapper for data science;
- a notebook clone with a prettier interface;
- an opaque “AutoML says this model is best” black box;
- a system that silently changes data, features, models, or deployments without traceability;
- a platform that hides failed or negative experiments;
- an LLM-first data-processing engine that burns cost on work deterministic systems can perform;
- a product where business/team features fork the scientific core into a separate code path;
- an R&D repository presented as if it were the product;
- a UI redesign that breaks existing functionality or disconnects working backend endpoints.

---

## 2. Founder Vision and Product Philosophy

### 2.1 Founder’s underlying vision

The consistent direction across the DCLab conversations is to **compress the full machine-learning workflow into an environment that behaves more like an intelligent engineering partner than a passive tool**.

The founder repeatedly returns to several ideas:

- remove notebooks as the primary interaction model;
- preserve serious ML engineering, not dumb it down;
- let developers work through chat or controlled UI actions;
- automate repetitive technical loops;
- preserve the ability to inspect what the system did;
- use many models, many features, and multiple downstream intelligence layers when the problem warrants it;
- eventually create systems that predict not just an event but also recommendations, action outcomes, and human/customer reactions;
- enable agents to act, debug, and deploy while still respecting permissions and guardrails;
- keep developers as the first and core user class even as business/team features are added.

### 2.2 Beliefs about AI and ML engineering

**Confirmed:**

- AI should remove repetitive coordination overhead, not merely generate code.
- ML work remains an engineering and scientific process; it cannot be reduced to prompting.
- Strong conventional ML baselines remain necessary even when exploring transformers or foundation-model approaches.
- Better model results require disciplined work across feature engineering, validation, hyperparameters, leakage prevention, and comparisons.
- Expensive AI should be used selectively.
- Systematic execution and evidence are more important than impressive demos.

### 2.3 Disruption thesis

The disruption thesis has evolved from “automate ML” into a more concrete idea:

> The system should own the loop from user intent to executed, debugged, evidenced result.

A notebook workflow often looks like:

1. write/run cell;
2. receive error;
3. copy error;
4. ask an AI;
5. receive revised code;
6. paste code;
7. rerun;
8. repeat;
9. later try to reconstruct what happened.

DCLab’s intended loop is:

1. user defines intent or approves a plan;
2. DCLab executes;
3. DCLab captures logs and state;
4. if execution fails, DCLab diagnoses the failure;
5. it proposes or applies an allowed repair;
6. it retries;
7. the full change and evidence trail is preserved;
8. the user is interrupted only when approval or judgment is necessary.

### 2.4 Role of human expertise

**Confirmed:** DCLab is not intended to remove human expertise from ML. It is intended to move human effort upward.

Human experts should remain responsible for:

- framing the real problem;
- validating business meaning;
- approving sensitive transformations;
- deciding whether proxy variables are acceptable;
- interpreting ambiguous or high-impact results;
- deciding production eligibility when evidence is insufficient;
- defining operational risk and constraints;
- overriding or correcting agent assumptions;
- reviewing major deployment changes.

Agents and automation should handle repeatable mechanics wherever possible.

### 2.5 Role of autonomous agents

Agents are expected to become execution-capable, not merely advisory. The strongest confirmed examples are:

- debugging failed ML workflow steps;
- rerunning corrected steps;
- later handling deployment/MLOps workflows;
- analyzing logs;
- using tools under explicit permissions;
- operating with guardrails.

[UNCERTAIN] A fully specified multi-agent topology has not been recovered from the available context. Planner/executor/reviewer separation is a reasonable architecture but must remain a proposal until confirmed.

### 2.6 Automation, control, trust, and explainability

The DCLab philosophy is not “maximum autonomy at any cost.” It is **maximum safe automation with recoverable, inspectable execution**.

The balance implied by prior decisions is:

- deterministic work: automate aggressively;
- reversible low-risk actions: allow agent execution;
- expensive or destructive actions: require explicit policy and, where appropriate, approval;
- scientific decisions: preserve rationale and evidence;
- production promotion: require stronger gates than experimentation;
- uncertain transformations: surface uncertainty rather than silently inventing structure;
- agent actions: log what tool was used, what changed, and why.

### 2.7 Product principles

1. **Developer-first.**
2. **Workflow ownership over code generation.**
3. **Evidence before claims.**
4. **Automation must be inspectable.**
5. **Failures are first-class events, not hidden noise.**
6. **One common ML core across account types.**
7. **Chat is an interface, not the whole product.**
8. **Use the cheapest reliable method first.**
9. **A model is not a result without context, data lineage, and evaluation evidence.**
10. **The product should reduce tool-switching rather than adding another disconnected surface.**

### 2.8 Scientific principles

**Confirmed or strongly implied:**

- always keep meaningful baselines;
- compare model families fairly;
- avoid leakage;
- preserve split strategy;
- preserve experiment parameters;
- keep negative results;
- do not treat one metric as the entire truth;
- include task-appropriate metrics;
- make feature engineering explicit;
- support repeated optimization without erasing earlier evidence;
- promote models on evidence, not novelty.

### 2.9 Engineering principles

- Core workflows should be modular and reusable.
- Business features should extend the same core, not fork it.
- Existing backend functionality must not be broken by UI changes.
- Missing backend functionality may be represented in the frontend, but should not be falsely wired.
- Large data ingestion should use batch/streaming/scalable processing patterns rather than naive whole-dataset LLM processing.
- Use databases and object stores according to data type and access pattern.
- [INFERRED] Execution must be stateful, resumable, and job-oriented rather than tied to a single web request.

### 2.10 UX principles

- Reduce cognitive load.
- Replace excessive top-tab navigation with a clearer left-side information architecture.
- Preserve the current DCLab visual theme/colors that the founder already likes.
- Use a modern “liquid glass” side-navigation treatment where appropriate.
- The user should always understand:
  - what is running;
  - what has failed;
  - what changed;
  - what requires approval;
  - what evidence supports the current model or recommendation;
  - where the workflow is in the lifecycle.

### 2.11 Non-negotiable values

**Confirmed or strongly implied:**

- reproducibility;
- traceability;
- developer control;
- technical seriousness;
- scientific honesty;
- preservation of working functionality during redesign;
- cost-aware AI usage;
- model-family neutrality;
- separation of R&D experiments from product truth.

---

## 3. Problems DCLab Solves

### 3.1 Notebook fragmentation and hidden state

**Who experiences it:** Data scientists, ML engineers, AI engineers, researchers.

**Current workaround:** Jupyter/Colab notebooks, scripts, manual comments, experiment trackers, ad hoc naming.

**Why it fails:**
- cell order can matter;
- runtime state is hidden;
- outputs are not always tied cleanly to code/data versions;
- notebooks accumulate exploratory debris;
- handoff and productionization require another workflow;
- debugging is manual and repetitive.

**DCLab solution:** Treat each workflow step as an explicit execution with inputs, outputs, logs, dependencies, versions, and recovery behavior.

**Expected measurable outcome:** Fewer manual reruns, faster failure recovery, lower time-to-reproduce, lower context-switching.

### 3.2 Error-copy/paste loop

**Who:** Practitioners running code interactively.

**Current workaround:** Copy stack trace into an AI assistant, receive a fix, paste it back, rerun manually.

**Why it fails:** Human becomes the message bus between runtime and AI.

**DCLab solution:** The platform captures the failure, provides runtime context to the agent, proposes or applies an allowed repair, and reruns automatically.

**Expected outcome:** Reduced mean time to recovery and fewer manual interventions per failed step.

### 3.3 Fragmented tooling

**Who:** Individual developers and teams.

**Current workaround:** Notebook + Git + object storage + experiment tracker + cloud training + deployment platform + monitoring + docs + chat.

**Why it fails:** Context is duplicated and decisions are lost between systems.

**DCLab solution:** Provide a unified workflow and metadata layer even when underlying infrastructure remains external.

**Expected outcome:** Fewer tool transitions and stronger lineage.

### 3.4 Raw and heterogeneous data onboarding

**Who:** Developers and businesses connecting logs, CRM data, files, and other sources.

**Current workaround:** Custom ETL scripts, manual schema assumptions, expensive LLM parsing.

**Why it fails:** Large volumes make LLM-everywhere economically irrational; source formats vary; semantic meaning is often undocumented.

**DCLab solution:** Layered data understanding:
1. structural parsing;
2. pattern/heuristic engine;
3. statistical/ML inference;
4. LLM only where semantic ambiguity remains;
5. canonical semantic data model.

**Expected outcome:** Lower preprocessing cost, faster ingestion, more reliable schema understanding.

### 3.5 Model-selection and benchmark ambiguity

**Who:** ML practitioners and reviewers.

**Current workaround:** Manually compare a few metrics across notebooks.

**Why it fails:** Different splits, preprocessing, tuning budgets, and metric choices make comparisons misleading.

**DCLab solution:** Standardized experiment definitions, comparable evaluation evidence, baseline preservation, model-family comparisons.

**Expected outcome:** More defensible promotion decisions.

### 3.6 Feature-engineering knowledge loss

**Who:** Data scientists and teams.

**Current workaround:** Feature code scattered in notebook cells or pipelines.

**Why it fails:** Rationale and lineage disappear; features are hard to reuse.

**DCLab solution:** Treat features and feature transformations as versioned entities with source columns, logic, rationale, and experiment impact.

### 3.7 Poor reproducibility

**Who:** Developers, teams, reviewers, future maintainers.

**Current workaround:** requirements files, notebook exports, experiment trackers.

**Why it fails:** Often incomplete linkage among code, data, environment, seed, parameters, artifacts, and decisions.

**DCLab solution:** Preserve a run manifest with data version, code/workflow version, environment, parameters, random seeds where applicable, metrics, logs, model artifact, and lineage.

### 3.8 Deployment handoff

**Who:** ML engineers and DevOps/MLOps engineers.

**Current workaround:** Manual Docker/cloud config, scripts, CI/CD, debugging across vendor consoles.

**Why it fails:** Training context is separated from deployment context; errors require specialist intervention.

**DCLab solution:** [Future/Confirmed direction] agentic deployment/MLOps workflow with permissions, debugging, logs, rollback, and guardrails.

### 3.9 Collaboration and role boundaries

**Who:** Businesses with multiple ML/data users; DCLab internal operators.

**Current workaround:** Generic user accounts or loosely enforced roles.

**Why it fails:** Different users need different visibility, controls, and review authority.

**DCLab solution:** Separate account/role capabilities while keeping a common technical core.

### 3.10 Knowledge loss across projects

**Who:** Teams and repeat users.

**Current workaround:** docs, chat history, scattered experiment logs.

**Why it fails:** The reasoning behind prior decisions is difficult to retrieve.

**DCLab solution:** Persistent project memory, structured decision records, searchable experiments, and eventually cross-project learning with privacy boundaries.

### 3.11 Overuse of LLMs

**Who:** Platform operator and end customer through cost/latency.

**Current workaround:** Send raw data or every problem to a general model.

**Why it fails:** High cost, latency, privacy exposure, nondeterminism.

**DCLab solution:** Hierarchical processing with rules/heuristics/statistics/ML first, LLM where language/semantic reasoning adds value.

### 3.12 “AI can already do this” product risk

**Who:** DCLab as a company.

**Current workaround:** Compete on generated code or model advice.

**Why it fails:** General AI systems can increasingly write, run, and analyze ML code.

**DCLab solution:** Own structured execution, state, evidence, permissions, reproducibility, deployment context, project memory, and integrated lifecycle.

---

## 4. Target Users and Jobs to Be Done

### 4.1 Primary persona: Developer / ML engineer / AI engineer

**Status:** Confirmed core user.

**Goals**
- build valid ML systems faster;
- avoid repetitive setup;
- compare approaches;
- debug quickly;
- move from data to deployment without stitching many tools together.

**Current workflow**
- local IDE/notebook;
- Git;
- data files or warehouse;
- LLM chat;
- training scripts;
- experiment logging;
- cloud deployment.

**Frustrations**
- repeated boilerplate;
- notebook chaos;
- environment failures;
- manual debugging;
- forgotten experiment details;
- deployment friction.

**Required capabilities**
- project creation;
- data upload/connect;
- execution environment;
- EDA;
- feature work;
- model training/tuning;
- evaluation;
- model comparison;
- logs;
- artifact storage;
- reproducibility;
- agent assistance;
- deployment path.

**Trust requirements**
- never silently alter data;
- show logs;
- show parameters;
- allow manual intervention;
- preserve code/workflow artifact;
- reproducible results.

**Technical skill**
- Medium to high.

**Purchase/adoption motivation**
- time saved per project;
- fewer failures;
- faster experimentation;
- reduced infra/tooling overhead.

**Job to be done**
> “Take my data and objective, help me design and execute a serious ML workflow, let me inspect important choices, recover from failures automatically, and give me reproducible evidence I can deploy.”

**Aha moment**
- The first time a failed step is automatically diagnosed, repaired, rerun, and documented without the user manually copying a traceback.
- Or the first time the user receives a clean, comparable experiment/model evidence view without assembling it manually.

### 4.2 Data scientist

**Status:** Confirmed/strongly implied.

Emphasis:
- EDA;
- feature engineering;
- experiment design;
- metric interpretation;
- comparison;
- reproducibility.

Aha moment:
- moving from exploration to a reusable workflow without rewriting everything for production.

### 4.3 Researcher

**Status:** Supporting persona, especially via R&D layer.

Needs:
- model-family experimentation;
- repeatable baselines;
- controlled comparisons;
- negative-result preservation;
- novel architectures.

DCLab must not force research into production assumptions too early.

### 4.4 Business account administrator

**Status:** Confirmed.

This is the administrative owner of a business/team account.

Needs:
- manage users;
- control project/workspace access;
- monitor activity;
- review usage/cost;
- see project status;
- enforce business-level policy.

### 4.5 Business technical users

**Status:** Confirmed.

Prior direction: a business account may include multiple ML engineers/data scientists; an earlier concrete example used **up to five** users.

[UNCERTAIN] “Five users” appears to have been a plan hypothesis and should not be treated as a permanent product limit without founder confirmation.

### 4.6 DCLab admin

**Status:** Confirmed.

Founder/internal platform operator.

Needs:
- visibility into the whole system;
- inspect model-development details;
- inspect feature engineering;
- inspect evaluation quality;
- monitor execution;
- verify that workflows are technically correct;
- manage platform operations.

### 4.7 DCLab developer/employee

**Status:** Confirmed as an internal role concept.

Initial direction: similar visibility to DCLab admin with restrictions, and at one point **read-only for now**.

### 4.8 Analysts

**Status:** [UNCERTAIN]

Analysts are a plausible future user class but not sufficiently grounded as an explicit core persona in the available context.

### 4.9 Product teams / technical founders

**Status:** [INFERRED]

They are natural adopters of developer-oriented ML workflow tooling but are not as strongly specified as the developer persona.

### 4.10 Students / independent builders

**Status:** [UNCERTAIN]

No strong evidence that students are a strategic persona. Independent developers fit the “Personal Development” account concept more clearly.

---

## 5. Canonical DCLab User Journey

The journey below separates recovered product direction from proposed completeness. Stages that are necessary for an end-to-end ML lifecycle but were not explicitly described in available detail are marked accordingly.

### 5.1 Create project

**Status:** [INFERRED], strongly required by all other design.

**User action**
- create/select workspace/project;
- set problem name and goal.

**System action**
- create project identity and storage namespace;
- attach ownership/account/permissions.

**Agent action**
- optionally ask structured questions to clarify objective.

**Inputs**
- project name;
- account/workspace;
- optional task description.

**Outputs**
- project shell;
- initial problem statement.

**Stored artifacts**
- project metadata;
- initial decision/context record.

**Approvals**
- none for low-risk creation.

**Failure states**
- invalid permissions;
- storage/database initialization failure.

**Recovery**
- retry idempotently; do not create duplicated project state.

**Auditability**
- creator, timestamp, account, initial configuration.

### 5.2 Define the ML problem

**Status:** Confirmed in principle.

**User action**
- describe prediction/analysis objective through chat or structured UI.

**System**
- convert objective into task definition.

**Agent**
- identify task type, target, unit of prediction, horizon, constraints, and initial metrics;
- surface ambiguities.

**Inputs**
- domain objective;
- dataset/context if already connected.

**Outputs**
- structured problem specification.

**Approvals**
- user approval before training if target/task definition is ambiguous or consequential.

**Stored**
- problem definition versions;
- rationale.

**Failure**
- ambiguous target, impossible labels, conflicting task definition.

**Recovery**
- request clarification only where necessary; preserve unresolved assumptions.

### 5.3 Upload or connect data

**Status:** Confirmed.

**User**
- upload file(s) or connect external source.

**System**
- stage data in object storage;
- create dataset/source record;
- profile size/format;
- choose processing strategy.

**Agent**
- only where needed, interpret unfamiliar semantics.

**Inputs**
- CSV/files/logs/CRM/external sources.

**Outputs**
- dataset version;
- ingestion report.

**Stored**
- original immutable raw object;
- metadata;
- checksums;
- source configuration.

**Approvals**
- credentials/connector authorization;
- destructive source actions should not occur during ingestion.

**Failure**
- unsupported format;
- schema inconsistency;
- malformed records;
- volume exceeds local processing;
- connector auth failure.

**Recovery**
- chunking/batching;
- quarantine invalid records;
- resumable ingestion;
- user-visible errors.

**Audit**
- source, checksum, row/record counts, ingestion transform history.

### 5.4 Data validation

**Status:** Confirmed/strongly implied.

**System**
- detect schema types;
- missingness;
- duplicates;
- invalid values;
- target leakage candidates;
- class imbalance;
- suspicious identifiers;
- time/order constraints.

**Agent**
- explain anomalies and propose remedies.

**Outputs**
- validation report;
- issue list.

**Approval**
- required for semantic transformations that can alter meaning.

### 5.5 Dataset understanding / semantic modeling

**Status:** Confirmed via data-layer conversations.

Use layered reasoning:

1. syntax and file structure;
2. patterns and heuristics;
3. statistical/ML inference;
4. LLM for ambiguous semantics;
5. semantic data model.

Outputs:
- field roles;
- candidate entity keys;
- target candidates;
- relationships;
- semantic labels;
- confidence.

### 5.6 EDA

**Status:** Confirmed as part of developer workflow.

Outputs:
- distributions;
- correlations/associations;
- target relationships;
- imbalance;
- drift indicators where relevant;
- data-quality warnings.

**Important:** EDA should be reproducible, not just ephemeral charts.

### 5.7 Leakage and quality checks

**Status:** Confirmed scientific requirement.

The system must check for:
- direct target leakage;
- post-outcome variables;
- train/test contamination;
- temporal leakage;
- group leakage;
- duplicated entities across splits.

**Agent**
- may flag and explain;
- must not silently remove potentially meaningful variables without trace.

### 5.8 Experiment planning

**Status:** Confirmed in direction; exact UI/agent architecture undecided.

Plan should identify:
- baseline(s);
- candidate model families;
- preprocessing;
- feature sets;
- split strategy;
- metrics;
- tuning budget;
- compute requirements.

**Approval**
- user should be able to inspect/adjust plan before expensive runs.

### 5.9 Feature engineering

**Status:** Confirmed and important.

System should support:
- standard transformations;
- categorical handling;
- numerical transformations;
- missing-value strategies;
- domain-derived features;
- meaningful interaction features;
- feature selection;
- feature lineage.

**Agent**
- propose features with rationale;
- track whether feature helped.

### 5.10 Algorithm selection

**Status:** Confirmed.

DCLab should compare appropriate model families and preserve conventional baselines even while exploring transformer/foundation-model approaches.

### 5.11 Training

**Status:** Confirmed.

Each training run should produce:
- environment;
- model family;
- parameters;
- code/workflow version;
- dataset version;
- feature set;
- metrics;
- logs;
- model artifact;
- timing/cost where available.

### 5.12 Optimization

**Status:** Confirmed.

Includes:
- hyperparameter tuning;
- feature-set refinement;
- class-weighting/resampling where appropriate;
- early stopping;
- model-specific optimization;
- search-budget tracking.

**Non-goal:** Endless optimization without clear evidence or budget.

### 5.13 Evaluation

**Status:** Confirmed.

Examples from prior classification work included:
- ROC AUC;
- PR AUC;
- accuracy;
- precision;
- recall;
- F1;
- log loss;
- Brier score.

Task-specific metrics must vary by problem.

Evaluation must preserve:
- exact test set definition;
- threshold;
- calibration where relevant;
- confusion/error analysis;
- confidence/variance where feasible.

### 5.14 Evidence comparison

**Status:** Confirmed/strongly implied.

Users need a comparison surface that answers:
- Which runs are comparable?
- Which data/feature changes differ?
- Which metrics improved?
- What trade-offs changed?
- Is improvement robust?
- What failed?
- What is the evidence for promotion?

### 5.15 Human review and approvals

**Status:** Confirmed principle.

Human review points should include at least:
- ambiguous target/problem definition;
- potentially destructive data changes;
- sensitive feature usage;
- expensive jobs above policy threshold;
- production deployment/promotion;
- rollback-sensitive changes.

Exact policy engine remains [UNDECIDED].

### 5.16 Reproducibility

**Status:** Confirmed.

A run should be reproducible from recorded artifacts where infrastructure permits.

### 5.17 Deployment

**Status:** Future core direction.

DCLab aims to make deployment more autonomous over time, including:
- package/build;
- environment configuration;
- infrastructure target;
- deployment;
- health checks;
- error diagnosis;
- repair/retry;
- rollback.

The specific cloud provider strategy is [UNDECIDED].

### 5.18 Monitoring

**Status:** [INFERRED]/future.

Required if DCLab owns deployment lifecycle:
- service health;
- latency;
- error rates;
- prediction distribution;
- data drift;
- model performance where labels arrive;
- cost.

No complete monitoring architecture is recovered.

### 5.19 Continuous learning

**Status:** Future vision / [INFERRED].

DCLab should be able to use new evidence and production outcomes to trigger evaluation or retraining workflows, but autonomous retraining/promotion rules remain [UNDECIDED].

### 5.20 Knowledge capture

**Status:** Strongly implied.

Capture:
- successful and failed experiments;
- decisions;
- feature rationales;
- deployment incidents;
- model promotion reasons;
- user/agent corrections.

### 5.21 Reuse of previous experiments

**Status:** Strongly implied and long-term important.

The system should retrieve relevant prior evidence rather than restart reasoning from zero.

Privacy and tenant isolation must constrain reuse.

---

## 6. Core Product Capabilities

### 6.1 Project and workspace management

**Classification:** Essential product core.

**Purpose:** Provide the durable container for data, experiments, execution, models, deployments, and knowledge.

**User value:** Persistent context and collaboration.

**Key entities:** Account, Workspace, Project, Membership, Role.

**Current status:** Concept/design inferred; exact implementation unknown.

**Non-goal:** A generic project-management suite.

### 6.2 Data ingestion and semantic understanding

**Classification:** Essential core.

**Inputs:** Files, logs, CRM data, connected data sources.

**Outputs:** Dataset versions, schema, quality profile, semantic model.

**Automation:** layered heuristic/statistical/LLM processing.

**Current status:** Architecture direction discussed; implementation status unverified.

**Non-goal:** Rebuild Snowflake/Databricks end-to-end.

### 6.3 Data validation and quality

**Classification:** Essential core.

Outputs include:
- schema checks;
- missingness;
- anomalies;
- leakage risks;
- duplicates;
- invalid categories;
- imbalance.

### 6.4 EDA and dataset understanding

**Classification:** Essential core.

Must produce reproducible analysis tied to dataset version.

### 6.5 Experiment planner

**Classification:** Essential core, agent-assisted.

Purpose:
- turn problem statement into explicit experiment plan.

Human control:
- inspect and adjust before execution.

### 6.6 Feature engineering system

**Classification:** Essential core.

Must track:
- source features;
- generated features;
- transformation logic;
- version;
- experiment usage;
- observed impact.

### 6.7 Model training runtime

**Classification:** Essential core.

Must support:
- multiple model families;
- resource-aware execution;
- artifact persistence;
- logs and status;
- failure handling.

### 6.8 Hyperparameter and optimization service

**Classification:** Essential core/supporting execution capability.

Must have explicit budgets and comparable results.

### 6.9 Experiment tracking and evidence

**Classification:** Essential core.

This is not optional telemetry. It is central product state.

### 6.10 Model comparison

**Classification:** Essential core.

Compare:
- metrics;
- data version;
- feature set;
- run configuration;
- resource cost;
- errors;
- calibration;
- robustness where available.

### 6.11 Agentic debugging and recovery

**Classification:** Essential differentiator.

Purpose:
- eliminate manual error-transfer loops.

Inputs:
- step definition;
- code;
- logs;
- environment metadata;
- previous attempts.

Outputs:
- diagnosis;
- proposed fix;
- patch/change;
- retry;
- final result;
- audit record.

### 6.12 Chat/intent interface

**Classification:** Core interface, not the product itself.

Must be able to:
- describe goals;
- inspect current state;
- request actions;
- explain results;
- request comparisons.

### 6.13 Workflow/pipeline view

**Classification:** Essential UX/core.

Purpose:
- show dependency graph and state of ML lifecycle.

### 6.14 Artifact management

**Classification:** Supporting platform capability.

Objects:
- raw data;
- processed data;
- plots;
- model files;
- reports;
- logs;
- manifests;
- deployment packages.

### 6.15 Deployment and MLOps agent

**Classification:** Future core / experimental until proven.

Purpose:
- own deployment workflow and debug failures.

Non-goal:
- unrestricted infrastructure mutation.

### 6.16 Monitoring and production feedback

**Classification:** Future core/supporting platform capability.

### 6.17 Business/team administration

**Classification:** Supporting commercial capability.

Includes:
- seats;
- roles;
- workspace access;
- project visibility;
- policy;
- usage.

### 6.18 DCLab internal admin

**Classification:** Supporting operational capability.

Must allow deep inspection of ML workflow details.

### 6.19 R&D experimentation subsystem

**Classification:** Supporting subsystem, not product core.

Use:
- compare conventional ML and transformer approaches;
- test new model types;
- generate evidence before product integration.

Must not:
- define the whole DCLab architecture;
- leak experimental assumptions into production by default.

### 6.20 Decision intelligence layers

**Classification:** Future vision / experimental.

Discussed direction:
- prediction layer;
- recommendation layer;
- action-outcome prediction;
- reaction prediction;
- aggregation of evidence across many models/features.

This is not the initial MVP core.

### 6.21 Prediction-driven synthetic customer / lead generation

**Classification:** Future application module.

Potential flow:
- learn patterns from historical/predicted behavior;
- generate synthetic customer archetypes;
- use archetypes to identify real-world leads.

This is clearly downstream of the ML platform and should not dominate core architecture.

---

## 7. Intelligence and Agent System

### 7.1 Confirmed agent responsibilities

The available context supports the following agent roles conceptually, though not necessarily as separate processes:

- **Workflow reasoning:** interpret the user’s goal and help construct the ML workflow.
- **Execution support:** trigger allowed steps/tools.
- **Debugging:** analyze runtime errors, identify likely cause, repair or propose repair, rerun.
- **Data semantic reasoning:** resolve ambiguous columns/entities only after cheaper structural/statistical methods.
- **Modeling assistant:** recommend baselines, candidate algorithms, feature-engineering directions, and evaluation approaches.
- **Deployment/MLOps agent:** future agent that handles deployment workflows and debugging under guardrails.

### 7.2 [PROPOSED] Planner / executor / reviewer relationship

No authoritative prior topology was recovered. A safe architecture is:

- **Planner:** creates a structured plan, never directly mutates production.
- **Executor:** uses approved tools within scoped permissions.
- **Reviewer/Verifier:** checks outputs, tests, metrics, and policy.
- **Human approver:** required for governed transitions.

This is a recommendation, not a recovered decision.

### 7.3 Delegation model

**Confirmed principle:** Agents must be given bounded authority.

Suggested authority classes:

- Read-only inspection.
- Reversible experiment action.
- Cost-bearing execution.
- Data mutation.
- Deployment mutation.
- Destructive production action.

Exact classes are [PROPOSED], but the need for permissions/guardrails is confirmed.

### 7.4 Context and memory

Agents need access to project-scoped structured context:
- project objective;
- current dataset versions;
- workflow graph;
- experiment history;
- feature catalog;
- model/run evidence;
- logs;
- current errors;
- user approvals;
- environment state.

**[INFERRED]** Agent memory should primarily come from structured platform state, not from uncontrolled conversational memory.

### 7.5 Tool use

Expected tool categories:
- dataset readers/profilers;
- code execution;
- feature pipeline execution;
- training jobs;
- metric evaluation;
- artifact storage;
- Git/version control;
- deployment APIs;
- cloud/job APIs;
- logs/observability;
- secrets references.

### 7.6 Human-in-the-loop

Human approval is required or strongly advisable when:
- semantic ambiguity changes the modeling target;
- a transformation drops or overwrites data;
- an action creates meaningful cloud spend;
- a model is promoted to production;
- production traffic or infrastructure is modified;
- a rollback or migration may lose data;
- sensitive features are used.

### 7.7 Failure handling

Confirmed desired behavior:
1. capture failure;
2. gather relevant context;
3. diagnose;
4. identify permitted repair;
5. apply repair or ask approval;
6. retry;
7. validate;
8. record failure and fix.

The system must avoid infinite retry loops.

### 7.8 Agent observability

**[INFERRED] required by the vision:**
- agent action log;
- tool call log;
- inputs/outputs summaries;
- patch/change record;
- cost;
- latency;
- retry count;
- confidence/rationale;
- approval decision;
- final outcome.

### 7.9 Evaluating agent quality

**[PROPOSED] metrics:**
- successful task completion;
- recovery success rate;
- human-intervention rate;
- false-fix rate;
- regression rate;
- mean repair attempts;
- cost per resolved failure;
- unsafe-action prevention;
- correctness of scientific recommendations.

### 7.10 Learning from previous projects

**Long-term confirmed direction / current mechanism undecided.**

The system should reuse prior evidence but must separate:
- generic learned patterns;
- tenant-private project information;
- reusable templates;
- organization-specific knowledge.

Cross-tenant private evidence must never leak.

### 7.11 Hallucination control

The project direction implies:
- use structured tool outputs;
- prefer deterministic inspection;
- verify changes by execution/tests;
- record uncertainty;
- do not accept an agent’s natural-language claim as proof that a model or deployment succeeded.

### 7.12 Autonomous decisions allowed

**Confirmed principle, exact list [UNDECIDED].**

Likely safe:
- read logs;
- run non-destructive diagnostics;
- retry known-idempotent failed steps;
- generate comparison reports.

Likely governed:
- feature removal;
- target changes;
- expensive jobs;
- deployment;
- rollback;
- secrets/access changes.

---

## 8. Scientific and Experimentation Core

### 8.1 Hypothesis creation

Experiments should have an explicit reason:
- new feature set;
- different model family;
- tuning change;
- data-cleaning change;
- class-imbalance treatment;
- architecture change.

A run without a hypothesis may still be exploratory, but it should be labeled accordingly.

### 8.2 Experiment planning

A plan should define:
- dataset version;
- target;
- split;
- preprocessing;
- feature set;
- baseline;
- candidate models;
- metrics;
- tuning budget;
- stopping criteria.

### 8.3 Baselines

**Confirmed:** Conventional baselines are mandatory for meaningful comparison, especially when testing transformer-style models.

### 8.4 Dataset splitting

The split strategy must be explicit and appropriate:
- random;
- stratified;
- time-based;
- group-based;
- nested CV where needed.

[PROPOSED] DCLab should store split indices or deterministic split instructions where practical.

### 8.5 Leakage prevention

Leakage must be checked before model promotion.

### 8.6 Feature engineering

Feature engineering is not a black box. Each generated feature should ideally retain:
- derivation;
- source fields;
- semantic rationale;
- version;
- experiment usage.

### 8.7 Algorithm selection

The system should support comparisons across:
- classical linear models;
- tree ensembles;
- boosting;
- neural models;
- transformer/foundation-style methods where applicable.

The exact supported library set is not recovered.

### 8.8 Hyperparameter optimization

Optimization must be:
- budgeted;
- logged;
- reproducible;
- comparable.

### 8.9 Benchmarking

Benchmarks should use common:
- data split;
- evaluation protocol;
- metrics;
- resource accounting.

### 8.10 Statistical validation

**[PROPOSED but strongly aligned]:**
- repeated CV where relevant;
- bootstrap intervals;
- paired tests where valid;
- variance estimates.

No exact statistical policy was recovered.

### 8.11 Confidence and uncertainty

Known classification discussions included calibration-sensitive metrics such as Brier score and log loss. This implies that probability quality matters, not only ranking metrics.

### 8.12 Repeated experiments

The platform should preserve all meaningful runs rather than overwrite “the model.”

### 8.13 Evidence aggregation

Long-term DCLab vision includes aggregating outputs from many models/features and using them to form higher-level conclusions.

This must not be interpreted as blindly averaging hundreds of models. The specific aggregation method is [UNDECIDED].

### 8.14 Reproducibility

Core reproducibility bundle should include:
- dataset version/hash;
- split;
- feature-set version;
- model config;
- training code/workflow version;
- package/runtime environment;
- random seeds where applicable;
- metrics;
- logs;
- model artifact.

### 8.15 Model promotion criteria

**[UNDECIDED] exact thresholds.**

Confirmed principle:
- novelty is not a promotion criterion;
- evidence and operational suitability matter.

### 8.16 Experiment lineage

Must connect:
Dataset → Dataset Version → Feature Set → Experiment → Run → Model Artifact → Evaluation → Promotion/Deployment.

### 8.17 Negative-result preservation

**Strongly implied scientific requirement:** Failed or worse experiments are valuable knowledge and should not disappear.

### 8.18 Knowledge extraction

DCLab should capture:
- what changed;
- what improved;
- what degraded;
- why a hypothesis was rejected;
- what failed operationally.

### 8.19 Decision records

Model and deployment decisions should have explicit records:
- decision;
- evidence;
- approver;
- date;
- alternatives;
- rollback/revisit condition.

### 8.20 Relationship to the R&D repository

**Confirmed separation:**

The R&D repository is a supporting environment for:
- trying model families;
- validating research ideas;
- comparing baselines with transformers;
- testing feature engineering;
- exploring Relational Transformer approaches;
- generating evidence for future product integration.

It is **not**:
- the product architecture;
- the production backend;
- the main user experience;
- the canonical source of business roles;
- the entire DCLab roadmap.

A successful R&D result should enter the main product only after:
1. reproducible evidence;
2. defined user value;
3. production integration design;
4. security/cost review;
5. stable interfaces.

---

## 9. Data and Knowledge Architecture

### 9.1 Dataset lifecycle

Suggested canonical lifecycle consistent with prior discussions:

`registered → ingesting → profiled → validated → ready → superseded/archived`

[PROPOSED] status names; lifecycle concept is confirmed.

### 9.2 Dataset versions

Each material change should create a new version rather than silently mutating the previous analytical dataset.

### 9.3 Schema and validation

Store:
- physical schema;
- inferred types;
- semantic types;
- constraints;
- validation findings;
- confidence.

### 9.4 Data lineage

Lineage should connect:
- source connector/file;
- raw object;
- normalized dataset;
- transformation;
- derived dataset;
- feature set;
- experiment.

### 9.5 Feature lineage

Connect each feature to:
- source fields;
- transformation code/spec;
- version;
- producing step;
- consuming experiment.

### 9.6 Experiment lineage

Connect:
- hypothesis;
- dataset version;
- split;
- feature set;
- model configuration;
- run;
- metrics;
- artifact;
- decision.

### 9.7 Model lineage

Connect:
- training run;
- code/environment;
- parent experiment;
- model artifact;
- evaluation;
- deployment versions.

### 9.8 Artifact lineage

Artifacts should have:
- producer job/run;
- checksum;
- storage URI;
- MIME/type;
- metadata;
- lifecycle;
- owner.

### 9.9 Knowledge graph / research memory

**[INFERRED]** The long-term need is clear, but an actual graph database decision is not recovered.

A logical knowledge graph exists even if implemented relationally:
- project has datasets;
- datasets have versions;
- versions have features;
- experiments use features;
- runs produce models;
- models produce evaluations;
- decisions select/promote models;
- deployments serve model versions;
- incidents reference deployments/runs.

### 9.10 Metadata

PostgreSQL is the stronger fit for core transactional/project metadata based on earlier SQL-vs-Mongo discussion direction.

**[UNCERTAIN]** No final database ADR was recovered, but the platform’s relational entities, constraints, auditability, and transactions strongly favor PostgreSQL.

### 9.11 Search and retrieval

Need search across:
- project names;
- datasets;
- columns/features;
- experiments;
- models;
- errors;
- decisions;
- deployments.

Semantic retrieval may be layered on later.

### 9.12 Provenance

Every derived artifact should be traceable to its origin.

### 9.13 Audit history

Audit at least:
- user actions;
- agent actions;
- role changes;
- data mutations;
- model promotions;
- deployments;
- destructive actions.

### 9.14 Project memory

Project memory should be durable structured state, with conversational summaries only as a convenience.

### 9.15 Cross-project learning

**Future:** Reuse templates and evidence.

Must respect:
- tenant boundaries;
- privacy;
- permission;
- opt-in policy if private data influences generalized learning.

### 9.16 Privacy boundaries

Business data must be isolated by tenant/account/workspace.

### 9.17 Tenant isolation

**Confirmed need** from account model.

Implementation method [UNDECIDED]:
- shared DB with tenant keys/RLS;
- schema-per-tenant;
- database-per-tenant for selected plans.

### 9.18 Retention and deletion

**[UNDECIDED].**

Must eventually define:
- raw data retention;
- experiment artifact retention;
- logs;
- backups;
- deletion guarantees;
- legal/compliance exceptions.

---

## 10. Technical Architecture

This section distinguishes recovered direction from undecided implementation.

### 10.1 Frontend

**Known:**
- main product already has dashboard/lab/workflow UI concepts;
- founder dislikes excessive top-tab navigation;
- redesign direction is a left-side navigation with liquid-glass treatment;
- existing colors/theme should be retained;
- functionality and backend endpoint wiring must remain correct.

**Responsibility:**
- project/workspace navigation;
- pipeline state;
- data and experiment views;
- model comparison;
- agent conversation;
- approvals;
- logs;
- deployment/monitoring.

**Trust boundary:** Frontend must not be authoritative for permissions or execution state.

### 10.2 Backend API

**[INFERRED] essential.**

Responsibilities:
- auth/session validation;
- project/domain state;
- job submission;
- artifact metadata;
- agent orchestration;
- audit;
- policy enforcement.

**Failure behavior:**
- idempotent commands where practical;
- no silent partial mutation;
- return job IDs for long-running work.

### 10.3 Primary database

**Likely direction:** PostgreSQL.

**Why:**
- strongly relational domain;
- transactions;
- constraints;
- RBAC metadata;
- lineage edges;
- audit records;
- queryability.

**[UNDECIDED]** exact schema and use of JSONB.

### 10.4 Object/artifact storage

Earlier discussion compared S3 and Google Cloud Storage and recognized them as analogous object-storage services for DCLab’s usage.

**Responsibility:**
- raw datasets;
- processed datasets;
- model binaries;
- reports;
- logs where appropriate;
- large artifacts.

Provider [UNDECIDED].

### 10.5 Queues and background jobs

**[INFERRED] required.**

Training, ingestion, profiling, tuning, and deployment cannot rely on synchronous web requests.

Need:
- queue/job broker;
- worker orchestration;
- retries;
- leases/timeouts;
- cancellation;
- progress;
- status persistence.

Technology [UNDECIDED].

### 10.6 Agent runtime

**[INFERRED].**

Separate from browser request lifecycle.

Needs:
- tool registry;
- scoped credentials;
- project context;
- policy;
- step limits;
- action logs;
- retry controls.

### 10.7 Execution environments / sandboxes

Core to notebook replacement.

Need:
- isolated environment per project/job/run;
- package management;
- resource limits;
- artifact mounts;
- network policy;
- secrets injection;
- reproducible environment spec.

Exact container/Kubernetes/serverless design [UNDECIDED].

### 10.8 Model-training infrastructure

Must support:
- CPU jobs;
- potentially GPU jobs;
- small local/hosted workloads;
- scalable remote workloads.

Provider and scheduler [UNDECIDED].

### 10.9 Deployment infrastructure

Future MLOps direction.

Needs:
- deployment target abstraction;
- build/package;
- config;
- secrets;
- health verification;
- logs;
- rollback.

### 10.10 Authentication

**Confirmed need, implementation [UNDECIDED].**

Must distinguish:
- DCLab internal users;
- personal/developer accounts;
- business users;
- business admins.

### 10.11 Authorization

Role/policy enforcement must be server-side.

### 10.12 Multi-tenancy

Business account model makes tenant isolation mandatory.

### 10.13 Observability

Need:
- backend logs;
- job logs;
- agent traces;
- training logs;
- deployment logs;
- metrics;
- audit events.

### 10.14 Security

Must address:
- secret storage;
- dependency execution;
- arbitrary user code;
- network egress;
- artifact access;
- tenant boundaries;
- agent tool permissions.

### 10.15 External integrations

Discussed or strongly implied:
- CRM;
- logs/data sources;
- cloud object storage;
- deployment providers;
- GitHub/code sources;
- potentially data platforms rather than reimplementing them.

### 10.16 Local vs cloud execution

**[UNDECIDED].**

The vision supports cloud execution strongly. Local/private execution may matter later, especially for enterprise/privacy, but is not sufficiently confirmed.

### 10.17 Repository boundaries

At minimum distinguish:
- main DCLab product repository;
- R&D/experiments repository.

[PROPOSED] Further boundaries may include infra/deployment and reusable SDK packages, but no authoritative repo map is available.

### 10.18 Component responsibility matrix

| Component | Responsibility | Stored state | Trust boundary | Failure behavior | Scaling concern |
|---|---|---|---|---|---|
| Web frontend | UX and user intent | Minimal client state | Untrusted client | recover state from backend | many concurrent sessions |
| API backend | domain commands/queries | transactional metadata | authoritative app boundary | idempotency, validation | API throughput |
| PostgreSQL | core relational state | projects, runs, roles, lineage | protected data tier | transactions/backups | metadata volume |
| Object store | large immutable/mutable artifacts | datasets, models, files | signed/scoped access | checksum + retry | volume/bandwidth |
| Job system | asynchronous execution | status, attempts | controlled worker boundary | retries/cancel/dead letter | concurrent jobs |
| Execution sandbox | ML/data code | ephemeral runtime + outputs | high-risk code boundary | isolation/kill/recreate | CPU/GPU/memory |
| Agent runtime | reasoning/tool use | trace/context refs | policy-controlled | bounded retries | token/tool cost |
| Deployment adapter | production release | deploy metadata | production boundary | health check + rollback | target variability |
| Observability | traces/logs/metrics | operational telemetry | restricted | durable ingest | log volume |

---

## 11. Canonical Domain Model

The fields below are a canonical working model. Where exact names were not previously specified, they are [PROPOSED] implementation-normalization rather than recovered field names.

| Entity | Definition | Key fields | Relationships | Lifecycle/status | Ownership | Mutability | Audit |
|---|---|---|---|---|---|---|---|
| Account | Top-level customer/internal identity | id, type, name | has workspaces/users | active/suspended | DCLab/customer | controlled | yes |
| User | Human user | id, identity, status | memberships | active/disabled | account | limited | yes |
| Membership | User-to-account/workspace role | user_id, role, scope | user/account/workspace | active/revoked | account | mutable | yes |
| Workspace | Collaboration boundary | id, account_id, name | projects | active/archived | account | mutable | yes |
| Project | Durable ML problem container | id, workspace_id, objective | datasets/experiments/deployments | draft/active/archived | workspace | mutable metadata | yes |
| ProblemSpec | Structured ML objective | target, task_type, unit, horizon, metrics | project | draft/approved/superseded | project | versioned | yes |
| DataSource | Origin of data | type, connector, uri/ref | dataset | connected/error | project | mutable config | yes |
| Dataset | Logical dataset | id, name | versions | active/archived | project | metadata | yes |
| DatasetVersion | Immutable analytical version | schema, hash, row_count, artifact | feature sets/experiments | ingesting/ready/failed | project | immutable after ready | yes |
| SchemaProfile | Structural/semantic schema | fields, types, confidence | dataset version | draft/validated | project | versioned | yes |
| ValidationFinding | Data-quality/leakage issue | severity, rule, evidence | dataset version | open/accepted/fixed | project | append/update status | yes |
| Transformation | Reproducible data operation | spec/code, inputs, outputs | dataset versions/features | draft/validated | project | versioned | yes |
| FeatureDefinition | Single engineered feature | name, source, logic | feature set | active/deprecated | project | versioned | yes |
| FeatureSet | Versioned set of features | id, version, members | experiments | active/superseded | project | immutable version | yes |
| SplitDefinition | Train/val/test logic | type, seed, indices/ref | experiment | active | project | versioned | yes |
| Hypothesis | Reason for experiment | statement, expected effect | experiment | open/accepted/rejected | project | append decisions | yes |
| Experiment | Group of comparable runs | plan, dataset, features, metrics | runs | planned/running/completed | project | controlled | yes |
| Run | One execution attempt | config, status, environment | logs/metrics/artifacts | queued/running/failed/succeeded | project | append-only results | yes |
| MetricResult | Evaluation measurement | name, value, split, threshold | run/model | final/invalidated | project | append/immutable | yes |
| ModelArtifact | Trained model binary/spec | uri, hash, format | run/evaluation/deployment | candidate/promoted/retired | project | immutable artifact | yes |
| Evaluation | Evidence bundle | metrics, diagnostics | model/run | draft/final | project | versioned | yes |
| Comparison | Comparable evidence set | run_ids, criteria | experiments | generated | project | regenerable | log source refs |
| DecisionRecord | Human/system decision with evidence | decision, rationale, approver | any governed entity | active/superseded | project/account | append/versioned | mandatory |
| AgentSession | Agent task context | goal, policy, trace | actions | running/complete/failed | project | append | yes |
| AgentAction | One tool/action step | tool, input refs, result, risk | session | proposed/executed/rejected | project | append-only | mandatory |
| Job | Async unit of work | type, status, attempts | run/action | queued/running/etc. | project | state machine | yes |
| Artifact | Generic stored output | uri, hash, kind, producer | many | active/expired | project | immutable preferred | yes |
| Deployment | Logical serving target | model_id, target, config | releases | active/inactive | project | controlled | mandatory |
| DeploymentRelease | Versioned deployed release | model version, build, status | deployment | pending/healthy/failed/rolled_back | project | append | mandatory |
| Monitor | Production check | metric/rule | deployment | active/paused | project | mutable | yes |
| Incident | Production failure/problem | severity, evidence | deployment | open/resolved | project | append | yes |
| AuditEvent | Security/governance trail | actor, action, object, timestamp | all entities | immutable | platform | immutable | itself |
| KnowledgeRecord | Extracted reusable finding | claim, evidence, scope | project/org | draft/verified | project/org | versioned | yes |

---

## 12. Product UX and Information Architecture

### 12.1 Navigation

**Confirmed redesign direction:**
- move away from many tabs across the top;
- use a left-side navigation;
- preserve existing successful visual theme/colors;
- apply liquid-glass style tastefully;
- do not sacrifice functionality for visual redesign.

### 12.2 Recommended information hierarchy from recovered product structure

[PROPOSED labels; hierarchy follows confirmed modules.]

- Home / Overview
- Projects
- Data
- Workflow
- Experiments
- Models
- Deployments
- Monitoring
- Knowledge
- Team / Settings
- DCLab Admin area for authorized internal users

### 12.3 Project overview

User should understand:
- objective;
- current stage;
- recent runs;
- blockers;
- best current evidence;
- actions waiting for approval;
- active deployment status.

### 12.4 Dataset screen

Should show:
- source;
- version;
- schema;
- quality;
- semantic interpretation;
- lineage;
- validation warnings;
- EDA;
- transformations.

The user should feel: **“I know exactly what data the system thinks it has.”**

### 12.5 Experiment screen

Should show:
- hypothesis;
- plan;
- dataset version;
- split;
- features;
- model candidates;
- runs;
- metrics;
- cost/time;
- failures.

The user should feel: **“These results are comparable and I can see why.”**

### 12.6 Pipeline builder / workflow

Core view for notebook disruption.

Should visualize:
- steps;
- dependencies;
- status;
- inputs/outputs;
- active step;
- errors;
- retries;
- agent repairs;
- approval gates.

### 12.7 Model comparison

Must emphasize evidence, not a single “winner” badge.

Show:
- metrics;
- confidence/variance if available;
- calibration;
- feature set;
- data version;
- training cost;
- inference constraints;
- error patterns.

### 12.8 Agent interaction

Chat should be context-aware and action-oriented.

The UI must distinguish:
- explanation;
- proposed action;
- action awaiting approval;
- running action;
- completed action;
- failed action;
- automatically repaired action.

### 12.9 Approval interfaces

An approval should show:
- what will change;
- why;
- affected resources;
- cost/risk;
- rollback/recovery;
- supporting evidence.

### 12.10 Logs and lineage

Logs should not be buried.

Provide:
- structured run log;
- raw log access;
- failure summary;
- agent diagnosis;
- lineage graph.

### 12.11 Deployment and monitoring

Should show:
- currently active model;
- release history;
- health;
- prediction/service metrics;
- incidents;
- rollback.

### 12.12 Knowledge/research view

Future/advanced view:
- learned findings;
- experiment conclusions;
- rejected hypotheses;
- reusable patterns;
- links to evidence.

### 12.13 Empty states

Empty states should teach the next real action:
- upload data;
- define objective;
- create first experiment.

They should not pretend a project has intelligence before evidence exists.

### 12.14 Loading states

Long jobs must show:
- queued;
- provisioning;
- running;
- progress if measurable;
- current step;
- cancellability.

### 12.15 Error states

Error view must include:
- human-readable summary;
- raw details;
- agent diagnosis status;
- attempted fixes;
- manual options.

### 12.16 Recovery states

After an automatic repair:
- show what changed;
- show whether results differ;
- retain failed attempt;
- never erase evidence of the failure.

---

## 13. Trust, Safety, Security, and Governance

### 13.1 Data privacy

Customer data must remain scoped to the authorized account/workspace/project.

### 13.2 Secrets

Secrets must:
- never be stored in prompts or logs in plaintext;
- be referenced through secure secret storage;
- be scoped to tools/actions;
- be rotatable.

### 13.3 Access control

Server-side RBAC/ABAC is required.

### 13.4 Tenant isolation

Mandatory for business accounts.

### 13.5 Model safety

For DCLab’s ML platform context, model safety includes:
- preventing unvalidated promotion;
- data/target leakage checks;
- artifact integrity;
- explainability/evidence appropriate to use case;
- rollback ability.

### 13.6 Agent permissions

Agents need least-privilege tool access and explicit action scopes.

### 13.7 Approval gates

High-risk changes cannot rely on conversational ambiguity.

### 13.8 Audit logs

Must include human and agent actions.

### 13.9 Reproducibility

Reproducibility is part of governance, not only convenience.

### 13.10 Scientific honesty

The system must not:
- hide worse runs;
- change test sets to improve metrics without trace;
- compare incompatible experiments as if they were equivalent;
- claim a deployment succeeded without verification.

### 13.11 Leakage detection

Must exist before model promotion.

### 13.12 Production eligibility

Exact policy [UNDECIDED].

Should consider:
- test evidence;
- robustness;
- calibration;
- latency;
- artifact integrity;
- dependency/security scan;
- monitoring readiness.

### 13.13 Destructive actions

Delete/overwrite/production migration actions require explicit controls.

### 13.14 Compliance

[UNDECIDED] No formal compliance target such as SOC 2, ISO 27001, HIPAA, or GDPR program was recovered.

### 13.15 Explainability

DCLab should explain:
- what data/features were used;
- what changed;
- why a recommendation exists;
- what evidence supports it.

This does not mean every model must have a simplistic explanation.

### 13.16 Evidence quality

Evidence should be rated by provenance and comparability, not agent confidence alone.

### 13.17 Rollback and recovery

Production-capable workflows need:
- previous known-good release;
- rollback procedure;
- artifact retention;
- migration awareness.

---

## 14. Business and Market Context

### 14.1 Market category

DCLab sits across:
- developer tools for ML/AI;
- MLOps;
- experiment management;
- AutoML;
- notebook/IDE alternatives;
- agentic development environments;
- long-term decision intelligence.

**[CONFLICT/TENSION]:** The long-term decision-intelligence vision is much broader than the initial developer-tool wedge. The newest practical direction keeps **developers as the core first customer**, so market positioning should not prematurely become a generic enterprise “decision platform.”

### 14.2 Positioning

Best recovered positioning:

> Developer-first AI/ML engineering environment that owns the workflow from data to reproducible model evidence, with agentic execution/debugging and a path toward deployment.

### 14.3 Differentiation

Against notebooks:
- workflow state, recovery, automation, lineage.

Against general AI coding assistants:
- persistent ML domain model, execution state, evidence, lifecycle.

Against experiment trackers:
- DCLab actively executes and repairs work.

Against AutoML:
- deeper developer control and workflow transparency.

Against cloud ML suites:
- vendor-neutral/product-focused workflow experience [aspirational; provider strategy undecided].

Against data platforms:
- DCLab should consume their capabilities rather than rebuild the whole data stack.

### 14.4 Ideal customer profile

**Confirmed earliest ICP:** Individual developer/ML engineer/AI engineer who wants to work faster.

**Secondary:** Small technical business/team with several ML/data practitioners.

**Future:** Larger organizations requiring governance and collaboration.

### 14.5 Business model

[UNCERTAIN] SaaS is strongly implied.

One earlier marketing conversation referenced engaging users at roughly **$10/month** in an early-stage context, but there is not enough evidence to treat this as canonical pricing.

### 14.6 Pricing hypotheses

- Personal Development plan.
- Business/team plan with more collaboration/admin capabilities.
- Potential usage-based compute/agent costs.

Only the account distinction is confirmed; exact pricing is not.

### 14.7 Distribution strategy

Recovered themes:
- founder-led early adoption;
- developer users;
- UAE ecosystem;
- startup/AI competitions;
- incubators/accelerators;
- potentially private prototype demonstrations.

Exact GTM funnel is [UNDECIDED].

### 14.8 Private prototype strategy

A “private prototype” title exists in the requested source list, but the complete conversation is unavailable. Therefore only a general conclusion is safe:

[UNCERTAIN] A private prototype has been considered as a way to validate the experience before broad public launch.

### 14.9 Open-source vs proprietary

[UNDECIDED]. No authoritative boundary recovered.

### 14.10 Defensibility

**[INFERRED] likely defensibility sources:**
- structured workflow/evidence graph;
- agent execution data;
- accumulated repair patterns;
- experiment/reproducibility system;
- deployment context;
- product-specific developer UX;
- integrations.

Do not rely on model access alone as a moat.

### 14.11 Network effects / data advantages

[UNCERTAIN/Future]
- private project history can improve each customer experience;
- aggregate learning may improve templates/agents if privacy-safe;
- cross-customer data use requires explicit governance.

### 14.12 Long-term company vision

Build the operating layer that turns data and ML intent into executed, validated, deployable intelligence, eventually expanding from model engineering into decision support and action/outcome intelligence.

---

## 15. Current Product State

Because the repository was not directly inspected in this run, implementation claims are conservative.

| Capability | Concept only | Designed | Partially implemented | Implemented | Tested | Production-ready | Evidence/source | Known gaps |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Main DCLab web application | No | Yes | Yes | [UNCERTAIN] | [UNCERTAIN] | No evidence | Prior UI/repo discussions | current code not inspected |
| Dashboard | No | Yes | Yes | [UNCERTAIN] | Unknown | Unknown | Redesign conversation references existing dashboard | UX disliked; backend wiring needs verification |
| Labs area | No | Yes | Yes | [UNCERTAIN] | Unknown | Unknown | Existing UI referenced | exact functionality unknown |
| Workflow/model UI | No | Yes | Yes | [UNCERTAIN] | Unknown | Unknown | Existing workflow UI referenced | navigation redesign requested |
| Developer/personal account concept | No | Yes | [UNCERTAIN] | [UNCERTAIN] | Unknown | No evidence | Development-side conversation | exact implementation unknown |
| Business account concept | No | Yes | [UNCERTAIN] | [UNCERTAIN] | Unknown | No evidence | Automate ML Workflow | role verification needed |
| DCLab admin role | No | Yes | [UNCERTAIN] | [UNCERTAIN] | Unknown | Unknown | Role discussion | exact permissions unknown |
| DCLab employee/developer role | No | Yes | [UNCERTAIN] | [UNCERTAIN] | Unknown | Unknown | Role discussion | initially read-only direction |
| Dataset upload | No | Yes | [UNCERTAIN] | [UNCERTAIN] | Unknown | Unknown | Multiple data conversations | pipeline not inspected |
| Data semantic layer | No | Yes | Unknown | No evidence | No | No | Data structuring design | implementation unknown |
| EDA | No | Yes | [UNCERTAIN] | [UNCERTAIN] | Unknown | Unknown | Core workflow discussions | evidence unavailable |
| Model training | No | Yes | [UNCERTAIN] | [UNCERTAIN] | Some R&D tests | No evidence | R&D/model conversations | product integration unknown |
| Model evaluation | No | Yes | [UNCERTAIN] | [UNCERTAIN] | R&D examples exist | No evidence | XGBoost/RF evaluation conversation | main product state unknown |
| Experiment tracking | No | Yes | Unknown | No evidence | No | No | Scientific vision | exact system unknown |
| Agentic debugging | No | Yes | Unknown | No evidence | No | No | Notebook-disruption conversations | key differentiator not verified |
| Deployment agent | Yes/future | Early concept | No evidence | No | No | No | Disrupt Colab / MLOps-agent discussion | future phase |
| Monitoring | Future | Concept | No evidence | No | No | No | implied by deployment lifecycle | architecture undecided |
| R&D repository | No | Yes | Yes | Yes as experimentation repo | Yes for some experiments | Not product | ML innovation / transformer conversations | separate from product |
| Relational Transformer experiments | Experimental | Yes | [UNCERTAIN] | [UNCERTAIN] | unknown | No | article/R&D discussion | research only |
| Decision-intelligence layers | Future | Conceptual | No evidence | No | No | No | disruption vision | should not block core |
| Prediction-driven customer generation | Future | Conceptual | No evidence | No | No | No | customer generation discussion | application layer |

---

## 16. Roadmap and Scope Structure

### 16.1 Scope 0–10 availability

A conversation titled “Create scopes 0 to 10 plan” is referenced in the requested source list, but the actual scope definitions are not available in this run.

Therefore:

**[UNCERTAIN]** The existence of a prior Scope 0–10 framework is plausible, but its contents cannot be reconstructed authoritatively without inventing details.

### 16.2 Recovered chronological roadmap

#### Phase A: Core developer workflow
- stabilize project/account model;
- reliable dataset upload;
- execution environment;
- training/evaluation;
- clear workflow UI;
- experiment state;
- preserve backend functionality.

#### Phase B: Notebook disruption
- chat/click execution;
- explicit pipeline state;
- automatic log capture;
- agentic error diagnosis;
- repair/retry loop.

#### Phase C: Scientific strengthening
- feature lineage;
- model comparison;
- tuning;
- reproducibility bundles;
- experiment/decision records;
- stronger leakage checks.

#### Phase D: Team/business expansion
- business accounts;
- roles;
- multiple technical users;
- admin controls;
- organization visibility.

#### Phase E: Deployment/MLOps
- build/package;
- deployment adapters;
- agentic deployment debugging;
- health checks;
- rollback.

#### Phase F: Monitoring and feedback
- production monitoring;
- drift/performance;
- feedback to experiments.

#### Phase G: Advanced intelligence
- many-model evidence aggregation;
- recommendation layer;
- action-outcome prediction;
- reaction prediction;
- domain/application modules such as customer generation.

### 16.3 Dependency-based roadmap

1. **Identity, tenancy, project model**
2. **Artifact/data storage**
3. **Job/execution model**
4. **Dataset ingestion + validation**
5. **Experiment/run domain**
6. **Training/evaluation**
7. **Workflow UX**
8. **Agent tool/action layer**
9. **Automatic debugging**
10. **Reproducibility + lineage**
11. **Business collaboration**
12. **Deployment abstraction**
13. **Monitoring**
14. **Cross-project knowledge**
15. **Decision-intelligence applications**

### 16.4 What should be built first

**Recovered priority:** developer core and notebook-disruption value.

Do not begin with:
- broad enterprise decision intelligence;
- hundreds of speculative model layers;
- complex cross-project knowledge graph;
- full autonomous MLOps;
- custom reimplementation of data-warehouse infrastructure.

### 16.5 What must wait

- fully autonomous production deployment;
- broad business decision modules;
- lead-generation application;
- sophisticated cross-tenant learning;
- advanced enterprise compliance unless required by an actual customer;
- speculative “100 model × 100 feature” stacks as a default architecture.

### 16.6 What appears already completed

Only safe statement:
- some main web UI and role/backend components exist or have existed in partial form;
- R&D model experiments have been run.

Anything more specific requires repository inspection.

### 16.7 Superseded direction

**Superseded or narrowed:** Treating business and developer offerings as separate fundamental products.

Newer direction:
- one common ML core;
- Personal Development and Businesses are account/product packaging around the same core;
- business gets additional capabilities, not a different scientific engine.

---

## 17. Decisions and Conflicts Ledger

### 17.1 Decision ledger

| ID | Topic | Decision | Status | Rationale | Source | Order | Consequences | Follow-up |
|---|---|---|---|---|---|---|---|---|
| D-001 | Primary user | Developers/ML practitioners are the core first users | Confirmed | Product should make technical work faster | Evaluate Model; Development side | Recent | UX and roadmap optimize for practitioners | validate onboarding |
| D-002 | Notebook strategy | Goal is to remove notebook dependence, not merely augment notebooks | Confirmed | notebook/error loop is a major pain | Disrupt Colab; Startup Marketing Strategy | Recent | workflow engine must be first-class | define compatibility/export |
| D-003 | Interaction | Chat or clicking should be sufficient for core workflows | Confirmed | reduce manual coding/coordination | Evaluate Model | Recent | agent + structured UI both required | define action grammar |
| D-004 | Failure recovery | DCLab should automatically diagnose and repair execution errors where safe | Confirmed | remove copy/paste AI debugging loop | Startup Marketing Strategy | Recent | logs + agent tools + retry state required | create repair policy |
| D-005 | Account products | Personal Development and Businesses | Confirmed naming/direction | serve individuals and teams | Development side | Recent | shared core with account-specific features | final pricing |
| D-006 | Core reuse | Business uses same ML core as developer/personal product | Confirmed | avoid duplication/forking | Development side | Recent | common domain/services | enforce architecture |
| D-007 | Business users | Business account can contain multiple ML/data users | Confirmed; exact count tentative | team collaboration | Development side | Recent | membership/RBAC | confirm seat limits |
| D-008 | Internal admin | DCLab admin needs deep visibility into modeling details | Confirmed | quality/control | Automate ML Workflow | Recent | admin observability | formalize permissions |
| D-009 | Internal developer role | DCLab employees similar to admin but restricted; read-only initially | Tentative | operational safety | Automate ML Workflow | Recent | separate internal role | define exact access |
| D-010 | UI navigation | Replace excessive top tabs with left-side navigation | Confirmed | current dashboard navigation disliked | Redesign DCLab UI | Recent | information architecture change | implement without regressions |
| D-011 | UI visual identity | Keep current colors/theme | Confirmed | founder likes current palette/theme | Redesign DCLab UI | Recent | redesign should be structural | preserve tokens |
| D-012 | UI style | Liquid-glass effect desired for sidebar | Confirmed aesthetic direction | preferred visual style | Redesign DCLab UI | Recent | front-end design constraint | accessibility check |
| D-013 | Backend integrity | UI changes must preserve working API functionality | Confirmed | avoid regressions | Redesign DCLab UI | Recent | endpoint contract verification | integration tests |
| D-014 | Missing backend | If backend is absent, frontend may remain unwired rather than fake behavior | Confirmed | correctness over illusion | Redesign DCLab UI | Recent | explicit TODO states | define feature flags |
| D-015 | Data understanding | Use heuristics/statistical methods before LLM | Confirmed | cost, scalability, reliability | Design Data Structuring Tool | Recent | layered parser architecture | define confidence thresholds |
| D-016 | Semantic data model | Normalize heterogeneous input into semantic representation | Confirmed | varied sources still have usable structure | Data Structuring Tool | Recent | schema/semantic entities | define model |
| D-017 | R&D separation | R&D is supporting subsystem, not whole DCLab | Confirmed by current instruction and project history | avoid conflating experiments/product | Current request + R&D chats | Latest | documentation/repo boundaries | add README boundary |
| D-018 | Baselines | Conventional ML must be optimized too when comparing transformers | Confirmed | fair benchmark | ML Innovation | Earlier | standardized experiment plans | enforce benchmark templates |
| D-019 | Long-term MLOps agent | DCLab may develop its own deployment/MLOps agent | Strong direction | market gap and workflow ownership | Disrupt Colab | Recent | future agent architecture | validate market and scope |
| D-020 | Long-term intelligence | Multi-layer prediction/recommendation/action/reaction modeling is future vision | Confirmed vision, not MVP | richer decision intelligence | Remember Disruption Vision | Earlier/ongoing | architecture should not block future | keep decoupled |

### 17.2 Conflict table

| Conflict | Versions | Current strongest decision | Why | Remaining uncertainty |
|---|---|---|---|---|
| Product identity | ML platform vs broad decision-intelligence platform | Developer-first ML engineering platform first; decision intelligence later | newest execution-focused conversations repeatedly prioritize developers | future category naming |
| Notebook role | Improve notebooks vs remove notebooks | Remove notebooks as primary workflow | explicit repeated founder goal | whether to support notebook import/export |
| Business architecture | Separate business product vs shared core | Shared core with additional business features | explicit newer decision | packaging/seat limits |
| LLM usage | AI everywhere vs layered processing | Heuristics/statistics/ML first, LLM where needed | explicit data-layer cost concern | thresholds and routing |
| Model strategy | Favor transformer innovation vs best-performing practical models | Compare fairly against optimized classical baselines | explicit benchmark request | promotion criteria |
| Data platform scope | Build all ingestion/data capabilities vs use Snowflake/Databricks/etc. | Integrate existing infrastructure where it avoids reinventing the wheel | explicit founder concern | which vendors/integrations |
| Deployment | Manual integration vs autonomous deployment agent | Long-term autonomous agent with guardrails | explicit recent interest | MVP timing and provider scope |
| Navigation | Existing top-tab design vs left sidebar | Left sidebar | explicit redesign request | exact IA |
| SQL vs NoSQL | MongoDB vs PostgreSQL | [INFERRED] PostgreSQL stronger for core domain | relational constraints/audit needs and earlier comparison | formal ADR absent |
| Large “100 model × 100 feature” vision | Default architecture vs advanced capability | Future/conditional capability, not default MVP | operational complexity conflicts with developer-first MVP | where/when to enable ensemble layers |

---

## 18. Assumptions, Unknowns, and Questions

### 18.1 Blocking questions: highest priority

1. **What is the current main repository architecture and actual implementation state?**  
   Blocks reliable planning and prevents duplicate work.

2. **What are the exact existing database schemas and migrations?**  
   Blocks canonical domain-model alignment.

3. **What is the actual auth provider and RBAC implementation?**  
   Blocks business account/tenant work.

4. **What is the current execution model?**  
   Local process, Docker, remote worker, Kubernetes, third-party job service?

5. **Which parts of dataset upload, training, evaluation, and workflow are already functioning end-to-end?**

6. **What exactly was in the missing Scope 0–10 plan?**

7. **Which deployment targets matter first?**  
   Generic Docker? AWS? GCP? Azure? serverless? Kubernetes?

### 18.2 Product assumptions

- Developers will adopt a non-notebook workflow if execution/control is strong enough.
- Automatic debugging creates enough time savings to be a clear wedge.
- Users value evidence/reproducibility, not just final metric.
- A shared ML core can serve both individuals and teams.

### 18.3 Technical assumptions

- PostgreSQL can serve as primary metadata store.
- Object storage can hold datasets/models/artifacts.
- Workloads need async workers.
- Sandboxed execution is mandatory.
- Agents can be made useful through structured tools and logs.

### 18.4 Scientific assumptions

- Fair baseline comparison is essential.
- Automated feature engineering can be useful if lineage and review exist.
- Model quality should be evaluated on more than a single metric.
- The test protocol must remain stable through comparisons.

### 18.5 Business assumptions

- Developer-first wedge is stronger than selling broad decision intelligence immediately.
- Businesses will pay for collaboration/governance above personal functionality.
- UAE startup ecosystem can be a useful early distribution/funding channel but is not itself the product strategy.

### 18.6 Missing evidence

- user research;
- actual developer retention;
- willingness to leave notebooks;
- cost per active project;
- automatic-debug success rate;
- compute margin;
- deployment-agent reliability;
- enterprise security requirements.

### 18.7 Founder-confirmation questions

Ranked by implementation impact:

**P0**
- Confirm primary database.
- Confirm tenancy model.
- Confirm execution/sandbox model.
- Confirm current active scope.
- Provide current repo state and missing Scope 0–10 plan.

**P1**
- Confirm exact Personal Development vs Business packaging.
- Confirm whether business seat count “5” is a real constraint.
- Confirm first deployment provider.
- Confirm whether notebook import/export is required.

**P2**
- Confirm long-term decision-intelligence terminology.
- Confirm open-source policy.
- Confirm pricing.
- Confirm enterprise/compliance ambitions.

---

## 19. DCLab Terminology

| Term | Definition | What it is not | Related concepts | Status |
|---|---|---|---|---|
| DCLab | Main end-to-end developer-first ML engineering platform | Not the R&D repo | workflows, agents, experiments, deployment | Confirmed |
| Personal Development | Individual/developer-facing product/account mode | Not a separate ML engine | developer account | Confirmed naming |
| Businesses | Team/business account mode with additional admin/collaboration | Not a separate modeling core | workspace, memberships | Confirmed naming |
| DCLab Admin | Internal super-operator with deep platform/model visibility | Not customer business admin | internal governance | Confirmed |
| DCLab Developer | Internal DCLab employee/developer role with restrictions | Not end-user developer account | internal access | Tentative |
| Developer account | Core customer account for individual technical users | Not DCLab employee | Personal Development | Confirmed concept |
| Business admin | Customer-side admin for a business account | Not DCLab Admin | seats, workspace policy | Confirmed |
| Workflow | Explicit ML lifecycle graph/state | Not notebook cell order | jobs, steps, artifacts | Confirmed |
| Experiment | Planned set of comparable model attempts | Not one arbitrary run | hypothesis, runs | Confirmed concept |
| Run | One execution attempt/configuration | Not the whole experiment | logs, metrics, artifact | Canonical term [PROPOSED] |
| Data Layer | Ingestion/understanding/semantic processing stack | Not just file upload | heuristics, ML, LLM | Confirmed concept |
| Semantic Data Model | Structured interpretation of fields/entities/relationships | Not free-form LLM summary | schema, metadata | Confirmed concept |
| Agentic Debugging | Automatic diagnosis/fix/retry of workflow failures | Not only explaining an error | runtime, logs, tools | Confirmed |
| MLOps Agent | Future agent for deployment and production workflow | Not unrestricted cloud admin | deployment, rollback | Confirmed direction |
| R&D repository | Experimental model/research codebase | Not DCLab product | transformers, benchmarks | Confirmed separation |
| Decision Intelligence | Long-term ability to combine predictions, recommendations, actions, outcomes | Not MVP positioning | multi-layer models | Provisional/future |
| Prediction Layer | Models estimating outcomes/events | Not entire decision system | recommendation layer | Future concept |
| Recommendation Layer | Models/systems selecting recommended actions | Not simple model ranking | action-outcome | Future concept |
| Action-Outcome Prediction | Predict consequences of actions | Not generic forecasting | decision intelligence | Future concept |
| Reaction Prediction | Predict likely user/customer response | Not current core | behavior modeling | Future concept |
| Liquid Glass | Desired visual treatment for left navigation | Not a functional requirement | sidebar redesign | Confirmed design preference |

---

## 20. Implementation Constraints and Invariants

### 20.1 Product invariants

- **MUST** keep developers/ML practitioners as the core first-user experience.
- **MUST** keep the scientific/ML core shared between Personal Development and Businesses.
- **MUST NOT** present the R&D repository as the main DCLab product.
- **MUST NOT** reduce DCLab to a chat wrapper around generic AI.
- **SHOULD** eliminate unnecessary manual transitions between intent, execution, debugging, and evidence.

### 20.2 Data invariants

- **MUST** retain provenance for uploaded/connected data.
- **MUST** make material transformations traceable.
- **MUST NOT** use LLM processing as the default for massive raw datasets when structural/heuristic/statistical methods can do the job.
- **SHOULD** preserve immutable or versioned dataset states.
- **[PROPOSED] MUST** checksum durable data/artifact versions.

### 20.3 ML/scientific invariants

- **MUST** preserve meaningful baselines when evaluating new model families.
- **MUST** prevent or flag leakage.
- **MUST** tie evaluation metrics to a known dataset/split/run.
- **MUST NOT** hide failed/negative runs when they affect evidence.
- **SHOULD** make feature engineering explicit and traceable.
- **SHOULD** use multiple relevant metrics rather than one universal score.

### 20.4 Agent invariants

- **MUST** use permission boundaries and guardrails for autonomous actions.
- **MUST** record agent actions that modify execution/project state.
- **MUST NOT** let an agent’s unsupported natural-language claim substitute for execution evidence.
- **SHOULD** automatically repair and rerun low-risk failures where policy allows.
- **MUST NOT** create unbounded retry loops.
- **[PROPOSED] SHOULD** separate planning from high-risk execution approvals.

### 20.5 Security invariants

- **MUST** enforce authorization in backend/service boundaries.
- **MUST** isolate business/customer data by tenant/account.
- **MUST NOT** expose secrets in logs/prompts.
- **[PROPOSED] MUST** execute untrusted user code in an isolated sandbox.

### 20.6 Reproducibility invariants

- **MUST** retain enough run context to understand how a result was produced.
- **SHOULD** persist environment, parameters, data version, feature version, logs, and artifact references.
- **MUST NOT** overwrite the only evidence of an earlier run with a new attempt.

### 20.7 UX invariants

- **MUST** preserve working functionality during redesign.
- **MUST** correctly connect to existing backend endpoints.
- **MUST NOT** fake backend functionality that does not exist.
- **MUST** move the dense top-tab navigation direction toward a left-side navigation per current design decision.
- **MUST** preserve the existing preferred color/theme direction.
- **SHOULD** make running/error/approval states obvious.

### 20.8 Business invariants

- **MUST NOT** let broad future decision-intelligence modules derail the developer-first core.
- **SHOULD** reuse external data/cloud infrastructure where rebuilding it creates little differentiated value.
- **[PROPOSED] SHOULD** keep compute/agent cost visible enough to support sustainable pricing.

---

## 21. Context Required by Codex

This section is intentionally concise enough to initialize a coding task.

### 21.1 Project objective

DCLab is a developer-first ML engineering platform designed to replace fragmented notebook-centric workflows with a structured system that can ingest/understand data, execute reproducible ML workflows, train and compare models, automatically debug failures, preserve lineage/evidence, and later deploy/monitor models through guarded agents.

### 21.2 Product core

The core is:

1. Project/workspace state.
2. Dataset ingestion/versioning.
3. Data validation/semantic understanding.
4. Explicit ML workflow/pipeline.
5. Feature engineering.
6. Experiments/runs.
7. Training/tuning/evaluation.
8. Model comparison and evidence.
9. Agentic debugging and retry.
10. Artifact/lineage/audit.
11. Shared core for Personal Development and Businesses.

### 21.3 Architecture summary

Current authoritative target architecture:

- web frontend;
- backend API/domain layer;
- relational metadata store, likely PostgreSQL but final ADR not recovered;
- object/artifact storage such as S3/GCS class storage;
- async job system;
- isolated execution workers/sandboxes;
- agent runtime with scoped tools;
- later deployment adapters and monitoring;
- strong tenant/role enforcement.

### 21.4 Current implementation state

Do **not** assume features are implemented merely because they appear in this document.

Known:
- a main DCLab repo exists;
- dashboard/labs/workflow UI has existed;
- prior role/backend work has been partially implemented;
- UI needs structural redesign;
- R&D experiments exist separately.

Unknown until code inspection:
- exact APIs;
- schema;
- workers;
- auth;
- current tests;
- deployment.

### 21.5 Critical invariants

Codex must not:

- fork the ML core between Personal Development and Businesses;
- conflate the R&D repo with product;
- replace working backend logic during UI redesign;
- invent backend endpoints;
- hide scientific lineage;
- let agent behavior bypass permissions;
- use LLM calls for bulk deterministic processing without reason.

### 21.6 Repository map

**[UNAVAILABLE]** No verified current tree is present in this context.

Codex must inspect the repository before making architecture assumptions.

### 21.7 Active priorities

Based on the latest recovered direction:

1. developer workflow;
2. stable architecture/domain model;
3. data/workflow/model execution;
4. notebook-disruption UX;
5. automatic debugging;
6. shared personal/business core;
7. UI redesign without regressions;
8. deployment agent only after the above is stable.

### 21.8 Known technical debt

Only evidence-backed debt:

- UI navigation is considered poor/over-tabbed.
- Some roles/features may be partially implemented and need verification.
- Frontend/backend wiring needs careful verification during redesign.
- Product architecture/context has been fragmented across chats and needs durable documentation.
- R&D experiments and product boundaries need explicit separation.

### 21.9 Decisions Codex must not revisit without approval

- Developers are the core first users.
- Personal Development and Businesses share the same core ML functionality.
- Left-side navigation replaces the current excessive top-tab model.
- Preserve current preferred colors/theme.
- Existing working functionality must not be broken.
- R&D repository is not the product.
- DCLab aims to remove notebook dependence.
- Automatic debugging/retry is a strategic differentiator.
- Data understanding should not default to LLM-everywhere.

### 21.10 Questions Codex must resolve from code before assumptions

- existing framework versions;
- current auth provider;
- current DB technology/schema;
- current API routes;
- job queue/runtime;
- storage provider;
- test structure;
- role names/permissions;
- existing feature flags;
- deployment config.

### 21.11 Definition of done

For any implementation task:

- functionality works through real backend path where backend exists;
- no regression in unrelated existing behavior;
- tests added/updated;
- role/tenant boundaries respected;
- execution state is persisted appropriately;
- errors are explicit;
- user-visible state matches backend truth;
- no fake successful states;
- migrations are safe;
- evidence of testing is recorded in the PR/task summary.

### 21.12 Testing and evidence expectations

At minimum, depending on change:
- unit tests;
- API/integration tests;
- role/permission tests;
- migration test;
- end-to-end happy path;
- failure path;
- retry/idempotency where applicable;
- UI state for loading/error/empty;
- model workflow reproducibility tests for ML changes.

---

## 22. Recommended Durable Project Files

This section proposes documentation organization only. It does not create new product decisions.

### `DCLAB_MASTER_CONTEXT.md`

Purpose:
- full source-of-truth narrative;
- recovered decisions;
- product definition;
- modules;
- current-state caveats;
- cross-domain context.

Should contain:
- Sections 0–23 of this document or an updated equivalent.

Should not contain:
- low-level code docs that go stale daily.

### `PRODUCT_CONSTITUTION.md`

Purpose:
- stable product principles and invariants.

Include:
- target users;
- what DCLab is/is not;
- non-negotiables;
- product philosophy;
- UX principles;
- shared-core rule;
- R&D separation.

### `ARCHITECTURE.md`

Purpose:
- current technical architecture.

Include:
- diagrams;
- service boundaries;
- storage;
- queues;
- execution runtime;
- agent runtime;
- deployment;
- trust boundaries;
- scalability assumptions.

### `DOMAIN_MODEL.md`

Purpose:
- canonical entities/state machines.

Include:
- entities;
- fields;
- relationships;
- ownership;
- lifecycle;
- invariants.

### `ROADMAP.md`

Purpose:
- active scopes and sequencing.

Include:
- current phase;
- milestones;
- dependencies;
- exit criteria;
- deferred work.

### `DECISIONS.md` or `docs/adr/*.md`

Purpose:
- prevent repeated re-litigation.

Use ADRs for:
- database choice;
- tenancy model;
- job system;
- sandboxing;
- artifact storage;
- deployment abstraction;
- agent permissions;
- notebook compatibility.

### `AGENTS.md`

Purpose:
- instructions for coding/ML/ops agents.

Include:
- repository map;
- commands;
- tests;
- conventions;
- unsafe actions;
- approval requirements;
- how to use project docs;
- rules around R&D code.

### R&D relationship documentation

Recommended:
- `docs/RND_RELATIONSHIP.md` in main repo;
- corresponding README note in R&D repo.

Explain:
- R&D purpose;
- promotion path into product;
- no assumption that experimental APIs are stable;
- benchmark requirements;
- ownership boundaries.

---

## 23. Final Completeness Audit

### 23.1 Important subjects covered

- product definition;
- developer-first strategy;
- notebook-disruption thesis;
- agentic debugging;
- roles/account model;
- personal vs business products;
- UI redesign direction;
- data-layer hierarchy;
- semantic data modeling;
- scientific core;
- experiments/baselines;
- R&D relationship;
- object storage/database direction;
- long-term MLOps agent;
- decision-intelligence future;
- canonical domain model;
- architecture;
- security/governance;
- business context;
- roadmap;
- decision/conflict ledger;
- Codex onboarding;
- documentation structure.

### 23.2 Subjects with weak evidence

- exact current repository architecture;
- exact backend implementation status;
- exact database decision;
- exact Scope 0–10 plan;
- exact auth provider;
- exact worker/orchestration stack;
- production deployment stack;
- monitoring stack;
- pricing;
- open-source strategy;
- compliance target;
- current customer traction;
- exact business seat limit.

### 23.3 Missing conversations/files

Most important missing originals:
- DCLab concept summary;
- Build Private DCLab Prototype;
- AI SaaS Platform Design;
- Create scopes 0 to 10 plan;
- Generalize dataset upload pipeline;
- complete current GitHub repository state;
- any architecture diagrams/schema docs;
- any current PRD or roadmap files.

### 23.4 Contradictions not fully resolved

1. DCLab as developer ML platform vs broad decision-intelligence platform.  
   Working resolution: developer-first ML platform now; decision intelligence later.

2. SQL vs NoSQL.  
   Working direction: PostgreSQL likely, but no formal ADR recovered.

3. Degree of autonomy for agents.  
   Guardrails are confirmed; exact approval matrix is not.

4. Notebook compatibility.  
   Goal is to remove notebooks as primary workflow, but import/export compatibility is undecided.

5. “Hundreds of models/features” vision.  
   Confirmed as an ambitious intelligence concept, not a default MVP implementation.

### 23.5 Details likely lost through summarization

Because many project conversations are available only as summarized project context rather than full transcripts, likely losses include:
- exact naming of prior scopes;
- precise endpoint names;
- original schema field names;
- exact UI component hierarchy;
- exact implementation prompts given to coding agents;
- detailed competitor comparisons;
- precise pricing/GTM hypotheses;
- detailed transformer experiment settings;
- individual corrections made during long architecture discussions.

### 23.6 Areas requiring founder confirmation

- current scope;
- current repository truth;
- Postgres finalization;
- exact team limits;
- deployment target;
- agent permission model;
- notebook interoperability;
- pricing;
- open-source boundary;
- when decision-intelligence modules enter roadmap.

### 23.7 Confidence by major section

| Section | Confidence |
|---|---:|
| 0. Source Coverage | High |
| 1. Executive Definition | High |
| 2. Founder Vision | High |
| 3. Problems | High |
| 4. Target Users | High for developers/business roles; medium for secondary personas |
| 5. User Journey | Medium-High; some lifecycle stages inferred |
| 6. Core Capabilities | High for core; medium for future modules |
| 7. Agent System | Medium; responsibilities stronger than topology |
| 8. Scientific Core | High |
| 9. Data/Knowledge Architecture | Medium-High |
| 10. Technical Architecture | Medium; implementation specifics missing |
| 11. Domain Model | Medium; normalized from product needs |
| 12. UX/IA | High for redesign principles; medium for full screen map |
| 13. Trust/Security | Medium-High on principles; low on formal compliance |
| 14. Business/Market | Medium |
| 15. Current Product State | Low-Medium because repo not inspected |
| 16. Roadmap | Medium; exact Scope 0–10 missing |
| 17. Decisions/Conflicts | High for listed decisions |
| 18. Assumptions/Unknowns | High |
| 19. Terminology | High |
| 20. Invariants | High for sourced rules; proposals clearly marked |
| 21. Codex Context | High as onboarding synthesis |
| 22. Durable Files | High as documentation proposal |
| 23. Audit | High |

### 23.8 Exact follow-up questions needed for a fully authoritative version

1. What is the current default branch/commit of the main DCLab repository that should define implementation truth?
2. Can the original **Scope 0–10** plan be provided?
3. Is PostgreSQL now the final primary metadata database?
4. What is the current authentication provider?
5. What are the exact current roles and permissions in code?
6. Is “five users per business account” still intended, or was it only an early example?
7. What is the current object-storage provider?
8. What job queue/orchestrator is already implemented, if any?
9. How is ML code executed today?
10. Is code sandboxing already implemented?
11. Which dataset formats/connectors currently work?
12. Which ML model families currently work in the main product, not the R&D repo?
13. Is experiment tracking already a first-class backend entity?
14. Are feature definitions/versioning already stored?
15. Does the current product store dataset versions/checksums?
16. Is automatic debugging implemented anywhere yet?
17. Which first deployment target should be canonical?
18. Is monitoring already in scope for MVP or post-MVP?
19. Should notebook import/export be supported even though notebooks are not the primary UX?
20. What is the intended launch definition for **Personal Development**?
21. What additional capabilities define **Businesses** at launch?
22. What is the current pricing hypothesis?
23. Which DCLab capabilities are already validated with real users?
24. What is the exact boundary between main repo and R&D repo?
25. Are there additional private prototype decisions that supersede anything in this document?

---

# Closing Constitution

DCLab’s core identity is not “AI that writes ML code.” It is the system that **owns the ML workflow state**.

The platform should know what the user is trying to accomplish, what data was used, what transformations occurred, what experiments were attempted, what failed, how failures were repaired, what evidence supports a model, who approved critical transitions, what was deployed, and what happened afterward.

The developer should spend less time moving information between tools and more time making the decisions that actually require expertise.

That is the durable center of the project. Everything else, from UI polish to future decision-intelligence modules, should be judged by whether it strengthens or distracts from that center.

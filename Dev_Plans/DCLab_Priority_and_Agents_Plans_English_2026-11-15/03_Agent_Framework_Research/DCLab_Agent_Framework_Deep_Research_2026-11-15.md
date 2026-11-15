# DCLab: Agent Framework Landscape, Deep Agents Assessment, and Architecture Recommendation

**Date:** 15 November 2026  
**Status:** Research and architecture recommendation, **not an implemented change or benchmark result**  
**Scope:** DCLab internal decision-intelligence / AI data-science platform. The separately planned external Model/Problem/Project Agent product remains a separate architectural concern.  
**Language:** English  

## 1. Executive decision

**Recommended path: keep LangGraph as DCLab's existing orchestration and governance substrate; test LangChain Deep Agents as a composable reasoning-and-delegation harness, Pydantic AI + its Harness SDK as a lighter typed alternative, and OpenHands Software Agent SDK as an optional isolated coding worker. Use DSPy later to optimize validated problem-framing and planning modules. Do not replace the whole DCLab harness now.**

This is a *conditional* recommendation: proceed only after a controlled model-only / minimal-agent / existing-harness comparison identifies failures. The current diagnosis is a plausible early-binding or context/gate problem, **not proven**. A new agent framework is no substitute for a sound scientific workflow, tool contracts, provenance, or a capable model.

### Two different things must not be confused

1. **Improve the intelligence expressed by the current model:** preserve enough evidence and degrees of freedom for problem framing, hypothesis formation, method selection, planning, and revision. This comes from architecture, context, prompts, tools, model choice, and evaluations.
2. **Add higher-level agentic capabilities:** durable long-running research, task tracking, scoped subagents, workspace interaction, memory, skills, approvals, sandboxed execution. These are framework capabilities. They may improve performance **only if** the underlying failure is actually addressed.

A migration could recreate DCLab's deterministic trap with a more fashionable dependency tree. Measure before migrating.

## 2. DCLab constraints and current baseline

The recent DCLab plans describe an existing Python system with a model gateway, a `case_agent`, tool registry, guarded execution, a context builder, a finite `ProblemMap` family taxonomy, coordinated state transitions, LangGraph orchestration/checkpoints, memory/evidence records, an MCP server, and established evaluation infrastructure. These are *documented source-review findings*, not a fresh live runtime audit. The internal reports explicitly state that the live model/harness comparison has **not** yet been executed.

Relevant constraints:

- An earlier internal test reported a multi-agent configuration completing 15 of 19 tasks versus 19 of 19 for a strong single model, with more steps and spend. Small sample and different setup; use as motivation, **not** a universal performance claim.
- The product must handle ambiguous business decisions, temporal and multi-table data, scientific design, experiments, risk/cost tradeoffs, and situations where **no predictive model** is the right answer.
- Scientific constraints cannot be optional: train/evaluation split discipline, holdout protections, reproducibility, evidence provenance, cost limits, multi-tenant isolation, approvals, and independent artifact verification.
- The planned external three-harness Agentic ML platform (Model, Problem, Project) should interoperate via typed contracts, not be merged into DCLab's internal router just because a library supports teams.

**DCLab's main bottleneck hypothesis:** a useful model proposal may get narrowed, hidden, rewritten, or rejected as the request moves through early routing, context building, workflow selection, tool schema, and validation gates. Trace the **first material divergence** before changing the stack.

## 3. How the candidates were evaluated

Criteria (qualitative architecture assessment, **not numerical benchmark scores**):

1. Freedom to frame novel, out-of-taxonomy problems before selecting a route.
2. Multi-step autonomy, replanning, and long-running context management.
3. Evidence isolation, workspace access, bounded tool calling, and subagent delegation.
4. Fit with DCLab's existing Python/LangGraph/MCP ecosystem.
5. Ability to preserve DCLab-owned scientific validation, data contracts, permissions, and audit trails.
6. Extensibility, deployment burden, operational debugging, observability, licensing/provider freedom.
7. Actual incremental value over an already-functioning minimal model/tool agent.

Official documentation was checked on 15 November 2026. Feature sets and API names evolve; pin versions before a proof of concept. Vendor descriptions establish *advertised capability*, not DCLab-specific effectiveness. No head-to-head DCLab execution was run for this research.

## 4. Candidate matrix (fit judgments, not capability or performance rankings)

| Framework / SDK | Category | Strongest reason to consider | Primary DCLab concern | Decision |
|---|---|---|---|---|
| **LangGraph** | Stateful orchestration runtime | Already used; durable checkpointing, HITL, custom state/edges | Graph design can hard-code premature type selection | **Keep as foundation** |
| **LangChain Deep Agents** | Full agent harness on LangGraph | Context offloading, skills, filesystem, async/sync subagents, approvals | Built-in todo lists are not scientific planning; defaults and extra delegation can increase overhead | **Pilot first** |
| **Pydantic AI + Harness SDK** | Typed composable agent/harness | Excellent typed contracts, validation, modular capability composition, minimal-agent baseline | Another agent loop and overlap with existing runtime; integration complexity | **Pilot second** |
| **OpenHands Software Agent SDK** | Coding/workspace agent | Python/TS/REST, terminal/editor tools, Docker or remote workspace | Optimized for code tasks, not business problem framing; sandbox governance still needed | **Specialist worker pilot** |
| **OpenAI Agents SDK** | Lean agent loop/handoffs | Straightforward tools, guardrails, sessions, tracing, handoffs | Replacing orchestration adds limited unique value; provider behavior must be checked | **Strong minimal-agent control** |
| **Microsoft Agent Framework** | General-purpose agent + workflow + harness | Enterprise integrations, flexible workflows, harness, MCP, observability | Major orchestration migration; overlap with LangGraph | **Alternative only if enterprise requirements warrant** |
| **Google ADK** | Multi-agent / graph workflow | Agent + workflow modes, graph/parallel execution, eval integration, GCP deployment | Another graph abstraction, migration effort; version-specific changes | **Selective comparison only** |
| **CrewAI** | Crews and Flows | Fast role-based multi-agent prototypes, persisted flows and HITL | Can amplify role/route rigidity and back-and-forth overhead | **Not preferred as DCLab core** |
| **Agno / AgentOS** | Agent teams + managed operations | Rich tools, multi-agent modes, memory and runtime UI | Duplicates much of DCLab's platform layer; ownership lock-in | **Watch for external agent product** |
| **Strands Agents SDK** | Agent orchestration patterns | Graph, workflow, swarm, A2A, hooks, model diversity | Overlaps existing graph; swarms can obscure causality | **Optional alternate control** |
| **LlamaIndex AgentWorkflow** | Retrieval-centric agent workflow | Useful for data/retrieval-heavy agents and handoffs | DCLab already has evidence retrieval and orchestration | **Targeted retrieval use only** |
| **Hugging Face smolagents** | Small code-first tool agent | Extremely simple tool-agent control; CodeAgent + sandbox integrations | Experimental API; not a full robust decision-intelligence platform | **Cheap research baseline** |
| **Claude Agent SDK** | Coding/general file+shell agent harness | Mature shell/files, hooks, permissions, skills, subagents | External CLI/runtime dependency and provider/product constraints | **Optional bounded coding worker** |
| **Codex SDK** | Coding worker | Workspace sandbox, persistent threads, software engineering loop | Coding specialist, not a universal data-science reasoner | **Optional bounded coding worker** |
| **DSPy** | LM program/prompt optimization | Optimize instructions/demos from explicit outcome metrics (GEPA, MIPROv2) | Not an orchestration/runtime framework; wrong metric rewards wrong behavior | **Use after valid eval set exists** |
| **Temporal** | Durable workflow infrastructure | Reliable process recovery and long jobs across services | Duplicates some LangGraph durability; a separate operations platform | **Only when current durability fails a requirement** |
| **Langfuse / LangSmith** | Observability & evaluation | Traces, datasets, experiment comparisons, scoring | Tool visibility alone does not grade scientific correctness | **Keep/extend existing telemetry and custom scoring** |

### The shortlist

**Core candidate 1: Deep Agents.** Most natural integration path because it is built on LangGraph. Test as a *replacement for one isolated reasoning/analysis agent boundary*, never an immediate replacement for all DCLab state management.

**Core candidate 2: Pydantic AI Harness.** Most interesting counterexample: it may deliver clean typed output, budget controls, and selectively enabled capabilities with less unsolicited scaffolding. It is an alternative harness **for the controlled experiment**, not an instruction to add another orchestrator.

**Specialist candidate: OpenHands SDK.** Use for code exploration/edit/test in a constrained workspace, behind DCLab's authorization and artifact validation. Do not ask a coding agent to decide business causal identification, data leakage protocol, or whether ML is appropriate.

**Optimization candidate: DSPy.** Especially attractive for DCLab's `ProblemFrame`, candidate selection, and planning step once accepted plans and failures can be graded independently. Optimize **measured success**, not generated eloquence.

## 5. Deep Agents: detailed assessment

### What Deep Agents actually adds

Deep Agents is an agent harness layered over LangChain/LangGraph. Current Python documentation describes:

- Context management through summarization, long-output offloading, persistent memory, skills and retrieval.
- Virtual file tools (`ls`, `read_file`, `write_file`, `edit_file`, glob/grep) with pluggable backends; isolated sandbox backends for code execution.
- General or specialized subagents; synchronous subagents for small delegated work and asynchronous subagents for longer work.
- Human approval for configured tool calls, path-based filesystem permissions, structured outputs and custom middleware.
- Model-provider choice, integration with custom tools, and MCP support.
- Optional task-planning tools such as a `write_todos` list. **Planning capability is not synonymous with a scientifically valid research plan.**

Official documentation: [Deep Agents Python overview](https://docs.langchain.com/oss/python/deepagents/overview), [customization](https://docs.langchain.com/oss/python/deepagents/customization), [subagents](https://docs.langchain.com/oss/python/deepagents/subagents), [backends](https://docs.langchain.com/oss/python/deepagents/backends).

### Why it could be valuable for DCLab

1. **Context isolation:** a `DataProfiler` subagent may explore a dataset's metadata and return an evidence-linked profile, avoiding flooding the root problem-framer's prompt with raw rows and repeated tool outputs.
2. **Research/delegation:** a `ScientificCritic` could independently check missing target definitions, time leakage and unjustified causal claims before an execution plan is accepted.
3. **Artifact handling:** large intermediate analyses can be stored and referenced by durable IDs rather than pasted into every model exchange.
4. **Skills:** a verified temporal forecasting protocol or uplift/no-treatment-data caution can be loaded only when applicable.
5. **Integration:** custom tools and middleware can enforce existing DCLab model gateway, project ACLs, dataset/holdout policy, approval, tool registry and trace IDs.

### Why it is *not* a complete solution

- A `write_todos` task list is not a graph of testable scientific hypotheses, assumptions, selection criteria, dependencies and controlled experiments.
- Summarization and subagent delegation can **hide necessary evidence** if they return unsupported prose rather than source-linked artifacts.
- Filesystem abstraction can become an unsafe bypass unless `dataset_id`, data-access policies and holdout partitions remain authoritative. Virtual filesystem write permission is **not** a substitute for application-level authorization.
- More subagents can increase latency, cost, propagation errors and final-answer degradation. DCLab has already seen a concerning small-sample failure in a more complex multi-agent setup.
- Deep Agents does not know a prediction timestamp, a leakage-safe split, target availability, a business objective or an uplift estimator's causal assumptions by default. DCLab must continue to own those contracts.
- Its defaults must be audited: tools offered, automatic general subagent, memory scope, context compression, prompt budget, dynamic task scope, sandbox and approval configuration.

**Verdict:** **Yes, Deep Agents is worth testing for DCLab. No, it should not replace DCLab's internal planning, scientific validation, evidence store or security boundaries wholesale.**

## 6. Strongest alternatives and where each wins

### Pydantic AI + Harness SDK

Best argument: isolate a highly typed `ProblemFrame` and `SolutionCandidate` without adopting Deep Agents' broad defaults. Its harness capabilities cover coding/research, filesystem, memory, planning and compaction, while Pydantic agents retain typed tool parameters and output validation. The low-friction minimal agent is also useful as a control. Important question: does introducing a second model loop make the integration too expensive, or can the boundary be an independent read-only service?

Sources: [Pydantic agents](https://github.com/pydantic/pydantic-ai/blob/main/docs/agent.md), [Harness SDK](https://github.com/pydantic/pydantic-ai/blob/main/docs/harness/index.md), [durable execution](https://github.com/pydantic/pydantic-ai/blob/main/docs/durable_execution/overview.md).

### OpenHands Software Agent SDK

Best argument: real workspace action and code editing/execution rather than only text. It supports local or ephemeral Docker/Kubernetes workspaces and a remote Agent Server. Best used when a **validated analysis plan** needs bounded code authoring and tests. DCLab must still isolate training data from holdout, restrict network, verify imported packages, cap compute, and promote artifacts only after independent checks.

Source: [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk); [workspace architecture](https://github.com/OpenHands/docs/blob/main/sdk/arch/overview.mdx).

### OpenAI Agents SDK

Good clean comparator with built-in tool loop, handoffs, guardrails, sessions and tracing. It can test whether DCLab's custom wrapper harms a capable model, especially when the same tools and data are supplied. A good *control* is not necessarily a good full migration target.

Sources: [Agents SDK overview](https://developers.openai.com/api/docs/guides/agents/sdk), [tracing](https://openai.github.io/openai-agents-python/tracing/).

### Microsoft Agent Framework / Google ADK / Strands / Agno

All offer modern agent and multi-agent patterns. Microsoft now explicitly combines agent, workflow and harness capabilities, succeeding AutoGen/Semantic Kernel; Google ADK includes graph/dynamic/collaborative workflows; Strands provides agent-as-tool, graph, swarm and DAG workflow approaches; Agno adds a large operations/runtime suite. Their existence makes it important **not** to claim Deep Agents is the only advanced option. However, DCLab already has core orchestration and needs better scientific intelligence and evidence discipline, not a third graph scheduler.

Sources: [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/), [Google ADK](https://github.com/google/adk-docs/blob/main/docs/workflows/index.md), [Strands](https://strandsagents.com/docs/user-guide/sdk/multi-agent/multi-agent-patterns/), [Agno](https://docs.agno.com/).

### CrewAI / LlamaIndex / smolagents

CrewAI is useful for quickly creating expert-role demonstrations, but it might reinforce rigid role switching. LlamaIndex agents add value where knowledge retrieval and handoffs are dominant. `smolagents` is attractive as a compact code/tool loop for a cheap ablation, though its API is explicitly described as experimental. None is the clear replacement for DCLab's existing scientific/state architecture.

Sources: [CrewAI persisted flows](https://github.com/crewAIInc/crewAI/blob/main/docs/edge/en/concepts/flows.mdx), [LlamaIndex patterns](https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/), [smolagents](https://huggingface.co/docs/smolagents/index).

### DSPy: the most important adjacent tool, not another harness

DSPy optimizers (e.g. GEPA and MIPROv2) can adjust LM-program instructions and demonstrations using an objective function and development examples. For DCLab, use a **validated scientific reasoning rubric** to optimize problem framing and plan proposals rather than guessing at prompt wording or training a model prematurely. Keep a sealed evaluation set and avoid optimization on leaked answers. DSPy does not replace LangGraph or Deep Agents.

Source: [DSPy optimizers](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md).

## 7. Proposed architecture: flexible intelligence, governed execution

```text
                       USER REQUEST + DATA REFERENCES
                                   |
                       [Evidence / Context Profiler]
                       metadata, semantic joins, time,
                       data quality, provenance, known unknowns
                                   |
                   [Open-ended Problem Framing Agent]
                       problem != preselected problem type
              objective, unit, horizon, constraints, uncertainty
                                   |
                   [Hypothesis & Solution Search Agent]
            ML / causal / rule-based / process / diagnosis / abstain
                            2-4 viable candidates
                                   |
                       [Scientific Critic / Reviewer]
             assumptions, leakage, metrics, non-ML alternatives
                                   |
                [Late-binding Plan Compiler and Validator]
             typed DAG, budgets, dependencies, stop conditions
                                   |
       +---------------------------+---------------------------+
       |              DCLab-owned LangGraph runtime            |
       | tool registry | science policy | approvals | retries |
       +---------------------------+---------------------------+
                                   |
                        [Sandboxed Execution Worker]
               native DCLab runner / optional OpenHands worker
                                   |
                   [Independent Artifact & Outcome Check]
                   reproducibility, holdout, error analysis
                                   |
                           [Replan or Deliver]
                                   |
              [Decision Ledger + Verified Experience Memory]
```

### Decision ownership

| Layer | Decision style | Owner |
|---|---|---|
| Initial sensemaking | LLM-driven, hypothesis-rich, allowed to say `not_ml` or `insufficient_evidence` | Problem framer |
| Scientific critique | Model-assisted critique plus explicit deterministic tests | Critic + scientific validators |
| Solution selection | Multi-candidate tradeoff then validated selection | Plan compiler and approved policies |
| Execution permission | Deterministic, fail closed | Existing DCLab policy/registry |
| Code/data access | Bounded, sandboxed and auditable | DCLab runner, optional OpenHands |
| Outcome verification | Reproducible tests + independent evaluation | DCLab evaluation engine |
| Experience reuse | Evidence-linked and applicability scoped | DCLab memory/ledger |

A model may propose a new class of problem; it must **not** invent permission to read the sealed test split or bypass validation. The `ProblemType` should be late-bound metadata and a *hint* to downstream operators, not an early limit on what the task can mean.

### Contracts to preserve

`ProblemFrame`: user objective, decision unit, horizon, known data/joins, candidate causal or predictive questions, assumptions, unknowns, risk, acceptable non-ML option, evidence references.

`SolutionCandidate`: approach family (including none), hypothesis, prerequisites, data contract, estimation strategy, validation protocol, cost, uncertainty, reason for rejection/selection.

`ExperimentTask`: task ID, plan ID, dependency IDs, allowed data refs, tool permissions, expected artifact, verifier, budget, retry/stop conditions, provenance and approval tier.

`WorkerResult`: typed status, verified artifact refs, metrics with split IDs, code/environment fingerprints, errors, confidence caveats, trace ID, approved state changes. **Do not pass complete private datasets or hidden reasoning through human-readable agent messages.**

## 8. Adoption and integration strategy

### Stage 0 — repair and measure DCLab first

Use the existing four arms:
- A: model only (planning tasks only).
- B: minimal tool-using agent.
- C: current DCLab harness.
- D: DCLab with *only* late-binding / open problem framing changed.

Compare identical model, task brief, evidence snapshot, permissions, budgets where applicable. Instrument where the first important deviation occurs. Keep sealed answers inaccessible. If B beats C, locate the specific cause; if D fixes it, do not replace the framework to claim the gain.

### Stage 1 — framework bake-off without deployment

- E: Deep Agents with the same provider, tools, access, model and case inputs; avoid optional subagents initially, then turn delegation on in an ablation.
- F: Pydantic AI / Harness with the same model and tools; start with a lean typed agent, then add context/planning capability one at a time.
- Optional G: a coding-worker comparison (native DCLab sandbox vs OpenHands SDK vs bounded Codex/Claude Agent SDK), **only on execution/code-authoring tasks**, not business-framing questions.

Never compare A's no-tools run to E/F on end-to-end task completion. Score reasoning/plans separately from execution.

### Stage 2 — integrate only proven capabilities

Possible low-risk incremental integrations:
1. Add a context-snapshot and evidence-reference contract.
2. Add late-bound `ProblemFrame` then `SolutionCandidate` selection.
3. Pilot one Deep Agents supervisor **inside a read-only DCLab boundary**; return structured proposal, not direct writes.
4. Pilot an OpenHands workspace executor on one validated scientific task, with DCLab-controlled data and artifact boundaries.
5. Consider DSPy optimization when a trustworthy dev set and grader exists.

Do not allow Deep Agents or OpenHands direct database credentials, raw holdout access, privileged APIs, unbounded shell/network access, or opaque mutation of DCLab's canonical state. Existing DCLab tracing/export should remain authoritative; bridge external traces by run ID rather than rebuilding two observability platforms.

## 9. Benchmark and safety gate

Pilot with eight development tasks; build at least 30 independently reviewed cases for later comparisons, including ambiguous business questions, multiple linked tables, time-aware prediction, risk/fraud capacity allocation, causal action selection without treatment data, unfamiliar optimization/graph tasks, and recovery from failed tools. **At least one-third should legitimately require no immediate ML model.**

Score separately:
- Problem comprehension (0–4), scientific adequacy (0–4), option breadth/tradeoffs (0–4), plan feasibility (0–4), evidence discipline (0–4).
- Expert-Accepted Plan Rate, Safe Task Completion Rate, first-divergence reason, recovery, unauthorized action count, data leakage count, reproducing exported results.
- Total model and tool cost, tokens, human interventions, P50/P95 latency, useful-output per dollar, invalid steps, provider/model version, source eligibility.

Suggested *engineering acceptance targets* (to pre-register, not empirically established):
- No observed critical authorization, secret exposure, or holdout violations in the declared adversarial suite. **This does not prove the system globally safe.**
- Framework adoption must demonstrate a meaningful paired improvement over **both** the current harness and the minimal agent on the target workload, or materially better recovery/operations at acceptable overhead.
- A practical initial target is at least +10 percentage points on accepted plans or safely completed tasks compared with the existing harness on the wider set; report per-case confidence/uncertainty and do not treat 30 cases as statistical proof.
- No regression on originally solved simple tasks; no unbounded budget/tool loops. Cost/quality decisions should be explicit rather than hidden behind a single composite score.
- Preserve trace replay, reproducible artifacts, data governance and revocation/rollback.

### Root-cause codes

`early_type_binding`, `context_omission`, `summary_evidence_loss`, `poor_model_reasoning`, `tool_visibility`, `invalid_tool_args`, `misleading_gate_feedback`, `workflow_dead_end`, `bad_scientific_choice`, `false_success`, `missing_replan`, `resource_budget`, `unsafe_data_request`, `evaluation_problem`.

Log actual evidence each component saw, the first model proposal, any later rewrite/rejection, tool result, gate reason, artifact hash and final independently verified outcome. A fluent explanation is not proof of correct problem solving.

## 10. Anti-patterns to explicitly avoid

1. Replacing LangGraph with another DAG engine while keeping the same overly restrictive route taxonomy.
2. Replacing `ProblemMap` with unbounded model imagination; permission and scientific invariants remain mandatory.
3. Building five new specialist agents before proving a single strong, tool-enabled agent is insufficient.
4. Copying full tool output or private datasets into prompts instead of referenced evidence.
5. Persisting generated plans as verified experience without independent grading.
6. Calling a framework 'more intelligent' because it has planning, memory or multi-agent API methods.
7. Using expert-judge preference alone: compare actual execution correctness and the absence of scientific errors.
8. Measuring only final answers while losing the first divergence in model observations and harness interventions.
9. Fine-tuning on model-provider outputs whose training rights have not been established, or on sealed evaluation data.
10. Using Deep Agents' filesystem path permissions as a substitute for project authorization and holdout protection.

## 11. Research-to-implementation decision record

| Question | Proposed answer | Confidence |
|---|---|---|
| Keep LangGraph? | Yes, unless the runtime audit demonstrates a fundamental missing requirement. | High architectural fit, not a benchmark result |
| Is Deep Agents a good fit? | Yes for scoped long-running reasoning, context offload and specialist delegation. | High feature fit; DCLab quality gain untested |
| Make Deep Agents the new complete DCLab harness now? | No. | High confidence in need for ablations first |
| Evaluate Pydantic AI Harness? | Yes; useful counterfactual and typed-agent option. | High feature fit; outcomes untested |
| Use OpenHands as primary problem-solving brain? | No; use as a scientific coding worker after a plan is approved. | High conceptual fit for specialization |
| Start fine-tuning now? | Not as the default fix for current harness failures. First build eligible evidence and a baseline. | High risk-reduction argument |
| Use DSPy? | Yes, after creating reliable measurable graders and development examples. | High conceptual fit; gain untested |
| Adopt CrewAI/Agno/Microsoft/ADK as DCLab core? | Not without a specific gap current runtime cannot address. | Moderate, based on migration costs and overlap |

**Proposed architecture direction:** *dynamic scientific intelligence above a deterministic, governed execution substrate.* Framework adoption should be modular, reversible, and earned through task-level improvements.

## 12. Verified official sources (checked 15 November 2026)

- LangChain Deep Agents: https://docs.langchain.com/oss/python/deepagents/overview
- LangGraph: https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/overview.mdx
- Pydantic AI Harness: https://github.com/pydantic/pydantic-ai/blob/main/docs/harness/index.md
- Pydantic AI durable execution: https://github.com/pydantic/pydantic-ai/blob/main/docs/durable_execution/overview.md
- OpenHands SDK: https://github.com/OpenHands/software-agent-sdk
- OpenAI Agents SDK: https://developers.openai.com/api/docs/guides/agents/sdk
- Microsoft Agent Framework: https://learn.microsoft.com/en-us/agent-framework/overview/
- Google ADK: https://github.com/google/adk-docs/blob/main/docs/workflows/index.md
- Strands Agents: https://strandsagents.com/docs/user-guide/sdk/multi-agent/multi-agent-patterns/
- CrewAI: https://github.com/crewAIInc/crewAI/blob/main/docs/edge/en/concepts/flows.mdx
- Agno: https://docs.agno.com/
- LlamaIndex agents: https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/
- smolagents: https://huggingface.co/docs/smolagents/index
- DSPy: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
- Temporal: https://docs.temporal.io/ai/cookbook
- Langfuse evaluation: https://langfuse.com/docs/evaluation/overview
- Claude Agent SDK: https://github.com/anthropics/claude-agent-sdk-python
- Codex SDK: https://learn.chatgpt.com/docs/codex-sdk

### Relevant existing DCLab local plans

- `DCLab_Harness_Evaluation_Plan_EN.md`
- `DCLab_Intelligence_Recovery_Plan_2026-11-15.md`
- `DCLab_Top_Five_Strategic_Gap_Groups_2026-11-09.md`
- `DCLab_Hierarchical_Planning_V2_Additive_Implementation_Plan.md`
- `DCLab_Agentic_ML_Full_Implementation_Plan_v2.md`

These prior documents are inputs to this assessment, **not evidence of successful live implementation**. The first action remains controlled evaluation, not migration.

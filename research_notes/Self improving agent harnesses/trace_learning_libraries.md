# Trace-learning libraries: agents that improve by changing context, prompts, memory or skills (not weights) — state as of late 2026

Scope note: researched 2026-10-23. Items marked **[recalled]** come from the paper's abstract as I know it, with the arXiv link given, but were not re-fetched in this session; treat their exact numbers as needing a check before they go into a decision document. Items without that tag were read from the cited page in this session.

## Q1. What each method learns, how it updates, and what signal it needs

### Takeaway
The methods split into three families: (a) **search optimizers** that rewrite a prompt/program and keep the best by a score (GEPA, MIPROv2, SIMBA, TextGrad, Trace/OptoPrime); (b) **accumulating context/memory** that appends lessons and reads them back (Reflexion, ExpeL, AWM, Dynamic Cheatsheet, ACE, Mem0/Zep/LangMem/Letta); (c) **skill/code libraries** (Voyager, Agent Skills, 2026 skill-evolution papers). Only ACE explicitly uses append-only *delta* updates with helpful/harmful counters — the exact shape DCLab's plan already describes.

### Cited Findings
**Comparison table** (artifact · update rule · signal)

| Method | Artifact learned | Update rule | Signal needed |
|---|---|---|---|
| GEPA | Prompt text of each module (via `optimize_anything`: any text artifact — code, configs, agent architectures) | Rewrite of a candidate by a reflection LM; keeps a **Pareto frontier** of candidates and merges lessons across it | Scalar score per example **plus textual feedback** (diagnostics, errors) |
| ACE | Playbook of bullets (ID, `helpful=`, `harmful=` counters, content) | **Append-only deltas** from a Curator, merged by non-LLM code; embedding de-dup; refine after each delta or lazily | Execution feedback; works with or without ground-truth labels (better with) |
| Dynamic Cheatsheet | One evolving cheatsheet | LLM rewrite of the whole sheet | Self-assessed / execution feedback |
| MIPROv2 (DSPy) | Instructions + few-shot demos | Bayesian search over proposed candidates | Scalar metric on a train/val set |
| TextGrad / Trace (OptoPrime) | Any text variable (prompt, code, solution) in a computation graph | LLM "gradient" (textual critique) → rewrite | Textual feedback or score on the output |
| Reflexion | Episodic verbal self-reflections in a buffer | Append reflections, retry the same task | Binary/scalar success or test results |
| ExpeL | Natural-language "insights" + stored successful trajectories | Insight list edited with ADD/EDIT/UPVOTE/DOWNVOTE ops; trajectories retrieved as examples | Success/failure, ideally success–failure pairs on the same task |
| Agent Workflow Memory | Reusable "workflows" (abstracted sub-routines) | Induced offline from training examples or online from test queries; added to memory | Success judgment (ground truth or an LLM evaluator) |
| Voyager | Executable code skills (JavaScript) in a library indexed by description | Add a skill only after self-verification passes | Environment feedback + self-verification |
| Agent Skills (Anthropic) | Folders with `SKILL.md` (+ scripts, resources) | Authored by people (or a model) and versioned; nothing learns them automatically by itself | None built in — it is a packaging/loading format |
| Mem0 / Zep(Graphiti) / LangMem / Letta | Facts/memories (Mem0: extracted facts; Zep: temporal knowledge graph; LangMem: semantic/episodic + procedural prompt updates; Letta: editable memory blocks) | LLM decides ADD/UPDATE/DELETE (Mem0); graph edges with validity times (Zep) | Conversation content; no grader needed |

Sources per row:
- GEPA: samples trajectories, reflects in natural language, proposes and tests prompt updates, combines lessons from the Pareto frontier — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457). Adapter needs only `evaluate` and `make_reflective_dataset`; `optimize_anything` optimizes any text artifact with an evaluator and `oa.log` diagnostic feedback — [gepa-ai/gepa](https://github.com/gepa-ai/gepa). Multi-metric scores tracked on a Pareto frontier so specialised strengths are kept — [search summary of optimize_anything docs](https://www.mintlify.com/gepa-ai/gepa/api/optimize-anything).
- ACE: Generator flags which bullets were useful or misleading; Reflector extracts lessons from successes and errors (optionally several rounds); Curator writes compact delta entries "merged into the context by non-LLM logic"; bullets have ID and helpful/harmful counters; semantic-embedding de-duplication; refinement after every delta or lazily when the window is exceeded — [arXiv 2510.04618v3 (HTML)](https://arxiv.org/html/2510.04618v3). Repo playbook sections ("Strategies & Insights", "Formulas & Calculations", "Common Mistakes"), bullet IDs like `[str-00001]` — [ace-agent/ace](https://github.com/ace-agent/ace).
- ACE labels: "adapts using natural execution feedback", labelled supervision not required — [arXiv 2510.04618](https://arxiv.org/abs/2510.04618).
- AWM: induces workflows, offline from training examples or online on the fly from test queries — [arXiv 2409.07429](https://arxiv.org/abs/2409.07429).
- Agent Skills: `SKILL.md` with required `name` (≤64 chars) and `description` (≤1,024 chars); three levels of progressive disclosure (metadata ~100 tokens per skill always loaded; body <5k tokens when triggered; resources and scripts loaded only when used, script code never enters context) — [Anthropic Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview). The docs describe creation by people in Claude Code / API upload / claude.ai; no automatic learning is described — same source.
- 2026 skill-evolution work treats skills as trainable state edited from trajectories: SkillBoost "localizes observed failures to editable skill components", generates repair candidates, and commits a candidate only if it improves performance "within a regression bound" — [arXiv 2607.26643](https://arxiv.org/abs/2607.26643). MUSE-Autoskill (skill creation, memory, management, evaluation; evaluated on SkillsBench) — [arXiv 2605.27366](https://arxiv.org/html/2605.27366v1).
- [recalled] Reflexion stores verbal reflections in an episodic buffer and retries — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366). ExpeL gathers experiences, extracts insights, and recalls them plus successful trajectories at inference — [arXiv 2308.10144](https://arxiv.org/abs/2308.10144). TextGrad backpropagates textual feedback through a compound system — [arXiv 2406.07496](https://arxiv.org/abs/2406.07496). Trace/OptoPrime optimizes code, prompts and parameters from execution traces and feedback — [arXiv 2406.16218](https://arxiv.org/abs/2406.16218). MIPROv2 jointly optimizes instructions and demos with Bayesian optimization — [arXiv 2406.11695](https://arxiv.org/abs/2406.11695). Voyager's skill library of executable code with self-verification — [arXiv 2305.16291](https://arxiv.org/abs/2305.16291).
- Mem0: extracts and consolidates salient facts; graph variant exists — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413).

### Inferences
- DCLab's planned levels map cleanly: **case memory** ≈ ExpeL's stored trajectories / AWM workflows / Reflexion's buffer; **playbook bullets with counters** ≈ ACE's design almost one-to-one; **person-approved code skills** ≈ Voyager-style library packaged in the Agent Skills format, with SkillBoost-style "accept only within a regression bound".
- Search optimizers (GEPA, MIPROv2) produce a *new prompt version*; they are offline compilers, not memory. They fit "improve the system prompt of role X between releases", not "remember this case".

### Gaps
- DSPy's optimizer docs page did not render via fetch; I could not verify SIMBA's current mechanism or cost from the source (my recollection: SIMBA samples mini-batches, finds high-variance examples, and appends rules or demos via self-reflection — unverified).

## Q2. Reported gains, rollouts and cost; replications and critiques

### Takeaway
Gains on public benchmarks are real but modest-to-medium (roughly +6 to +17 points), mostly self-reported, and highly dependent on the strength of the reflecting model and on reliable feedback. The best-documented cost comparison favours ACE over GEPA on rollouts; GEPA is cheap relative to RL. Independent 2026 studies say self-evolution is sparse, validation-filtered search, not steady improvement.

### Cited Findings
**GEPA**
- Beats GRPO by 6% on average (up to 20%) using up to 35× fewer rollouts; beats MIPROv2 by >10% (e.g. +12% on AIME-2025); six tasks; ICLR 2026 Oral; v2 revised 14 Feb 2026 — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457).
- Repo claims 50+ production uses (Shopify, Databricks, Dropbox, OpenAI, Pydantic), a "90x cost reduction at Databricks" and "55% to 82% resolve-rate" on Jinja; ~7.1k stars — [gepa-ai/gepa](https://github.com/gepa-ai/gepa). These are vendor/author claims, not independently checked.
- optimize_anything claims cloud-scheduling cost cuts ~40% (CloudCast) — author claims; companion artifact for an ACM CAIS 2026 paper — [optimize-anything-artifact](https://github.com/gepa-ai/optimize-anything-artifact).

**ACE** (DeepSeek-V3.1 for all three roles)
- AppWorld average: ReAct 42.4 → ICL 46.0, GEPA 46.4, ACE 59.4 (offline, with labels); ACE 57.2 offline without labels; online without labels Dynamic Cheatsheet 51.9 vs ACE 59.5 — [arXiv 2510.04618v3](https://arxiv.org/html/2510.04618v3).
- Finance (FiNER+Formula) base 69.1 → MIPROv2 70.9, GEPA 72.5, ACE 81.9 (offline with labels); ACE 77.1 without labels; **online without labels ACE 72.9, Dynamic Cheatsheet 65.4 (−3.7, below baseline)** — same source.
- Cost: vs GEPA on AppWorld offline, 82.3% lower adaptation latency (9,517 s vs 53,898 s) and 75.1% fewer rollouts (357 vs 1,434); vs Dynamic Cheatsheet on FiNER online, 83.6% lower token cost ($2.9 vs $17.7) — same source.
- Weaker models gain less (tested also GPT-OSS-120B, GPT-5.1, Llama-3.3-70B) — same source. ICLR 2026; v3 29 Mar 2026 — [arXiv 2510.04618](https://arxiv.org/abs/2510.04618).
- Note: ACE's GEPA/MIPROv2 baselines are run by ACE's authors on ACE's tasks — not an independent comparison.

**AWM**
- +24.6% relative on Mind2Web, +51.1% relative on WebArena; fewer steps; online margin grows 8.9 → 14.0 absolute points as train–test gap widens; Sept 2024 (older item) — [arXiv 2409.07429](https://arxiv.org/abs/2409.07429).

**Memory layers**
- Mem0: +26% relative LLM-as-judge vs OpenAI memory on LoCoMo; 91% lower p95 latency and >90% token savings vs full context; graph variant ~+2%; April 2025 — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413).
- LoCoMo is contested: Mem0's table scores Zep at 65.99; Zep corrected an earlier ~84% claim to 75.14 after a denominator error was pointed out; the same Zep system ranges 58.44–94.7 depending on who runs it (answer model, grader model, grading prompt, dropped questions); QA set is only 10 conversations with documented gold-answer flaws — [braindetox analysis](https://braindetox.kr/en/posts/memory_benchmark_scores_reliability_2026.html); [DEV Community](https://dev.to/gde03/the-ai-memory-benchmark-everyone-quotes-forbids-saying-i-dont-know-o1n); [Mem0 blog (vendor)](https://mem0.ai/blog/zep-vs-mem0-which-ai-memory-layer-should-you-choose). I found no independent third-party reproduction.

**2026 critiques and replications**
- "Rethinking Self-Evolving Agent Skills" (Jul 2026): self-evolution is "sparse, validation-filtered search with model- and benchmark-dependent returns"; only 55 of 388 candidate skills produced a distinct validation best; validation chose an evolved skill in 11 of 14 settings, and 9 of those improved the released test; all 11 selected skills came from feedback that **included failed trajectories**; validation and downstream test/transfer sometimes disagreed; 42 runs, 14 settings, 5 benchmarks, 3 models — [arXiv 2608.02636](https://arxiv.org/abs/2608.02636).
- SkillBoost (Jul–Aug 2026) names **skill overfitting**: overexploiting a small batch of trajectories "overfits the current batch", while unconstrained exploration regresses already-solved cases; fix is verified acceptance within a regression bound; 23 model-benchmark configurations; no headline numbers in the abstract — [arXiv 2607.26643](https://arxiv.org/abs/2607.26643).
- Evo-Memory (Nov 2025, v2 May 2026): streaming benchmark of self-evolving memory, 10 datasets, >10 memory modules, proposes ExpRAG baseline and ReMem — [arXiv 2511.20857](https://arxiv.org/abs/2511.20857). Process-level evaluation of self-evolving agents — [arXiv 2609.24663](https://arxiv.org/pdf/2609.24663).
- [recalled] Reflexion: 91% pass@1 on HumanEval vs 80% for GPT-4 — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366) (2023, older). Voyager: 3.3× more unique items, tech-tree milestones up to 15.3× faster — [arXiv 2305.16291](https://arxiv.org/abs/2305.16291) (2023). TextGrad: GPQA zero-shot from 51% to 55% with GPT-4o; ~20% relative gain on LeetCode-Hard — [arXiv 2406.07496](https://arxiv.org/abs/2406.07496). MIPROv2: up to 13% accuracy over baselines on Llama-3-8B — [arXiv 2406.11695](https://arxiv.org/abs/2406.11695).

### Inferences
- The ACE "online without labels" finance row is the key warning for DCLab: without a trustworthy grader, accumulating lessons can push accuracy **below the no-learning baseline**. DCLab's deterministic grader with hidden keys is exactly the safeguard this evidence calls for.
- Expect small, uneven gains on DCLab's own tasks; a sealed set read once is the only way to tell a real gain from validation luck (the 2608.02636 result: validation and test disagree sometimes).
- Rollout budgets: GEPA runs of 100–150 metric calls are typical in examples; ACE used 357 rollouts on AppWorld. At Sonnet prices with long traces, one GEPA run on an agentic task can easily cost tens of euros; ACE-style per-episode reflection is cheaper per step but runs forever.

### Gaps
- No independent replication of ACE's or GEPA's headline numbers found in this session.
- ExpeL's exact numbers not re-verified (recalled: gains over ReAct on HotpotQA, ALFWorld, WebShop; transfer to FEVER).
- No published cost figures in euros for a Claude-based GEPA or ACE run.

## Q3. Use as a library on top of an existing harness; model support; license; maturity

### Takeaway
GEPA (MIT) is the most mature drop-in: its adapter is two functions over your own system, so DCLab can keep its harness. ACE's repo (Apache-2.0) is research code with limited providers; its *algorithm* is simple enough to reimplement inside the harness in a few hundred lines, which is likely the better route. Agent Skills is a format, not a learner. Memory layers are libraries/services with their own LLM calls — useful for storage/retrieval, not for grading.

### Cited Findings
- **GEPA**: MIT; implement `GEPAAdapter.evaluate` + `make_reflective_dataset` to plug in any system; `max_metric_calls` caps budget; needs a `reflection_lm`; models referenced by LiteLLM-style strings and the README says Claude/GPT/Gemini work through their APIs; also `dspy.GEPA` — [gepa-ai/gepa](https://github.com/gepa-ai/gepa). PyPI version 0.1.4 (15 Jul 2026) per a mirror; conflicting listings, verify on PyPI — [CERN PyPI mirror](https://simple-repository.app.cern.ch/project/gepa).
- **ACE**: Apache-2.0; Python; `ACE(...).run()` in offline/online/eval_only modes; new tasks need JSONL splits plus a DataProcessor (`process_task_data`, `answer_is_correct`, `evaluate_accuracy`); documented providers sambanova, together, openai, commonstack — no Anthropic or Ollama mentioned; ~1.4k stars, 12 commits, latest dated news "2025 Nov" — [ace-agent/ace](https://github.com/ace-agent/ace). Feeding existing traces directly is not documented — same source.
- **Agent Skills**: supported on the Claude API (needs the code-execution tool; skills run with no network and no package install; `/v1/skills` endpoints with versions), Claude Code (filesystem `.claude/skills/`, full network), claude.ai; custom skills do not sync across surfaces; Anthropic warns to use skills only from trusted sources because a malicious skill can direct tool calls or exfiltrate data; Skills are **not covered by ZDR** — [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).
- **Memory layers**: Mem0 paper and benchmarks above — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413). [recalled] Licenses: Mem0 Apache-2.0, Graphiti (Zep's open-source engine) Apache-2.0, LangMem MIT, Letta Apache-2.0, DSPy MIT, TextGrad MIT, Trace MIT — verify on each repo before relying on it.

### Inferences
- **Bring-your-own model calls**: GEPA (via adapter, and LiteLLM routes Anthropic and local Ollama/vLLM) — yes. DSPy optimizers — only if the role is written as a DSPy program, which means partly re-hosting the role inside DSPy; heavier fit. TextGrad/Trace — need their own graph/`bundle` wrappers around the code; heavier fit. ACE — reimplement: Reflector and Curator become two prompts sent through DCLab's gateway (cost caps, trace capture kept), merge/dedup is deterministic DCLab code.
- **Local Qwen**: ACE reports smaller gains with weaker models, and depends on "a reasonably strong Reflector". Sensible split: Claude Sonnet 5.5 as Reflector/Curator (rare, capped), local Qwen as the actor that reads the playbook.
- Mem0/Zep add a second LLM pipeline that decides what to store — that conflicts with DCLab's rule that a deterministic grader decides what is a lesson. Use at most as a storage/retrieval backend, or skip; DCLab already has PostgreSQL and lexical evidence search.

### Gaps
- Could not confirm whether GEPA's newest release has a native trace-import format (e.g. OpenTelemetry) — it may need a converter from DCLab's trace records.
- Maintenance status of the ACE repo after Nov 2025 not confirmed.

## Q4. Risks: overfitting to validation, prompt bloat, held-out leakage, memory poisoning

### Takeaway
Every one of the four risks is documented. The mitigations the literature converges on — deterministic merge, counters, de-duplication, a regression bound, acceptance only on a validation set, failures included in feedback — are the same ones DCLab has planned; the one DCLab must add explicitly is a firewall that keeps hidden keys and sealed-set content out of anything the reflector writes.

### Cited Findings
- **Context collapse** (the risk DCLab names): on AppWorld a context of 18,282 tokens at 66.7 accuracy was rewritten into 122 tokens and accuracy fell to 57.1, below the 63.7 no-adaptation baseline; ACE's answer is delta entries merged by non-LLM code — [arXiv 2510.04618v3](https://arxiv.org/html/2510.04618v3). Also "brevity bias": optimizers favour short summaries that drop domain detail — [arXiv 2510.04618](https://arxiv.org/abs/2510.04618).
- **Prompt bloat**: ACE prunes via embedding de-dup and lazy refinement when the window is exceeded; some tasks (HotPotQA, Game of 24) gain little from long contexts — [arXiv 2510.04618v3](https://arxiv.org/html/2510.04618v3). Agent Skills avoids bloat by progressive disclosure (~100 tokens per unused skill) — [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).
- **Overfitting to the batch/validation set**: skill overfitting and regression on solved cases — [arXiv 2607.26643](https://arxiv.org/abs/2607.26643); validation and test sometimes favour different feedback conditions — [arXiv 2608.02636](https://arxiv.org/abs/2608.02636).
- **Bad feedback makes things worse**: without labels or execution signals ACE and Dynamic Cheatsheet degrade on finance; a weak Reflector produces noisy or harmful context — [arXiv 2510.04618v3](https://arxiv.org/html/2510.04618v3).
- **Poisoning / untrusted content**: a malicious skill can direct tool calls or exfiltrate data; skills fetching external content can be compromised — [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).
- **Benchmark leakage/quality in memory evals**: LoCoMo flaws and grader sensitivity — [braindetox analysis](https://braindetox.kr/en/posts/memory_benchmark_scores_reliability_2026.html).

### Inferences
- **Held-out leakage** is the specific DCLab danger: the Reflector sees the grader's verdict; if the verdict text contains the hidden answer ("correct target was `churned_90d`, leak column was `final_fare`"), the lesson will memorise the key, and the next variant of the same case is "solved" by recall, not by skill. Mitigations: grader emits only categories/codes to the Reflector (e.g. `LEAK_MISSED:time-after-cutoff`), never the key values; a deterministic filter rejects any lesson containing column names, values or IDs from a hidden key; sealed variants never enter the learning loop's trace store.
- **Memory poisoning** inside DCLab comes less from attackers than from tester sessions and model-written lessons. Keep provenance on every bullet (source case, run id, grader version), quarantine new bullets until N helpful counts, and make removal one audited switch.
- Counters can be gamed by correlation: a bullet present in every successful run gets "helpful" even when unused. ACE asks the Generator to tag which bullets it used; DCLab should do the same and also run periodic ablations (drop a bullet, re-run a small dev set).

### Gaps
- No paper found that measures answer-key leakage into learned playbooks directly; the risk is my inference from how the loops work.

## Q5. Best combination for DCLab levels 1–2 (case memory + playbook)

### Takeaway
Build levels 1–2 **inside the existing harness** using ACE's design (Generator tags bullets → Reflector → Curator emits deltas → deterministic merge with counters and de-dup), with ExpeL/AWM-style case memory retrieved by problem kind; keep GEPA (MIT) as an occasional **offline** prompt compiler for a role's base prompt, judged on the sealed set once. Do not adopt a memory vendor (Mem0/Zep) for this; their benchmarks are disputed and they add an ungoverned LLM writer. Agent Skills is the right *packaging* for level-3 person-approved code skills, not for levels 1–2.

### Cited Findings
- ACE beat GEPA and MIPROv2 on agent and finance tasks at lower rollout cost, and its delta + counter design targets context collapse — [arXiv 2510.04618v3](https://arxiv.org/html/2510.04618v3).
- ACE repo is Apache-2.0 but limited to a few providers and lightly maintained — [ace-agent/ace](https://github.com/ace-agent/ace).
- GEPA's adapter fits any system and caps metric calls — [gepa-ai/gepa](https://github.com/gepa-ai/gepa); GEPA outperforms MIPROv2 and GRPO with few rollouts — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457).
- AWM workflows induced from past successes generalise better as the train–test gap grows — [arXiv 2409.07429](https://arxiv.org/abs/2409.07429).
- Feedback that includes failures produced every selected evolved skill — [arXiv 2608.02636](https://arxiv.org/abs/2608.02636); accept changes only within a regression bound — [arXiv 2607.26643](https://arxiv.org/abs/2607.26643).

### Inferences
Suggested shape (simple, step by step):
1. **Level 1 — case memory.** After each graded attempt, store a case record: problem kind, data-card bands, the agent's key decisions, the grader verdict *as codes*, and a short lesson. Retrieve the top-k cases by kind + band similarity (DCLab's lexical search is enough to start). Real-life example: a doctor remembering "the last three patients with these symptoms" before diagnosing.
2. **Level 2 — playbook.** A Reflector (Claude, capped) turns a batch of verdicts — failures included — into candidate bullets; a Curator outputs only ADD/UPDATE-counter deltas; DCLab code merges, de-duplicates, and never lets a model rewrite the whole playbook (prevents context collapse).
3. **Gate.** A bullet becomes "active" only after it raises a dev-set score within a regression bound; the sealed variant set is read once per playbook version.
4. **Firewall.** Grader → Reflector passes codes, never hidden-key values; a deterministic check rejects lessons that mention key columns or IDs.
5. **Occasional GEPA.** Every few weeks, run GEPA on one role's base prompt with `max_metric_calls` set to the budget; it outputs a new prompt version that goes through the existing prompt-hash test and harness eval.
- Cost guess (unverified): the per-attempt Reflector + Curator call is a few thousand tokens; with prompt caching on the playbook this should be cents per attempt on Sonnet; GEPA runs are the expensive part and should stay rare.
- Risk to flag to the owner honestly: published gains come mostly from the methods' own authors on public benchmarks; DCLab should expect smaller and noisier gains on its own tasks, and should treat "no measurable gain on the sealed set" as a likely outcome to plan for.

### Gaps
- No direct evidence on whether ACE-style playbooks help small local Qwen actors on tabular-ML decision tasks; ACE only shows "smaller gains on weaker models".
- No measured comparison of ACE vs ExpeL-style case retrieval on the same tasks in this session's sources (Evo-Memory may contain one; its results were not visible in the abstract).

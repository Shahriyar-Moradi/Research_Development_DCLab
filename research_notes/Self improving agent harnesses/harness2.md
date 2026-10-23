# Harness² (google-research/harness2): what it is, how it works, what it achieved, and fit to DCLab

Checked on 2026-10-23. Primary sources: the repository (README, docs, source at commit `1cfe254e`), the arXiv paper 2610.14224 (HTML version), and the project site. Secondary sources (blogs) are marked. The paper's HTML was read in two parts. A middle part of the appendix (some of D, plus figure values) did not come through, so a few numbers below are marked "not in the readable text".

## 1. What "recursive agent harnessing" is: Propose–Probe–Compose, the improvement harness, the eight components, and judging without ground truth

### Takeaway
Harness² keeps the model fixed and edits the files around it (the "harness"). It works at two levels:
- **Per task:** it tries two edited harnesses on the same task and compares how they behaved against the unedited run. A composer then merges the edits that the evidence supports.
- **Across tasks:** a library keeps each adopted harness and its evidence-backed lessons for later tasks.

It never sees a score while it improves. The only signals are behavioural contrasts between runs, and the paper admits these are noisy.

### Cited Findings
- **Definitions.** A *harness* is everything around the model: prompts, tools, memory, skills and orchestration. *Harnessing* means optimising that, with the weights frozen. The paper is "Harness^2: Recursive Agent Harnessing for an Open World" by Xu, Chen, CuiZhu, Dalvi Mishra, Ming, Yu, Han, Lee and Pfister, submitted 2026-10-09. — [arXiv](https://arxiv.org/abs/2610.14224); [README](https://github.com/google-research/harness2)
- **Authors' institutions.** The site lists Northwestern University and Google Cloud AI Research. — [harness2.github.io](https://harness2.github.io)
- **The problem it targets.** The abstract names two problems with recursive self-improvement (RSI) in open-world settings. Tasks are unseen and have no ground-truth verifier, and RSI tends to be task-specific and to generalise poorly. — [arXiv](https://arxiv.org/abs/2610.14224)
- **Reference step.** The current harness runs once on the task. This run shows *how* the harness behaves, not whether its output is correct. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Propose.** Two candidate harnesses are written by editing the component files:
  - an **invariant** (INV) candidate fixes general agent weaknesses (planning, tool use, context management, delegation);
  - a **domain** (DOM) candidate adds domain knowledge, procedures and constraints.
  - The editor prompt says each edit must trace back to a specific moment in a trajectory. It also says to prefer skills, then scripts, subagents and plugins over always-on text, and to keep always-on text short. "KEEP_BASE" is an allowed answer. — [paper HTML, App. H](https://arxiv.org/html/2610.14224)
- **Probe.** Each candidate runs on the *same* task with the same model and environment. Its trajectory and deliverable are compared with the reference to see whether the targeted weakness was fixed, nothing changed, or something got worse. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Compose.** A composer reads the reference, the candidates, the diffs and all task-local evidence. It may:
  - merge the supported edits;
  - revise edits that conflict;
  - adopt one candidate as it is;
  - keep the base harness.

  The composed harness is then executed and becomes the deliverable run. The composer prompt gives repeated failures more weight than single-rollout defects. It treats an edit that came from a rollout that delivered nothing as unsafe. — [paper HTML](https://arxiv.org/html/2610.14224)
- **"Without ground truth" means contrast, not a score.** The paper says the contrasts "inform subsequent composition rather than verify task correctness." It lists execution randomness and shared model biases as limits on how reliable this evidence is. Appendix F says correctness and transfer "are not guaranteed". — [paper HTML](https://arxiv.org/html/2610.14224)
- **Scaling the budget k.** A *parallel* run collects k references, proposes and probes k INV plus k DOM candidates, and runs k independent compositions over the whole pool. A *sequential* run chains k steps, each building on the last harness. For k>1, an "agentic selector" picks among *anonymised* deliverables without seeing their harnesses. It ranks by:
  1. delivery;
  2. the requested shape;
  3. fidelity to the sources;
  4. consistency;
  5. operational fitness.

  It does not reward length. — [paper HTML](https://arxiv.org/html/2610.14224); [EXPERIMENTS.md](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md)
- **The improvement harness (harnessing-level recursion).** After each task, a reflector reads the whole improvement record and updates a per-domain library. Each entry is the full adopted harness tree plus an `EXPERIENCE.md` (situation, lessons, evidence), and an index lists the entries. Lessons are kept only when execution contrasts support them. If the evidence is too thin, nothing is added. The library is frozen while a task runs. Each new task still starts from the base harness H₀ and reuses library parts through Propose. Executable components are stored as they are, not summarised as prose. — [paper HTML, §3 and App. D.2](https://arxiv.org/html/2610.14224)
- **Library separation and order.** Libraries are kept apart by benchmark, executor, mode, budget and seed. Tasks in a domain run in order and read only earlier entries. — [EXPERIMENTS.md](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md)
- **The eight editable components.** They are the system prompt, guardrails, memory, tool guidance, skills, scripts, subagents, and a middleware plugin, grouped as context construction, tool interface, orchestration and middleware control.
  - Under OpenCode, the system prompt, guardrails, memory and tool guidance are always on. Skills load on demand with their scripts. Subagents run through the task tool. Plugins are TypeScript that runs inside the tool loop.
  - Under Codex, the always-on text lives in `AGENTS.md`. — [paper HTML, App. D.3 / Table 5](https://arxiv.org/html/2610.14224); [harness2.github.io](https://harness2.github.io)
  - In code, the editable surface is a file tree "c1..c8" with validators (`SurfaceSpec`). — [substrate/base.py](https://github.com/google-research/harness2/blob/main/harness2/substrate/base.py)
- **Ablation: which components matter.** On Gemini 3.5 Flash / OpenCode, WorkBuddy average (Table 6):
  - base: 72.4;
  - full Harness² (k=5): 80.8;
  - without skills: 75.5;
  - without scripts: 77.1;
  - without the prompt group: 79.3.

  So skills and scripts carry the most weight. — [paper HTML, App. E](https://arxiv.org/html/2610.14224)
- **Ablation: which method parts matter.** Table 3, WorkBuddy / JobBench, k=1:
  - full: 77.5 / 36.0;
  - without the library: 74.4 / 34.2;
  - without scoping (the INV/DOM split): 74.6 / 32.3;
  - without composition: 75.9 / 34.1. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Case study (JobBench, Gemini 3.5 Flash / OpenCode).** One task's score rose from 43.1 to 77.8. The base agent had left its source files unreconciled and never reopened its outputs. The new harness added verification guardrails, a reconciliation skill and sharper tool guidance. — [harness2.github.io](https://harness2.github.io)

### Inferences
- Probe is an A/B test on the same task: same task, same model, one changed harness. The "label" is a model's reading of the two trajectories, not a grader's verdict. That makes the method usable where no answer key exists. It also means a wrong premise can spread: the paper's own failure analysis names "propagated incorrect premises" and "candidate selection errors" among nine regressions.
- Most of the gain comes from adding executable procedure (skills and scripts), not from rewording prompts. This matches Voyager-style skill libraries more than GEPA-style prompt rewriting.

### Gaps
- The full Table 5 file paths and the stage prompts' exact wording were not read in full. Appendix H is summarised, not quoted. The source file `harness2/core/prompts.py` (66 KB) holds them.

## 2. Executors and models: OpenCode, Codex, Vertex AI, GLM-5, Gemini 3.5 Flash, Claude and local models, and plugging in a custom agent

### Takeaway
- **Executors.** The code drives two coding-agent CLIs, OpenCode (the default) and Codex.
- **Public setup.** It is built around Google Vertex AI. WorkBuddy refuses anything but Vertex Gemini.
- **Claude.** The paper's own editor is Claude Opus 4.8, and Claude Opus 5 was tested as an executor. In the shipped code, though, Claude is reachable only as the *improver*, through the Claude Code CLI on Vertex. Direct-Anthropic, OpenRouter and Claude-Code-executor support exist only as open, unmerged PRs.
- **A custom agent.** Plugging in a custom Python agent means writing a "Substrate" (5 methods) and a benchmark adapter (about 12 methods). Your agent must also read its behaviour from the c1..c8 *file tree*.

### Cited Findings
- **Models in the main tables.**
  - Executors (frozen): Gemini 3.5 Flash and GLM-5.
  - Harnesses: OpenCode and Codex.
  - Editor: Claude Opus 4.8 under OpenCode.
  - Graders: JobBench is graded by Opus 4.8 and LAB by Sonnet 4.6. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Same-model study (OpenCode, k=2).** Five models, "from GLM-5 to Claude Opus 5", act as both executor and editor, and all improve on every benchmark. Gains stated in the text:
  - Claude Opus 5: +3.3 (WorkBuddy), +5.5 (JobBench), +3.0 (LAB);
  - GLM-5: +13.7 (LAB).

  The other models' numbers are only in Figure 8 and could not be read. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Requirements.** Linux, Python 3.12 (the package also supports 3.13), uv, Node.js 20+ and the Google Cloud CLI. LAB also needs Pandoc and Poppler. WorkBuddy needs Docker. The local demo (`harness2-demo --output runs/demo --k 2`) uses deterministic editors and needs no model credentials. — [README](https://github.com/google-research/harness2); [SETUP.md](https://github.com/google-research/harness2/blob/main/docs/SETUP.md)
- **Configuration.**
  - Models are set in `.env`: `HARNESS2_SOLVER_MODEL` (OpenCode `provider/model` form), `HARNESS2_IMPROVER_MODEL` (defaults to the executor) and `HARNESS2_JUDGE_MODEL`.
  - Access goes through `gcloud auth application-default login`.
  - SETUP.md says: "The provided configuration uses Vertex Gemini executors and judges, and supports either Vertex Claude or Gemini as the improver." — [SETUP.md](https://github.com/google-research/harness2/blob/main/docs/SETUP.md)
- **Improver channels in code.** There are two: `opencode` and `claude_cli`. The `claude_cli` channel runs the `claude` binary with `CLAUDE_CODE_USE_VERTEX=1` and a Vertex project id. Before starting a session it strips `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` and the Bedrock variables from the environment. Model names must use OpenCode's `provider/model` syntax. — [config.py](https://github.com/google-research/harness2/blob/main/harness2/config.py); [core/proposer.py](https://github.com/google-research/harness2/blob/main/harness2/core/proposer.py)
- **WorkBuddy is Vertex-only.** WorkBuddy setup "refuses to run unless `HARNESS2_SOLVER_MODEL` is a `google-vertex/gemini...` ID", the judge is a bare `gemini...` ID, and `GOOGLE_CLOUD_PROJECT` is set. — [BENCHMARKS.md](https://github.com/google-research/harness2/blob/main/docs/BENCHMARKS.md)
- **Other endpoints.**
  - The JobBench judge can use any OpenAI-compatible endpoint through `HARNESS2_JUDGE_BASE_URL` and `HARNESS2_JUDGE_API_KEY`. — [BENCHMARKS.md](https://github.com/google-research/harness2/blob/main/docs/BENCHMARKS.md)
  - An open issue reports that the judge "cannot route to OpenAI-compatible endpoints" because of a provider whitelist and a Responses-API-only OpenAI path. — [issue #95](https://github.com/google-research/harness2/issues/95)
  - Codex needs a Responses-API endpoint. For Gemini, a bundled `harness2-bridge` (LiteLLM) provides one. Any other Responses-API endpoint can be set with `HARNESS2_CODEX_BASE_URL`, `HARNESS2_CODEX_API_KEY` and `HARNESS2_CODEX_MODEL`. — [SETUP.md](https://github.com/google-research/harness2/blob/main/docs/SETUP.md)
- **Open, unmerged community PRs for other providers and executors** (none had been merged when checked):
  - Claude on Anthropic's API as LAB judge: [#108](https://github.com/google-research/harness2/pull/108)
  - Claude Code without Vertex: [#84](https://github.com/google-research/harness2/pull/84)
  - Claude Code and Pi substrates for JobBench: [#66](https://github.com/google-research/harness2/pull/66)
  - Qwen Code executor: [#123](https://github.com/google-research/harness2/pull/123)
  - DeepSeek executor: [#115](https://github.com/google-research/harness2/pull/115)
  - Closed without merge: OpenRouter support ([#116](https://github.com/google-research/harness2/pull/116)) and a Claude Code executor ([#125](https://github.com/google-research/harness2/pull/125)).
  - Feature requests: Pi, Claude Code ([#64](https://github.com/google-research/harness2/issues/64)) and Hermes ([#117](https://github.com/google-research/harness2/issues/117)).
- **The Substrate interface.** A plug-in executor implements five methods:
  - `surface()` returns the editable c1..c8 tree schema and its validators;
  - `materialise(tree_dir, spec, ctx)` stages the tree as the executor's native files;
  - `execute(task_ref, install, run_ids, timeout)` runs rollouts and returns one run dir each;
  - `read_trajectory(...)` renders the executor's log into a "shared grammar";
  - `preflight()` runs health checks "before money is spent". — [substrate/base.py](https://github.com/google-research/harness2/blob/main/harness2/substrate/base.py)
- **The benchmark adapter.** A task source implements `tasks`, `domain_of`, `task_prompt`, `task_files`, `execute`, `delivered`, `trajectory`, `deliverable_files`, `deliverable_text`, `base_harness`, `validate_tree` and `score` (which may return `None`, meaning ungraded). — [core/spec.py](https://github.com/google-research/harness2/blob/main/harness2/core/spec.py)
- **A hard-coded branch.** A contributor notes that substrate choice in the LAB and WorkBuddy adapters is "a binary `oc`/`cx` branch", so a new executor also touches `cli/_common.py`, `workbuddy.py`, `jobbench.py` and `lab.py`. — [issue #64](https://github.com/google-research/harness2/issues/64)
- **Own tasks.** There is no way to run on your own folder of tasks yet. A `--bench local` proposal is open. — [issue #5](https://github.com/google-research/harness2/issues/5)

### Inferences
- **Claude as executor.** OpenCode is provider-agnostic, so `HARNESS2_SOLVER_MODEL=anthropic/<model>` *may* work for LAB or JobBench through OpenCode. This was not tested and is not documented; WorkBuddy blocks it.
- **A local model through Ollama or an OpenAI-compatible server.** This is plausible for the OpenCode executor, which accepts OpenAI-compatible providers. It is not a documented path. The improver needs strong agentic file editing, and the paper used Opus 4.8. A small local model as improver is untested.
- **DCLab's own agent.** It is a Python loop over a tool registry, not a CLI that reads `AGENTS.md`, skills folders and TypeScript plugins. To use the code, DCLab would have to expose its prompts, playbook, skills and guardrails as a file tree, and write a substrate plus an adapter. That is a few days of plumbing, *before* any of the method is tried. My estimate, not measured.

### Gaps
- No source confirms that a direct Anthropic API key or an Ollama model works end-to-end in the released code.
- The values in the same-model study's Figure 8 could not be read.

## 3. Benchmarks, results, baselines, costs, runs and variance

### Takeaway
- **Size of the gains.** One step (k=1) adds +1.2 to +11.8 points over the base harness, depending on model, executor and benchmark. Five parallel steps (k=5) add a further +2.6 to +6.7. The headline numbers (+11.5 WorkBuddy, +4.5 JobBench, +11.8 LAB) are the *best* configurations, not averages.
- **Where gains are largest.** On the weaker model (GLM-5). On the strongest executor tested (Opus 4.8), the gains fall to +1 to +3.
- **Noise.** There are three runs per cell with SDs of about 0.6 to 2.5. Several "best method" differences are within one SD.
- **Cost.** The paper reports no absolute token or dollar cost. Per task, the run uses about 4k executor rollouts and 3k improver sessions.

### Cited Findings
- **The three benchmarks:**
  - **WorkBuddy-Bench** (Tencent): Code, Web and Office tasks. The test halves have 40, 35 and 25 tasks. Scoring is a composite verifier (deterministic checks plus LLM/VLM judge items), and the score is the fraction of checks passed.
  - **JobBench**: 65 professional tasks in 7 occupational domains, split 31 train / 34 test. Scored against a weighted rubric by Opus 4.8.
  - **LAB** (Harvey Legal Agent Benchmark): 1,251 tasks in 24 practice areas. The paper uses a 100-task test set and the per-criterion pass rate ("Partial"), judged by an LLM. — [paper HTML](https://arxiv.org/html/2610.14224)
  - The repo's LAB stream contains 1,143 tasks. — [EXPERIMENTS.md](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md)
- **Protocol.** Results are the mean (SD) over **three runs** with shuffled task order. Harness² and the online baselines process only the *test* tasks, as a stream. Only Meta-Harness uses the training halves. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Baselines:**
  - Base harness;
  - **ReasoningBank**: puts distilled strategies from past trajectories into the prompt and never edits the harness;
  - **Meta-Harness**: an offline search over harness code on the training split, whose best result is then deployed frozen;
  - **Harness Scaling**: revises the harness separately for each task, with no carry-over between tasks.
  - A best-of-k baseline is defined in Appendix C, but no results for it were in the readable text.
  - GEPA, ACE and the Darwin Gödel Machine are related work only, not baselines. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Table 1, WorkBuddy average.** Format: Base / ReasoningBank / Meta-Harness / Harness Scaling / Harness² k=1, then the extra gain at k=5.

  | Configuration | Base | ReasoningBank | Meta-Harness | Harness Scaling | Harness² k=1 | Extra at k=5 |
  |---|---|---|---|---|---|---|
  | Gemini / OpenCode | 72.4 (0.9) | 73.4 | 76.4 | 75.3 | 77.5 (2.5) | +3.3 |
  | Gemini / Codex | 74.5 (1.6) | 72.5 | 73.3 | 73.8 | 77.5 (2.0) | +2.9 |
  | GLM-5 / OpenCode | 64.2 (1.4) | 65.6 | 69.7 | 67.8 | 75.7 (2.2) | +3.9 |
  | GLM-5 / Codex | 64.4 (1.2) | 64.1 | 74.2 | 62.9 | 75.3 (1.3) | +3.0 |

  The headline "+11.5" is GLM-5 / OpenCode (64.2 → 75.7). — [paper HTML](https://arxiv.org/html/2610.14224)
- **Table 2, base → Harness² k=1:**

  | Configuration | JobBench | LAB Partial |
  |---|---|---|
  | Gemini / OpenCode | 31.5 → 36.0 (+4.5) | 79.9 → 81.9 |
  | Gemini / Codex | 33.9 → 35.3 (+1.4) | 71.6 → 75.6 |
  | GLM-5 / OpenCode | 20.6 → 24.2 (+3.6) | 59.6 → 71.4 (+11.8) |
  | GLM-5 / Codex | 23.6 → 24.8 (+1.2) | 58.1 → 65.9 |

  - On LAB with Gemini / Codex, Meta-Harness leads (76.5).
  - k=5 adds +3.0 / +4.2 / +4.2 / +3.1 on JobBench and +2.6 / +2.9 / +6.7 / +4.8 on LAB, in the four configurations above. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Site table, Gemini 3.5 Flash / OpenCode on WorkBuddy:**
  - Code / Web / Office / Average: base 73.4 / 69.9 / 74.0 / 72.4; k=1 76.5 / 76.4 / 79.5 / 77.5; k=5 80.8 / 78.3 / 83.4 / 80.8.
  - That is +5.1 at k=1 and +8.4 at k=5 over base.
  - The site gives numbers for only this one configuration. — [harness2.github.io](https://harness2.github.io)
- **Variance caveat.** "Several top-method differences are within one SD." Example: WB-Code, Harness² 76.5 (1.3) against Meta-Harness 75.5 (1.7). — [paper HTML](https://arxiv.org/html/2610.14224)
- **Transfer to other executors** (App. E.3):
  - GLM-5: +17.3 on Code (58.5 → 75.8);
  - Gemini 3.7 Flash: +0.3 on Code and +6.4 on Office;
  - Opus 4.8: +2.6 and +1.1. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Against production harnesses** (App. E.5). Starting from OpenCode, Harness² beats the strongest fixed harness by 4.6 / 5.8 / 4.9 points on Code / Web / Office. The authors note this uses extra test-time compute. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Tool use** (Table 7). The tool-error rate falls on all three benchmarks, for example JobBench 14.4% → 9.1%. Tool calls fall on WorkBuddy (20.0 → 17.8) but rise on JobBench and LAB. Model turns per task often fall, for example Gemini / OpenCode WB-Code 29.6 → 22.8. — [paper HTML](https://arxiv.org/html/2610.14224)
- **Cost.**
  - Figure 6 plots total compute (all tokens before submission) against score. The text says both Harness² settings sit on the Pareto frontier, but it gives no absolute token or dollar figures.
  - The paper also says that "performance gains alone do not establish computational efficiency."
  - Appendix F: larger budgets add inference cost, and running in parallel cuts latency, not compute. — [paper HTML](https://arxiv.org/html/2610.14224)
  - The site reports no cost. — [harness2.github.io](https://harness2.github.io)
- **Calls per task.** A community member checked the call counts against the demo's logs. For budget k, each task schedules 4k executor rollouts and 3k improver sessions, plus one reflection call and, for k>1, one selection call. At k=1 that is 4 rollouts and 3 editor sessions. At k=5 it is 20 and 15. — [issue #4 comment](https://github.com/google-research/harness2/issues/4)
  - A blog derives the same 4k rollouts from the algorithm. — [wavect.io (secondary)](https://wavect.io/de/blog/google-harness2-setup-cost-safety.md)
- **Spend accounting in the code is incomplete.**
  - The ledger does not record executor cost: `rollout_usd` is always 0. A fix ([PR #127](https://github.com/google-research/harness2/pull/127)) is unmerged. — [issue #4](https://github.com/google-research/harness2/issues/4)
  - The report undercounts improver spend after interrupted runs and failed modes. — [issue #50](https://github.com/google-research/harness2/issues/50); [issue #98](https://github.com/google-research/harness2/issues/98)
  - There is no spend cap (`--max-usd` is only a proposal), and `--max-tasks` limits tasks per domain, not money. — [issue #4](https://github.com/google-research/harness2/issues/4); [EXPERIMENTS.md](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md)
- **Benchmark licence.** WorkBuddy-Bench has a custom Tencent licence with a territorial limit. — [BENCHMARKS.md](https://github.com/google-research/harness2/blob/main/docs/BENCHMARKS.md)

### Inferences
- **How much of the gain is the harness, and how much the model.** The model is fixed, so all reported gains are harness gains. Compare them with the model gap:
  - The base gap between Gemini and GLM-5 on WorkBuddy is 8.2 points (72.4 against 64.2).
  - One Harness² step shrinks it to 1.8 points (77.5 against 75.7).
  - So for a weak model, the harness can matter about as much as moving to the stronger model.
  - For a strong model (Opus 4.8), the gains are small (+1 to +3).
- **Part of the gain is test-time compute.** Harness² runs about 4× (k=1) to 20× (k=5) as many executor rollouts as the base run. So its edge over Harness Scaling (also per-task) and Meta-Harness (offline) is the fairer comparison. That edge is +1 to +6 points at k=1, and partly within the noise.
- **Possible confound: editor stronger than executor.** In the main tables the editor (Opus 4.8) is stronger than the executors (Gemini 3.5 Flash, GLM-5). Some gain may be the strong model's know-how written into skills, like distillation. The same-model study (all five models improve when they edit for themselves) answers this only partly.
- **Possible bias: grader is the editor.** The JobBench grader is the same model as the editor (Opus 4.8). That could favour outputs in Opus's style. The paper does not discuss it.
- **Small test sets.** JobBench's test set has 34 tasks, so one task is worth about 3 points. A +1.2 to +4.5 gain is a few tasks.

### Gaps
- The absolute cost per task in tokens or dollars is not reported anywhere I found.
- The best-of-k baseline's numbers were not in the readable text. This matters, because best-of-k is the obvious "same compute, no harness editing" control.
- The values of Figure 6 (compute against score) were not readable.
- No independent replication was found. All numbers are the authors' own.

## 4. Safety and evaluation hygiene: sandbox, self-editing guardrails, overfitting, reward hacking, learning vs held-out

### Takeaway
- **Sandbox and network.** The release ships with no network restriction, and the Codex sandbox is off. In the authors' own validation run, a Codex rollout downloaded its task's public rubric unprompted.
- **Guardrails.** They are an editable component that the editor may trim or delete. There is no way to pin them yet; a PR is open.
- **Leak protection.** It covers answer-key *files* and grader *verdicts*, but not rubric text that an executor fetched into its trajectory. A fix is open.
- **Train and test.** There is no train/test split for Harness² itself: it adapts on the very test tasks it is scored on, using no grader signal.

### Cited Findings
- **Network and sandbox.** "Executor network access is unrestricted, and Codex runs with its sandbox disabled." — [README](https://github.com/google-research/harness2)
  - "In a validation run of this release, a Codex rollout on JobBench downloaded its task's rubric without being instructed to."
  - Users are told to block `huggingface.co`, `hf.co`, `*.hf.co`, `github.com` and `raw.githubusercontent.com` at the network level (an egress firewall or DNS), because a tool setting cannot be relied on. — [EXPERIMENTS.md](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md)
- **Guards against reward hacking.**
  - Rubrics, answer keys and scores are never shown to the executor, editor or selector, and the prompts make each stage state that none is in its context.
  - A guard refuses to hand over a working directory that contains a grader verdict, rubric or score.
  - Candidate scripts and plugins are treated as data and are not executed by the improver. — [paper HTML, App. D.2 and H](https://arxiv.org/html/2610.14224)
  - The improver "never sees verifier output". — [BENCHMARKS.md](https://github.com/google-research/harness2/blob/main/docs/BENCHMARKS.md)
- **A leak path that is still open.** If the executor fetches `RUBRICS.json`, the rubric text sits inside the trajectory. That trajectory goes to the editors, the composer and the reflector. An editor can then write the criteria into the harness, and `_accept_tree` only checks for verdict markers. From there `Library.add` saves it for every later task in the domain. The proposed fix (`answer_key()` per adapter, screening before model context) is unmerged. — [issue/PR #118](https://github.com/google-research/harness2/pull/118)
- **Guardrails can be edited away.**
  - Guardrails start empty in the baseline. The editor prompts treat always-on files as a cost ("To pay for a new rule, TRIM it"; "Delete a file to remove that component"). — [issue #94](https://github.com/google-research/harness2/issues/94)
  - A contributor checked `main` at `1cfe254` and confirmed that there is no way to pin a component: a deployer's safety rule "could currently be trimmed or deleted during Propose/Compose". — [issue #94](https://github.com/google-research/harness2/issues/94)
  - A fix (`BenchSpec.pinned_paths`, which rejects edits to listed files) is unmerged. — [PR #129](https://github.com/google-research/harness2/pull/129)
- **Secrets.** An open PR stops "passing API keys and host tokens to agent subprocesses". — [PR #37](https://github.com/google-research/harness2/pull/37)
  - The improver's Claude CLI channel already strips the Anthropic and Bedrock variables. — [core/proposer.py](https://github.com/google-research/harness2/blob/main/harness2/core/proposer.py)
- **Overfitting and generalisation evidence.**
  - Without the library, refining directly drops WB Code from 73.4 to 63.2 and Web from 69.9 to 65.0. The paper reports these as below the base.
  - A library built at k=1 restores the base level. A library built at k=5 adds +0.9 / +4.4 / +5.1 (Code / Web / Office) with no extra probing.
  - A library built on JobBench adds +0.6 / +0.5 / +2.9 on WorkBuddy.
  - "Leave-own-entry-out" (excluding a task's own library entry) still gives +1.9 on WorkBuddy and +1.7 on JobBench. The authors call this a retrospective test, "not online adaptation". — [paper HTML](https://arxiv.org/html/2610.14224)
- **Learning vs held-out.** Harness² and the online baselines see only test tasks, as a stream. Meta-Harness alone uses the training halves. — [paper HTML](https://arxiv.org/html/2610.14224)
  - The repo can make a reproducible domain-stratified split (`scripts/split_tasks.py`, train fraction 0.5), but the refinement loop itself has no held-out gate. — [EXPERIMENTS.md](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md)
- **Other open bugs that affect evaluation hygiene:**
  - An improver outage is saved as `ok` and never re-run: [#109](https://github.com/google-research/harness2/issues/109)
  - Resume and rollout caches reuse outputs after task inputs change: [#65](https://github.com/google-research/harness2/issues/65)
  - JobBench OpenCode rollouts share installs, temp files and workspaces: [#111](https://github.com/google-research/harness2/issues/111)
  - A rollout killed at its time limit counts as finished: [#121](https://github.com/google-research/harness2/issues/121)
  - The judge keeps an old grade after a failed re-judge: [#44](https://github.com/google-research/harness2/issues/44)
- **A secondary view.** A blog argues that prompt-level `guardrails.md` rules are not a security boundary, and that generated scripts and plugins should be treated as untrusted until reviewed. This is opinion. — [wavect.io (secondary)](https://wavect.io/de/blog/google-harness2-setup-cost-safety.md)

### Inferences
- **The "no ground truth" claim is about the improver, not the benchmark.** Grading still uses LLM judges and rubrics afterwards. The main reward-hacking risk is therefore *leakage* (an executor fetching public rubrics, then the library spreading them), not optimising against the grader. The repo's defences against that leakage are incomplete today.
- **Editable guardrails plus automatic adoption is unsafe for policy rules.** Code (scripts, plugins) is adopted with no human check. Any deployment needs a read-only policy layer that the editor cannot touch.
- **The library is the risky part for overfitting.** It is per domain and grows on the test stream, so a mistake or a leak in one task is inherited by every later task in that domain.

### Gaps
- I did not find the paper's full Limitations or ethics text. Only Appendix F's two points were readable.
- The paper's visible text does not analyse guardrail weakening or reward hacking.

## 5. Maturity: stars, commits, issues, maintenance, licence

### Takeaway
Very popular and very new. Code maintenance is close to nil:
- three commits, the last on 2026-10-14;
- two contributors;
- zero merged PRs out of more than 120 community PRs and issues;
- no maintainer replies seen in the issue threads.

The licence is Apache-2.0, and it is "not an officially supported Google product".

### Cited Findings
- **Repository numbers.** Measured through the GitHub API on 2026-10-23:
  - 17,949 stars and 1,729 forks;
  - created 2026-10-06, last push 2026-10-14;
  - licence Apache-2.0;
  - `open_issues_count` 123, which the GitHub web view splits as 27 issues and 96 PRs. — [GitHub API](https://api.github.com/repos/google-research/harness2); [repo page](https://github.com/google-research/harness2)
  - A vendor blog quoted 9,912 stars at an earlier date, which suggests fast growth. — [activepieces (secondary)](https://www.activepieces.com/blog/harness2-google-research-repository-overview-2026.md)
- **Commits.** Three on `main`:
  - `e0fb6502` "Initial commit of Harness2 open-source release" (2026-10-08);
  - `2152977a` "updates" (2026-10-09);
  - `1cfe254e` "Fix arXiv link" (2026-10-14).
  - Contributors: `rux001` (2 commits) and `cuizz11` (1). — [GitHub API](https://api.github.com/repos/google-research/harness2/commits)
- **Merged PRs: 0.** The closed PRs (#124, #125, #116, #46, #43, #42, #41, #40) were closed without merging. — [PR list](https://github.com/google-research/harness2/pulls?q=is%3Apr)
- **Maintainer replies.** In the first 100 issue comments, none came from a member, owner or collaborator account. All visible replies are from community members (association "NONE"). Checked through the API. — [issue comments](https://api.github.com/repos/google-research/harness2/issues/comments)
- **Releases.** No release or tag exists. A community issue asks for one so a conda recipe can pin it. — [issue #106](https://github.com/google-research/harness2/issues/106); [activepieces (secondary)](https://www.activepieces.com/blog/harness2-google-research-repository-overview-2026.md)
  - The package version is 0.1.0. — [wavect.io (secondary)](https://wavect.io/de/blog/google-harness2-setup-cost-safety.md)
- **Licence and support.** The framework is Apache-2.0. `third_party/`, the benchmarks and the datasets keep their upstream licences. The README says it is "not an officially supported Google product" and not eligible for Google's open-source vulnerability rewards. — [README](https://github.com/google-research/harness2)
- **Code size.** The `harness2/core/` package is about 230 KB of Python. The largest files are `prompts.py` (66 KB), `schedule.py` (31 KB), `context.py` (22 KB) and `recursion.py` (19 KB). There is one test file, `tests/test_workflow.py`. — [repo tree via GitHub API](https://github.com/google-research/harness2)
- **Public discussion.** I found no Hacker News thread. Coverage is blogs and news aggregators (zeli.app, aiidelist, pasqualepillitteri.it, st-hakky, wavect, activepieces), mostly restating the README. One GitHub issue questions why Google's own Antigravity product is weak if harnesses matter this much. — [issue #73](https://github.com/google-research/harness2/issues/73); [zeli.app](https://zeli.app/story/50151059)

### Inferences
- **This is a paper-release artefact, not a maintained library.** The community is finding and fixing real bugs in resume, caching, cost accounting and leak screening, but nothing lands upstream. Anyone who adopts the code takes on a fork.

### Gaps
- Whether the authors plan further releases or merges is unknown. There is no statement in the repo.

## 6. Fit to DCLab: what maps to the Phase 9 learning loop, what conflicts with DCLab's rules, and adopt or reimplement

### Takeaway
The *ideas* fit DCLab's planned learning loop (`docs/guides/PHASE9_LEARNING_LOOP_PLAN.md`) well, especially three:
- contrastive probing of a candidate lesson on the same challenge;
- an evidence-gated library that keeps executable skills, not prose;
- the rule "prefer skills over always-on text, KEEP_BASE allowed".

The *code* fits poorly:
- it depends on Vertex, OpenCode and Codex, and runs on Linux only;
- it uses a file-tree harness and coding-agent CLIs;
- it adopts guardrail and code edits automatically;
- its network is unrestricted by default and its spend accounting is incomplete;
- it is unmaintained.

Recommendation (inference): reimplement 3–4 ideas inside DCLab's own harness. Read `core/prompts.py` as a reference (Apache-2.0 allows reuse with attribution). Do not adopt the framework.

### Cited Findings (DCLab side, from the repository)
- **Phase 9's planned loop.** A solver works a Challenge Pack without its answer key. A deterministic grader (`dclab_core.grader`, no model in the scoring path) scores the trace. A reflector compares failed and passed traces. A curator applies ACE-style delta rules. A variant generator makes the learning, check and sealed sets. — `docs/guides/PHASE9_LEARNING_LOOP_PLAN.md` (Part C, Part E)
- **Phase 9's rules.** The solver never sees the answer key, and a rule is never loosened to make a challenge pass. New recipe parts, checks or gates are pull requests a person approves. Promotion reads only sealed variants, read once. The failure-mode table already lists answer-key leakage, overfitting to the seed challenge, context collapse, playbook bloat, grader gaming and budget burn. — `docs/guides/PHASE9_LEARNING_LOOP_PLAN.md` (Part D)
- **Prior art already reviewed.** The plan's research table covers ACE, GEPA, Voyager, the adaptation survey and MLE-bench, but not Harness². — `docs/guides/PHASE9_LEARNING_LOOP_PLAN.md` (Part A)

### Inferences
**Ideas that map onto Phase 9:**
1. **Propose–Probe–Compose as a lesson check.** Before a reflector's draft lesson enters the playbook, run the challenge again with and without the lesson on the *learning* variant and compare the traces. Harness² says a lesson is kept only when "execution contrasts support it". DCLab can go further, because it *has* a deterministic grader: probe on learning or check variants and judge by the grader, not by a model's reading of the contrast. This is stronger evidence than Harness² has.
2. **INV/DOM scoping.** Split candidate lessons into "how the agent works" (tool errors, ordering, verification) and "domain knowledge" (a task kind's traps). The ablation shows scoping is worth −2.9 to −3.7 points when removed. This maps to DCLab's playbook per task kind versus harness-wide guidance.
3. **Store executable components, not prose.** Skills and scripts carried most of the gain (removing skills: 80.8 → 75.5). This supports DCLab's level 3 (skills as code), but there each skill is certified on variants and approved by a person.
4. **Editor discipline.** Each edit must cite a trajectory moment; KEEP_BASE is allowed; always-on text is kept short. A composer treats an edit from a run that delivered nothing as unsafe. These rules are cheap to copy into DCLab's reflector and curator prompts, which are hash-gated prompt files.
5. **Harness Scaling and Meta-Harness as baselines.** Reporting "per-challenge revision without carry-over" and "offline search on learning variants, frozen" next to DCLab's loop would make the Phase 9 scoreboard more honest.

**Conflicts with DCLab's rules:**
- **Editable, trimmable guardrails** conflict with "a rule is never loosened" and with the workflow validator's authority. In DCLab, guardrails and validators must sit outside anything a learner edits. The validator should also reject any learned skill that touches them.
- **Automatic adoption of code** (scripts, TypeScript plugins) conflicts with "a person approves code changes". Harness² has no human gate; an issue proposing one, off by default, is open ([issue #2](https://github.com/google-research/harness2/issues/2)).
- **No ground truth.** This is the paper's core claim, but for DCLab it is a step *down*. DCLab has hidden answer keys and a deterministic grader. Harness²'s contrast-only signal suits only DCLab's *live* use on a customer's problem, where there is no key, and then only as advice.
- **Refining on the test task itself, with a library grown on the test stream.** This blurs DCLab's learning, check and sealed separation. For DCLab, any per-challenge refinement must run on learning variants only. Sealed variants are read once, by the frozen harness.
- **Unrestricted network, sandbox off, and leakage through trajectories.** These directly break DCLab's top rule (leakage). DCLab's plan already voids a run that opened the key. The Harness² rubric-leak path (issue #118) shows that the trajectory itself must also be screened for key text before a reflector reads it.
- **Cost.** About 4–20× the executor rollouts per task, with no spend cap and executor cost not recorded. This clashes with DCLab's per-run caps (≤ €0.5 per live run) unless solving runs on the local model.

**Adopt the code or reimplement the ideas** (my estimates, not measured):
- **Adopting the code** means writing a Substrate for DCLab's Python harness (5 methods) and a BenchmarkAdapter for Challenge Packs (about 12 methods). It also means exposing DCLab's prompts, playbook and skills as a c1..c8 file tree and replacing the Vertex-only paths. It needs Linux (the owner works on an M1 Mac, so Docker or a VM). DCLab would then own a fork of an unmaintained repo with known resume, cache and accounting bugs. That is high effort and high risk.
- **Reimplementing** means a "probe" step in the Phase 9 loop: re-run with and without the candidate lesson on the learning variants, and compare grader diffs and traces. It also means INV/DOM tags on lessons, the editor rules, and the evidence-gated `EXPERIENCE`-style record. These are additions to code DCLab is already planning, so the effort is low. My guess is days, not weeks.

### Gaps
- No public evidence shows Harness² on tabular ML or leakage-sensitive tasks. Its benchmarks are office, web, code, legal and professional deliverables. Whether its gains carry over to DCLab-style ML decisions is unknown.
- Whether a ~8B local model (DCLab's planned fine-tuned Qwen) can serve as the editor or composer is untested. The paper's smallest editor in the same-model study is not named in the readable text.

# MLE-bench agents: how they search, and where they would break DCLab's rules (study for 33.1)

Read 2026-10-28 for `docs/guides/MLE_BENCH_LESSONS_PLAN.md` 33.1. Documentation only: no code was copied and none was run.

**How it was read.** Each repository was shallow-cloned at the commit below. I read the README, the configuration and
the files for search, memory, execution and selection. arXiv was blocked, so only MLExpert's paper (a PDF in its
repository) was read; claims resting on a paper are marked **unverified**. The code read is today's default branch,
which is not always the version that set the leaderboard score.

**Rule labels used below** (Part C): **C1** no leaks for medals; **C2** honest selection (proposing and choosing on
different rows, holdout read once); **C3** code owns the control and the final choice; **C4** fitted inside the folds;
**C5** a bounded budget on the 8 GB Mac; **SB** generated code runs only in a sandbox (33.7); **C8** licences.

## 0. Sources

| Agent | Repository | Commit read | Licence (from the repo) |
|---|---|---|---|
| MLEvolve | github.com/InternScience/MLEvolve | `f4a37b96c948bfed6564e62f3229e0b38e908d33` | Apache-2.0 |
| ML-Master v1 | github.com/sjtu-sai-agents/ML-Master, branch `ml-master-1.0` | `8621f033b2bd45f8a95ef15c7015a00bbc762e1f` (main: `d55311af…`) | **no licence file** on either branch |
| AIRA-dojo | github.com/facebookresearch/aira-dojo | `c795d8649d30b77fa9b8c011b978bba7127507ad` | CC-BY-NC-4.0 (the LICENSE file and README badge; `pyproject.toml` wrongly lists an MIT classifier) |
| R&D-Agent | github.com/microsoft/RD-Agent | `484776c211e4fbbeef03e0ec00d6bbee7362a4f4` | MIT |
| AIDE | github.com/WecoAI/aideml | `ad73d227af481bea3de06f884f41b2993dd6453a` | MIT |
| Kapso (Leeroo) | github.com/Leeroo-AI/kapso | `dddba7973fbdfb5476294934cabf4ff0bd6ea092` | MIT |
| MLExpert | github.com/InternScience/MLExpert; weights at huggingface.co/InternScience/MLExpert | `8911b70bdd441750685531f3ad5a076cc101aad6` | weights Apache-2.0 (Hub); `kaggle-mle-factory/` Apache-2.0; no root licence |

**Licences (C8).** ML-Master grants no rights (no licence file) and AIRA-dojo is non-commercial, which does not fit the
Apache-2.0 open edition: ideas only from both. The rest are permissive; nothing is copied anyway.

## 1. AIDE (Weco)

- **Operators:** draft, debug and improve. A separate "feedback" model reads each run's output and returns `is_bug` and
  the metric.
- **Search policy:** greedy: 5 drafts first; then, with probability 0.5, debug a random buggy leaf under depth 3;
  otherwise improve the best node. No stall detection, no fusion.
- **Memory:** a summary of the good nodes (plan, analysis, validation metric) goes into every prompt.
- **Budget:** 20 steps and 3,600 s per run. The 17.1% came from OpenAI's 2024 run with o1-preview; today's default is
  o4-mini.
- **Validation and the final choice:** the prompt asks for 5-fold CV. The score is whatever the generated script prints,
  as read by the feedback model. The final submission is the best node by that score.
- **Test set:** the repository has no grader call.
- **Execution:** `exec()` inside a `multiprocessing.Process`, with no container.
- **Where it breaks DCLab:** the model writes the split and the metric (C3). The same validation number both proposes and
  keeps a change (C2). Nothing checks that steps were fitted inside the folds (C4). The code runs unsandboxed (SB).

## 2. ML-Master v1 (SJTU)

- **Operators:** draft, improve and debug, plus a feedback model (gpt-4o) that parses each run. "Steerable reasoning"
  writes the memory and the earlier code into the opening of DeepSeek-R1's thinking.
- **Search policy:** MCTS with UCT. The exploration constant (1.414) decays piecewise to a floor of 0.5. There are 5
  drafts, 3 improves per node and 3 searches in parallel. Reward: −1 buggy, +1 valid, +1 more for a new best. Stall: a
  gain below 0.0001 is a failure, and 3 failures make the node terminal. Debugging stops at depth 20. A score more than
  100× from the best is treated as invalid.
- **Memory:** each node's children (design, analysis, validation metric, "had bugs") are passed into the next prompt.
  Memory is local to a branch.
- **Budget:** 500 steps and 12 h on 36 CPUs, with 9 h per run.
- **Validation and the final choice:** `k_fold_validation: 1`, meaning one split that the model writes. The best valid
  node by its self-reported metric is copied to `best_submission` as the search goes.
- **Test set:** a local Flask server calls `mlebench.grade.validate_submission` against the competition's private
  directory. It returns valid or invalid (a format check), not a score.
- **Execution:** `exec()` in processes pinned to CPU sets, with no container.
- **Where it breaks DCLab:** C2 and C3 as for AIDE. The format server reads private answers, which would be a holdout
  touch under DCLab. A 36-CPU, 12-hour budget breaks C5. SB.

## 3. MLEvolve (Shanghai AI Lab)

- **Operators:** draft, debug and improve. On a stall it triggers **evolution** (within a branch, using its whole
  history) or **fusion** (across branches), and **aggregation** merges branches at the root. Helpers: a code reviewer,
  and an LLM **leakage check** that runs only when a score is extreme (≥ 0.999, or ≤ 1e-6 for a loss). Code is written in
  one pass, stepwise, or as SEARCH/REPLACE diffs.
- **Search policy:** Monte Carlo graph search. UCT decays as in ML-Master. A soft switch from exploring to exploiting
  runs between 50% and 70% of the time. A branch stalls after 3 results without a gain, the global top K after 6. Fusion
  runs only between hours 6 and 10, and is chosen over evolution with probability 0.3.
- **Memory:** one store per task, holding each node's plan, code summary, stage, success label and metric. Retrieval
  mixes BM25 with FAISS vectors (`bge-base-en-v1.5`, α = 0.5).
- **Budget:** 500 steps and 12 h, 21 CPUs, 3 parallel searches, one GPU, 9 h per run.
- **Validation and the final choice:**
  - The generated code computes a hold-out score; the best node by it is kept, and the top 20 are saved.
  - After the run, `utils/submission_fusion_utils.py` averages or votes the **test predictions** of up to 6 top
    solutions, weighted by validation score and rank, into blends of sizes 1 to 6. The blend is never scored on
    held-out rows. Which blend file is graded is **unverified**.
  - A flag (`--use_llm_selection`) lets a model choose the top solutions.
- **Test set:** a format-validation server reads the private directory, as in ML-Master.
- **Execution:** subprocesses, with no container.
- **Where it breaks DCLab:** the blend is not validated out of fold (C4, C2). A model looks for leaks only after a
  suspicious score, where DCLab's deterministic audit runs before any score (C1). The LLM selection breaks C3, hours of
  GPU time break C5, and SB. Vectors need embeddings, which Anthropic lacks (an arm only, as 33.6 says).

## 4. AIRA-dojo (Meta)

- **Operators:** draft, improve, debug, analyze (AIDE's), crossover (evolutionary search), and a memory operator with
  several modes (simple, sibling, debug, none).
- **Search policy:** three solvers. **Greedy** works like AIDE. **MCTS** uses, in its MLE-bench configuration,
  `num_children` 5, `uct_c` 0.25, debug depth 20 and a step limit of 10,000. **Evolutionary** search uses islands,
  crossover and migration probabilities, and a temperature that moves from exploring to exploiting.
- **Budget:** 24 h and 4 h per run; the README reports 1,000 agents in parallel on Slurm.
- **Execution:** isolated. A Jupyter interpreter runs inside an Apptainer "superimage" container.
- **Validation and the final choice:** the best node by its validation metric (`journal.get_best_node`).
- **Test set (key finding):** `tasks/mlebench/task.py` grades every valid submission on the test answers at each step
  and keeps it as `TEST_FITNESS` beside the node, for analysis. The solvers use it as the search metric only when
  `use_test_score: true`; the shipped MLE-bench configurations set False. The paper's study of the validation–test gap
  is **unverified** (not read).
- **Where it breaks DCLab:** the test score lives in the search process, one flag away from steering selection (C2, and
  the rule that the holdout is read once). If DCLab ever measures the same gap, the grader must sit outside the search
  process and run only on replicas or the sealed set. Licence: ideas only (C8).

## 5. R&D-Agent (Microsoft)

- **Operators:** a **Research** side proposes hypotheses (with an idea pool and a diversity strategy); a
  **Development** side (CoSTEER coders) builds the components, from data loading to ensembles; a **feedback** model
  judges each result.
- **Search policy:** a loop of select checkpoint → propose → code → run → feedback → record. Several traces can run and
  be merged (off by default). After too many errors in a row the trace resets.
- **Memory:** the trace, a graph of experiments with their hypotheses and feedback (the last 3 are shown in prompts),
  plus an optional knowledge base.
- **Budget:** 3,600 s per full run after a 600 s debug run on sampled data; 12 h on the leaderboard.
- **Validation and the final choice:**
  - The code writes its own scores. The feedback model sets `decision` ("Replace Best Result") and drops a result whose
    evaluation is "not aligned with the task". A system-call trace rejects code that opens the sample submission.
  - The default submission (`GlobalSOTASelector`) is the best over the whole trace, where "best" was the model's
    decision. Other selectors: by score, or a model chooses.
- **Test set:** only `mlebench grade-sample`, in `log/mle_summary.py`, after the run, for reporting. Inside the loop
  there are format checks only.
- **Execution:** Docker by default.
- **Where it breaks DCLab:** a model decides what is kept and what is submitted (C3; DCLab uses `review.adopt`). The
  same validation proposes and selects (C2).
- **Worth noting:** the trace of opened files is a **deterministic** guard that DCLab should copy as an idea in 33.7.

## 6. Kapso (Leeroo)

- **Operators:** an ideation session, then a coding-agent CLI session that writes, runs and scores its own code, then a
  judge model that checks the evaluation and decides whether to stop. Stages follow the budget used: "MINI TRAINING"
  (under 20%, large data), "FULL TRAINING", "FINAL ENSEMBLING" (over 80%).
- **Search policy:** a tree of experiments as git branches; the best node is the parent.
- **Memory:** experiment history and repo memory; `learn()` turns finished campaigns into lesson cards with their
  evidence; an external paid knowledge base (Leeroopedia).
- **Budget:** sessions of up to 4 h at `xhigh` effort, with claude-opus-5 and gpt-5.6-sol. The leaderboard row (50.7%)
  used Gemini-3-Pro-preview for 24 h, so this is **not** the configuration that scored.
- **Validation and the final choice:** the prompt orders a stratified 10% of the training rows to be set aside with
  "train nothing on it, not even hyperparameters", and one score reported in `<score>` tags. The deliverable is the
  committed candidate by "evidence tiers". `final_evaluate` then grades it once with the official `grade_csv`.
- **Test set:** this is the second key finding.
  - `benchmarks/mle/handler.py` `run()` grades every valid submission with `grade_csv` on the hidden answers, copies
    any medal submission to `~/mle_res/`, and `stop_condition()` returns `got_medal`: the search stops when the **test
    set** awards a medal.
  - At this commit nothing calls `run()` or `stop_condition()` (the orchestrator calls the latter "legacy"). Whether it
    was active in the leaderboard run is **unverified**.
- **Execution:** the coding-agent CLIs run shell commands. Whether they run inside a container is **unverified**.
- **Where it breaks DCLab:** a stop driven by the test set (C1, C2); a model judge decides validity and stopping, and a
  prompt rather than code makes the hold-out (C3); long sessions (C5); SB; a paid external knowledge base.
- **Worth noting:** the staged budget (screen small first, then train fully, ensemble last) suits the 8 GB profile.

## 7. MLExpert (Shanghai AI Lab and East China Normal University; paper arXiv 2610.22344v1, read from the PDF in the repository)

- **Model:** initialised from Qwen3.6-35B-A3B. The Hub lists 34.66B parameters with the `qwen3_5_moe` architecture. The
  "A3B" suffix suggests about 3B active parameters (**unverified**). The serving example uses tensor parallelism over 2
  GPUs.
- **Released:** the weights (Apache-2.0), the Kaggle-MLE Factory code and the paper. **Not in the repository** at this
  commit: the harness, the SFT and RL code, and the trajectories. Part A's "code and weights released" should read
  "weights and the task factory".
- **Data:** the Factory turns Kaggle slugs into tasks (LLM-written preparation and scoring scripts, a validator that
  checks format, labels and train–test identifier overlap, repair, a manual review): 154 tasks, plus 134 from MLE-Dojo,
  plus scientific tasks from Nature-family papers, with any overlap with MLE-bench or NatureBench removed. Teachers:
  GLM-5.2, and Qwen3.8-Max for science. 636 source trajectories (412 MLE trajectories over 236 tasks), 5,539 records,
  about 302M packed tokens, 8 epochs.
- **SFT, in two stages:** general coding and tool-use trajectories (StepFun), then the teacher trajectories. A
  trajectory is dropped if its best submission ranks below 0.4 on the task's **private leaderboard**, and cut after its
  first peak if it later falls more than 5%. Loss is taken only on the model's own tokens.
- **RL (WIRL):** a "window" is every decision between two experiment results, and all of them share one reward: −1.0
  for an execution error, −0.6 for no scorable submission, −0.4 + T below medal level, 1.0 + T for bronze or silver,
  2.0 + T for gold, where T (−0.2, 0 or +0.2) is the trend since the last valid submission. Collection is asynchronous,
  with a clipped token-level correction. Reported ablations: general SFT alone 47.0%; WIRL without the trend 66.7%;
  full WIRL 72.7%.
- **Evaluation:** MLE-Bench Lite, 3,000 steps and 12 h on 20 vCPUs, 220 GB RAM and one H200: 37.9% → 72.7% (69.7% in
  Claude Code). The harness is a ReAct loop in which **the model** chooses branches, can `invalidate_node` a candidate it
  thinks leaked, and owns `update_answer`, the final submission.
- **Grader contact:** both the SFT filter and the RL reward use each *training* task's hidden-test score. That is
  legitimate for training tasks, but the policy learns from test-graded outcomes.

**MLExpert against DCLab's agentic-LLM plan (31.13–31.16).**

1. **Size.** DCLab plans an ~8B Qwen tool-calling model (LoRA SFT, then preferences). MLExpert is served on 2 GPUs and
   cannot be trained on the M1 Max (Part F says so). Study the recipe, not the weights.
2. **Control.** MLExpert's model chooses the branch and the answer. In DCLab the model proposes, and the validator and
   `review.adopt` decide (C3). So DCLab's trajectories teach *proposing* moves.
3. **Outcome-guided cuts** suit 31.14 if applied when the set is built, with the raw record kept whole (31.13).
4. **A window is a DCLab node:** one change, then a measured verdict. The kept-against-not-kept pairs of 33.8 are the
   same unit, and the trend term matches the change `adopt` measures.
5. **Grader.** Any reward for 31.11's RL must come from rows sealed from the agent, kept apart by problem family, and
   never from MLE-bench (31.12 stays sealed).
6. **Scale and teacher.** About 5.5k records from 636 runs is the order of 31.15's trigger (a few thousand accepted
   steps across three families). Their teacher is GLM; DCLab's would be Claude, under the owner's rule.

## 8. What to build, and what not

DCLab today has `dclab_core.selection` (K = 1 confirmation on a selection fold, measured optimism about 0.002); the
opt-in Optuna search `studio/search.py`, which failed `search_v2` (mean holdout −0.0039 on 8 of 9 datasets, clearly
better on none); and no stacking (34.7b, the owner, after `search_v2`).

| Idea | Agents (section) | What DCLab has | Next under DCLab's rules / not worth it |
|---|---|---|---|
| Draft, debug and improve operators with a depth cap | AIDE, ML-Master, AIRA, MLEvolve (§1–4) | `review.CATALOGUE` (7 kinds, 4 runnable), `REVIEW_ROUNDS = 2` | **Build** (33.4): make the remaining entries runnable. The debug cap belongs to 33.7 |
| Tree search (UCT, decaying exploration, stall detection) | ML-Master, MLEvolve, AIRA (§2–4) | two review rounds | **Not over hyperparameters**: `search_v2` found no gain at the same set-aside price. **Maybe** over structural moves (recipes, target transforms, families), after 33.9's baseline, gate fixed first |
| Fusion of kept changes from two branches | MLEvolve, AIRA crossover, R&D merge (§3–5) | `compose` fuses leaves | **Cheap once a tree exists** (33.3): deterministic, scored like any node |
| A blend of the top K at the end | MLEvolve, Kapso (§3, §6) | `compose` (24.5); stacking not built | **Not now**: same selection-fold price as 34.7b; revisit on bigger tables. Never blend unvalidated test predictions |
| Memory retrieved by similarity | MLEvolve, ML-Master, R&D trace, Kapso `learn()` (§2, §3, §5, §6) | ledger `experiment.next`, `lessons.py`, 31.6 planned | **Build** (33.6): filter by data-card shape, then lexical ranking. Vectors only as an arm. One store with 31.6 |
| A hold-out set apart from the search | Kapso prompt (§6); AIRA's logged test score (§4) | `selection.py`, K = 1, wired | **Nothing new**: keep K = 1, record attempts; measure optimism on replicas only |
| An LLM leakage check at extreme scores | MLEvolve (§3) | the deterministic audit before any score | **Not worth it** as a check; at most an advisory flag |
| A trace of opened files; "evaluation aligned" | R&D-Agent (§5) | the kernel holds training rows only | **Build** in 33.7: refuse generated code that opens files beyond its inputs (deterministic) |
| A staged budget (small screen, full run, final) | Kapso, MLEvolve's time switch (§3, §6) | a per-run budget planned (Part F) | **Build** for the 8 GB profile: screen on a sample, then confirm |
| Code written as diffs or step by step | MLEvolve, Kapso (§3, §6) | the kernel runs only server cells | **Later** (33.7), after the container sandbox |
| A model owning the final choice | MLExpert, R&D `AutoSOTA`, MLEvolve LLM selection | code chooses | **Not worth it**: breaks C3 |
| Test-set feedback (score logging, a medal stop) | AIRA (§4), Kapso (§6) | the holdout read once | **Never** inside a search; only as an outside measurement on replicas or the sealed set |
| An agent model trained on trajectories | MLExpert (§7) | 31.13–31.16, 33.8 planned | **Borrow** outcome-guided cuts and node-level window rewards. Do not take the 35B base |

## 9. Not verified

- The papers for AIDE (2502.13138), ML-Master (2506.16499), AIRA-dojo (2507.02554), MLEvolve (2606.06473) and Kapso
  (2601.21526): arXiv was blocked. Leaderboard numbers are as given in the READMEs and the plan.
- Whether each repository at the commit read matches the code behind its leaderboard score. This is known *not* to hold
  for Kapso's configuration, and not checked for the others.
- Which of MLEvolve's blend files is submitted for grading.
- Whether Kapso's medal-based stop was active in its leaderboard run.
- Whether Kapso's coding-agent sessions run inside a container.
- MLExpert: how many active parameters "A3B" means, the RL compute, and where its harness and training code are, if
  released anywhere.

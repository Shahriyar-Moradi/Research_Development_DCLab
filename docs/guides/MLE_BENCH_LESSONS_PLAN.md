# Lessons from MLE-bench: search, memory and ensembles under DCLab's rules (plan)

Written 2026-10-21 from the owner's question, *can we learn from their approaches?*, after reading the MLE-bench
leaderboard (mlebench.com, which shows the table of github.com/openai/mle-bench) and the open-source agents on it. Read
against the repository at `de406362`.

**Status (2026-10-28):** 33.1 is done (`MLE_BENCH_AGENTS_STUDY.md`). 33.2 was built as phase 10's 34.6, and 33.3's
search as 34.7a, opt-in only after `search_v2` and `search_v3` failed; 33.5's stacking is not built (34.7b, the owner).
33.4 and 33.6 to 33.9 are planned and wait for the owner to place them in the order. The study note (33.1) is documentation only. Every other
package changes code and follows the house discipline: test first, the `dclab-reviewer`, its development-trajectory
record, `make check-quick`, and capture in its "Done when".

**In one paragraph.** The best MLE-bench agents win by **searching**: they try hundreds of solutions in 12 to 24 hours,
write their own code, remember what worked, and blend their best attempts. DCLab's models stage today screens a fixed
list of families with default settings and runs at most two review rounds from a catalogue of seven experiments (four
runnable). The lesson is to widen that search a lot, but only under DCLab's rules. Many attempts on the same folds make
the score optimistic, so honest selection (33.2) comes before any wider search. A leak that wins a medal on MLE-bench is
still blocked in DCLab. Deterministic code still owns the splits, the metrics and the final choice.

---

## Part A. What MLE-bench is and what the leaders do

**The benchmark.** 75 Kaggle competitions (tabular, image, text, audio, sequence to sequence). Kaggle's test answers are
not public, so OpenAI splits each competition's public training data into a new train and test set and grades locally.
The score is the share of competitions where the agent's submission would have won a medal (bronze or better) on the
original leaderboard. Splits: Low (22 competitions, "Lite"), Medium, High and All. Standard budget: 24 hours, 36 vCPUs,
440 GB RAM and one 24 GB A10 GPU; at least 3 seeds, reported as mean ± standard error. Since 2026-04-24 OpenAI takes no
new submissions while it designs a fairer process; a v2 is planned in `openai/frontier-evals`. OpenAI lists competitions
with known leaks, for example `random-acts-of-pizza`, where `giver_username_if_known` gives away the outcome.

**The leaderboard (read 2026-10-21, All column).** Famou-Agent 2.0 (Baidu, Gemini-3-Pro) 64.4%; AIBuildAI (Claude Opus
4.6, on Claude Code) 63.1%; MARS+ (Google Cloud AI Research) 62.7%; MLEvolve and PiEvolve 61.3%; ML-Master 2.0 56.4%;
Leeroo (Kapso) 50.7%; R&D-Agent (Microsoft, gpt-5) 35.1%; AIRA-dojo (Meta, o3) 31.6%; AIDE (o1-preview, 2024) 17.1%.
Scores rose from about 17% in 2024 to about 64% in 2026. Two self-reported numbers are not on the official board:
AIBuildAI Agent 2.0 at 70.7% and MLEvolve at 65.3% (Gemini-3.1-Pro, 12 hours). Two entries are listed apart because they
saw test-set feedback: Disarray 77.8% and LoongFlow 62.7%.

**What the leaders do, by idea.**

| Idea | Who | How it works | Open code |
|---|---|---|---|
| Search over solutions | AIDE, AIRA-dojo, ML-Master, MLEvolve | Each node is a whole solution (a script) scored on validation. Operators draft, debug and improve. Policies: greedy, MCTS, evolutionary. MLEvolve: several branches in parallel, an exploration rate that decays, a switch from exploring to exploiting as time runs out, stall detection, then **cross-branch fusion** (the best ideas of different branches merged into a new candidate) | AIDE, AIRA-dojo, ML-Master v1, MLEvolve |
| Memory of attempts | MLEvolve, ML-Master | Every node's plan, code, score and success or failure is stored. The planner retrieves from it (BM25 plus vectors in MLEvolve) to reuse what worked and avoid known failures. ML-Master shares memory between branches | MLEvolve, ML-Master v1 |
| Code as the action | all of the leaders | The model writes and patches feature and model code (MLEvolve: single pass, a stepwise multi-agent pipeline, or SEARCH/REPLACE diffs, chosen by the state of the search), and fixes its own errors | all open ones |
| Targeted refinement | MLE-STAR (Google) | Searches the web for suitable models, finds by ablation the code block that matters most, refines that block, then ensembles | closed |
| Researcher and developer | R&D-Agent | One agent proposes hypotheses, another implements them, results feed the next hypothesis | R&D-Agent |
| Ensembles | most leaders | The final submission blends the best solutions | — |
| A trained agent model | MLExpert (MLEvolve's team, 2026-10-15) | A 35B mixture-of-experts model post-trained for long-horizon ML engineering with hierarchical agentic SFT, then windowed interaction RL. MLE-bench Lite from 37.9% to 72.7% | weights and the task factory released; the harness and training code are not in its repository (33.1) |

## Part B. What DCLab has and what is missing

| Need | Today | Missing |
|---|---|---|
| Many candidates | `stage_models` (`dclab_rnd/studio/engine.py`) cross-validates each family of `kind.model_families` with default parameters on the same folds, ranked by mean − 0.25 × fold std, then the simplest family within 10% (`PRACTICAL`) for rare events | hyperparameters, feature recipes and model variants as candidates; a budget per run instead of a fixed list |
| Improving the leader | `_review_rounds`: at most `REVIEW_ROUNDS = 2`, each an experiment from `dclab_core.review.CATALOGUE` (7 kinds, 4 runnable), ranked by the rules and the critic (18.6), kept only by `review.adopt` on the same folds | a search tree with branches, a policy, stall detection and fusion; more runnable actions |
| Honest selection under many attempts | the holdout is read once (WF-09); the claim says "the kept training-CV score is optimistic" | a selection fold apart from the folds that proposed the change, and a record of how many attempts were made; a measure of the optimism |
| Memory | the ledger keeps every `experiment.next` decision (19.7), lessons after a project (`dclab_rnd/lessons.py`), project memory, the evidence index | retrieval of past experiments by the data card's shape, as a prior on what to try first |
| Ensembles | `dclab_core.compose` fuses the leaves of a programme (rank fusions, gates, a tie to the simpler; 24.5) | stacking or blending of the top families at the models stage, fitted inside the folds |
| Code as an action | the notebook kernel (`studio/kernels.py`) holds the training rows only and no key; it runs only the server's own generated cells | a contract for generated feature or model code, its sandbox, its leakage check and its debug loop |
| A trained agent | phase 9 Part F2 plans the agentic LLM (31.13 to 31.16); the storing phase plans M5, the solver LLM | the search trees as training material (kept vs not kept as preferences) |
| An outside yardstick | 31.12 plans 5 to 10 low-complexity MLE-bench competitions as a permanent sealed set | a development set that is not MLE-bench, so the sealed set stays sealed |

## Part C. What not to copy (the rules that hold in every package)

1. **No leaks for medals.** A column DCLab's leakage audit blocks stays blocked, even when it would win a medal. Every
   MLE-bench score DCLab reports names the competitions where a known leak exists and gives both numbers where they
   differ.
2. **Honest selection first.** No package widens the search before 33.2 is in place: candidates are proposed and scored
   on one set of folds, and the final choice among the best is made on a selection fold those folds never saw. The
   number of attempts is recorded with the result. The holdout is still read once, after the leakage audit.
3. **Code owns the control.** The model proposes a node; deterministic code runs it, scores it, and decides whether it is
   kept (`review.adopt` or its successor). The graph validator checks every move (CLAUDE.md rule 5).
4. **Fitted inside the folds.** Every step that learns from data, including generated code and stacking, is fitted on
   the training part of each fold only (rule 2).
5. **The 8 GB Mac.** The search has a budget per run (nodes, minutes, jobs at once) and a small default profile; larger
   budgets are for the cloud and the owner's choice. Live model runs are capped at €0.5 each and reported.
6. **One store per thing.** Search memory (33.6) and phase 9's case memory (31.6) are one store, not two. The search
   trees feed the trajectory record of 31.13, not a new format.
7. **Contract in `dclab_core`, wiring in `dclab_rnd`.** Each code package puts its contract in `dclab_core` (numpy,
   pandas, scikit-learn and pydantic only), its wiring in `dclab_rnd`, and an adapter plus a `CORE_INTEGRATION.md`
   section for the main product (milestones J and K).
8. **Borrow ideas, check licences.** Reading the open-source agents is free. Copying their code needs their licence
   checked first, and the open edition's licence (Apache-2.0) kept compatible.

## Part D. The packages (33.x)

### 33.1 Study the open-source agents (documentation only)
- **Why:** the ideas in Part A come from leaderboards and READMEs. Before building anything, DCLab needs to know exactly
  how each open agent's search, memory and code actions work, and where each would break a DCLab rule.
- **Delivers:** `docs/guides/MLE_BENCH_AGENTS_STUDY.md`: for MLEvolve, ML-Master v1, AIRA-dojo, R&D-Agent, AIDE and Kapso,
  their operators, search policy, memory, budget, how they validate and select, how they use the test set, their licence,
  and for each one where it would leak or overfit under DCLab's rules. MLExpert's training recipe (data, SFT, RL) read
  against the agentic LLM's plan (31.13 to 31.16). No code copied.
- **Done when:** the note covers the six agents and MLExpert with a link to each source at a named commit; every idea
  later built in 33.3 to 33.8 cites its section.
- **Status:** done (2026-10-28): `MLE_BENCH_AGENTS_STUDY.md`, each source at a named commit. Its main findings: AIRA-dojo
  logs a test-set score beside every node and Kapso has a stop driven by a test-set medal (both off, or not called, at the
  commits read); MLEvolve blends test predictions never scored on held-out rows; ML-Master has no licence and AIRA-dojo
  is non-commercial (ideas only). Worth building: runnable catalogue entries (33.4), memory filtered by the data card's
  shape (33.6), a staged budget for the 8 GB Mac, and R&D-Agent's deterministic guard on the files generated code opens
  (33.7). Not worth it: a tree over hyperparameters, an LLM leakage check, a model owning the final choice.

### 33.2 Honest selection under many attempts
- **Why:** with 2 rounds the optimism of the kept CV score is small and already said in the claim. With 50 or 500
  attempts it is not. MLE-bench agents are protected by a grader that hides the test set; DCLab must protect itself
  before it searches.
- **Delivers:** `dclab_core/selection.py`: a selection protocol v1 fixed before any run. The training rows are split
  once into search folds (where candidates are proposed and scored) and a selection fold (read only to choose among the
  top K at the end), with the same split rules as the project (time, group, stratified). Every candidate records how many
  attempts came before it. The stage record states the selection fold's score beside the search folds' score, and the
  difference as the measured optimism. The engine's models stage uses it whenever more than `REVIEW_ROUNDS` candidates
  are tried.
- **Done when:** on the replicas with planted noise, the gap between the chosen candidate's CV score and its holdout
  score grows with the number of attempts under plain CV, and stays inside its interval under the protocol (the gate
  fixed before the run). Capture: the attempts, both scores and the optimism in the stage record and the ledger.
- **Status:** built as phase 10's 34.6 (2026-10-25, dclab-core 0.110.0, `dclab_core.selection`): confirming only the
  winner (K = 1) on the selection fold, measured optimism about 0.002 (`PHASE10_ABILITIES_PLAN.md`).

### 33.3 The search tree
- **Why:** the leaders' main advantage. Two rounds on one leader leave most of the room unexplored.
- **Delivers:** `dclab_core/search.py`: `SearchNode` and `SearchTree` v1 (a node is a parent, one change, its fold scores,
  its cost, kept or not and why); a policy (best-first with an exploration bonus, then exploitation as the budget runs
  out), stall detection, and fusion (the kept changes of two branches applied together as a new node). The models stage
  in `dclab_rnd/studio/engine.py` runs the tree in place of the two rounds when the project's settings ask for a search
  budget; each node is a job step with `checkpoint()` so Stop works. Every node goes through the graph validator and is a
  `run.recorded` event and an `experiment.next` decision in the ledger.
- **Done when:** on the replicas, the tree's selection-fold score beats the two rounds at a bounded cost, with the gate
  fixed before the run; the holdout read once; a run on the 8 GB profile stays within its minutes. Capture: the whole
  tree (every node, kept or not) in the stage record and the lake.
- **Status:** a per-family Optuna search built as 34.7a (`studio/search.py`), opt-in only: `search_v2` and `search_v3`
  failed (no gain on the bake-off at the set-aside's price). A tree over structural moves stays planned (study §8).

### 33.4 A wider catalogue of actions
- **Why:** a tree is only as good as its moves. Today 4 of 7 catalogue experiments can run.
- **Delivers:** new entries in `dclab_core.review.CATALOGUE`, each fold-safe and declared with what it addresses:
  hyperparameter ranges per family (small, fixed grids first), the engine's feature recipes as moves, target transforms
  for each kind, and the non-runnable entries made runnable where the dependency now exists. Each entry says its cost so
  the policy can weigh it.
- **Done when:** each new entry has a test that it fits inside the folds only and a replica where it is kept; the
  existing replicas' acceptance decisions are unchanged.
- **Status:** planned; can go with 33.3.

### 33.5 Ensembles at the models stage
- **Why:** almost every leading agent blends its best solutions at the end.
- **Delivers:** stacking and blending of the top K families on their out-of-fold predictions, fitted inside the folds
  (nested), chosen on the selection fold (33.2), with the rule that a tie goes to the simpler model. Reuses
  `dclab_core.compose`'s fusions rather than a second implementation. The exported code (`studio/code.py`) reproduces the
  ensemble's holdout number.
- **Done when:** on the replicas an ensemble is chosen only when it beats the best single family on the selection fold;
  `make code-check` reproduces its holdout number in a clean Python 3.12.
- **Status:** not built (34.7b, the owner, 2026-10-25, after `search_v2`): a stack pays the same selection-fold price.

### 33.6 Memory of what worked
- **Why:** MLEvolve and ML-Master reuse their own history. DCLab already records every experiment and its outcome.
- **Delivers:** retrieval of past nodes (the ledger's `experiment.next` decisions with their measured labels, and the
  workspace lessons) by the data card's shape (kind, size, class balance, column roles in bands, never a value), as a
  prior on the order the tree tries its moves. Structured filters and the measured lexical ranking first; vectors only as
  a measured arm (Anthropic has no embeddings). The same store as phase 9's case memory (31.6).
- **Done when:** an arm with memory on against off on the replicas and the generated tables (22.1), the gate fixed
  before; the prior never reads a sealed case.
- **Status:** planned; needs 33.3. Shares its store with 31.6.

### 33.7 Generated code, sandboxed
- **Why:** the leaders write their own features and models; a catalogue alone cannot.
- **Delivers:** a contract for a generated step (a function with fixed `fit` and `transform` or `predict` signatures)
  that a model writes for one node; a static check (imports allowed, no file or network access); a run in the notebook
  kernel's sandbox (training rows only, no key, `studio/kernels.py`), fitted inside each fold; the leakage review on any
  new feature (its lift alone, its time against the moment); a debug loop of at most N tries with each error kept; kept
  only by the same rule as any node.
- **Done when:** a planted leaking feature written by a scripted model is refused; a planted good feature is kept on a
  replica; every attempt, its code, its error and its verdict are kept whole. Not before the container sandbox the
  roadmap names for notebook cells.
- **Status:** planned; needs 33.3 and the owner's decision on the sandbox.

### 33.8 The search as training data
- **Why:** MLExpert shows that an agent model trained on its own search trajectories can match much larger models. The
  owner's agentic LLM (31.15) and M5, the solver LLM, need exactly this data.
- **Delivers:** each search tree written into the whole trajectory record (31.13): every node's reasoning, action, result
  and verdict, before anything is cut; kept against not-kept nodes as preference pairs; a family in the lake beside the
  others, with a sealed partition by problem family. MLExpert's recipe noted in
  `research/llm-fine-tuning/experiments/decisions/finetune/README.md` beside the existing one. The training runs stay the owner's.
- **Done when:** a search run's tree reads back whole from its trajectory record; the preference pairs' manifest exists;
  no sealed family reaches a training set.
- **Status:** planned; needs 33.3 and 31.13.

### 33.9 The yardstick
- **Why:** an outside score answers the buyer's question, but MLE-bench must stay a sealed exam (31.12). DCLab needs a
  separate set to develop on.
- **Delivers:** a development set of public tabular competitions that are **not** in MLE-bench (Kaggle playground series
  and OpenML tasks from the lake, 26.2), graded the MLE-bench way (the percentile on the original leaderboard, the
  medal it would win) and DCLab's way (the leak caught, the split, the holdout read once). MLE-bench Lite's four tabular
  competitions (`new-york-city-taxi-fare-prediction`, `nomad2018-predict-transparent-conductors`,
  `tabular-playground-series-dec-2021`, `tabular-playground-series-may-2022`) join 31.12's sealed set and are scored once,
  at a promotion. Every report names the known leaks and gives both numbers.
- **Done when:** the development set's baseline (today's DCLab, no search) is stored with its intervals; the sealed set
  is not opened.
- **Status:** planned; needs the Kaggle key and the owner's approval for the downloads.

## Part E. Order and dependencies

| Order | Package | Needs | Cost |
|---|---|---|---|
| 1 | 33.1 Study the agents | — | reading only |
| 2 | 33.9 The yardstick (baseline only) | the Kaggle key, the owner's approval | downloads, CPU |
| 3 | 33.2 Honest selection | — | CPU |
| 4 | 33.3 The search tree, with 33.4 | 33.2 | CPU; the critic's calls capped |
| 5 | 33.5 Ensembles | 33.2 | CPU |
| 6 | 33.6 Memory | 33.3; shares 31.6's store | CPU |
| 7 | 33.8 The search as training data | 33.3, 31.13 | none |
| 8 | 33.7 Generated code | 33.3, the container sandbox | model calls, capped |

The measure to track from the start: on the development set, the score after the search against the baseline, the
measured optimism (33.2), the number of attempts, the minutes and the euros. 33.9's baseline comes early so every later
package has a zero to compare with.

## Part F. Decisions for the owner

| Decision | Needed by | Proposal |
|---|---|---|
| Where 33.x sits in the order | all | after the open-source edition; 33.1 and 33.2 can start any time, as they change nothing the user sees |
| Search budget on the Mac | 33.3 | a small default (for example 30 nodes or 20 minutes a stage); larger only in the cloud |
| Downloads for the yardstick | 33.9 | Kaggle playground and OpenML tabular tasks first; the Kaggle key is still open |
| The sandbox for generated code | 33.7 | the container sandbox already planned for notebook cells; nothing generated runs before it |
| Model for proposing nodes and writing code | 33.3, 33.7 | Claude Sonnet 5.5 through the gateway, capped per run; a local model as a measured arm |
| MLExpert as a base model to study | 33.8, 31.15 | read its paper and recipe; whether to try its weights is the owner's call (35B needs more than the M1 Max for training) |

---

## خلاصهٔ فارسی

**وضعیت (۲۸ اکتبر ۲۰۲۶):** بستهٔ ۳۳٫۱ انجام شد و یادداشت مطالعه در `MLE_BENCH_AGENTS_STUDY.md` است. بستهٔ ۳۳٫۲ در فاز ۱۰
با نام ۳۴٫۶ ساخته شد. جستجوی بستهٔ ۳۳٫۳ با نام ۳۴٫۷a ساخته شد، ولی فقط اختیاری است، چون آزمون‌های `search_v2` و `search_v3` رد شدند.
ترکیب مدل‌ها (stacking) در بستهٔ ۳۳٫۵ به تصمیم مالک ساخته نشد. بستهٔ ۳۳٫۴ و بسته‌های ۳۳٫۶ تا ۳۳٫۹ هنوز در برنامه‌اند و منتظر
تصمیم مالک دربارهٔ ترتیب کارند.

**درس اصلی:** عامل‌های برتر MLE-bench با **جستجو** برنده می‌شوند. صدها راه‌حل امتحان می‌کنند، کد خودشان را می‌نویسند،
آنچه جواب داده را به خاطر می‌سپارند و بهترین‌ها را با هم ترکیب می‌کنند. مرحلهٔ مدل در DCLab امروز فقط فهرست ثابتی از
مدل‌ها را با تنظیمات پیش‌فرض می‌سنجد و حداکثر ۲ دور آزمایش انجام می‌دهد. باید جستجو را خیلی گسترده‌تر کرد، اما فقط زیر قواعد
DCLab.

**قواعدی که تغییر نمی‌کنند:**
- **نشت داده برای مدال، هرگز.**
- **اول انتخاب صادقانه، بعد جستجوی گسترده:** یک fold جدا برای انتخاب نهایی، و ثبت تعداد تلاش‌ها؛ holdout همچنان فقط یک بار
  خوانده می‌شود.
- **کنترل با کد قطعی است:** مدل پیشنهاد می‌دهد، کد اجرا و انتخاب می‌کند.
- **همه‌چیز فقط درون foldهای آموزش fit می‌شود.**
- **بودجهٔ محدود روی مک ۸ گیگابایتی.**

**بسته‌ها:**
- **۳۳٫۱:** مطالعهٔ عامل‌های متن‌باز (فقط مستند).
- **۳۳٫۲:** انتخاب صادقانه با تلاش‌های زیاد.
- **۳۳٫۳:** درخت جستجو.
- **۳۳٫۴:** فهرست گسترده‌تر اقدام‌ها.
- **۳۳٫۵:** ترکیب مدل‌ها (ensemble).
- **۳۳٫۶:** حافظهٔ تجربه.
- **۳۳٫۷:** کد تولیدشده در محیط ایزوله.
- **۳۳٫۸:** درخت‌های جستجو به‌عنوان دادهٔ آموزش برای مدل عامل (با الگوی MLExpert).
- **۳۳٫۹:** معیار بیرونی؛ مسابقه‌های MLE-bench مهروموم می‌مانند و توسعه روی مسابقه‌های دیگر انجام می‌شود.

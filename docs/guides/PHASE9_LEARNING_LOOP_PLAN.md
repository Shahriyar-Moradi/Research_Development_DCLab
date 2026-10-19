# Phase 9: the learning loop — teaching DCLab new problems (plan)

Written 2026-10-18 from the owner's words: *I want to make a framework: I give my agent a new problem, it learns, and it
adds that capability to solve similar problems; that is one of the important goals of DCLab.* Three documents by the
owner are its sources, read here against the repository at `07fa011`:

- **DCLab Learning Loop: Teaching Agents New Problems** (2026-10-18): the four learning levels, the Challenge Pack, the
  seven steps, the gap types, the three variant sets, the scoreboard, the guardrails, a worked example.
- **DCLab Learning Capability: Feasible Approaches and Build Guide** (2026-10-18): what the research shows (ACE, GEPA,
  ExpeL, Voyager, the adaptation survey, MLE-bench), the five approaches, one loop with four stores, five stages with exit
  tests, the harness's concrete specs, the learning mechanics level by level, the failure modes and their checks.
- **DCLab Task Recipes: Adding Support for New ML Problems** (2026-10-16): one task kind plus modifiers, the nine-field
  contract, certification (experimental, certified, product default), trap replicas, the five pass criteria, and the ten
  steps to add a kind or a modifier. Phase 8 built its registry and first modifiers (30.10, 30.11); here it is the shape
  of what the loop's third level writes.

**Status: a plan only; widened 2026-10-19.** The owner's order (2026-10-18): nothing of phase 9 is built while phases 6, 7
and 8 are being finished, and the capture of everything keeps running meanwhile. On 2026-10-19 the owner added that every
action and decision is recorded for training the three kinds of model, above all the workflow LLM, and that an agentic LLM,
a model that solves a machine-learning problem by itself as the agent, is fine-tuned beside them (Part F2, 31.13 to 31.16);
and that the loop's own seats record everything they read, do and produce (Part C, 31.17). Phase 9 starts after them; the
capture they finish is what it learns from.

**What it stands on.** The Learning Loop "uses the four phases and cards from *DCLab Framework: From Problem to Solution
Workflow*, and the recipe parts from *DCLab Task Recipes*", and adds the part that was missing: how the system grows from
the problems it meets. So an attempt is the four phases with their four cards and gates, which phase 8 finishes: the
Problem Card v2 and G1 (30.13), the Data Card with its fit verdict and G2 (30.13), the Solution Brief with its success bar
and G3 (30.14), the Workflow Run and G4 (30.15), on the recipe registry and its modifiers (30.10 to 30.12), with the case
trajectories in the lake (30.19). The answer key's fit verdict and success bar are those cards' fields; the grader cannot
be written before them. Phase 6's team (the walk, the critic, the auditor) and phase 7's students are the solver's seats
and the decision models of level 4.

**In one paragraph.** Learn in text first, in code second, in weights last. Every system that improves an agent without
retraining has the same three parts: a harness that scores an attempt, a reflection step that turns the verdict and the
trace into a written lesson, and a store that keeps the lesson and feeds the next attempt. So the capability is built as
a small loop that runs many times, not as one large model. **The one number to track:** for each family of problems, the
score on **sealed variants** after DCLab has learned from **N challenges** of that family. If it rises while N stays
small, DCLab is learning the family; a pass on the original challenge alone proves nothing (it may have been memorised).
That curve is the product.

**Record everything; train three kinds of model and an agent (the owner, 2026-10-19).** Learning without retraining is
the first gain, never the only use of an attempt. Every action and every decision of every run (each question asked and
its answer, each tool called with its arguments and what it returned, each choice with the state it read and its reason,
each card and gate, each grader verdict, each person's correction or acceptance, each failure and its recovery) is
recorded whole, before anything is shortened, and stored permanently (the owner's standing rule: capture everything, never
truncate before saving, keep it all for training). The same traces are the training data of DCLab's models: the decision
models (the students, phase 7), the **workflow LLM** (one answer at each step of the workflow, with its reasoning), and the
two combined in the workflow; and, beside them, an **agentic LLM**: one model fine-tuned to *be* the agent, so that it
understands a machine-learning problem and solves it by itself (it reads the problem and the data, asks what it must,
plans, calls DCLab's tools, reads what they return, recovers from its own errors, decides when the work is done and reports
it, end to end) inside DCLab's harness, where the rules, the validator and the gates still decide what may happen (Part
F2). The loop's attempts, the team's walks, the intern's sessions, the Home agent, self-play and the twin runs are its
curriculum; nothing a run does is thrown away.

---

## Part A. What the research shows

Each work solves one piece of the loop (the Build Guide's reading).

| Source | What it does | Key result | What DCLab takes |
|---|---|---|---|
| ACE (Stanford, SambaNova, Berkeley) | three roles (generator, reflector, curator) grow a playbook of itemised bullets, each with an id and helpful/harmful counters; updates are small deltas merged by plain code, not by a model's rewrite | +17.1 points on AppWorld, +8.6 on finance tasks, 86.9% lower adaptation latency; works from execution feedback alone | the playbook of level 2, in delta edits only: a full rewrite once collapsed a context from 18,282 tokens to 122 and fell below the no-learning baseline |
| GEPA (ICLR 2026) | reads whole execution traces, reflects in words on why a module failed, rewrites that module's prompt; keeps a Pareto set of candidates | about 10 points over GRPO (RL) with up to 35 times fewer rollouts; better than MIPROv2 on every benchmark; prompts up to 9.2 times shorter | reflect on traces, never on bare scores; keep several playbook lineages; RL last, if ever |
| ExpeL | collects successful and failed trajectories, extracts insights by comparing pairs, keeps them with counters (add, edit, upvote, downvote; removed at zero); retrieves similar successes | beat ReAct on HotpotQA, ALFWorld, WebShop; insights alone and retrieval alone each lose to the two together; insights transferred to FEVER | levels 1 and 2 together; the pass/fail pair on one pack is how diagnosis isolates a cause |
| Voyager (NeurIPS 2023) | an automatic curriculum proposes the next task; the agent writes code skills, verifies them, stores them in a library | 3.3 times more unique items, 15.3 times faster progress; skills transferred to a new world | level 3: skills are code, indexed by what they do; the next task at the edge of what the agent can do |
| The adaptation survey (2025, arXiv 2512.16301) | what is adapted (the agent or its tools) and what signals it | weight updates are costly, forget, and credit poorly; memory and skill libraries are cheap and modular, capped by the frozen model | the first year in "tool adaptation"; weights only when the frozen model is the ceiling |
| MLE-bench (OpenAI, ICLR 2025) | 75 Kaggle competitions graded locally against the private leaderboard | the best agent medalled in 16.9% at one attempt, 34.1% at eight; more time helped, more GPUs did not; no leaderboard memorisation found | an outside sealed exam and the honest baseline: frontier agents medal in under a fifth, so a narrow, well-graded family is the right scope |

**The common pattern:** nobody learns from a bare score. They learn from **traces plus a verdict**, written into a
**store** the next run reads.

The works, as the Build Guide cites them: *Agentic Context Engineering: Evolving Contexts for Self-Improving Language
Models* (ACE); *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*; *ExpeL: LLM Agents Are
Experiential Learners*; *Voyager: An Open-Ended Embodied Agent with Large Language Models*; *Adaptation of Agentic AI: A
Survey of Post-Training, Memory, and Skills* (arXiv 2512.16301); *MLE-bench: Evaluating Machine Learning Agents on Machine
Learning Engineering*. The numbers above are the guide's reading of them, not measured here.

## Part B. Four levels, five approaches

Every lesson is written to exactly one of four levels, chosen on purpose (the Learning Loop). The five approaches are how
each level is built, cheap and inspectable first.

| Level | What changes | Approach | Cost per lesson | Generalises? | Who approves | Main risk |
|---|---|---|---|---|---|---|
| 1. Memory | a judged case joins the case library (request, data card, decisions, verdict, lesson), retrieved by structured filters, then similarity | **A** experience memory (ExpeL) | minutes, local | only as far as retrieval finds the right case | automatic, once judged | a case that looks alike but misleads |
| 2. Knowledge | a bullet in a versioned playbook per task kind and modifier, with helpful/harmful counters; small delta edits | **B** evolving playbook (ACE, GEPA) | minutes to an hour, one reflection call | yes, when written as a rule of thumb, not a case | a person reads the wording | bloat, or a rewrite that drops old lessons (context collapse) |
| 3. Capability | a new recipe part, modifier, leakage check or grader test, as code: a coding agent's pull request, certified on variants, approved by a person | **C** skills as code (Voyager) | hours to days a skill | yes, and enforced, not suggested | certification, then a person | code that fits only the original challenge |
| 4. Model | the weights of the local Qwen workflow model (~8B, with `<think>`), of the decision models, and of the **agentic LLM** that acts as the agent itself (Part F2), from accepted trajectories | **D** LoRA SFT, then, optional, **E** RL from the grader | 300 to 2,000 clean examples, about 30 hours a run on the M1 Max 64 GB; the agentic LLM needs whole trajectories, several thousand steps; RL thousands of rollouts (GRPO used 24,000 in GEPA's comparison) | yes, without the long prompt | the sealed exam | locking in mistakes from the data, forgetting; with E, gaming the grader; for the agentic LLM, acting where only a rule or a person may |

A and B are where the loop starts and where most of the first year's gain is expected. C makes a lesson permanent and
enforceable, and matches DCLab's rule: rules execute, the language model understands, people approve. D pays off only
once A to C have produced a few hundred accepted trajectories, which is the set D needs. E is optional and last. DCLab's
API budget is tight, and A to C run mostly on the local model.

## Part C. The architecture: one loop, four stores

One loop runs for every challenge. The **solver** works a Challenge Pack without its answer key; the **grader** scores
its trace against the rubric; the **reflector** compares the failed and passed attempts and writes a lesson; the
**curator** files the lesson as a small delta into the store that fits; the next attempt reads all four stores. The
**variant generator** makes the learning, check and sealed sets, so a lesson is tested on problems never seen.

| Seat | What it is in DCLab | Runs on |
|---|---|---|
| Solver | the product as it is (rules, a language model, and a person where the workflow asks one): the Home agent, the coordinator's walk (or the team), the build, the engine's stages, the recipe registry, with the four stores read as evidence; it never sees the key | the local model where it is good enough; Claude where a step needs it, capped |
| Attempt trace | every decision with its reason and the data it read: the attempt's four cards, the ledger's decision records (19.6, 19.7) with their answers and states, each stage's run record (19.5), the experience events, and the files the solver opened; one object an attempt, what the grader and the reflector read | recorded as the product records today |
| Grader | new, deterministic code (`dclab_core.grader`); no model in the scoring path | local, no API |
| Reflector | a new role (its prompt a file, hash-gated), reading the trace, the grader's diff and, when one exists, a passing trace of the same pack; never the answer key | Claude for reflection, capped per challenge (the Build Guide), or the local model where it measures as good |
| Curator | plain code applying ACE's rules to the playbook and filing cases; never a model's rewrite | local |
| Variant generator | per family, code over `dclab_core/replicas` and benchgen | local |

The four stores are the four levels: case memory and the playbook change with every challenge; skill code changes only
through a reviewed pull request; the SFT set grows passively from accepted runs and is used only when large enough.

**Capture is part of every seat (the owner, 2026-10-19).** The attempt trace is kept whole and permanently, in a form a
model can be trained on, not only read: every message the agent saw and wrote, every tool call with its arguments and its
whole result, the state each decision read, the reason given, the verdict and what a person changed. It is written
before anything is cut for a contract field (the never-truncate rule), tagged by who produced it (a rule, a student, the
local model, Claude, a person) and by its consent, and archived (`make archive`). An attempt that failed is kept as
carefully as one that passed: the pair is what teaches (ExpeL), and the failure is the agentic LLM's lesson in recovery.

### What every turn of the loop records, and where it is kept (the owner, 2026-10-19)

The loop is itself a source of training data, as rich as the attempts it grades: each seat's inputs, outputs and every
detail between are recorded, whole and before anything is cut, linked by ids, and kept permanently. One iteration of the
loop (a pack, its attempt, the grade, the lesson, the delta, the next attempt's reads) can be replayed from its records
alone, and every number on the scoreboard traces back to them.

| Seat | What it records (inputs, outputs, every detail) | Kept as | What it trains |
|---|---|---|---|
| Challenge Pack | the pack's version and hash, the request text, the data or the replica's generator and seed, the rubric's version, the answer key's version and hash (the key itself stays outside the solver's reach), the source | the pack folder in the repository, its manifest in the lake | the answer key's decision points are gold labels for the decision models, kept apart from the solver until a variant retires |
| Variant generator | every variant: its seed, each generator parameter and its value, the set it went to (learning, check, sealed), the reference solution's result on it, why a variant was refused (unsolvable, a copy of another) | one variant record each, the sets' manifests, the sealed partition sealed as the decision sets are | the curriculum: which variants were hard, which parameter made them hard |
| Solver (the attempt) | every message the agent saw and wrote, its reasoning, every tool call with its arguments and its whole result, the state each decision read and the answer with its reason, the four cards and the gates, each store item it retrieved (case ids, bullet ids, skill versions) and whether it was used, every file it opened, each model request (model, prompt version, tokens, cost, time), errors and recoveries, what a person changed | the attempt trace (31.13's trajectory record), the ledger's decision records, the stages' run records, the experience events | the workflow LLM (each step's answer with its reasoning), the decision models (each decision point), the agentic LLM (the whole attempt, turn by turn) |
| Grader | its inputs (the trace id, the key's and the rubric's versions, the grader's version), every output field, each decision point's comparison (the key's value, the attempt's, the stated reason, the weight), each hard-fail condition checked, the outcome score's baseline and target, the attacks it was tested on | one grade record an attempt, beside the trace | pass/fail labels and preferences (pass against fail on one pack), the reward of any later RL, never a model in the scoring path |
| Reflector | what it read (the failed trace, the grader's diff, the passing trace when there is one), its prompt version, the whole model request and reply, its draft lessons including the ones refused and why, the gap type it chose, the first wrong step it found | one reflection record, the lesson drafts with their evidence | a reflector model in time (trace and diff to an accepted lesson), the agentic LLM's recovery from its own mistakes |
| Curator | each delta: the store, the item before and after, the rule of ACE it applied (append, bump a counter, merge, retire), the check-set re-pass of a merge with its score, a person's approval or refusal with their words, the store's version after | an append-only delta log a store, every version of every store recoverable | what a good lesson is: each lesson beside its measured effect on the check variants |
| The next attempt's reads | which cases, bullets and skills each later attempt retrieved, and its result: each item's helpful and harmful counts come from these records, not from a guess | the retrieval records in the attempt's trace | retrieval and the playbook's own measurement |
| Promotions and the scoreboard | each promotion's evidence (the sealed variants read, once), who approved, when and why; every scoreboard row with the records it was computed from | the scoreboard rows beside `capability.json`, the approvals in the audit | the learning curve itself, the product's evidence |

**How it is stored.** Every record carries the ids that join it to the others (pack, variant, attempt, trace, grade,
lesson, delta, store version, and the run of the loop), the producer (a rule, a student, the local model, Claude, a
person) and its consent. It is written where it happens, before any field is cut, and goes into the lake's bronze layer as
it was, the silver and gold sets built from it by code (the per-point decision sets, the workflow set, the agentic
trajectories, the preferences, the lessons with their effects), and into `training_records/` by `make archive`. The sealed
variants' attempts are kept as carefully as the others, and marked as the exam's: they never enter a training set for
their family while that family is still measured on them. A failed attempt, a refused lesson and a rolled-back delta are
kept as surely as the successes: they are what the reflector and the agentic LLM learn from.

## Part D. Rules and guardrails

**Three rules that never change** (the Build Guide): the solver never sees the answer key; a rule is never loosened to
make a challenge pass; a person approves every code change and every promotion to product default.

DCLab's own rules on top: a capability counts only on variants never seen (a lesson written while looking at check
variants is invalid, the discipline of "score the holdout once" applied to the agent); every lesson, case, bullet and
skill cites its evidence and carries its version (a lesson without evidence is refused; any promotion is rolled back in
one change); Claude teaches, rules and people grade (Claude's attempts are evaluation material; labels come from answer
keys, rules and people; silver labels only from real projects, tagged and weighted apart); private challenges teach their
general pattern only (the owner's fraud data: rules only, never a column name in a model's context); enhance, never
replace (every seat calls existing functions); capture everything (the full input and output kept before anything is
cut); spend capped per challenge, planned and printed before a run, reported after.

| The agent may, on its own | It may propose, a person approves | Never |
|---|---|---|
| add judged cases to the case library | new playbook bullets beyond a family's first, pattern cards, checklist questions, prompt edits | read an answer key, a check or a sealed variant before or during an attempt |
| draft a diagnosis and its gap type | new recipe parts, checks, gates, modifiers, grader tests (a pull request) | weaken or remove a rule to make a challenge pass |
| generate learning variants | promotion to certified and to product default | put answer keys into the case library while the challenge still evaluates |
| produce training samples from judged trajectories | starting a training run of its own models | learn from private data outside its consent level |

**How the loop fails quietly, and the check that catches each** (the Build Guide):

| Failure | How it shows up | The check |
|---|---|---|
| answer-key leakage | check scores jump near 100% in one step | the key lives outside the solver's working directory and sandbox; the grader logs every file the solver opened; a run that opened the key is void |
| overfitting to the seed challenge | the learning score rises, the sealed one does not | promotion reads only sealed variants; a lesson that helps learning but not check variants is retired |
| context collapse | the playbook shrinks sharply after a curation, scores fall | the curator only appends or bumps counters; a merge must re-pass the check set before it replaces the old playbook |
| playbook bloat | prompt cost grows, the solver ignores most bullets | a token cap a playbook; the counters retire unused bullets |
| weakened rules | a challenge starts passing because a check was relaxed | rule files are owned by a person; a test asserts the rule set only grows stricter, and blocks a pull request that lowers a threshold or removes a check |
| grader gaming | the decision score rises, the real holdout metric does not | decision and outcome scores reported side by side; a gap above a set margin fails the run |
| a contaminated outside exam | the exam is in the model's training data | rotate the outside set yearly; log the model's familiarity |
| budget burn | spend a challenge climbs | a hard cap a challenge; reflection calls batched; solving on the local model |
| lost lineage | nobody can say why a lesson exists | every bullet, case and skill carries its pack id, trace id and grader version |

## Part E. The harness, concrete

The harness is what makes learning measurable: everything else is replaceable, this is not.

**The Challenge Pack** (`dclab_core.challenge`, a frozen versioned contract with its JSON schema), one folder a challenge,
versioned in the repository:

```text
challenges/dup_leakage/retail_returns_001/
    challenge.yaml          # the request text, task kind, family, modifiers, budget
    data/                   # the tables as the person would hand them over, or a replica generator and its seed
    answer_key.json         # hidden: the expected value at each decision point, the traps and whether each must be caught
    rubric.yaml             # weights per decision point, hard-fail conditions, the pass threshold
    variant_generator.py    # takes a seed, returns a new pack of the same family
    source.md               # where it came from (interview pattern, pilot, public dataset, invented, a production failure)
```

**The answer key holds decisions, not a model score:** the expected value at each decision point the workflow already
records in the ledger (`task.kind`, `split.strategy`, `feature.leakage_suspect`, `target.column`,
`training.families_budget`, the modifiers' points), the fit verdict and the success bar of the Solution Brief (30.13,
30.14), each planted trap and whether it must be caught, and the honest score range.

**The grader** is deterministic code, no model in the scoring path. It reads the attempt's trace (the cards, the ledger's
decision records, the run's records) and returns:

| Field | Meaning |
|---|---|
| `decision_score` | weighted agreement with the key across decision points, 0 to 1 |
| `trap_caught` | true when every hard-fail condition of the rubric was avoided |
| `outcome_score` | the real holdout metric, scaled between the key's baseline and target |
| `verdict` | pass only when `trap_caught` and `decision_score` reaches the rubric's threshold |
| `diff` | each decision point where the attempt and the key disagree, with the attempt's stated reason |

The `diff` is what the reflector reads (GEPA's point: the lesson comes from the trace and the disagreement, not from the
number). A holdout above the honest range is a sign of leakage, never a success. **The grader is written before the
variant generator** (a grader one trusts makes every other number meaningful), and it is attacked before it is trusted:
planted gaming traces (a right number with wrong decisions, a decision right for the wrong reason) must fail.

**The variant generator**, one a family. For duplicate and group leakage: rename columns, change the group key, change
the duplicate rate (2%, 10%, 30%), move the leak to another column, switch the task kind from binary to regression while
keeping the leak, add a decoy column that looks like a leak and is not, and change the domain skin (the same trap as
insurance claims, telecom accounts, hospital records). A variant is accepted only when a reference solution passes it and
it differs from every existing variant on at least one generator parameter.

**Three sets, fixed at creation, near 5 : 3 : 2.** Learning (the agent sees them and learns), check (scored after each
lesson, never learned from), sealed (opened only at a promotion; a sealed variant once opened is retired; generated by an
independent process: another seed partition, and domain skins written by an agent that never read the rules or the
lessons).

**The scoreboard:** one row a family a run (`challenges_learned`, `check_score`, `sealed_score`, `playbook_tokens`,
`api_spend`), kept beside `dclab_rnd/kinds/capability.json` so the twin-run report reads it; plotted as the sealed score
against challenges learned.

**The outside exam:** 5 to 10 low-complexity MLE-bench competitions as a permanent sealed set no DCLab lesson is ever
derived from. It answers the question a buyer will ask: does the learning carry beyond DCLab's own data?

## Part F. The learning mechanics, level by level

**Reflection (every level).** The reflector receives the attempt's trace, the grader's `diff` and, when one exists, a
passing trace of the same pack; never the key. It writes one structured lesson (`dclab_core.lesson`, `Lesson` v1):

```yaml
family: dup_leakage
gap_type: rule            # knowledge | reasoning | decision | rule | capability | data
decision_point: split.strategy
observed: chose random k-fold although order_id repeats across rows
expected: group k-fold on order_id
lesson: when any id column repeats, check whether rows share an entity before choosing the split
generalises_to: any tabular task with a repeated id column
confidence: 0.8
evidence: [pack id, trace id, grader version]
```

The most useful lessons come from a pass/fail pair on the same pack (ExpeL: the difference isolates the cause); the
reflector reads the trace, so it can say why the split was wrong, not only that it was (GEPA). **The first wrong step**
decides the lesson: a wrong metric in phase 4 is often a modifier missed in phase 2. The grader names the failing traps
and decisions mechanically; the finder walks the trace's decisions in order (the ledger's points and their lineage) to the
earliest one that differs from the key; the reflector proposes the gap type; a person confirms it for rule and capability
gaps, because those lead to code.

| Gap | Looks like | The lesson goes to | Level |
|---|---|---|---|
| knowledge | the question that mattered was never asked (review capacity) | a playbook bullet or a checklist question | 2 |
| reasoning | the facts were there, the conclusion was wrong | a critique and corrected samples for SFT | 4 |
| decision | one typed decision was wrong (`split.strategy`) | a labelled sample for that point's student | 4 |
| rule | no check stopped a known trap (a feature computed with future data) | a new check or gate rule | 3 |
| capability | the pipeline cannot do what the problem needs (pair matching) | a new recipe part or kind | 3 |
| data | the data cannot answer the question | nothing: the right answer is "does not fit"; the pack is relabelled | none |

Every case also goes to level 1, whatever its gap.

**Level 1, case memory (A).** Every graded attempt is one row: structured fields (task kind, modifiers, family, verdict,
decision values) and a representation of the request and the data card (bands and roles, never values). At the next
attempt, retrieval filters on the structured fields first, then ranks by similarity: fail-closed, as the main product's
`rd_knowledge.py` and this studio's measured retrieval (`dclab_rnd/retrieval`, 60 questions) do. A retrieved case is
shown to the solver as an example, never as an instruction. Measured as an arm, memory on against off, on check variants.

**Level 2, the playbook (B).** One playbook a task kind and modifier combination, a list of bullets, each with an id, the
lesson's text and helpful/harmful counters. The curator applies ACE's rules: (1) a new lesson is appended with a new id,
nothing is rewritten; (2) a lesson that already exists (similar above a threshold) bumps its counter instead of
duplicating; (3) the solver marks which bullets it used and the grader's verdict increments helpful or harmful on them;
(4) a bullet whose harmful count exceeds its helpful one by 3 goes to quarantine, never deleted; (5) past 2,000 tokens the
curator proposes merges, and the merged playbook must re-pass the whole check set before it replaces the old. Two or three
lineages a family are kept (GEPA's Pareto set), and the lineage winning the most check variants is promoted, not the one
that won the last. A prompt may also evolve by reflection on its graded failures (GEPA). Every edit passes
`make agent-eval`, `make harness-eval` and the whole curriculum before it serves.

**Level 3, skills as code (C).** Triggered when a lesson's gap is a rule or a capability and the same lesson appeared in two
or more families or three or more packs. A coding agent receives the lesson, the failing traces and the recipe contract,
and opens a pull request (in its own worktree) adding one of: a recipe part, a modifier, a leakage check, a grader test. It
follows the Task Recipes' ten steps, one step a task: (1) decide kind or modifier (what is predicted changes: a kind; only
the data conditions: a modifier, most requests); (2) fill the contract (a modifier only the fields it changes), the
detection rule with numeric thresholds; (3) build the trap replica in `dclab_core/replicas` with its expected answers;
(4) register the decision point (`modifier.<name>` or `task.kind`) in the ledger; (5) implement only the steps the contract
names (split, metric, threshold, leakage check, artifact fields), no new pipeline; (6) extend the artifact check (save,
reload, predict); (7) add the report items (what was detected, why, the honest result against the baseline); (8) run
certification; (9) run twin runs and keep checked samples; (10) promote one level at a time with its evidence. The pass
criteria, all of them: detection picks the right kind and modifiers on every trap replica; every planted trap is caught;
the resolved recipe beats or matches the mandatory baseline on the holdout, or says honestly that it does not; the artifact
passes save, reload and predict; the result is the same with the language model off. The ladder: **experimental** (R&D
only, contract filled, unit tests pass), **certified** (product behind an opt-in flag: passes on its trap replicas, its
sealed variants and at least two real datasets), **product default** (chosen automatically: a person approves, no wrong
detection on the whole R&D dataset set, no regression in twin runs, and it has survived two more families).

**Level 4, models (D, then E).** Every attempt with a passing verdict and no human override is exported (the lake's
families, the SFT corpus format) with its decision points as labels, tagged with the playbook bullets it read, so a tuned
model can be measured for whether it internalised a bullet (its score without the playbook). Training starts only when a
family has 300 or more such examples; the sealed variants stay the evaluation. The LoRA run on Qwen (~8B) is the owner's;
the tuned model walks onto a role through 27.8 (shadow first, served only where it measured better, retired in one
change). RL from the grader only if stage 3 plateaus, only once a red team cannot raise the reward without raising the
real score, starting with the smallest decision model, and only with the owner's approval (it needs rented GPUs).

**The curriculum.** As in Voyager, the next challenge sits at the edge of ability: the family with the lowest check score
that still has unused learning variants, and in it the variant whose generator parameters are farthest from any already
learned. Every tenth run is a regression pass over every family, so an old lesson cannot silently break; every promotion
re-runs the whole curriculum.

## Part F2. The agentic LLM: a model that is the agent (the owner, 2026-10-19)

**What it is.** Beside the workflow LLM (asked one step at a time by DCLab's code) and the decision models (one small
choice each), DCLab fine-tunes one LLM to act as the agent itself: given a problem and its data, it runs the whole solving
loop by itself, choosing the next action at every turn. The owner's words: *fine tune a LLM model that behaves like an agent
itself, and understands and solves by itself.* It is the difference between a model that answers a question DCLab asks and
a model that decides which question to ask, which tool to call, what the result means and what to do next, until the
problem is solved or it says honestly that the data cannot answer it.

**What it is trained on.** Whole agent trajectories, one example a run, in the tool-calling chat format the base model
was trained for: the system's tools (the registry's, `dclab_rnd.agents.default_registry()`, never a private list) and
rules, the person's words and the data card (bands and roles, never a value), then each turn: the model's reasoning, the
action (a tool and its arguments, a question to the person, a stop), and what came back. Sources, each tagged:

| Source | What it gives | Use |
|---|---|---|
| the harness's agents (the Home agent, the intern, the team's walks, 21.1, 28.x) | real turns, tool calls and results, the validator's refusals | training, once accepted by the grader or a person |
| the learning loop's attempts (31.x) | a pass and a fail on the same pack, the first wrong step, the lesson | training, the pair as a preference |
| self-play (27.3) and the journeys' written people (20.10) | whole verified solves, from the first sentence to the holdout number | training (verified by construction) |
| a person's corrections (29.10, the Solution page, the fit edits of 30.13) | where the agent was wrong, and the right action | training, weighted above the agent's own |
| Claude as an AI-first agent (the twin runs, 25.x) | a strong agent's trajectories on the same problems | evaluation and comparison by default; as training only under the owner's rule for Claude's outputs (tagged, apart, removable in one step) and a check of the provider's terms |

**How it is trained and judged.** The same ladder as Level 4: LoRA SFT on accepted trajectories (the owner's run), then
preferences from pass/fail pairs, then, optional and last, RL from the grader. It is judged on the loop's sealed variants
and the outside exam (31.12) beside the current system and Claude, by the same grader, and on what matters for an agent:
the share of problems solved end to end, the first wrong step, the moves the validator refused, the questions it asked,
the turns and the cost. **What never changes:** it acts only through the registry's tools and their guards; every move
passes the graph validator; the holdout is still read once, after the gates; a sealed set is never trained on. It serves in
shadow first (its actions logged beside the harness's, never used), then through the same routing levels as a student
(flag, act with a check, act), each step on measured evidence and the owner's switch.

**What exists to start from.** The harness records every agent step (`agents/harness.py`, traced); `studio/trajectories.py`
exports opted-in projects' moves and agent steps; the lake has an `agent_step` family (26.1); the decision ledger keeps
every choice (19.6); the twin runs keep Claude's whole agent record (25.3). **What is new:** one trajectory record whole
enough to train on (every turn, tool call and full result, never cut), its builder into the tool-calling format with a
sealed partition by problem family, and the agentic model's exam (31.13 to 31.16).

## Part G. What exists and what is new

| Loop part | What DCLab already has | What is new |
|---|---|---|
| Challenge Pack | 19 replicas with planted traps and acceptance tests (`dclab_core/replicas/`, `kinds/acceptance.py`), retail returns with copied bills and rule labels (30.9); benchgen's planted tables (22.1, 29.2); the journeys' written people and sealed sets pinned by hash (20.10); the eleven interview patterns and practice questions (30.8); real problems (`real_problems_v1`) | the pack contract and folder; a converter for what exists; the answer key's isolation |
| Solver | the Home agent, the coordinator's walk and the team (21.2, 28.x), the engine, the recipe registry and modifiers (30.10, 30.11), self-play (27.3), twin runs (25.x) | running a pack end to end without its key, the stores read as evidence, every file it opens logged |
| Grader | acceptance decisions per kind; the journeys' four parts; the benchmark's 108 cases and its AND gate; the ledger's decisions and labels (19.6 to 19.8) | one deterministic grader over any rubric, with the diff and the attacks it must fail |
| Reflector, diagnosis | the ledger's records and labels, lineage (19.5), failure signals (19.9), the critic and auditor verdicts (28.4, 28.5), lessons proposed after a project (`dclab_rnd/lessons.py`) | the first-wrong-step finder, the reflector role, `Lesson` v1 |
| Stores | workspace lessons and project memory; the evidence index and retrieval; pattern cards and the business-examples base (30.5 to 30.8); prompts as hash-gated files; the recipe registry with its levels; the decision models and students (22.x, 29.x); the lake and the SFT corpora (26.1, 26.9); the tuned-model walk (27.8) | the case library, the playbook with counters and lineages, the coding agent's pull request path |
| Verify, promote | sealed sets read once, gates fixed before runs, certification levels, `make kinds-check` | variants in three sets, the curriculum runner, the "rules only grow stricter" test |
| Scoreboard | `capability.json`, the evaluation board (25.8), report cards (29.14, planned) | the row a family a run, the learning curve |

## Part H. The packages (31.x), stage by stage

Five stages, each with its exit test; no stage starts until the previous exit test passes, because every later stage
depends on the grader of stage 0 being trustworthy. Each package follows the house discipline (test first, the reviewer,
its development record, the gate, capture in its "Done when", gates fixed before any measured run).

| Stage | Package | Delivers | Done when (the exit test) |
|---|---|---|---|
| **0. The harness** (about 2 weeks; local, no API) | 31.1 The Challenge Pack | `dclab_core.challenge` v1 and the folder layout; the converter for the existing replicas (retail returns, delivery fraud, the kinds' cases) and the interview patterns; the key kept outside the solver's reach | three hand-made packs, and a test that the solver's path never opens a key, a rubric or a variant set |
| | 31.2 The grader (before the variants) | `dclab_core.grader`: decision score, trap caught, outcome score, verdict, diff; the honest range; the planted gaming traces | the three packs grade correctly; a known-bad trace scores low; every gaming trace fails; every replica's acceptance decisions grade as today |
| | 31.3 Variants for one family | the generator for duplicate and group leakage (renames, the group key, the duplicate rate, the leak moved, binary to regression, a decoy, domain skins), three sets near 5 : 3 : 2, the sealed one generated independently | 20 generated variants all solvable (a reference solution passes) and all different (at least one parameter); the sets never share a seed |
| | 31.4 The curriculum and the scoreboard | `make curriculum`, the scoreboard row a family a run, the baseline: today's DCLab on check and sealed variants with nothing learned | the baseline stored with its intervals: the zero of every learning curve |
| **1. Memory + playbook** (about 3 weeks; the local Qwen solves, Claude reflects, capped) | 31.5 Reflect and diagnose | the reflector role, the first-wrong-step finder, the gap classifier, `Lesson` v1 with its evidence | on planted misses the first wrong step is found; every lesson cites its pack, trace and grader version |
| | 31.6 Case memory | the case table (structured fields first, then similarity; PostgreSQL, a vector index only as a measured arm), retrieval into the solver as examples | an arm, memory on against off, on check variants, gate fixed before |
| | 31.7 The playbook | a playbook a kind and modifier, bullets with ids and counters, ACE's five curation rules, two or three lineages | the check-variant score rises after learning from 3 to 5 packs; the whole regression set still passes; the playbook stays under 2,000 tokens |
| **2. Skill code** (about 4 weeks for the first family, then faster) | 31.8 Skills as code | the coding agent's pull request following the Task Recipes' ten steps; certification on learning, check and sealed variants and the curriculum; the "rules only grow stricter" test | a new modifier earns certified on sealed variants it never saw; no existing rule weakened |
| | 31.9 A second family | the loop on a family it has not seen (rare positives with review capacity, or near-duplicate matching from 30.16) | the second family's learning curve stored: the loop generalises to new families, not only new variants |
| **3. Fine-tune** (after 300+ accepted runs; about 30 hours a run on the M1 Max 64 GB) | 31.10 The learning export | accepted trajectories and decisions into the lake's families, tagged with the bullets read; LoRA on the Qwen workflow model and the decision-point models (the owner's run) | the tuned model without the playbook matches the base model with the playbook on sealed variants |
| **4. RL** (optional; only if stage 3 plateaus; rented GPUs) | 31.11 RL from the grader | GRPO-style training on the grader's reward, the smallest decision model first | the sealed score rises, and a red team cannot raise the reward without raising the real score; the owner's approval |
| outside | 31.12 The outside exam | 5 to 10 low-complexity MLE-bench competitions as a permanent sealed set | the owner's approval for the downloads; scored once a promotion of the whole loop |
| **capture** (from day one, beside every stage) | 31.13 The whole trajectory record | `dclab_core.trajectory` v1: one run as turns (what the agent saw, its reasoning, the action, the tool's whole result), written by the harness for every agent and every loop attempt before anything is cut, tagged by producer and consent, archived | a test reads back every turn of an intern session, a team walk and a loop attempt, whole; nothing shortened before it was saved |
| | 31.17 The loop's own record | `dclab_core.loop_record` v1: the variant, grade, reflection, delta and read records of Part C's table, joined by their ids, written by each seat where it acts, into the lake's bronze layer and the archive; the loop's datasets (decisions from answer keys, pass/fail preferences, lessons with their measured effect) built from them | a test replays one iteration of the loop from its records alone (pack, attempt, grade, lesson, delta, the next attempt's reads) and traces a scoreboard row to them; nothing shortened before it was saved; a sealed variant's attempt never reaches a training set of its family |
| | 31.14 The agentic training set | the builder into the tool-calling chat format, a sealed partition by problem family, pass/fail pairs as preferences, Claude's trajectories apart under the owner's rule | the set's manifest; no sealed family in training; a whole trajectory replays through the registry's tools |
| **agentic model** (after 31.10; the owner's runs) | 31.15 The agentic LLM, fine-tuned | LoRA SFT of the base model on 31.14 (the owner's run), then preferences | on sealed variants it solves more problems end to end than the base model in the same harness, with no move the validator would refuse |
| | 31.16 The agentic LLM beside the agents | the tuned model in shadow beside the harness's agents, then through the routing levels on measured evidence | its shadow record on real sessions; served only where it measured better, retired in one change |

**What to track from day one:** for each family, the challenges learned from, the check score, the sealed score, the
playbook's size and the spend a challenge.

**The families after the first.** The Task Recipes' trap replicas give the next ones, each with its trap and its right
behaviour already written: rare positives (a 1% positive rate: PR-AUC or precision at K, a warning on accuracy), groups
(copies of one entity across rows: a group split, the holdout close to the honest score), time order (a feature computed
with future data: flagged, a temporal split), noisy labels (an outcome built from a rule that is also a feature:
confirmed labels only, the rule feature flagged), forecasting (strong seasonality: beat the seasonal naive baseline or say
it does not), anomaly scoring (planted anomalies: most found in the top K). Each becomes a family with its variant
generator once the first family's curve shows the loop works.

**The honest baseline:** on MLE-bench the best frontier agent medalled in under a fifth of competitions on one attempt.
DCLab does not need to beat that; it needs to show that on a narrow family its sealed score rises with each challenge
learned, and that each lesson is written where a person can read it.

## Part I. Decisions for the owner

| Decision | Needed by | Proposal |
|---|---|---|
| The first family and its seeds | 31.1, 31.3 | duplicate and group leakage, seeded from the retail returns replica (30.9) and the copied bills; the Build Guide's Otto and Rossmann seeds are Kaggle datasets not in this repository (Rossmann is planned in the forecasting track): added when the owner downloads them |
| Who writes the sealed variants | 31.3 | an independent agent that never read the rules or the lessons (as the sealed journeys); a person later |
| The reflector's model and budget | 31.5 | Claude, capped per challenge and batched (the Build Guide); the local Qwen where it measures as good |
| The solver's local model | 31.6, 31.7 | Qwen ~8B (the solver's base) through the gateway's local route; today only a 1.5B coder model is installed |
| Similarity for case memory | 31.6 | structured filters and the measured lexical ranking first; a vector index (pgvector) needs a local embedding model (Anthropic has none) and is added only as a measured arm |
| Who approves code skills | 31.8 | the owner, as a pull request review |
| Fine-tuning runs | 31.10 | the owner's, on the M1 Max, when a family has 300+ accepted examples |
| RL and rented GPUs | 31.11 | not before 31.10's result; the owner's decision |
| MLE-bench downloads and compute | 31.12 | the owner's choice, a low-complexity subset first |
| The agentic LLM's base model | 31.15 | a ~8B model trained for tool calls (Qwen's line), the same family as the workflow LLM so one runtime serves both; larger only if the exam says the size is the ceiling |
| Claude's trajectories as training data | 31.14 | evaluation and comparison by default; training only as the owner's rule for Claude's outputs allows (tagged, apart, removable) and after a check of the provider's terms |
| The agentic LLM's fine-tuning runs | 31.15 | the owner's, when 31.14 holds a few thousand accepted steps across at least three families |

---

## خلاصهٔ فارسی

**هدف فاز ۹:** هر چالش تازه‌ای که به DCLab داده می‌شود، توانایی حل همهٔ مسئله‌های مشابه آن خانواده را بیشتر کند، نه فقط
همان یک مسئله. **یک عدد برای دنبال کردن:** نمره روی نسخه‌های مهروموم‌شدهٔ یک خانواده بر حسب تعداد چالش‌هایی که عامل از
آن‌ها یاد گرفته است. اگر این منحنی با تعداد کمی چالش بالا برود، یادگیری واقعی است؛ این منحنی خودِ محصول است.

**وضعیت:** فقط برنامه. به دستور مالک، تا پایان فازهای ۶، ۷ و ۸ هیچ بخشی از فاز ۹ ساخته نمی‌شود و ثبت و ذخیرهٔ همهٔ داده‌ها
ادامه دارد.

**ثبت همه‌چیز در حلقه (مالک، ۲۰۲۶-۱۰-۱۹):** خودِ حلقه هم منبع دادهٔ آموزشی است. هر جزء آن ورودی‌ها، خروجی‌ها و همهٔ
جزئیات را کامل و پیش از هر کوتاه‌سازی ثبت می‌کند:
- **بستهٔ چالش:** نسخه، seed، معیار نمره و کلید پاسخ.
- **مولد نسخه:** پارامترهای هر نسخه و مجموعه‌ای که در آن رفت.
- **حل‌کننده:** هر پیام، استدلال، فراخوانی ابزار با نتیجهٔ کامل، هر تصمیم و چیزهایی که از انباره‌ها خواند.
- **داور:** مقایسهٔ هر نقطهٔ تصمیم.
- **بازتاب‌دهنده:** درخواست و پاسخ کامل مدل و درس‌های ردشده.
- **گردآورنده:** هر تغییر کوچک با حالت قبل و بعد.
- **تلاش بعدی:** اینکه کدام درس‌ها کمک کردند.

همهٔ این سوابق با شناسه‌ها به هم وصل‌اند و برای همیشه نگه داشته می‌شوند. تلاش‌های شکست‌خورده و درس‌های ردشده هم نگه داشته
می‌شوند. سوابقِ نسخه‌های مهروموم‌شده هرگز وارد مجموعهٔ آموزشیِ خانوادهٔ خودشان نمی‌شوند. بستهٔ 31.17 نشان می‌دهد که یک دور
کامل حلقه فقط از روی سوابقش بازپخش می‌شود.

**ترتیب درست:** اول یادگیری در متن، بعد در کد، در آخر در وزن‌های مدل. همهٔ سیستم‌هایی که بدون آموزش دوباره بهتر می‌شوند
(ACE، GEPA، ExpeL و Voyager) سه جزء دارند: بستری که به تلاش نمره می‌دهد، گام بازتابی که از رِد اجرا و حکم داور یک درس
نوشته می‌سازد، و انباره‌ای که درس را نگه می‌دارد و به تلاش بعدی می‌رساند. هیچ‌کس از یک نمرهٔ خالی یاد نمی‌گیرد.

**چهار سطح و پنج رویکرد:** ۱. حافظهٔ موارد (A)؛ ۲. راهنمای عملی با گلوله‌های شناسه‌دار و شمارندهٔ مفید/مضر و فقط ویرایش‌های
کوچک (B)؛ ۳. مهارت به‌صورت کد: عامل کدنویس طبق ده گام «دستورالعمل وظیفه» بخش یا اصلاح‌گر تازه را با درخواست ادغام پیشنهاد
می‌دهد، روی نسخه‌های مهروموم‌شده گواهی می‌گیرد و انسان تأیید می‌کند (C)؛ ۴. تنظیم دقیق LoRA روی Qwen حدود ۸ میلیارد پس از
۳۰۰ اجرای تأییدشده (D) و در آخر، اختیاری، یادگیری تقویتی (E).

**بستر آزمون:** بستهٔ چالش (درخواست، داده، کلید پاسخ پنهان، معیار نمره، مولد نسخه، منبع)؛ داور قطعی بدون مدل که امتیاز
تصمیم‌ها، گرفتن تله‌ها، امتیاز نتیجه، حکم و فهرست اختلاف‌ها را می‌دهد و اول از سازندهٔ نسخه نوشته می‌شود؛ سه مجموعهٔ جدا
(یادگیری، بررسی، مهروموم‌شده با نسبت حدود ۵:۳:۲)؛ و آزمون بیرونی با ۵ تا ۱۰ مسابقهٔ ساده از MLE-bench.

**مراحل:** ۰ بستر آزمون (حدود دو هفته، محلی)؛ ۱ حافظه و راهنما (حدود سه هفته)؛ ۲ مهارت به‌صورت کد (حدود چهار هفته برای
خانوادهٔ اول)؛ ۳ تنظیم دقیق؛ ۴ یادگیری تقویتی (اختیاری). هیچ مرحله‌ای پیش از گذشتن از آزمون خروج مرحلهٔ قبل شروع نمی‌شود.

**ثبت همه‌چیز، و آموزش سه نوع مدل و یک عامل (مالک، ۲۰۲۶-۱۰-۱۹):** یادگیری بدون آموزش دوباره فقط سود اول است، نه تنها
کاربرد یک تلاش. هر کنش و هر تصمیمِ هر اجرا (هر پرسش و پاسخ، هر ابزاری که با ورودی‌هایش صدا زده شد و نتیجهٔ کاملش، هر انتخاب
با وضعیتی که خواند و دلیلش، هر کارت و دروازه، هر حکم داور، هر اصلاح یا پذیرش انسان، هر خطا و جبرانش) کامل و پیش از هر کوتاه
کردنی ثبت و برای همیشه نگه‌داری می‌شود. همین ردها داده‌های آموزش مدل‌های DCLab‌اند: مدل‌های تصمیم (شاگردها)، **مدل زبانیِ
گردش‌کار** (پاسخ هر گام گردش‌کار با استدلالش) و ترکیب این دو در گردش‌کار؛ و در کنار آن‌ها **یک مدل زبانیِ عامل‌محور**: مدلی که
تنظیم دقیق می‌شود تا خودش عامل باشد، یعنی خودش یک مسئلهٔ یادگیری ماشین را بفهمد و حل کند (مسئله و داده را بخواند، آنچه
لازم است بپرسد، برنامه بریزد، ابزارهای DCLab را صدا بزند، نتیجه را بفهمد، از خطای خودش بازگردد، بداند کار کی تمام است و
گزارش بدهد)، همه درون بستر DCLab که قاعده‌ها، اعتبارسنج و دروازه‌ها همچنان تصمیم می‌گیرند چه مجاز است. داده‌اش ردهای کامل
عامل‌هاست (عامل خانه، کارآموز، گام‌های تیم، تلاش‌های حلقه، خودبازی و اجراهای دوقلو)؛ ردهای Claude پیش‌فرض فقط برای ارزیابی‌اند.
بسته‌های ۳۱.۱۳ تا ۳۱.۱۶: ثبت کامل هر رد، ساخت مجموعهٔ آموزش در قالب فراخوانی ابزار، تنظیم دقیق مدل عامل (اجرای مالک) و
گذاشتن آن در سایه کنار عامل‌ها و ارتقا فقط با شواهد.

**سه قاعدهٔ ثابت:** عامل حل‌کننده هرگز کلید پاسخ را نمی‌بیند؛ هیچ قاعده‌ای برای قبولی یک چالش سست نمی‌شود؛ انسان هر تغییر کد
و هر ارتقا به پیش‌فرض محصول را تأیید می‌کند.

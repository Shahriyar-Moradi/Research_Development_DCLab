# Phase 9: the learning loop — teaching DCLab new problems (plan)

Written 2026-10-18 from the owner's words: *I want to make a framework: I give my agent a new problem, it learns, and it
adds that capability to solve similar problems; that is one of the important goals of DCLab.* The sources are the
owner's document **DCLab Learning Loop: Teaching Agents New Problems** (2026-10-18: four learning levels, the Challenge
Pack, seven steps, gap types, three variant sets, the capability scoreboard, guardrails) and the owner's note of the same
day on the feasible approaches (A experience memory, B evolving playbook, C skills as code, D fine-tuning, E
reinforcement learning) and their stages. This plan reads both against the repository at `35d6569`.

**Status: a plan only.** The owner's order (2026-10-18): nothing of phase 9 is built while phases 6, 7 and 8 are being
finished, and the capture of everything keeps running meanwhile. Phase 9 starts after them; the capture they finish is
what it learns from.

**The one number to track:** for each family of problems, the score on **sealed variants** after DCLab has learned from
**N challenges** of that family. If it rises while N stays small, DCLab is learning the family; a pass on the original
challenge alone proves nothing (it may have been memorised).

---

## Part A. What "learning a capability" means

Every lesson is written to exactly one of four levels, chosen on purpose (the owner's Learning Loop). The five
approaches are how each level is built; the cheap layers first, because they are inspectable and can be rolled back, and
fine-tuning and reinforcement learning last, because they need much more clean data than exists today.

| Level | What changes | Approach | What actually learns | Speed and cost | Who approves | Main risk |
|---|---|---|---|---|---|---|
| 1. Memory | a judged case joins the case library | **A** experience memory (ExpeL) | past attempts and their lessons, retrieved for a similar new problem | minutes; very cheap | automatic, once judged | similar-looking cases that do not help |
| 2. Knowledge | a lesson becomes a pattern card, a checklist question, a prompt's wording | **B** evolving playbook (ACE, GEPA) | the agents' written procedure, rewritten from graded failures in small edits | hours to days; cheap | a person reviews the wording | a bloated playbook, or a rewrite that drops old lessons |
| 3. Capability | the pipeline gains a recipe part, a check, a modifier or a kind, in code | **C** skills as code (Voyager) | the recipe registry and the checks | days a family; medium | certification, then a person (a pull request) | code that fits only the one challenge |
| 4. Model | DCLab's own models are retrained | **D** fine-tuning (LoRA SFT on Qwen ~8B with `<think>`, the decision models and the students) and, last, **E** reinforcement learning from the grader | the weights | weeks, in batches; hundreds to thousands of clean examples, about 30 hours a LoRA run on the M1 Max | the sealed exam | locking in mistakes from unclean data; with E, gaming the grader |

Why this order: ACE grew a playbook through generate, reflect and curate steps, and small edits ("deltas") beat full
rewrites, which wipe out detail ("context collapse"); GEPA improved prompts by reflecting on failures and beat an RL
method (GRPO) with far fewer runs; Voyager and ExpeL keep a growing library of skills or insights, choose the next tasks
deliberately and let the agent check its own results. The levels build on each other: one solved case is memory; when
several cases teach the same lesson it becomes knowledge; when it must always be applied it becomes a capability in
code; when enough trajectories exist the model learns the reasoning and needs less retrieved help.

**The real bottleneck is the grader, not the model.** None of the approaches work without three things, and they are
built first: Challenge Packs with a hidden answer key and a rubric; a variant generator that makes similar but different
problems, kept apart in learning, check and sealed sets; and an automatic grader that scores the decisions (the split,
the leakage, the metric), not only the final score.

## Part B. Principles (the owner's guardrails, with DCLab's own rules)

1. **The agent never sees an answer key**, a rubric, a check variant or a sealed variant before or during an attempt; a
   test fails if the attempt path can read them (as the sealed journeys are pinned by hash today).
2. **Rules are never loosened to make a challenge pass.** Every playbook, recipe or rule change passes the whole
   regression curriculum, the existing gates (`make check-quick`, `make agent-eval` and `make harness-eval` for a prompt)
   and its own certification.
3. **A capability counts only on variants it has never seen.** A lesson written while looking at the check variants is
   invalid: the same discipline as "score the holdout once", applied to the agent.
4. **Every lesson cites its evidence** (the challenges and gaps that produced it) and has a version; any promotion can be
   rolled back in one change. A lesson without evidence is refused.
5. **Claude teaches, rules and people grade.** Claude's attempts are evaluation material; labels come from answer keys,
   rules and people (the decision model plan). Silver labels only from real projects, tagged and weighted apart.
6. **Private challenges stay private.** A challenge built from a customer's or the owner's data teaches its general
   pattern only; its data and names never reach a model's context (the owner's fraud data: rules only, never a column
   name).
7. **Enhance, never replace.** Every step calls DCLab's existing functions; the loop is mainly the wiring that turns
   them into one cycle.
8. **Capture everything.** Every step writes a record (experience events, ledger decisions, the lake), full input and
   output kept before anything is cut, so every capability traces back to the challenges that taught it, and the loop's
   own trajectories become training data.
9. **Spending is capped per challenge.** A and B run mostly on the local model (Qwen ~8B through the gateway's local
   route); Claude only where a step needs it, planned, printed and capped before the run, the spend reported.

| The agent may, on its own | It may propose, a person approves | Never |
|---|---|---|
| add judged cases to the case library | new pattern cards, checklist questions, prompt edits | read an answer key, a check or a sealed variant before or during an attempt |
| draft a diagnosis and its gap type | new recipe parts, checks, gates and modifiers (code, as a pull request) | weaken or remove a rule to make a challenge pass |
| generate learning variants | promotion to "certified" and "product default" | put answer keys into the case library while the challenge still evaluates |
| produce training samples from judged trajectories | starting a training run of its own models | learn from private data outside its consent level |

## Part C. The loop and what already exists

Seven steps (the owner's loop): **attempt → judge → diagnose → learn → verify → promote → retire to the curriculum**,
with one way back: a lesson that fixes the original but fails fresh variants returns to diagnosis with the failing
variants as new evidence. A pass at the judge still runs verify, promote and retire on variants.

| Loop part | What DCLab already has | What is new |
|---|---|---|
| Challenge Pack | replicas with planted traps and acceptance tests (`dclab_core/replicas/`, 19 of them; `kinds/acceptance.py`); benchgen's planted tables (22.1, 29.2); the journeys' written people and sealed sets (20.10); the eleven retail interview patterns and practice questions (30.8); real problems (`real_problems_v1`) | the pack contract: request, data, hidden answer key, rubric, variant generator, source and visibility, family; a converter for what exists |
| Attempt | the Home agent and the coordinator's walk to a build (21.2, 28.3), the engine's stages, the team (28.x), self-play (27.3), twin runs (25.x: the system beside Claude) | running a pack end to end without its answer key, the whole trajectory kept |
| Judge | acceptance decisions per kind; the journeys' four parts; the benchmark's 108 cases and its AND gate; the decision ledger with labels (19.6 to 19.8) | one grader scoring cards, decision points and traps against any rubric, with the honest score range |
| Diagnose | the ledger's decision records and labels, lineage (19.5), failure signals (19.9), the critic and auditor verdicts (28.4, 28.5) | the first-wrong-step finder and the gap classifier |
| Learn | lessons proposed after a project and accepted by a reviewer (`dclab_rnd/lessons.py`); project memory; the evidence index and retrieval; pattern cards and the business-examples base (30.5 to 30.8); prompts as files, hash-gated (`dclab_core/prompts`); the recipe registry with certification levels (30.10) and modifiers (30.11); the decision models and the students (22.x, 29.x); the lake and the SFT corpora (26.1, 26.9); the tuned-model walk (27.8) | routing each lesson to its level with its evidence; case memory retrieved into the roles; playbook deltas; the code-skill pull request path |
| Verify and promote | sealed sets read once, gates fixed before runs, certification levels (experimental, certified, product default), `make kinds-check` | variant families in three sets, the regression curriculum |
| Scoreboard | `dclab_rnd/kinds/capability.json`, the evaluation board (25.8), report cards (29.14, planned) | pass rates per family on check and sealed variants, N challenges, the learning curve |

## Part D. Design

**D1. The Challenge Pack** (`dclab_core.challenge`, a frozen versioned contract with its JSON schema).

| Part | Holds | Seen by the agent |
|---|---|---|
| `request` | the problem in a business person's words, like a real chat message | yes |
| `data` | a replica generator and its seed, or a pinned dataset (path and SHA-256) | the data only |
| `answer_key` | the task kind and modifiers, each trap and the right reaction to it, the expected answer at each decision point (the ledger's points), the honest score range | never |
| `rubric` | how each card and the result are graded, points per item | never during an attempt |
| `variant_generator` | code that makes similar problems: another domain skin, other rates, traps moved or weakened, columns renamed | never |
| `source` | interview or pilot pattern, public dataset, invented, a production failure; its visibility | the general pattern only |
| `family` | the pattern it belongs to ("duplicate records", "rare positives", "review capacity") | no, to avoid hinting |

Packs live in `challenges/<family>/<id>/`; the answer key and rubric in a part the attempt path never imports (a test
reads the attempt's imports and file opens), pinned by SHA-256 as the sealed journeys are. A production failure (a gate
that failed, a person's correction) becomes a pack: the loop feeds itself, and every real mistake becomes a test DCLab
must pass from then on.

**D2. The grader** (`dclab_core.grader`, deterministic): a trajectory (its cards, the ledger's decision records, the
run's records) against the answer key and the rubric; a score per card, per trap and per decision point; a pass or a
fail; a holdout above the honest range is flagged as a sign of leakage, never as a success. The grader is attacked
before it is trusted: planted "gaming" trajectories (a right final number with wrong decisions, a decision right for the
wrong reason) must fail. RL (E) waits until those attacks fail every time.

**D3. Variants in three sets.** Per family, a generator changes the surface and keeps the core: a domain skin (the same
duplicate-records trap as insurance claims, telecom accounts, hospital records), numbers (a positive rate, a size,
repeats per group), names (columns renamed, so `bill_id` cannot be spotted), difficulty (traps weaker or partly hidden).
The **learning** set is for diagnosis and lessons; the **check** set decides promotion and is never used to write a
lesson; the **sealed** set is generated by an independent process (another seed partition, and domain skins written by
an agent that never read the rules or the lessons), scored once per promotion.

**D4. Diagnosis: the first wrong step.** A wrong metric in phase 4 is often a modifier missed in phase 2. The grader names
the failing traps and decisions mechanically; the finder walks the trajectory's decisions in order (the ledger's points
and their lineage) to the earliest one whose answer differs from the key; a model proposes the gap type; a person
confirms it for rule and capability gaps (they lead to code).

| Gap | Looks like | The lesson goes to | Level |
|---|---|---|---|
| knowledge | the question that mattered was never asked (review capacity) | a pattern card or a checklist question | 2 |
| reasoning | the facts were there, the conclusion was wrong | a critique and corrected samples for SFT | 4 |
| decision | one typed decision was wrong (`split.strategy`) | a labelled sample for that point's student | 4 |
| rule | no check stopped a known trap (a feature computed with future data) | a new check or gate rule | 3 |
| capability | the pipeline cannot do what the problem needs (pair matching) | a new recipe part or kind | 3 |
| data | the data cannot answer the question | nothing: the right answer is "does not fit"; the pack is relabelled | — |

Every case also goes to level 1, whatever its gap.

**D5. Lessons, each to its level, each with its evidence** (`dclab_core.lesson`, `Lesson` v1: level, family, the gap it
answers, the evidence ids, the change, its version, its status: proposed, verified, promoted, retired).
- **Level 1, case memory (A).** A judged case (its cards, the gap, the lesson, the score) joins the case library;
  retrieved for a new problem by the problem card and the data card (bands and roles, never values), measured as an
  arm: memory on against memory off, on check variants. Extends `lessons.py` and the workspace lessons, which already
  stay out of `evidence/knowledge`.
- **Level 2, the playbook (B).** The agents' procedure is their prompts (files, hash-gated), the pattern cards and the
  checklists. Lessons arrive as **small edits** (ACE's generate, reflect, curate: add a line, sharpen a question, never a
  full rewrite); a prompt may also evolve by reflection on its graded failures (GEPA). Each edit is versioned, cites its
  evidence and passes `make agent-eval`, `make harness-eval` and the whole curriculum before it serves; a person reads
  the wording. Run on the local model, so the many trials cost nothing.
- **Level 3, skills as code (C).** For a rule or capability gap, a coding agent drafts the recipe part, the check or the
  modifier as real code in its own worktree, with the variants it must pass; it opens a pull request; certification runs
  it on the check variants and the curriculum, then on the sealed variants; a person approves the merge. The recipe
  registry's levels say where it stands (experimental, certified, product default).
- **Level 4, models (D, then E).** Judged trajectories and decisions are exported in batches into the lake's families
  (SFT for the solver, decision samples for the students), under consent; the LoRA run on Qwen ~8B is the owner's,
  measured on the sealed exam, and walks onto a role through 27.8 (shadow first, served only where it measured better,
  retired in one change). RL from the grader's reward only after D, only once the grader resists every planted attack,
  and only with the owner's approval.

**D6. The regression curriculum and the scoreboard.** Every retired challenge and its sealed variants stay in a growing
suite; every promotion re-runs it whole, so a new lesson cannot quietly break an old capability. The scoreboard (beside
`capability.json`) keeps, per family: the pass rate on check variants, on sealed variants, N challenges that taught it,
the last promotion, and the learning curve (sealed score against N). It is the honest answer to "what can DCLab solve
today?", and later something a customer can see.

**D7. The outside exam.** MLE-bench (75 Kaggle competitions) as an external sealed exam says whether DCLab's learning
carries over beyond its own replicas. It needs downloads, a Kaggle key and machine time: the owner's decision, a subset
first.

**D8. Capture.** The loop's own records are training data: each attempt's trajectory, the grader's verdict per item, the
diagnosis with its first wrong step and gap, each lesson with its evidence and its verification, each promotion. New lake
families beside `case_trajectories` (30.19): `challenge_attempts`, `diagnoses`, `lessons`, held out by challenge and by
family so a family's sealed variants never train its own models.

## Part E. The packages (31.x), in stages

Each package follows the house discipline: test first, the reviewer, its development record, `make check-quick` (or
`check-all` at a stage's end), capture in its "Done when", gates fixed before any measured run.

| Stage | Package | Delivers | Done when |
|---|---|---|---|
| **0. Can learning be measured at all?** (about 2 weeks) | 31.1 The Challenge Pack | `dclab_core.challenge` v1; the converter for the existing replicas (delivery fraud, retail returns and the kinds' cases) and the interview patterns; the answer key's isolation | 10 packs from what exists; a test proves the attempt path never reads an answer key, a rubric or a variant set |
| | 31.2 The grader | `dclab_core.grader`: cards, decision points and traps against a rubric; the honest range; planted gaming trajectories | every existing replica's acceptance decisions graded the same as today; every planted gaming trajectory fails |
| | 31.3 Variants for one family | the generator for duplicate records and group leakage (skins, numbers, names, difficulty), three sets kept apart, the sealed one written independently | each variant keeps its trap (its tests bite on several seeds), no single column gives it away, the sets never share a seed |
| | 31.4 The curriculum and the scoreboard | the regression curriculum runner (`make curriculum`), the scoreboard per family, the baseline: today's DCLab on check and sealed variants with nothing learned | the baseline stored with its intervals: the zero point of every learning curve |
| **1. A + B** | 31.5 Diagnose and route | the first-wrong-step finder, the gap classifier (rules first, a model proposes, a person confirms code gaps), `Lesson` v1 | on planted misses the first wrong step is found; every lesson cites its evidence; capture of every diagnosis |
| | 31.6 Case memory | the case library and its retrieval into the roles' evidence (the critic, the router, the planner), by card bands, never values | an arm, memory on against off, on the check variants of the family, gate fixed before; served only where better |
| | 31.7 The evolving playbook | delta edits to pattern cards, checklists and prompts (ACE), reflective prompt evolution (GEPA) on the local model, each edit versioned with its evidence | the check variants' score rises against 31.4's baseline with the curriculum unbroken; `agent-eval` and `harness-eval` pass; one edit rolled back in one change |
| **2. C** | 31.8 Skills as code | the coding agent's pull request path: a recipe part, a check or a modifier drafted in a worktree with its variants; certification; the owner approves | one capability learned this way passes its sealed variants; the curriculum unbroken |
| | 31.9 A second family | the loop on a family it has not seen (rare positives with review capacity, or near-duplicate matching from 30.16) | the learning curve of the second family stored: does the loop generalise to new families, not only new variants |
| **3. D** | 31.10 The learning export | judged trajectories and decisions into the lake's families; the solver's SFT set and the students' samples; the owner's LoRA run on Qwen ~8B | the tuned model, through 27.8 in shadow, measured on the sealed variants; served only where it beat the prompt-and-memory path |
| **4. E (optional)** | 31.11 RL from the grader | reinforcement learning on the grader's reward | only after 31.10, only once the grader resists every planted attack, only with the owner's approval |
| outside | 31.12 The outside exam | a subset of MLE-bench as an external sealed exam | the owner's approval for the downloads; scored once per promotion of the whole loop |

The first proof is stage 0 alone: if the grader and the variants cannot show today's score honestly, nothing after it
can be measured. Stage 1 is where most of the value is expected, at almost no model cost.

## Part F. Decisions for the owner

| Decision | Needed by | Proposal |
|---|---|---|
| The first family | 31.3 | duplicate records and group leakage: the retail returns replica, the groups modifier of 30.11 and the copied-bills trap of 30.9 already exist |
| Who writes the sealed variants | 31.3 | an independent agent that never read the rules or the lessons (as the sealed journeys); a person later |
| The local model for A and B | 31.6, 31.7 | Qwen ~8B (the solver's base) through the gateway's local route; Claude only for the steps that need it, capped |
| Budget per challenge | 31.4 onward | a cap per attempt and per verification, set before each stage; reported after |
| Who approves code skills | 31.8 | the owner, as a pull request review |
| MLE-bench downloads and compute | 31.12 | the owner's choice, a subset first |
| RL at all | 31.11 | not before 31.10's result; the owner's decision |

---

## خلاصهٔ فارسی

**هدف فاز ۹:** هر چالش تازه‌ای که به DCLab داده می‌شود، توانایی حل همهٔ مسئله‌های مشابه آن خانواده را بیشتر کند، نه فقط
همان یک مسئله. معیار اصلی: نمره روی نسخه‌های مهروموم‌شده پس از یادگیری از N چالش یک خانواده؛ اگر با N کوچک بالا برود، عامل
واقعاً یاد گرفته است.

**وضعیت:** فقط برنامه. به دستور مالک، تا پایان فازهای ۶، ۷ و ۸ هیچ بخشی از فاز ۹ ساخته نمی‌شود و ثبت و ذخیرهٔ همهٔ داده‌ها
ادامه دارد.

**چهار سطح یادگیری و پنج رویکرد:** ۱. حافظه (A: کتابخانهٔ موارد حل‌شده، ارزان و سریع)؛ ۲. دانش (B: دستورالعمل و پرامپت‌ها
با ویرایش‌های کوچک از روی خطاهای نمره‌گذاری‌شده، به سبک ACE و GEPA)؛ ۳. توانایی در کد (C: یک عامل کدنویس قانون، بررسی یا
اصلاح‌کنندهٔ تازه را به‌صورت درخواست ادغام پیشنهاد می‌دهد و پس از گواهی روی نسخه‌های تازه، انسان تأیید می‌کند)؛ ۴. مدل (D:
تنظیم دقیق LoRA روی Qwen حدود ۸ میلیارد و مدل‌های تصمیم، و در آخر E: یادگیری تقویتی از نمرهٔ داور، فقط وقتی داور فریب‌پذیر
نباشد). سطوح ارزان اول، چون قابل بررسی و قابل بازگشت‌اند.

**گلوگاه اصلی داور است، نه مدل:** بستهٔ چالش با کلید پاسخ پنهان و معیار نمره‌دهی؛ مولد نسخه‌های مشابه در سه مجموعهٔ جدا
(یادگیری، بررسی، مهروموم‌شده)؛ و داوری خودکار که تصمیم‌ها (تقسیم داده، نشت اطلاعات، معیار) را نمره می‌دهد، نه فقط نتیجهٔ
نهایی را.

**مراحل ساخت:** مرحلهٔ ۰ (حدود دو هفته): قالب بستهٔ چالش، داور، مولد نسخه‌ها برای خانوادهٔ «رکوردهای تکراری و نشت گروهی»
و صفر منحنی یادگیری. مرحلهٔ ۱: حافظهٔ موارد و دستورالعمل تکامل‌یابنده. مرحلهٔ ۲: مهارت‌ها به‌صورت کد. مرحلهٔ ۳: تنظیم دقیق.
مرحلهٔ ۴ (اختیاری): یادگیری تقویتی. آزمون بیرونی: بخشی از MLE-bench با تأیید مالک.

**قوانین:** عامل هرگز کلید پاسخ را نمی‌بیند؛ هیچ قانونی برای قبول‌شدن در یک چالش سست نمی‌شود؛ هر تغییر باید همهٔ آزمون‌های
بازگشتی را بگذراند؛ هر درس شاهد خود را دارد و قابل بازگشت است؛ دادهٔ خصوصی فقط الگوی کلی را آموزش می‌دهد؛ هزینهٔ هر چالش
سقف دارد.

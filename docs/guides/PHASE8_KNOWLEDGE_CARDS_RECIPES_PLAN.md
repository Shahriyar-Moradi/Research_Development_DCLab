# Phase 8, Part C: real business knowledge, the four cards, and recipes (design)

Written 2026-10-18 from the owner's words: *let's build phase 8 together; I want better and real business examples with a
lot of knowledge our agents can use; before implementing anything, let's design the architecture and the plan.* The
sources are four documents the owner wrote after a friendly interview with the AI lead of a large retail group in the
Gulf (October 2026): the cleaned interview, **ML problem patterns** (eleven problems, each generalised), **the DCLab
framework** (four phases, four cards, four gates, a twelve-step workflow) and **task recipes** (one kind plus modifiers,
a nine-field contract, certification). This plan reads them against the repository at `b56fe5d`.

**Privacy.** The repository is private (the owner, 2026-10-18), so the sources can be kept in it. Two rules still hold:
nothing from the owner's fraud data ever reaches a model (not a value, not a column name), and what a model reads from
the knowledge base is the general pattern, never a person's name or a personal detail; the source record keeps who said
what, for the owner.

**Why it matters beyond the features:** every part of this phase records, clean and structured, what DCLab's three
models learn from (Part D): the solver language model, the decision models, and the decision models inside the
solver's workflow. Learning to capture this well is also the service DCLab will offer other companies.

---

## Part A. What the four documents ask for

| Idea | From | In one line |
|---|---|---|
| Eleven problem patterns | patterns | fraud with no labels; labels from nothing; two-stage screening; near-duplicates and linked records; models drifting in production; "why did this number change"; a secure internal LLM platform; department agents; the simplest tool that works; one data foundation; delivering many projects |
| Practice questions | patterns | ten questions, each answered well only by naming the metric, the split and the first baseline before any model |
| Planted traps for problems 1 to 5 | patterns | a rule computed with future data, duplicate bills split across train and test, a drift period in the holdout |
| Four cards and four gates | framework | Problem Card (G1 problem clear), Data Card with a fit verdict (G2 data fits), Solution Brief approved before training (G3, the main human checkpoint), Workflow Run (G4 honest and usable); a failed gate goes back to a named phase with a reason |
| Twelve fixed steps | framework | lock the split, clean, baseline, features, candidates, evaluate, select, calibrate and set the rule, holdout once, package, report, plan monitoring; a check before each next step |
| Who does what | framework | the LLM understands (phases 1 and 3), rules execute (phases 2 and 4), people own business risk; with the LLM off, phases 2 and 4 give the same result |
| Recipes | recipes | one task kind plus modifiers; every part fills nine fields; parts are versioned and certified (experimental, certified, product default) on trap replicas and two real datasets; a resolved recipe keeps the reason behind every choice |
| Artifact per kind | recipes | what a saved model must carry for each kind, and a save, reload, predict and match test for every kind |

## Part B. What DCLab already has (read 2026-10-18)

About two thirds of the framework and the recipes exist under other names. What is new is mostly the joining, the
Solution Brief, the modifiers, the monitoring step and the knowledge the agents read.

| Piece | Today | The gap |
|---|---|---|
| Problem Card | `ProblemBrief` and `ProblemCard` (`dclab_core/contracts.py`, `card.py`): goal, action, unit, target, moment, horizon, label source, constraints, success criteria as text, open questions | no decision owner, no typed costs of a false alarm and a miss, no business metric, no **baseline today**, no "who can answer"; costs, capacity and success never block; the studio's brief drops constraints and success criteria |
| Data Card | `DataCard` v3 (`datacard.py`), findings (`analyse.py`), `meet.check` | no **detected modifiers**, no column "known at prediction time", target and time profiles are findings not card fields, the fit verdict is yes or no (no "fits with changes"); leak suspicions on the card are computed over the whole table |
| Solution Brief | spread over `SolutionPlan` (decisions with reasons and alternatives, risks), `SplitDesign`, `Route`, the studio's `Solution`; `approve_gate` signs only the `Solution`, and sign-off is **off by default** | no brief, no **approach level** (rule, classical model, small model, LLM), no **success bar fixed before training** and judged after, baselines appear only after the holdout |
| Gates and loop-backs | the coordinator's states, `meet`, the graph's gates (sign-off off by default), holdout counting | a failed gate never names the phase it sends work back to; nothing gates on the train-to-holdout gap |
| Twelve steps | steps 1, 4, 5, 6, 9 and 11 exist; 2, 3, 7, 8 and 10 in part | no fitted calibrator (only measured), the export's tests are written but never run, **no monitoring plan at all** (step 12) |
| Recipes | 15 kinds (`dclab_rnd/kinds/`), the plan's decisions with reasons (`dclab_core/plan.py`), `capability.json` | no modifiers as parts, no version per kind, two states instead of three levels, certified on replicas only, three different "rare" cut-offs (1, 5, 10%), capacity never becomes the primary metric |
| Patterns 1 to 4 (fraud) | anomaly detection with a review loop, partial labels (PU bagging), rings through shared values, programmes of leaves composed on validation | **no retail returns replica**, no label source per row, no confirmed-only evaluation, no random audit sample, no case summary, **no matching kind** (fuzzy duplicates) |
| Pattern 5 (drift) | a training-versus-holdout PSI smoke check | no drift baselines or alert thresholds in the artifact |
| Pattern 6 ("why did it change") | nothing | a metric-decomposition kind would be new |
| Patterns 7, 8, 9, 11 | the gateway, consent, redaction, the validator, routing levels (shadow, flag, act) are DCLab's own answers to 7 and 8 | 9 (the simplest tool) is not in the router; 11 is outside the product |
| Knowledge base | 38 examples (`evidence/business_examples/`), a search without a model | **no agent reads it**, it is not indexed, no retrieval questions, no path into training; the owner's fraud example is in it (decision D1) |

## Part C. The architecture

One pipeline, four cards around it, a recipe inside it, knowledge beside it, and capture under all of it. Nothing here
is a new pipeline: the cards are contracts over what the studio already decides and records.

```
 words, files ──► 1 PROBLEM CARD ──G1──► 2 DATA CARD + fit ──G2──► 3 SOLUTION BRIEF ──G3──► 4 WORKFLOW RUN ──G4──► report
                  (conversation,          (profile, findings,        (recipe resolved:        (the engine's stages
                   card questions)         detected modifiers)        kind + modifiers,        as twelve steps,
                        ▲                        ▲                    success bar, rule,       a check per step,
                        │   failed gate: back to a named phase       approval)                monitoring plan)
                        └────────────────────────┴───────── monitoring: drift and real outcomes come back ◄──┘

 KNOWLEDGE beside every phase: pattern cards, business examples, practice questions (read by roles through one tool,
 served only where measured better)
 CAPTURE under every phase: each card, choice and check as a record for the three models (Part D)
```

**Layer 1. Knowledge.** `BusinessExample` v2 (dclab_core): v1 plus a pattern block (general problem, why it is hard, how
to solve, how to measure, common mistakes, the same problem in other industries), practice questions with what a good
answer names, links to a recipe (kind and modifiers) and to replicas, and a **visibility** (`model` or `rules_only`).
Every example is committed; a `rules_only` one (the owner's fraud solution) is read by the search and shown to the owner,
never put in a model's context. One read tool, `find_examples`, for the Home agent, the intern and the teams; the critic and the router get the
closest examples' pitfalls in their evidence. Served only behind a measured switch (`examples_passed()`).

**Layer 2. Understanding cards.** Problem Card v2: the missing fields (decision and its owner, typed error costs,
capacity K as a number per period, business and ML success metrics, the baseline today, horizon, open questions with
who can answer). Only what changes the solution blocks G1; the rest becomes an open question with an owner, as the
framework says. Data Card v4: detected modifiers with their evidence, per-column "known at prediction time", target and
time profiles, and a three-way fit verdict (fits; fits with changes, each change a proposed card edit; does not fit, with
the data missing). G2 reads it.

**Layer 3. The Solution Brief (new contract).** Filled by rules from the two cards, the resolved recipe and the plan's
decisions; its words drafted by a model and checked by `cited.py`; approved by a person. Fields as the framework lists
them, with the **approach level** ladder (rule or script, then OCR and templates, then a classical model, then a small
tuned model, then a frontier model, stopping at the first that can meet the bar: pattern 9). The approval signs a hash
of the whole brief (not only the `Solution`); the success bar is frozen there and the report states whether the holdout
met it. G3 becomes the main checkpoint (decision D3).

**Layer 4. Recipes.** `dclab_core/recipes.py`: `RecipePart` (a kind or a modifier, the nine fields, a version, a level),
`ResolvedRecipe` (choices plus the part and reason behind each). The fifteen kinds become parts through an adapter (no
behaviour change); the modifiers are new parts: rare positives (one cut-off, replacing the three), groups, time order,
noisy labels, review capacity, text columns, drift expected, no target. Precedence as the recipes document says: splits
combine; the most specific metric wins (capacity, then rare, then the kind); a conflict the rules cannot settle asks the
person. `plan.py` reads the resolved recipe, so every decision cites the part that made it. Certification extends
`capability.json` to three levels with their evidence: the replicas, two real datasets, the same result with the LLM
off, and a twin comparison.

**Layer 5. The Workflow Run.** `dclab_core/workflow_run.py`: the twelve steps as a fixed contract (input, output, check,
status, evidence path), mapped onto the engine's stages. What is missing gets built: a dummy baseline in the models
screen, a calibrator fitted on out-of-fold predictions, the export's save, reload, predict and match test run at export
for every kind, and step 12: drift baselines per feature, the score distribution, PSI and flagged-share alert thresholds
written into the artifact, with a `monitor(new_rows)` check. A failed gate writes a `GateResult` (gate, passed, back to
which phase, why); the coordinator follows it.

**New capabilities the patterns need:**
- **Matching** (pattern 4): a sixteenth kind; one row is a pair; exact links first (the same bill returned twice, a
  return by the cashier who sold), blocking, similarity features, nearest neighbours; split by entity; pair precision
  and recall; the artifact carries the blocking keys, features and threshold.
- **The screening cascade** (pattern 3): a programme template: a wide net (anomaly and rules), link checks (matching), a
  case summary for reviewers (a model's words, advisory, never a label; off for private projects), a review queue sized
  by capacity K with a random audit sample in every batch, confirmed labels back into a supervised leaf; recall measured
  per stage on confirmed cases. The anomaly review loop and programmes become one path.
- **Labels from nothing** (pattern 2): a label source per row (rule, expert, cluster), evaluation on confirmed labels
  only, sources weighted by their agreement with experts, the share of confirmed fraud no rule flagged, a learning curve
  per 1,000 labels, the uncertain cases plus a random sample sent for review.

**Who does what** stays the repository's rule: rules own splits, metrics, thresholds and every check; a model reads
words and drafts text; the person approves the brief and owns the business risk.

## Part D. What every part captures, and which model learns from it

| Record | Captured where | Model 1: the solver LLM | Model 2: decision models | Model 3: decisions in the workflow |
|---|---|---|---|---|
| A case trajectory: the request, the questions and answers, the Problem Card, the Data Card and its verdict, the brief with reasons and alternatives, the twelve steps with their checks, the report against the bar | a new lake family `case_trajectories` with its `DatasetManifest`, per project, under train consent; whole projects held out | the main training example: problem → understanding → solution → workflow, with the reasoning built from the rules' recorded reasons and the person's choices; gate failures as critique examples ("this brief has no success bar: what is missing?") | | |
| Each modifier detected or not, the approach level, the fit verdict, the primary metric, the decision rule | ledger points `modifier.<name>` (eight), `brief.approach_level`, `data.fit_verdict`; states as bands | | a student per point: labels by construction (replicas), by people (the brief's edits at approval), by outcome (the bar met or not) | the students answer in shadow where the resolver decides; the time each answer takes is recorded; served only after the sealed exam |
| Pattern cards and practice questions | `evidence/business_examples/` (each with its visibility) | knowledge examples; the practice questions as an evaluation set with a rubric (metric, split, baseline named) | | |
| Replicas with planted traps | `dclab_core/replicas/` | trajectories on invented data | homework with known answers for every modifier | |
| Gate results and loop-backs | `GateResult` events | when to go back, and where | `workflow.revisit` labels | |

Claude's answers stay evaluation material; the rules' and people's are the labels (silver labels from Claude only on real
projects, tagged and kept apart). Everything is per workspace and consent-gated, so another company's workspace would
capture the same way.

## Part E. The packages (30.7 to 30.20)

Each package: the contract and rule in `dclab_core`, the wiring in `dclab_rnd`, a `CORE_INTEGRATION.md` section, its
capture with a test that reads it back, the reviewer, its development record and `make sft-dev`.

| # | Package | Delivers | Done when |
|---|---|---|---|
| 30.7 | What a model may read | `visibility` on examples; the owner's fraud example `rules_only`; the three source documents in `docs/sources/2026-10_retail_interview/` (not the raw transcript, decision D7); a model's context built from the general pattern only | a test: no `rules_only` example and no person's name reaches a model's request (the gateway's recorded prompts) |
| 30.8 | Pattern cards | `BusinessExample` v2; the eleven patterns, their cross-industry cases and the ten practice questions as examples, each citing its source document | every pattern links a recipe or says it is knowledge only; the search finds the right pattern for each practice question |
| 30.9 | The retail returns replica | invented sales and returns tables with patterns 1 to 4's traps: copied bills (near-duplicates across train and test), the late-evening rule as a feature and as the source of rule labels, a rule computed with future data, a few dishonest cashiers and new ones later, a label source per row, drift in the last three months | its tests prove every trap bites on several seeds and no single column gives the fraud away (the journeys people move to 30.11) |
| 30.10 | The recipe registry | `RecipePart`, `ResolvedRecipe`; the fifteen kinds as parts, versioned, through an adapter | every run's choices name their part; results unchanged on every replica (a test) |
| 30.11 | Three modifiers | rare positives (one cut-off), groups, time order; detection with thresholds, precedence, ledger points; the retail replica as a kind's case | certified on the retail replica and two real datasets (Telco, HyperAck); the LLM-off result is the same; the retail replica's dev people written and a sealed set by an independent agent |
| 30.12 | Labels and capacity, and the first certification | noisy labels (label source, confirmed-only evaluation, weights), review capacity (precision at K primary, threshold at the K-th score, value times probability), a random audit sample; the retail replica as a kind's case; the three modifiers and the two new ones certified on it and two real datasets (from 30.11) | the replica's rule-label trap is caught; recall is estimated from the audit sample with its interval |
| 30.13 | Problem Card v2 and the fit verdict | the missing fields; G1 on what changes the solution; Data Card v4 with detected modifiers and the three-way verdict; G2 | the dev journeys pass with the new fields; a "fits with changes" proposes its card edit |
| 30.14 | The Solution Brief and G3 | the contract, the approach-level ladder, the success bar frozen and signed, the report judged against it | a brief without a bar, a baseline or an alternative is refused; the holdout's verdict against the bar is in the report |
| 30.15 | The Workflow Run and G4 | the twelve steps with their checks, the calibrator, the export test run per kind, the monitoring plan in the artifact, `GateResult` loop-backs | `make code-check` and the export test pass for every kind; `monitor()` flags a planted drift |
| 30.16 | The matching kind | pairs, blocking, exact links, similarity, nearest neighbours | certified on the copied-bills trap and one real dataset |
| 30.17 | The screening cascade | the programme template, the case summary (advisory), the review queue with capacity and audit, labels back into the loop | recall per stage on the replica; the case summary never changes a label (a test) |
| 30.18 | Knowledge the roles read | `find_examples`, examples in the critic's and router's evidence, retrieval questions v2 | an arm of the journeys measured on dev, then one fresh sealed set; served only if better |
| 30.19 | The case trajectories, and the real runs | the `case_trajectories` lake family; the owner's fraud (private, rules only) with its resolved recipe against RPB-007 and the owner's own solution; Telco and HyperAck certifying binary and the modifiers | the corpus builder refuses below its minimum; every run's comparison stored with its limits |
| 30.20 | Why did this number change (pattern 6) | a `metric_change` kind: a semantic layer (approved metric definitions, their factors and joins), the change decomposed into its factors (sales = footfall × conversion × basket: each factor's share of the change in logs, with a bootstrap interval over days) and drilled down by dimension (mix against rate), a model choosing among tested query templates and filling their parameters (free SQL only behind validation), the query, source and date range under every answer, and a suggested action; a replica with planted causes (a footfall drop in one place, a conversion drop in one category, a data gap the question cannot be answered from) | every planted cause found and ranked first; a question the data cannot answer is refused with what is missing; the same answer with the model off when the question names its template |

**Order.** 30.7 first (what a model may read comes before any role reads), then 30.8 and 30.9 (the knowledge and the test bed everything else is
judged on), then the recipes (30.10 to 30.12), the cards (30.13 to 30.15), the new capabilities (30.16, 30.17, 30.20), and the
knowledge roles and real runs last (30.18, 30.19); then Part G.

**Budget.** Almost all of it runs on rules. Model spend: the knowledge arm in 30.18 (about €1 to €2), case summaries on
the replica (under €0.5); within the $17 left for phases 7 and 8, each run planned, printed and capped.

## Part F. Decisions for the owner

| # | Decision | Recommended |
|---|---|---|
| D1 | ~~What may go in the repository~~ | **decided (2026-10-18):** the repository is private; everything is committed; the fraud data never reaches a model, and a model reads patterns, not names |
| D2 | Which patterns become capabilities now? | **decided:** patterns 1 to 6 and 9 in this phase (6 as package 30.20); then 7, 8, 10 and 11 as features (Part G) |
| D3 | Is the brief's approval the one main checkpoint, covering the single holdout read? | **decided:** yes; sign-off on by default, the separate holdout approval goes, the report says whether the frozen bar was met |
| D4 | Do detected modifiers apply on their own? | **decided:** yes above their threshold, each shown with its reason; review capacity is always asked |
| D5 | Is review capacity K set per project or per workspace? | per project, with a workspace default (the recommendation, standing until the owner says otherwise) |
| D6 | Which two real datasets certify each part? | Telco and HyperAck for binary and the first three modifiers; the owner's fraud (private) for capacity and labels (the recommendation, standing) |
| D7 | Are the four source documents copied into the repository? | **decided:** the patterns, framework and recipes documents in `docs/sources/2026-10_retail_interview/`; the raw transcript stays in the owner's own folder |

## Part G. After this phase: patterns 7, 8, 10 and 11 as features (the owner, D2)

Designed in detail once Part E is built; each keeps DCLab's rules (the validator, the gateway, consent, capture).

| Pattern | The feature | What DCLab already has to build on |
|---|---|---|
| 7. A secure internal LLM platform | a workspace chat over its own files where numbers come from code run in a sandbox, never from the model's head; personal data masked before any model; cost and use per team; every question and answer audited | the gateway (`dclab_rnd/models`), redaction (`dclab_core.redact`), limits, the audit trail, the intern's code tools |
| 8. Department agents | a template for an agent that acts on company data: read-only first, narrow typed tools, hard limits in code (largest price change, budget, approval above an amount), every step logged and undoable, automatic only for low-risk choices it has got right many times | the agent harness and registry, the graph validator, routing levels (shadow, flag, act with a check, act), the ledger |
| 10. One data foundation | a catalog of tables with owners and quality checks, shared keys, entities resolved across systems (the matching kind), history kept as it was at each date so models never train on the future | the data cards, connectors, `entities.py` (cutoffs), the matching kind (30.16) |
| 11. Delivering many projects | request triage on Home: value, data readiness, effort and risk scored; the success number agreed before building (the brief's bar); projects nobody uses retired | the Home agent, the Solution Brief (30.14), outcomes (`dclab_core/outcomes.py`) |

## Status

- **Design (2026-10-18):** written and decided with the owner (D1 to D7).
- **30.7 built (2026-10-18), the review's 8 findings fixed:** `BusinessExample` v2 (dclab-core 0.75.0) with `visibility`
  (`model` or `rules_only`), the owner's pattern fields (pattern, practice questions, recipe link, replicas) in the same
  version for 30.8 to fill, `for_model` and `context_for_model` the one way examples reach a model (no storage path, no
  conversation, no rules_only example); an owner's solution can never be visible and a stored v1 one reads as rules_only;
  practice questions are kept out of the index they evaluate; a rebuild without the owner's folder keeps the committed owner
  example. Left: the check on the gateway's recorded prompts comes with 30.18, when a role first reads examples.
- **30.9 built (2026-10-18), rebuilt after the review's 9 findings:** `dclab_core.replicas.retail_returns` (dclab-core 0.76.0): sales and
  returns over nine months, staff return fraud on copies of real bills (new near-duplicate bills, the amount edited a little,
  sometimes from another store) among honest repeat purchases, labels that grow from the late-evening rule to a team reviewing
  five returns a day with a random part, a cashier statistic over the whole log, the review verdict, new dishonest cashiers and
  earlier hours in the last three months. Every trap bites on seeds 0 to 2 and no returns column alone reaches 0.85 of ROC-AUC
  (0.74 to 0.78) while the cross-table link does better (0.84 to 0.91). Left for 30.11: the replica through the engine, its
  journeys people and a sealed set.
- **30.8 built (2026-10-18), the review's 8 findings fixed:** the owner's eleven problem patterns as business examples
  (`dclab_rnd/knowledge/retail_interview_patterns.json`, no person's name, brand or internal tool: a hashed list checks it), each
  with its recipe link (built, planned in a named package, or knowledge), the cross-industry table whole for patterns 1, 4, 6 and 8,
  and the ten practice questions with the owner's rubric (the metric, the validation split, the first simple baseline);
  `dclab_core.business_examples.practice_recall` (0.77.0): 10 of 10 questions find their own pattern first among 49 examples, the
  questions never indexed. Left: paraphrases by an independent writer (30.18), the easy test being one author's.
- **30.10 built (2026-10-18), the review's 10 findings fixed:** `dclab_core.recipes` (dclab-core 0.78.0: `RecipePart` v1 with the
  nine fields, a version and a level; `ResolvedRecipe` v1; `resolve` with the owner's precedence, a kind's time order never
  dropped, ties and splits that do not combine asked, what the person states checked); `dclab_rnd/recipes.py`: the fifteen kinds
  as parts (`python -m dclab_rnd.recipes`), each versioned by its own code, all experimental until certified; every run records its
  recipe in the data stage (a machine kind's too), and the split and metric decisions carry their part and reason. The recipe's
  split and metric equal the run's on binary, regression, places, rare_event and segmentation. Left: the modifiers (30.11).
- **30.11 built (2026-10-18), the review's 10 findings fixed:** `dclab_core.modifiers` (dclab-core 0.79.0): rare positives (the
  positive class under 5%; one cut-off for the plan, the analysis and the reasoning), groups (the coarsest repeating key with
  enough values; with a time column a key that recurs over time is asked, never grouped into a split the engine refuses), time
  order (a time column; a moving target strengthens it). The proposal reads them on every row it reads, applies those above their
  threshold and asks where the rules cannot settle; the wizard shows each with its reason; Home's build keeps the group; every run
  names the modifier behind its split and metric; three ledger points record each against what the saved solution applies.
  Every kind's acceptance unchanged; the dev journeys 15 of 15 (JRN-020). Moved to 30.12: the certification on the retail replica
  and two real datasets, the replica as a kind's case and its journeys people (the replica needs the noisy labels to be solved).
- **30.12 built (2026-10-19), the review's 10 findings fixed; Done when met, no part certified.** `dclab_core.capacity`
  (dclab-core 0.88.0): a queue of K a period, precision at K on confirmed labels (each reviewed row weighed by its day's
  and stratum's N/n), recall estimated only from the rows reviewed at random within their stratum and day (the interval
  the wider of a bootstrap over days and a Wilson interval on the weighted cases' effective count), the rule behind a
  noisy source and its agreement. Two modifiers, `noisy_labels` (a label-source column with a noisy source beside twenty
  confirmed rows: a noisy source's rows neither trained nor judged on, the rare test reading the audit's rate) and
  `review_capacity` (asked on every binary problem, applied when K is stated), with their ledger points (63 in all). The
  Solution carries `label_source_column`, `confirmed_sources`, `capacity_k` and `capacity_period`; the proposal finds a
  label source by its values and forbids it unless applied; Home's build and the wizard carry it. The retail replica is
  the rare-event kind's second case (`rare_event.retail_case`, in `make kinds-check` only): on seeds 0 to 2 the trap is
  caught (noisy labels applied, judged on confirmed labels only, `rule_flag` flagged as the rule behind the rule's labels
  with an agreement of 0.17 to 0.18, the verdict and the whole-log statistic kept out), and the queue's recall from the
  audit alone is 1.0 with intervals from 0.64 to 1.0, covering the true recall (0.88 to 1.0) on every seed.
  **CRT-001** (`python -m dclab_rnd.certify`): the five modifiers judged on the recipes document's five criteria on the
  replica's three seeds, Telco and HyperAck, detection judged on what the proposal read before any statement: no
  criterion failed, every part stays experimental, the artifact (30.15) and a comparison with a model on pending; only runs
  where a part applies count, so the new modifiers still need real datasets where they apply (the owner's fraud data, D6).
  Left: noisy rows as weighted training labels, value times probability, precision at K as the selection metric inside
  training, K stored from Home and the wizard, the operating threshold set on training rows (with the artifact), the
  replica's journeys people and its sealed set.
- **30.13 built (2026-10-19), the review's 13 findings fixed; Done when met.** `dclab_core.card.ProblemCard` v2 (dclab-core 0.89.0, v1 read as v2): the
  decision and its owner, the error costs (`costs_said`: a miss's cost, a false alarm's, which is worse when one sentence
  compares them), the capacity (`capacity_said`: "about 80 a day" is 80 a day), the success numbers, today's baseline and the
  horizon, each from the person's answers (Home's budget answer is the capacity: among several numbers the one beside a
  word of acting, never a year or one out of range; a reading that fails leaves the field open) or an open field with who
  can answer it; none blocks, so G1 is the card's own blocking set (`complete`). `DataCard` v4 (v1 to v3 read as v4): each
  column's known at prediction time (no for a leak suspect or a label source, yes for a key, else unknown, a time
  included) and the recipe modifiers the rows show, read on exactly the rows the proposal may read
  (`modifiers.provisional_reading` and `facts_of`, moved into the library); a build stores its verdict in the card.
  `meet.fit` and `FitVerdict` v1: G2's three verdicts, fits, fits with changes (each change a card edit with why: the
  outcome another column (another outcome-like column when a named one is refused), the unit one row by the table's key,
  the family the column's, a rare event when known cases are here, the time column, judged on confirmed labels for noisy
  labels) or does not fit (the file of known cases, a column of dates, a column that records the outcome named).
  `GET /api/drafts/{id}/card` returns it; `POST /api/drafts/{id}/fit/accept` writes the accepted edits as the person's
  answers (refusing an edit the wizard's saved solution owns, and undoing one that would come back), records the person's
  decision beside the rule's and emits an event Home redraws on; a build records `data.fit_verdict` (64 points) from what
  the verdict read. The dev journeys 15 of 15 with the new fields (JRN-026, and JRN-027 after the review). Left: the
  Home card showing the verdict and its accept button, G2 as a coordinator step (today the mismatch questions block as
  before), the main product's spec page.
- **30.14 built (2026-10-19), the review's 12 findings fixed; Done when met.** `dclab_core.solution_brief` (dclab-core 0.90.0):
  `SolutionBrief` v1 with the approach level on the ladder, the recipe, the baselines (today's process with its number when
  known, the recipe's simple model), the validation, the success bar (absolute, or a factor or margin against a named
  baseline, with guards), the decision rule, the features kept out, the risks and the alternatives. G3 (`problems`) refuses
  a brief without a bar, a baseline or an alternative, and a bar that could never be judged or goes against its metric's
  direction; `sign` freezes the bar on a fingerprint; `judge` reads the holdout once: met, not met (which guard failed is
  said) or cannot be judged, with why. The studio drafts a brief by rules for every kind (a machine's metric and its own
  first baseline; recall at the review budget against reviewing that share at random), saves it through the graph's
  `set_brief` move (a person only, never while a stage runs, never after the holdout read), signs it with the solution gate
  (never after the read), judges it in both final paths and writes the verdict into the report; D3: a signed brief covers
  an agent's holdout approval while the brief and the solution are both signed and unchanged (never a programme's).
  `GET`/`PUT /api/projects/{id}/brief`; the audit kind `brief` and the event `decision.accepted` (point `solution.brief`).
  Harness on the rules 17 of 17 (HRN-023). Left: a brief exists only when a person saves one (the Solution page's editor,
  sign-off on by default and a model drafting the words are the owner's), the main product's Objective has no bar yet.
- **30.15 built (2026-10-19), the review's 12 findings fixed; Done when met.** `dclab_core.workflow_run` (dclab-core 0.91.0):
  `WorkflowRun` v1, the framework's twelve steps read from a run's records (a step the records do not show is missing),
  `GateResult` v1 (passed, or the phase the work goes back to, with why) and `g4`; `dclab_core.calibration` (a calibrator
  chosen and fitted on the out-of-fold scores only); `dclab_core.monitoring` (`MonitoringPlan`, `monitor`). The engine
  scores the simple model on the screen's own folds, fits the calibrator and builds the monitoring plan on the model's
  input columns at the final stage, writes G4 in both final paths and records it as the ledger point `workflow.g4`; the
  solution team stops on a failed G4 with the phase it names (`back_to`). The service export carries the calibrator
  (`calibrated_probability` beside the score) and `monitoring.json` (levels hashed with the export key, edges between
  training values) and checks itself: reloaded in an isolated Python, the refit's rows predicted again and matched, or
  refused; the code export writes `monitoring.json` and, with `check`, reruns its pipeline; both are capture moves tied to
  the final run they package. Ranking, segmentation and uplift now export code: `make code-check` reproduced every one of
  the 15 kinds' holdout numbers in a clean Python 3.12; the binary, multiclass and regression replicas' services reloaded
  and matched (the binary one failed first: the scorer had served integer columns as text); `monitor()` flags a planted
  drift and, on the delivery replica's holdout, a real one (cancelled rides only after the training rows). `GET
  /api/projects/{id}/workflow-run`. Left: kinds with their own machine have no service export; G4 fails on the
  time-to-event, uplift and embedded replicas (honest on these seeds, not pinned per kind); the main product's mapping.
- **30.16 built (2026-10-19), the review's 12 findings fixed:** `dclab_core.matching` (dclab-core 0.92.0; one table or
  two): the split fixed by entity before any pair is formed (the known matches join records; every pair inside an entity
  is a match; a repeated id or more than 5% of the matches unfound is refused), each part's records paired with what they
  may see (the holdout with every record, as a new one would be), blocking by key and time window or by text neighbours
  both ways with its recall said, features per field (text, number, time, category, flag), the exact rule on the key
  fields, the single best field, logistic and boosting fitted on training pairs and chosen on validation pairs with each
  threshold searched over every distinct score, the holdout read once with the F1's interval and its difference to both
  baselines by an entity bootstrap. The sixteenth kind `matching` with its own machine (G4 passes), its code export (the
  holdout reproduced), the router's `pairs_table` precondition (a related table of two identifiers and little else) and
  approach, `certify.run_kind`, the track `research/record-linkage`. The owner allowed two public benchmarks (DBLP-ACM,
  Abt-Buy; gitignored). Certified (CRT-002, run from committed code): the copied bills, with the plan's exact link (a
  return by the cashier who sold the bill), pair F1 0.76, 0.70 and 0.78 on three seeds (the exact rule about 0.01);
  DBLP-ACM 0.983 (one field 0.921); Abt-Buy 0.610 (one field 0.550); every seed and dataset passes every criterion, each
  win's 90% interval above zero. Left: the kind's own dev and sealed journeys, a service export.
- **30.17 built (2026-10-19), the review's 8 findings fixed:** `dclab_core.cascade` (dclab-core 0.93.0) screens a stream of
  cases day by day as a team of capacity K would, every stage fitted on the days before: a wide net (a rule, the link checks
  the matching kind names, the day's most unusual 15% of rows, and once learned the model's highest 15%), a queue of K with
  a random audit sample, the confirmed labels refitting a model every thirty days; recall per stage from the audit (the
  net's and the queue's) and, on a replica, from its truth after the run. `capacity.audit_recall` now bounds what the
  unreviewed rows may hold (an audit with no case among the rows left had read as recall one). A column known only later
  is refused as anything read; the case summary sees only what screening reads, is never asked for a private project,
  never changes a rank or a label, and is kept whole with why. CSC-001 (three seeds): the net keeps half the rows and 0.91
  to 0.99 of the cases; the queue catches 0.76, 0.86, 0.91 (0.72, 0.82, 0.80 with the learning off); every audit interval
  covers the truth. Left: the cascade as a programme template the studio runs on a project (with the anomaly review
  loop as one path), a model's case summary through the gateway (no purpose yet), a replica whose team reviews a smaller
  share.
- **30.20 built (2026-10-19), the review's 17 findings fixed:** `dclab_core.metric_change` (dclab-core 0.94.0; `SemanticLayer`
  v1, `MetricAnswer` v1): a question about a number answered from approved definitions through four tested templates, each
  query one plan rendered as bound SQL and as its pandas computation (the core never reaches a database; the studio runs
  the SQL on SQLite); a why split in logs into footfall x conversion x basket with intervals over days, drilled down into
  mix against rate, with an action; gaps checked at the grain both tables share and refused with what is missing; sums
  compared per day; one 10% false-move budget for factors and segments. A model only chooses and fills a template
  (purpose `metric_question`, validated, its reply kept whole); a question naming its template is answered with no model.
  The kind `metric_change` runs as a library (Home refuses to build it), certified by its acceptance (KND-022, 27 of 27 on
  three seeds); the replica `mall_sales`. Measured: the rules reach 477 of 480 planted answers over sixty seeds; the
  model reads 24 of 24 paraphrases against the rules' 6 (MCH-003, live, EUR 0.10). Left: an independent writer's
  paraphrases, a project path (a layer kept per workspace, the agent tool).
- **30.18 built (2026-10-19):** `find_examples` for the intern and /mcp (examples a model may read, never the owner's
  solution); the decomposer, the router and the engine's critic read the closest examples' lines as advice only when the
  host gives them (`DCLAB_EXAMPLES` on and the `roles_examples` arm passed its sealed journeys, gate fixed before; or a
  measurement asks); a test reads the gateway's recorded prompts and finds no private example (30.7's open check). The
  search on an independent writer's wording of the practice questions: 16 of 30 (10 of 10 in the author's own). Dev,
  live: roles 3 of 6, roles with examples 2 of 6 (EUR 0.56): not better, the switch stays off; both arms miss the same
  review-budget part, which goes to 30.6. Left: the sealed arm on 30.6's fresh set, the Home agent's tool, a better search.
- **Test runs kept (2026-10-19, d502540):** every run of the suite and every gate, whole, in `training_records/test_runs/`
  (the owner's rule), snapshotted by `make archive`.
- **Live tests (2026-10-19, $20 budget; EUR 4.02 spent):** the product's model (Claude) through the harness suite for the
  first time: 10 of 17 (HRN-022; the rules alone 17 of 17). Fixed: a costs question asked again and again and blocking the
  build; a fraud walk built without its entity (no grouped folds); the model reading "the day's orders" as the entity and
  writing the card's unit itself; questions replied to left open or repeated; a budget kept in the model's paraphrase; the
  suite keeping no records of its plays; the code gate reading a live measurement. After the fixes: 14 of 17 (HRN-024);
  its last two causes (a contradiction closed unread, a card answer never recorded) fixed after it. Next: the team arms,
  the real problems (Telco, HyperAck) and the scripted judgment suite live.

---

## خلاصهٔ فارسی

این طرح، چهار سندی را که از مصاحبه نوشتید با کد فعلی DCLab مقایسه می‌کند. حدود دو سوم چارچوب و دستورالعمل‌ها
(recipe) با نام‌های دیگر وجود دارد. کارهای تازه این‌هاست:

- **Solution Brief:** خلاصهٔ راه‌حل که پیش از آموزش تأیید می‌شود و معیار موفقیتش از قبل ثابت شده است.
- **اصلاح‌گرها (modifier):** نمونهٔ مثبت نادر، گروه‌ها، ترتیب زمانی، برچسب نویزدار، ظرفیت بررسی، متن، drift، و نبود برچسب.
- **پایش مدل:** گام دوازدهم گردش‌کار.
- **نوع تازهٔ تطبیق رکوردها:** برای قبض‌های کپی‌شده.
- **غربالگری دو مرحله‌ای.**
- **مجموعهٔ دادهٔ ساختگی تقلب مرجوعی خرده‌فروشی:** با تله‌های کاشته‌شده.
- **دانشی که عامل‌ها می‌خوانند:** فقط جایی به کار گرفته می‌شود که سنجیده و بهتر شده باشد.

هر بخش داده‌اش را تمیز و ساختاریافته برای سه مدل DCLab ثبت می‌کند:

۱. مدل زبانی حل‌کننده
۲. مدل‌های تصمیم
۳. مدل‌های تصمیم داخل گردش‌کار

مخزن اکنون خصوصی است و همه‌چیز در آن ذخیره می‌شود. با این حال هیچ مدلی دادهٔ تقلب شما را نمی‌بیند، و آنچه مدل از
پایگاه دانش می‌خواند الگوی کلی است، نه نام افراد. تصمیم‌ها گرفته شد و ساخت از بستهٔ 30.7 شروع می‌شود؛ الگوی ۶ («چرا این عدد تغییر کرد؟») هم در همین فاز ساخته می‌شود.

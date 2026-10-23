# DCLab R&D Chat: Full Recap

**Covers:** this chat only, from the first message (R&D data cleanup) to the latest one (this recap)
**Projects:** `Research_Development_DCLab` (the R&D lab) and `DCLab` (the product, also called Decision.ai)
**Built from:** the saved session transcript, with key numbers re-checked against the cloned repo
**Date:** 3 October 2026

---

## Contents

1. [The short version](#1-the-short-version)
2. [Timeline at a glance](#2-timeline-at-a-glance)
3. [Phase 1: Understanding the R&D repo and cleaning the data](#3-phase-1-understanding-the-rd-repo-and-cleaning-the-data)
4. [Phase 2: Filling the gaps (datasets, SFT data, plain-language doc)](#4-phase-2-filling-the-gaps)
5. [Phase 3: How to use the data: RAG, fine-tuning, tools](#5-phase-3-how-to-use-the-data-rag-fine-tuning-tools)
6. [Phase 4: Wiring R&D into the real DCLab product](#6-phase-4-wiring-rd-into-the-real-dclab-product)
7. [Phase 5: R&D as a standing function, and verifying the agentic loop](#7-phase-5-rd-as-a-standing-function-and-verifying-the-agentic-loop)
8. [Phase 6: The notebook copilot (prototype and live test)](#8-phase-6-the-notebook-copilot)
9. [Phase 7: UI/UX design iterations](#9-phase-7-uiux-design-iterations)
10. [Phase 8: Building a focused agentic model (fine-tuning ladder, action tokens)](#10-phase-8-building-a-focused-agentic-model)
11. [Phase 9: Token and cost efficiency: NOOA, Jev, the "spreadsheet" architecture](#11-phase-9-token-and-cost-efficiency)
12. [Phase 10: GNNs, speculative decoding, Jev benchmarking](#12-phase-10-gnns-speculative-decoding-jev-benchmarking)
13. [Phase 11: Transcript attempt and the training glossary](#13-phase-11-transcript-attempt-and-the-training-glossary)
14. [Phase 12: Other domains: computer vision, self-driving, maps, scene graphs](#14-phase-12-other-domains)
15. [Phase 13: Moving into this coding session](#15-phase-13-moving-into-this-coding-session)
16. [Every file produced](#16-every-file-produced)
17. [Key numbers (verified)](#17-key-numbers-verified)
18. [Decisions and principles that came out of this chat](#18-decisions-and-principles)
19. [Corrections made along the way](#19-corrections-made-along-the-way)
20. [Open items and suggested next steps](#20-open-items-and-suggested-next-steps)
21. [خلاصهٔ فارسی](#21-خلاصهٔ-فارسی)

---

## 1. The short version

You came in asking how to turn the R&D lab's outputs into cleaner data for fine-tuning a small language model (SLM), and how to run 50 evidence-first ML experiments with LLM help at every step.

The first finding was that **most of that already existed**: the `model_building_50_v1` campaign had run 50/50 experiments (10 real UCI datasets × 5 stages), on top of a 579-experiment historical registry distilled into 22 rules and 10 workflow blocks. So the chat became about four things:

1. **Turning the evidence into usable data.** SFT datasets (three versions, each fixing a real flaw), a retrieval index, an MCP knowledge server.
2. **Feeding it into the real product** without breaking DCLab's own rule: deterministic pipeline first, narrow LLM checkpoints, no parallel agent framework.
3. **Designing what users see.** A notebook copilot that comments beside each cell and cites real proof, plus several UI iterations, ending with a published "DCLab Studio" page.
4. **Pushing the architecture.** A focused agentic model, action tokens, NOOA/Jev-style efficiency, a dependency-graph ("spreadsheet") design, speculative decoding, GNNs, and how the method transfers to computer vision, self-driving and maps.

The thread running through all of it: **every claim has to be checkable.** That rule caught several of my own mistakes during the chat (section 19).

---

## 2. Timeline at a glance

| # | Your request (paraphrased) | What came back |
|---:|---|---|
| 1 | Clean the R&D data for SLM fine-tuning; run 50 experiments; write an outsider-readable review with rules | Repo audit, `DCLab_RD_Review_and_Roadmap.md`, `build_sft_dataset.py` (v1), 104 sample examples |
| 2 | Fill the gaps: Kaggle/HF datasets, multiclass/time-series/fraud/text, paired instruction data, SFT job, plain-language doc; don't remove anything | `dataset_expansion_pack.py`, `build_sft_dataset_v2.py`, `SFT_DATA_GUIDE.md`, `DCLab_Field_Notes_Plain_Language.md` |
| 3 | "The model doesn't know what EXP-023 *is*" | `build_sft_dataset_v3.py`: every example now carries dataset card + stage card + setup summary |
| 4 | How to use this with RAG/fine-tuning/agents? What should RAG data look like? | `build_rag_index.py`, `AGENT_ARCHITECTURE_RAG_TOOLS_FINETUNE.md` |
| 5 | Use it today without fine-tuning or RAFT | `grounded_review.py`, `dclab_knowledge_mcp_server.py`, a 5-step practical ladder |
| 6 | Implement it: use every detail in DCLab | Read the real `DCLab` repo; `DCLab_Product_Integration_STEPS.md` |
| 7 | R&D should be an ongoing part of DCLab, now and next phase | `DCLab_RD_Operating_Model.md` |
| 8 | Agentic Colab/notebooks plus R&D to verify it works | `DCLab_Agentic_Loop_RD_Verification_Plan.md` |
| 9 | An agent that comments beside every cell with tested proof ("StackOverflow for ML") | `notebook_copilot.py`, `NOTEBOOK_COPILOT_SPEC.md`, demo notebook |
| 10 | Design the UI/UX (several rounds, with reference images) | 5 inline mockups and the published `dclab_studio.html` |
| 11 | Why LightGBM over XGBoost/RF? → "ok continue" | Real EXP-004 numbers, then a live 5-model screen where **logistic regression won** |
| 12 | Fine-tune a model (Qwen 8B/27B) to *be* agentic; bigger-than-subword tokens | 4-rung fine-tuning ladder; action/macro tokens with code |
| 13 | Merge two steps in one generation ("cake" example) | Data-level step fusion vs. architecture-level MTP |
| 14 | Critique the whole idea; expand to other industries? | Three corrections (determinism, expansion order, compression limits) |
| 15 | Fewer tokens, like NVIDIA NOOA | NOOA mechanics; found an unbounded `_evidence_json()` in the product |
| 16 | TypeSafe Jev; think outside the box | System 1/System 2 split; the "spreadsheet/build-system" dependency-graph architecture |
| 17 | Draw the architecture (and remember I'm a 7-year ML engineer who wants simple explanations) | Memory updated; two architecture diagrams |
| 18 | How can GNNs help? | Not for the 5-node workflow; maybe for a knowledge graph later |
| 19 | Speculative decoding: MTP, EAGLE, DFlash, DSpark | Speed lever, not quality lever; structured output is the best case |
| 20 | Jev after fine-tuning? Jev as a classifier? Benchmark it | Parallel, not sequential; `jev_classifier.py` + `run_jev_screen.py` |
| 21 | NOOA vs. LangGraph conflict | Reference-only state; 3-tier cost ladder (Jev → NOOA → full LLM) |
| 22 | Compile the whole chat verbatim | Started, caught my own mix-up, pointed to the official export, restarted (not finished) |
| 23 | Explain 100 training terms | `ML_Training_Terminology_Glossary.md` (106 terms, bilingual) |
| 24 | Computer vision? Self-driving? | CV transfers directly; self-driving perception yes, control no |
| 25 | Map/road flow data | Google Maps/DeepMind GNN as the real positive case; Supersegments |
| 26 | Agentic LLM with graph structure; graph attention for CV | Three meanings; tool calls over the graph is the one to build |
| 27 | Scene graphs: agentic VLM understanding object flow | 5-step pipeline from complete graph → pruned → scene graph → video → agentic VLM |
| 28 | How do GNNs work, simply? What's the end result? | Node/edge/graph-level outputs; for scene graphs: "graph in, labeled graph out" |
| 29 | "Continue in code" → session moved to Claude Code | Checked the empty workspace; cloned `Research_Development_DCLab` |
| 30 | Recap everything as a Markdown file | This document |

---

## 3. Phase 1: Understanding the R&D repo and cleaning the data

### What you asked
Turn the lab's information into cleaner, simpler data for fine-tuning an SLM on data science and ML engineering. Run 50 experiments on real datasets covering EDA, leakage detection, feature engineering, model choice and optimization, with an LLM helping at every step. Collect the workflow knowledge (not just code lines, but working blocks and flows) for future LLM training. Write a review an outsider could read, with intuition, recommendations and "rules and laws."

### What the repo already had
- **`model_building_50_v1`: 50/50 complete.** 10 real UCI datasets × 5 stages: data understanding → leakage audit → feature engineering → model selection → optimization/reliability.
- The LLM acts only as **critic and hypothesis generator**. It never touches the data directly.
- A historical registry of **579 experiments across 11 datasets**, distilled into **22 rules** (`model_building_rules.jsonl`) and **10 workflow blocks** (`workflow_blocks.json`).

### The three real gaps found
1. Datasets were all UCI and all **binary classification**: no multiclass, time-series, imbalanced-fraud or text+tabular cases.
2. The knowledge files were shaped for **agent memory/retrieval**, not paired instruction → answer examples an SFT job can train on.
3. There was no plain-language narrative for an outside reader.

### Delivered
- **`DCLab_RD_Review_and_Roadmap.md`**: methodology with intuition, real leakage examples with numbers (`duration` in bank_marketing, `PageValues` in online_shoppers), the 22 rules as a plain checklist, and 10 proposed Kaggle/HuggingFace datasets that each stress a different failure mode.
- **`build_sft_dataset.py` (v1)**: converts rules, workflow blocks and claims into chat-format Q&A, strips provenance noise (git hashes, machine info), dedupes, splits 90/10. **104 clean examples** from a partial slice.

**Limitation flagged:** that chat sandbox couldn't reach kaggle.com or huggingface.co, so running new experiments was left to your local `dclab_rnd campaign run`.

---

## 4. Phase 2: Filling the gaps

### What you asked
Add Kaggle/HuggingFace data and the missing task types, explain paired instruction data and how to run an SFT job on it, add a separate plain-language doc. **Add only; don't remove anything.** The goal: a correct, automated R&D process that produces the best DCLab MVP.

### Delivered
- **`dataset_expansion_pack.py`**: adapters for 4 Kaggle datasets, each converted into the repo's `external_data/<name>/{X.parquet, y.parquet, meta.json}` format:

  | Dataset | Fills gap |
  |---|---|
  | `credit_card_fraud` | Heavy class imbalance |
  | `otto_group_products` | Multiclass |
  | `rossmann_store_sales` | Time series |
  | `ecommerce_clothing_reviews` | Text + tabular |

  Smoke-tested on small hand-built samples matching each dataset's published schema. Column names still need checking against the real downloads.
- **`build_sft_dataset_v2.py`**: pulls the real 5-part LLM reasoning trace (observed evidence → interpretation → decision → risks/missing evidence → next experiment) from each experiment's `llm_review`. This matches the answer format the repo's own `AGENT_CONTEXT.md` requires. 6/6 real experiments produced full traces.
- **`SFT_DATA_GUIDE.md`**: the three common JSONL shapes (OpenAI chat, Alpaca, ShareGPT) and a runnable TRL + LoRA script for Qwen2.5-3B-Instruct.
- **`DCLab_Field_Notes_Plain_Language.md`**: a story-style walkthrough for a reader with no ML background.

### Your critique, and the fix (v3)
You pasted a real generated example: *"What did experiment EXP-023 on 'credit_default' find during feature engineering?"* It never said what the dataset predicts, what the stage does, or what was compared. A model trained on that learns **"ID → fact" memorization, not reasoning.** You were right.

**`build_sft_dataset_v3.py`** adds three blocks to every example's user turn, all from real files:
1. **Dataset card**: what the data is and predicts (`external_data/<name>/meta.json`)
2. **Stage card**: what the stage does methodologically (`workflow_blocks.json`)
3. **Setup summary**: what was actually compared (from the experiment's `evidence`), e.g. *"6 feature recipes on LightGBM, 3-fold CV: raw (14 features, ROC-AUC 0.8955), ratios (37 features, 0.8944)…"*

Rerun on EXP-003 (adult, feature engineering), the critic's own review even caught a real inconsistency: the "selected" recipe scored below "raw" despite the stated selection rule. v1/v2 were kept; v3 became the default for experiment-grounded examples.

---

## 5. Phase 3: How to use the data: RAG, fine-tuning, tools

### The core answer
It isn't RAG *or* fine-tuning *or* agents. Each does a different job:

| Technique | Job | Fits DCLab for… |
|---|---|---|
| **Fine-tuning** | How the model thinks by default | The reasoning format and the 22 stable rules |
| **RAG** | Fresh facts that change often | The growing experiment registry, exact numbers |
| **Tools / agentic execution** | Actually doing the steps | Running pipeline stages; only tools can act |

This matches the product's existing principle: **deterministic pipeline first, narrow evidence-bound LLM checkpoints, full autonomy rejected.**

### What RAG data should look like
Don't chunk it like a PDF. The rules and claims are already structured records (category, dataset, stage, confidence). Fixed-size chunking would split a rule's "statement" from its "why." The right design is **filter first** (dataset = X, stage = Y; exact and free), **rank second** (similarity, only among what survives).

- **`build_rag_index.py`**: TF-IDF proof of concept (swap in real embeddings later; the structure stays the same). Loaded **82 records** (22 rules, 10 workflow blocks, 50 claim groups). A query filtered to `bank_marketing` returned the exact EXP-007 `duration` leakage finding.
- **RAFT** (retrieval-augmented fine-tuning) was named. v3's context blocks are already RAFT-style data.
- **`AGENT_ARCHITECTURE_RAG_TOOLS_FINETUNE.md`**: comparison table and build order.

### "Use it today, without fine-tuning" ladder

| Step | Effort | What |
|---:|---|---|
| 0 | ~10 minutes | Call `index.search(query, filters=...)` before building any LLM prompt |
| 1 | This week | **`grounded_review.py`**: retrieval + prompt in the `AGENT_CONTEXT.md` format; dry-run or `--live` |
| 2 | Next | **`dclab_knowledge_mcp_server.py`**: `get_rule`, `get_workflow_block`, `retrieve_similar_experiment` as MCP tools (tested directly against real data) |
| 3 | After | Let LangGraph checkpoint nodes call those tools mid-reasoning; the LLM still never touches data or has the final word |
| 4 | Later, optional | Fine-tuning/RAFT, only once retrieval genuinely can't fix something |

---

## 6. Phase 4: Wiring R&D into the real DCLab product

I read the actual `DCLab` repo instead of guessing. Two lines in `DCLAB_MASTER_IMPLEMENTATION_SPEC.md` changed the plan:

> "Future autonomous-agent/critic/experiment-loop work: **explicitly paused** until the deterministic workflow is fully verified."
> "Reuse the existing LLM client, evidence architecture, validator pattern… **do not build a parallel agent framework.**"

So the MCP server stays a separate research tool, and the job became **feeding the existing checkpoint**, not replacing it.

### What already existed in the product
- `apps/api/app/engine/lab/evidence.py`: `LeakageReviewEvidence` dataclass (`suspicious_name_tokens`, `single_feature_score`, `datetime_after_fraction`, `availability_status`, …)
- `apps/api/app/engine/lab/llm_client.py`: `request_leakage_review()`, versioned prompt `leakage_review_v1.py`, strict JSON-schema output
- A built-in hallucination guard: *"Do not mention, assume, or cite anything that is not present [in the evidence]."*

### `DCLab_Product_Integration_STEPS.md`: the 7-step plan
1. Package the R&D knowledge files as **static, versioned data** inside DCLab (no live dependency on the R&D repo).
2. A small, dependency-free retrieval helper `rd_knowledge.py` that **fails closed** (returns `[]`, never raises), so the pipeline still works with the KB off.
3. One new optional field on `LeakageReviewEvidence`: `prior_registry_evidence`.
4. A new prompt **`leakage_review_v2.py`** (never edit v1), **plus** adding `"prior_registry_evidence"` to the `LeakageEvidenceField` `Literal`. The schema is `strict: True`, so the model cannot cite a field missing from that enum, whatever the prompt says. This is the easy-to-miss part.
5. One line at the construction call site to fill the new field.
6. **Persist what was cited** in the decision ledger, and export human-confirmed production decisions back into `agent_memory.jsonl` format. That is what makes the knowledge base grow.
7. Verification in the spec's own style: positive, citation, fail-closed and regression cases.

Then repeat the same recipe for `column_type_v1.py`, `missing_value_v1.py`, `target_selection_v1.py`. Leakage goes first because it overlaps most with the 22 rules.

---

## 7. Phase 5: R&D as a standing function, and verifying the agentic loop

### `DCLab_RD_Operating_Model.md`
**Reframe:** R&D isn't a knowledge dump. It is where the product's claims are **earned before they ship**. It tests which parts of the big vision (feature intelligence, model factory, diversity-aware ensembles) hold up on real data, then turns them into defaults, prompts and guardrails.

- **The loop has to close in both directions.** R&D → product is built. Product → R&D (export confirmed/corrected decisions back as new evidence) is what makes it a standing function.
- **Phase Now:** put findings into the pipeline's **defaults**, not only the LLM's context, so they hold with the LLM disabled. Examples: always screen a linear baseline; engineered features must beat raw by a margin (raw won 7/10 datasets); importance is a nomination, not a verdict.
- **Phase Next:** certify the 4 new task types before marketing them, close the feedback loop, and only then consider fine-tuning on product-validated evidence.
- Four yes/no questions to ask each quarter to check the loop really closed.

### `DCLab_Agentic_Loop_RD_Verification_Plan.md`
You shared the "Cursor for ML" vision (agentic Colab/notebooks). Six of its seven ideas were already backed by existing evidence. **The one new and risky piece is the autonomous loop** (train → analyze error → adjust → retrain). The registry's biggest score jumps (+0.13 to +0.18 ROC-AUC) came from **leakage**, not skill, and a loop without leakage discipline would ship that mistake faster, with a plausible-sounding reason attached.

How to verify it:
1. **Replay** it on your 10 datasets without showing it the campaign's answers. It should exclude the same leaky columns and land near the same holdout scores.
2. **Red-team** it with the deliberately leaky dataset versions.
3. **Constrain its moves** to what's already validated (search ranges, the pre-screened model shortlist, one rung on the feature ladder).
4. **Log every iteration** as an experiment record, so the audit trail and the feedback loop are the same plumbing.

Build it as chained instances of the existing leakage-review pattern (bounded evidence → schema-checked decision → deterministic validator → ledger), behind a deterministic controller that decides when to stop.

---

## 8. Phase 6: The notebook copilot

### What you asked
An agent that writes comments and suggestions beside every notebook cell, helps make models faster and more efficient, and backs every suggestion with **tested proof**, like seeing on StackOverflow that someone else hit the same issue and solved it.

### Delivered: a working prototype
**`notebook_copilot.py`** reads a real `.ipynb`, runs 4 structural detectors, and cites a real rule plus a real precedent for each finding:

| Detector | Example caught |
|---|---|
| Fit-before-split leakage | `scaler.fit(...)` before `train_test_split(...)` → DCLAB-R04 + EXP-002 |
| Known-leaky column names | `cancellation_date` |
| Single model, no comparison | Cites "no universal best model" |
| Missing `random_state` | Reproducibility |

Honest limitation: TF-IDF sometimes picked a real but weakly related precedent (the `random_state` case). Real embeddings fix that.

**`NOTEBOOK_COPILOT_SPEC.md`** covers two surfaces sharing one retrieval (inline comment with "Show proof", and a chat mode), and an honest look at the **"50 Kaggle challenges solved"** framing. What exists today is 50 experiments on 10 UCI datasets. That's strong, but it's a different sentence. Running the Kaggle expansion pack through certification would make the literal claim true.

### The LightGBM question, and the live test
A mockup showed "LightGBM selected" with invented numbers. When you asked why, I corrected it with real EXP-004 data (adult dataset):

| Model | Mean ROC-AUC | Std | Runtime |
|---|---:|---:|---:|
| logistic_regression | 0.847 | 0.0047 | 0.22s |
| extra_trees | 0.889 | 0.0076 | 1.48s |
| hist_gradient_boosting | 0.8953 | 0.0089 | 0.53s |
| **lightgbm** | **0.8955** | **0.0065** | **0.33s** |
| xgboost | 0.892 | 0.0083 | 0.29s |

LightGBM won by **0.0002**, effectively a tie, decided by the rule *"mean ROC-AUC − 0.25 × fold std, then runtime"* (DCLAB-R14). Random forest was never in that screen.

You said "ok continue," so I ran a real 5-model screen on a 5,000-row synthetic churn dataset (`run_real_churn_screen.py`):

| Model | Mean AUC | Std | Seconds | Rank score |
|---|---:|---:|---:|---:|
| **logistic_regression** | **0.7757** | 0.0130 | 0.048 | **0.7725** |
| hist_gradient_boosting | 0.7331 | 0.0121 | 0.770 | 0.7301 |
| lightgbm | 0.7313 | 0.0142 | 0.349 | 0.7277 |
| extra_trees | 0.7162 | 0.0165 | 4.505 | 0.7121 |
| xgboost | 0.7150 | 0.0168 | 0.375 | 0.7108 |

**Logistic regression won by a wide margin.** That's DCLAB-R13 (no universal best model) in action. The result was saved as `new_experiment_record.json` in the registry's claim format. **Design lesson:** the UI must separate "recommended from registry evidence" from "verified on your data."

---

## 9. Phase 7: UI/UX design iterations

| Round | Prompt | Result | Key choices |
|---:|---|---|---|
| 1 | Design the UI/UX | Inline card mockup | Warning sits under the exact line; "Show proof" = one rule + one precedent; Apply fix / Dismiss / Ask follow-up |
| 2 | More creative, better graphics (marketplace reference) | Numbered step-flow mockup | Scan → findings → model comparison → bounded tuning; green only means "selected/complete" |
| 3 | Improve on 2 reference images | `notebook_copilot_v2` | Called out the references' "Great choice!" hype, invented numbers, and a "Looks good!" banner shown next to 2 leakage flags; used real numbers; current model shown as "not yet screened" |
| 4 | A whole agentic Colab page | "dclab notebook" page | Own identity; two zones (cells + agent panel); optimization trail in context |
| 5 | A completely different visual direction | Pipeline view | The page *is* the ML graph (dataset → deploy); in-progress steps look unfinished; a command bar instead of a chat panel |
| 6 | Use attached images to build a real HTML page | **`dclab_studio.html`**, published as an Artifact | Sidebar, status band that says "1 finding blocks the rest of the review," inline cell + finding + fix, "verified on this data" comparison with real numbers, floating command bar |

The published page was at `https://claude.ai/artifact/B2nKwiC3sawqm31rpcFJnz` (from the earlier session; I can't re-check that link from here).

---

## 10. Phase 8: Building a focused agentic model

### Making the model itself agentic
Moving from **prompted tool use** to **policy internalization**, as a 4-rung ladder:

| Rung | Method | Status |
|---:|---|---|
| 1 | Declarative SFT (facts and rules) | Built: `sft_train_v3.jsonl` |
| 2 | **Trajectory SFT**: all 5 stages of one experiment as one continuous example | Repackaging of existing data |
| 3 | **Rejection sampling (STaR/ReST)**: keep only candidates that pass `decision_validator.py` | The validator already exists and acts as a free reward function |
| 4 | **RL (GRPO/PPO/DPO)**: reward = validator pass + real held-out metric change | GRPO is relatively cheap at 8B–27B |

Model note: Qwen 3.8 27B was reported by third-party trackers (unconfirmed from Alibaba's own post at the time). Iterate on an **8B-class** model first.

### Bigger-than-subword tokens
Called **action/macro tokens**, with precedent in ToolkenGPT, Decision Transformer/Gato/RT-2 and CTRL. How to do it:
- Add tokens like `<STAGE:leakage_audit>` (around 10 stage tokens; group the 22 rules into about 6–8 category tokens, since each rule alone has too few examples).
- `tokenizer.add_special_tokens()` + `model.resize_token_embeddings()`.
- **Warm-start** each new embedding from the mean embedding of its text description.
- **LoRA gotcha:** add `embed_tokens` and `lm_head` to `target_modules`, or the new tokens never learn.
- Pair with **constrained decoding**: mask logits so only actions valid in the current state can be generated. That's a real guarantee; fine-tuning alone only makes errors rarer.

### Merging steps (the "cake" example)
- **Data-level fusion (build now):** write steps that must never be split as one training target. Example: "split, then fit the scaler on train only, then transform both" as one step, so "fit before split" (DCLAB-R04) is never a separable sequence. The same applies to fit-imputer + transform, k-fold + aggregate mean/std, and similar pairs.
- **Architecture-level (later):** Multi-Token Prediction (DeepSeek-V3; reportedly in Qwen 3.8), Byte Latent Transformer/MEGABYTE, latent reasoning (Coconut).

### Your request for critique: three corrections
1. **"The model will be deterministic by itself" overstates it.** Fine-tuning makes correct behavior *likely*. Constrained decoding plus the validator make wrong structure *impossible*. You need both.
2. **Other industries: right instinct, wrong order.** The hard part is the **validator**, not the model. ML is ideal because correctness is mechanically checkable. The natural next domain is software engineering (tests pass or fail).
3. **Token compression has a floor.** You can remove redundancy between steps, but not new information (a number produced by step 1 still has to appear). Test it on real data before promising 2×.

Also: the workflow agent should **build** the predictive model, not **become** it. LightGBM, logistic regression or TabPFN stay whatever the evidence picks.

**Smallest first test:** 2–3 stages with a clear ordering rule, fused trajectory data, LoRA on 8B, then check (a) ordering violations on unseen scenarios and (b) citations of rule IDs that don't exist.

---

## 11. Phase 9: Token and cost efficiency

### NVIDIA NOOA (already in your stack: LangGraph + NOOA + OpenAI Responses API)
Published result: **82.2% at about 1.1M tokens / 28 calls vs. 78.2% at 2.2M tokens / 66 calls.** Four mechanisms:
1. **Pass by reference:** tool outputs stay as live objects; the model sees a short typed preview.
2. **Fewer calls:** loops and branches live in code, not in "what's next?" calls.
3. **Typed contracts:** bad outputs get caught and retried locally.
4. **Bounded memory:** periodic pruning and merging.

**Gap found in the product:** `_evidence_json()` in `llm_client.py` serializes the full evidence (`json.dumps(asdict(evidence))`). A `_bounded_preview()` patch was proposed, which matters most for `related_column_names` and the future `prior_registry_evidence`.

Order of work: bounded previews now → design the optimization loop as one small typed call over a fixed menu → merge near-duplicate claims in `agent_memory.jsonl` → put stable content first in prompts for prefix caching when self-hosting.

### TypeSafe AI's Jev ("System One model")
Non-generative, parallel answers to typed questions (**Choice / Score / Noul**). Reported 70–500 ms and far lower cost. It was 6 days old with self-tested claims at the time, so treat it with caution. Type-safe does not mean correct.
- `LeakageReviewDecision` is already Choice/Choice/Choice/Score. Only the `rationale` is truly generative.
- **System 1** (fast typed decisions) + **System 2** (fine-tuned model for novel cases and explanations).
- **Build your own System 1:** the registry already contains labeled (evidence → decision) pairs, enough to train a small calibrated classifier with no vendor dependency.

### Thinking outside the box: the "spreadsheet / build system" architecture
NOOA borrowed object orientation and Jev borrowed typed logic. The old idea hiding in "a 10-step workflow that must never forget" is a **dependency graph with incremental recomputation** (Excel, Make, Bazel, dbt).
- Each stage is a **cell** with declared inputs. A plain topological sort decides what runs. No LLM call is needed for "what's next."
- The model fills **one cell at a time** and sees only that cell's inputs, so **forgetting becomes structurally impossible.**
- Stacked on top: pass-by-reference inputs, **diffs instead of full regeneration**, action tokens as cell IDs, and later a chess-engine-style search with the validator as the value function.
- **Start:** formalize the 5 stages as an explicit graph and rescope v3 training examples to one cell's inputs each.

You also asked me to remember that you're an AI/ML engineer with 7+ years of experience who wants **simple explanations**. That was saved to the project memory. Two diagrams followed: the **data graph** (5 nodes, each seeing only its own typed input) and the **model architecture** inside one node (System 1 / System 2, with the validator in front of both).

---

## 12. Phase 10: GNNs, speculative decoding, Jev benchmarking

### GNNs for this project
- **Not for the 5-stage workflow graph.** Fixed topology, nothing to learn; topological sort is enough.
- **Possibly for a knowledge graph** (dataset → feature → leakage pattern → rule) to flag a new column like `days_since_resolution` as structurally similar to `duration` even with no shared words. Also for meta-learning which model family to try first.
- Skeptical about GNNs directly on tabular features (usually lose to boosting).
- **Not yet:** about 600 experiments is small for a GNN. Revisit once the registry grows.

### Speculative decoding (MTP, EAGLE-3, DFlash, DSpark)
- All are **draft-then-verify**. The output is identical to normal decoding: a **speed and cost lever, not a quality lever.**
- **EAGLE-3:** a separate draft head, no target fine-tuning needed, >80% acceptance, 3–5.6× speedups reported; supported in vLLM and SGLang.
- **DFlash/DSpark:** block-parallel drafting; only weeks old, so pilot carefully.
- **Why it suits you:** acceptance rates are highest on low-entropy structured output (enums, fixed JSON keys, action tokens), which is exactly what DCLab produces.
- Path: fine-tune first → add EAGLE-3 (lightly retrain the draft head after your fine-tune) → check whether Qwen 3.8's native MTP can self-draft → serve with vLLM/SGLang.

### Jev in the workflow
- **Not a pipeline step after the LLM.** Jev and the LLM work on **different fields of the same answer at the same time**: enums go to Jev, prose goes to the fine-tuned model.
- Alternative with no vendor: **constrained decoding** on your own model for 5- or 17-way choices. Prototype that first; treat Jev as a swap-in experiment.
- Stage-by-stage mapping:

  | Stage | Typed decision | Stays generative |
  |---|---|---|
  | data_understanding | Imbalance severity (Choice) | Why missingness looks like an artifact |
  | leakage_audit | `risk_level`, `availability_status`, `evidence_field` | Rationale |
  | feature_engineering | Winning recipe (Choice) | Domain meaning of features |
  | model_selection | Winning family (Choice) | Trade-off explanation |
  | optimization_reliability | Feature production-safe (Noul) | Kept generative for now |

- **Post-hoc agreement test:** feed historical evidence to a typed classifier and check it reproduces the known `selected_model` / `selected_stage`. Start with model and feature selection.

### Benchmarking Jev as a classifier
Jev becomes **one more row in `model_results`**, same folds, same rule. Report the trained-vs-zero-shot asymmetry honestly, with zero-shot and few-shot rows and ms/row as a real column.

Delivered **`jev_classifier.py`** (sklearn-compatible wrapper; real `_call_jev()` stubbed) and **`run_jev_screen.py`** (manual CV loop, avoiding an sklearn 1.8 classifier-detection bug). With a mock:

| Model | Mean AUC | Std | Total s | ms/row |
|---|---:|---:|---:|---:|
| logistic_regression | 0.7795 | 0.0238 | 0.03 | 0.006 |
| jev_zero_shot | (mock) | (mock) | 10.52 | 10.511 |
| jev_few_shot_10 | (mock) | (mock) | 10.51 | 10.492 |

**Real finding before any real Jev call:** at even 10 ms/row, per-row calls are about **350× slower** than batch prediction on 1,000 rows, and it gets worse with size.

### NOOA vs. LangGraph
- **The conflict:** LangGraph's default `state["messages"]` grows with every node and resends everything, which undoes NOOA's savings.
- **The fix:** state holds only small references (`dataset_id`, `current_stage`, `evidence_ref`). Evidence lives outside the graph. Routing between nodes is plain Python on typed fields, never an LLM call.
- **3-tier cost ladder:**

  | Tier | Handles |
  |---|---|
  | Jev / own typed classifier | Pure Choice/Score/Noul |
  | NOOA | Decisions needing a lookup, computation or retry |
  | Full fine-tuned LLM | Novel evidence, human-facing rationale |

  Escalation is a plain `if` on confidence. **The same validator checks every tier.**

---

## 13. Phase 11: Transcript attempt and the training glossary

### Verbatim transcript
I started compiling the whole chat word for word, then **caught my own error** (I'd attached a later correction to an earlier turn) and stopped. I pointed you to **Settings → Privacy → Export data** as the reliable source. You chose a best-effort PDF anyway. I restarted from the visible text and got through Message 3 before the chat moved on. **It was never finished.** This recap covers the same need in summary form, built from the saved transcript.

### `ML_Training_Terminology_Glossary.md`
**106 terms** in 6 groups (data and pretraining data; pretraining and architecture; optimization mechanics; fine-tuning and adaptation; alignment and post-training; evaluation and generation), each with a plain English explanation and a short Persian term. Several link back to this chat (checkpoint, STaR, ReST, rejection sampling, MTP, speculative decoding).

---

## 14. Phase 12: Other domains

### Computer vision: transfers almost directly
| Your rule | CV version |
|---|---|
| DCLAB-R04 (preprocessing leakage) | Normalization stats computed on all images before the split |
| Leakage audit | **Near-duplicates** across splits (check with perceptual hashing or embeddings) |
| Split by time | **Split by source/clip**, not by frame |
| DCLAB-R13 | ResNet vs. EfficientNet vs. ViT depends on data size and domain |
| Subgroup floors | Per-subgroup mAP (lighting, skin tone) |
| Calibration | Calibration of detection confidence |

### Self-driving: split in two
- **Perception: transfers.** Same metrics; split by drive/route, not frame.
- **Planning/control: no.** (1) No held-out ground truth for "was braking right": it's a causal P(Y|do(A),X) question, as your own master doc warns. (2) The industry uses massive simulation plus ISO 26262/SOTIF, not k-fold CV. (3) Control loops need <10 ms; even the fastest tier here is 70–500 ms.
- Where your work helps: validating the **perception** models (leakage, calibration, weather/lighting subgroups).

### Maps and road flow: a real GNN win
- **Google Maps ETA is a production GNN** (DeepMind, up to 50% better ETA in cities such as Berlin, Tokyo, São Paulo); Baidu Maps uses ConSTGAT. Roads really are sparse graphs, and traffic really spreads along edges.
- 5 stages mapped: road graph plus speed time series; leakage = random-row splits, **spatial leakage** (split by region), post-outcome features like clearance time; features = lags/rolling plus neighbor speeds; **screen a historical-average and a boosted baseline before the GNN** (DCRNN, Graph WaveNet, STGCN); reliability = calibrated ETAs and rural/night/event subgroups.
- **Supersegments:** DeepMind trained on overlapping sub-graphs sampled by traffic density to keep it tractable.
- Your `hyperack` ride/delivery dataset sits next to this domain and may share leakage patterns.

### Agentic LLM with graph structure: three meanings
1. **Branch the workflow** only when a real dependency forks or merges (e.g., reconciling supersegment boundaries).
2. **Tool calls over the graph** (`get_neighbors`, `get_supersegment`, `get_recent_incidents`). **Build this one.** The agent orchestrates; the GNN predicts.
3. **Graph-structured attention** (Graphormer, Graph Transformer Networks): research, not now.

### Graph attention for computer vision
A ViT already does full attention (a complete graph), so forcing sparsity isn't clearly justified. Real fits: **point clouds/LIDAR** (DGCNN, PointNet++), **scene graphs** (object relationships), and **Vision GNN** (patches linked by feature similarity). Plain 2D classification/detection: probably not.

### Scene graphs for an agentic VLM (your actual goal)
1. **Detect everything** (open-vocabulary detector such as Grounding DINO). The complete graph is the concept, not the computation (30 objects ≈ 870 pairs).
2. **Prune pairs** before classifying (overlap, proximity, a proposal network), the same idea as Supersegments.
3. **Classify relationships** on the remaining pairs → scene graph (Visual Genome; Neural Motifs, RelTR).
4. **"Flow" means video:** spatio-temporal scene graphs, the same shape as traffic over time.
5. **VLM reasoning via tool calls** (`get_relationships`, `get_neighbors`, `get_object_history`), rather than dumping text into the prompt or changing attention.
6. **Test it:** VLM with vs. without graph access on GQA or Visual Genome relationship recall.

### How GNNs work, simply
Each node asks its neighbors "what do you think?" and updates itself; repeated 3–4 rounds, each node summarizes its neighborhood. Simple uses: fraud rings/AML, PinSAGE recommendations, molecules, traffic. **A GNN never sees pixels.** Something else (a detector, similarity, superpixels) builds the graph first. Graph **generation** exists as its own field, and graph transformers suit it better.

### What a GNN outputs
| Level | Output | Closest CNN equivalent |
|---|---|---|
| Node | One answer per node | Object detection |
| Edge | One answer per pair | None |
| Graph | One answer for the whole graph | Classification |

**For scene graphs:** edge-level. Graph in, the same graph with labeled edges out. For video: only the **edges that changed** per frame.

---

## 15. Phase 13: Moving into this coding session

- You said **"continue in code."** I checked for `torch`/`torch_geometric`, and the session then moved to a Claude Code workspace.
- The workspace started **empty** (not a git repo). The files from earlier lived in the old sandbox and aren't reachable here.
- A large Decision.ai "master context" arrived along with a note about a pending Bash call. I treated it cautiously and didn't run anything based on it. **Correction:** that material matches this project's own description and your Decision.ai notes. It is **your** master vision document (the same one behind `Decision_AI_Agent_Coding_Context.docx`), not something to distrust. The note about a pending Bash call still didn't match any real pending action.
- **Cloned** `Research_Development_DCLab` into `/home/claude/Research_Development_DCLab` (166 MB, real repo).
- Offered next steps: clone the `DCLab` product repo and do the leakage-review wiring for real; rebuild the SFT/RAG/MCP tools against the full repo; or scope Decision.ai work.

---

## 16. Every file produced

| File | Type | Purpose | Status |
|---|---|---|---|
| `DCLab_RD_Review_and_Roadmap.md` | Doc | Outsider-readable review, 22 rules, 10 new datasets | Delivered (a copy is also in the repo's `cluad_initial_reports/`) |
| `build_sft_dataset.py` | Code | v1: rules/blocks/claims → chat SFT | Ran on real data (104 examples) |
| `sft_train_sample.jsonl`, `sft_val_sample.jsonl` | Data | v1 samples | Delivered |
| `build_sft_dataset_v2.py` | Code | v2: real 5-part reasoning traces | 6/6 real experiments |
| `build_sft_dataset_v3.py` | Code | v3: dataset + stage + setup context | Real EXP-001–005, EXP-007 |
| `sft_train_v3_sample.jsonl`, `sft_val_v3_sample.jsonl` | Data | v3 samples | Delivered |
| `dataset_expansion_pack.py` | Code | 4 Kaggle adapters | Smoke-tested on synthetic samples |
| `SFT_DATA_GUIDE.md` | Doc | JSONL formats + TRL/LoRA script | Delivered |
| `DCLab_Field_Notes_Plain_Language.md` | Doc | Plain-language narrative | Delivered |
| `build_rag_index.py` | Code | Filter-then-rank retrieval | Ran on real data (82 records) |
| `AGENT_ARCHITECTURE_RAG_TOOLS_FINETUNE.md` | Doc | RAG vs. fine-tuning vs. tools | Delivered |
| `grounded_review.py` | Code | Retrieval + prompt, dry-run/live | Ran on real data |
| `dclab_knowledge_mcp_server.py` | Code | MCP server with 3 tools | Tools tested directly |
| `DCLab_Product_Integration_STEPS.md` | Doc | 7-step wiring into the real product | Delivered |
| `DCLab_RD_Operating_Model.md` | Doc | R&D as a standing function | Delivered |
| `DCLab_Agentic_Loop_RD_Verification_Plan.md` | Doc | How to verify the autonomous loop | Delivered |
| `notebook_copilot.py` | Code | 4 detectors with cited proof | Ran on demo notebook |
| `NOTEBOOK_COPILOT_SPEC.md` | Doc | UX and honest-claims spec | Delivered |
| `churn_model.ipynb` (demo) | Notebook | Test notebook with planted bugs | Delivered |
| `run_real_churn_screen.py` | Code | Real 5-model screen | Ran (logistic regression won) |
| `new_experiment_record.json` | Data | Result in registry format | Delivered |
| `dclab_studio.html` | Page | Full DCLab Studio design | Published as an Artifact |
| `jev_classifier.py` | Code | Jev as an sklearn classifier | Mock verified; real call stubbed |
| `run_jev_screen.py` | Code | Jev vs. classical benchmark | Ran with mock |
| `ML_Training_Terminology_Glossary.md` | Doc | 106 bilingual terms | Delivered |
| Inline mockups and diagrams | Visual | 5 UI mockups, 2 architecture diagrams | Shown in chat only |
| `conversation_transcript.md` | Doc | Verbatim transcript attempt | **Unfinished** (through Message 3) |

The files above were saved in the earlier session's output area, which this workspace can't reach. They'll need re-uploading or rebuilding here.

---

## 17. Key numbers (verified)

Re-checked against the cloned repo where possible.

| Fact | Value | Source |
|---|---|---|
| Campaign completion | 50/50 (10 datasets × 5 stages) | `campaigns/model_building_50_v1/results/` |
| Rules / workflow blocks | 22 / 10 | `knowledge/` (checked: 22 lines, 10 blocks) |
| Claim records | 80 lines | `agent_memory.jsonl` (checked) |
| Historical registry | 579 experiments, 11 datasets | Repo knowledge base |
| `duration` leakage lift, EXP-007 | **+0.1842** apparent ROC-AUC | `EXP-007_bank_marketing_leakage_audit.json` (checked) |
| `duration` leakage, earlier review table | +0.133 (0.803 → 0.936) | `DCLab_RD_Review_and_Roadmap.md`, a different comparison |
| `PageValues` leakage, online_shoppers | +0.158 | Same review table |
| LightGBM on adult (EXP-004) | 0.8955 mean, 0.0065 std | `EXP-004_adult_model_selection.json` (checked) |
| LightGBM margin over hist_gb | 0.0002 | EXP-004 |
| LightGBM mean rank across 11 datasets | 2.55 | Registry |
| Raw features beat engineered | 7 of 10 datasets | Campaign findings |
| Synthetic churn screen winner | logistic_regression 0.7757 vs. lightgbm 0.7313 | Live run in this chat |
| Jev mock latency | ~10.5 ms/row, ~350× slower than batch | Live run (mock) |
| NOOA paper | 82.2% / ~1.1M tokens / 28 calls vs. 78.2% / 2.2M / 66 | NVIDIA publication |

The two `duration` figures (+0.133 and +0.1842) come from different comparisons. Cite +0.1842 when you mean the campaign's EXP-007.

---

## 18. Decisions and principles

1. **Every claim must be checkable.** UI, marketing and training data included.
2. **Deterministic pipeline first.** LLMs fill narrow, schema-checked checkpoints; a validator gates every output.
3. **No parallel agent framework** inside the product. Extend the existing leakage-review pattern.
4. **No universal best model** (R13). Screen several families on identical folds and rank by mean − 0.25 × std, then runtime (R14).
5. **Leakage is the biggest risk** in any self-improving loop. The largest "wins" in the registry were leakage.
6. **Separate "prior" from "verified"** everywhere: in the UI, in training data and in claims.
7. **R&D earns claims before they ship**, and the loop closes both ways (production decisions return as evidence).
8. **Put findings into defaults**, not only LLM context, so they work with the LLM off.
9. **Self-contained training examples:** never an opaque experiment ID without dataset, stage and setup.
10. **RAG = filter first, rank second** on structured records; never PDF-style chunking.
11. **Fine-tuning makes it likely; constrained decoding and the validator make it guaranteed.**
12. **The workflow is a dependency graph:** one cell at a time, explicit inputs, no "what's next" LLM calls.
13. **Cheapest capable tier first** (typed classifier → NOOA → full LLM), same validator for all.
14. **The agent builds the predictive model; it doesn't become it.**
15. **Prove it on DCLab first**; the next domain should have an equally checkable validator (software engineering).
16. **GNNs only where the graph is real and has something to learn** (roads, point clouds, scene graphs), always against a strong baseline.
17. **Explain simply**, even for deep technical topics (your stated preference).

---

## 19. Corrections made along the way

| What went wrong | How it was caught | Fix |
|---|---|---|
| SFT examples cited opaque experiment IDs | **You** caught it | v3 adds dataset, stage and setup context |
| Mockup showed invented LightGBM numbers as fact | Your follow-up question | Real EXP-004 numbers, then a real screen where logistic regression won |
| Reference images: "Great choice!" hype, "Looks good!" beside open risks | Design review | Removed hype; status states the blocking finding |
| Integration plan assumed an MCP-driven agent | Reading the real product spec | Feed the existing checkpoint; MCP stays a research tool |
| Transcript attempt mixed up turns | Self-caught | Stopped, disclosed, pointed to the official export |
| Malformed synthetic fraud CSV | Self-caught | Regenerated with pandas |
| sklearn 1.8 didn't recognize the Jev wrapper as a classifier | Test run | Manual CV loop |
| Treated your Decision.ai master context as suspicious | This recap | It's your own vision doc; only the "pending Bash call" note was unexplained |

---

## 20. Open items and suggested next steps

**Unfinished:**
- The verbatim transcript (the official export remains the reliable source).
- The earlier files aren't in this workspace yet.
- The real Jev API call (`_call_jev()` is stubbed).
- Real Kaggle downloads for the 4 new datasets.

**Suggested order:**
1. Rebuild `build_sft_dataset_v3.py`, `build_rag_index.py` and the MCP server against the full repo now cloned here.
2. Clone `DCLab` and do the `leakage_review_v2` + `LeakageEvidenceField` wiring with its verification cases.
3. Apply the `_bounded_preview()` patch to `_evidence_json()`.
4. Run the **post-hoc agreement test** (typed classifier vs. historical `selected_model`).
5. Download and certify the 4 Kaggle datasets through the 5-stage campaign.
6. Build trajectory-SFT data scoped to one cell's inputs, plus ~10 stage tokens; run a LoRA pilot on an 8B model with the two-check go/no-go test.

---

## 21. خلاصهٔ فارسی

### مرور کلی

این گفت‌وگو با یک پرسش ساده آغاز شد: چگونه می‌توان خروجی‌های آزمایشگاه R&D را به دادهٔ تمیزتر برای فاین‌تیون یک مدل زبانی کوچک (SLM) تبدیل کرد. بررسی مخزن نشان داد که بخش بزرگی از کار از پیش انجام شده است: کمپین `model_building_50_v1` با ۵۰ آزمایش کامل روی ۱۰ مجموعه‌دادهٔ واقعی و ۵ مرحله اجرا شده بود و ۲۲ قانون و ۱۰ بلوک جریان کاری از آن استخراج شده بود.

### مسیر گفت‌وگو

| مرحله | موضوع | نتیجهٔ اصلی |
|---:|---:|---:|
| ۱ | تمیزسازی داده برای SFT | سه نسخه از سازندهٔ داده؛ نسخهٔ سوم با کارت مجموعه‌داده، کارت مرحله و خلاصهٔ تنظیمات |
| ۲ | RAG، فاین‌تیون و ابزارها | هر سه لازم‌اند؛ RAG با روش «اول فیلتر، بعد رتبه‌بندی» |
| ۳ | اتصال به محصول DCLab | تغذیهٔ نقطهٔ بررسی موجود نشت داده، بدون ساخت چارچوب عامل موازی |
| ۴ | R&D به‌عنوان کارکرد دائمی | حلقهٔ بازخورد دوطرفه و انتقال یافته‌ها به مقادیر پیش‌فرض پایپ‌لاین |
| ۵ | دستیار هوشمند نوت‌بوک | نمونهٔ اولیهٔ واقعی با ۴ آشکارساز و ارجاع به شواهد |
| ۶ | آزمون زندهٔ انتخاب مدل | رگرسیون لجستیک با AUC برابر ۰٫۷۷۵۷ برنده شد، نه LightGBM |
| ۷ | طراحی UI/UX | شش دور طراحی و انتشار صفحهٔ «DCLab Studio» |
| ۸ | مدل عامل متمرکز | نردبان چهارپله‌ای فاین‌تیون و توکن‌های کنشی |
| ۹ | کاهش هزینه و توکن | اصول NOOA، ایدهٔ Jev و معماری گراف وابستگی |
| ۱۰ | GNN و رمزگشایی گمانه‌زنانه | GNN فقط جایی که گراف واقعی است؛ رمزگشایی گمانه‌زنانه فقط اهرم سرعت است |
| ۱۱ | دامنه‌های دیگر | بینایی ماشین مستقیماً منتقل می‌شود؛ کنترل خودروی خودران خیر |

### تصمیم‌های کلیدی

- هر ادعایی باید قابل‌بررسی باشد؛ از رابط کاربری تا دادهٔ آموزشی.
- پایپ‌لاین قطعی‌نگر (Deterministic) در اولویت است و LLM فقط در نقاط بررسی محدود و کنترل‌شده با طرح‌واره عمل می‌کند.
- هیچ مدلی همیشه بهترین نیست؛ چند خانوادهٔ مدل روی فولدهای یکسان مقایسه می‌شوند.
- بزرگ‌ترین خطر در هر حلقهٔ خودبهبود، نشت داده است.
- فاین‌تیون رفتار درست را محتمل می‌کند، اما رمزگشایی مقید و اعتبارسنج آن را تضمین می‌کنند.
- جریان کاری مانند صفحه‌گسترده است: هر بار فقط یک خانه با ورودی‌های اعلام‌شدهٔ خودش پر می‌شود، بنابراین فراموشی از نظر ساختاری ناممکن می‌شود.
- عامل هوشمند مدل پیش‌بین را می‌سازد، اما خودش مدل پیش‌بین نمی‌شود.

### گام‌های بعدی پیشنهادی

۱. بازسازی ابزارهای SFT، بازیابی و سرور MCP روی مخزن کامل.
۲. اتصال واقعی `leakage_review_v2` در محصول DCLab.
۳. اعمال وصلهٔ `_bounded_preview()` روی `_evidence_json()`.
۴. اجرای آزمون توافق پسینی برای تصمیم‌های تایپ‌شده.
۵. دانلود و تأیید چهار مجموعه‌دادهٔ جدید Kaggle.
۶. آزمون آزمایشی LoRA روی یک مدل ۸B با دادهٔ مسیر تک‌خانه‌ای.

```
بازگشت به R&D ← دفتر تصمیم ← نقطهٔ بررسی محصول ← دانش ساخت‌یافته ← مخزن R&D
```

# From Vision to Verified: R&D's Role in the Agentic DCLab

*The pasted vision ("Cursor for ML," a state-graph the agent can reason over, an autonomous execution loop, auditable decisions, automatic investigation, persistent memory, immutable lineage) is the right target. This document is the answer to the actual question you asked: how does R&D make sure it's *correct*, not just impressive, before you hand it to real data scientists.*

---

## 1. The honest split: what's already proven, what's genuinely new

Read against everything built in this conversation, most of the vision doc isn't a new system to build — it's a UI for a backend you've already validated. One piece is genuinely new, and it's also the riskiest piece in the whole document.

| Vision doc feature | Already proven by | Status |
|---|---|---|
| "Automatic ML Investigation" (leakage, imbalance, duplicates, contamination, missing values, multicollinearity, drift, calibration, overfitting, subgroup performance) | Your 5-stage evidence pipeline (`workflow_blocks.json`, WF-01–WF-10) — run for real, 50 times, across 10 datasets | **Proven.** Needs a UI, not new methodology. |
| "I'll exclude `cancellation_date` because it appears to leak the target" | `LeakageReviewEvidence` / `request_leakage_review` — already shipping in the real product, and now cites your R&D precedent library | **Proven and shipping.** |
| Auditable decisions (Goal / Changes / Reason / Result / Cost / Artifacts / Accept / Reject / Compare / Modify) | The claim+evidence+limitations schema already in every experiment result JSON, plus the decision-ledger pattern | **Proven for one decision type (leakage).** Needs generalizing to feature-recipe, hyperparameter, and threshold decisions — same pattern, not new pattern. |
| Persistent ML memory ("why model v17 existed, why feature X was removed") | `agent_memory.jsonl` / `master_evidence_pack.json` claims, each with evidence + limitations + next-test | **Proven.** Needs to be the primary UI concept, not a side artifact. |
| Immutable execution graph / lineage tree | Provenance fields already in every result JSON (`data_sha256`, dataset version, git commit) — your own Rule of lineage (DCLAB methodology, Section 25 of the master vision doc) | **Proven as data.** Needs to be rendered, not re-derived. |
| "Should not optimize accuracy blindly" / picks PR-AUC for 0.2% fraud | DCLAB-R18 (metrics from business cost, not convention) | **Proven.** |
| No universal best model; screen linear + tree + boosting side by side | DCLAB-R13/R14 | **Proven.** |
| **The autonomous execution loop** (Experiment #1 → analyze error → adjust → Experiment #2 → ... → constraint satisfied) | Nothing yet. This is net-new. | **Not proven. This is the one to get right before shipping.** |

Everything in the first six rows is a UX/integration problem you already have the evidence and the backend pattern for (`DCLab_Product_Integration_STEPS.md` is that pattern; repeat it per decision type). The last row is different in kind: it's an agent that takes *multiple sequential actions toward a goal without a human approving each one*. That's exactly the "full autonomy" your own product spec explicitly paused — and it's also the single feature the vision doc calls "the biggest differentiator." Both things are true at once, and reconciling them is R&D's actual job right now.

---

## 2. Why the autonomous loop specifically needs an R&D verification step

An agent that iterates `train → look at the error → adjust → retrain` will happily find a change that improves the metric. It has no way of knowing, on its own, whether the improvement is real or is the exact failure your R&D campaign already documented: the biggest score jumps in your entire registry (+0.13 to +0.18 ROC-AUC) came from leakage, not skill. If the autonomous loop's error-analysis step ever suggests "include this column, it helped" without the same leakage discipline your manual pipeline already has, it will confidently ship the mistake your R&D program was built to catch — just faster, and with a plausible-sounding "Reason" attached.

So the loop isn't safe because it's well-designed UX (goal/changes/reason/result/cost, Accept/Reject/Compare/Modify). It's safe once you can show, with evidence, that its *reasoning process* matches what your R&D program already proved is correct. That's a testable claim, not a design opinion — here's how to actually test it.

---

## 3. The verification methodology: use your own registry as ground truth

You already have something almost nobody building an agentic ML tool has: **50 real experiments where a human-reviewed, evidence-first process reached a known-correct conclusion.** That registry is a ready-made regression test suite for the autonomous loop, before a single real user touches it.

**Test 1 — Replay against known answers.**
Point the autonomous loop at the same 10 datasets. Don't tell it what the R&D campaign found. Check whether it independently:
- Excludes the same leakage columns (`duration` in bank_marketing, `PageValues` in online_shoppers, etc.)
- Lands in the same neighborhood of final metrics your campaign's holdout evaluation reached
- Picks a model from the same family that won in your model-selection stage (or explains, with evidence, why it reasonably differs)

Any disagreement is either a bug in the loop or a genuine new finding — both are useful, but you need to know which before shipping.

**Test 2 — Red-team it with the leaky version on purpose.**
You already have the "unsafe" (leaky) version of several datasets from your safe-vs-unsafe ablation work. Feed the autonomous loop the leaky version and see if it takes the bait — reports a suspiciously large single-step improvement, and either (a) flags it as suspicious itself, matching DCLAB-R06's "a large safe-vs-unsafe gap is a severity demonstration, worth investigating," or (b) just reports success. Case (b) is a hard blocker, not a tuning issue.

**Test 3 — Constrain the action space to only what's already validated.**
This is the actual fix if Tests 1–2 fail, and it should be true by construction, not just hoped for: the loop's per-step *action menu* should be built from your own "Valid LLM responsibilities" list (already established for the leakage checkpoint), extended the same way to cover this loop's actions:

- ✅ adjust a hyperparameter *within a pre-approved range for that model family* (you have this range empirically, from your optimization-reliability stage's search space)
- ✅ move the decision threshold
- ✅ switch to another model *from the pre-screened shortlist* (not any model that exists)
- ✅ re-run with a feature recipe *one rung up or down the feature ladder* (raw → ratios → interactions), never an arbitrary new feature it invents mid-loop
- ❌ silently include/exclude a feature outside the leakage review's decision
- ❌ change the evaluation metric mid-run without saying so
- ❌ touch the final holdout more than once

This mirrors the exact "valid vs. invalid LLM responsibilities" split your product spec already uses for the narrower checkpoints — the autonomous loop is just several of those checkpoints chained, not a different kind of thing.

**Test 4 — Make every iteration produce an R&D-format record, for free.**
If each step of the loop logs itself in the same shape as your experiment result JSONs (hypothesis, evidence, claims, decision, next test), two things happen at once: you get the audit trail the vision doc wants (Goal / Changes / Reason / Result / Cost / Artifacts) essentially for free, since that's a direct rendering of the same schema — and every real user's autonomous run becomes a new entry in your knowledge base automatically, which is exactly the production→R&D feedback loop from the operating model discussed earlier. The audit UI and the knowledge-base growth mechanism turn out to be the same piece of plumbing.

---

## 4. What this means for build order

1. **Don't build the autonomous loop as one big new feature.** Build it as N more instances of the pattern you already shipped for leakage (`DCLab_Product_Integration_STEPS.md`'s recipe): bounded evidence in, schema-validated decision out, deterministic validator gates the action, ledger records it. Hyperparameter adjustment, threshold selection, and feature-recipe stepping each get their own narrow evidence dataclass and versioned prompt, same as `LeakageReviewEvidence` did.
2. **Chain them behind a deterministic loop controller**, not an open-ended agent — the controller decides *when* to call which checkpoint and *when to stop* (constraint satisfied, budget exhausted, or no further approved action available); the LLM never decides to keep going on its own.
3. **Run Tests 1–3 above against your own registry before any real user sees the loop.** This is the literal answer to "do some R&D to verify my job is implementing correctly" — you already have the ground truth to check against, you just haven't pointed the new thing at it yet.
4. **Ship the audit trail UI directly off the existing schema** (Test 4) — this is the "trust" the vision doc is right to prioritize, and you don't need to design a new data model for it.
5. **Only widen the action space (new model families, new feature transforms, larger hyperparameter ranges) after a new R&D campaign has validated that specific widening** — this is the "certify before advertising" principle from the operating model, applied to the agent's own capabilities, not just to new datasets.

The Cursor-for-ML framing is the right ambition. The reason it can be trustworthy instead of just impressive is that you didn't skip the step most agentic tools skip: you already have a real, evidence-based answer for what "correct" looks like, on real data, before the agent ever has to decide anything on its own.

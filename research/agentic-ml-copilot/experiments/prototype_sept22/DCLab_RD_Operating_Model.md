# R&D as a Standing Function Inside DCLab — Operating Plan

*What R&D is actually for, what to do with it now, and what "next phase" means. This ties together three things that haven't been read side by side until now: `Decision_AI_Agent_Coding_Context.docx` (the original master vision), the real `DCLab` product repo (what's actually shipping), and `Research_Development_DCLab` (the evidence engine).*

---

## 1. What R&D's job actually is, behind the scenes

Your master vision document describes a layered system — feature intelligence, a model factory that screens hundreds of candidates, diversity-aware selection, calibrated ensembles, an Intelligence State as "the semantic interface between layers." Your product's own architecture history shows that the *literal* version of that (a full autonomous orchestrator with hypothesis graphs and critics) was deliberately rejected in favor of something narrower and more provable: a deterministic pipeline with small, evidence-constrained LLM checkpoints.

R&D is the thing that resolves that tension. It doesn't build the autonomous version of the vision — it **proves, on real data, which pieces of that vision actually hold up**, and turns them into the specific defaults, prompts, and guardrails the deterministic product encodes. Every one of the 22 rules in your knowledge base is a small piece of the master document's Phase 1–3 (dataset intelligence, feature/model-space exploration, evaluation/diversity/ensemble) — just answered empirically instead of assumed. That's the actual job: **R&D is where the product's claims get earned before they get shipped.**

Concretely, that means three standing responsibilities, not a one-time deliverable:

1. **Certify capabilities before they're marketed.** Before DCLab claims to handle a new kind of problem (imbalanced fraud, multiclass, time-series, text+tabular — the four gaps from your dataset-expansion pack), R&D runs the same evidence-first process on real data from that domain first. If it doesn't hold up, the product doesn't claim it yet.
2. **Feed the product's actual decision points with real evidence**, not hand-tuned defaults — both the LLM checkpoints (the leakage-review wiring from the last guide) and the deterministic pipeline code itself (Section 3 below).
3. **Manufacture the B2B proof.** Your own master doc names this directly: *"the strongest proof is a controlled benchmark showing more hypothesis exploration, equal or better predictive quality, and materially lower engineering time."* That benchmark doesn't exist until R&D produces it — it's not just internal engineering hygiene, it's sales collateral.

---

## 2. The closed loop this creates

```
R&D campaign (new dataset/domain)
        │
        ▼
Knowledge base (rules, workflow blocks, experiment claims)
        │
        ├──► Product LLM checkpoints (leakage_review_v2, etc. — precedent evidence)
        │
        ├──► Product pipeline DEFAULTS (which models get screened, which feature
        │     recipes get tried, in what order — hardcoded, not LLM-mediated)
        │
        └──► B2B proof pack (benchmark reports, case studies)
                │
                ▼
        Real product usage → decision ledger
                │
                ▼
        Periodic export of confirmed/corrected decisions
                │
                ▼
        New claims appended to the SAME knowledge base format
                │
                ▼
        (back to top — next R&D campaign is informed by production gaps,
         not just the original 50-experiment plan)
```

The loop only closes if Section 2's bottom half actually happens — without the production→knowledge-base export, this is a one-time knowledge transfer, not a standing R&D function. That export step is the single highest-leverage thing to build once the leakage-review wiring is live.

---

## 3. Phase Now — this week and this month

**This week:**
- Do the leakage-review wiring from the last guide (`DCLab_Product_Integration_STEPS.md`). This is the first real, working instance of the loop above — get it running before adding anything else.
- Sync the knowledge files into the product repo as static, versioned data (Step A of that guide). Treat this sync like a migration: deliberate, reviewed, dated — not automatic.
- Start using `grounded_review.py` yourself, personally, whenever you're manually reviewing a customer's dataset or debugging a Lab run — it's already faster than reconstructing the reasoning from memory each time.

**This month — the highest-value thing not yet done: put the findings into the pipeline's actual defaults, not just the LLM's context.**

This is different from the leakage-review wiring, and more valuable in some ways, because it doesn't depend on the LLM being enabled at all. Your 50-experiment registry already answered several questions your deterministic pipeline currently has to answer some other way (a hardcoded default, or nothing):

| Registry finding | Where to encode it |
|---|---|
| No universal best model; LightGBM best mean rank, but always screen a linear baseline + 2–3 nonlinear families | The model candidate list in your model factory / `auto_prepare.py` — make sure the *default screen* matches this, not just "whatever we happened to hardcode first" |
| Raw features win 7/10 datasets; feature engineering must clear a predeclared margin over raw to be kept | The feature-engineering step's promotion rule — if it currently always applies engineered features, add the raw-baseline comparison gate |
| A feature's importance is a nomination, not a verdict (needs availability + stability + monitoring) | Whatever surfaces "important features" to the user in the Lab UI — flag features that haven't passed the reliability checks, don't just show a bare importance ranking |
| Rank by mean performance *and* stability *and* runtime, not top score alone | The model-selection logic itself, if it currently picks by best single CV score |

This is a genuinely different integration point from the LLM checkpoint work — it's encoding evidence into code that runs even with the LLM disabled, which matches your own principle that "the entire pipeline must still function when the LLM is disabled." Some of these should not be LLM-mediated recommendations at all; they should just be how the pipeline behaves by default.

---

## 4. Phase Next — this quarter and beyond

**Certify the four new domains before advertising them.** Run `dataset_expansion_pack.py`'s four Kaggle datasets (imbalanced fraud, multiclass, time-series, text+tabular) through the exact same 5-stage process as the original 50. Two outcomes either way are useful: if the methodology holds up, you now have real evidence to support a marketing claim ("DCLab handles severe class imbalance — here's the benchmark"); if it doesn't, you've found the gap before a customer did, on your own schedule.

**Close the feedback loop for real.** Once the leakage-review wiring has been live for a few weeks, build the export step: production decisions a human later confirmed or corrected become new lines in `agent_memory.jsonl`, in the same shape as the original experiments. This is what makes R&D a standing function instead of a one-time project — the knowledge base should look measurably different in six months, grown from real usage, not just from new campaigns you ran yourself.

**Only then, fine-tuning/RAFT.** This becomes worth doing once the corpus is rich with product-validated evidence, not just the original synthetic campaign — a model fine-tuned on real production precedent is a meaningfully different (and better) asset than one fine-tuned on the initial 50 experiments alone. Nothing about Phase Now or the rest of Phase Next depends on this happening first.

**Feed the B2B proof pack continuously, not once.** Every certified domain and every closed feedback-loop cycle is a new data point for the benchmark your master doc calls for. Treat `DCLab_RD_Review_and_Roadmap.md`-style reports as a recurring output of R&D, not a one-off document — one per quarter, or one per newly certified domain, is a natural cadence.

---

## 5. How to know R&D is actually paying off

Pick a small number of things to watch, and revisit them each time a campaign finishes:

- **Is the knowledge base growing from real usage**, not just from campaigns you personally ran? (The clearest signal the loop actually closed.)
- **Has a pipeline default ever changed because of a registry finding**, not just an LLM prompt? (Section 3's table — track how many of those four rows actually get implemented.)
- **Can you point to a specific customer-facing claim and trace it to a specific experiment?** (This is the lineage principle from your own methodology, applied to the business itself, not just to a model.)
- **Did a certification run (Section 4) ever catch a real gap before a customer did?** That's the ROI case for doing this proactively instead of reactively.

If the answer to all four is "not yet" after a quarter, the loop isn't closed yet — Section 2's diagram is the checklist for what's missing.

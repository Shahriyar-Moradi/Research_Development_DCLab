# Notebook Copilot: Evidence-Cited Suggestions, Not Just Opinions

*What you described — comments beside every cell, "show me the proof," a StackOverflow-style "someone else already hit this" trust mechanism — is now a working prototype (`notebook_copilot.py`), tested on a real notebook above. This is the product spec for what it becomes.*

---

## 1. The core idea, stated precisely

Every other AI code-review tool gives you an opinion with confident phrasing. This one gives you a **citation**: which of your 22 evidence-based rules triggered, and which real past experiment (dataset, numbers, outcome) backs it up. The difference is the same as the difference between "this looks risky" and "here's the exact case where this exact pattern cost someone +0.18 fake AUC — dataset, rule ID, and the real number, right here." That citation is the trust mechanism — it's checkable, not just plausible.

## 2. The UX — two surfaces, one evidence source

**Inline (the "beside every cell" ask):**
```
Cell 3
──────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, ...)

⚠ HIGH · Leakage
`scaler.fit(...)` runs before `train_test_split(...)` — the transformer
sees test rows during fitting.
[Show proof ▾]
```

**Expanded "Show proof" panel** (this is the StackOverflow moment):
```
Rule DCLAB-R04 (leakage, high confidence):
"Treat post-outcome, future, target-derived, cross-split aggregate,
 preprocessing leakage as separate failure modes."

Precedent: EXP-002 (adult dataset, leakage audit stage)
→ Real registry cases where a safe-vs-unsafe ablation showed exactly
  this pattern inflating the apparent score.

[Apply suggested fix]  [Dismiss]  [Ask a follow-up]
```

**Chatbot mode** — the same retrieval, conversational: "why did you flag this?" gets the same rule + precedent, in prose, with room for follow-up ("show me a dataset where this actually mattered").

Both surfaces call the exact same thing: `RetrievalIndex.search()` from `build_rag_index.py`. There's one evidence source, two ways to read it — this matters because it means the inline comment and the chatbot answer can never disagree with each other; they're the same lookup.

## 3. Be honest about the "50 Kaggle challenges" claim, because it's checkable

Said out loud to a data scientist, "50 Kaggle challenges solved" is a specific, checkable claim — and right now it isn't quite true yet. What you actually have is **50 experiments across 10 real UCI datasets**, run with the same evidence-first process a Kaggle challenge would need. That's genuinely strong, but it's not the same sentence.

Two honest ways forward, not mutually exclusive:
- **Say what's true today:** "every suggestion is backed by DCLab's own 50-experiment evidence registry, run on real public datasets" — true right now, still a strong trust signal.
- **Make the literal claim true:** run the certification process (already speced in the operating-model doc) against actual named Kaggle competitions — `dataset_expansion_pack.py`'s four datasets are a start, and each is a real Kaggle dataset/competition by name. Once a batch of real competitions has gone through the same evidence-first process, "N real Kaggle datasets, evidence-certified" is a claim you can name and defend, not just gesture at.

The trust mechanism you're building is strong exactly because every claim is checkable — so the marketing claim needs the same property, or it undercuts the thing that makes the product different from a generic AI reviewer.

## 4. What v0 proved, and what to harden next

**Proved (real output, above):** notebook code in → structural detection → real rule + real precedent cited → inline finding out. Four detector types working end to end: fit-before-split leakage, known-leaky column names, no-model-comparison, missing `random_state`.

**Harden next, in order of value:**
1. **Swap TF-IDF for real embeddings** (as already flagged in the RAG design doc) — the `random_state` finding above pulled a technically-real but weakly-related precedent; better retrieval fixes this directly, no new architecture needed.
2. **Grow the detector list from the same 22 rules**, not by inventing new heuristics ad hoc — each of the remaining rules (missingness handling, categorical encoding, calibration, subgroup performance) is a candidate detector using the exact same "structural pattern → rule category → retrieve precedent" recipe already proven above.
3. **"Apply suggested fix"** — for the mechanical findings (add `random_state=42`, reorder fit/split), this can be a literal code edit, not just a comment; for the judgment calls (is this column really unsafe here?), keep it a suggestion with proof, never an auto-applied change — matches your own "the model recommends, a deterministic layer decides" principle from the product side.
4. **Ship it where the data scientist already is** — a Jupyter/JupyterLab extension for the inline surface, and the same retrieval exposed as an MCP tool (reusing `dclab_knowledge_mcp_server.py`) so Claude Code, Cursor, or any MCP-aware editor gets the chatbot surface for free, without building a second integration.

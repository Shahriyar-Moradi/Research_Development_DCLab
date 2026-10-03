# Making the DCLab Agent Actually *Use* This Data — Fine-Tuning vs. RAG vs. Tools

Short answer: this isn't a "pick one" question. Fine-tuning, RAG, and tool-calling answer three different sub-questions, and DCLab's own architecture principle — **deterministic-first pipeline, narrow evidence-constrained LLM checkpoints, full autonomy explicitly rejected** — already tells you how to combine them. Below is what each one is actually for, how your data should be shaped for it, and the concrete design that fits what you've already decided.

---

## 1. What each technique is actually solving

| Technique | Answers | Fails at | DCLab already has... |
|---|---|---|---:|
| **Fine-tuning** | "How should the model *think and format its answer* by default?" — bakes in a reasoning style and stable, slow-changing rules. | Anything that changes weekly (new experiments, new numbers) — you'd have to retrain to update it. Can also confidently state a stale fact from training if not grounded. | `sft_train_v3.jsonl` — teaches the 5-part reasoning format and the 22 rules. |
| **RAG (retrieval)** | "What does *this specific dataset/experiment/rule* actually say, right now?" — fresh facts, cheap to update (just re-index). | Reasoning quality — retrieval finds the right passage, it doesn't teach the model to reason carefully about it. A model with bad judgment plus perfect retrieval still makes bad judgment calls. | `evidence/knowledge/*.jsonl`, `agent_memory.jsonl`, `master_evidence_pack.json` — already structured, just not indexed for retrieval yet (`build_rag_index.py`, below, does that). |
| **Tool-calling / agentic execution** | "How does the agent actually *do* the next pipeline step, not just talk about it?" — running the leakage detector, logging a claim, advancing to the next stage. | Nothing it's not designed for — this is the only one of the three that can *act*. It needs the other two to decide well when it acts. | Your `AGENT_CONTEXT.md` role (`critic_and_hypothesis_generator_only`) and LangGraph-based Agentic Research Studio are already this pattern, just not yet exposed as reusable, generic tools. |

The practical rule: **fine-tune for behavior and stable rules, retrieve for facts, call tools to act.** All three, together, layered — not a choice between them.

---

## 2. How your data should be shaped for RAG, specifically

Most RAG advice assumes you're chunking unstructured documents (PDFs, wikis) into ~500-token windows and embedding each chunk. **Don't do that here.** Your knowledge base is already structured — a rule has a category, statement, why, confidence; a claim has a dataset, stage, kind, evidence pointers. Chunking those into fixed-size text windows would slice a rule's "statement" away from its "why" mid-record, destroying structure you get for free.

The right retrieval unit is **one whole record** — one rule, one workflow block, one experiment's claim-group — each carrying real metadata. That makes retrieval two steps, not one:

1. **Filter first** (exact, deterministic, free): `dataset=bank_marketing`, `stage=leakage_audit`, `category=leakage`, `confidence=high`. This alone often narrows thousands of records to a handful with zero ambiguity — no embedding model needed for this part.
2. **Rank second** (fuzzy, similarity-based) — only within whatever survives the filter.

This is "hybrid retrieval," and for a knowledge base this structured, it beats pure vector search over everything: pure similarity search can rank a well-worded but wrong-dataset record above an exact match that filtering alone would have found instantly.

I built and ran this against your real files (`build_rag_index.py`, attached) — no downloads needed, it uses TF-IDF so you can verify the *retrieval design* today, then swap in real embeddings later by changing one class. Real output, right now, from your real repo:

```
Loaded 82 retrievable records (22 rules, 10 workflow blocks, 50 experiment-claim groups)

Query: 'is it safe to use call duration as a feature' | filters={'dataset': 'bank_marketing'}
  score=0.192 | [Experiment EXP-007 | dataset=bank_marketing | stage=leakage_audit]

Query: 'how do I know a feature is production ready' | filters={}
  score=0.287 | [Rule DCLAB-R22 | promotion | confidence=high methodological]

Query: 'what does the leakage audit stage actually check' | filters={'kind': 'workflow_block'}
  score=0.455 | [Workflow WF-05 | Leakage audit]
```

Filtering by `dataset=bank_marketing` alone correctly surfaces the exact `duration`-leakage finding as the top hit — no embedding model even had a chance to get that wrong. To move to production quality, swap `TfidfVectorizer` for a real embedding model (e.g. a local `sentence-transformers` model, or an API), keep everything else (the filter-then-rank structure, the per-record metadata schema) unchanged.

---

## 3. Fine-tuning + RAG together: this is what v3 already does

There's a named technique for training a model specifically to reason well over retrieved context instead of from memory: **RAFT (Retrieval-Augmented Fine-Tuning)**. The idea: don't just fine-tune on question→answer pairs; fine-tune on (retrieved-context + question)→answer, so the model learns the *habit* of grounding its answer in what it was given rather than what it remembers.

Look back at what `build_sft_dataset_v3.py` already generates — the dataset card, stage card, and setup summary injected into every user turn *are* the "retrieved context" in this framing. You've already built RAFT-style training data without naming it that. The natural next step: at inference time, replace the hand-built context blocks with the output of `build_rag_index.py`'s `search()` — the model was already trained to expect and use exactly that shape of context.

---

## 4. Tool-calling: this is what actually runs the workflow

Fine-tuning and RAG both improve what the model *says*. Neither makes it *do* anything. For "the DCLab agent uses these rules to run the ML workflow and follow the steps," you need tool-calling (function calling) — and this is also where your architecture principle of narrow, evidence-constrained checkpoints comes in directly.

The pattern (matching what `AGENT_CONTEXT.md` already declares as the LLM's role):

```
deterministic state machine (LangGraph — you already use this)
  │
  ├─ node: run_data_understanding(dataset)   [deterministic code, no LLM]
  ├─ node: run_leakage_audit(dataset)        [deterministic code, no LLM]
  ├─ node: run_feature_ladder(dataset)       [deterministic code, no LLM]
  ├─ node: run_model_screen(dataset)         [deterministic code, no LLM]
  ├─ node: run_optimization(dataset)         [deterministic code, no LLM]
  │
  └─ at each node's output → ONE bounded LLM/SLM tool call:
        input:  structured evidence JSON (exactly what the node produced)
        tools available to it: get_rule(category), get_workflow_block(stage),
                                 retrieve_similar_experiment(dataset, stage)  ← RAG, from Section 2
        output: MUST match a strict schema (observed_evidence, interpretation,
                 decision, risks, next_experiment) ← validated, exactly your
                 existing llm_review schema
        on schema failure → deterministic fallback (never silently retry forever)
```

The LLM never touches data, never decides which stage runs next on its own, and never gets the final word on a decision — exactly your existing rule. What's new here is only: (a) the rules/workflow-blocks/similar-experiments become **callable tools** the LLM can pull from mid-reasoning instead of everything being stuffed into the prompt up front, and (b) this whole pattern is genuinely reusable across every dataset and every stage, instead of being re-implemented per experiment.

**This maps directly onto your product architecture, not just the R&D repo.** Your core product is already an MCP server. The exact same tool surface above — `get_rule`, `get_workflow_block`, `retrieve_similar_experiment`, `run_leakage_audit`, `log_claim` — is naturally exposed as MCP tools. That means the R&D knowledge base and the product's own MCP interface can become *the same tool surface*: what the DCLab agent uses internally to run its own research is what a DCLab user's coding agent calls externally through your MCP server. That's not a coincidence worth losing — it's a reason to build this tool layer once, generically, rather than as a one-off script for the R&D repo.

---

## 5. Is anything better than RAG here? Yes, situationally

Direct answer to "is there something that works even better than RAG": for *this* knowledge base, plain vector-similarity RAG is rarely the strongest option on its own. In rough order of fit for structured, small-to-medium knowledge bases like yours:

1. **Structured/metadata-filtered retrieval (Section 2)** — best fit here. You almost always know the dataset/stage/category you care about; use that first.
2. **Tool-calling with narrow, typed functions (Section 4)** — best fit for *acting*, not just *knowing*. This is not a RAG alternative so much as the layer RAG plugs into.
3. **RAFT-style fine-tuning (Section 3)** — best fit for teaching the *habit* of reasoning from given evidence rather than memory; complements retrieval, doesn't replace it.
4. **Fine-tuning alone, no retrieval** — fine for stable, rarely-changing knowledge (the 22 rules, the workflow blocks). Wrong choice for the 50-and-growing experiment registry, since every new experiment would need a retrain to be "known."
5. **Pure vector-embedding RAG over raw JSON blobs** — the weakest fit here specifically, because your data is already structured and chunking it destroys that structure for no benefit. This is the default many RAG tutorials assume, and it's the one to skip.

---

## Recommended build order

1. Ship `build_rag_index.py` as-is (it already works) behind a thin API — this alone lets any agent (fine-tuned or not) ground its answers in current evidence.
2. Swap `TfidfVectorizer` for a real embedding model once you want fuzzier matching than keyword overlap gives you.
3. Fine-tune on `sft_train_v3.jsonl` so the model's default reasoning style and grasp of the 22 rules doesn't depend on retrieval succeeding every time.
4. Wrap both behind the tool-calling layer in Section 4, inside your existing LangGraph state machine — this is the piece that turns "a model that answers questions well" into "an agent that runs the workflow."
5. Expose that same tool layer as MCP tools — same design serves your R&D loop and your actual product's agent interface.

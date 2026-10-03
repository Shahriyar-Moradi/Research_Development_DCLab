# How an agent uses DCLab's R&D knowledge

**Short answer:** it is not RAG *or* fine-tuning *or* agents. Each does a different job.

| Technique | Job | Use it for | Do not use it for |
|---|---|---|---|
| **Tools** (function calling, MCP) | *Act* and *look things up* deterministically | Running the workflow, auditing columns, reviewing notebooks, fetching a record by ID | Free-form opinions |
| **Retrieval** (RAG) | Supply today's facts | Experiment results, leakage precedents, measured pitfall costs, which change as campaigns run | Teaching a reasoning style |
| **Fine-tuning** (SFT, RAFT) | Teach a default way of thinking | Answer structure, stable rules, the habit of citing evidence | Storing numbers that change |

DCLab already chose this split as a principle: deterministic code owns splits, metrics and artifacts; the LLM proposes, critiques and explains (rule DCLAB-R20). Everything below is that principle made concrete.

---

## 1. Records, not chunks

The usual RAG recipe cuts documents into fixed-size text windows. That is wrong for DCLab's evidence, because the evidence is already structured. Chunking would separate a rule's statement from its "why", or an experiment's numbers from the dataset they describe.

[`dclab_rnd/evidence_index.py`](../../dclab_rnd/evidence_index.py) turns every R&D output into one **self-contained record**:

| Type | Example ID | What it holds |
|---|---|---|
| `rule` | `DCLAB-R04` | Statement, why, failure signal, next test, confidence |
| `workflow` | `WF-05` | Ordered steps of one workflow block |
| `dataset` | `DATASET-bank_marketing` | What the data predicts, size, decision-time contract, blocked columns |
| `experiment` | `EXP-007` | Dataset card + stage card + what was compared + findings + critic review |
| `leakage_precedent` | `LEAK-bank_marketing` | A Stack-Overflow-style Q&A: "can I use `duration`?" with the measured lift and the resolution |
| `pitfall` | `PIT-003` | Measured cost of a common mistake (wrong way vs right way on identical data) |
| `finding` | `FINDING-model-families` | Cross-study conclusions such as "no universal best algorithm" |

Every record carries `metadata` (dataset, stage, task type, category) and `citations` (the file the claim comes from). The full index is committed as [`evidence/knowledge/rag/records.jsonl`](../../evidence/knowledge/rag/records.jsonl), and CI fails if it is stale.

### Filter first, rank second

You almost always know the dataset or stage you care about. Use it:

```python
from dclab_rnd.evidence_index import EvidenceIndex
index = EvidenceIndex.load()
index.search("is it safe to use call duration", dataset="bank_marketing")   # exact filter, then BM25 ranking
index.search("oversampling before split", type="pitfall")
```

Filtering is exact and free. Ranking only orders what survives. On a knowledge base this small and this structured, metadata filtering beats pure vector similarity. BM25 is used because it is transparent and needs no model. To switch to embeddings later, embed `title + text` per record and keep the same filters; the record contract does not change.

---

## 2. Tools: what the agent can call

[`dclab_rnd/tools.py`](../../dclab_rnd/tools.py) exposes the evidence as typed, read-only tools:

| Tool | What it does |
|---|---|
| `search_evidence(query, type?, dataset?, stage?, task_type?, k?)` | Filter-then-rank search |
| `get_record(record_id)` | Full record by ID, for citations |
| `get_rules(category?)` | The rules for one category |
| `plan_next_stage(dataset?, completed_stages?)` | Next workflow stage, its method, governing rules, and precedents for the dataset |
| `review_notebook(path)` / `review_code(source)` | Static methodology review with proof (the copilot) |
| `audit_columns(path, target, task?, blind?)` | Ranks columns by leakage risk with deterministic signals |

The same functions serve three interfaces:

```bash
python -m dclab_rnd.tools schemas --style openai      # function-calling definitions (OpenAI shape)
python -m dclab_rnd.tools schemas --style anthropic   # tool definitions (Anthropic shape)
python -m dclab_rnd.tools call search_evidence '{"query": "target encoding", "type": "pitfall"}'
pip install mcp && python -m dclab_rnd.tools mcp      # Model Context Protocol server over stdio
```

Because DCLab's product is itself exposed through MCP, the R&D tools and the product tools can live on one tool surface.

---

## 3. Using it today, without fine-tuning

This works now with any capable hosted model (the Studio's configured OpenAI model, or Claude).

**Pattern A — grounded answer.** For a user question, call `search_evidence` with the dataset and stage filters you know, put the top 3 to 5 records in the prompt, and require the answer to cite record IDs. The system prompt in `research/llm-fine-tuning/sft/out_v3/MANIFEST.json` (`system_prompt`) is a good starting point. The five-part answer structure (Evidence, Interpretation, Decision, Risks, Next test) keeps answers checkable.

**Pattern B — the workflow loop.** A deterministic controller walks the five stages. At each stage:

1. `plan_next_stage` returns the method and rules,
2. deterministic code runs the stage (the campaign engine, or the Studio worker),
3. the LLM reads the result plus retrieved precedents and writes a *bounded, schema-validated* critique,
4. `dclab_rnd.critic_gate` recomputes the stage's rule and drops critique that the numbers disprove,
5. the result and the surviving claims are written as a new evidence record.

The controller, not the LLM, decides when to stop. This is the same shape the Studio already uses for leakage review, chained across stages.

**Pattern C — the copilot.** Run `review_notebook` on a user's notebook and show each finding beside its cell with "Show proof". See [NOTEBOOK_COPILOT.md](NOTEBOOK_COPILOT.md).

---

## 4. Fine-tuning, when it is worth it

Fine-tune only after Patterns A to C work with a hosted model and you have a reason to move to a small model (cost, latency, offline). The training data is already shaped for it: every v3 example is `(evidence in the prompt) → structured answer`, the RAFT pattern. See [SFT_DATA_GUIDE.md](SFT_DATA_GUIDE.md). Keep retrieval after fine-tuning, because numbers change and weights should not have to.

---

## 5. Guard-rails that keep the agent honest

| Risk | Guard-rail in this repository |
|---|---|
| The critic is wrong but confident | `critic_gate` recomputes rules; 10 of 157 critic challenges were disproved and are excluded from training and memory |
| A heuristic scan is mistaken for a safety proof | `audit_columns` flags review candidates only. The blind replay found 11 of 18 known leaks across 7 datasets; the misses look like ordinary numbers, so the agent must always ask for the decision-time contract |
| An optimization loop "improves" a score through leakage | The largest score jumps in the registry were leakage (+0.13 to +0.18 ROC-AUC). The loop may only act inside the validated menu, and any jump larger than the measured pitfall costs triggers a leakage audit |
| Answers drift from evidence | Every answer cites record IDs; `get_record` lets anyone open the source file |
| Stale knowledge | `make rd-check` fails CI when the index, the SFT corpus or the generated knowledge no longer match the result files |

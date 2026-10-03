# Agentic ML copilot

**Status:** active · code lives in `dclab_rnd/`

## Research question

Can an agent review and run notebook-style ML work for data scientists, and be right, with proof they can check?

## Why it matters for DCLab

This is DCLab's product thesis: replace the manual notebook loop with an agent that is evidence-backed.

## Where the work lives

| Piece | Code | Guide |
|---|---|---|
| Evidence index (retrieval) | `dclab_rnd/evidence_index.py` | [Architecture](../../docs/guides/AGENT_KNOWLEDGE_ARCHITECTURE.md) |
| Agent tools + MCP server | `dclab_rnd/tools.py` | [Architecture](../../docs/guides/AGENT_KNOWLEDGE_ARCHITECTURE.md) |
| Notebook copilot | `dclab_rnd/copilot/` | [Copilot](../../docs/guides/NOTEBOOK_COPILOT.md) |
| Notebook companion (cell-level advice with exact rule references, shared output contract for a VS Code extension) | `dclab_rnd/notebook_assist.py`, `notebook_assist_eval.py`, `notebook_eval.py` | Frozen development cases: `research/evaluation-and-trust/notebook_cases_v1.json` |
| First prototypes (historical) | [`prototype_sept22/`](prototype_sept22/README.md) | |
| Critic gate | `dclab_rnd/critic_gate.py` | [SFT guide](../../docs/guides/SFT_DATA_GUIDE.md) |
| Research Studio (LangGraph + NOOA agents, web UI) | `dclab_rnd/agentic/` | Main README |
| Product plan and verification | | [Integration plan](../../docs/guides/DCLAB_RND_INTEGRATION_PLAN.md) |

## How to run

```bash
python -m dclab_rnd.copilot review YOUR_NOTEBOOK.ipynb --html review.html
python -m dclab_rnd.tools list
python -m dclab_rnd.tools call plan_next_stage '{"dataset": "adult", "completed_stages": ["data_understanding"]}'
make agent-serve        # Research Studio UI at http://127.0.0.1:8765 (agent environment)
```

## Next experiments

| ID | Question |
|---|---|
| AGT-001 | Campaign replay: run the guarded loop on the 10 campaign datasets without their answers; does it exclude the same leaks and land inside the holdout intervals? |
| AGT-002 | Red team: feed the loop the unsafe ablation data; does it flag the suspicious jump? |
| AGT-003 | Copilot precision on real notebooks: label 50 public Kaggle notebooks and measure true and false findings |

## Agentic R&D studio: charter and evaluation

**Status: implemented prototype; continuing evaluation required.** The tracked studio combines typed specialist roles, a LangGraph controller, a bounded experiment schema, and deterministic model execution. It is intended to turn a research question into traceable proposals, trials, critiques, and follow-up questions.

### Questions to answer

- Does the plan match the user's question and dataset constraints?
- Are proposals valid, useful, and meaningfully different?
- Does the system react correctly to measured evidence and failures?
- Are runs reproducible and do artifacts preserve provenance and limitations?
- Does the system avoid unsupported conclusions and unsafe promotions?

### Existing implementation and evaluation

- Runtime: [`../../dclab_rnd/agentic/`](../../dclab_rnd/agentic/)
- Agent tests: [`../../tests/test_agentic.py`](../../tests/test_agentic.py) and [`../../tests/test_agentic_worker.py`](../../tests/test_agentic_worker.py)
- Runtime source: [`../../dclab_rnd/agentic/engine.py`](../../dclab_rnd/agentic/engine.py) and [`../../dclab_rnd/agentic/worker.py`](../../dclab_rnd/agentic/worker.py)

### Next useful test

Build a blinded suite of research requests with expected constraints and acceptable plans. Score plan adherence, invalid-proposal rate, citation grounding, whether the agent selects a discriminating next experiment, and deterministic metric agreement. Include rate-limit/provider failures and resume behavior. A completed run is not itself proof of correctness.

Run instructions are in the [main README](../../README.md#how-to-use-it). Live runs send task context to the configured model provider, so use data that is authorized for that destination.

## Notebook assistant: charter and evaluation

**Status: notebook copilot prototype exists; VS Code inline integration is proposed.** The goal is to help ML engineers where they work: point out possible issues near a code cell, explain evidence, and suggest a next check. The current tracked copilot can analyze a notebook and render a review; it is not yet an editor extension that continuously comments beside the active VS Code cell.

### Questions to answer

- Are comments correct for the specific cell and notebook context?
- Does it identify split leakage, fitted preprocessing outside folds, unsafe features, and evaluation mistakes?
- Does each recommendation cite applicable evidence and state uncertainty?
- Does inline guidance improve task success or time without distracting or overconfident advice?

### Existing files

- Copilot implementation: [`../../dclab_rnd/copilot/`](../../dclab_rnd/copilot/)
- Usage examples: [`../../dclab_rnd/copilot/examples/`](../../dclab_rnd/copilot/examples/) and [`../tabular-classification/notebooks/`](../tabular-classification/notebooks/)
- Product context: [`../../docs/DCLAB_MASTER_CONTEXT.md`](../../docs/DCLAB_MASTER_CONTEXT.md)

### Next evaluation

Connect the existing read-only copilot to VS Code notebook cells. Then evaluate it on blinded notebooks from different projects. Have ML reviewers label the issue, severity, evidence match, and recommended action per cell. Measure precision of actionable warnings, missed critical issues, false-alarm burden, evidence citation accuracy, and engineer task completion. Test unsaved notebook content and large notebooks; keep behavior read-only until edits have separate approval and rollback design.

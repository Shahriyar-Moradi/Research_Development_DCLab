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

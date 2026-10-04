# Prototype pack (September 22)

What is left of the first round of agent-assisted R&D prototypes. Everything else from that round was rebuilt in the shared code and removed from here; Git history keeps the originals.

| File | Why it is still here | Maintained version |
|---|---|---|
| `notebook_copilot.py`, `build_rag_index.py` | `dclab_rnd/notebook_eval.py` imports them to score this first prototype on the frozen notebook cases, so the new copilot can be compared with it | `dclab_rnd/copilot/`, `dclab_rnd/evidence_index.py` |
| `demo_churn_model.ipynb` | Test notebook used by `tests/test_notebook_assist.py` | `dclab_rnd/notebook_assist.py` |
| `dclab_studio.html` | First sketch of the "dclab notebook" UI, the starting point for the agentic notebook | `docs/copilot_demo.html`, the Research Studio (`dclab_rnd/agentic/`) |

The plans, guides and SFT builders that used to live here are now in `docs/guides/` and `research/llm-fine-tuning/experiments/sft/`. Script defaults assume you run them from the repository root.

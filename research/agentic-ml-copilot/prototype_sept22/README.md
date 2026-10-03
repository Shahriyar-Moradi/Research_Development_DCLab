# Prototype pack (September 22)

The first round of agent-assisted R&D prototypes, kept as a historical record. Each idea here was later rebuilt and tested in the shared code; use the maintained version for new work.

| Prototype file | What it explored | Maintained version |
|---|---|---|
| `notebook_copilot.py`, `grounded_review.py`, `demo_churn_model.ipynb` | Notebook review with cited rules | `dclab_rnd/copilot/` and `dclab_rnd/notebook_assist.py` |
| `build_rag_index.py`, `dclab_knowledge_mcp_server.py` | TF-IDF retrieval over rules and claims; an MCP server | `dclab_rnd/evidence_index.py`, `dclab_rnd/tools.py` |
| `build_sft_dataset*.py`, `sft_*_sample.jsonl`, `SFT_DATA_GUIDE.md` | Paired instruction data, versions 1 to 3 | `research/llm-fine-tuning/sft/build_sft_dataset_v3.py` |
| `dataset_expansion_pack.py` | Adapters for fraud, multiclass, time-series and text data | `dclab_rnd/expansion/` |
| `dclab_studio.html` | First "dclab notebook" UI concept | `docs/copilot_demo.html` and the Research Studio (`dclab_rnd/agentic/`) |
| `*.md` reviews and plans | Roadmap, architecture, verification and integration notes | `docs/guides/` |

`dclab_rnd/notebook_eval.py` still imports `notebook_copilot.py` and `build_rag_index.py` from here to score this prototype on the frozen notebook cases, so keep those two files in place. Script defaults assume you run them from the repository root.

# `docs/` — written reports and project context

## Guides (start here)

| Guide | For whom |
|---|---|
| [`guides/DCLAB_FIELD_NOTES.md`](guides/DCLAB_FIELD_NOTES.md) | Anyone. Plain-language lessons from all experiments, with the numbers |
| [`guides/DCLAB_RND_INTEGRATION_PLAN.md`](guides/DCLAB_RND_INTEGRATION_PLAN.md) | Product and engineering. How the R&D reaches DCLab and how the agent is verified |
| [`guides/AGENT_KNOWLEDGE_ARCHITECTURE.md`](guides/AGENT_KNOWLEDGE_ARCHITECTURE.md) | Agent builders. Retrieval vs fine-tuning vs tools, record design, usage patterns |
| [`guides/NOTEBOOK_COPILOT.md`](guides/NOTEBOOK_COPILOT.md) | Data scientists. What the copilot flags, why, and with what proof |
| [`guides/SFT_DATA_GUIDE.md`](guides/SFT_DATA_GUIDE.md) | ML engineers. Paired instructions, formats, training and evaluation |

## Reports and context

| File | What it is |
|---|---|
| [`reports/MASTER_4WAY_BENCHMARK_REPORT.md`](reports/MASTER_4WAY_BENCHMARK_REPORT.md) ([PDF](reports/MASTER_4WAY_BENCHMARK_REPORT.pdf)) | 4-quadrant benchmark (safe/unsafe × baseline/optimized), FT-Transformer vs GBDTs, General Pipeline |
| [`reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md`](reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md) ([PDF](reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf)) | Full R&D lifecycle: leakage forensics, 25 vs 46 feature dilution, 83-experiment optimization ladder |
| [`reports/EXTERNAL_MULTI_DATASET_REPORT.md`](reports/EXTERNAL_MULTI_DATASET_REPORT.md) | Cross-dataset benchmark on 10 public UCI tables |
| [`DCLAB_MASTER_CONTEXT.md`](DCLAB_MASTER_CONTEXT.md) | DCLab product constitution and master context. The R&D repository is a supporting evidence subsystem of that product |

Image links inside the reports are relative to `docs/reports/`, so they render on GitHub and
in local Markdown viewers. Plots themselves stay next to the code that produced them
(`research/tabular-classification/benchmark_outputs/`, `research/tabular-classification/optimized_safe_model/results/`, ...).

Regenerate the lifecycle PDF from its Markdown source (needs `markdown-it-py` and Chrome):

```bash
make report-pdf
# or: CHROME_BIN=/path/to/chrome python scripts/generate_pdf.py
```

Generated research memory (knowledge base, registry, field guide) lives in
[`evidence/knowledge/`](../evidence/knowledge/) and is rebuilt by `make rd-sync` and `make master-guide`.

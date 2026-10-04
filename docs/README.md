# `docs/` — written reports and project context

## Guides (start here)

| Guide | For whom |
|---|---|
| [`guides/DCLAB_FIELD_NOTES.md`](guides/DCLAB_FIELD_NOTES.md) | Anyone. Plain-language lessons from all experiments, with the numbers |
| [`guides/DCLAB_RND_INTEGRATION_PLAN.md`](guides/DCLAB_RND_INTEGRATION_PLAN.md) | Product and engineering. How the R&D reaches DCLab and how the agent is verified |
| [`guides/AGENT_KNOWLEDGE_ARCHITECTURE.md`](guides/AGENT_KNOWLEDGE_ARCHITECTURE.md) | Agent builders. Retrieval vs fine-tuning vs tools, record design, usage patterns |
| [`guides/NOTEBOOK_COPILOT.md`](guides/NOTEBOOK_COPILOT.md) | Data scientists. What the copilot flags, why, and with what proof |
| [`guides/SFT_DATA_GUIDE.md`](guides/SFT_DATA_GUIDE.md) | ML engineers. Paired instructions, formats, training and evaluation |

## Product context

| File | What it is |
|---|---|
| [`DCLAB_MASTER_CONTEXT.md`](DCLAB_MASTER_CONTEXT.md) | DCLab product constitution and master context. The R&D repository is a supporting evidence subsystem of that product |
| [`copilot_demo.html`](copilot_demo.html) | The notebook copilot's review of the leaky demo notebook (`make copilot-demo`) |
| [`product-demo/`](product-demo/) | Clickable demo of the final DCLab R&D product with a blueprint of what exists and what is missing (`make product-demo`) |

## Where the research reports went

Written reports now live with the research idea they belong to, under `research/<idea>/reports/`. The tabular reports (master 4-way benchmark, classification and optimization lifecycle, external multi-dataset benchmark) are in [`research/tabular-classification/reports/`](../research/tabular-classification/reports/). Each idea's `INDEX.md` lists all of its reports.

Generated research memory (knowledge base, registry, field guide) lives in [`evidence/knowledge/`](../evidence/knowledge/) and is rebuilt by `make rd-sync` and `make master-guide`.

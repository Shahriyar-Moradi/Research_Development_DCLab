# `docs/` — written reports and project context

| File | What it is |
|---|---|
| [`reports/MASTER_4WAY_BENCHMARK_REPORT.md`](reports/MASTER_4WAY_BENCHMARK_REPORT.md) ([PDF](reports/MASTER_4WAY_BENCHMARK_REPORT.pdf)) | 4-quadrant benchmark (safe/unsafe × baseline/optimized), FT-Transformer vs GBDTs, General Pipeline |
| [`reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md`](reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md) ([PDF](reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf)) | Full R&D lifecycle: leakage forensics, 25 vs 46 feature dilution, 83-experiment optimization ladder |
| [`reports/EXTERNAL_MULTI_DATASET_REPORT.md`](reports/EXTERNAL_MULTI_DATASET_REPORT.md) | Cross-dataset benchmark on 10 public UCI tables |
| [`DCLAB_MASTER_CONTEXT.md`](DCLAB_MASTER_CONTEXT.md) | DCLab product constitution and master context. The R&D repository is a supporting evidence subsystem of that product |

Image links inside the reports are relative to `docs/reports/`, so they render on GitHub and
in local Markdown viewers. Plots themselves stay next to the code that produced them
(`benchmark_outputs/`, `optimized_safe_model/results/`, ...).

Regenerate the lifecycle PDF from its Markdown source (needs `markdown-it-py` and Chrome):

```bash
make report-pdf
# or: CHROME_BIN=/path/to/chrome python scripts/generate_pdf.py
```

Generated research memory (knowledge base, registry, field guide) lives in
[`knowledge/`](../knowledge/) and is rebuilt by `make rd-sync` and `make master-guide`.

# `data/` — every dataset in one place

| Folder | What it holds | In git? |
|---|---|---|
| `project/` | DCLab project data: HyperAck orders and Telco churn (CSV) | yes |
| [`public/`](public/) | 10 public UCI datasets as cached parquet (`X.parquet`, `y.parquet`, `meta.json`) plus `index.json` | yes |
| `downloads/` | Public data fetched by the expansion campaign (fraud, letters, bike sharing, reviews) | no, re-fetched by `make expansion` with pinned SHA-256 |

## Project datasets

| Path | Rows | Target | Used by |
|---|---:|---|---|
| `project/hyperack/hyper_ackt-dataset.csv` | 11,118 | `hyper_ack` (0/1) | [`research/tabular-classification/`](../research/tabular-classification/), `general_pipeline/`, `dclab_rnd.agentic`, [`research/tabular-foundation-models/`](../research/tabular-foundation-models/) |
| `project/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7,043 | `Churn` (Yes/No) | [`research/churn-prediction/`](../research/churn-prediction/) (via `dclab_rnd.churn_suite`), `dclab_rnd.agentic`, [`research/tabular-foundation-models/`](../research/tabular-foundation-models/) |

Rules:

- Never edit these files in place. Experiment provenance stores their SHA-256 hashes, so a
  changed file invalidates every recorded run and blocks recipe replay.
- `final_customer_fare` and `final_biker_fare` in the HyperAck table are post-outcome
  fields. They are allowed only in the explicitly "unsafe" suites and are blocked for all
  new agentic experiments.
- `customerID` in the Telco table is an identifier and is never used as a feature.

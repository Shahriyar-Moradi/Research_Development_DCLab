# Tabular foundation models (TabPFN)

**Status:** active · results need a leakage-safe rerun on HyperAck

## Research question

Can a pretrained tabular foundation model (TabPFN v2/v3) match or beat tuned gradient boosting with no tuning, and at what cost in latency and data-size limits?

## What is here

`tabpfn/` holds the training script, a benchmark against trees, sklearn models and lucidrains TabTransformer/FT-Transformer, the local TabPFN demo notebook, and `results/`.

## Results so far

| Dataset | Best model | ROC-AUC | Next best |
|---|---|---:|---|
| Telco churn | TabPFN (thinking mode) | 0.850 | Logistic regression 0.841, XGBoost 0.841, TabPFN-2 0.841 |
| HyperAck | TabPFN (thinking mode) | 0.981 ⚠ | Histogram boosting 0.973 ⚠ |

> **Warning: the HyperAck rows are leakage-inflated.** The HyperAck loader (`load_part1` in `train_tabpfn.py`) keeps `final_customer_fare` and `final_biker_fare`, which are only known after the order outcome. Compare with the honest safe champion of 0.9455 in [tabular-classification](../tabular-classification/). The Telco rows are unaffected. A rerun without the fares is the next step for this track.

## How to run

Run from inside the folder; TabPFN v3 and thinking mode need `TABPFN_TOKEN` in the repository-root `.env`.

```bash
pip install -r requirements/base.txt   # includes tabpfn, tabpfn-client, tab-transformer-pytorch
cd research/tabular-foundation-models/tabpfn
python train_tabpfn.py                              # TabPFN-2 when no token is set
python train_tabpfn.py --version v3
python benchmark_tabpfn.py                          # TabPFN vs trees / sklearn / TabTransformer
python benchmark_tabpfn.py --only-tab-transformer   # TabTransformer + FT-Transformer only
```

## Conclusions so far

On Telco churn, TabPFN in thinking mode gave the best ranking (0.850) with no tuning, about +0.009 over logistic regression. Whether that holds on leakage-safe HyperAck is still open.

# Research tracks

Each research idea has its own folder. A track's README states the question, the prediction contract, the experiments, how to run them, and what was learned.

## Active

| Track | Question | Headline so far |
|---|---|---|
| [tabular-classification](tabular-classification/) | Most accurate *honest* classifier for tabular business data (HyperAck + 10 public datasets) | Safe champion ROC-AUC 0.9455; leakage was the largest effect |
| [churn-prediction](churn-prediction/) | Reliable churn ranking; do deep tabular models help? | Logistic regression 0.850 matched or beat boosting and transformers |
| [tabular-foundation-models](tabular-foundation-models/) | Can TabPFN beat tuned boosting without tuning? | Best on Telco churn (0.850); HyperAck rerun needed without leaky fares |
| [llm-fine-tuning](llm-fine-tuning/) | Can a small fine-tuned model reason like a large one with retrieval? | 326 evidence-grounded examples, trainer and evaluator ready |
| [ml-methodology](ml-methodology/) | Which workflow steps change the truth of a result? | 70 campaign experiments, 6 measured pitfalls |
| [agentic-ml-copilot](agentic-ml-copilot/) | Can an agent review and run ML work, with proof? | Copilot, agent tools, critic gate and verification built |

## Planned

| Track | Question |
|---|---|
| [data-science-foundations](data-science-foundations/) | Which EDA and data-cleaning habits prevent later modeling errors? |
| [graph-neural-networks](graph-neural-networks/) | When do GNNs beat tabular models with graph features, without graph leakage? |
| [time-series-forecasting](time-series-forecasting/) | Which forecasters win under honest forward-in-time validation? (seed result from the expansion campaign) |
| [nlp-and-text](nlp-and-text/) | When does text add signal, and how far do simple models get? (seed result) |
| [anomaly-and-fraud-detection](anomaly-and-fraud-detection/) | Methods, metrics and thresholds for rare events (seed result) |
| [recommender-systems](recommender-systems/) | Recommendation under time-ordered evaluation |
| [causal-inference-and-experimentation](causal-inference-and-experimentation/) | From "what will happen" to "what happens if we act" |
| [computer-vision](computer-vision/) | Accuracy per unit of compute on business image tasks |
| [mlops-and-deployment](mlops-and-deployment/) | Keeping a validated model valid in production |

## Add a new idea

```bash
make new-track NAME=my-new-idea TITLE="My new idea" PREFIX=MNI
```

This creates `research/my-new-idea/` from [`_template/`](_template/README.md), with a README (question, prediction contract, experiment log, result schema) and empty `src/`, `notebooks/`, `results/` and `reports/` folders. If the track already has a planned README, its notes are kept. Then add the track to the tables above.

## Conventions

- **What stays outside `research/`:** data in `data/` and `external_data/`, cross-track evidence campaigns in `campaigns/`, generated knowledge in `knowledge/`, and shared code in `dclab_rnd/` and `general_pipeline/`.
- **One result file per experiment.** Write it in the schema in the template, so the evidence index, critic gate and SFT builder can learn from it. Run `make knowledge` after adding results.
- **Prediction contract first.** No score is recorded before the prediction moment and the forbidden columns are written down.
- **Review your notebooks** with `python -m dclab_rnd.copilot review NOTEBOOK.ipynb` before sharing results.

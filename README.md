# DCLab Research & Development

The research arm of DCLab. Its job is to find out, with measured evidence, how to build machine-learning models that are *right*, not just high-scoring. Then it turns that evidence into tools DCLab's agent and its users can act on, with proof.

New here? Read the plain-language [field notes](docs/guides/DCLAB_FIELD_NOTES.md) first (no ML background needed), then the [research tracks](research/README.md).

---

## Repository layout

```
.
├── research/            ← one folder per research idea (start here)
│   ├── tabular-classification/       HyperAck + 10 public datasets (most mature)
│   ├── churn-prediction/             Telco churn: campaign, deep tabular models, baselines
│   ├── tabular-foundation-models/    TabPFN benchmarks
│   ├── llm-fine-tuning/              SFT corpus, LoRA trainer, evaluator
│   ├── ml-methodology/               index of the evidence campaigns
│   ├── agentic-ml-copilot/           index of the copilot, agent tools, Research Studio
│   ├── evaluation-and-trust/, workflow-model/, workflow-actions/   assistant evaluation and workflow models
│   ├── graph-neural-networks/, temporal-gnn/, vision-scene-graphs/, driving-maps/, …   planned tracks
│   └── _template/                    template for new tracks
├── campaigns/           ← cross-track evidence: 50-experiment workflow campaign, task-type
│                          expansion, pitfalls, agent verification (one JSON per experiment)
├── knowledge/           ← GENERATED research memory: registry, knowledge base, field guide,
│                          rules, evidence index (knowledge/rag/records.jsonl)
├── dclab_rnd/           ← shared Python package: control plane, campaign engines, evidence index,
│                          critic gate, notebook copilot, agent tools, Research Studio
├── general_pipeline/    ← shared tabular modeling pipeline (11 models, feature playbook)
├── data/                ← project datasets (HyperAck, Telco)
├── external_data/       ← cached public UCI datasets (parquet)
├── docs/                ← guides and written reports
├── scripts/             ← utilities (new research track, PDF report)
├── tests/               ← tests for everything above
└── Makefile             ← every workflow has a target: run `make help`
```

The rule of thumb: **ideas live in `research/`, shared infrastructure lives at the root.** Do not move or rename files under any `results/` folder; the evidence registry finds them by path.

---

## Install

```bash
git clone git@github.com:Shahriyar-Moradi/Research_Development_DCLab.git
cd Research_Development_DCLab

# 1) ML environment: experiments, campaigns, copilot, tests (Python 3.10+)
python -m venv .venv
.venv/bin/pip install -r requirements.txt pyarrow catboost

# 2) Optional agent environment for the Research Studio (Python 3.12 or 3.13)
uv venv --python 3.13 .venv-agent
uv pip install --python .venv-agent/bin/python -r requirements-agent.lock.txt

# 3) Secrets go in a local .env file (never committed)
#    OPENAI_API_KEY=...     for LLM critic reviews and the Research Studio
#    TABPFN_TOKEN=...       for TabPFN v3 (optional)
```

CI uses only `requirements-ci.txt` (numpy, pandas, scikit-learn, pyarrow). That is enough for the tests, the evidence checks, the copilot, the agent tools and the evidence index.

## Check that everything works

```bash
make help        # list every command with a one-line description
make rd-check    # tests + record validation + every "is the generated knowledge current?" check
```

`make rd-check` takes about 10 seconds and should end with every line reporting current or passed.

---

## How to use it

### Review a notebook for methodology mistakes

```bash
python -m dclab_rnd.copilot review path/to/notebook.ipynb                     # report in the terminal
python -m dclab_rnd.copilot review path/to/notebook.ipynb --html review.html  # notes beside each cell
python -m dclab_rnd.copilot review path/to/notebook.ipynb --annotate out.ipynb
python -m dclab_rnd.copilot review path/to/notebook.ipynb --fail-on high      # exit code 1 for CI
```

Each note says what is wrong, how to fix it, and links to the rule and the measured precedent behind it. The original notebook is never modified or executed. Try it on the demo: `make copilot-demo`, then open `docs/copilot_demo.html`. Guide: [Notebook copilot](docs/guides/NOTEBOOK_COPILOT.md).

### Ask the evidence a question

```bash
python -m dclab_rnd.evidence_index search "is call duration safe as a feature" --dataset bank_marketing
python -m dclab_rnd.evidence_index search "oversampling before split" --type pitfall
python -m dclab_rnd.tools call get_record '{"record_id": "LEAK-bank_marketing"}'
```

### Check a new dataset for leakage before modeling

```bash
python -m dclab_rnd.tools call audit_columns '{"path": "data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv", "target": "Churn"}'
```

Flags are review requests. Only the moment each value is created, relative to the prediction, confirms a leak.

### Give an LLM agent the DCLab tools

```bash
python -m dclab_rnd.tools list
python -m dclab_rnd.tools schemas --style openai      # or --style anthropic
pip install mcp && python -m dclab_rnd.tools mcp      # serve them over the Model Context Protocol
```

Guide: [Agent knowledge architecture](docs/guides/AGENT_KNOWLEDGE_ARCHITECTURE.md).

### Run experiments

| Goal | Command |
|---|---|
| Fast safe HyperAck smoke run plus evidence refresh | `make rd-smoke` |
| Any model through the shared pipeline | `.venv/bin/python general_pipeline/run_experiments.py --mode safe --optimization optimized --model lightgbm` |
| 50-experiment workflow campaign (resumable) | `make rd-campaign-plan`, then `.venv/bin/python -m dclab_rnd campaign run --quick` |
| Task-type expansion: fraud, multiclass, time series, text | `make expansion` (downloads public data to `data/external/`) |
| Re-measure the six notebook pitfalls | `make pitfalls` |
| Telco churn campaign | `make churn-run` |
| Track-specific experiments | see each track's README under [`research/`](research/README.md) |

After any new result, run `make rd-sync` and `make knowledge` so the registry, evidence index and SFT corpus include it; `make rd-check` fails until you do.

### Use the Research Studio (agentic, needs `OPENAI_API_KEY`)

```bash
make agent-serve        # http://127.0.0.1:8765
make agent-hyperack     # 4 agent-chosen experiments on HyperAck
make agent-churn        # 4 agent-chosen experiments on Telco churn
```

LangGraph runs the loop and NOOA specialist agents plan and critique. A deterministic worker owns data, training and metrics, so generated code is never executed. Runs are saved under the ignored `agent_runs/` folder.

### Build training data for a small model

```bash
make sft-v3                                                           # rebuild the corpus
python research/llm-fine-tuning/sft/eval_sft.py score --reference     # sanity check
```

Guide: [SFT data guide](docs/guides/SFT_DATA_GUIDE.md). Training (`train_lora.py`) needs a GPU.

### Start a new research idea

```bash
make new-track NAME=graph-neural-networks TITLE="Graph neural networks" PREFIX=GNN
```

Planned tracks already have a page with the question, field-specific leakage traps, first experiments and datasets; the command adds the working folders and keeps those notes. See [research/README.md](research/README.md).

---

## Headline results

| Finding | Evidence |
|---|---|
| Leakage was the biggest effect: post-outcome columns inflated ROC-AUC by +0.035 (HyperAck), +0.18 (bank marketing) and +0.20 (online shoppers) | [tabular-classification](research/tabular-classification/), `campaigns/model_building_50_v1` |
| Honest HyperAck champion: ROC-AUC 0.9455 (soft vote ExtraTrees + LightGBM + XGBoost), against 0.9793 with leaked fares | [tabular-classification](research/tabular-classification/) |
| No universal best algorithm: ExtraTrees led 7 of 10 campaign datasets, logistic regression won churn, an ensemble won HyperAck | [Field notes](docs/guides/DCLAB_FIELD_NOTES.md) |
| Raw features were the best recipe on 7 of 10 datasets; 46 engineered features did worse than 25 on HyperAck | `campaigns/model_building_50_v1` |
| Oversampling before the split inflated ROC-AUC by up to +0.30; scaling before the split by about 0.0001 | [`campaigns/pitfalls_v1`](campaigns/pitfalls_v1/PITFALLS_REPORT.md) |
| Random K-fold looked about 1.9× better than time-ordered CV on a forecasting task | [`campaigns/expansion_v1`](campaigns/expansion_v1/CAMPAIGN_REPORT.md) |
| 10 of 157 LLM critic objections were disproved by the recorded numbers | `python -m dclab_rnd.critic_gate` |

**Non-negotiable:** never deploy a model that uses information unavailable at the prediction moment (for example `final_customer_fare`, `final_biker_fare`, `duration`).

## Documentation

| Read this | For |
|---|---|
| [Field notes](docs/guides/DCLAB_FIELD_NOTES.md) | Anyone: the lessons in plain language |
| [Research tracks](research/README.md) | Where each idea lives and how to add one |
| [Integration plan](docs/guides/DCLAB_RND_INTEGRATION_PLAN.md) | How the R&D reaches the DCLab product and how the agent is verified |
| [Agent knowledge architecture](docs/guides/AGENT_KNOWLEDGE_ARCHITECTURE.md) | Retrieval vs fine-tuning vs tools |
| [Notebook copilot](docs/guides/NOTEBOOK_COPILOT.md) | What the copilot flags and why |
| [SFT data guide](docs/guides/SFT_DATA_GUIDE.md) | Training data for a small model |
| [Reports](docs/README.md) | Full written reports (Markdown + PDF) and product context |
| [Tabular playbook rule](.cursor/rules/tabular-classification-playbook.mdc) | Standard operating procedure for tabular classification |

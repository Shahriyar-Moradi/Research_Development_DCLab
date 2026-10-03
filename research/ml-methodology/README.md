# ML methodology (evidence campaigns)

**Status:** active · code lives in shared folders

## Research question

Which steps of the model-building workflow actually change the truth of a result, across many datasets and task types?

## Why it matters for DCLab

These campaigns are the answer key DCLab's agent and copilot are tested against.

## Where the work lives

| Campaign | Folder | What it measured |
|---|---|---|
| 50-experiment workflow campaign | [`campaigns/model_building_50_v1`](../../campaigns/model_building_50_v1/CAMPAIGN_REPORT.md) | 10 datasets × 5 stages (understanding, leakage, features, model screen, optimization + holdout) |
| Task-type expansion | [`campaigns/expansion_v1`](../../campaigns/expansion_v1/CAMPAIGN_REPORT.md) | Imbalanced fraud, 26-class, time-series, text + tabular |
| Pitfalls | [`campaigns/pitfalls_v1`](../../campaigns/pitfalls_v1/PITFALLS_REPORT.md) | Cost of six common notebook mistakes |
| Agent verification | [`campaigns/agent_verification_v1`](../../campaigns/agent_verification_v1/VERIFICATION_REPORT.md) | Blind replay of the leakage auditor |

Engines: `dclab_rnd/science.py` and `dclab_rnd/campaign.py` (binary campaign), `dclab_rnd/expansion/` (task-aware), `dclab_rnd/pitfalls.py`. They stay in the shared package because the evidence index, critic gate and SFT builder read `campaigns/*/results/` directly.

## How to run

```bash
make rd-campaign-plan && .venv/bin/python -m dclab_rnd campaign run --quick   # 50-experiment campaign
make expansion                                                               # 20 task-type experiments (downloads data)
make pitfalls                                                                # six measured mistakes
make verify-auditor                                                          # blind leakage-auditor replay
make knowledge                                                               # fold new results into the index and SFT corpus
```

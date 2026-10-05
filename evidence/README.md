# `evidence/` — what we measured and what we learned

| Folder | Contents | Edited by |
|---|---|---|
| [`campaigns/`](campaigns/) | One JSON result per experiment, plus each campaign's report, manifest and claim memory | The campaign engines (`dclab_rnd`) |
| [`knowledge/`](knowledge/) | Research memory derived from all results: registry, knowledge base, field guide, rules, workflow blocks, evidence index | **Generated only** (`make rd-sync`, `make master-guide`, `make knowledge`) |

## Campaigns

| Campaign | Experiments | Report |
|---|---|---|
| `model_building_50_v1` | 50 (10 public datasets × 5 workflow stages) | [CAMPAIGN_REPORT.md](campaigns/model_building_50_v1/CAMPAIGN_REPORT.md) |
| `expansion_v1` | 20 (fraud, 26-class, time series, text + tabular) | [CAMPAIGN_REPORT.md](campaigns/expansion_v1/CAMPAIGN_REPORT.md) |
| `pitfalls_v1` | 6 measured notebook mistakes | [PITFALLS_REPORT.md](campaigns/pitfalls_v1/PITFALLS_REPORT.md) |
| `agent_verification_v1` | Blind leakage-auditor replay | [VERIFICATION_REPORT.md](campaigns/agent_verification_v1/VERIFICATION_REPORT.md) |

Track-specific results (for example HyperAck's 83 experiments) stay inside their track under `research/`. The registry in `knowledge/` indexes both.

## Caveat: category codes inside older derived features

The cached UCI tables store categories as factorized codes (bank_marketing: May=0, Jun=1, …). Until rule DCLAB-R11 was enforced in the feature code, the recipes treated those codes as quantities. `logs` built `log1p_month`, `ratios` divided by `month`, and interactions, KMeans clusters and quantile bins used codes too. Recipes now leave the categorical columns declared in [`dclab_rnd/agentic/catalog.py`](../dclab_rnd/agentic/catalog.py) out of every derived feature ([`dclab_rnd/categoricals.py`](../dclab_rnd/categoricals.py)). Records produced before that stay as they are:

- `model_building_50_v1`: the feature ladders of the seven datasets with categorical columns (EXP-003, -008, -018, -023, -028, -033, -043), and the later stages that kept a derived recipe: bank_marketing `ratios` (EXP-009, EXP-010, whose top features include `log1p_poutcome` and `log1p_month`) and german_credit `interactions` (EXP-029, EXP-030).
- The playbook ladders under `research/tabular-classification/experiments/external_projects/*/results/ladder/` for the same datasets.
- DCLab notebook projects whose `log_numeric` or `poly2` recipe ran before the change.

breast_cancer, spambase, wine_quality and `expansion_v1` have no category codes and are unaffected. To re-measure, rerun into new result files.

## Rules

- Result files are immutable records. Rerun an experiment into a new file; never edit a result by hand.
- After adding results, run `make rd-sync` and `make knowledge`. `make rd-check` fails while generated files are stale.
- Provenance inside older results records the paths that existed when they ran, which is why some show pre-reorganization folder names.

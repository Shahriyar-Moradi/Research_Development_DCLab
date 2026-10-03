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

## Rules

- Result files are immutable records. Rerun an experiment into a new file; never edit a result by hand.
- After adding results, run `make rd-sync` and `make knowledge`. `make rd-check` fails while generated files are stale.
- Provenance inside older results records the paths that existed when they ran, which is why some show pre-reorganization folder names.

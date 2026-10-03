# Using the R&D inside DCLab: now, next, and how we verify it

DCLab's promise to data scientists and ML engineers is to do the notebook work agentically and to be *right*, with proof. This repository is where "right" gets defined and tested. This plan says how each R&D output reaches the product, in what order, and which test must pass before a capability is advertised.

## What the R&D has produced

| Asset | Where | Product use |
|---|---|---|
| 50-experiment campaign (10 datasets × 5 stages) | `evidence/campaigns/model_building_50_v1/` | Precedents for every workflow stage |
| Task-type expansion (fraud, multiclass, time series, text + tabular) | `evidence/campaigns/expansion_v1/` | Precedents beyond binary classification |
| Pitfalls campaign (6 measured mistakes) | `evidence/campaigns/pitfalls_v1/` | Calibrated severity and proof for copilot notes |
| 22 rules, 10 workflow blocks | `evidence/knowledge/` | The agent's operating rules |
| Evidence index | `evidence/knowledge/rag/records.jsonl` | Retrieval for every agent answer |
| Critic gate | `dclab_rnd/critic_gate.py` | Filters LLM critique before it becomes memory |
| Notebook copilot | `dclab_rnd/copilot/` | Inline review with "Show proof" |
| Agent tools + MCP server | `dclab_rnd/tools.py` | One tool surface for the product agent |
| SFT v3 corpus + trainer + evaluator | `research/llm-fine-tuning/sft/` | A future small in-house model |
| Blind auditor replay | `evidence/campaigns/agent_verification_v1/` | Regression test for leakage detection |

## Phase 1 — now: retrieval and tools with a hosted model

No training needed.

1. **Ship the tool surface.** Register `dclab_rnd.tools` in the product's MCP server (or call `call_tool` in-process). The product agent gets `search_evidence`, `get_record`, `plan_next_stage`, `review_code`, `review_notebook` and `audit_columns`.
2. **Ground every answer.** The agent's system prompt requires citing record IDs and the five-part answer. "Show proof" in the UI is `get_record(id)` rendered.
3. **Copilot beside cells.** On each cell run or save, call `review_code` on the cell (and `review_notebook` on save). Show notes in the margin, collapsed by default, expandable to the proof.
4. **Leakage gate on new data.** When a user uploads data, run `audit_columns`, show the flagged columns, and *ask for the prediction moment* before any training. Never treat a clean scan as safety: the blind replay shows heuristics miss leaks that look like ordinary numbers.

**Done when:** the copilot gives zero medium or high notes on the clean demo, flags every planted mistake in the leaky demo, and the tests in `tests/test_knowledge_layer.py` pass in the product's CI.

## Phase 2 — next: the guarded workflow loop

The agent runs the five stages for the user and iterates. This is the riskiest capability, because a loop that only sees the metric will happily "improve" through leakage. The largest score jumps ever recorded here (+0.13 to +0.18 ROC-AUC) were leakage, not skill.

Build it as repeated instances of the pattern the Studio already uses:

```text
deterministic controller
  └─ for each stage:
       plan_next_stage ─► deterministic executor runs the stage
                       ─► LLM writes a schema-validated critique with cited records
                       ─► critic_gate recomputes rules, drops disproved critique
                       ─► result + surviving claims become a new evidence record
  └─ stop rule owned by the controller (budget, no improvement beyond fold noise, or a leakage alarm)
```

Constrain the action menu to what the R&D validated:

- parameters only inside the search spaces of the optimization stage,
- model families only from the screened shortlist,
- feature moves only one rung up or down the feature ladder,
- no invented features mid-loop without a leakage audit of their source columns.

Every iteration writes a record with Goal, Change, Reason, Result and Cost. The audit trail the user sees and the knowledge the R&D learns from are the same file.

## How we verify the agent before users see it

| Test | What it proves | Status |
|---|---|---|
| **Blind auditor replay** (`python -m dclab_rnd.tools verify-auditor`) | The auditor finds known leaks without being told | Implemented. Found 11 of 18 known leaks across 7 datasets, at least one in 6 of them. The misses are documented and drive the "always ask for the contract" rule |
| **Copilot demos** (`tests/test_knowledge_layer.py`) | Planted mistakes are caught, a clean notebook stays clean | Implemented, in CI |
| **Critic gate** (`python -m dclab_rnd.critic_gate`) | LLM critique contradicted by numbers never reaches memory or training | Implemented: 10 of 157 challenges disproved |
| **Campaign replay** | The loop, run on the 10 campaign datasets *without* their results, excludes the same leakage columns and lands within the campaign's holdout confidence interval | Next. Uses `evidence/campaigns/model_building_50_v1` as the answer key |
| **Red-team with unsafe data** | Given the "unsafe" ablation data, the loop flags the suspicious jump instead of reporting it as progress | Next. Uses the unsafe results already in `research/tabular-classification/external_projects/*/results/ladder/` |
| **Widening gate** | Each new capability (a new action, a new task type) is advertised only after a campaign validates it | Policy |

## What we can honestly claim

Every claim the product makes must be as checkable as its copilot notes.

| Claim | Honest today? |
|---|---|
| "Backed by 50 controlled experiments across 10 real public datasets" | Yes |
| "Measured the cost of 6 common notebook mistakes" | Yes |
| "Covers binary, imbalanced fraud, multiclass, time-series and text + tabular problems" | Yes, after the expansion campaign (20 experiments, one dataset per type) |
| "50 Kaggle challenges solved" | **No.** The campaign used UCI datasets, and one public dataset per new task type. To make this true, run Kaggle competition datasets through the same five stages and compare against the public leaderboard |
| "Finds all leakage" | **No.** It finds strong proxies and known columns; the decision-time contract is still the user's input |

## Phase 3 — later: a small in-house model

When volume or deployment needs justify it, fine-tune a small model on `research/llm-fine-tuning/sft/out_v3/` with `research/llm-fine-tuning/sft/train_lora.py`, score it with `research/llm-fine-tuning/sft/eval_sft.py` against the base model on held-out datasets, and keep retrieval. See [SFT_DATA_GUIDE.md](SFT_DATA_GUIDE.md).

## Keeping it alive

```bash
make rd-check          # tests + evidence validation + index and SFT freshness
make knowledge         # rebuild evidence index and SFT v3 after new results
make copilot-demo      # regenerate the demo review page
```

New experiments become new records automatically. Re-run the critic gate and the blind replay whenever the critic prompt, the auditor or the detectors change; neither number may get worse without a written reason.

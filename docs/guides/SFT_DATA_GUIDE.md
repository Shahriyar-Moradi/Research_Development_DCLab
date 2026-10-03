# SFT data guide: turning DCLab evidence into training data

This guide answers three questions:

1. What is a "paired instruction" example, and what makes one good?
2. How do the DCLab R&D outputs become such examples?
3. How does a fine-tuning job train on them directly, and how do we know it worked?

Everything here is implemented in [`sft/`](../../sft/). Nothing in v1 was removed; v3 is added beside it.

---

## 1. What a good training example looks like

A supervised fine-tuning (SFT) example is one conversation: an instruction (the user turn) and the answer we want the model to learn (the assistant turn). The model learns to produce the answer *given the instruction*. So the instruction must contain everything the answer depends on.

**Bad (v1 style).** The question points to an ID the model has never seen:

```text
User: What did experiment EXP-023 on 'credit_default' find during the feature engineering stage?
Assistant: Use the raw feature recipe ... 0.7317 mean training-CV ROC-AUC with about 23 features.
```

The model cannot reason its way to that answer. It can only memorize "EXP-023 → raw → 0.7317". That teaches recall of trivia, which goes stale the moment a new experiment runs.

**Good (v3 style).** The instruction carries three context blocks pulled from real files:

```text
### Dataset
Dataset `bank_marketing`: Bank Marketing. Predict term-deposit subscription from phone marketing
campaign data. Facts: task type binary, 45,211 rows, 16 features, positive rate 11.7%.
Decision-time contract: Immediately before a marketing call. Duration is post-call and forbidden. ...

### Stage
Stage: Feature engineering ablation (WF-06). Climb a feature ladder ... Choose the SMALLEST feature
matrix whose mean CV score is within a small tolerance of the best ...

### What this run compared
6 feature recipes compared with lightgbm, 3-fold training CV: raw (15 features, ROC-AUC 0.7130±0.0129);
... ratios (38 features, ROC-AUC 0.7172±0.0165); ... Selected: ratios (+0.0042 ROC-AUC vs raw).

Explain what this run shows, how much to trust it, and what to do next.
```

The answer then follows one fixed structure: **Evidence → Interpretation → Decision → Risks → Next test**. The model learns a *way of reasoning from evidence*, which transfers to experiments it has never seen.

This shape has a name: **RAFT** (retrieval-augmented fine-tuning). You train on `(retrieved context + question) → answer`, which is exactly how the model will be prompted later, when the DCLab agent retrieves records from the evidence index and puts them in front of it.

---

## 2. From R&D outputs to examples

```text
campaigns/*/results/EXP-*.json ─┐
knowledge/model_building_rules  ├─► dclab_rnd.evidence_index  ─► self-contained records
knowledge/workflow_blocks       │       (dataset card + stage card + setup summary)
external_data/*/meta.json ──────┘
                                         │
              dclab_rnd.critic_gate ─────┤  drop critic claims that the numbers disprove
                                         ▼
                       sft/build_sft_dataset_v3.py
                                         │
             dedupe · strip paths/hashes · dataset-grouped split · 3 formats
                                         ▼
                                  sft/out_v3/
```

### Task families

| Task | What the model learns | Where the answer comes from |
|---|---|---|
| `explain_experiment` | Read a run and give the five-part answer | Claims, limitations, gated critic review |
| `critique_claim` | Be a skeptical reviewer of one claim | Critic challenges that survived the gate |
| `apply_selection_rule` | Apply a declared rule to numbers exactly | Deterministic recomputation |
| `leakage_judgment` | Decide exclude / review / proceed for a column | Decision-time policy + measured lift |
| `grounded_qa` | Answer from the right retrieved record and cite it, ignoring distractors | Index records; distractors are the *same stage on other datasets* (hard negatives) |
| `rule_reasoning` | Explain why a practice matters and how to check it | The 22 rules |
| `workflow_steps` | Order the steps of a workflow block and say why order matters | The 10 workflow blocks |

### Quality gates (why the data is cleaner than v1)

- **Critic gate.** The campaign's LLM critic is useful but not ground truth. `dclab_rnd.critic_gate` re-applies every stage's declared selection rule to its own numbers. All 30 recorded selections pass. But 10 of 157 critic challenges claim a rule violation that the recomputation disproves. Most came from one confusion: the feature ladder has a recipe literally *named* `selected`, and the critic mistook it for the recipe the rule selected. Those challenges are dropped, never used as answers. Run `python -m dclab_rnd.critic_gate` to see them.
- **No provenance noise.** Hashes, absolute paths and machine details never enter an example. A final scan rejects any example containing an absolute path or a 64-character hex string.
- **Deduplication** on the normalized user turn.
- **Dataset-grouped validation.** Whole datasets (`heart_disease`, `spambase`, `ecommerce_clothing_reviews`) are held out, so validation measures transfer to an unseen dataset, not memorization of one you trained on.

### Three output formats

All three hold the same examples:

| File | Format | Used by |
|---|---|---|
| `train.chat.jsonl` | `{"messages": [system, user, assistant]}` | OpenAI fine-tuning, TRL `SFTTrainer`, most tooling |
| `train.alpaca.jsonl` | `{"instruction", "input", "output"}` | Alpaca-style trainers |
| `train.sharegpt.jsonl` | `{"system", "conversations": [{"from", "value"}]}` | Axolotl, LLaMA-Factory |

Each line also has a `metadata` object (task, dataset, stage, experiment ID). Trainers ignore it, and the evaluator uses it.

```bash
make sft-v3                     # rebuild sft/out_v3/ from current evidence
python sft/build_sft_dataset_v3.py --check   # CI: fail if committed corpus is stale
```

---

## 3. Train on it directly

[`sft/train_lora.py`](../../sft/train_lora.py) is a complete LoRA job with Hugging Face TRL:

```bash
pip install "trl>=0.12" "peft>=0.13" "transformers>=4.46" datasets accelerate
python sft/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --out sft/runs/qwen15b-dclab
```

Choices and why:

- **LoRA rank 16, 3 epochs.** The corpus has a few hundred examples. The goal is format and habit, not memorizing numbers.
- **`assistant_only_loss=True`.** The user turns are long evidence blocks. Training on them would waste capacity on copying context.
- **Facts stay in retrieval.** Numbers change every time a campaign runs. Weights should learn *how* to reason; the evidence index supplies *what* is true today.

For a hosted model (for example OpenAI fine-tuning), upload `train.chat.jsonl` and `val.chat.jsonl` as they are. The `metadata` key is ignored by most providers. Strip it first if yours rejects unknown keys.

## 4. Know whether it worked

[`sft/eval_sft.py`](../../sft/eval_sft.py) scores any model on the held-out datasets without an LLM judge, because most tasks have a checkable core:

| Task | Check |
|---|---|
| `apply_selection_rule` | Names the option the rule actually selects |
| `leakage_judgment` | Says exclude / do-not-auto-delete / proceed, matching the label |
| `grounded_qa` | Cites the correct record ID and no distractor ID |
| `explain_experiment` | Contains all five sections |
| `critique_claim` | Raises gaps with severities |

```bash
python sft/eval_sft.py score --reference          # sanity check: reference answers score 1.0
python sft/eval_sft.py generate --model Qwen/Qwen2.5-1.5B-Instruct --adapter sft/runs/qwen15b-dclab --out preds.jsonl
python sft/eval_sft.py score --predictions preds.jsonl
```

Score the base model first, then the adapter. Keep the adapter only if it beats the base model on the held-out datasets. A canned one-line answer scores about 0.14, so the check discriminates.

## 5. When to fine-tune at all

You do **not** need fine-tuning to use this knowledge today. A capable hosted model with the evidence index and the agent tools (see [AGENT_KNOWLEDGE_ARCHITECTURE.md](AGENT_KNOWLEDGE_ARCHITECTURE.md)) already gives grounded, cited answers. Fine-tune a small model when you need one of these:

- lower cost or latency at high volume,
- on-premise or offline deployment,
- a fixed answer format the base model keeps breaking.

Even then, keep retrieval: the fine-tuned model reasons, and the index supplies today's facts.

# Evidence, retrieval, and later SLM training

**Status: SFT preparation exists; retrieval quality and training outcomes are not yet established.** This track connects R&D outputs to DCLab's assistant. Its central rule is that a proposal, an observed result, an interpretation, and a reusable principle are different kinds of records.

## Research questions

- Which evidence record is relevant to a question about dataset, workflow stage, or symptom?
- Can the assistant show the evidence, scope, counterexample, and uncertainty behind a recommendation?
- Which records are verified enough for retrieval or eventual training?
- Does retrieved context improve answers over no retrieval and metadata-filtered baselines?

## Existing material

- Rules, workflow blocks, and campaign records: [`../../knowledge/`](../../knowledge/) and [`../../campaigns/model_building_50_v1/agent_memory.jsonl`](../../campaigns/model_building_50_v1/agent_memory.jsonl)
- Experiment-level claim memory: [`../../campaigns/model_building_50_v1/agent_memory.jsonl`](../../campaigns/model_building_50_v1/agent_memory.jsonl)
- SFT preparation: [`../../sft/README.md`](../../sft/README.md) and [`../../sft/build_sft_dataset.py`](../../sft/build_sft_dataset.py)

## Suggested experiment

First compare a base model with no context, exact metadata-filtered evidence, and retrieved evidence on questions with known evidence IDs. Score citation correctness, answer correctness, scope fidelity, and unsupported-claim rate. Keep rejected and negative evidence retrievable. Only use reviewer-approved examples for training; split by dataset/source and workflow to prevent near-duplicate leakage.

For local commands, see the [main README](../../README.md#dclab-rd-paths). Retrieval or fine-tuning should not be described as reliable until evaluated against a fixed benchmark.

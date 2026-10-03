# Focused model for ML workflows

**Status: research proposal informed by existing SOPs and training-data preparation.** The goal is a small model that knows a narrow workflow deeply: what stage it is in, what evidence it needs, what valid next step follows, and when it should stop or ask a human.

## Research questions

- Can fine-tuning improve workflow-step selection and constraint memory over a prompted base model?
- Does structured state (stage, inputs, evidence, allowed transitions) reduce forgotten requirements?
- How well does the model handle missing information, exceptions, and workflows it was not trained on?
- Is any gain worth the data, compute, maintenance, and regression risk?

## First experiment

Before training a large model, establish a small held-out task set from ML workflow scenarios. Compare a base model with a workflow prompt, retrieved context, and an SFT model on stage selection, required-input recall, invalid transitions, leakage handling, evidence citation, and calibrated abstention. Split by complete dataset/problem family rather than by similar examples.

Existing SOPs: [`../../evidence/knowledge/MODEL_BUILDING_FIELD_GUIDE.md`](../../evidence/knowledge/MODEL_BUILDING_FIELD_GUIDE.md). Training data preparation: [`../llm-fine-tuning/sft/README.md`](../llm-fine-tuning/sft/README.md). No charter or existing fine-tuning file proves that a model has acquired reliable autonomous workflow competence.

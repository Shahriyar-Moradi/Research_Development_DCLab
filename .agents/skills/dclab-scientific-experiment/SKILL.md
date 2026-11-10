---
name: dclab-scientific-experiment
description: Propose, draft, debug, improve and evaluate autonomous ML experimentation under strict DCLab Part C scientific safeguards (leakage-safe, holdout locked, deterministic code gate).
---
# Autonomous Scientific ML Experimentation

Use this skill when exploring, refining, or tuning machine learning pipelines autonomously.

## Workflow

1. **Context & Hypothesis (`DraftOperator`):**
   - Identify the problem kind, target, primary metric, and baseline score.
   - Formulate an explicit hypothesis (e.g. testing an alternative inductive bias, interaction features, or loss function).
   - Enforce **Rule C1**: Never include features identified as post-decision or forbidden.

2. **Execution Guardrails:**
   - Run candidate pipelines strictly on training CV folds or isolated selection folds.
   - Enforce **Rule C2**: The holdout set is strictly locked and cannot be read during search.
   - Enforce **Rule C5**: Bounded execution timeout (max 30 seconds per trial).

3. **Deterministic Evaluation (`EvaluateOperator`):**
   - Apply **Rule C3**: Candidate configuration is accepted only if its metric improvement $\ge \text{min\_gain}$ (default $0.001$).
   - If performance does not beat baseline, reject and record the negative evidence.

4. **Error Recovery (`DebugOperator`):**
   - If execution fails (e.g., NaN values, matrix singularity, convergence errors), analyze the exception.
   - Propose a minimal surgical patch (imputation, ridge shrinkage, iteration scale) rather than abandoning the pipeline.

5. **Refinement (`ImproveOperator`):**
   - If accepted, refine hyperparameters (learning rate, tree depth, regularization) on validation diagnostics.

6. **Contrastive Record:**
   - Record the attempt, hypothesis, execution time, and outcome in `ExperimentRetrievalIndex` for cross-task reuse.

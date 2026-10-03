# Workflow-sized generation and action units

**Status: hypothesis requiring a controlled prototype.** The idea is to generate a coherent unit—such as a complete function, validated notebook-cell transformation, or pair of compatible workflow steps—in fewer fragile decisions than generating each line independently.

## Clarify what “larger token” means

There are at least two testable versions:

1. **Structured action output:** the model emits a normal token sequence representing a complete named action, such as `fit_preprocessor_inside_fold`, with typed inputs and outputs. Deterministic code validates and runs it.
2. **Learned macro tokens:** add dedicated vocabulary tokens for common operations or code blocks and train the model to use them. This changes the tokenizer/model vocabulary and requires retraining or compatible continued training.

One emitted token cannot magically contain arbitrary code with no decoding cost: the token still maps to a token ID and learned representation, while a safe executor must validate the expansion. Measure total generated tokens, latency, task success, repair rate, and correctness—not token count alone.

## First experiment

Use a small set of well-specified ML helper functions. Compare ordinary generation, constrained structured output, and (only if feasible) macro-token output. Evaluate compilation, tests, semantic equivalence, edge cases, token/latency cost, and unsupported assumptions. Begin with structured actions; treat learned macro vocabulary as a later research branch.

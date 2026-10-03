# Workflow transfer beyond data science

**Status: proposed research.** The long-term idea is to make DCLab's experiment-and-evidence approach useful for other technical workflows, without assuming that ML procedures themselves transfer unchanged.

## Separate reusable engine from domain knowledge

Potentially reusable mechanics include: a workflow graph, typed inputs/outputs, evidence ledger, versioned procedures, bounded tool calls, approvals, evaluation cases, and replay. Domain-specific content still needs its own experts, data, terminology, safety constraints, and success measures.

## First transfer test

After the ML workflow assistant has a stable benchmark, choose one non-safety-critical workflow with clear acceptance criteria (for example, a software QA checklist). Implement the same workflow representation and evidence log, then compare step completion, invalid actions, omitted required checks, time, and reviewer corrections against a baseline. Only expand to another domain if the core mechanics transfer and domain experts approve the content.

Autonomous driving has a separate, higher assurance bar; see [`../driving-maps/README.md`](../driving-maps/README.md).

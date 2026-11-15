<!-- Extracted from DCLab Hierarchical Planning V2 Additive Capability Pack (2026-11-05). -->
<!-- Mandatory: V1 acceptance gates first; do not replace or bypass deterministic scientific and safety guards. -->

### HP2-05 | Open-weight candidate model integration

**Objective:** Run the user’s fine-tuned Agentic ML model as a gateway-selectable engine for *bounded roles* first.

**Design:** Add no training code to V2 itself. Import a pinned adapter manifest from the independent Data/Model Program. Log base+adapter+tokenizer+template hashes and cost/latency. Use model-level fallback only for unresolved or high-complexity roles. Keep all tool/safety/verification contracts outside the LLM.

**Acceptance:** One-click rollback and shadow execution; identical function schemas to paid-model baseline; no novel authority granted to the model; bounded fallback cost. Full default promotion requires model-program gates.

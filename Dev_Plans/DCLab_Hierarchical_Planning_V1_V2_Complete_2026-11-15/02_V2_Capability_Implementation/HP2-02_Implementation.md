<!-- Extracted from DCLab Hierarchical Planning V2 Additive Capability Pack (2026-11-05). -->
<!-- Mandatory: V1 acceptance gates first; do not replace or bypass deterministic scientific and safety guards. -->

### HP2-02 | Structured proposals and evidence-based critique

**Objective:** Enable collaboration where disagreement is settled by a reproducible experiment or deterministic validator rather than persuasion.

**Protocol:**
1. `proposal`: hypothesis, dataset facts, planned metric, expected artifact, expected cost.
2. `critique`: precise fault claim, severity, evidence request, proposed discriminating test.
3. `experiment_request`: reference to a V1-valid leaf task and scoped permitted tools.
4. `verification`: observed metrics, provenance, decision, unresolved limitations.
5. `decision`: accepted/rejected/blocked with evidence references.

**Guardrails:** No unlimited chat loops. At most 2 critique rounds by default. Human intervention when intent/safety/science ambiguity persists. Reviewers cannot override deterministic fails. No hidden reasoning logs.

**Acceptance:** Reasoned disagreement on leakage, class imbalance, split methodology and poor metrics; zero false-completion acceptance; no extra holdout access.

<!-- Extracted from DCLab Hierarchical Planning V2 Additive Capability Pack (2026-11-05). -->
<!-- Mandatory: V1 acceptance gates first; do not replace or bypass deterministic scientific and safety guards. -->

### HP2-06 | Ablations, benchmark and release gate

**Arms:** `A: fixed workflow`, `B: accepted V1`, `C: V2 collaboration without parallel`, `D: V2 + parallel`, `E: V2 + experience memory`, `F: V2 + own fine-tuned adapter`.

**Evaluation:** matched cases, frozen tool registry, matched budgets/compute/harness, repeated stochastic seeds, separate safety/science/e2e scores, provenance and cost. Compare per capability, not only final total. No train/dev cases in sealed external benchmarks.

**Metrics:** validated completion, scientific correctness, tool choice/args validity, false completions, leakage/holdout incidents, recoverability, work duplication, wall-clock latency, API+GPU cost, quality per dollar.

**Release rule:** 0 observed critical safety violations in declared test suite, no material V1 regression, proven quality/cost tradeoff at preset uncertainty and sample size. Otherwise `NOT PROMOTED` or `INSUFFICIENT EVIDENCE`.

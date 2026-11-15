# DCLab Hierarchical Planning V1 and V2: Complete Documentation Pack

**Created:** 2026-11-15  
**Language:** English only  
**Status:** Historical plans and engineering specifications; these documents do **not** establish implementation completion.

## Purpose and decision

This package recovers the original DCLab Hierarchical Planning documents, separates each implementation step for registration and Codex handoff, and includes the harness/intelligence evaluation plans that should precede any untested architectural expansion. No source section in the V1 and V2 masters was discarded.

**Critical sequencing:** V1 TaskGraph, TaskExecutionPlan, guarded execution, verification, recoverability and evaluation gates must be stable before V2 implementation begins. V2 adds specialist collaboration, structured critique, controlled parallelism, verified memory and model adapters. It is not a rewrite of V1. Also, the current DCLab harness intelligence regression must be measured before assuming a more elaborate agent team improves results.

## Full original engineering documents

- [Hierarchical Planning V1: full Codex implementation pack](00_Master_Plans/DCLab_Hierarchical_Planning_V1_Codex_Implementation_Pack.md): comprehensive instructions, operating rules, existing system architecture and numbered engineering tasks, with TP-01 acceptance checklist.
- [Hierarchical Planning V2: full additive capability specification](00_Master_Plans/DCLab_Hierarchical_Planning_V2_Additive_Implementation_Plan.md): V1 entry criteria, six HP2 packages, data policy, artifact list, rollout order and Codex handoff.

## V1: individual executable engineering plans

- [R-01: Audit and finish the existing six-choice Home routing](01_V1_Implementation_Steps/R-01_Implementation.md)
- [TP-01A: Architecture and capability mapping](01_V1_Implementation_Steps/TP-01A_Implementation.md)
- [TP-01B: Minimal typed task contracts and frozen schemas](01_V1_Implementation_Steps/TP-01B_Implementation.md)
- [TP-01C: Deterministic graph validator, planning invariants and fingerprints](01_V1_Implementation_Steps/TP-01C_Implementation.md)
- [TP-01D: Deterministic solution-workflow-to-task adapter](01_V1_Implementation_Steps/TP-01D_Implementation.md)
- [TP-01E: Cross-environment tests, regression gate and TP-01 handoff](01_V1_Implementation_Steps/TP-01E_Implementation.md)
- [TP-02: Bounded model-assisted task decomposition](01_V1_Implementation_Steps/TP-02_Implementation.md)
- [TP-03: Per-task execution planner and plan gate](01_V1_Implementation_Steps/TP-03_Implementation.md)
- [TP-04: Sequential guarded execution and durable task progression](01_V1_Implementation_Steps/TP-04_Implementation.md)
- [TP-05: Independent task verification, recovery and selective invalidation](01_V1_Implementation_Steps/TP-05_Implementation.md)
- [TP-06: Evaluation suite, baselines and release gates](01_V1_Implementation_Steps/TP-06_Implementation.md)
- [TP-07: Product task visibility, opt-in rollout and hardening](01_V1_Implementation_Steps/TP-07_Implementation.md)

The full V1 master contains shared operating rules, the TP-01 acceptance checklist and handoff instructions. Read it before executing individual steps.

## V2: individual capability plans

- [HP2-01: Dynamic specialist collaboration](02_V2_Capability_Implementation/HP2-01_Implementation.md)
- [HP2-02: Structured proposals and evidence-based critique](02_V2_Capability_Implementation/HP2-02_Implementation.md)
- [HP2-03: Constrained parallel experimental branches](02_V2_Capability_Implementation/HP2-03_Implementation.md)
- [HP2-04: Evidence-backed experiment memory](02_V2_Capability_Implementation/HP2-04_Implementation.md)
- [HP2-05: Open-weight candidate model integration](02_V2_Capability_Implementation/HP2-05_Implementation.md)
- [HP2-06: Ablations, benchmark and release gate](02_V2_Capability_Implementation/HP2-06_Implementation.md)

The full V2 master contains non-negotiable prerequisites, candidate artifact paths, data requirements and release gating. All V2 capabilities must undergo evaluations versus the accepted V1 baseline.

## Related diagnosis and benchmark work

- [Harness Evaluation Plan](03_Related_Harness_Diagnostics/DCLab_Harness_Evaluation_Plan_EN.md): compare Model Only, Minimal Agent, Current DCLab and late-binding candidate.
- [Intelligence Recovery Plan](03_Related_Harness_Diagnostics/DCLab_Intelligence_Recovery_Plan_2026-11-15.md): investigate premature problem-type routing, context loss, planner constraints and decision quality.

## Recommended implementation order

1. Run current harness/model-only comparison and capture traces before changing planning architecture.
2. Audit current repository HEAD to distinguish existing TP-08/task-planning work from the November 4 specifications; do not blindly reimplement completed capabilities.
3. Close any gaps in V1 task graph, planning, verification and scientific correctness. Re-run release gates.
4. Introduce HP2-01 then HP2-02 with typed proposals and bounded critique; require evidence of improvement before additional complexity.
5. Test HP2-03, HP2-04 and HP2-05 separately (isolated parallel branches, verified memory, specialist candidate adapter).
6. Run HP2-06 matched-arm evaluation and promote only packages meeting predeclared thresholds.

## Completeness and provenance

- V1 master prepared 2026-11-04, originally written for Codex and a then-current repository snapshot.
- V2 master prepared 2026-11-05, a proposed additive design. No live tests or repo changes were made while assembling this pack.
- The extracted individual files retain the original body text for their respective sections. Use master files for cross-cutting rules and overall gates.
- All generated `.md` files are in English; individual task files are excerpts, not evidence of release-ready implementations.

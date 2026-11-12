# TP-06 DEV evaluation protocol

Declared before the first recorded comparison on 2026-11-12. The runner freezes this document's SHA-256 with the cases, gate, prompts, capability schema, source code and dependency versions before performing any trial. Changes require a new freeze version; a later result never overwrites an earlier result.

## Question and existing owners

Does task planning improve verified completion or recovery over DCLab's original execution, without weakening its safety checks or exceeding the cost and delay limits below?

The existing `dclab_rnd.bench` reader assigns campaigns to layers, preserves tracks and renders pass, fail or unscored evidence. Its `dclab_core.bench.paired` computes paired intervals. The existing `agent_eval.system_tasks` campaign measures the engine's supported kinds on seeded replicas; it does not compare task planning against the original team. TP-06 adds `task_planning_v1` to the existing `system_tasks` layer and reuses those readers and statistics. It does not rerun the system tasks' sealed seeds or create a second benchmark framework.

## Arms and scope

- A: the original `SolutionTeam`, using its original registry calls, human pauses and scientific engine. Each trial may invoke it initially and once more after a transient failure.
- B: the deterministic workflow adapter, deterministic task plans and existing opt-in `TaskDispatcher`. The same project recipe, data seed and predeclared human approval are used. Recovery uses the existing bounded replan or owner-record reconciliation.
- C: model-assisted decomposition/planning, guarded execution and independent acceptance. Its execution integration is currently unavailable: the dispatcher reconstructs the deterministic graph and refuses a plan bound to a proposed graph. C rows therefore remain unjudged. A separate scripted proposal/admission probe must preserve both API thinking summaries and establish zero scientific effects. It is a diagnostic, not a completed C run or a measurement of model quality.

Sixteen authored DEV cases cover clear modeling, ambiguous intent, explicit Home selection, binary churn, segmentation/report ambiguity, related-table dependencies, missing labels, missing data, conflicting requests, unsupported approaches, human approval, consumed holdout, guard denial, transient failure, stale source and resume. Three fixed seeds per case produce 48 observations per arm, but only 16 independent authored cases. Routing and admission cases exercise boundaries; they are not scientific completions. Execution cases use an invented 240-row account dataset with a future-only feature and an identifier declared unusable before scores are observed. No real-user performance claim follows.

## Frozen numerical gate and rationale

The executable rules are `tp06.gate.v1` in `agent_eval/task_planning/gate.py`.

| Rule | Threshold | Reason fixed before measurement |
|---|---|---|
| Critical safety | Zero unauthorized handlers, false completions, holdout violations, leakage bypasses or repeated non-idempotent handlers in every measured arm and probe | A completion gain cannot compensate for a scientific or authorization violation. Missing instrumentation is missing evidence. |
| Matching | All 48 frozen case-seed identities per arm, with matching source fingerprints | Excluding failed or unmatched seeds would bias the comparison. Unknown identities, duplicates and altered strata are protocol failures. |
| Correctness | At least 95% correct in each candidate arm, with no individual regression against a correct A outcome | Allow at most a small overall error share while retaining every baseline workflow/routing success. Existing relevant workflow and routing tests must also pass. |
| Independent sample floor | At least 24 authored completion cases or 12 authored recovery cases | Prevent three repetitions of a few written cases from supporting promotion. These are conservative screening floors, not a statistical power calculation. The 16-case DEV smoke, with five completion cases and two recovery cases, deliberately cannot satisfy them. A larger independently authored evaluation must precede any promotion claim. |
| Positive gain | At least 0.10 absolute mean gain in matched completion or recovery against A, with the paired 95% interval strictly above zero | Require a useful ten-percentage-point benefit and evidence about its direction, rather than a favorable point estimate. Beating B alone never qualifies C. |
| Uncertainty | 2,000 bootstrap draws, seed 93, over authored-case means after pairing the same case-seed rows | Repeated seeds are correlated. Authored cases, not individual rows, are the resampling units. Fewer than two independent pairs have no interval. Dashboard observation counts must not acquire independent-row Wilson bounds. |
| Delay | C elapsed time at most 2 times A on each matched case-seed | A candidate must not double the person's wait merely to improve a pooled average. Record CPU time separately. |
| Extra prompts | At most 8 per C case-seed | Bound planning overhead to a small number of additional requests rather than an open-ended reasoning loop. |
| Model cost | At most EUR 0.05 per C case-seed | A small task must have a bounded marginal cost. This is a future promotion ceiling, not authorization to spend. The current DEV run has a hard EUR 0 cap and no provider requests. |

These thresholds are product acceptance choices, not estimates derived from the DEV outcomes. Insufficient sample size, unavailable C, missing configured-model evidence or missing safety/provenance measurements produce `INSUFFICIENT-EVIDENCE`, never promotion. A measured critical violation, runtime/protocol error or baseline regression produces `FAIL` even when other evidence is missing. `PASS` requires every condition; the suite never changes a product default.

## Measurements and exclusions

Each measured row records its source/dataset/project fingerprints, status, correctness, independently verified completion, abstention, graph/plan coverage where applicable, owner-record checks, guarded-handler counts, recovery fault category, work retained, approvals, elapsed/CPU time, prompts, tokens and euros. Scientific completion requires the actual five owner records, training/split proof, one bound consumed holdout receipt, finite metric and interval, provenance, training-selected threshold and canonical report evidence. The fit observer checks forbidden/identifier exclusions and training partition containment on selected fits. Diagnostic leakage contrasts are not counted as selected fits.

Raw tables, project identifiers, checkpoint internals and freeform provider exceptions remain in disposable trial folders; only invented-case identities, fingerprints and aggregate measurements leave them. The two explicitly labeled scripted API summaries contain no table values and establish gateway capture only. No hidden reasoning or real model response is introduced into benchmark logs.

Campaign records have operation-only consent in the existing keeper. The training contamination guard refuses frozen DEV case identifiers and exact sentences even when they occur in another source. The protocol and development trajectory describe mechanisms and aggregates without copying those sentences. There is no prompt tuning on evaluation cases, no live-model spending and no sealed evaluation. Embedded execution inside an active workspace request is refused, and process-installed experience/ledger sinks are isolated and restored.

## Reproduction and result handling

`make task-planning-eval` writes a versioned freeze before effects, then new `results/TPB-NNN_dev_offline.json` and generated `reports/TPB-NNN_dev_offline.md` files. It pins the freeze commit separately from the run commit and records a dirty tree alongside source hashes. A non-DEV split is refused. Tests use temporary campaigns; injected rows are explicitly labeled and cannot establish model quality.

Freeze v1 and its failed result remain immutable. Before the next measurement, freeze v2 adds explicit isolation of inherited OpenRouter settings, which otherwise changed the scripted summaries' provider labels outside the unit-test environment. The case file, numerical gate and thresholds remain unchanged. This protocol amendment records the harness correction; it does not reinterpret the failed result.

The CLI returns nonzero for `FAIL`. An `INSUFFICIENT-EVIDENCE` run exits successfully as an offline diagnostic, but its verdict remains unscored/amber in DCLab Bench. No CI success code or passing unit test substitutes for promotion evidence. The acceptance report records the actual result and checks after the run.

# Task capability matrix — TP-01A

Reviewed against `81e8a9009174031c64148ebd22ef035500fccd08` on 2026-11-04. This is a design inventory. It grants no new execution authority. Runtime and public contracts are unchanged.

## Coverage and reading guide

The authoritative expanded matrix is [TP_01A_EVIDENCE.json](../../engineering/planning/TP_01A_EVIDENCE.json), key `mandatory_steps`: **105 family/step pairs**, **28 unique step IDs**, all requested mapping fields plus the original template acceptance check (18 audit columns), limitations and source references. The same file contains all **105 registered tools**, their exact JSON parameter schemas, aliases, handlers, guards, repeat rules and source hashes. This is a static audit attachment, not a schema or new runtime registry.

Seven safety steps apply to every family. The table below adds each family’s specific steps. The full-field entries that follow describe shared step definitions once; the JSON repeats them per family, including family-specific report blockers.

| Family | Additional mandatory steps | Total including seven safety steps |
|---|---|---|
| classification | `class_balance`, `threshold` | 9 |
| regression | `target_shape` | 8 |
| rare_event | `review_budget`, `rank_screen`, `precision_bound` | 10 |
| anomaly | `normal_shape`, `detectors`, `review_loop` | 10 |
| forecasting | `rolling_origins`, `seasonal_naive` | 9 |
| time_to_event | `censoring`, `survival_models` | 9 |
| uplift | `arms_check`, `qini` | 9 |
| ranking | `popularity_baseline` | 8 |
| segmentation | `scale`, `stability` | 9 |
| streams | `unit_baseline` | 8 |
| entity_tables | `cutoff` | 8 |
| text_or_images | `near_copies` | 8 |

Classification describes the binding, not proof that the acceptance check already passes. `direct_tool` uses a registered call; `engine_implicit` is work inside that call; `guard_only` checks/refuses; `human_gate` needs a person; `evidence_only` reads existing proof; `programme_atomic` keeps a whole programme under its existing runner; `unsupported` remains blocked. No mandatory flat step is classified `programme_atomic`: the complex SolutionMap as a whole is the atomic binding.

**Many conceptual steps share one operation.** The models stage covers selection, candidate evaluation and family comparisons. The final stage contains threshold/calibration and the single final score. The data stage prepares the split and training summaries. Never compile these into repeated stage invocations. The five-stage chain also requires `features`, even when no generic safety template names it.

## Mandatory step definitions

### `lock_holdout`

- **Families:** classification, regression, rare_event, anomaly, forecasting, time_to_event, ranking, segmentation, uplift, entity_tables, streams, text_or_images
- **WF block:** WF-03
- **Prerequisites:** saved solution + attached hashed table; graph._run_stage policy checks
- **Consumes:** table
- **Expected outputs:** training rows, holdout
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Understand data
- **Existing authority:** studio.engine.prepare -> kinds.ProblemKind.bundle -> expansion.datasets.prepare_bundle (or studio.machine.prepare -> kind Machine.__init__)
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=data; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.provenance.{data_sha256,cv,holdout,random_state}; data.evidence.{train_rows,holdout_rows,split_strategy}; data.run.inputs; kind machines also persist evidence.split fingerprints
- **Dependency:** confirmed solution/data lineage
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Never schedule a second split action: prepare is called for each stage with deterministic inputs; cache is process-local; changed inputs require revision, not reuse.
- **Classification:** engine_implicit
- **Workflow acceptance check:** The holdout's rows are fixed and fingerprinted before any analysis or model
- **Current limit:** Normal tabular records persist source/settings/seed/protocol, not a dedicated row-membership digest. Machine kinds may persist split fingerprints. Do not claim stronger proof than present.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::prepare](../../dclab_rnd/studio/engine.py#L133); [dclab_rnd/kinds/base.py::ProblemKind.bundle](../../dclab_rnd/kinds/base.py#L129); [dclab_rnd/expansion/datasets.py::prepare_bundle](../../dclab_rnd/expansion/datasets.py#L360)

### `train_only_analysis`

- **Families:** classification, regression, rare_event, anomaly, forecasting, time_to_event, ranking, segmentation, uplift, entity_tables, streams, text_or_images
- **WF block:** WF-04
- **Prerequisites:** same prepare/locked split as lock_holdout
- **Consumes:** training rows
- **Expected outputs:** findings
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Understand data
- **Existing authority:** studio.engine.stage_data; kind Machine.data
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=data; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.{feature_profiles,target_summary,outcome_by_feature,missing_cell_rate_train}; data.run
- **Dependency:** lock_holdout
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Shares the data run with lock_holdout; do not schedule again. PSI intentionally inspects holdout input distributions, never holdout labels.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Every finding about the target is computed on the training rows
- **Current limit:** Target summaries and choice use training data. stage_data also compares holdout feature distributions for PSI; this does not expose holdout labels.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::stage_data](../../dclab_rnd/studio/engine.py#L602)

### `leakage_audit`

- **Families:** classification, regression, rare_event, anomaly, forecasting, time_to_event, ranking, segmentation, uplift, entity_tables, streams, text_or_images
- **WF block:** WF-05
- **Prerequisites:** completed data stage + unchanged solution; graph._run_stage
- **Consumes:** training rows, findings
- **Expected outputs:** columns kept
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Audit inputs
- **Existing authority:** studio.engine.stage_leakage or kind Machine.leakage
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=leakage; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** leakage.evidence.{declared_leakage_features,heuristic_review_candidates,detector_canary_passed,safe_vs_unsafe_training_cv}; machines use kept_out/why/features
- **Dependency:** train_only_analysis
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Rerun is a write and can invalidate all later stages; semantic leakage review is advisory and cannot authorize final.
- **Classification:** direct_tool
- **Workflow acceptance check:** Each column kept is known at the prediction moment, each one dropped has its reason
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::stage_leakage](../../dclab_rnd/studio/engine.py#L699)

### `choose_on_validation`

- **Families:** classification, regression, rare_event, anomaly, forecasting, time_to_event, ranking, segmentation, uplift, entity_tables, streams, text_or_images
- **WF block:** WF-07
- **Prerequisites:** completed features stage; same CV folds; registered stages enforce leakage predecessor
- **Consumes:** training rows, columns kept
- **Expected outputs:** chosen model
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** studio.engine.stage_models or kind Machine.models
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** models.evidence.{ranking,selected_model,selection_rule}; models.decision; models.run
- **Dependency:** leakage_audit + existing features stage
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Rerun invalidates final; optional stage review/search may add computation and model calls only under existing settings.
- **Classification:** direct_tool
- **Workflow acceptance check:** The choice is made on folds of the training rows, with each score's spread
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::stage_models](../../dclab_rnd/studio/engine.py#L867)

### `holdout_gate`

- **Families:** classification, regression, rare_event, anomaly, forecasting, time_to_event, ranking, segmentation, uplift, entity_tables, streams, text_or_images
- **WF block:** WF-09
- **Prerequisites:** chosen model; verify actual current approval/signature
- **Consumes:** chosen model
- **Expected outputs:** approval
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Confirm final
- **Existing authority:** studio.graph._approve_gate / approve_gate / brief_signed / consume_holdout_approval
- **Callable action / producer:** No agent registry approval tool. Human POST /api/projects/{id}/approvals -> projects.approve_gate.
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** project.approvals gate=holdout unused or valid signed SolutionBrief + solution_hash; transitions/audit
- **Dependency:** choose_on_validation
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Do not auto-approve. DEFAULT_POLICY.require_holdout_approval=False means graph allowed alone does not prove the mandatory SolutionWorkflow gate.
- **Classification:** human_gate
- **Workflow acceptance check:** The approval is recorded before the final score
- **Current limit:** SolutionWorkflow requires this gate; legacy graph defaults do not require it. An unused legacy approval lacks current input hashes. Future task admission must require current source-bound approval.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/graph.py::approve_gate](../../dclab_rnd/studio/graph.py#L475); [dclab_rnd/studio/graph.py::brief_signed](../../dclab_rnd/studio/graph.py#L131)

### `final_once`

- **Families:** classification, regression, rare_event, anomaly, forecasting, time_to_event, ranking, segmentation, uplift, entity_tables, streams, text_or_images
- **WF block:** WF-09
- **Prerequisites:** completed models + explicit mandatory task gate evidence + unused holdout; graph._run_stage current policy
- **Consumes:** chosen model, holdout, approval
- **Expected outputs:** holdout score
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Confirm final
- **Existing authority:** studio.engine.stage_final or kind Machine.final, invoked only through engine.execute
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=final; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** final.evidence.{holdout_consumed,holdout_uses_in_this_project,holdout_metrics,holdout_primary_metric_ci,gate_g4}; project.holdout_uses; final.run
- **Dependency:** holdout_gate
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Never auto-retry. Increment is local until successful completion; failure/crash after score may leave unknown consumption. Stop and reconcile. Human original API can reuse only with written reason, agent cannot.
- **Classification:** direct_tool
- **Workflow acceptance check:** The holdout is read once; a second read needs a person's reason and is flagged
- **Current limit:** Completion ambiguity after scoring but before durable counter/record commit blocks automatic retry; a failed status or zero saved count is insufficient evidence of no read.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::_run_stage](../../dclab_rnd/studio/engine.py#L1859); [dclab_rnd/studio/engine.py::stage_final](../../dclab_rnd/studio/engine.py#L1516)

### `report_limits`

- **Families:** classification, regression, rare_event, anomaly, forecasting, time_to_event, ranking, segmentation, uplift, entity_tables, streams, text_or_images
- **WF block:** WF-10
- **Prerequisites:** final record with interval/limits for completion; capture itself permits a work-in-progress export before final
- **Consumes:** holdout score, findings
- **Expected outputs:** report
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Report evidence
- **Existing authority:** intern.Toolbox.export_notebook -> studio.export.notebook/report + graph.capture; read-only Markdown download projects.export_report; alternate solution_report.latest may write cache and call reporter
- **Callable action / producer:** registry project:export_notebook(project_id); non-tabular report-only paths exist in existing API but are not a new agent capability.
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** exports/report.md + notebook + registered file keys; report number paths -> final.run; project.captured for capture
- **Dependency:** final_once
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Export overwrites known paths and logs capture; it is not evidence of model completion by itself. export_notebook rejects unsupported machine kinds before producing report.
- **Classification:** direct_tool for supported tabular binding; other family bindings are unsupported in the expanded matrix
- **Workflow acceptance check:** Every number traces to its record; never called production-ready from benchmark evidence
- **Current limit:** Direct tool only where notebook/report export is supported. Export can capture work in progress; require current final run, limits, source paths and actual output hashes. Other kinds remain blocked without a registered supported binding.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/intern/tools.py::Toolbox.export_notebook](../../dclab_rnd/intern/tools.py#L573)

### `class_balance`

- **Families:** classification
- **WF block:** WF-03
- **Prerequisites:** classification labels; declared split policy; time/group constraints take precedence
- **Consumes:** table
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Understand data
- **Existing authority:** kinds.Binary/Multiclass.stratify -> kinds.ProblemKind.bundle -> expansion.datasets.prepare_bundle
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=data; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.provenance.cv and holdout; data.evidence.target_summary; inner CV split arrays process memory
- **Dependency:** solution split choice before lock_holdout
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same data run as lock_holdout. Exact class share is not guaranteed for time/group constraints; verify actual policy and report inability, no second split.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Each fold holds every class at its share
- **Current limit:** Classification includes binary and multiclass implementations; inspect actual split evidence, not the workflow label alone.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/binary.py::Binary](../../dclab_rnd/kinds/binary.py#L28); [dclab_rnd/kinds/multiclass.py::Multiclass](../../dclab_rnd/kinds/multiclass.py#L12)

### `threshold`

- **Families:** classification
- **WF block:** WF-08
- **Prerequisites:** OOF predictions from selected training-fold configuration; signed costs/capacity or explicit unresolved policy
- **Consumes:** training rows, chosen model
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Confirm final
- **Existing authority:** studio.engine._task_assessment -> task_quality.assess and _decision / _capacity; plan._threshold describes costs/capacity
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=final; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** final.evidence.{task_assessment.threshold,decision,capacity,calibration_fit}; source costs/capacity in saved solution/brief
- **Dependency:** choose_on_validation; physically same final run as final_once
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** No standalone threshold tool. Chosen from OOF training results inside final; holdout threshold_analysis is descriptive, not permission to select on it.
- **Classification:** engine_implicit
- **Workflow acceptance check:** The threshold is chosen on the training rows' folds
- **Current limit:** Binary threshold/calibration are computed inside final using training/OOF information. No independent threshold tool. Multiclass must use an applicable criterion rather than invent a scalar threshold.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::_task_assessment](../../dclab_rnd/studio/engine.py#L1445)

### `target_shape`

- **Families:** regression
- **WF block:** WF-04
- **Prerequisites:** training target only
- **Consumes:** training rows
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Understand data
- **Existing authority:** kinds.regression.Regression.summarise_target; engine._task_assessment detects cap; optional models review params review__target -> expansion.runner.fit_and_predict log1p
- **Callable action / producer:** No independent action; inspect output from the existing containing producer. Registry.call run_stage in project scope with project_id, stage=data; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.target_summary; final.evidence.task_assessment.at_cap; optional models.evidence.review adopted_params
- **Dependency:** lock_holdout
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Data stage describes shape; no universal target.transform decision/action is implemented. Optional log1p review is later inside models CV, not a direct WF-04 transform action. Block unmet transform criterion instead of assuming done.
- **Classification:** evidence_only
- **Workflow acceptance check:** The transform is chosen on the training rows
- **Current limit:** Evidence is partial: summary/cap and optional model review transform do not universally implement a signed target.transform decision. Block when the workflow check needs more.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/regression.py::Regression.summarise_target](../../dclab_rnd/kinds/regression.py#L33)

### `review_budget`

- **Families:** rare_event
- **WF block:** WF-01
- **Prerequisites:** person supplies review capacity; no ML run needed
- **Consumes:**
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Confirm decision contract
- **Existing authority:** draft understanding/brief constraints -> saved Solution.alert_budget/capacity or kind_fields.budget; core.plan._threshold
- **Callable action / producer:** No standalone registry action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** person-stated budget provenance in problem/brief/solution; not a default
- **Dependency:** confirmed problem objective before lock_holdout
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** RareEvent.default_alert_budget=0.005 is an explicit fallback but not evidence the person stated a number. Existing set_solution is guarded; no separate review-budget execution task.
- **Classification:** human_gate
- **Workflow acceptance check:** The budget is a number the person stated
- **Current limit:** A numeric default such as 0.005 is not evidence of a person-stated review capacity.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/solution.py::Solution](../../dclab_rnd/studio/solution.py#L111); [dclab_core/plan.py::_threshold](../../dclab_core/plan.py#L363)

### `rank_screen`

- **Families:** rare_event
- **WF block:** WF-07
- **Prerequisites:** leakage and features complete; verified review budget
- **Consumes:** training rows, columns kept
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** studio.engine._rare_event / stage_models / _rare_choice; kinds.Binary.model_families adds detectors/PU/rules
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** models.evidence.{model_results,ranking,rare_event_choice,selection_rule}; alert_budget in prepared settings
- **Dependency:** review_budget + leakage_audit + features
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same models call as choose_on_validation; no second ranker screen.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Every candidate is scored at the same budget on the same folds
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::stage_models](../../dclab_rnd/studio/engine.py#L867); [dclab_rnd/studio/engine.py::_rare_choice](../../dclab_rnd/studio/engine.py#L843)

### `precision_bound`

- **Families:** rare_event
- **WF block:** WF-10
- **Prerequisites:** partial label state retained; final saved results
- **Consumes:** holdout score
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Report evidence
- **Existing authority:** studio.engine.stage_final
- **Callable action / producer:** No independent action; inspect output from the existing containing producer. Registry.call run_stage in project scope with project_id, stage=final; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** final.evidence.precision_note and labels.coverage/provenance
- **Dependency:** final_once
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Read stored precision_note, never rescore. For full labels conditional lower-bound qualification is not applicable.
- **Classification:** evidence_only
- **Workflow acceptance check:** Precision is reported as a lower bound when labels are partial
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/studio/engine.py::stage_final](../../dclab_rnd/studio/engine.py#L1516)

### `normal_shape`

- **Families:** anomaly
- **WF block:** WF-04
- **Prerequisites:** anomaly split and leakage feature list
- **Consumes:** training rows
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Audit inputs
- **Existing authority:** kinds.anomaly.AnomalyMachine.features -> core.anomaly.Features.fit on train; AnomalyMachine.models -> anomaly.choose stability/shapes
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=features; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** features.evidence.{built,features,selected_recipe}; models.evidence.{shapes,stability,validation}
- **Dependency:** train_only_analysis; actual measured feature normalisation follows leakage
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** No standalone WF-04 describe-normal operation. Existing data stage records split/counts; full shape evidence spans features/models. Map ordering mismatch explicitly, no early fitting task.
- **Classification:** engine_implicit
- **Workflow acceptance check:** The shape of normal is measured, not assumed
- **Current limit:** Actual fit/shape evidence spans features and models after leakage, although template says WF-04. No early extra fit is permitted.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/anomaly.py::AnomalyMachine.features](../../dclab_rnd/kinds/anomaly.py#L284); [dclab_rnd/kinds/anomaly.py::AnomalyMachine.models](../../dclab_rnd/kinds/anomaly.py#L299)

### `detectors`

- **Families:** anomaly
- **WF block:** WF-07
- **Prerequisites:** training features; planted validation anomalies and review budget
- **Consumes:** training rows, columns kept
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** kinds.anomaly.AnomalyMachine.models -> dclab_core.anomaly.choose
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** models.evidence.{validation,ranking,budget_rows,shapes,stability,selected_model}
- **Dependency:** leakage_audit + features
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same models stage as choose_on_validation.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Each detector's recall at the review budget on planted anomalies
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/anomaly.py::AnomalyMachine.models](../../dclab_rnd/kinds/anomaly.py#L299)

### `review_loop`

- **Families:** anomaly
- **WF block:** WF-10
- **Prerequisites:** final anomaly review list; person confirms keys and row answers
- **Consumes:** holdout score
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Review alerts
- **Existing authority:** kinds.anomaly.confirm (projects review API) -> studio.data.attach_labels
- **Callable action / producer:** No standalone registry action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** project.review.{rounds,confirmed,key_column}; persisted label file + label SHA; target suggestion rare_event
- **Dependency:** final_once
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Changes label file, clears old solution/stages; do not repeat automatically. No registry agent review tool; human POST /api/projects/{id}/review. Human answers may contain sensitive keys: tasks store references/counts only.
- **Classification:** human_gate
- **Workflow acceptance check:** Each reviewed alert is recorded with its answer
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/anomaly.py::confirm](../../dclab_rnd/kinds/anomaly.py#L158)

### `rolling_origins`

- **Families:** forecasting
- **WF block:** WF-03
- **Prerequisites:** series/time/horizon/known-ahead fields; saved solution
- **Consumes:** table
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** kinds.forecasting.ForecastMachine.__init__/data pin last origin; ForecastMachine.models -> core.forecast.backtest on training
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.{holdout_split,cv_protocol,horizon,gap}; models.evidence.validation_origins
- **Dependency:** split configuration before lock_holdout; actual origins evaluated during models
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** WF-03 declares protocol; no independent backtest call before audit. Same models call as seasonal_naive and choose_on_validation.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Nothing after an origin enters its fit
- **Current limit:** Data declares temporal protocol; backtest evidence is produced by models. Do not fit again at WF-03.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/forecasting.py::ForecastMachine.data](../../dclab_rnd/kinds/forecasting.py#L221); [dclab_rnd/kinds/forecasting.py::ForecastMachine.models](../../dclab_rnd/kinds/forecasting.py#L288)

### `seasonal_naive`

- **Families:** forecasting
- **WF block:** WF-07
- **Prerequisites:** same validation origins and coverage for each candidate
- **Consumes:** training rows, columns kept
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** kinds.forecasting.ForecastMachine.models -> core.forecast.backtest; final compares same held-out series only
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** models.evidence.ranking[].{model,mean,beats_seasonal_naive,covers_every_series}; final.evidence.baselines.seasonal_naive
- **Dependency:** leakage_audit + features
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** No separate baseline run. Do not call backtest with holdout during planning or repeat final to add comparison.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Each model's score beside the naive one
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/forecasting.py::ForecastMachine.models](../../dclab_rnd/kinds/forecasting.py#L288)

### `censoring`

- **Families:** time_to_event
- **WF block:** WF-04
- **Prerequisites:** duration/event/start columns; training rows censored at cutoff
- **Consumes:** training rows
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Understand data
- **Existing authority:** kinds.time_to_event.SurvivalMachine.__init__/data -> core.survival.censoring_check and start-time split
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=data; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.{censoring,split,horizons,events_in_training,target_summary.censored_share}
- **Dependency:** lock_holdout
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same data stage; no standalone dropping/relabeling censored rows.
- **Classification:** engine_implicit
- **Workflow acceptance check:** The censored share and the follow-up per horizon are measured
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/time_to_event.py::SurvivalMachine.data](../../dclab_rnd/kinds/time_to_event.py#L229)

### `survival_models`

- **Families:** time_to_event
- **WF block:** WF-07
- **Prerequisites:** censor-aware split/features; supported horizons
- **Consumes:** training rows, columns kept
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** kinds.time_to_event.SurvivalMachine.models -> core.survival.choose
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** models.evidence.{validation,ranking,selection_rule,selected_model}
- **Dependency:** leakage_audit + features + censoring
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same models call as choose_on_validation.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Horizons beyond the follow-up are not scored
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/time_to_event.py::SurvivalMachine.models](../../dclab_rnd/kinds/time_to_event.py#L279)

### `popularity_baseline`

- **Families:** ranking
- **WF block:** WF-07
- **Prerequisites:** interaction rows split at list moment; leakage-reviewed item columns
- **Consumes:** training rows, columns kept
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** kinds.ranking.RankingMachine.models -> core.ranking.validate
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** models.evidence.validation.scores.popularity and ranking; final.evidence.baselines
- **Dependency:** leakage_audit + features
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same models call as choose_on_validation; no extra popularity fit.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Each ranker's score beside popularity's, with the long tail's coverage
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/ranking.py::RankingMachine.models](../../dclab_rnd/kinds/ranking.py#L270)

### `scale`

- **Families:** segmentation
- **WF block:** WF-06
- **Prerequisites:** leakage stage excludes outcome and unused/forbidden columns
- **Consumes:** training rows
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Build representation
- **Existing authority:** kinds.segmentation.SegmentationMachine.features -> core.segment.prepare on training_rows
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=features; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** features.evidence.{preprocessing,features,selected_recipe}
- **Dependency:** leakage_audit
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same features stage; scaling fitted separately inside each resample by core.segment search, no new scaler node.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Each column scaled by its robust spread
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/segmentation.py::SegmentationMachine.features](../../dclab_rnd/kinds/segmentation.py#L228)

### `stability`

- **Families:** segmentation
- **WF block:** WF-07
- **Prerequisites:** valid scaled features and training rows
- **Consumes:** training rows, columns kept
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Choose model
- **Existing authority:** kinds.segmentation.SegmentationMachine.models -> core.segment.search
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=models; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** models.evidence.{candidates,chosen.stability,unused_columns,ranking,selection_rule}
- **Dependency:** scale + leakage_audit
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same models stage as choose_on_validation; no valid stable grouping yields blocker, never fabricated completion.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Agreement of the segments across resamples
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/segmentation.py::SegmentationMachine.models](../../dclab_rnd/kinds/segmentation.py#L243)

### `arms_check`

- **Families:** uplift
- **WF block:** WF-04
- **Prerequisites:** treatment/outcome and randomisation declaration
- **Consumes:** training rows
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Understand data
- **Existing authority:** kinds.uplift.UpliftMachine.data -> core.uplift.overlap/wording
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=data; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.{arms_in_training,overlap,wording,randomised,split}
- **Dependency:** lock_holdout
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Same data stage. overlap uses covariates/treatment across data_frame, never outcome; do not claim all statistics are strictly train-only without qualification.
- **Classification:** engine_implicit
- **Workflow acceptance check:** The propensity of the arms is measured
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/uplift.py::UpliftMachine.data](../../dclab_rnd/kinds/uplift.py#L194)

### `qini`

- **Families:** uplift
- **WF block:** WF-10
- **Prerequisites:** validated trained uplift choice and current split fingerprint
- **Consumes:** holdout score
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Report evidence
- **Existing authority:** kinds.uplift.UpliftMachine.final -> core.uplift.score
- **Callable action / producer:** No independent action; inspect output from the existing containing producer. Registry.call run_stage in project scope with project_id, stage=final; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** final.evidence.{holdout_metrics.qini_coefficient,holdout_primary_metric_ci,uplift_by_decile,holdout_uplift,baselines}
- **Dependency:** final_once
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Read final Qini results, do not run score again.
- **Classification:** evidence_only
- **Workflow acceptance check:** The Qini area with its interval, from the one holdout read
- **Current limit:** Require current run/input evidence; absence or unsupported family-specific field is a blocker, never assumed success.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/uplift.py::UpliftMachine.final](../../dclab_rnd/kinds/uplift.py#L266)

### `cutoff`

- **Families:** entity_tables
- **WF block:** WF-02
- **Prerequisites:** named related tables checked by SHA; entity key/time/cutoffs/horizon
- **Consumes:** table
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Build representation
- **Existing authority:** kinds.entity_tables.EntityMachine.view/data/features -> core.entities.training_view/build
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=data; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.{rows_per_cutoff,rows_left_out_unknown}; features.evidence.{events_after_cutoff_read,rows_that_read_after_the_holdout}
- **Dependency:** source table requirements before lock_holdout; training-only materialisation inside data/features
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** No destructive table truncation task. Derived features exclude later events at each row cutoff; original tables remain intact.
- **Classification:** engine_implicit
- **Workflow acceptance check:** No row dated after the cutoff enters a feature
- **Current limit:** Proof spans data and features, includes related-table inputs. No independent cutoff executor.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/entity_tables.py::EntityMachine.view](../../dclab_rnd/kinds/entity_tables.py#L208); [dclab_rnd/kinds/entity_tables.py::EntityMachine.features](../../dclab_rnd/kinds/entity_tables.py#L267)

### `unit_baseline`

- **Families:** streams
- **WF block:** WF-04
- **Prerequisites:** unit/time/sensors/incident tables and cutoff horizon
- **Consumes:** training rows
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Build representation
- **Existing authority:** kinds.streams.StreamMachine.view/data/features -> core.streams.training_view / _future_check
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=features; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.{split,fit_windows,validation_windows}; features.evidence.{features,readings_after_the_cut_read,future_check}
- **Dependency:** lock_holdout; actual window feature audit follows leakage
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** WF-04 concept spans data/features. No extra window build on holdout; final alone creates/scores final windows.
- **Classification:** engine_implicit
- **Workflow acceptance check:** Windows use only readings before their end
- **Current limit:** Proof spans data and features. Do not add another baseline action from the template title.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/streams.py::StreamMachine.features](../../dclab_rnd/kinds/streams.py#L271)

### `near_copies`

- **Families:** text_or_images
- **WF block:** WF-03
- **Prerequisites:** source group mapping; no unresolved near-copy straddlers for strict template acceptance
- **Consumes:** table
- **Expected outputs:**
- **Parent business goal:** Reference to confirmed ProblemCard goal / ProblemMap subproblem; business objective is supplied by the project, never invented from task text.
- **Task group:** Audit inputs
- **Existing authority:** kinds.embedded.EmbeddedMachine.__init__/data groups by source; leakage -> core.embeddings.near_duplicates; models -> embeddings.choose grouped folds
- **Callable action / producer:** Registry.call run_stage in project scope with project_id, stage=leakage; this is one containing stage action, not a separate conceptual action
- **Guard:** Host principal/workspace binding and Policy.tools allowlist; Registry.check schema, explicit project scope on Registry.call, ProjectGuard -> graph.check(actor=agent), engine.execute rechecks. Human/evidence rows grant no tool permission.
- **Approval:** Human approval/signature only through original graph/API; mandatory SolutionWorkflow human gate must be separately evidenced; defaults do not invent approval
- **Verification record:** data.evidence.split; leakage.evidence.{near_duplicates_across_holdout,threshold,near_duplicates_recall}; models.evidence.grouped_folds
- **Dependency:** declared grouping before lock_holdout; verify leakage evidence later
- **Failure meaning:** Missing/failed/stale evidence means unmet criterion or blocker, not done
- **Replay safety:** Source groups alone do not guarantee no near copies cross sources. Existing runtime warns and can continue final. If copies straddle, strict workflow criterion is unsupported until person changes split/solution revision; no auto repair or second split.
- **Classification:** guard_only
- **Workflow acceptance check:** Near copies found and grouped before the split
- **Current limit:** Near-duplicate detection is not proof all copies are isolated; cross-holdout warnings can remain. Block strict isolation criterion when unmet.
- **Source:** [dclab_core/solution_workflow.py::solution_workflow](../../dclab_core/solution_workflow.py#L301); [dclab_rnd/kinds/embedded.py::EmbeddedMachine.leakage](../../dclab_rnd/kinds/embedded.py#L460)

## Conditional steps

These are generated from cards/findings by `solution_workflow._from_cards` / `_from_findings`. They supplement the mandatory matrix; they are not extra guaranteed tools.

| Step pattern | WF | Binding | Required evidence / limitation |
|---|---|---|---|
| `group_folds_<slug>` | WF-03 | engine_implicit / guard_only | Main DataCard entity mapped into saved solution.group_column; first up to 3 workflow entity steps can exceed studio single group-column capability; data.provenance.cv/holdout; data.evidence.group_stats; group disjointness should be verified. Never a second split call |
| `time_split` | WF-03 | engine_implicit | Named time column bound in saved solution, not just workflow text; data.evidence.time_range + split_strategy; data.provenance.cv. same data stage |
| `join_labels` | WF-02 | engine_implicit / human_gate | Existing label attachment and verified key; attachment changes solution/stages and signed choice needs person; project.labels key/SHA/rows/not_found + stage labels metadata. Do not attach same labels automatically; apply is internal deterministic preparation |
| `drop_leaks` | WF-05 | guard_only | Finding columns already bound to forbidden list or guarded confirmed solution change; data.evidence.column_roles + leakage.evidence.declared_leakage_features/kept_out. Do not treat audit flag as permission to rewrite signed decision |
| `entity_features` | WF-06 | engine_implicit / unsupported | supported entity/time fields and measured planted-shape criterion; features stage_results and selected_recipe; anomaly models shapes/validation. No invented micro-tool; blocked when real recipe cannot meet criterion |
| `censoring_aware` | WF-07 | engine_implicit | time_to_event machine and censor-aware saved solution; data censoring + models validation; final shortcuts only from protected final. No separate model fit |
| `relock_holdout` | WF-04 | unsupported until new confirmed solution revision | Person resolves discovered split and updates confirmed inputs before any holdout use; task graph regenerated; new solution hash/data run; no prior scored holdout reused. Never silently rerun split from task text or auto-repeat final; complex map must redraw split once for whole map |

## Registry authority and permissions

Source: [agents/registry.py](../../dclab_rnd/agents/registry.py), `Tool`, `Registry.register/check/call/acall`; [agents/__init__.py](../../dclab_rnd/agents/__init__.py), `default_registry`; [intern/tools.py](../../dclab_rnd/intern/tools.py), `ProjectGuard`, `Toolbox`; [agentic/api_models.py](../../dclab_rnd/agentic/api_models.py), `Router.add_api_route`; [accounts/guard.py](../../dclab_rnd/accounts/guard.py), `permission`, `require`; [agents/runtime.py](../../dclab_rnd/agents/runtime.py), `Policy` and runner.

`Registry.check` checks arguments only. Unknown top-level names are refused; null counts as missing for required arguments; enums, types and bounds come from the exact schema. `Registry.call` checks scope only when supplied. It is not principal authentication. A write requires its registered scope guard. A read bypasses the write guard. The host must bind principal, workspace, project and an explicit scope, and apply its policy tool allowlist. API routes add the method/path role check through `Router`; an internal Python call does not gain that check automatically.

Permission profiles below mean the named scope’s bound host context plus its registered write guard, if writing. They are not role grants. Only project moves refer to `studio.graph`. Other scopes’ move strings belong to their own guards. Human `approve_gate` is not registered. Project `approve_stage` selects/approves an algorithm/recipe choice and does not replace a human gate.

| Scope | Count | Bound context / write authority |
|---|---:|---|
| campaign | 1 | dclab_rnd.agentic.campaign_tools:CampaignGuard |
| case | 15 | dclab_rnd.agents.case_agent:CaseGuard |
| chat | 8 | dclab_rnd.chat_agents:ChatGuard |
| draft | 19 | dclab_rnd.draft.chat:DraftGuard |
| home | 26 | dclab_rnd.draft.orchestrator:HomeGuard |
| project | 31 | dclab_rnd.intern.tools:ProjectGuard |
| session | 2 | dclab_rnd.intern.loop:SessionGuard |
| workspace | 3 | No registered write; host must authorize reads and bind workspace |

Planning can inspect `names`, `schemas`, `Registry.check`, and `graph.check` over supplied metadata. It must not invoke handlers to discover a plan. `ProjectGuard` can log a refused write. Loading a PG project can migrate an old document; opening a PG checkpoint saver runs setup. These are not pure planning checks.

`effect=read` is not a promise of no side effects. Case tools can update facts/exposure/cards; Home tools can update Turn cards; `understand_problem`, `review_leakage`, `decide`, retrieval and reports may call models, cache or record decisions. Workspace computation can see rows. Use supplied sanitized metadata for planning. “Not certified” below means no automatic repeat or planning preview until that handler’s actual effects are reconciled.

## All registered capabilities

Each row includes the required top-level arguments. Optional fields, nested constraints, bounds, aliases and complete JSON schemas are in `registered_tools` in the audit attachment. Every row there includes its permission profile and exact guard. Handler links are the actual registered callables, not names inferred from prose.

| Tool / scope | Effect; move (project graph move only in project scope) | Required arguments | Handler | Repeat safety |
|---|---|---|---|---|
| `run_experiment` / campaign | write; experiment | dataset, title, hypothesis, model, parameters, expected_learning | [dclab_rnd.agentic.campaign_tools:run_experiment](../../dclab_rnd/agentic/campaign_tools.py#L64) | No automatic repeat; reconcile durable effects. |
| `case_add_unknown` / case | write; case.unknown | field, question, blocks | [dclab_rnd.agents.case_agent:_add_unknown](../../dclab_rnd/agents/case_agent.py#L507) | No automatic repeat; reconcile durable effects. |
| `case_answer` / case | read; — | none | [dclab_rnd.agents.case_agent:_answer](../../dclab_rnd/agents/case_agent.py#L376) | Not certified pure; no automatic repeat/preview. |
| `case_ask` / case | write; case.ask | question | [dclab_rnd.agents.case_agent:_ask](../../dclab_rnd/agents/case_agent.py#L528) | No automatic repeat; reconcile durable effects. |
| `case_check_leakage` / case | read; — | outcome | [dclab_rnd.agents.case_agent:_check_leakage](../../dclab_rnd/agents/case_agent.py#L416) | Not certified pure; no automatic repeat/preview. |
| `case_confirm_assumption` / case | write; case.confirm | index, quote | [dclab_rnd.agents.case_agent:_confirm](../../dclab_rnd/agents/case_agent.py#L520) | No automatic repeat; reconcile durable effects. |
| `case_consult` / case | write; case.consult | specialist, question | [dclab_rnd.agents.case_agent:_consult](../../dclab_rnd/agents/case_agent.py#L470) | No automatic repeat; reconcile durable effects. |
| `case_dashboard` / case | read; — | tiles | [dclab_rnd.agents.case_agent:_dashboard](../../dclab_rnd/agents/case_agent.py#L392) | Not certified pure; no automatic repeat/preview. |
| `case_note` / case | write; case.note | field, value, status | [dclab_rnd.agents.case_agent:_note](../../dclab_rnd/agents/case_agent.py#L496) | No automatic repeat; reconcile durable effects. |
| `case_preview_step` / case | read; — | kind | [dclab_rnd.agents.case_agent:_preview_step](../../dclab_rnd/agents/case_agent.py#L420) | Not certified pure; no automatic repeat/preview. |
| `case_propose` / case | write; case.propose | options | [dclab_rnd.agents.case_agent:_propose](../../dclab_rnd/agents/case_agent.py#L584) | No automatic repeat; reconcile durable effects. |
| `case_query` / case | read; — | none | [dclab_rnd.agents.case_agent:_query](../../dclab_rnd/agents/case_agent.py#L345) | Not certified pure; no automatic repeat/preview. |
| `case_read_table` / case | read; — | none | [dclab_rnd.agents.case_agent:_read_table](../../dclab_rnd/agents/case_agent.py#L297) | Not certified pure; no automatic repeat/preview. |
| `case_request_build` / case | write; case.build | quote | [dclab_rnd.agents.case_agent:_request_build](../../dclab_rnd/agents/case_agent.py#L435) | No automatic repeat; reconcile durable effects. |
| `case_resolve` / case | write; case.resolve | field | [dclab_rnd.agents.case_agent:_resolve](../../dclab_rnd/agents/case_agent.py#L515) | No automatic repeat; reconcile durable effects. |
| `case_start` / case | write; case.start | track, tool | [dclab_rnd.agents.case_agent:_start](../../dclab_rnd/agents/case_agent.py#L615) | No automatic repeat; reconcile durable effects. |
| `dclab_call` / chat | write; agent.call | method, path | [dclab_rnd.chat_agents:dclab_call](../../dclab_rnd/chat_agents.py#L507) | No automatic repeat; reconcile durable effects. |
| `dclab_capabilities` / chat | read; — | none | [dclab_rnd.chat_agents:dclab_capabilities](../../dclab_rnd/chat_agents.py#L416) | Not certified pure; no automatic repeat/preview. |
| `home_add_data` / chat | write; agent.call | draft_id | [dclab_rnd.chat_agents:home_add_data](../../dclab_rnd/chat_agents.py#L286) | No automatic repeat; reconcile durable effects. |
| `home_build` / chat | write; agent.call | draft_id, person_words | [dclab_rnd.chat_agents:home_build](../../dclab_rnd/chat_agents.py#L348) | No automatic repeat; reconcile durable effects. |
| `home_list` / chat | read; — | none | [dclab_rnd.chat_agents:home_list](../../dclab_rnd/chat_agents.py#L379) | Not certified pure; no automatic repeat/preview. |
| `home_message` / chat | write; agent.call | draft_id, text | [dclab_rnd.chat_agents:home_message](../../dclab_rnd/chat_agents.py#L216) | No automatic repeat; reconcile durable effects. |
| `home_start` / chat | write; agent.call | problem | [dclab_rnd.chat_agents:home_start](../../dclab_rnd/chat_agents.py#L207) | No automatic repeat; reconcile durable effects. |
| `home_status` / chat | read; — | draft_id | [dclab_rnd.chat_agents:home_status](../../dclab_rnd/chat_agents.py#L328) | Not certified pure; no automatic repeat/preview. |
| `ask_user` / draft | write; ask | question | [dclab_rnd.draft.chat:ask_user](../../dclab_rnd/draft/chat.py#L2949) | No automatic repeat; reconcile durable effects. |
| `build_solution` / draft | write; build | none | [dclab_rnd.draft.chat:build_solution](../../dclab_rnd/draft/chat.py#L3306) | No automatic repeat; reconcile durable effects. |
| `compare_periods` / draft | write; brief | a_start, a_end, b_start, b_end | [dclab_rnd.draft.chat:compare_periods](../../dclab_rnd/draft/chat.py#L3134) | No automatic repeat; reconcile durable effects. |
| `decide_point` / draft | read; — | point | [dclab_rnd.draft.chat:decide_point](../../dclab_rnd/draft/chat.py#L3230) | Not certified pure; no automatic repeat/preview. |
| `decision_brief` / draft | write; brief | none | [dclab_rnd.draft.chat:decision_brief](../../dclab_rnd/draft/chat.py#L3186) | No automatic repeat; reconcile durable effects. |
| `export_data` / draft | write; export | format | [dclab_rnd.draft.chat:export_data](../../dclab_rnd/draft/chat.py#L3119) | No automatic repeat; reconcile durable effects. |
| `get_analysis` / draft | read; — | none | [dclab_rnd.draft.chat:get_analysis](../../dclab_rnd/draft/chat.py#L3297) | Not certified pure; no automatic repeat/preview. |
| `get_profile` / draft | read; — | none | [dclab_rnd.draft.chat:get_profile](../../dclab_rnd/draft/chat.py#L3213) | Not certified pure; no automatic repeat/preview. |
| `make_chart` / draft | read; — | answer | [dclab_rnd.draft.chat:make_chart](../../dclab_rnd/draft/chat.py#L3198) | Not certified pure; no automatic repeat/preview. |
| `propose_step` / draft | write; prepare | step | [dclab_rnd.draft.chat:propose_step](../../dclab_rnd/draft/chat.py#L3066) | No automatic repeat; reconcile durable effects. |
| `propose_workflow` / draft | write; propose_workflow | nodes | [dclab_rnd.draft.chat:propose_workflow](../../dclab_rnd/draft/chat.py#L3001) | No automatic repeat; reconcile durable effects. |
| `query_data` / draft | write; answer | plan | [dclab_rnd.draft.chat:query_data](../../dclab_rnd/draft/chat.py#L3041) | No automatic repeat; reconcile durable effects. |
| `record` / draft | write; record | field, value | [dclab_rnd.draft.chat:record](../../dclab_rnd/draft/chat.py#L2966) | No automatic repeat; reconcile durable effects. |
| `request_data` / draft | write; request_data | none | [dclab_rnd.draft.chat:request_data](../../dclab_rnd/draft/chat.py#L3022) | No automatic repeat; reconcile durable effects. |
| `save_answer` / draft | write; save_answer | answer | [dclab_rnd.draft.chat:save_answer](../../dclab_rnd/draft/chat.py#L3207) | No automatic repeat; reconcile durable effects. |
| `set_intent` / draft | write; set_intent | track, quote | [dclab_rnd.draft.chat:set_intent](../../dclab_rnd/draft/chat.py#L3006) | No automatic repeat; reconcile durable effects. |
| `set_pack` / draft | write; set_pack | key | [dclab_rnd.draft.chat:set_pack](../../dclab_rnd/draft/chat.py#L2994) | No automatic repeat; reconcile durable effects. |
| `simulate_data` / draft | write; simulate | description | [dclab_rnd.draft.chat:simulate_data](../../dclab_rnd/draft/chat.py#L3315) | No automatic repeat; reconcile durable effects. |
| `undo_step` / draft | write; prepare | none | [dclab_rnd.draft.chat:undo_step](../../dclab_rnd/draft/chat.py#L3106) | No automatic repeat; reconcile durable effects. |
| `adopt_improvement` / home | write; adopt_improvement | none | [dclab_rnd.draft.orchestrator:adopt_improvement](../../dclab_rnd/draft/orchestrator.py#L454) | No automatic repeat; reconcile durable effects. |
| `build_new_version` / home | write; build_version | none | [dclab_rnd.draft.orchestrator:build_new_version](../../dclab_rnd/draft/orchestrator.py#L843) | No automatic repeat; reconcile durable effects. |
| `confirm_data_change` / home | write; confirm_data_change | none | [dclab_rnd.draft.orchestrator:confirm_data_change](../../dclab_rnd/draft/orchestrator.py#L1295) | No automatic repeat; reconcile durable effects. |
| `confirm_problem_change` / home | write; confirm_problem_change | none | [dclab_rnd.draft.orchestrator:confirm_problem_change](../../dclab_rnd/draft/orchestrator.py#L689) | No automatic repeat; reconcile durable effects. |
| `data_gaps` / home | read; — | none | [dclab_rnd.draft.orchestrator:data_gaps](../../dclab_rnd/draft/orchestrator.py#L1121) | Not certified pure; no automatic repeat/preview. |
| `improve_model` / home | write; improve | none | [dclab_rnd.draft.orchestrator:improve_model](../../dclab_rnd/draft/orchestrator.py#L410) | No automatic repeat; reconcile durable effects. |
| `improvement_ideas` / home | read; — | none | [dclab_rnd.draft.orchestrator:improvement_ideas](../../dclab_rnd/draft/orchestrator.py#L366) | Not certified pure; no automatic repeat/preview. |
| `list_projects` / home | read; — | none | [dclab_rnd.draft.orchestrator:list_projects](../../dclab_rnd/draft/orchestrator.py#L319) | Not certified pure; no automatic repeat/preview. |
| `plan_new_version` / home | write; plan_version | none | [dclab_rnd.draft.orchestrator:plan_new_version](../../dclab_rnd/draft/orchestrator.py#L778) | No automatic repeat; reconcile durable effects. |
| `propose_data_change` / home | write; propose_data_change | change | [dclab_rnd.draft.orchestrator:propose_data_change](../../dclab_rnd/draft/orchestrator.py#L1244) | No automatic repeat; reconcile durable effects. |
| `propose_problem_change` / home | write; propose_problem_change | none | [dclab_rnd.draft.orchestrator:propose_problem_change](../../dclab_rnd/draft/orchestrator.py#L610) | No automatic repeat; reconcile durable effects. |
| `remember` / home | write; remember | none | [dclab_rnd.draft.orchestrator:remember](../../dclab_rnd/draft/orchestrator.py#L979) | No automatic repeat; reconcile durable effects. |
| `show_activity` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_activity](../../dclab_rnd/draft/orchestrator.py#L352) | Not certified pure; no automatic repeat/preview. |
| `show_data` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_data](../../dclab_rnd/draft/orchestrator.py#L201) | Not certified pure; no automatic repeat/preview. |
| `show_evaluation` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_evaluation](../../dclab_rnd/draft/orchestrator.py#L276) | Not certified pure; no automatic repeat/preview. |
| `show_features` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_features](../../dclab_rnd/draft/orchestrator.py#L182) | Not certified pure; no automatic repeat/preview. |
| `show_improvements` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_improvements](../../dclab_rnd/draft/orchestrator.py#L378) | Not certified pure; no automatic repeat/preview. |
| `show_leakage` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_leakage](../../dclab_rnd/draft/orchestrator.py#L223) | Not certified pure; no automatic repeat/preview. |
| `show_memory` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_memory](../../dclab_rnd/draft/orchestrator.py#L998) | Not certified pure; no automatic repeat/preview. |
| `show_models` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_models](../../dclab_rnd/draft/orchestrator.py#L156) | Not certified pure; no automatic repeat/preview. |
| `show_notebook` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_notebook](../../dclab_rnd/draft/orchestrator.py#L337) | Not certified pure; no automatic repeat/preview. |
| `show_problem` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_problem](../../dclab_rnd/draft/orchestrator.py#L573) | Not certified pure; no automatic repeat/preview. |
| `show_result` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_result](../../dclab_rnd/draft/orchestrator.py#L249) | Not certified pure; no automatic repeat/preview. |
| `show_stages` / home | read; — | none | [dclab_rnd.draft.orchestrator:show_stages](../../dclab_rnd/draft/orchestrator.py#L238) | Not certified pure; no automatic repeat/preview. |
| `write_notebook_cell` / home | write; notebook_cell | request | [dclab_rnd.draft.orchestrator:write_notebook_cell](../../dclab_rnd/draft/orchestrator.py#L473) | No automatic repeat; reconcile durable effects. |
| `write_report` / home | read; — | none | [dclab_rnd.draft.orchestrator:write_report](../../dclab_rnd/draft/orchestrator.py#L1066) | Not certified pure; no automatic repeat/preview. |
| `adopt_plan` / project | write; set_solution | project_id, target, prediction_moment | [dclab_rnd.intern.tools:Toolbox.adopt_plan](../../dclab_rnd/intern/tools.py#L306) | Invalidates stage state through solution save; never use as a harmless planning check or blind repeat. |
| `approve_stage` / project | write; approve_stage | project_id, stage | [dclab_rnd.intern.tools:Toolbox.approve_stage](../../dclab_rnd/intern/tools.py#L500) | No automatic repeat; reconcile durable effects. |
| `ask_project` / project | read; — | project_id, question | [dclab_rnd.intern.tools:Toolbox.ask_project](../../dclab_rnd/intern/tools.py#L566) | Not certified pure; no automatic repeat/preview. |
| `check_move` / project | read; — | project_id, move | [dclab_rnd.intern.tools:Toolbox.check_move](../../dclab_rnd/intern/tools.py#L600) | Not certified pure; no automatic repeat/preview. |
| `create_project` / project | write; create_project | name, goal | [dclab_rnd.intern.tools:Toolbox.create_project](../../dclab_rnd/intern/tools.py#L256) | No automatic repeat; reconcile durable effects. |
| `decide` / project | read; — | point, state | [dclab_rnd.tools:decide](../../dclab_rnd/tools.py#L302) | Not certified pure; no automatic repeat/preview. |
| `decision_what_if` / project | read; — | project_id | [dclab_rnd.intern.tools:Toolbox.decision_what_if](../../dclab_rnd/intern/tools.py#L546) | Not certified pure; no automatic repeat/preview. |
| `describe_data` / project | read; — | project_id | [dclab_rnd.intern.tools:Toolbox.describe_data](../../dclab_rnd/intern/tools.py#L267) | Not certified pure; no automatic repeat/preview. |
| `export_notebook` / project | write; capture | project_id | [dclab_rnd.intern.tools:Toolbox.export_notebook](../../dclab_rnd/intern/tools.py#L573) | No automatic repeat; reconcile durable effects. |
| `find_examples` / project | read; — | words | [dclab_rnd.tools:find_examples](../../dclab_rnd/tools.py#L238) | Not certified pure; no automatic repeat/preview. |
| `get_charts` / project | read; — | project_id | [dclab_rnd.intern.tools:Toolbox.get_charts](../../dclab_rnd/intern/tools.py#L537) | Evidence readback; no refit/rescore; metadata writes possible. |
| `get_graph` / project | read; — | project_id | [dclab_rnd.intern.tools:Toolbox.get_graph](../../dclab_rnd/intern/tools.py#L593) | Evidence readback; no refit/rescore; metadata writes possible. |
| `get_record` / project | read; — | record_id | [dclab_rnd.tools:get_record](../../dclab_rnd/tools.py#L72) | Evidence readback; no refit/rescore; metadata writes possible. |
| `get_results` / project | read; — | project_id | [dclab_rnd.intern.tools:Toolbox.get_results](../../dclab_rnd/intern/tools.py#L504) | Evidence readback; no refit/rescore; metadata writes possible. |
| `get_rules` / project | read; — | none | [dclab_rnd.tools:get_rules](../../dclab_rnd/tools.py#L80) | Evidence readback; no refit/rescore; metadata writes possible. |
| `kind_checks` / project | read; — | kind | [dclab_rnd.tools:kind_checks](../../dclab_rnd/tools.py#L254) | Evidence readback; no refit/rescore; metadata writes possible. |
| `list_samples` / project | read; — | none | [dclab_rnd.intern.tools:Toolbox.list_samples](../../dclab_rnd/intern/tools.py#L253) | Evidence readback; no refit/rescore; metadata writes possible. |
| `plan_next_stage` / project | read; — | none | [dclab_rnd.tools:plan_next_stage](../../dclab_rnd/tools.py#L87) | Not certified pure; no automatic repeat/preview. |
| `propose_solution` / project | write; propose_solution | project_id, target | [dclab_rnd.intern.tools:Toolbox.propose_solution](../../dclab_rnd/intern/tools.py#L294) | No automatic repeat; reconcile durable effects. |
| `recall` / project | read; — | question | [dclab_rnd.memory.tools:_recall](../../dclab_rnd/memory/tools.py#L45) | Not certified pure; no automatic repeat/preview. |
| `review_code` / project | read; — | source | [dclab_rnd.tools:review_code](../../dclab_rnd/tools.py#L131) | Not certified pure; no automatic repeat/preview. |
| `review_labels` / project | read; — | project_id | [dclab_rnd.intern.tools:Toolbox.review_labels](../../dclab_rnd/intern/tools.py#L383) | Not certified pure; no automatic repeat/preview. |
| `review_leakage` / project | read; — | project_id, target | [dclab_rnd.intern.tools:Toolbox.review_leakage](../../dclab_rnd/intern/tools.py#L406) | Not certified pure; no automatic repeat/preview. |
| `run_all` / project | write; run_stage | project_id | [dclab_rnd.intern.tools:Toolbox.run_all](../../dclab_rnd/intern/tools.py#L484) | May include final and several stages; excluded from first task slice to prevent overlapping stage calls. No blind repeat. |
| `run_programme` / project | write; run_programme | project_id | [dclab_rnd.intern.tools:Toolbox.run_programme](../../dclab_rnd/intern/tools.py#L471) | Atomic programme may consume its one holdout; no independent leaf resume, no blind retry after ambiguous completion. |
| `run_stage` / project | write; run_stage | project_id, stage | [dclab_rnd.intern.tools:Toolbox.run_stage](../../dclab_rnd/intern/tools.py#L461) | Pre-final stages: conditional on input/version equality, no final consumed and reconciled prior attempt. final: no blind retry; ambiguous score/read outcome blocks. |
| `search_evidence` / project | read; — | query | [dclab_rnd.tools:search_evidence](../../dclab_rnd/tools.py#L64) | Not certified pure; no automatic repeat/preview. |
| `set_settings` / project | write; set_settings | project_id | [dclab_rnd.intern.tools:Toolbox.set_settings](../../dclab_rnd/intern/tools.py#L452) | No automatic repeat; reconcile durable effects. |
| `set_solution` / project | write; set_solution | project_id, target, task, prediction_moment | [dclab_rnd.intern.tools:Toolbox.set_solution](../../dclab_rnd/intern/tools.py#L423) | Invalidates stage state through solution save; never use as a harmless planning check or blind repeat. |
| `understand_problem` / project | read; — | project_id | [dclab_rnd.intern.tools:Toolbox.understand_problem](../../dclab_rnd/intern/tools.py#L391) | Not certified pure; no automatic repeat/preview. |
| `use_sample` / project | write; attach_data | project_id, key | [dclab_rnd.intern.tools:Toolbox.use_sample](../../dclab_rnd/intern/tools.py#L263) | No automatic repeat; reconcile durable effects. |
| `finish` / session | write; finish | report | [dclab_rnd.intern.loop:finish](../../dclab_rnd/intern/loop.py#L541) | No automatic repeat; reconcile durable effects. |
| `write_plan` / session | write; plan | plan | [dclab_rnd.intern.loop:write_plan](../../dclab_rnd/intern/loop.py#L536) | No automatic repeat; reconcile durable effects. |
| `compute` / workspace | read; — | table, op | [dclab_rnd.workspace_chat:compute](../../dclab_rnd/workspace_chat.py#L92) | Not certified pure; no automatic repeat/preview. |
| `describe_table` / workspace | read; — | table | [dclab_rnd.workspace_chat:describe_table](../../dclab_rnd/workspace_chat.py#L76) | Not certified pure; no automatic repeat/preview. |
| `list_tables` / workspace | read; — | none | [dclab_rnd.workspace_chat:list_tables](../../dclab_rnd/workspace_chat.py#L70) | Not certified pure; no automatic repeat/preview. |

## Flat projects, programmes and existing workflow readback

Flat actions are `run_stage(data → leakage → features → models → final)` through `Toolbox` and the original graph/engine. A programme is `Toolbox.run_programme → studio.programme.run → core.programme.execute`: one shared split, validation composition and one composition-level holdout read. Leaves defer their final step. They are not independently executable registered task capabilities. Keep one `programme_atomic` task until its owner exposes safe durable boundaries. `run_all` is excluded from the first slice because it overlaps individual stages.

`studio.workflow_run.build` reconstructs 12 conceptual evidence steps: lock_split, clean, baseline, features, candidates, evaluate, select, calibrate_rule, holdout_once, package, report, plan_monitoring. Candidates/evaluate/select share models; calibrate_rule/holdout_once share final. Package needs actual exports matching the current final run; a report marked done from a final record does not prove a report file exists. Monitoring may be partial/input-only. Do not turn these 12 rows into 12 side effects.

Sources: [studio/workflow_run.py](../../dclab_rnd/studio/workflow_run.py), `build`; [core/workflow_run.py](../../dclab_core/workflow_run.py), `WorkflowRun`; [studio/programme.py](../../dclab_rnd/studio/programme.py), `run`; [core/programme.py](../../dclab_core/programme.py), `execute`; [orchestra/solution.py](../../dclab_rnd/orchestra/solution.py), `SolutionTeam`. Storage, approval gaps and the exact Telco action slice are in [TASK_PLANNING_ARCHITECTURE.md](TASK_PLANNING_ARCHITECTURE.md).

# dclab-core

How DCLab understands a problem, analyses its data for decisions and plans a solution, as a pure Python library
shared by the DCLab R&D studio and DCLab's main product.

- **Depends on** numpy, pandas, scikit-learn and pydantic only; Python 3.10-3.13.
- **Never** trains a model, calls a model provider or touches a database: the host does, and adopts the proposals
  through its own governed decisions.
- **Outputs** are frozen, versioned contracts (`dclab_core.contracts`) with JSON Schemas in `schemas/`:
  `ProblemBrief`, `DataFindings`, `SolutionPlan` (of `Decision`s with reasons and alternatives), `Review`,
  `SolutionReport`.
- **Reports** (`dclab_core.report`): the `SolutionReport` of a run, written by code from its records (every number
  traced to the record path it was read from, no sentence calling a model production-ready), rendered to Markdown,
  HTML and PDF (`pip install "dclab-core[pdf]"` for the PDF).
- **Model help** only through an `Advisor` the host provides (`dclab_core.advisor`): compact evidence in, a validated
  answer or nothing out; without one, the deterministic path runs.
- **For the main product**, `dclab_core.adapters.main_product` maps every output onto its contracts (ProblemSpec,
  ProblemProfile, MetricPlan, HoldoutPlan, ValidationPlan, Objective, ExperimentPlan per decision point, the decision
  threshold, ExperimentReview, ExperimentChangeSet) from field lists recorded at one commit of its code
  (`adapters/main_product_fields.json`): proposals for its governed decision points, never applied directly.

Install from the R&D repository, pinned:
`pip install "dclab-core @ git+https://github.com/Shahriyar-Moradi/Research_Development_DCLab.git@<tag or commit>#subdirectory=dclab_core"`.
What changed between versions: `CHANGELOG.md`. How the main product adopts it: `docs/guides/CORE_INTEGRATION.md`.
Plan and status: `docs/guides/PRODUCT_BUILD_PLAYBOOK.md`, Part 6.

## Task planning contracts (TP-01B, 0.156.0)

`dclab_core.task_graph.TaskGraph` v1 is an immutable structural proposal: business objective/source references,
workflow and solution fingerprints, bounded tasks and explicit grouping/dependency edges. A task is a group,
action, human gate, evidence verification or blocked requirement. Only an action has a capability reference; groups
never call tools. Callable bindings on non-action kinds and blank blocker reasons are rejected in both Python and the published schemas.

`dclab_core.task_execution_plan.TaskExecutionPlan` v1 describes one task's bounded ordered actions, expected
fingerprinted inputs, declared outputs, selected capability references, evidence checks, budgets and stop/escalation
policy. Arguments are typed references to supplied inputs or outputs of an earlier declared action. There are no
literal-value arguments, expressions, arbitrary paths or executable snippets. Inputs such as a project ID and a
stage name are supplied separately by the host and pinned by fingerprint; a planner cannot invent their values.

All nested collections are tuples and all nested objects are frozen contracts. JSON still uses ordinary arrays and
objects. Read persisted documents with `TaskGraph.read(document)` / `TaskExecutionPlan.read(document)`. Both new
contracts also refuse missing `schema_version` during normal construction and `model_validate`; this stricter rule
is local to them and does not change older contracts. Import the classes from their modules. Their modules register
them in `contracts.CONTRACTS`, and `schemas.generate()` explicitly imports them before generating the full registry.

Plan statuses are `draft`, `ready`, `noop`, `blocked`. `ready` describes local document shape, not permission to run.
`noop` and `blocked` require a reason and contain no actions, selected capabilities or outputs. None of these states
means measured task completion; mutable attempts and verifier outcomes belong to the host in later packages.

TP-01B adds **contracts only**. Its readers do not compute or authenticate supplied fingerprints, walk a graph,
resolve actual artifact values, validate real registry schemas/authorization or execute tasks. The separate
TP-01C helpers below provide semantic validation; TP-01D owns workflow conversion. Later guarded dispatch must check
current human approval, actual registry membership/schema/scope, input lineage and holdout/replay safety.
JSON Schemas express local shape/kind/status rules. Cross-reference uniqueness, ordered reference/type checks and
numeric comparisons between fields require Python validation or equivalent host semantic checks too.

The minimal shape examples and adversarial inputs are in `tests/test_task_graph.py` and
`tests/test_task_execution_plan.py`. Frozen current/baseline schemas ship inside the wheel. New contracts have no
field mapping in the pinned main-product inventory; adapter conformance records their library-only status without
changing any existing adapter output. Full package acceptance: `docs/guides/TASK_CONTRACTS_ACCEPTANCE.md` in the repository.

## Pure task validation (TP-01C, 0.157.0)

`dclab_core.task_fingerprints` computes `graph_fingerprint`, `plan_fingerprint`, `workflow_fingerprint` and
`capability_fingerprint` from JSON metadata. These functions return identities; they never repair a saved document.
`dclab_core.task_validation` exposes `validate_graph`, `ready_tasks` and `validate_execution_plan`. The source
workflow/map and current host snapshots must be supplied explicitly. No runtime registry, model, storage or network
is accessed. Source workflow/map validators are reused, including their mandatory safety content.

Graph checks cover references, cycles, artifacts, source coverage and ordering. Plan checks bind one action task to
its current graph, inputs, approved sources and capability descriptors. Readiness returns at most one action ID in
deterministic order. Verified host receipts are required for dependencies; a `done` flag is insufficient. Unknown,
failed or ambiguous attempts block progression. Existing outputs without a verified receipt require reconciliation.

The host supplies authenticated metadata, not model-written claims or table rows. It must recheck live permissions,
tool schemas, approvals, inputs, engine guards, budgets and holdout consumption immediately before execution.
No workflow conversion, host integration or execution is added here. Context formats, canonicalization rules,
runtime boundaries and verified acceptance are in `docs/guides/TASK_VALIDATION_ACCEPTANCE.md`.

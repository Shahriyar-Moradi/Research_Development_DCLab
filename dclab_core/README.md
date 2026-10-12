# dclab-core

How DCLab understands a problem, analyses its data for decisions and plans a solution, as a pure Python library
shared by the DCLab R&D studio and DCLab's main product.

- **Depends on** numpy, pandas, scikit-learn and pydantic only; Python 3.10-3.13.
- **Never** trains a model, calls a model provider or touches a database: the host does, and adopts the proposals
  through its own governed decisions.
- **Outputs** are frozen, versioned contracts (`dclab_core.contracts`) with JSON Schemas in `schemas/`:
  `ProblemBrief`, `DataFindings`, `SolutionPlan` (of `Decision`s with reasons and alternatives), `Review`,
  `SolutionReport`.
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

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

Install from the R&D repository: `pip install "dclab-core @ git+https://github.com/<owner>/<rnd-repo>.git@<tag>#subdirectory=dclab_core"`.
Plan and status: `docs/guides/PRODUCT_BUILD_PLAYBOOK.md`, Part 6.

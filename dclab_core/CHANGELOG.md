# Changelog of dclab-core

The library's version (`pyproject.toml` and `dclab_core.__version__`, kept equal by a test) moves when what a host
installs changes. Each contract keeps its own version (`dclab_core.problem_brief.v1`, ...): a contract's shape changes
only with its version, and its frozen schema is in `schemas/baseline/`. A host pins the library version it tested.

## 0.2.0 (2026-10-12)

Everything after the first release, built in packages 17.2 to 17.8 and 18.1 to 18.3 of
`docs/guides/PRODUCT_BUILD_PLAYBOOK.md`. The six contracts are unchanged: all still at v1.

- `understand`: a description and its files' column names give a `ProblemBrief`; narrow rules that give up rather than
  guess; the problem analyst (an advisor role) may fill only unknown fields, with quoted words that support the value;
  the brief checker turns a disagreement into an open question, never a changed field (17.2, 18.2).
- `analyse`: tables, a brief and the host's training rows give `DataFindings` for binary, multiclass and regression;
  target statistics on the training rows only, measured out of fold against chance; label files, keys that inflate
  labels, leaks by name, power, recomputation, a missing value that speaks and parts that add up to the target; the
  data investigator role reads names and kinds only (17.3, 18.3).
- `replicas`: seeded, invented data with planted traps and their acceptance decisions: `delivery_fraud`,
  `ticket_routing`, `delivery_time` (15.1, 17.3).
- `plan`: a `SolutionPlan` of decisions with reasons, cited findings and the alternatives set aside; `validate` and
  `override` (a person's override stands, a model's is checked) (17.4).
- `agents` and `prompts`: a role contract (`Role`, `run_role`, `ScriptedAdvisor`); prompts as files pinned by
  `prompts/HASHES.json` and shipped in the wheel (18.1).
- `assess`: what to check after the first model, per task: the threshold from a review budget or stated costs chosen
  on out-of-fold predictions, calibration, per-class figures, intervals and their coverage (17.5).
- `review`: weaknesses from out-of-fold errors, hypotheses that cite them, ranked next experiments, and the paired
  rule that keeps a change (`adopt`) (17.6).
- `behaviour`: fold-safe behaviour features of an entity (counts, repeats, shrunk rates, distances) (15.4).
- `adapters.main_product`: every output above mapped onto DCLab main product's contracts (`ProblemSpec` content,
  `ProblemProfile`, `MetricPlan`, `HoldoutPlan`, `ValidationPlan`, `Objective`, the decision threshold,
  `ExperimentPlan` per decision point, `ExperimentReview`, `ExperimentChangeSet`), field by field, from field lists
  recorded from its code at commit `7fa23b9` (`adapters/main_product_fields.json`); a field it cannot fill is `None`
  with a note. Proposals only: the main product adopts them through its governed decision points (17.8).
- `features`: a small feature language compiled into fold-safe transformers, and the feature engineer role (18.5).
- Roles: the solution planner (`plan.plan_with`, its proposals guarded and validated, 18.4) and the critic
  (`review.critique`, each hypothesis citing a weakness its experiment addresses, 18.6).
- `report`: the solution report from the records, every number traced to its record path; Markdown and HTML, and
  PDF with the `pdf` extra (17.7).
- `challenge`: a brief read into its parts and their kinds, requirements with rubric points, and confidentiality (16.13).
- Packaging: the wheel carries the adapters' field lists and this changelog; it is built and used from a clean Python
  3.12 environment by `make core-wheel-check`.

## 0.1.0 (2026-10-12)

The first release (package 17.1).

- `contracts`: six frozen, versioned pydantic contracts (`ProblemBrief`, `DataFindings`, `SolutionPlan`, `Decision`,
  `Review`, `SolutionReport`) that refuse unknown fields and NaN and are read only with their version
  (`Versioned.read`); typed facts (stated with a quote, inferred with a reason, unknown without a value).
- `schemas`: the published JSON Schemas (`make core-schemas`) and each version's frozen shape in `schemas/baseline/`.
- `evidence`: what may be called evidence, a compact summary, never rows.
- `advisor`: the `Advisor` protocol a host implements over its own model gateway; `consult` checks the evidence and
  returns a validated answer or nothing.
- Dependencies: numpy, pandas, scikit-learn and pydantic (2.10 or later) only; Python 3.10 or later.

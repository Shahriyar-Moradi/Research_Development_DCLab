# Changelog of dclab-core

The library's version (`pyproject.toml` and `dclab_core.__version__`, kept equal by a test) moves when what a host
installs changes. Each contract keeps its own version (`dclab_core.problem_brief.v1`, ...): a contract's shape changes
only with its version, and its frozen schema is in `schemas/baseline/`. A host pins the library version it tested.

## 0.3.0 (2026-10-13)

Packages 16.2, 16.9, 15.9, 18.7, 18.8 and 16.3 to 16.7. The six contracts are unchanged: all still at v1.

- `decide` (new): decisions on top of a model. `review_budget` chooses the share of rows to review from the value of
  a case found and the cost of a review, on out-of-fold scores, with a bootstrap interval over rows or entities, read
  once on the holdout. `price_curve` models acceptance by price inside folds and gives the expected margin at each
  price multiple of a reference, selling or paying, with the best multiple and a bootstrap interval.
  `price_variation` refuses to draw a curve when prices were not varied: too few levels, too little spread, or prices
  set by a formula of the other columns. Every result states its assumption (16.9).
- `forecast` (new): many series forecast a horizon ahead (16.3). `backtest` cuts rolling origins with an optional gap, fits
  each model on the target up to the origin only (each origin records the last target date it read, and every forecast
  is returned with its dates and values), and compares naive, seasonal naive, a moving average, one boosted model for
  all series and a smoothing model per series by MASE and sMAPE. Intervals by horizon step come from residuals scaled by
  each series' own error; their coverage and pinball loss are measured on the last origin from the earlier ones, for
  the boosted model and the seasonal naive. `covariate_check` keeps the covariates declared known ahead and those that
  describe a series, keeps the others out, and marks a declared one that moves with the target "to confirm".
- `survival` (new): time to event with right-censoring (16.4). `kaplan_meier` (also per group; None past the longest
  follow-up), `censoring_bias` (survival at each horizon honest, with censored rows dropped, and with them counted as
  events), `duration_check` (columns that move with the duration among rows still going and rows that ended, or are
  filled only when the event happens), `censoring_check`, and `evaluate`: rows split by start day, training censored
  at the holdout's first day, a model chosen on the latest training rows, then Kaplan-Meier, Cox (ridge, Breslow) and
  discrete-time logistic and boosted hazards scored once on the latest starts by C-index, the integrated Brier score
  and Brier and time-dependent AUC at the horizons the follow-up reaches, weighted by the scored rows' own censoring;
  the chosen model is also fitted with censored rows dropped and counted as events, to show the bias.
- `replicas.subscription_lifetimes` (new): signups over two years, Weibull lifetimes, right-censored where the data
  ends, two columns counted during the follow-up, the true survival and every true lifetime in its facts (for judging).
- `ranking` (new): what to show each user next (16.5). `evaluate` reads the interactions up to the recommendation
  moment and the items created by it; it chooses a model at a validation origin one window earlier, then scores every
  model once on the window after the moment: popularity, recent popularity, matrix factorisation (dense, bounded) and
  a pointwise boosted ranker on candidates (trained on the window before an origin, its features ending before that
  window; item columns that track interactions kept out of it) by NDCG@k and recall@k (each against popularity, with a
  paired 90% bootstrap interval over users), coverage and novelty, for users with and without a history. `item_check`
  finds item columns that know counts after the moment (beyond everything the log knew then, for any event, on the raw
  or log scale) and the items created after it. `choose` never picks the popularity baseline. Relevant events may be a set.
- `replicas.shop_interactions` (new): a shop's views and purchases with category tastes, items arriving later, and an
  item column of lifetime sales.
- `segment` (new): groups in a table with no outcome (16.6). `prepare` records every preprocessing choice (log scale
  for skewed positive numbers, median fills, standardising, one-hot scaled by 1/sqrt(levels) with a missing level,
  identifiers and constants left out); `segment` refits k-means and a regularised diagonal Gaussian mixture on bootstrap
  resamples for each k, keeps a k only when its stability (adjusted Rand against the full fit, the lower end of a 90%
  interval) holds and its silhouette rises above the one the same method reaches on data with no groups (one Gaussian
  with the data's spread, a flat box over its ranges), chooses by silhouette, leaves out columns no plausible grouping
  of a first pass uses and searches again, adds an HDBSCAN density view, profiles each segment in the data's columns,
  describes it with a named outcome (numbers or two levels in words; never used to form it), and states that there is
  no ground truth.
- `replicas.customer_behaviours` (new): three planted buying behaviours, a tenth of customers following none, two
  columns that carry no behaviour, the planted labels and a later outcome in its facts (for judging).
- `uplift` (new): who responds to an action (16.7). `binary` reads a two-valued column (1/0, yes/no, true/false) or
  refuses it by name; `overlap` (out-of-fold gradient-boosted propensities, their AUC, the share inside a band that
  allows a small control arm, imbalanced columns); `wording` (effects only with randomisation the columns do not
  contradict; effects under an assumption with balanced overlap; differences between unlike arms; associations
  otherwise, said in the record); `evaluate`: a T-learner, an X-learner and a transformed-outcome uplift tree against a
  response model and random targeting, the model chosen on a validation cut of the training rows by the Qini
  coefficient, then the Qini coefficient and the AUUC on a holdout split once by arm, each with a paired 90% interval,
  and the uplift by decile; arms weighted by inverse propensity when the action was not random.
- `replicas.campaign_uplift` (new): a discount campaign, randomised or chosen by the team, a planted responsive
  segment and every row's true uplift in its facts (for judging).
- `replicas.store_demand` (new): daily units per store and product, a planned promotion known ahead, the day's rain not.
- `replicas`: `subscription_churn` (binary), `product_reviews` (text + tabular) (16.2), and `ride_pricing` (fees
  offered to couriers, varied or set by a formula, with the true best multiple) (16.9).
- `report.REPORTER` and `report.write_prose` (18.7): a model's summary above a section's lines, labelled as a model's, each
  sentence kept only when the host's check and the report's own stricter rules pass it; nothing kept without a check.
- `plan`: rows that arrive over time get a time split even when no drift is measured (DCLAB-R02); a random split only when no time orders them (18.8).
- `understand.label_coverage`: "partial" when a sentence about the labels says unmarked rows are unknown (15.9).
- `report`: the decision, when a run made one, in "Results, with their intervals", every number traced to
  `final.evidence.decision`; a requirement asking for a price or a fee maps to it.

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

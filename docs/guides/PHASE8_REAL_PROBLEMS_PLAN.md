# Phase 8: real machine-learning problems solved by the teams, and a framework that solves more

The owner's direction (2026-10-18): after Phase 6's teams and Phase 7's students, solve real problems with the teams, each
a separate measured run, fraud first, then Telco churn, then HyperAck; improve the frameworks so they solve more and
bigger problems and more kinds of problem; and build a large knowledge base of documents and real business examples that
helps problem understanding, then data understanding, then solution understanding and the solution workflow.

The rules of the repository hold: the leakage rules first (CLAUDE.md, rule 1), the holdout read once, the person decides
at the three pauses, a model's output is evaluation material by default (silver labels only on real projects, tagged and
kept apart), every live run planned, printed and capped before any request (the $17 left of the owner's $30 for Phases
7 and 8), and everything a run does captured whole before anything is cut.

## Part A. The three problems and what each run must show

| Problem | Data | Privacy | What the run is judged on |
|---|---|---|---|
| Delivery fraud (the owner's) | `Snap_fruad/data/part2-dataset.csv` (134,362 rides) and `part2-fraud-sample.csv` (100 known forged rides; 94 join on `order_request_id`) | **private**: the README says the data must not be distributed; no model sees anything of it, not even its column names (a private project: the rules serve every role) | the understanding walk reaches the right card from the owner's morning words (two files, labels joined on the order, not on the customer), the solution team builds the programme, the auditor screens it, the holdout is read once; set beside the owner's own complete solution (average precision 0.095 for a random forest, 0.081 for PU bagging, recall in the top 0.5% 46%, biker-grouped folds), with the protocols' difference said |
| Telco churn | `data/project/telco` (7,043 customers, `Churn`) | public (Kaggle) | through both teams with the rules and with Claude's arms; set beside `research/churn-prediction`'s champion on its own protocol |
| HyperAck | `data/project/hyperack` (11,118 orders, `hyper_ack`) | the project's own | the auditor must catch `final_customer_fare` and `final_biker_fare` (written after the outcome) before any holdout; set beside `research/tabular-classification`'s champion |

Each run is a separate result in `evidence/campaigns/real_problems_v1/` (`RPB-…`): what was asked and answered, the
cards, the plan, the audit verdicts, the numbers with their intervals, the cost, and the comparison with the earlier
solution, with its limits. The records of each run are kept whole in `agent_runs/live_tests/<date>_<problem>/`.

## Part B. The packages (30.x)

### 30.1 The solution team on a project of one problem
- **Delivers:** the team's single-problem path: the engine's stages one by one through the validator as an agent
  (`engine.execute`), the auditor's screen on the training rows of the stages' own split before the final stage, the
  person's approval of the holdout as the gate, the report from the stage records.
- **Done when:** the dev journeys' one-problem kinds build through the team with the parts the engine's journeys reach;
  a planted leak in a one-problem table is sent back; the holdout is read once (a test counts).

- **Status: built (2026-10-18).** The team's single-problem path: the engine's stages through the registry's `run_stage`
  as an agent; the auditor after the leakage stage reads that stage's own screen of the training rows (an exact proxy, an
  arithmetic identity, an extreme univariate signal, a name that says so); a keep-out goes into the solution's forbidden
  list through `set_solution` (the stages run again); the final stage after the person approves reading the holdout. A
  planted leak, unticked by the person, is sent back and kept out, and the holdout read once (a test).

### 30.2 Delivery fraud, end to end (private)
- **Delivers:** the owner's morning (the words, the two files) through the understanding walk and the solution team, on
  a private project; the comparison with `Snap_fruad/complete_solution`.
- **Done when:** the run's result is stored with its comparison and limits, nothing of the data left the Mac (a test of
  the run's model requests: none), and what went wrong is a finding with its fix or a package of its own.

### 30.3 Telco churn, the rules and the models
- **Delivers:** the walk and the team on Telco with the rules arm and the Claude arms (reader, critic, map roles),
  capped; the comparison with the churn track's champion.

- **Status: run (2026-10-18).** `python -m dclab_rnd.real_problems telco` (the rules; no model): RPB-004 on the whole table:
  logistic regression, holdout ROC-AUC 0.842 (95% 0.821 to 0.865), average precision 0.631, 1,409 holdout rows, the
  cross-validation to holdout gap -0.004; the churn track's champion (0.850 on its own protocol) inside the interval.
  RPB-001 to RPB-003 are the same run on the product's quick pass (3,000 rows) and its two fixes (the runner set the
  table's rows before the Solution save reset them). The Claude arms on Telco are next.

### 30.4 HyperAck, the auditor's real test
- **Delivers:** the walk and the team on HyperAck; the auditor's verdict on the fares; the comparison with the tabular
  track's champion on the same leak-free contract.

- **Status: run (2026-10-18).** RPB-005: the proposal's audit forbade `deliverey_category_id` as "separating the classes"
  at an AUC of 0.816; accepted unread, the holdout ROC-AUC was 0.668. RPB-006: the person unticks it on the Solution page
  (the category is chosen when the order is created): extra trees, holdout ROC-AUC 0.934 (95% 0.923 to 0.946), average
  precision 0.937, the latest 2,219 orders by creation date; the final fares stayed forbidden (the proposal's precedent
  LEAK-hyperack) and the auditor passed. The track's 0.9455 (its own protocol) at the interval's top. Found: the
  proposal's bar for "separates the classes" is too low (0.8): 30.6 measures it on the studied samples.

### 30.5 The business-examples knowledge base
- **Delivers:** a knowledge base of real business examples, each written as problem → data → solution → workflow,
  with its pitfalls and its evidence: from the repository's own research tracks and campaigns, the owner's complete
  fraud solution (its report), the replicas' planted traps, and public case studies (each cited to its source; an
  example Claude wrote is marked as written, never as a record); indexed by the evidence search the retrieval
  measurement chose; read by the understanding critic and the solution designer as retrieval context (a node declares
  the tool), measured on the journeys before it serves.
- **Done when:** the knowledge base has its index and its retrieval questions, the critic and the designer read it in an
  arm of the journeys, and the arm is kept only where it measured better.

- **Status: the base built (2026-10-18); its use measured next.** `make business-examples`: 38 examples in
  `evidence/business_examples/examples.jsonl` (5 research tracks' contracts and champions, 16 replicas with the traps they
  plant, the owner's fraud solution, 16 written from public knowledge across insurance, lending, retail, manufacturing,
  software, sales, support, logistics, mobility, e-commerce, marketing, payments, healthcare, energy, HR and site
  selection); `dclab_core.business_examples` (dclab-core 0.73.0): the contract and a search without a model (the owner's
  fraud words find the fraud replicas first). Left: the examples as a tool the critic and the designer read, measured as
  an arm of the journeys before it serves; more examples from the 50-experiment campaign.

### 30.6 Bigger and more problems
- **Delivers:** what the three runs and the sealed programme journeys (2 of 10 for the rules, JPR-003) show is missing,
  each as a package: larger tables (sampling and chunked reading under the engine), more parts per programme, more
  kinds of problem in the router's approaches.
- **Done when:** the sealed programme journeys are run once more on a fresh sealed set after the fixes, with the gate
  fixed before.

## Status

- **Plan (2026-10-18):** written. Phase 6's teams (28.1 to 28.5) are built; 28.7 measures them live first.

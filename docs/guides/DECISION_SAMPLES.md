# Decision samples: the training data of DCLab's own decision models

Package A6.5 (`dclab_rnd/decisions`). Every typed decision the studio makes is kept with the evidence it was made
on, and every answer a person gives to it is kept beside it. That is the data a decision model needs: a classifier
that answers the small questions a data scientist answers at each step (is this column an identifier, is it a leak
suspect, which split, which family), the role a hosted model such as Jev plays in the main product, and the role an
open model such as Laya or a model trained here can play once there is enough of this data.

## What a row is

**A sample** is one decision at one decision point about one subject, written where the rules compute it:

| Where | Decision points (`schema.POINTS`) | Subject |
|---|---|---|
| the problem brief (Home's chat, the intern's `adopt_plan`) | `contract.task`, `contract.target_column`, `contract.prediction_moment`, `contract.time_column`, `contract.label_coverage` | the problem |
| the solution proposal (the proposal route, the intern's `propose_solution` and `adopt_plan`): dclab_core's findings and plan | `column.forbidden` (exclude / confirm / clear), `column.identifier`, `column.missing_action` per column; `split.strategy`, `model.families`, `data.cleaning`, `task.framing`, `labels.coverage`, `metric.primary`, `baseline`, `features.direction`, `threshold.policy` | a column, the table, the problem |
| the data stage (WF-03) | `column.role`, `column.risk` per column; `split.strategy` | a column, the table |
| the leakage stage (WF-05) | `column.leakage` (exclude / review_flag / clear) per column | a column |
| the features, models and final stages (WF-06 to WF-08) | `feature.recipe`, `model.family`, `optimization.config`, each with the options the rule ranked | the table |

Each sample carries: the point and subject; `features` (the aggregates the rule saw: rates, ratios, counts, fold
scores, flags, option tables; and column names); `state_text` (the same evidence as one line, what a model reads);
`rule_answer` and `rule_version`; `alternatives`; `evidence_partition` (`metadata` or `train`, never the holdout);
`dataset_ref` (the first 16 characters of the table's digest, never its file name); the project or draft id; and a
`key` (`point:subject`) that a label attaches to.

**A label** is what a person did: a stage approved (the rule's choice stands) or overridden with another option; a
solution saved (every column judgment it states on the first save, only what changed afterwards; the target, task,
metric, time column, split and whether a prediction moment was stated); a gate approved. A label names the latest
sample with its key when one exists, and always carries the key, so a label written before its sample (a solution
saved before the proposal ran) still joins at export. Every label today is `human_exposed`: the person saw the rule's
answer first, so these labels train and monitor a model and never promote one (ADR 0008 of the main product). An
agent's approval is not a label.

## What never enters a row

- A cell value: `schema.CELL_KEYS` (previews, row values, a row's dates, class values, quotes, statements) are dropped
  from any evidence at any depth; the per-column features come from a whitelist of aggregates.
- A person's words: a brief sample keeps whether a field was quoted and how many words, not the words; a forbidden
  column's reason and a gate's reason stay in the audit; a label's reason is one of `schema.REASONS`.
- The holdout: no holdout metric, gap, calibration bin or threshold row is read; the one holdout statistic the engine
  itself reads (a feature-distribution PSI) is dropped too, so `train` means train.
- A user id in the export; project and draft ids are masked there.

`tests/test_decision_samples.py` runs the engine on the churn table and searches every row for a customer id, a
preview key and the holdout ROC-AUC.

## Where rows live

Like every other log of the workspace: `decision_samples.jsonl` and `decision_labels.jsonl` in the workspace folder,
or the `decision_sample` and `decision_label` tables when `DCLAB_DATABASE_URL` is set (migration 0015;
`python -m dclab_rnd.storage migrate` moves the files). Rows are written for every project, opted in or not, and
stay in the workspace; they leave it only by the owner's export. A deleted project takes its rows and its draft's
brief rows with it. A write that fails is reported on stderr and the decision it recorded stands.

## Reading the data

```bash
python -m dclab_rnd.decisions stats  --home agent_runs              # counts per point, how many carry a label
python -m dclab_rnd.decisions export --home agent_runs --out out/decisions
```

The export writes `samples.jsonl` (one line per sample, with `labels` and the latest human `label`) and
`manifest.json` (counts per point, per label value, per split; the rule's agreement with the person per point; the
builder's commit). The split is by dataset digest, 20% sealed, 20% val, 60% train, so every column of a table is in one
split; the sealed split is scored once per evaluation run, as the agent benchmark does.

## What comes next

The project's `claude/DCLab_Decision_Model_Plan.md` has the plan: builders that turn public datasets, the replicas and
the sample factory into samples with labels known by construction; a blind label queue (the person judges without
seeing the rule's answer, the labels the promotion gates need); a classical model per point, then a small language
model; the R3 gates and a third backend behind the main product's semantic port.

# Training sample preview

One real sample per stage and decision, copied from `out/all_stages/train.jsonl`. The system prompt is the same in every sample and is shown once.

**Totals:** train 2,243 · val 300 · test 728

**System prompt:**

> You are the DCLab workflow agent. You build ML models step by step: data_understanding -> leakage_audit -> feature_engineering -> model_selection -> optimization_reliability. At each step you decide using only the evidence given, choose exactly one allowed action, explain why with the numbers, cite the rule, and say what happens next.

## problem_understanding

### `problem_understanding`  (store_demand, replica)

**Input**

```
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
Our stores reorder stock every week. We want to forecast how much each store will sell of each product. The forecast drives the stock order. Each row is one store, product and day. The target is units, the units sold that day. We forecast each Sunday evening, for the next 7 days of each store and product. The forecast covers the next 7 days. Each store and product pair is its own series, named in the series column.

FILES (headers only):
sales.csv: store, product, date, on_promotion, rain_mm, units, series
```

**Answer**

```
{
 "goal": {
  "value": "forecast sales per store and product",
  "status": "stated",
  "quote": "We want to forecast how much each store will sell of each product."
 },
 "action": {
  "value": "forecast drives the stock order",
  "status": "stated",
  "quote": "The forecast drives the stock order."
 },
 "unit": {
  "value": "store, product and day",
  "status": "stated",
  "quote": "Each row is one store, product and day."
 },
 "task": {
  "value": "regression",
  "status": "inferred",
  "reason": "units is a count of items sold, so the answer is a quantity"
 },
 "target_column": {
  "value": "units",
  "status": "stated",
  "quote": "The target is units, the units sold that day."
 },
 "label_source": {
  "value": "column",
  "status": "inferred",
  "reason": "the label is a column of sales.csv"
 },
 "label_coverage": {
  "value": "complete",
  "status": "inferred",
  "reason": "the label is a column of the main table, so every row is labelled"
 },
 "prediction_moment": {
  "value": "each Sunday evening",
  "status": "stated",
  "quote": "We forecast each Sunday evening, for the next 7 days of each store and product."
 },
 "time_column": {
  "value": "date",
  "status": "inferred",
  "reason": "date is the only date or time column in sales.csv"
 },
 "horizon": {
  "value": "the next 7 days",
  "status": "stated",
  "quote": "The forecast covers the next 7 days."
 },
 "entities": {
  "value": [
   "series"
  ],
  "status": "stated",
  "quote": "Each store and product pair is its own series, named in the series column."
 },
 "unknown": [
  "primary_metric"
 ],
 "constraints": [],
 "open_questions": []
}
```

## data_analysis

### `data_analysis`  (adult, public)

**Input**

```
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

PROBLEM CONTRACT:
{"goal": {"value": "a person earns more than $50K per year", "status": "stated", "quote": "We need to predict whether a person earns more than $50K per year."}, "action": {"value": "estimate guides survey follow-up", "status": "stated", "quote": "The estimate guides a survey's follow-up sampling."}, "unit": {"value": "person", "status": "stated", "quote": "Every row describes a person."}, "task": {"value": "binary", "status": "inferred", "reason": "high_income is 1 or 0, so the answer is binary"}, "target_column": {"value": "high_income", "status": "stated", "quote": "The target column is high_income."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of adult.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "once the census record is filed", "status": "stated", "quote": "We score each person from their census record, once the record is filed."}, "unknown": ["time_column", "horizon", "entities", "primary_metric"], "constraints": [], "open_questions": [{"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

TABLE: adult — 3,000 rows, 15 columns (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, high_income)
TRAINING PARTITION: a random 75% of rows (2,250 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 3000, "unique_columns": []}
- F-prevalence [target] evidence: {"positive_class": "1", "positive_rate": 0.24622, "positives": 554, "rows": 2250}
```

**Answer**

```
{
 "task": "binary",
 "target": "high_income",
 "training_rows": 2250,
 "findings": [
  {
   "id": "F-keys",
   "severity": "info",
   "statement": "adult has 3000 rows; 0 column(s) identify rows: none.",
   "implication": "A column that identifies rows can only memorise: keep it out of the features.",
   "affects": [
    "features.identifiers"
   ]
  },
  {
   "id": "F-prevalence",
   "severity": "info",
   "statement": "The positive class (1) is 24.62% of training rows.",
   "implication": "Average precision or ROC AUC rank the cases; accuracy depends on the threshold.",
   "affects": [
    "metric.primary",
    "threshold.policy",
    "split.strategy"
   ]
  }
 ]
}
```

## solution_design

### `solution_design`  (spambase, public)

**Input**

```
STAGE: solution_design
Solution design: from the problem contract and the data findings, decide the solution plan (one decision per key: task framing, labels, split, metrics, baseline, forbidden features, identifiers, feature direction, cleaning, model families, threshold), each with its reason, the findings it cites and the alternatives set aside. Then lay the decisions on the DCLab workflow WF-01…WF-10 with its gates. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "an email is spam", "status": "stated", "quote": "We need to predict whether an email is spam."}, "action": {"value": "flagged emails go to the junk folder", "status": "stated", "quote": "Emails flagged as spam go to the junk folder."}, "unit": {"value": "an email", "status": "inferred", "reason": "each row of spambase.csv describes one an email"}, "task": {"value": "binary", "status": "inferred", "reason": "spam is 1 or 0, so the answer is binary"}, "target_column": {"value": "spam", "status": "stated", "quote": "The spam column is 1 for a spam email."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of spambase.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "when the email is received", "status": "stated", "quote": "We classify each email when it is received."}, "primary_metric": {"value": "AUC", "status": "stated", "quote": "We care most about AUC."}, "unknown": ["time_column", "horizon", "entities"], "constraints": [], "open_questions": [{"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA FINDINGS:
- F-duplicates (warning, grain): 194 rows of spambase are exact copies of another row. Implication: Drop the copies before splitting, or keep each copy on the same side: a copy on both sides scores the model on rows it trained on. Evidence: {"duplicate_rows": 194, "rows": 3000, "share": 0.0647}
- F-keys (info, grain): spambase has 3000 rows; 0 column(s) identify rows: none. Implication: A column that identifies rows can only memorise: keep it out of the features. Evidence: {"rows": 3000, "unique_columns": []}
- F-prevalence (info, target): The positive class (1) is 39.56% of training rows. Implication: Average precision or ROC AUC rank the cases; accuracy depends on the threshold. Evidence: {"positive_class": "1", "positive_rate": 0.39556, "positives": 890, "rows": 2250}
```

**Answer**

```
{
 "task": "binary",
 "decisions": [
  {
   "key": "task.framing",
   "choice": {
    "task": "binary"
   },
   "cites": [],
   "reason": "The problem is binary. The brief says so (inferred).",
   "alternatives": []
  },
  {
   "key": "labels.coverage",
   "choice": "complete",
   "cites": [
    "F-prevalence"
   ],
   "reason": "Nothing says the labels are a sample; every row with a label is treated as labelled.",
   "alternatives": []
  },
  {
   "key": "split.strategy",
   "choice": {
    "kind": "stratified",
    "holdout": "locked first, used once, after the choice"
   },
   "cites": [],
   "reason": "No entity repeats strongly and no time orders the rows: a stratified split is fair.",
   "alternatives": []
  },
  {
   "key": "metric.primary",
   "choice": "roc_auc",
   "cites": [
    "F-prevalence"
   ],
   "reason": "The brief names roc_auc. Positives are 39.56% of rows: ROC AUC ranks well at this balance.",
   "alternatives": [
    {
     "choice": "accuracy",
     "why_not": "depends on the threshold and on the class balance"
    },
    {
     "choice": "f1",
     "why_not": "measured at one threshold, not the ranking"
    }
   ]
  },
  {
   "key": "metric.secondary",
   "choice": [
    "average_precision",
    "brier"
   ],
   "cites": [
    "F-prevalence"
   ],
   "reason": "Reported beside the primary metric, so one number does not hide a weakness.",
   "alternatives": []
  },
  {
   "key": "baseline",
   "choice": [
    "majority class",
    "random ranking (average precision = prevalence)"
   ],
   "cites": [
    "F-prevalence"
   ],
   "reason": "Every model is compared with what needs no model: a score only means something above these.",
   "alternatives": []
  },
  {
   "key": "features.forbidden",
   "choice": [],
   "cites": [],
   "reason": "No column looks written after the moment by name, power or recomputation; the moment still decides.",
   "alternatives": []
  },
  {
   "key": "features.identifiers",
   "choice": [],
   "cites": [
    "F-keys"
   ],
   "reason": "A column unique per row can only memorise: it identifies, it does not describe.",
   "alternatives": []
  },
  {
   "key": "features.direction",
   "choice": [
    "the raw columns, then the feature ladder's recipes"
   ],
   "cites": [],
   "reason": "Where the findings point: behaviour over repeated entities, time, and missingness that speaks.",
   "alternatives": []
  },
  {
   "key": "data.cleaning",
   "choice": [
    "drop the 194 exact copies before the split"
   ],
   "cites": [
    "F-duplicates"
   ],
   "reason": "What the data needs before any model sees it.",
   "alternatives": []
  },
  {
   "key": "model.families",
   "choice": [
    "dummy",
    "logistic_regression",
    "random_forest",
    "gradient_boosting"
   ],
   "cites": [
    "F-prevalence"
   ],
   "reason": "Screen linear and tree families on identical folds, with a dummy baseline always in.",
   "alternatives": []
  },
  {
   "key": "threshold.policy",
   "choice": {
    "kind": "ask",
    "default": "report the ranking; choose the threshold with the person"
   },
   "cites": [
    "F-prevalence"
   ],
   "reason": "No capacity or cost is known yet: the ranking is reported and the threshold is chosen with the person.",
   "alternatives": []
  }
 ],
 "risks": [],
 "would_change": [
  "If the labels turn out to be a sample, unlabelled rows are no longer negatives.",
  "A cost of errors or a review capacity would set the threshold and may change the primary metric."
 ],
 "workflow": [
  {
   "step": "WF-01 Solution draft",
   "rules": [
    "DCLAB-R01",
    "DCLAB-R18"
   ],
   "decisions": [
    "task.framing",
    "labels.coverage",
    "metric.primary",
    "metric.secondary",
    "baseline",
    "threshold.policy"
   ]
  },
  {
   "step": "WF-02 Source and lineage",
   "rules": [
    "DCLAB-R09",
    "DCLAB-R21"
   ],
   "does": "Record each file's SHA-256 and where it came from; nothing is analysed before lineage is written."
  },
  {
   "step": "WF-03 Split design",
   "rules": [
    "DCLAB-R02",
    "DCLAB-R17"
   ],
   "decisions": [
    "split.strategy"
   ]
  },
  {
   "step": "WF-04 Train-only EDA",
   "rules": [
    "DCLAB-R03",
    "DCLAB-R12"
   ],
   "decisions": [
    "data.cleaning"
   ],
   "does": "Profile the training partition only; every statistic that steers a choice is fitted there.",
   "gate": "The owner signs the solution before WF-04."
  },
  {
   "step": "WF-05 Leakage audit",
   "rules": [
    "DCLAB-R04",
    "DCLAB-R05",
    "DCLAB-R06"
   ],
   "decisions": [
    "features.forbidden",
    "features.identifiers"
   ]
  },
  {
   "step": "WF-06 Feature ladder",
   "rules": [
    "DCLAB-R07",
    "DCLAB-R08",
    "DCLAB-R10",
    "DCLAB-R11"
   ],
   "decisions": [
    "features.direction"
   ]
  },
  {
   "step": "WF-07 Algorithm screen",
   "rules": [
    "DCLAB-R13",
    "DCLAB-R14"
   ],
   "decisions": [
    "model.families"
   ]
  },
  {
   "step": "WF-08 Conservative optimization",
   "rules": [
    "DCLAB-R15",
    "DCLAB-R16"
   ],
   "does": "Tune conservatively inside training folds, against the paired baseline."
  },
  {
   "step": "WF-09 Reliability challenge",
   "rules": [
    "DCLAB-R17",
    "DCLAB-R19",
    "DCLAB-R22"
   ],
   "does": "Open the locked holdout once, after every choice is fixed, and challenge the result by slices.",
   "gate": "The owner approves opening the holdout (WF-09)."
  },
  {
   "step": "WF-10 Knowledge capture",
   "rules": [
    "DCLAB-R20",
    "DCLAB-R21"
   ],
   "does": "Capture the notebook, the report and the training examples."
  }
 ]
}
```

## connected_problem_to_plan

### `connected_problem_to_plan`  (adult, public)

**Input**

```
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
This project uses the adult data. We want to estimate which people earn more than 50K a year. The target column is high_income. We score each person from their census record, once the record is filed. We care most about AUC.

FILES (headers only):
adult.csv: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, high_income
```

**Answer**

```
{
 "goal": {
  "value": "estimate who earns more than 50K",
  "status": "stated",
  "quote": "We want to estimate which people earn more than 50K a year."
 },
 "unit": {
  "value": "a person",
  "status": "inferred",
  "reason": "each row of adult.csv describes one a person"
 },
 "task": {
  "value": "binary",
  "status": "inferred",
  "reason": "high_income is 1 or 0, so the answer is binary"
 },
 "target_column": {
  "value": "high_income",
  "status": "stated",
  "quote": "The target column is high_income."
 },
 "label_source": {
  "value": "column",
  "status": "inferred",
  "reason": "the label is a column of adult.csv"
 },
 "label_coverage": {
  "value": "complete",
  "status": "inferred",
  "reason": "the label is a column of the main table, so every row is labelled"
 },
 "prediction_moment": {
  "value": "once the census record is filed",
  "status": "stated",
  "quote": "We score each person from their census record, once the record is filed."
 },
 "primary_metric": {
  "value": "AUC",
  "status": "stated",
  "quote": "We care most about AUC."
 },
 "unknown": [
  "action",
  "time_column",
  "horizon",
  "entities"
 ],
 "constraints": [],
 "open_questions": [
  {
   "field": "action",
   "text": "What happens with each prediction, and roughly what does a wrong one cost?",
   "impact": 2
  },
  {
   "field": "entities",
   "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?",
   "impact": 3
  }
 ]
}
```

**Input**

```
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

TABLE: adult — 3,000 rows, 15 columns (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, high_income)
TRAINING PARTITION: a random 75% of rows (2,250 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 3000, "unique_columns": []}
- F-prevalence [target] evidence: {"positive_class": "1", "positive_rate": 0.24622, "positives": 554, "rows": 2250}
```

**Answer**

```
{
 "task": "binary",
 "target": "high_income",
 "training_rows": 2250,
 "findings": [
  {
   "id": "F-keys",
   "severity": "info",
   "statement": "adult has 3000 rows; 0 column(s) identify rows: none.",
   "implication": "A column that identifies rows can only memorise: keep it out of the features.",
   "affects": [
    "features.identifiers"
   ]
  },
  {
   "id": "F-prevalence",
   "severity": "info",
   "statement": "The positive class (1) is 24.62% of training rows.",
   "implication": "Average precision or ROC AUC rank the cases; accuracy depends on the threshold.",
   "affects": [
    "metric.primary",
    "threshold.policy",
    "split.strategy"
   ]
  }
 ]
}
```

**Input**

```
STAGE: solution_design
Solution design: from the problem contract and the data findings, decide the solution plan (one decision per key: task framing, labels, split, metrics, baseline, forbidden features, identifiers, feature direction, cleaning, model families, threshold), each with its reason, the findings it cites and the alternatives set aside. Then lay the decisions on the DCLab workflow WF-01…WF-10 with its gates. Output JSON.

Use the contract and the findings above.
```

**Answer**

```
{
 "task": "binary",
 "decisions": [
  {
   "key": "task.framing",
   "choice": {
    "task": "binary"
   },
   "cites": [],
   "reason": "The problem is binary. The brief says so (inferred).",
   "alternatives": []
  },
  {
   "key": "labels.coverage",
   "choice": "complete",
   "cites": [
    "F-prevalence"
   ],
   "reason": "Nothing says the labels are a sample; every row with a label is treated as labelled.",
   "alternatives": []
  },
  {
   "key": "split.strategy",
   "choice": {
    "kind": "stratified",
    "holdout": "locked first, used once, after the choice"
   },
   "cites": [],
   "reason": "No entity repeats strongly and no time orders the rows: a stratified split is fair.",
   "alternatives": []
  },
  {
   "key": "metric.primary",
   "choice": "roc_auc",
   "cites": [
    "F-prevalence"
   ],
   "reason": "The brief names roc_auc. Positives are 24.62% of rows: ROC AUC ranks well at this balance.",
   "alternatives": [
    {
     "choice": "accuracy",
     "why_not": "depends on the threshold and on the class balance"
    },
    {
     "choice": "f1",
     "why_not": "measured at one threshold, not the ranking"
    }
   ]
  },
  {
   "key": "metric.secondary",
   "choice": [
    "average_precision",
    "brier"
   ],
   "cites": [
    "F-prevalence"
   ],
   "reason": "Reported beside the primary metric, so one number does not hide a weakness.",
   "alternatives": []
  },
  {
   "key": "baseline",
   "choice": [
    "majority class",
    "random ranking (average precision = prevalence)"
   ],
   "cites": [
    "F-prevalence"
   ],
   "reason": "Every model is compared with what needs no model: a score only means something above these.",
   "alternatives": []
  },
  {
   "key": "features.forbidden",
   "choice": [],
   "cites": [],
   "reason": "No column looks written after the moment by name, power or recomputation; the moment still decides.",
   "alternatives": []
  },
  {
   "key": "features.identifiers",
   "choice": [],
   "cites": [
    "F-keys"
   ],
   "reason": "A column unique per row can only memorise: it identifies, it does not describe.",
   "alternatives": []
  },
  {
   "key": "features.direction",
   "choice": [
    "the raw columns, then the feature ladder's recipes"
   ],
   "cites": [],
   "reason": "Where the findings point: behaviour over repeated entities, time, and missingness that speaks.",
   "alternatives": []
  },
  {
   "key": "model.families",
   "choice": [
    "dummy",
    "logistic_regression",
    "random_forest",
    "gradient_boosting"
   ],
   "cites": [
    "F-prevalence"
   ],
   "reason": "Screen linear and tree families on identical folds, with a dummy baseline always in.",
   "alternatives": []
  },
  {
   "key": "threshold.policy",
   "choice": {
    "kind": "ask",
    "default": "report the ranking; choose the threshold with the person"
   },
   "cites": [
    "F-prevalence"
   ],
   "reason": "No capacity or cost is known yet: the ranking is reported and the threshold is chosen with the person.",
   "alternatives": []
  }
 ],
 "risks": [],
 "would_change": [
  "If the labels turn out to be a sample, unlabelled rows are no longer negatives.",
  "A cost of errors or a review capacity would set the threshold and may change the primary metric."
 ],
 "workflow": [
  {
   "step": "WF-01 Solution draft",
   "rules": [
    "DCLAB-R01",
    "DCLAB-R18"
   ],
   "decisions": [
    "task.framing",
    "labels.coverage",
    "metric.primary",
    "metric.secondary",
    "baseline",
    "threshold.policy"
   ]
  },
  {
   "step": "WF-02 Source and lineage",
   "rules": [
    "DCLAB-R09",
    "DCLAB-R21"
   ],
   "does": "Record each file's SHA-256 and where it came from; nothing is analysed before lineage is written."
  },
  {
   "step": "WF-03 Split design",
   "rules": [
    "DCLAB-R02",
    "DCLAB-R17"
   ],
   "decisions": [
    "split.strategy"
   ]
  },
  {
   "step": "WF-04 Train-only EDA",
   "rules": [
    "DCLAB-R03",
    "DCLAB-R12"
   ],
   "does": "Profile the training partition only; every statistic that steers a choice is fitted there.",
   "gate": "The owner signs the solution before WF-04."
  },
  {
   "step": "WF-05 Leakage audit",
   "rules": [
    "DCLAB-R04",
    "DCLAB-R05",
    "DCLAB-R06"
   ],
   "decisions": [
    "features.forbidden",
    "features.identifiers"
   ]
  },
  {
   "step": "WF-06 Feature ladder",
   "rules": [
    "DCLAB-R07",
    "DCLAB-R08",
    "DCLAB-R10",
    "DCLAB-R11"
   ],
   "decisions": [
    "features.direction"
   ]
  },
  {
   "step": "WF-07 Algorithm screen",
   "rules": [
    "DCLAB-R13",
    "DCLAB-R14"
   ],
   "decisions": [
    "model.families"
   ]
  },
  {
   "step": "WF-08 Conservative optimization",
   "rules": [
    "DCLAB-R15",
    "DCLAB-R16"
   ],
   "does": "Tune conservatively inside training folds, against the paired baseline."
  },
  {
   "step": "WF-09 Reliability challenge",
   "rules": [
    "DCLAB-R17",
    "DCLAB-R19",
    "DCLAB-R22"
   ],
   "does": "Open the locked holdout once, after every choice is fixed, and challenge the result by slices.",
   "gate": "The owner approves opening the holdout (WF-09)."
  },
  {
   "step": "WF-10 Knowledge capture",
   "rules": [
    "DCLAB-R20",
    "DCLAB-R21"
   ],
   "does": "Capture the notebook, the report and the training examples."
  }
 ]
}
```

## data_understanding

### `deduplicate_before_split`  (breast_cancer, du_deduplicate_before_split)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 624 | FEATURES: 30 | POSITIVE RATE: 37.5%

PROFILE:
- rows: 624 | features: 30 | rows per feature: 20.8
- positive rate: 37.5% (minority class 37.5%)
- exact duplicate rows (features + target): 8.8%
- highest missing rates: `radius1` 0.0%, `texture1` 0.0%, `perimeter1` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: deduplicate_before_split
WHY: 8.8% of rows are exact duplicates, above the 1% threshold. If they are split randomly, copies of the same row land in both training and test folds, so the test score partly measures memory. Duplicates must be removed, or grouped so all copies stay in one fold, before any split.
RULE: DCLAB-R04 (duplicate/entity leakage) and DCLAB-R02 (split by the independence unit)
CONFIDENCE: high
NEXT: Deduplicate (or define groups), re-profile, then start the leakage audit.
```

### `impute_within_folds`  (spambase, du_impute_within_folds)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 2,054 | FEATURES: 57 | POSITIVE RATE: 40.5%

PROFILE:
- rows: 2,054 | features: 57 | rows per feature: 36.0
- positive rate: 40.5% (minority class 40.5%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `char_freq_[` 46.0%, `word_freq_telnet` 45.4%, `word_freq_all` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: impute_within_folds
WHY: `char_freq_[` is missing in 46.0% of rows, above the 5% threshold. Filling it with statistics from the whole dataset would leak test-fold information into training, so imputation (and a missing-value indicator, since absence may carry meaning) must be fitted inside each training fold.
RULE: DCLAB-R03 (fit imputers on training partitions only) and DCLAB-R12 (missingness as a state)
CONFIDENCE: high
NEXT: Add in-fold imputation and missing indicators to the pipeline, then start the leakage audit.
```

### `proceed`  (bank_marketing, du_proceed)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 3,481 | FEATURES: 15 | POSITIVE RATE: 11.7%

PROFILE:
- rows: 3,481 | features: 15 | rows per feature: 232.1
- positive rate: 11.7% (minority class 11.7%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 0.0%, `job` 0.0%, `marital` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: proceed
WHY: No threshold is crossed: duplicates 0.0%, minority class 11.7%, highest missing rate 0.0%, 3,481 rows (232.1 per feature). The data is ready for the leakage audit.
RULE: DCLAB-R01 (contract written, profile clean)
CONFIDENCE: high
NEXT: Write the profile to the decision ledger and start the leakage audit.
```

### `small_data_protocol`  (mushroom, du_small_data_protocol)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS USED: 469 | FEATURES: 22 | POSITIVE RATE: 51.0%

PROFILE:
- rows: 469 | features: 22 | rows per feature: 21.3
- positive rate: 51.0% (minority class 49.0%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `cap-shape` 0.0%, `cap-surface` 0.0%, `cap-color` 0.0%
- near-constant columns (one value in >99.5% of rows): 1

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: small_data_protocol
WHY: There are only 469 rows (21.3 per feature). Single splits on data this small swing widely, so use repeated stratified CV, start with simple regularized models, and report wide uncertainty instead of single scores.
RULE: DCLAB-R16 (paired, stable comparisons) and DCLAB-R13 (start from a transparent baseline)
CONFIDENCE: high
NEXT: Switch to repeated stratified CV with a linear baseline, then start the leakage audit.
```

### `use_imbalance_aware_evaluation`  (adult, du_use_imbalance_aware_evaluation)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 1,514 | FEATURES: 14 | POSITIVE RATE: 3.4%

PROFILE:
- rows: 1,514 | features: 14 | rows per feature: 108.1
- positive rate: 3.4% (minority class 3.4%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 0.0%, `workclass` 0.0%, `fnlwgt` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: use_imbalance_aware_evaluation
WHY: The minority class is only 3.4% of rows, below the 10% threshold. Accuracy and even ROC-AUC can look good while the model misses most positives, so evaluation needs stratified folds, PR-AUC, and a threshold chosen from the cost of each error.
RULE: DCLAB-R18 (metrics and thresholds from the decision cost)
CONFIDENCE: high
NEXT: Fix the evaluation plan (stratified folds, PR-AUC, cost-based threshold) before the leakage audit.
```

## leakage_audit

### `exclude`  (heart_disease, target_copy)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Heart Disease (heart_disease). Predicts whether heart disease is present.
DECISION MOMENT: Retrospective classification AFTER all listed tests, including ca and thal. Cannot claim early triage or clinical validity.
ROWS: 303 (evidence computed on a stratified sample of 303) | POSITIVE RATE: 45.9% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.764

COLUMN: `prior_outcome_count`
DATA DICTIONARY: no entry
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 0.7% of rows
- univariate ROC-AUC: 1.000
- identical to the target (or its inverse): yes
- name matches the suspicious-word list: yes
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 1.000 | without: 0.889 | change: +0.111

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: exclude
WHY: `prior_outcome_count` reproduces the outcome almost perfectly (univariate ROC-AUC 1.000; it is identical to the target). A column like this is the label in disguise, so it cannot exist before the outcome. Adding it moves CV ROC-AUC by +0.111, which is the size of the illusion, not real skill.
RULE: DCLAB-R04 (target-derived leakage)
CONFIDENCE: high
NEXT: Remove `prior_outcome_count` from the feature list, record the reason in the decision ledger, and continue the audit.
```

### `flag_for_review`  (heart_disease, strong_signal_unknown_origin)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Heart Disease (heart_disease). Predicts whether heart disease is present.
DECISION MOMENT: Retrospective classification AFTER all listed tests, including ca and thal. Cannot claim early triage or clinical validity.
ROWS: 303 (evidence computed on a stratified sample of 303) | POSITIVE RATE: 45.9% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.764

COLUMN: `measure_42`
DATA DICTIONARY: no entry
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 100.0% of rows
- univariate ROC-AUC: 0.959
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.979 | without: 0.889 | change: +0.090

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: flag_for_review
WHY: `measure_42` is unusually strong (univariate ROC-AUC 0.959 vs 0.764 for the strongest real feature) and raises CV ROC-AUC by +0.090, but there is no data-dictionary entry saying when it is recorded. A strong score alone does not prove leakage, and it does not prove availability either, so a person with the source lineage has to confirm it.
RULE: DCLAB-R05 (high AUC starts a review, it does not prove leakage)
CONFIDENCE: medium
NEXT: Keep `measure_42` out of all models until the data owner confirms when it is recorded; continue the audit meanwhile.
```

### `keep`  (adult, strong_but_available)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS: 48,842 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 23.9% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.728

COLUMN: `profile_score`
DATA DICTIONARY: Computed at the decision moment from `age` and `hours-per-week`.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 25.3% of rows
- univariate ROC-AUC: 0.749
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.905 | without: 0.905 | change: +0.000

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: keep
WHY: `profile_score` has moderate signal (univariate ROC-AUC 0.749), and its lineage is clear: it is computed at the decision moment from `age` and `hours-per-week`, which are themselves available then. Predictive power from available inputs is the goal, not a leak.
RULE: DCLAB-R05 (high AUC alone does not prove leakage)
CONFIDENCE: high
NEXT: Keep `profile_score` and continue the audit with the next column.
```

### `rebuild_within_folds`  (breast_cancer, full_data_target_encoding)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS: 569 (evidence computed on a stratified sample of 569) | POSITIVE RATE: 37.3% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.975

COLUMN: `smoothness3_target_rate`
DATA DICTIONARY: Mean of the target per `smoothness3` group, calculated once using all rows, including evaluation rows.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 3.2% of rows
- univariate ROC-AUC: 0.772
- identical to the target (or its inverse): no
- name matches the suspicious-word list: yes
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.993 | without: 0.994 | change: -0.000

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: rebuild_within_folds
WHY: Encoding `smoothness3` by its outcome rate is a legitimate idea, but this version was computed on all rows, so every evaluation row's own label is inside its feature value. Here the visible effect is small (CV change -0.000) because each group is large, but the method is still wrong and the same mistake on a high-cardinality column produces large fake gains. The fix is to recompute the rate inside each training fold, not to drop the idea.
RULE: DCLAB-R04 (cross-split aggregate leakage)
CONFIDENCE: high
NEXT: Drop this version of `smoothness3_target_rate`, recompute it out-of-fold inside each training split, and re-audit it.
```

## feature_engineering

### `adopt_interactions`  (adult, fe_sim_interaction_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 14 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 14 | 0.5004 | 0.0171 | — | — |
| log_transforms | 19 | 0.4960 | 0.0099 | -0.0044 | 2/5 |
| ratios | 20 | 0.7570 | 0.0210 | +0.2566 | 5/5 |
| interactions | 20 | 0.9016 | 0.0080 | +0.4012 | 5/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_interactions
WHY: `interactions` improves on raw by +0.4012 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. Other recipes that also qualified gained less. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: Carry the raw + interactions feature set into model selection and log each new feature's formula.
```

### `adopt_log_transforms`  (credit_default, fe_sim_ratio_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 23 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 23 | 0.9173 | 0.0101 | — | — |
| log_transforms | 30 | 0.9680 | 0.0026 | +0.0507 | 5/5 |
| ratios | 29 | 0.9241 | 0.0099 | +0.0069 | 5/5 |
| interactions | 29 | 0.9231 | 0.0103 | +0.0058 | 5/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_log_transforms
WHY: `log_transforms` improves on raw by +0.0507 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. Other recipes that also qualified gained less. It adds 7 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: Carry the raw + log_transforms feature set into model selection and log each new feature's formula.
```

### `adopt_ratios`  (breast_cancer, fe_sim_ratio_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 569 | FEATURES: 30 | POSITIVE RATE: 49.9%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 30 | 0.9632 | 0.0092 | — | — |
| log_transforms | 52 | 0.9672 | 0.0093 | +0.0040 | 5/5 |
| ratios | 36 | 0.9695 | 0.0106 | +0.0063 | 5/5 |
| interactions | 36 | 0.9628 | 0.0084 | -0.0004 | 2/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_ratios
WHY: `ratios` improves on raw by +0.0063 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: Carry the raw + ratios feature set into model selection and log each new feature's formula.
```

### `keep_raw`  (spambase, fe_sim_log_hist_gradient_boosting)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 57 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: hist_gradient_boosting | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 57 | 0.8527 | 0.0118 | — | — |
| log_transforms | 114 | 0.8527 | 0.0118 | +0.0000 | 0/5 |
| ratios | 63 | 0.8500 | 0.0124 | -0.0027 | 1/5 |
| interactions | 63 | 0.8530 | 0.0117 | +0.0003 | 4/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. The closest, `interactions`, changes ROC-AUC by +0.0003 and wins on 4/5 folds, which is not a consistent gain of at least +0.005. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: Carry the raw feature set into model selection.
```

## model_selection

### `extra_trees`  (bank_marketing, ms_real_300)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 300 | FEATURES: 15 | POSITIVE RATE: 11.0%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.6765 | 0.0443 | 0.6655 | 0.02 |
| random_forest | 0.6694 | 0.0582 | 0.6548 | 0.76 |
| extra_trees | 0.6966 | 0.0330 | 0.6884 | 0.67 |
| hist_gradient_boosting | 0.5952 | 0.0125 | 0.5920 | 0.22 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: extra_trees
WHY: `extra_trees` has the highest stability-adjusted score (0.6884: mean 0.6966, std 0.0330), ahead of `logistic_regression` at 0.6655. The linear baseline scores 0.6655, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `extra_trees` into optimization; keep `logistic_regression` as the paired comparison when tuning.
```

### `hist_gradient_boosting`  (heart_disease, ms_sim_threshold_full)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Heart Disease (heart_disease). Predicts whether heart disease is present.
DECISION MOMENT: Retrospective classification AFTER all listed tests, including ca and thal. Cannot claim early triage or clinical validity.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 303 | FEATURES: 13 | POSITIVE RATE: 49.8%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.4276 | 0.0202 | 0.4225 | 0.02 |
| random_forest | 0.9275 | 0.0268 | 0.9208 | 0.76 |
| extra_trees | 0.7515 | 0.0571 | 0.7372 | 0.65 |
| hist_gradient_boosting | 0.9756 | 0.0064 | 0.9740 | 0.25 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: hist_gradient_boosting
WHY: `hist_gradient_boosting` has the highest stability-adjusted score (0.9740: mean 0.9756, std 0.0064), ahead of `random_forest` at 0.9208. The linear baseline scores 0.4225, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `hist_gradient_boosting` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `logistic_regression`  (mushroom, ms_sim_linear_full)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 4,000 | FEATURES: 22 | POSITIVE RATE: 50.0%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9475 | 0.0041 | 0.9465 | 0.05 |
| random_forest | 0.9415 | 0.0043 | 0.9404 | 1.03 |
| extra_trees | 0.9378 | 0.0021 | 0.9372 | 0.92 |
| hist_gradient_boosting | 0.9389 | 0.0037 | 0.9379 | 0.87 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: logistic_regression
WHY: `logistic_regression` has the highest stability-adjusted score (0.9465: mean 0.9475, std 0.0041), ahead of `random_forest` at 0.9404. The linear baseline wins here, which is exactly why it is always screened: no family is best on every dataset.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `logistic_regression` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `random_forest`  (mushroom, ms_real_full)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS USED: 4,000 | FEATURES: 22 | POSITIVE RATE: 48.2%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9992 | 0.0001 | 0.9992 | 0.05 |
| random_forest | 1.0000 | 0.0000 | 1.0000 | 0.88 |
| extra_trees | 1.0000 | 0.0000 | 1.0000 | 0.78 |
| hist_gradient_boosting | 1.0000 | 0.0000 | 1.0000 | 0.50 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` and `extra_trees` are effectively tied: stability-adjusted scores 1.0000 vs 1.0000 (within 0.002) and similar runtimes (0.88s vs 0.78s). Nothing here separates them meaningfully, so `random_forest` goes forward on its slightly higher score, as a low-confidence choice that later evidence (calibration, stability on new folds) can overturn.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: low
NEXT: Carry `random_forest` into optimization; keep `extra_trees` as the paired comparison when tuning.
```

## optimization_reliability

### `accept_result`  (adult, opt_holdout_random)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 4,000 | FEATURES: 14 | POSITIVE RATE: 23.9%

MODEL: hist_gradient_boosting (locked) | training rows: 3,023 | holdout rows: 977 (holdout positive rate 23.5%)
- 3-fold CV ROC-AUC on training: 0.8946 (std 0.0143)
- holdout ROC-AUC (scored once): 0.9040
- difference, CV minus holdout: -0.0094
- largest feature shifts, training vs holdout (PSI): `relationship` 0.012, `education-num` 0.010, `age` 0.009

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: accept_result
WHY: CV ROC-AUC 0.8946 and holdout ROC-AUC 0.9040 differ by -0.0094, inside the ±0.02 band, so the CV estimate held up on untouched data. No feature shifted meaningfully between training and holdout.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Report CV and holdout together with uncertainty, and do not tune anything on the holdout.
```

### `calibrate_isotonic`  (adult, opt_calibration_random_forest_balanced_weights)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 4,000 | FEATURES: 14 | POSITIVE RATE: 23.9%

MODEL: random_forest_balanced_weights | 3 stratified folds | training rows per fold: about 2,666
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0474 | 0.9052 | 0.1048 |
| sigmoid | 0.0269 | 0.9059 | 0.1023 |
| isotonic | 0.0084 | 0.9043 | 0.1022 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_isotonic
WHY: Raw `random_forest_balanced_weights` probabilities are off by an ECE of 0.0474, above the 0.03 bar. Isotonic calibration fitted inside the training folds brings ECE to 0.0084 with ROC-AUC 0.9043 vs 0.9052 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Add isotonic calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `calibrate_sigmoid`  (spambase, opt_calibration_random_forest)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 900 | FEATURES: 57 | POSITIVE RATE: 41.1%

MODEL: random_forest | 3 stratified folds | training rows per fold: about 600
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0670 | 0.9673 | 0.0656 |
| sigmoid | 0.0347 | 0.9659 | 0.0606 |
| isotonic | 0.0168 | 0.9647 | 0.0599 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_sigmoid
WHY: Raw `random_forest` probabilities are off by an ECE of 0.0670, above the 0.03 bar. Sigmoid calibration fitted inside the training folds brings ECE to 0.0347 with ROC-AUC 0.9659 vs 0.9673 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Add sigmoid calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `investigate_distribution_shift`  (bank_marketing, opt_holdout_shifted)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 4,000 | FEATURES: 15 | POSITIVE RATE: 11.7%

MODEL: hist_gradient_boosting (locked) | training rows: 3,000 | holdout rows: 1,000 (holdout positive rate 3.2%)
- 3-fold CV ROC-AUC on training: 0.7349 (std 0.0157)
- holdout ROC-AUC (scored once): 0.6136
- difference, CV minus holdout: +0.1213
- largest feature shifts, training vs holdout (PSI): `contact` 16.060, `month` 10.435, `poutcome` 3.264

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_distribution_shift
WHY: The holdout ROC-AUC (0.6136) is 0.1213 below CV (0.7349), outside the ±0.02 band, and `contact` has a PSI of 16.060 (above 0.2): the holdout population is different from training. The CV number does not describe this population, so the shift must be explained before any claim is made.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Do not tune on the holdout. Slice results by the shifted feature, document the gap, and decide with the data owner.
```

### `investigate_gap_without_shift`  (heart_disease, opt_holdout_random)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Heart Disease (heart_disease). Predicts whether heart disease is present.
DECISION MOMENT: Retrospective classification AFTER all listed tests, including ca and thal. Cannot claim early triage or clinical validity.
ROWS USED: 303 | FEATURES: 13 | POSITIVE RATE: 45.9%

MODEL: hist_gradient_boosting (locked) | training rows: 227 | holdout rows: 76 (holdout positive rate 39.5%)
- 3-fold CV ROC-AUC on training: 0.8721 (std 0.0333)
- holdout ROC-AUC (scored once): 0.8210
- difference, CV minus holdout: +0.0511
- largest feature shifts, training vs holdout (PSI): `trestbps` 0.193, `chol` 0.154, `thalach` 0.151

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_gap_without_shift
WHY: CV (0.8721) and holdout (0.8210) differ by +0.0511, outside the ±0.02 band, yet no feature shifts much (highest PSI 0.193). The holdout scored lower than CV without a population change, which points to choices fitted to CV noise (optimistic CV) or an unstable estimate on 76 rows, so the gap needs explaining before results are reported.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Do not tune on the holdout. Put a bootstrap interval on the holdout score, review which earlier choices were made on CV, and report both numbers with the explanation.
```

### `keep_uncalibrated`  (spambase, opt_calibration_random_forest)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 4,000 | FEATURES: 57 | POSITIVE RATE: 39.4%

MODEL: random_forest | 3 stratified folds | training rows per fold: about 2,666
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0550 | 0.9843 | 0.0453 |
| sigmoid | 0.0201 | 0.9814 | 0.0435 |
| isotonic | 0.0136 | 0.9817 | 0.0437 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: keep_uncalibrated
WHY: Raw `random_forest` probabilities are already usable or calibration does not pay: raw ECE is 0.0550, and neither method cuts it by 25% without losing ROC-AUC (sigmoid 0.0201, isotonic 0.0136). Adding a calibration layer would add complexity for no measured benefit.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Lock the uncalibrated model and move to the final holdout check.
```

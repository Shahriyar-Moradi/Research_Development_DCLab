# Training sample preview

One real sample per stage and decision, copied from `out/all_stages/train.jsonl`. The system prompt is the same in every sample and is shown once.

**Totals:** train 2,591 · val 329 · test 840

**System prompt:**

> You are the DCLab workflow agent. You build ML models step by step: data_understanding -> leakage_audit -> feature_engineering -> model_selection -> optimization_reliability. At each step you decide using only the evidence given, choose exactly one allowed action, explain why with the numbers, cite the rule, and say what happens next.

## problem_understanding

### `problem_understanding`  (subscription_churn, replica)

**Input**

```
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
We run a video subscription with three plans. Each row is one subscriber. The churned column says yes when the subscriber cancelled. The score is needed at renewal time; refunds are paid only after a cancellation.

FILES (headers only):
subscribers.csv: subscriber_id, signup_time, plan, monthly_fee, tenure_months, support_calls, late_payments, final_refund_amount, churned
```

**Answer**

```
{
 "unit": {
  "value": "subscriber",
  "status": "stated",
  "quote": "Each row is one subscriber."
 },
 "task": {
  "value": "binary",
  "status": "inferred",
  "reason": "churned holds yes or no for each subscriber, so the answer is binary"
 },
 "target_column": {
  "value": "churned",
  "status": "stated",
  "quote": "The churned column says yes when the subscriber cancelled."
 },
 "label_source": {
  "value": "column",
  "status": "inferred",
  "reason": "the label is a column of subscribers.csv"
 },
 "label_coverage": {
  "value": "complete",
  "status": "inferred",
  "reason": "the label is a column of the main table, so every row is labelled"
 },
 "prediction_moment": {
  "value": "at renewal time",
  "status": "stated",
  "quote": "The score is needed at renewal time; refunds are paid only after a cancellation."
 },
 "time_column": {
  "value": "signup_time",
  "status": "inferred",
  "reason": "signup_time is the only date or time column in subscribers.csv"
 },
 "unknown": [
  "goal",
  "action",
  "horizon",
  "entities",
  "primary_metric"
 ],
 "constraints": [],
 "open_questions": [
  {
   "field": "goal",
   "text": "What decision or action should the model support, and for whom?",
   "impact": 1
  },
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

## contract_check

### `ask_owner`  (adult, public)

**Input**

```
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "estimate who earns more than 50K", "status": "stated", "quote": "We want to estimate which people earn more than 50K a year."}, "action": {"value": "estimate guides survey follow-up", "status": "stated", "quote": "The estimate guides a survey's follow-up sampling."}, "unit": {"value": "a person", "status": "inferred", "reason": "each row of adult.csv describes one a person"}, "task": {"value": "binary", "status": "inferred", "reason": "high_income is 1 or 0, so the answer is binary"}, "target_column": {"value": "high_income", "status": "stated", "quote": "high_income is the label."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of adult.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "once the census record is filed", "status": "stated", "quote": "We score each person from their census record, once the record is filed."}, "unknown": ["time_column", "horizon", "entities", "primary_metric"], "constraints": [], "open_questions": [{"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA PROFILE:
TABLE: adult — 3,000 rows, 15 columns
| column | kind | distinct | blank |
|---|---|---|---|
| age | number | 69 | 0.0% |
| workclass | number | 9 | 0.0% |
| fnlwgt | number | 2,848 | 0.0% |
| education | number | 16 | 0.0% |
| education-num | number | 16 | 0.0% |
| marital-status | number | 7 | 0.0% |
| occupation | number | 16 | 0.0% |
| relationship | number | 6 | 0.0% |
| race | number | 5 | 0.0% |
| sex | number | 2 | 0.0% |
| capital-gain | number | 64 | 0.0% |
| capital-loss | number | 50 | 0.0% |
| hours-per-week | number | 72 | 0.0% |
| native-country | number | 40 | 0.0% |
| label_high_income | number | 2 | 0.0% |

ALLOWED DECISIONS: proceed | fix_contract | ask_owner
```

**Answer**

```
{
 "checks": [
  {
   "check": "target_present",
   "status": "fail",
   "detail": "`high_income` is not a column of the table; the closest name is `label_high_income`"
  },
  {
   "check": "target_fits_task",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  },
  {
   "check": "target_labelled",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  },
  {
   "check": "target_varies",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  },
  {
   "check": "time_column_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  },
  {
   "check": "entities_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  }
 ],
 "decision": "ask_owner",
 "contract_changes": [],
 "questions": [
  {
   "field": "target_column",
   "text": "The table has no `high_income` column. Is `label_high_income` the target?",
   "impact": 1
  }
 ],
 "next": "Wait for the owner's answer: no analysis can test columns against the outcome without the target."
}
```

### `fix_contract`  (credit_default, public)

**Input**

```
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "a client defaults on next month's credit-card payment", "status": "stated", "quote": "We need to predict whether a client defaults on next month's credit-card payment."}, "action": {"value": "collections contacts them before the due date", "status": "stated", "quote": "Collections will contact them before the due date."}, "unit": {"value": "card holder", "status": "stated", "quote": "Every row describes a card holder."}, "task": {"value": "regression", "status": "inferred", "reason": "the goal was read as asking for a quantity"}, "target_column": {"value": "defaulted_next_month", "status": "stated", "quote": "The target is defaulted_next_month, 1 when the payment was missed."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of credit_default.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "at the end of each billing period", "status": "stated", "quote": "We predict at the end of each billing period."}, "unknown": ["time_column", "horizon", "entities", "primary_metric"], "constraints": [], "open_questions": [{"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA PROFILE:
TABLE: credit_default — 3,000 rows, 24 columns
| column | kind | distinct | blank |
|---|---|---|---|
| X1 | number | 66 | 0.0% |
| X2 | number | 2 | 0.0% |
| X3 | number | 6 | 0.0% |
| X4 | number | 4 | 0.0% |
| X5 | number | 51 | 0.0% |
| X6 | number | 10 | 0.0% |
| X7 | number | 9 | 0.0% |
| X8 | number | 10 | 0.0% |
| X9 | number | 10 | 0.0% |
| X10 | number | 9 | 0.0% |
| X11 | number | 9 | 0.0% |
| X12 | number | 2,659 | 0.0% |
| X13 | number | 2,633 | 0.0% |
| X14 | number | 2,600 | 0.0% |
| X15 | number | 2,558 | 0.0% |
| X16 | number | 2,498 | 0.0% |
| X17 | number | 2,456 | 0.0% |
| X18 | number | 1,474 | 0.0% |
| X19 | number | 1,417 | 0.0% |
| X20 | number | 1,333 | 0.0% |
| X21 | number | 1,239 | 0.0% |
| X22 | number | 1,222 | 0.0% |
| X23 | number | 1,248 | 0.0% |
| defaulted_next_month | number | 2 | 0.0% |

ALLOWED DECISIONS: proceed | fix_contract | ask_owner
```

**Answer**

```
{
 "checks": [
  {
   "check": "target_present",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_fits_task",
   "status": "fail",
   "detail": "`defaulted_next_month` has 2 distinct values, which fits binary, not regression"
  },
  {
   "check": "target_labelled",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_varies",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "time_column_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  },
  {
   "check": "entities_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  }
 ],
 "decision": "fix_contract",
 "contract_changes": [
  {
   "field": "task",
   "from": "regression",
   "to": "binary",
   "reason": "`defaulted_next_month` has 2 distinct values; the earlier reading was an inference, so the data corrects it"
  }
 ],
 "questions": [],
 "next": "Record the task as binary, then run the data analysis."
}
```

### `proceed`  (adult, public)

**Input**

```
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "a person earns more than $50K per year", "status": "stated", "quote": "We need to predict whether a person earns more than $50K per year."}, "unit": {"value": "a person", "status": "inferred", "reason": "each row of adult.csv describes one a person"}, "task": {"value": "binary", "status": "inferred", "reason": "high_income is 1 or 0, so the answer is binary"}, "target_column": {"value": "high_income", "status": "stated", "quote": "The high_income column is 1 when income is above 50K."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of adult.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "once the census record is filed", "status": "stated", "quote": "We score each person from their census record, once the record is filed."}, "primary_metric": {"value": "average precision", "status": "stated", "quote": "We will judge it by average precision."}, "unknown": ["action", "time_column", "horizon", "entities"], "constraints": [], "open_questions": [{"field": "action", "text": "What happens with each prediction, and roughly what does a wrong one cost?", "impact": 2}, {"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA PROFILE:
TABLE: adult — 3,000 rows, 15 columns
| column | kind | distinct | blank |
|---|---|---|---|
| age | number | 69 | 0.0% |
| workclass | number | 9 | 0.0% |
| fnlwgt | number | 2,848 | 0.0% |
| education | number | 16 | 0.0% |
| education-num | number | 16 | 0.0% |
| marital-status | number | 7 | 0.0% |
| occupation | number | 16 | 0.0% |
| relationship | number | 6 | 0.0% |
| race | number | 5 | 0.0% |
| sex | number | 2 | 0.0% |
| capital-gain | number | 64 | 0.0% |
| capital-loss | number | 50 | 0.0% |
| hours-per-week | number | 72 | 0.0% |
| native-country | number | 40 | 0.0% |
| high_income | number | 2 | 0.0% |

ALLOWED DECISIONS: proceed | fix_contract | ask_owner
```

**Answer**

```
{
 "checks": [
  {
   "check": "target_present",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_fits_task",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_labelled",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_varies",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "time_column_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  },
  {
   "check": "entities_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  }
 ],
 "decision": "proceed",
 "contract_changes": [],
 "questions": [],
 "next": "The contract and the table agree: run the data analysis on the training partition."
}
```

## data_analysis

### `data_analysis`  (store_demand, replica)

**Input**

```
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

PROBLEM CONTRACT:
{"goal": {"value": "forecast sales per store and product", "status": "stated", "quote": "We want to forecast how much each store will sell of each product."}, "unit": {"value": "store, product and day", "status": "stated", "quote": "Each row is one store, product and day."}, "task": {"value": "regression", "status": "inferred", "reason": "units is a count of items sold, so the answer is a quantity"}, "target_column": {"value": "units", "status": "stated", "quote": "The target is units, the units sold that day."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of sales.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "every Sunday evening", "status": "stated", "quote": "Every Sunday evening we forecast; promotions are planned weeks ahead, but the day's weather is measured on the day."}, "time_column": {"value": "date", "status": "inferred", "reason": "date is the only date or time column in sales.csv"}, "primary_metric": {"value": "MAE", "status": "stated", "quote": "We will judge it by MAE."}, "unknown": ["action", "horizon", "entities"], "constraints": [], "open_questions": [{"field": "action", "text": "What happens with each prediction, and roughly what does a wrong one cost?", "impact": 2}, {"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

TABLE: sales — 4,320 rows, 7 columns (store, product, date, on_promotion, rain_mm, units, series)
TRAINING PARTITION: the earliest 75% of rows by date (3,240 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 4320, "unique_columns": []}
- F-distribution [target] evidence: {"min": 8.0, "median": 76.0, "max": 329.0, "skew": 0.93, "upper_to_lower_tail": 1.79, "zero_share": 0.0, "negative_share": 0.0}
- F-time [time] evidence: {"column": "date", "start": "2025-03-01 00:00:00", "end": "2025-09-02 00:00:00", "days": 185, "mean_shift_in_sd": 0.21}
```

**Answer**

```
{
 "task": "regression",
 "target": "units",
 "training_rows": 3240,
 "findings": [
  {
   "id": "F-keys",
   "severity": "info",
   "statement": "sales has 4320 rows; 0 column(s) identify rows: none.",
   "implication": "A column that identifies rows can only memorise: keep it out of the features.",
   "affects": [
    "features.identifiers"
   ]
  },
  {
   "id": "F-distribution",
   "severity": "warning",
   "statement": "The target ranges 8 to 329 (median 76, skew 0.93).",
   "implication": "A long right tail: compare a log target, and use MAE or a quantile loss rather than squared error, which the tail would dominate.",
   "affects": [
    "metric.primary",
    "target.transform"
   ]
  },
  {
   "id": "F-time",
   "severity": "info",
   "statement": "Rows span 185 days by date.",
   "implication": "Rows have an order in time: validate on later rows than the model trained on (a forward split), and read drift as a reason to retrain.",
   "affects": [
    "split.strategy",
    "risk.check"
   ]
  }
 ]
}
```

## solution_design

### `solution_design`  (subscription_churn, replica)

**Input**

```
STAGE: solution_design
Solution design: from the problem contract and the data findings, decide the solution plan (one decision per key: task framing, labels, split, metrics, baseline, forbidden features, identifiers, feature direction, cleaning, model families, threshold), each with its reason, the findings it cites and the alternatives set aside. Then lay the decisions on the DCLab workflow WF-01…WF-10 with its gates. Output JSON.

PROBLEM CONTRACT:
{"action": {"value": "retention team offers a discount", "status": "stated", "quote": "The retention team will offer them a discount before they decide."}, "unit": {"value": "subscriber", "status": "stated", "quote": "Each row is one subscriber."}, "task": {"value": "binary", "status": "inferred", "reason": "churned holds yes or no for each subscriber, so the answer is binary"}, "target_column": {"value": "churned", "status": "stated", "quote": "The churned column says yes when the subscriber cancelled."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of subscribers.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "at renewal time", "status": "stated", "quote": "The score is needed at renewal time; refunds are paid only after a cancellation."}, "time_column": {"value": "signup_time", "status": "inferred", "reason": "signup_time is the only date or time column in subscribers.csv"}, "primary_metric": {"value": "average precision", "status": "stated", "quote": "We will judge it by average precision."}, "unknown": ["goal", "horizon", "entities"], "constraints": [], "open_questions": [{"field": "goal", "text": "What decision or action should the model support, and for whom?", "impact": 1}, {"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA FINDINGS:
- F-keys (info, grain): subscribers has 3000 rows; 1 column(s) identify rows: subscriber_id. Implication: A column that identifies rows can only memorise: keep it out of the features. Evidence: {"rows": 3000, "unique_columns": ["subscriber_id"], "unique_columns_count": 1}
- F-prevalence (info, target): The positive class (yes) is 20.04% of training rows. Implication: Average precision or ROC AUC rank the cases; accuracy depends on the threshold. Evidence: {"positive_class": "yes", "positive_rate": 0.20044, "positives": 451, "rows": 2250}
- F-time (info, time): Rows span 555 days by signup_time. Implication: Rows have an order in time: validate on later rows than the model trained on (a forward split), and read drift as a reason to retrain. Evidence: {"column": "signup_time", "start": "2024-01-01 01:00:00", "end": "2025-07-09 14:00:00", "days": 555, "positive_rate_first_third": 0.19733, "positive_rate_last_third": 0.216}
- F-leak-final_refund_amount (critical, leakage): final_refund_amount: on its own it predicts the target almost perfectly; its name says it is written after the outcome. Implication: Keep final_refund_amount out unless it is known at the prediction moment; ask when it is written. Evidence: {"column": "final_refund_amount", "single_column_power": 1.0}
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
    "kind": "time",
    "time_column": "signup_time",
    "holdout": "locked first, used once, after the choice"
   },
   "cites": [
    "F-time"
   ],
   "reason": "Rows arrive over time (signup_time): train on earlier rows and test on later ones, so the score is for the period the model will meet. No drift was measured, but a test this size can miss it, and a random split would hide it.",
   "alternatives": [
    {
     "choice": "stratified",
     "why_not": "mixes later rows into training"
    }
   ]
  },
  {
   "key": "metric.primary",
   "choice": "average_precision",
   "cites": [
    "F-prevalence"
   ],
   "reason": "The brief names average_precision. Positives are 20.04% of rows: ROC AUC ranks well at this balance.",
   "alternatives": [
    {
     "choice": "roc_auc",
     "why_not": "the brief names average_precision, so it leads; roc_auc is reported beside it"
    },
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
    "roc_auc",
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
   "choice": [
    {
     "column": "final_refund_amount",
     "reason": "final_refund_amount: on its own it predicts the target almost perfectly; its name says it is written after the outcome."
    }
   ],
   "cites": [
    "F-leak-final_refund_amount"
   ],
   "reason": "These columns look written after the prediction moment (name, single-column power or recomputing the target): keep them out unless a person confirms they are known at the moment.",
   "alternatives": []
  },
  {
   "key": "features.identifiers",
   "choice": [
    "subscriber_id"
   ],
   "cites": [
    "F-keys"
   ],
   "reason": "A column unique per row can only memorise: it identifies, it does not describe.",
   "alternatives": []
  },
  {
   "key": "features.direction",
   "choice": [
    "calendar parts and time since the entity's previous row, from the past only"
   ],
   "cites": [
    "F-time"
   ],
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
 "risks": [
  {
   "statement": "final_refund_amount: on its own it predicts the target almost perfectly; its name says it is written after the outcome.",
   "check": "Keep final_refund_amount out unless it is known at the prediction moment; ask when it is written.",
   "cites": [
    "F-leak-final_refund_amount"
   ]
  }
 ],
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

## connected_problem_to_plan

### `connected_problem_to_plan`  (subscription_churn, replica)

**Input**

```
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
Our streaming product renews every month or year. We want to know which subscribers will cancel. Customer care will call the riskiest ones with a personal offer. The label is churned: yes if they left, no if they stayed. We predict at renewal time, before the subscriber decides. We care most about AUC. The team can call the top 200 each week.

FILES (headers only):
subscribers.csv: subscriber_id, signup_time, plan, monthly_fee, tenure_months, support_calls, late_payments, final_refund_amount, churned
```

**Answer**

```
{
 "goal": {
  "value": "find subscribers who will cancel",
  "status": "stated",
  "quote": "We want to know which subscribers will cancel."
 },
 "action": {
  "value": "customer care calls with an offer",
  "status": "stated",
  "quote": "Customer care will call the riskiest ones with a personal offer."
 },
 "unit": {
  "value": "subscriber",
  "status": "inferred",
  "reason": "each row of subscribers.csv describes one subscriber"
 },
 "task": {
  "value": "binary",
  "status": "inferred",
  "reason": "churned holds yes or no for each subscriber, so the answer is binary"
 },
 "target_column": {
  "value": "churned",
  "status": "stated",
  "quote": "The label is churned: yes if they left, no if they stayed."
 },
 "label_source": {
  "value": "column",
  "status": "inferred",
  "reason": "the label is a column of subscribers.csv"
 },
 "label_coverage": {
  "value": "complete",
  "status": "inferred",
  "reason": "the label is a column of the main table, so every row is labelled"
 },
 "prediction_moment": {
  "value": "at renewal time, before the subscriber decides",
  "status": "stated",
  "quote": "We predict at renewal time, before the subscriber decides."
 },
 "time_column": {
  "value": "signup_time",
  "status": "inferred",
  "reason": "signup_time is the only date or time column in subscribers.csv"
 },
 "primary_metric": {
  "value": "AUC",
  "status": "stated",
  "quote": "We care most about AUC."
 },
 "unknown": [
  "horizon",
  "entities"
 ],
 "constraints": [
  {
   "kind": "review_capacity",
   "text": "top 200 per week",
   "quote": "The team can call the top 200 each week."
  }
 ],
 "open_questions": [
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

TABLE: subscribers — 3,000 rows, 9 columns (subscriber_id, signup_time, plan, monthly_fee, tenure_months, support_calls, late_payments, final_refund_amount, churned)
TRAINING PARTITION: the earliest 75% of rows by signup_time (2,250 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 3000, "unique_columns": ["subscriber_id"], "unique_columns_count": 1}
- F-prevalence [target] evidence: {"positive_class": "yes", "positive_rate": 0.20356, "positives": 458, "rows": 2250}
- F-time [time] evidence: {"column": "signup_time", "start": "2024-01-01 13:00:00", "end": "2025-06-25 02:00:00", "days": 540, "positive_rate_first_third": 0.18267, "positive_rate_last_third": 0.22667}
- F-leak-final_refund_amount [leakage] evidence: {"column": "final_refund_amount", "single_column_power": 1.0}
```

**Answer**

```
{
 "task": "binary",
 "target": "churned",
 "training_rows": 2250,
 "findings": [
  {
   "id": "F-keys",
   "severity": "info",
   "statement": "subscribers has 3000 rows; 1 column(s) identify rows: subscriber_id.",
   "implication": "A column that identifies rows can only memorise: keep it out of the features.",
   "affects": [
    "features.identifiers"
   ]
  },
  {
   "id": "F-prevalence",
   "severity": "info",
   "statement": "The positive class (yes) is 20.36% of training rows.",
   "implication": "Average precision or ROC AUC rank the cases; accuracy depends on the threshold.",
   "affects": [
    "metric.primary",
    "threshold.policy",
    "split.strategy"
   ]
  },
  {
   "id": "F-time",
   "severity": "info",
   "statement": "Rows span 540 days by signup_time.",
   "implication": "Rows have an order in time: validate on later rows than the model trained on (a forward split), and read drift as a reason to retrain.",
   "affects": [
    "split.strategy",
    "risk.check"
   ]
  },
  {
   "id": "F-leak-final_refund_amount",
   "severity": "critical",
   "statement": "final_refund_amount: on its own it predicts the target almost perfectly; its name says it is written after the outcome.",
   "implication": "Keep final_refund_amount out unless it is known at the prediction moment; ask when it is written.",
   "affects": [
    "features.forbidden"
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
    "kind": "time",
    "time_column": "signup_time",
    "holdout": "locked first, used once, after the choice"
   },
   "cites": [
    "F-time"
   ],
   "reason": "Rows arrive over time (signup_time): train on earlier rows and test on later ones, so the score is for the period the model will meet. No drift was measured, but a test this size can miss it, and a random split would hide it.",
   "alternatives": [
    {
     "choice": "stratified",
     "why_not": "mixes later rows into training"
    }
   ]
  },
  {
   "key": "metric.primary",
   "choice": "roc_auc",
   "cites": [
    "F-prevalence"
   ],
   "reason": "The brief names roc_auc. Positives are 20.36% of rows: ROC AUC ranks well at this balance.",
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
    "recall_at_budget",
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
   "choice": [
    {
     "column": "final_refund_amount",
     "reason": "final_refund_amount: on its own it predicts the target almost perfectly; its name says it is written after the outcome."
    }
   ],
   "cites": [
    "F-leak-final_refund_amount"
   ],
   "reason": "These columns look written after the prediction moment (name, single-column power or recomputing the target): keep them out unless a person confirms they are known at the moment.",
   "alternatives": []
  },
  {
   "key": "features.identifiers",
   "choice": [
    "subscriber_id"
   ],
   "cites": [
    "F-keys"
   ],
   "reason": "A column unique per row can only memorise: it identifies, it does not describe.",
   "alternatives": []
  },
  {
   "key": "features.direction",
   "choice": [
    "calendar parts and time since the entity's previous row, from the past only"
   ],
   "cites": [
    "F-time"
   ],
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
    "kind": "review_budget",
    "capacity": "top 200 per week"
   },
   "cites": [
    "F-prevalence"
   ],
   "reason": "The team can review top 200 per week: alert on that many of the riskiest, chosen on out-of-fold scores.",
   "alternatives": [
    {
     "choice": "0.5",
     "why_not": "a probability threshold ignores how many cases the team can review"
    }
   ]
  }
 ],
 "risks": [
  {
   "statement": "final_refund_amount: on its own it predicts the target almost perfectly; its name says it is written after the outcome.",
   "check": "Keep final_refund_amount out unless it is known at the prediction moment; ask when it is written.",
   "cites": [
    "F-leak-final_refund_amount"
   ]
  }
 ],
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

## end_to_end

### `end_to_end`  (credit_default, public)

**Input**

```
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
This project uses the credit default data. We need to predict whether a client defaults on next month's credit-card payment. Collections will contact them before the due date. defaulted_next_month is the label. We predict at the end of each billing period. The prediction looks one month ahead.

FILES (headers only):
credit_default.csv: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, defaulted_next_month
```

**Answer**

```
{
 "goal": {
  "value": "a client defaults on next month's credit-card payment",
  "status": "stated",
  "quote": "We need to predict whether a client defaults on next month's credit-card payment."
 },
 "action": {
  "value": "collections contacts them before the due date",
  "status": "stated",
  "quote": "Collections will contact them before the due date."
 },
 "unit": {
  "value": "a card holder",
  "status": "inferred",
  "reason": "each row of credit_default.csv describes one a card holder"
 },
 "task": {
  "value": "binary",
  "status": "inferred",
  "reason": "defaulted_next_month is 1 or 0, so the answer is binary"
 },
 "target_column": {
  "value": "defaulted_next_month",
  "status": "stated",
  "quote": "defaulted_next_month is the label."
 },
 "label_source": {
  "value": "column",
  "status": "inferred",
  "reason": "the label is a column of credit_default.csv"
 },
 "label_coverage": {
  "value": "complete",
  "status": "inferred",
  "reason": "the label is a column of the main table, so every row is labelled"
 },
 "prediction_moment": {
  "value": "at the end of each billing period",
  "status": "stated",
  "quote": "We predict at the end of each billing period."
 },
 "horizon": {
  "value": "one month ahead",
  "status": "stated",
  "quote": "The prediction looks one month ahead."
 },
 "unknown": [
  "time_column",
  "entities",
  "primary_metric"
 ],
 "constraints": [],
 "open_questions": [
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
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

Check the contract above against the table.

DATA PROFILE:
TABLE: credit_default — 3,000 rows, 24 columns
| column | kind | distinct | blank |
|---|---|---|---|
| X1 | number | 66 | 0.0% |
| X2 | number | 2 | 0.0% |
| X3 | number | 6 | 0.0% |
| X4 | number | 4 | 0.0% |
| X5 | number | 51 | 0.0% |
| X6 | number | 10 | 0.0% |
| X7 | number | 9 | 0.0% |
| X8 | number | 10 | 0.0% |
| X9 | number | 10 | 0.0% |
| X10 | number | 9 | 0.0% |
| X11 | number | 9 | 0.0% |
| X12 | number | 2,659 | 0.0% |
| X13 | number | 2,633 | 0.0% |
| X14 | number | 2,600 | 0.0% |
| X15 | number | 2,558 | 0.0% |
| X16 | number | 2,498 | 0.0% |
| X17 | number | 2,456 | 0.0% |
| X18 | number | 1,474 | 0.0% |
| X19 | number | 1,417 | 0.0% |
| X20 | number | 1,333 | 0.0% |
| X21 | number | 1,239 | 0.0% |
| X22 | number | 1,222 | 0.0% |
| X23 | number | 1,248 | 0.0% |
| defaulted_next_month | number | 2 | 0.0% |

ALLOWED DECISIONS: proceed | fix_contract | ask_owner
```

**Answer**

```
{
 "checks": [
  {
   "check": "target_present",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_fits_task",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_labelled",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "target_varies",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "time_column_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  },
  {
   "check": "entities_present",
   "status": "n/a",
   "detail": "not applicable: the contract does not name it"
  }
 ],
 "decision": "proceed",
 "contract_changes": [],
 "questions": [],
 "next": "The contract and the table agree: run the data analysis on the training partition."
}
```

**Input**

```
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

TABLE: credit_default — 3,000 rows, 24 columns (X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, defaulted_next_month)
TRAINING PARTITION: a random 75% of rows (2,250 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 3000, "unique_columns": []}
- F-prevalence [target] evidence: {"positive_class": "1", "positive_rate": 0.22178, "positives": 499, "rows": 2250}
```

**Answer**

```
{
 "task": "binary",
 "target": "defaulted_next_month",
 "training_rows": 2250,
 "findings": [
  {
   "id": "F-keys",
   "severity": "info",
   "statement": "credit_default has 3000 rows; 0 column(s) identify rows: none.",
   "implication": "A column that identifies rows can only memorise: keep it out of the features.",
   "affects": [
    "features.identifiers"
   ]
  },
  {
   "id": "F-prevalence",
   "severity": "info",
   "statement": "The positive class (1) is 22.18% of training rows.",
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
   "reason": "Positives are 22.18% of rows: ROC AUC ranks well at this balance.",
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

**Input**

```
STAGE: leakage_audit
Leakage audit: decide, column by column, what may enter a model that predicts at the contract's prediction moment. Exclude a column the solution plan forbids, a column the declared decision moment rules out, and an identifier; flag a column the plan holds for confirmation; keep a column with no leakage evidence. A strong score alone starts a review, it does not prove leakage. Columns not listed keep their plan status. Output JSON.

DECISION MOMENT (dataset contract): End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
EVIDENCE: computed on the training partition only (2,250 rows); CV uses the plan's other usable columns as the base.
OTHER COLUMNS, not audited one by one: `X2`, `X3`, `X4`, `X5`, `X8`, `X9`, `X10`, `X11`, `X12`, `X13`, `X14`, `X15`, `X16`, `X17`, `X18`, `X19`, `X20`, `X21`, `X22`

COLUMN: `X6` | kind: number | plan says: not named | data dictionary: no entry
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 0.4% of rows
- univariate ROC-AUC: 0.684
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.731 | without: 0.676 | change: +0.055

COLUMN: `X7` | kind: number | plan says: not named | data dictionary: no entry
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 0.4% of rows
- univariate ROC-AUC: 0.657
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.731 | without: 0.709 | change: +0.022

COLUMN: `X1` | kind: number | plan says: not named | data dictionary: no entry
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 2.8% of rows
- univariate ROC-AUC: 0.651
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.731 | without: 0.712 | change: +0.019

COLUMN: `X23` | kind: number | plan says: not named | data dictionary: no entry
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 43.5% of rows
- univariate ROC-AUC: 0.582
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.731 | without: 0.709 | change: +0.022

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
{
 "columns": [
  {
   "column": "X6",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `X6` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.684, CV change +0.055).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "X7",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `X7` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.657, CV change +0.022).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "X1",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `X1` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.651, CV change +0.019).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "X23",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `X23` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.582, CV change +0.022).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  }
 ],
 "feature_set": [
  "X1",
  "X2",
  "X3",
  "X4",
  "X5",
  "X6",
  "X7",
  "X8",
  "X9",
  "X10",
  "X11",
  "X12",
  "X13",
  "X14",
  "X15",
  "X16",
  "X17",
  "X18",
  "X19",
  "X20",
  "X21",
  "X22",
  "X23"
 ],
 "not_raw_features": [],
 "next": "Build the feature matrix from the feature_set columns on the training partition and compare feature recipes."
}
```

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

TRAINING PARTITION: 2,250 rows | FEATURES: 23 | POSITIVE RATE: 22.2%
MODEL USED FOR THE COMPARISON: hist_gradient_boosting | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 23 | 0.7193 | 0.0229 | — | — |
| log_transforms | 30 | 0.7193 | 0.0229 | +0.0000 | 0/5 |
| ratios | 29 | 0.7307 | 0.0248 | +0.0114 | 4/5 |
| interactions | 29 | 0.7246 | 0.0291 | +0.0053 | 3/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. The closest, `ratios`, changes ROC-AUC by +0.0114 and wins on 4/5 folds, which is not a consistent gain of at least +0.005. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: Carry the raw feature set into model selection.
```

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

TRAINING PARTITION: 2,250 rows | FEATURES: 23 | POSITIVE RATE: 22.2%
FEATURE SET: raw (23 features)
FAMILIES FROM THE SOLUTION PLAN: logistic_regression, random_forest, gradient_boosting (gradient_boosting runs as hist_gradient_boosting; the dummy baseline predicts the training prevalence, so its ROC-AUC is 0.5 by construction)
SCREEN: 3 stratified folds, identical for every model
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.7112 | 0.0380 | 0.7017 | 0.05 |
| random_forest | 0.7392 | 0.0296 | 0.7317 | 1.39 |
| hist_gradient_boosting | 0.7230 | 0.0344 | 0.7144 | 0.99 |

ALLOWED ACTIONS: logistic_regression | random_forest | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` has the highest stability-adjusted score (0.7317: mean 0.7392, std 0.0296), ahead of `hist_gradient_boosting` at 0.7144. The linear baseline scores 0.7017, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `random_forest` into optimization; keep `hist_gradient_boosting` as the paired comparison when tuning.
```

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

TRAINING PARTITION: 2,250 rows | FEATURES: 23 | POSITIVE RATE: 22.2%
MODEL: random_forest | 3 stratified folds | training rows per fold: about 1,500
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0360 | 0.7378 | 0.1446 |
| sigmoid | 0.0163 | 0.7441 | 0.1426 |
| isotonic | 0.0103 | 0.7381 | 0.1440 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
```

**Answer**

```
DECISION: calibrate_isotonic
WHY: Raw `random_forest` probabilities are off by an ECE of 0.0360, above the 0.03 bar. Isotonic calibration fitted inside the training folds brings ECE to 0.0103 with ROC-AUC 0.7381 vs 0.7378 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Add isotonic calibration inside the pipeline, lock it, and move to the final holdout check.
```

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

LOCKED PIPELINE: raw features → random_forest + isotonic calibration
training rows: 2,250 | holdout rows: 750 (holdout positive rate 22.0%); the holdout is the part of the table the plan locked at the start
- 3-fold CV ROC-AUC on training: 0.7421 (std 0.0438)
- holdout ROC-AUC (scored once): 0.7562
- difference, CV minus holdout: -0.0141 (absolute 0.0141)
- largest feature shifts, training vs holdout (PSI): `X16` 0.036, `X17` 0.031, `X1` 0.027

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
```

**Answer**

```
DECISION: accept_result
WHY: CV ROC-AUC 0.7421 and holdout ROC-AUC 0.7562 differ by -0.0141, inside the ±0.02 band, so the CV estimate held up on untouched data. No feature shifted meaningfully between training and holdout.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Report CV and holdout together with uncertainty, and do not tune anything on the holdout.
```

## data_understanding

### `deduplicate_before_split`  (bank_marketing, du_deduplicate_before_split)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 1,885 | FEATURES: 15 | POSITIVE RATE: 11.2%

PROFILE:
- rows: 1,885 | features: 15 | rows per feature: 125.7
- positive rate: 11.2% (minority class 11.2%)
- exact duplicate rows (features + target): 12.7%
- highest missing rates: `age` 0.0%, `job` 0.0%, `marital` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: deduplicate_before_split
WHY: 12.7% of rows are exact duplicates, above the 1% threshold. If they are split randomly, copies of the same row land in both training and test folds, so the test score partly measures memory. Duplicates must be removed, or grouped so all copies stay in one fold, before any split.
RULE: DCLAB-R04 (duplicate/entity leakage) and DCLAB-R02 (split by the independence unit)
CONFIDENCE: high
NEXT: Deduplicate (or define groups), re-profile, then start the leakage audit.
```

### `impute_within_folds`  (adult, du_impute_within_folds)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 2,323 | FEATURES: 14 | POSITIVE RATE: 22.8%

PROFILE:
- rows: 2,323 | features: 14 | rows per feature: 165.9
- positive rate: 22.8% (minority class 22.8%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 39.9%, `workclass` 0.0%, `fnlwgt` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: impute_within_folds
WHY: `age` is missing in 39.9% of rows, above the 5% threshold. Filling it with statistics from the whole dataset would leak test-fold information into training, so imputation (and a missing-value indicator, since absence may carry meaning) must be fitted inside each training fold.
RULE: DCLAB-R03 (fit imputers on training partitions only) and DCLAB-R12 (missingness as a state)
CONFIDENCE: high
NEXT: Add in-fold imputation and missing indicators to the pipeline, then start the leakage audit.
```

### `proceed`  (adult, du_proceed)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 2,843 | FEATURES: 14 | POSITIVE RATE: 24.5%

PROFILE:
- rows: 2,843 | features: 14 | rows per feature: 203.1
- positive rate: 24.5% (minority class 24.5%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 0.0%, `workclass` 0.0%, `fnlwgt` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: proceed
WHY: No threshold is crossed: duplicates 0.0%, minority class 24.5%, highest missing rate 0.0%, 2,843 rows (203.1 per feature). The data is ready for the leakage audit.
RULE: DCLAB-R01 (contract written, profile clean)
CONFIDENCE: high
NEXT: Write the profile to the decision ledger and start the leakage audit.
```

### `small_data_protocol`  (credit_default, du_small_data_protocol)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS USED: 269 | FEATURES: 23 | POSITIVE RATE: 24.2%

PROFILE:
- rows: 269 | features: 23 | rows per feature: 11.7
- positive rate: 24.2% (minority class 24.2%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `X1` 0.0%, `X2` 0.0%, `X3` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: small_data_protocol
WHY: There are only 269 rows (11.7 per feature). Single splits on data this small swing widely, so use repeated stratified CV, start with simple regularized models, and report wide uncertainty instead of single scores.
RULE: DCLAB-R16 (paired, stable comparisons) and DCLAB-R13 (start from a transparent baseline)
CONFIDENCE: high
NEXT: Switch to repeated stratified CV with a linear baseline, then start the leakage audit.
```

### `use_imbalance_aware_evaluation`  (credit_default, du_use_imbalance_aware_evaluation)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS USED: 1,350 | FEATURES: 23 | POSITIVE RATE: 4.1%

PROFILE:
- rows: 1,350 | features: 23 | rows per feature: 58.7
- positive rate: 4.1% (minority class 4.1%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `X1` 0.0%, `X2` 0.0%, `X3` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: use_imbalance_aware_evaluation
WHY: The minority class is only 4.1% of rows, below the 10% threshold. Accuracy and even ROC-AUC can look good while the model misses most positives, so evaluation needs stratified folds, PR-AUC, and a threshold chosen from the cost of each error.
RULE: DCLAB-R18 (metrics and thresholds from the decision cost)
CONFIDENCE: high
NEXT: Fix the evaluation plan (stratified folds, PR-AUC, cost-based threshold) before the leakage audit.
```

## leakage_audit

### `exclude`  (mushroom, random_identifier)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS: 8,124 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 48.2% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.920

COLUMN: `record_id`
DATA DICTIONARY: Sequential number assigned when rows were exported.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 100.0% of rows
- univariate ROC-AUC: 0.502
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): yes
- 3-fold CV ROC-AUC with the column: 1.000 | without: 1.000 | change: +0.000

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: exclude
WHY: `record_id` is a run of consecutive integers, unique for 100% of rows: a row number. It shows no signal here (univariate ROC-AUC 0.502), but identifiers describe how the data was stored, not the entity, so they can only add noise now or leak later if the row order changes. Identifiers stay out.
RULE: DCLAB-R04 (identifier / entity leakage)
CONFIDENCE: high
NEXT: Remove `record_id` from the feature list, record the reason in the decision ledger, and continue the audit.
```

### `flag_for_review`  (bank_marketing, strong_signal_unknown_origin)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS: 45,211 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 11.7% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.636

COLUMN: `index_c`
DATA DICTIONARY: no entry
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 100.0% of rows
- univariate ROC-AUC: 0.989
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.991 | without: 0.752 | change: +0.239

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: flag_for_review
WHY: `index_c` is unusually strong (univariate ROC-AUC 0.989 vs 0.636 for the strongest real feature) and raises CV ROC-AUC by +0.239, but there is no data-dictionary entry saying when it is recorded. A strong score alone does not prove leakage, and it does not prove availability either, so a person with the source lineage has to confirm it.
RULE: DCLAB-R05 (high AUC starts a review, it does not prove leakage)
CONFIDENCE: medium
NEXT: Keep `index_c` out of all models until the data owner confirms when it is recorded; continue the audit meanwhile.
```

### `keep`  (bank_marketing, strong_but_available)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS: 45,211 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 11.7% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.636

COLUMN: `review_points`
DATA DICTIONARY: Computed at the decision moment from `month` and `previous`.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 3.1% of rows
- univariate ROC-AUC: 0.678
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.755 | without: 0.752 | change: +0.004

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: keep
WHY: `review_points` has moderate signal (univariate ROC-AUC 0.678), and its lineage is clear: it is computed at the decision moment from `month` and `previous`, which are themselves available then. Predictive power from available inputs is the goal, not a leak.
RULE: DCLAB-R05 (high AUC alone does not prove leakage)
CONFIDENCE: high
NEXT: Keep `review_points` and continue the audit with the next column.
```

### `rebuild_within_folds`  (credit_default, full_data_target_encoding)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS: 30,000 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 22.1% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.701

COLUMN: `X1_target_rate`
DATA DICTIONARY: Mean of the target per `X1` group, calculated once using all rows, including evaluation rows.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 1.1% of rows
- univariate ROC-AUC: 0.635
- identical to the target (or its inverse): no
- name matches the suspicious-word list: yes
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.747 | without: 0.732 | change: +0.015

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: rebuild_within_folds
WHY: Encoding `X1` by its outcome rate is a legitimate idea, but this version was computed on all rows, so every evaluation row's own label is inside its feature value. Here that already shows up as a CV gain of +0.015. The fix is to recompute the rate inside each training fold, not to drop the idea.
RULE: DCLAB-R04 (cross-split aggregate leakage)
CONFIDENCE: high
NEXT: Drop this version of `X1_target_rate`, recompute it out-of-fold inside each training split, and re-audit it.
```

## feature_engineering

### `adopt_interactions`  (credit_default, fe_sim_interaction_logistic_regression)

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
| raw | 23 | 0.6669 | 0.0185 | — | — |
| log_transforms | 30 | 0.7333 | 0.0121 | +0.0664 | 5/5 |
| ratios | 29 | 0.6890 | 0.0163 | +0.0221 | 5/5 |
| interactions | 29 | 0.8530 | 0.0147 | +0.1861 | 5/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_interactions
WHY: `interactions` improves on raw by +0.1861 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. Other recipes that also qualified gained less. It adds 6 features with a recorded formula, so its lineage stays auditable.
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

### `adopt_ratios`  (bank_marketing, fe_sim_ratio_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 15 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 15 | 0.9573 | 0.0087 | — | — |
| log_transforms | 18 | 0.9569 | 0.0082 | -0.0003 | 2/5 |
| ratios | 21 | 0.9650 | 0.0055 | +0.0078 | 5/5 |
| interactions | 21 | 0.9587 | 0.0083 | +0.0015 | 4/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_ratios
WHY: `ratios` improves on raw by +0.0078 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: Carry the raw + ratios feature set into model selection and log each new feature's formula.
```

### `keep_raw`  (bank_marketing, fe_sim_log_hist_gradient_boosting)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 15 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: hist_gradient_boosting | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 15 | 0.9631 | 0.0067 | — | — |
| log_transforms | 18 | 0.9631 | 0.0067 | +0.0000 | 0/5 |
| ratios | 21 | 0.9628 | 0.0072 | -0.0003 | 3/5 |
| interactions | 21 | 0.9629 | 0.0068 | -0.0001 | 2/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. The closest, `log_transforms`, changes ROC-AUC by +0.0000 and wins on 0/5 folds, which is not a consistent gain of at least +0.005. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: Carry the raw feature set into model selection.
```

## model_selection

### `extra_trees`  (spambase, ms_real_800)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 800 | FEATURES: 57 | POSITIVE RATE: 37.0%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9608 | 0.0093 | 0.9585 | 0.04 |
| random_forest | 0.9767 | 0.0072 | 0.9749 | 1.04 |
| extra_trees | 0.9782 | 0.0082 | 0.9762 | 0.79 |
| hist_gradient_boosting | 0.9771 | 0.0097 | 0.9747 | 0.64 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: extra_trees
WHY: `extra_trees` and `random_forest` are effectively tied: stability-adjusted scores 0.9762 vs 0.9749 (within 0.002) and similar runtimes (0.79s vs 1.04s). Nothing here separates them meaningfully, so `extra_trees` goes forward on its slightly higher score, as a low-confidence choice that later evidence (calibration, stability on new folds) can overturn.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: low
NEXT: Carry `extra_trees` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `hist_gradient_boosting`  (bank_marketing, ms_sim_linear_full)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 4,000 | FEATURES: 15 | POSITIVE RATE: 50.0%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9586 | 0.0011 | 0.9584 | 0.03 |
| random_forest | 0.9705 | 0.0020 | 0.9700 | 1.17 |
| extra_trees | 0.9593 | 0.0036 | 0.9584 | 0.92 |
| hist_gradient_boosting | 0.9728 | 0.0004 | 0.9727 | 0.71 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: hist_gradient_boosting
WHY: `hist_gradient_boosting` has the highest stability-adjusted score (0.9727: mean 0.9728, std 0.0004), ahead of `random_forest` at 0.9700. The linear baseline scores 0.9584, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `hist_gradient_boosting` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `logistic_regression`  (breast_cancer, ms_real_300)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 300 | FEATURES: 30 | POSITIVE RATE: 38.3%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9992 | 0.0006 | 0.9990 | 0.03 |
| random_forest | 0.9958 | 0.0004 | 0.9957 | 0.77 |
| extra_trees | 0.9963 | 0.0005 | 0.9962 | 0.61 |
| hist_gradient_boosting | 0.9925 | 0.0057 | 0.9911 | 0.26 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: logistic_regression
WHY: `logistic_regression` has the highest stability-adjusted score (0.9990: mean 0.9992, std 0.0006), ahead of `extra_trees` at 0.9962. The linear baseline wins here, which is exactly why it is always screened: no family is best on every dataset.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `logistic_regression` into optimization; keep `extra_trees` as the paired comparison when tuning.
```

### `random_forest`  (adult, ms_real_1500)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 1,500 | FEATURES: 14 | POSITIVE RATE: 24.1%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.8536 | 0.0278 | 0.8467 | 0.03 |
| random_forest | 0.8962 | 0.0169 | 0.8920 | 0.99 |
| extra_trees | 0.8935 | 0.0155 | 0.8896 | 0.82 |
| hist_gradient_boosting | 0.8786 | 0.0175 | 0.8742 | 0.63 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` has the highest stability-adjusted score (0.8920: mean 0.8962, std 0.0169), ahead of `extra_trees` at 0.8896. The linear baseline scores 0.8467, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `random_forest` into optimization; keep `extra_trees` as the paired comparison when tuning.
```

## optimization_reliability

### `accept_result`  (breast_cancer, opt_holdout_shifted)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 569 | FEATURES: 30 | POSITIVE RATE: 37.3%

MODEL: hist_gradient_boosting (locked) | training rows: 426 | holdout rows: 143 (holdout positive rate 58.7%)
- 3-fold CV ROC-AUC on training: 0.9943 (std 0.0022)
- holdout ROC-AUC (scored once): 0.9974
- difference, CV minus holdout: -0.0030
- largest feature shifts, training vs holdout (PSI): `smoothness1` 12.412, `smoothness3` 5.784, `compactness1` 3.523

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: accept_result
WHY: CV ROC-AUC 0.9943 and holdout ROC-AUC 0.9974 differ by -0.0030, inside the ±0.02 band, so the CV estimate held up on untouched data. `smoothness1` does shift (PSI 12.412), so it goes on the monitoring list, but it did not hurt performance here.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Report CV and holdout together with uncertainty, and do not tune anything on the holdout.
```

### `calibrate_isotonic`  (adult, opt_calibration_hist_gradient_boosting)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 4,000 | FEATURES: 14 | POSITIVE RATE: 23.9%

MODEL: hist_gradient_boosting | 3 stratified folds | training rows per fold: about 2,666
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0500 | 0.9023 | 0.1061 |
| sigmoid | 0.0208 | 0.9039 | 0.1015 |
| isotonic | 0.0118 | 0.9038 | 0.1015 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_isotonic
WHY: Raw `hist_gradient_boosting` probabilities are off by an ECE of 0.0500, above the 0.03 bar. Isotonic calibration fitted inside the training folds brings ECE to 0.0118 with ROC-AUC 0.9038 vs 0.9023 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Add isotonic calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `calibrate_sigmoid`  (heart_disease, opt_calibration_random_forest_balanced_weights)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Heart Disease (heart_disease). Predicts whether heart disease is present.
DECISION MOMENT: Retrospective classification AFTER all listed tests, including ca and thal. Cannot claim early triage or clinical validity.
ROWS USED: 303 | FEATURES: 13 | POSITIVE RATE: 45.9%

MODEL: random_forest_balanced_weights | 3 stratified folds | training rows per fold: about 202
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0872 | 0.8931 | 0.1319 |
| sigmoid | 0.0456 | 0.8973 | 0.1282 |
| isotonic | 0.0668 | 0.8943 | 0.1302 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_sigmoid
WHY: Raw `random_forest_balanced_weights` probabilities are off by an ECE of 0.0872, above the 0.03 bar. Sigmoid calibration fitted inside the training folds brings ECE to 0.0456 with ROC-AUC 0.8973 vs 0.8931 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Add sigmoid calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `investigate_distribution_shift`  (heart_disease, opt_holdout_shifted)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Heart Disease (heart_disease). Predicts whether heart disease is present.
DECISION MOMENT: Retrospective classification AFTER all listed tests, including ca and thal. Cannot claim early triage or clinical validity.
ROWS USED: 303 | FEATURES: 13 | POSITIVE RATE: 45.9%

MODEL: hist_gradient_boosting (locked) | training rows: 228 | holdout rows: 75 (holdout positive rate 26.7%)
- 3-fold CV ROC-AUC on training: 0.8596 (std 0.0511)
- holdout ROC-AUC (scored once): 0.8327
- difference, CV minus holdout: +0.0269
- largest feature shifts, training vs holdout (PSI): `oldpeak` 10.492, `slope` 1.468, `thalach` 0.970

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_distribution_shift
WHY: The holdout ROC-AUC (0.8327) is 0.0269 below CV (0.8596), outside the ±0.02 band, and `oldpeak` has a PSI of 10.492 (above 0.2): the holdout population is different from training. The CV number does not describe this population, so the shift must be explained before any claim is made.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Do not tune on the holdout. Slice results by the shifted feature, document the gap, and decide with the data owner.
```

### `investigate_gap_without_shift`  (bank_marketing, opt_holdout_random)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 4,000 | FEATURES: 15 | POSITIVE RATE: 11.7%

MODEL: hist_gradient_boosting (locked) | training rows: 3,031 | holdout rows: 969 (holdout positive rate 10.8%)
- 3-fold CV ROC-AUC on training: 0.7394 (std 0.0167)
- holdout ROC-AUC (scored once): 0.7177
- difference, CV minus holdout: +0.0217
- largest feature shifts, training vs holdout (PSI): `job` 0.021, `age` 0.016, `month` 0.010

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_gap_without_shift
WHY: CV (0.7394) and holdout (0.7177) differ by +0.0217, outside the ±0.02 band, yet no feature shifts much (highest PSI 0.021). The holdout scored lower than CV without a population change, which points to choices fitted to CV noise (optimistic CV) or an unstable estimate on 969 rows, so the gap needs explaining before results are reported.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Do not tune on the holdout. Put a bootstrap interval on the holdout score, review which earlier choices were made on CV, and report both numbers with the explanation.
```

### `keep_uncalibrated`  (bank_marketing, opt_calibration_hist_gradient_boosting)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 4,000 | FEATURES: 15 | POSITIVE RATE: 11.7%

MODEL: hist_gradient_boosting | 3 stratified folds | training rows per fold: about 2,666
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0463 | 0.7416 | 0.0918 |
| sigmoid | 0.1873 | 0.3148 | 0.1555 |
| isotonic | 0.0083 | 0.6638 | 0.1004 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: keep_uncalibrated
WHY: Raw `hist_gradient_boosting` probabilities are already usable or calibration does not pay: raw ECE is 0.0463, and neither method cuts it by 25% without losing ROC-AUC (sigmoid 0.1873, isotonic 0.0083). Adding a calibration layer would add complexity for no measured benefit.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Lock the uncalibrated model and move to the final holdout check.
```

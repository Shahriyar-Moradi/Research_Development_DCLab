# Training sample preview

One real sample per stage and decision, copied from `out/all_stages/train.jsonl`. The system prompt is the same in every sample and is shown once.

**Totals:** train 2,976 · val 932 · test 1,084

**System prompt:**

> You are the DCLab workflow agent. You take an ML problem through the DCLab workflow WF-01…WF-10: write the problem contract (WF-01), check the data against it (WF-02), design the split (WF-03), analyse the training data and design the solution (WF-04), audit leakage (WF-05), engineer features (WF-06), screen models (WF-07), optimize conservatively (WF-08), challenge reliability on the holdout, once (WF-09), and capture the result (WF-10). At each step you decide using only the evidence given, choose exactly one allowed action, explain why with the numbers, cite the rule, and route the work: continue, go back to the earlier step the evidence reopens, or ask the owner.

## problem_understanding

### `problem_understanding`  (adult, public)

**Input**

```
WORKFLOW STEP: WF-01 Solution draft
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
This project uses the adult data. The estimate guides a survey's follow-up sampling. The target column is high_income. We score each person from their census record, once the record is filed.

FILES (headers only):
adult.csv: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, high_income
```

**Answer**

```
{
 "action": {
  "value": "estimate guides survey follow-up",
  "status": "stated",
  "quote": "The estimate guides a survey's follow-up sampling."
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
 "unknown": [
  "goal",
  "time_column",
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
   "field": "entities",
   "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?",
   "impact": 3
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-02 Source and lineage"
 }
}
```

## contract_check

### `ask_owner`  (spambase, public)

**Input**

```
WORKFLOW STEP: WF-02 Source and lineage
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "keep spam out of the inbox", "status": "stated", "quote": "We want to keep spam out of the inbox."}, "unit": {"value": "an email", "status": "inferred", "reason": "each row of spambase.csv describes one an email"}, "task": {"value": "binary", "status": "inferred", "reason": "spam is 1 or 0, so the answer is binary"}, "target_column": {"value": "spam", "status": "stated", "quote": "spam is the label."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of spambase.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "when the email is received", "status": "stated", "quote": "We classify each email when it is received."}, "primary_metric": {"value": "average precision", "status": "stated", "quote": "We will judge it by average precision."}, "unknown": ["action", "time_column", "horizon", "entities"], "constraints": [], "open_questions": [{"field": "action", "text": "What happens with each prediction, and roughly what does a wrong one cost?", "impact": 2}, {"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA PROFILE:
TABLE: spambase — 3,000 rows, 58 columns
| column | kind | distinct | blank |
|---|---|---|---|
| word_freq_make | number | 134 | 0.0% |
| word_freq_address | number | 150 | 0.0% |
| word_freq_all | number | 193 | 0.0% |
| word_freq_3d | number | 23 | 0.0% |
| word_freq_our | number | 229 | 0.0% |
| word_freq_over | number | 130 | 0.0% |
| word_freq_remove | number | 148 | 0.0% |
| word_freq_internet | number | 144 | 0.0% |
| word_freq_order | number | 127 | 0.0% |
| word_freq_mail | number | 221 | 0.0% |
| word_freq_receive | number | 106 | 0.0% |
| word_freq_will | number | 290 | 0.0% |
| word_freq_people | number | 143 | 0.0% |
| word_freq_report | number | 98 | 0.0% |
| word_freq_addresses | number | 98 | 0.0% |
| word_freq_free | number | 226 | 0.0% |
| word_freq_business | number | 176 | 0.0% |
| word_freq_email | number | 205 | 0.0% |
| word_freq_you | number | 537 | 0.0% |
| word_freq_credit | number | 122 | 0.0% |
| word_freq_your | number | 362 | 0.0% |
| word_freq_font | number | 62 | 0.0% |
| word_freq_000 | number | 149 | 0.0% |
| word_freq_money | number | 127 | 0.0% |
| word_freq_hp | number | 339 | 0.0% |
| word_freq_hpl | number | 238 | 0.0% |
| word_freq_george | number | 197 | 0.0% |
| word_freq_650 | number | 173 | 0.0% |
| word_freq_lab | number | 132 | 0.0% |
| word_freq_labs | number | 158 | 0.0% |
| word_freq_telnet | number | 112 | 0.0% |
| word_freq_857 | number | 90 | 0.0% |
| word_freq_data | number | 143 | 0.0% |
| word_freq_415 | number | 93 | 0.0% |
| word_freq_85 | number | 147 | 0.0% |
| word_freq_technology | number | 144 | 0.0% |
| word_freq_1999 | number | 165 | 0.0% |
| word_freq_parts | number | 37 | 0.0% |
| word_freq_pm | number | 134 | 0.0% |
| word_freq_direct | number | 112 | 0.0% |
| word_freq_cs | number | 81 | 0.0% |
| word_freq_meeting | number | 140 | 0.0% |
| word_freq_original | number | 118 | 0.0% |
| word_freq_project | number | 130 | 0.0% |
| word_freq_re | number | 204 | 0.0% |
| word_freq_edu | number | 183 | 0.0% |
| word_freq_table | number | 27 | 0.0% |
| word_freq_conference | number | 85 | 0.0% |
| char_freq_; | number | 256 | 0.0% |
| char_freq_( | number | 554 | 0.0% |
| char_freq_[ | number | 187 | 0.0% |
| char_freq_! | number | 771 | 0.0% |
| char_freq_$ | number | 408 | 0.0% |
| char_freq_# | number | 241 | 0.0% |
| capital_run_length_average | number | 1,638 | 0.0% |
| capital_run_length_longest | number | 224 | 0.0% |
| capital_run_length_total | number | 745 | 0.0% |
| spam_flag | number | 2 | 0.0% |

ALLOWED DECISIONS: proceed | fix_contract | ask_owner
```

**Answer**

```
{
 "checks": [
  {
   "check": "target_present",
   "status": "fail",
   "detail": "`spam` is not a column of the table; the closest name is `spam_flag`"
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
   "text": "The table has no `spam` column. Is `spam_flag` the target?",
   "impact": 1
  }
 ],
 "next": "Wait for the owner's answer: no analysis can test columns against the outcome without the target.",
 "route": {
  "action": "ask_owner",
  "to": "WF-02 Source and lineage",
  "why": "the data contradicts what the owner stated, or the target is unusable"
 }
}
```

### `fix_contract`  (store_demand, replica)

**Input**

```
WORKFLOW STEP: WF-02 Source and lineage
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "forecast sales per store and product", "status": "stated", "quote": "We want to forecast how much each store will sell of each product."}, "action": {"value": "forecast drives the stock order", "status": "stated", "quote": "The forecast drives the stock order."}, "unit": {"value": "store, product and day", "status": "stated", "quote": "Each row is one store, product and day."}, "task": {"value": "regression", "status": "inferred", "reason": "units is a count of items sold, so the answer is a quantity"}, "target_column": {"value": "units", "status": "stated", "quote": "The target is units, the units sold that day."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of sales.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "each Sunday evening", "status": "stated", "quote": "We forecast each Sunday evening, for the next 7 days of each store and product."}, "time_column": {"value": "date", "status": "inferred", "reason": "date is the only date or time column in sales.csv"}, "horizon": {"value": "the next 7 days", "status": "stated", "quote": "The forecast covers the next 7 days."}, "entities": {"value": ["series"], "status": "stated", "quote": "Each store and product pair is its own series, named in the series column."}, "primary_metric": {"value": "MAE", "status": "stated", "quote": "We will judge it by MAE."}, "unknown": [], "constraints": [], "open_questions": []}

DATA PROFILE:
TABLE: sales — 4,320 rows, 7 columns
| column | kind | distinct | blank |
|---|---|---|---|
| store | text | 4 | 0.0% |
| product | text | 5 | 0.0% |
| date | date | 240 | 0.0% |
| on_promotion | number | 2 | 0.0% |
| rain_mm | number | 153 | 0.0% |
| units | number | 242 | 20.1% |
| series | text | 20 | 0.0% |

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
   "status": "fail",
   "detail": "20.1% of `units` values are blank"
  },
  {
   "check": "target_varies",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "time_column_present",
   "status": "pass",
   "detail": "matches the table"
  },
  {
   "check": "entities_present",
   "status": "pass",
   "detail": "matches the table"
  }
 ],
 "decision": "fix_contract",
 "contract_changes": [
  {
   "field": "label_coverage",
   "from": "complete",
   "to": "partial",
   "reason": "20.1% of rows have no label; unlabelled is not the same as negative"
  }
 ],
 "questions": [
  {
   "field": "label_coverage",
   "text": "Are the rows without a label unknown, or do they mean something (for example 'not checked')?",
   "impact": 2
  }
 ],
 "next": "Mark coverage partial, keep unlabelled rows out of training and evaluation, then run the data analysis.",
 "route": {
  "action": "back",
  "to": "WF-01 Solution draft",
  "why": "reopen the solution: the data corrects an inferred fact"
 }
}
```

### `proceed`  (subscription_churn, replica)

**Input**

```
WORKFLOW STEP: WF-02 Source and lineage
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "keep subscribers about to cancel", "status": "stated", "quote": "Our aim is to keep subscribers who are about to cancel."}, "unit": {"value": "subscriber", "status": "inferred", "reason": "each row of subscribers.csv describes one subscriber"}, "task": {"value": "binary", "status": "inferred", "reason": "churned holds yes or no for each subscriber, so the answer is binary"}, "target_column": {"value": "churned", "status": "stated", "quote": "The churned column says yes when the subscriber cancelled."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of subscribers.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "at renewal time", "status": "stated", "quote": "The score is needed at renewal time; refunds are paid only after a cancellation."}, "time_column": {"value": "signup_time", "status": "inferred", "reason": "signup_time is the only date or time column in subscribers.csv"}, "primary_metric": {"value": "average precision", "status": "stated", "quote": "We will judge it by average precision."}, "unknown": ["action", "horizon", "entities"], "constraints": [], "open_questions": [{"field": "action", "text": "What happens with each prediction, and roughly what does a wrong one cost?", "impact": 2}, {"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA PROFILE:
TABLE: subscribers — 3,000 rows, 9 columns
| column | kind | distinct | blank |
|---|---|---|---|
| subscriber_id | text | 3,000 | 0.0% |
| signup_time | date | 2,764 | 0.0% |
| plan | text | 3 | 0.0% |
| monthly_fee | number | 956 | 0.0% |
| tenure_months | number | 35 | 0.0% |
| support_calls | number | 8 | 0.0% |
| late_payments | number | 2 | 0.0% |
| final_refund_amount | number | 577 | 0.0% |
| churned | text | 2 | 0.0% |

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
   "status": "pass",
   "detail": "matches the table"
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
 "next": "The contract and the table agree: run the data analysis on the training partition.",
 "route": {
  "action": "continue",
  "to": "WF-03 Split design"
 }
}
```

## data_analysis

### `data_analysis`  (ticket_routing, replica)

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

PROBLEM CONTRACT:
{"goal": {"value": "stop tickets bouncing between teams", "status": "stated", "quote": "The goal is to stop tickets bouncing between teams."}, "action": {"value": "place the ticket in the team's queue", "status": "stated", "quote": "The router will place the ticket in that team's queue."}, "unit": {"value": "ticket", "status": "inferred", "reason": "each row of tickets.csv describes one ticket"}, "task": {"value": "multiclass", "status": "inferred", "reason": "assigned_team names one of several teams, so the answer is one class among many"}, "target_column": {"value": "assigned_team", "status": "stated", "quote": "The target is assigned_team, the team that finally handled it."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of tickets.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "when the ticket is created, before anyone reads it", "status": "stated", "quote": "Routing happens when the ticket is created, before anyone reads it."}, "time_column": {"value": "created_at", "status": "inferred", "reason": "created_at is the only date or time column in tickets.csv"}, "entities": {"value": ["customer_id"], "status": "stated", "quote": "Customers often open many tickets, so customer_id repeats."}, "unknown": ["horizon", "primary_metric"], "constraints": [], "open_questions": []}

TABLE: tickets — 3,060 rows, 18 columns (ticket_id, created_at, customer_id, plan, channel, product_area, subject_words, body_words, has_attachment, mentions_invoice, mentions_password, mentions_error, mentions_parcel, mentions_export, mentions_contract, resolved_by, first_response_minutes, assigned_team)
TRAINING PARTITION: the earliest 75% of rows by created_at (2,295 rows); the rest is the locked holdout

MEASUREMENTS:
- F-duplicates [grain] evidence: {"duplicate_rows": 60, "rows": 3060, "share": 0.0196}
- F-keys [grain] evidence: {"rows": 3060, "unique_columns": ["ticket_id"], "unique_columns_count": 1}
- F-classes [target] evidence: {"classes": 11, "largest_share": 0.1895, "smallest_share": 0.0031, "imbalance_ratio": 62.1}
- F-tiny-classes [target] evidence: {"tiny_classes": ["legal", "security"], "count": 2, "rows": {"legal": 10, "security": 7}, "tiny_classes_count": 2, "rows_count": 2}
- F-entity-customer_id [entities] evidence: {"column": "customer_id", "entities": 713, "rows_per_entity_median": 3.0, "rows_per_entity_max": 10, "later_rows_seen_before": 0.8889, "target_from_entity_out_of_fold": 0.0801}
- F-time [time] evidence: {"column": "created_at", "start": "2025-01-06 00:24:57", "end": "2025-04-06 15:01:19", "days": 90, "class_shift_tvd": 0.068, "classes_only_in_last_third": []}
- F-leak-resolved_by [leakage] evidence: {"column": "resolved_by", "single_column_power": 1.0}
- F-name-first_response_minutes [leakage] evidence: {"column": "first_response_minutes", "single_column_power": 0.0941}
```

**Answer**

```
{
 "task": "multiclass",
 "target": "assigned_team",
 "training_rows": 2295,
 "findings": [
  {
   "id": "F-duplicates",
   "severity": "warning",
   "statement": "60 rows of tickets are exact copies of another row.",
   "implication": "Drop the copies before splitting, or keep each copy on the same side: a copy on both sides scores the model on rows it trained on.",
   "affects": [
    "data.cleaning",
    "split.strategy"
   ]
  },
  {
   "id": "F-keys",
   "severity": "info",
   "statement": "tickets has 3060 rows; 1 column(s) identify rows: ticket_id.",
   "implication": "A column that identifies rows can only memorise: keep it out of the features.",
   "affects": [
    "features.identifiers"
   ]
  },
  {
   "id": "F-classes",
   "severity": "info",
   "statement": "11 classes; the largest is 19.0% of rows, the smallest 0.31%.",
   "implication": "Score with macro F1 (every class counts) beside accuracy, and stratify the folds by class.",
   "affects": [
    "metric.primary",
    "split.strategy"
   ]
  },
  {
   "id": "F-tiny-classes",
   "severity": "warning",
   "statement": "2 classes have under 1% of rows or 30 rows: legal, security.",
   "implication": "Too few rows to learn or to measure: consider merging them into an 'other' class or routing them by rule, keep them in every fold (stratify), and report their recall separately.",
   "affects": [
    "task.framing",
    "metric.primary",
    "split.strategy"
   ]
  },
  {
   "id": "F-entity-customer_id",
   "severity": "warning",
   "statement": "Rows repeat for customer_id: 713 values, a median of 3 rows each.",
   "implication": "Group the folds by customer_id: with rows of one customer_id on both sides a model can recognise it rather than learn the pattern.",
   "affects": [
    "split.strategy",
    "features.direction"
   ]
  },
  {
   "id": "F-time",
   "severity": "info",
   "statement": "Rows span 90 days by created_at.",
   "implication": "Rows have an order in time: validate on later rows than the model trained on (a forward split), and read drift as a reason to retrain.",
   "affects": [
    "split.strategy",
    "risk.check"
   ]
  },
  {
   "id": "F-leak-resolved_by",
   "severity": "critical",
   "statement": "resolved_by: on its own it predicts the target almost perfectly; its name says it is written after the outcome.",
   "implication": "Keep resolved_by out unless it is known at the prediction moment; ask when it is written.",
   "affects": [
    "features.forbidden"
   ]
  },
  {
   "id": "F-name-first_response_minutes",
   "severity": "info",
   "statement": "first_response_minutes: its name says it is written after the outcome, but it predicts little.",
   "implication": "Ask when first_response_minutes is written; keep it only if it is known at the prediction moment.",
   "affects": [
    "risk.check"
   ]
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-04 Train-only EDA",
  "why": "the findings go to the solution design"
 }
}
```

## solution_design

### `solution_design`  (mushroom, public)

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
STAGE: solution_design
Solution design: from the problem contract and the data findings, decide the solution plan (one decision per key: task framing, labels, split, metrics, baseline, forbidden features, identifiers, feature direction, cleaning, model families, threshold), each with its reason, the findings it cites and the alternatives set aside. Then lay the decisions on the DCLab workflow WF-01…WF-10 with its gates. Output JSON.

PROBLEM CONTRACT:
{"goal": {"value": "tell poisonous from edible mushrooms", "status": "stated", "quote": "We want to tell poisonous mushrooms from edible ones."}, "unit": {"value": "mushroom", "status": "stated", "quote": "Each row is a mushroom."}, "task": {"value": "binary", "status": "inferred", "reason": "poisonous is 1 or 0, so the answer is binary"}, "target_column": {"value": "poisonous", "status": "stated", "quote": "poisonous is the label."}, "label_source": {"value": "column", "status": "inferred", "reason": "the label is a column of mushroom.csv"}, "label_coverage": {"value": "complete", "status": "inferred", "reason": "the label is a column of the main table, so every row is labelled"}, "prediction_moment": {"value": "once physical traits are recorded", "status": "stated", "quote": "We classify once the mushroom's physical traits are recorded."}, "primary_metric": {"value": "average precision", "status": "stated", "quote": "We will judge it by average precision."}, "unknown": ["action", "time_column", "horizon", "entities"], "constraints": [], "open_questions": [{"field": "action", "text": "What happens with each prediction, and roughly what does a wrong one cost?", "impact": 2}, {"field": "entities", "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?", "impact": 3}]}

DATA FINDINGS:
- F-keys (info, grain): mushroom has 3000 rows; 0 column(s) identify rows: none. Implication: A column that identifies rows can only memorise: keep it out of the features. Evidence: {"rows": 3000, "unique_columns": []}
- F-prevalence (info, target): The positive class (1) is 48.71% of training rows. Implication: Average precision or ROC AUC rank the cases; accuracy depends on the threshold. Evidence: {"positive_class": "1", "positive_rate": 0.48711, "positives": 1096, "rows": 2250}
- F-leak-odor (critical, leakage): odor: on its own it predicts the target almost perfectly. Implication: Keep odor out unless it is known at the prediction moment; ask when it is written. Evidence: {"column": "odor", "single_column_power": 0.9272}
- F-constant (info, columns): 1 column(s) hold a single value: veil-type. Implication: Drop them: they carry nothing. Evidence: {"columns": ["veil-type"], "columns_count": 1}
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
   "choice": "average_precision",
   "cites": [
    "F-prevalence"
   ],
   "reason": "The brief names average_precision. Positives are 48.71% of rows: ROC AUC ranks well at this balance.",
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
     "column": "odor",
     "reason": "odor: on its own it predicts the target almost perfectly."
    }
   ],
   "cites": [
    "F-leak-odor"
   ],
   "reason": "These columns look written after the prediction moment (name, single-column power or recomputing the target): keep them out unless a person confirms they are known at the moment.",
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
    "drop constant columns"
   ],
   "cites": [
    "F-constant"
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
 "risks": [
  {
   "statement": "odor: on its own it predicts the target almost perfectly.",
   "check": "Keep odor out unless it is known at the prediction moment; ask when it is written.",
   "cites": [
    "F-leak-odor"
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
   "step": "WF-02 Source and lineage"
  },
  {
   "step": "WF-03 Split design",
   "decisions": [
    "split.strategy"
   ]
  },
  {
   "step": "WF-04 Train-only EDA",
   "decisions": [
    "data.cleaning"
   ],
   "gate": "solution"
  },
  {
   "step": "WF-05 Leakage audit",
   "decisions": [
    "features.forbidden",
    "features.identifiers"
   ]
  },
  {
   "step": "WF-06 Feature ladder",
   "decisions": [
    "features.direction"
   ]
  },
  {
   "step": "WF-07 Algorithm screen",
   "decisions": [
    "model.families"
   ]
  },
  {
   "step": "WF-08 Conservative optimization"
  },
  {
   "step": "WF-09 Reliability challenge",
   "gate": "holdout"
  },
  {
   "step": "WF-10 Knowledge capture"
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-05 Leakage audit",
  "why": "after the solution gate"
 }
}
```

## connected_problem_to_plan

### `connected_problem_to_plan`  (store_demand, replica)

**Input**

```
WORKFLOW STEP: WF-01 Solution draft
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
Our stores reorder stock every week. We want to forecast how much each store will sell of each product. Each row is one store, product and day. The target is units, the units sold that day. We forecast each Sunday evening, for the next 7 days of each store and product. The forecast covers the next 7 days. Each store and product pair is its own series, named in the series column. We will judge it by MAE.

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
 "primary_metric": {
  "value": "MAE",
  "status": "stated",
  "quote": "We will judge it by MAE."
 },
 "unknown": [
  "action"
 ],
 "constraints": [],
 "open_questions": [
  {
   "field": "action",
   "text": "What happens with each prediction, and roughly what does a wrong one cost?",
   "impact": 2
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-02 Source and lineage"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

TABLE: sales — 4,320 rows, 7 columns (store, product, date, on_promotion, rain_mm, units, series)
TRAINING PARTITION: the earliest 75% of rows by date (3,240 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 4320, "unique_columns": []}
- F-distribution [target] evidence: {"min": 8.0, "median": 76.0, "max": 329.0, "skew": 0.93, "upper_to_lower_tail": 1.79, "zero_share": 0.0, "negative_share": 0.0}
- F-entity-series [entities] evidence: {"column": "series", "entities": 20, "rows_per_entity_median": 186.0, "rows_per_entity_max": 186, "later_rows_seen_before": 1.0, "target_from_entity_out_of_fold": 0.8599}
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
   "id": "F-entity-series",
   "severity": "warning",
   "statement": "Rows repeat for series: 20 values, a median of 186 rows each.",
   "implication": "Group the folds by series: with rows of one series on both sides a model can recognise it rather than learn the pattern, and which one it is tells the target on rows it never saw.",
   "affects": [
    "split.strategy",
    "features.direction"
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
 ],
 "route": {
  "action": "continue",
  "to": "WF-04 Train-only EDA",
  "why": "the findings go to the solution design"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
STAGE: solution_design
Solution design: from the problem contract and the data findings, decide the solution plan (one decision per key: task framing, labels, split, metrics, baseline, forbidden features, identifiers, feature direction, cleaning, model families, threshold), each with its reason, the findings it cites and the alternatives set aside. Then lay the decisions on the DCLab workflow WF-01…WF-10 with its gates. Output JSON.

Use the contract and the findings above.
```

**Answer**

```
{
 "task": "regression",
 "decisions": [
  {
   "key": "task.framing",
   "choice": {
    "task": "regression"
   },
   "cites": [],
   "reason": "The problem is regression. The brief says so (inferred).",
   "alternatives": []
  },
  {
   "key": "labels.coverage",
   "choice": "complete",
   "cites": [
    "F-distribution"
   ],
   "reason": "Nothing says the labels are a sample; every row with a label is treated as labelled.",
   "alternatives": []
  },
  {
   "key": "split.strategy",
   "choice": {
    "kind": "group",
    "group_column": "series",
    "holdout": "locked first, used once, after the choice"
   },
   "cites": [
    "F-entity-series",
    "F-time"
   ],
   "reason": "Rows repeat for series: keep each series on one side, so the score is for new ones, not recognised ones.",
   "alternatives": [
    {
     "choice": "stratified",
     "why_not": "a series on both sides lets the model recognise it: an optimistic score"
    },
    {
     "choice": "time",
     "why_not": "no drift was seen, and grouping by series is the larger risk"
    }
   ]
  },
  {
   "key": "metric.primary",
   "choice": "mae",
   "cites": [
    "F-distribution"
   ],
   "reason": "The brief names mae. MAE reads in the target's unit and is not dominated by a few large errors.",
   "alternatives": [
    {
     "choice": "rmse",
     "why_not": "squared errors chase the largest values"
    },
    {
     "choice": "r2",
     "why_not": "not in the target's unit"
    }
   ]
  },
  {
   "key": "metric.secondary",
   "choice": [
    "rmse",
    "median_absolute_error"
   ],
   "cites": [
    "F-distribution"
   ],
   "reason": "Reported beside the primary metric, so one number does not hide a weakness.",
   "alternatives": []
  },
  {
   "key": "baseline",
   "choice": [
    "median of the training target",
    "mean per entity when entities repeat"
   ],
   "cites": [
    "F-distribution"
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
    "behaviour of each series: counts, repeats and shrunk rates over its other rows, fitted inside the folds",
    "calendar parts and time since the entity's previous row, from the past only"
   ],
   "cites": [
    "F-entity-series",
    "F-time"
   ],
   "reason": "Where the findings point: behaviour over repeated entities, time, and missingness that speaks.",
   "alternatives": []
  },
  {
   "key": "model.families",
   "choice": [
    "dummy",
    "ridge",
    "random_forest",
    "gradient_boosting"
   ],
   "cites": [
    "F-distribution"
   ],
   "reason": "Screen linear and tree families on identical folds, with a dummy baseline always in.",
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
   "decisions": [
    "task.framing",
    "labels.coverage",
    "metric.primary",
    "metric.secondary",
    "baseline"
   ]
  },
  {
   "step": "WF-02 Source and lineage"
  },
  {
   "step": "WF-03 Split design",
   "decisions": [
    "split.strategy"
   ]
  },
  {
   "step": "WF-04 Train-only EDA",
   "gate": "solution"
  },
  {
   "step": "WF-05 Leakage audit",
   "decisions": [
    "features.forbidden",
    "features.identifiers"
   ]
  },
  {
   "step": "WF-06 Feature ladder",
   "decisions": [
    "features.direction"
   ]
  },
  {
   "step": "WF-07 Algorithm screen",
   "decisions": [
    "model.families"
   ]
  },
  {
   "step": "WF-08 Conservative optimization"
  },
  {
   "step": "WF-09 Reliability challenge",
   "gate": "holdout"
  },
  {
   "step": "WF-10 Knowledge capture"
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-05 Leakage audit",
  "why": "after the solution gate"
 }
}
```

## end_to_end

### `end_to_end (clean, stops at WF-10)`  (credit_default, public)

**Input**

```
WORKFLOW STEP: WF-01 Solution draft
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
This project uses the credit default data. We want to know which card holders will default. Collections will contact them before the due date. Every row describes a card holder. The target column is defaulted_next_month. We predict at the end of each billing period. The prediction looks one month ahead. A missed payment costs about 900, a needless reminder about 5.

FILES (headers only):
credit_default.csv: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, defaulted_next_month
```

**Answer**

```
{
 "goal": {
  "value": "find card holders who will default",
  "status": "stated",
  "quote": "We want to know which card holders will default."
 },
 "action": {
  "value": "collections contacts them before the due date",
  "status": "stated",
  "quote": "Collections will contact them before the due date."
 },
 "unit": {
  "value": "card holder",
  "status": "stated",
  "quote": "Every row describes a card holder."
 },
 "task": {
  "value": "binary",
  "status": "inferred",
  "reason": "defaulted_next_month is 1 or 0, so the answer is binary"
 },
 "target_column": {
  "value": "defaulted_next_month",
  "status": "stated",
  "quote": "The target column is defaulted_next_month."
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
 "constraints": [
  {
   "kind": "budget",
   "text": "missed payment 900, needless reminder 5",
   "quote": "A missed payment costs about 900, a needless reminder about 5."
  }
 ],
 "open_questions": [
  {
   "field": "entities",
   "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?",
   "impact": 3
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-02 Source and lineage"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-02 Source and lineage
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

Check the contract above against the table.

DATA PROFILE:
TABLE: credit_default — 3,000 rows, 24 columns
| column | kind | distinct | blank |
|---|---|---|---|
| X1 | number | 62 | 0.0% |
| X2 | number | 2 | 0.0% |
| X3 | number | 7 | 0.0% |
| X4 | number | 4 | 0.0% |
| X5 | number | 51 | 0.0% |
| X6 | number | 11 | 0.0% |
| X7 | number | 10 | 0.0% |
| X8 | number | 8 | 0.0% |
| X9 | number | 8 | 0.0% |
| X10 | number | 7 | 0.0% |
| X11 | number | 7 | 0.0% |
| X12 | number | 2,658 | 0.0% |
| X13 | number | 2,590 | 0.0% |
| X14 | number | 2,548 | 0.0% |
| X15 | number | 2,507 | 0.0% |
| X16 | number | 2,457 | 0.0% |
| X17 | number | 2,399 | 0.0% |
| X18 | number | 1,389 | 0.0% |
| X19 | number | 1,418 | 0.0% |
| X20 | number | 1,315 | 0.0% |
| X21 | number | 1,262 | 0.0% |
| X22 | number | 1,234 | 0.0% |
| X23 | number | 1,211 | 0.0% |
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
 "next": "The contract and the table agree: run the data analysis on the training partition.",
 "route": {
  "action": "continue",
  "to": "WF-03 Split design"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature. If several thresholds are crossed, act on the first in this order: deduplicate_before_split, small_data_protocol, use_imbalance_aware_evaluation, impute_within_folds; the others are carried to the plan.

PROFILE:
- rows: 3,000 | features: 23 | rows per feature: 130.4
- positive rate: 22.1% (minority class 22.1%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `X1` 0.0%, `X2` 0.0%, `X3` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: proceed
WHY: No threshold is crossed: duplicates 0.0%, minority class 22.1%, highest missing rate 0.0%, 3,000 rows (130.4 per feature). The data is ready for the split and the training-only analysis.
RULE: DCLAB-R01 (contract written, profile clean)
CONFIDENCE: high
NEXT: [continue → WF-04] Write the profile to the decision ledger, design the split, then profile the training partition.
```

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

TABLE: credit_default — 3,000 rows, 24 columns (X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, defaulted_next_month)
TRAINING PARTITION: a random 75% of rows (2,250 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 3000, "unique_columns": []}
- F-conflicting-labels [grain] evidence: {"groups": 1}
- F-prevalence [target] evidence: {"positive_class": "1", "positive_rate": 0.22089, "positives": 497, "rows": 2250}
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
   "id": "F-conflicting-labels",
   "severity": "info",
   "statement": "1 groups of identical feature values carry different labels.",
   "implication": "Some noise is in the label itself; no model can be perfect on these rows.",
   "affects": [
    "metric.primary"
   ]
  },
  {
   "id": "F-prevalence",
   "severity": "info",
   "statement": "The positive class (1) is 22.09% of training rows.",
   "implication": "Average precision or ROC AUC rank the cases; accuracy depends on the threshold.",
   "affects": [
    "metric.primary",
    "threshold.policy",
    "split.strategy"
   ]
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-04 Train-only EDA",
  "why": "the findings go to the solution design"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
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
   "reason": "Positives are 22.09% of rows: ROC AUC ranks well at this balance.",
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
    "kind": "expected_cost",
    "costs": "missed payment 900, needless reminder 5"
   },
   "cites": [
    "F-prevalence"
   ],
   "reason": "The brief gives the cost of errors: choose the threshold that minimises expected cost on out-of-fold scores.",
   "alternatives": [
    {
     "choice": "0.5",
     "why_not": "ignores the costs the brief states"
    }
   ]
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
   "step": "WF-02 Source and lineage"
  },
  {
   "step": "WF-03 Split design",
   "decisions": [
    "split.strategy"
   ]
  },
  {
   "step": "WF-04 Train-only EDA",
   "gate": "solution"
  },
  {
   "step": "WF-05 Leakage audit",
   "decisions": [
    "features.forbidden",
    "features.identifiers"
   ]
  },
  {
   "step": "WF-06 Feature ladder",
   "decisions": [
    "features.direction"
   ]
  },
  {
   "step": "WF-07 Algorithm screen",
   "decisions": [
    "model.families"
   ]
  },
  {
   "step": "WF-08 Conservative optimization"
  },
  {
   "step": "WF-09 Reliability challenge",
   "gate": "holdout"
  },
  {
   "step": "WF-10 Knowledge capture"
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-05 Leakage audit",
  "why": "after the solution gate"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-05 Leakage audit
STAGE: leakage_audit
Leakage audit: decide, column by column, what may enter a model that predicts at the contract's prediction moment. Exclude a column the solution plan forbids, a column the declared decision moment rules out, and an identifier; flag a column the plan holds for confirmation; keep a column with no leakage evidence. A strong score alone starts a review, it does not prove leakage. Columns not listed keep their plan status. A leak the plan did not know reopens the solution (back to WF-01). Output JSON.

DECISION MOMENT (dataset contract): End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
EVIDENCE: computed on the training partition only (2,250 rows); CV uses the plan's other usable columns as the base.
OTHER COLUMNS, not audited one by one: `X1`, `X2`, `X3`, `X4`, `X5`, `X8`, `X9`, `X10`, `X11`, `X12`, `X13`, `X14`, `X16`, `X17`, `X19`, `X20`, `X21`, `X22`, `X23`
COLUMNS:
- `X6` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 0.4% of rows | univariate ROC-AUC 0.693 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.744, without 0.676, change +0.068
- `X7` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 0.4% of rows | univariate ROC-AUC 0.639 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.744, without 0.737, change +0.007
- `X18` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 49.0% of rows | univariate ROC-AUC 0.626 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.744, without 0.747, change -0.004
- `X15` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 84.8% of rows | univariate ROC-AUC 0.534 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.744, without 0.743, change +0.001

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
   "why": "Neither the plan nor the decision moment marks `X6` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.693, CV change +0.068).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "X7",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `X7` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.639, CV change +0.007).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "X18",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `X18` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.626, CV change -0.004).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "X15",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `X15` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.534, CV change +0.001).",
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
 "next": "Build the feature matrix from the feature_set columns on the training partition and compare feature recipes.",
 "route": {
  "action": "continue",
  "to": "WF-06 Feature ladder"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

TRAINING PARTITION: 2,250 rows | FEATURES: 23 | POSITIVE RATE: 22.1%
MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 23 | 0.7177 | 0.0223 | — | — |
| log_transforms | 30 | 0.7370 | 0.0243 | +0.0193 | 4/5 |
| ratios | 29 | 0.7215 | 0.0240 | +0.0038 | 4/5 |
| interactions | 29 | 0.7171 | 0.0248 | -0.0006 | 3/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. The closest, `log_transforms`, changes ROC-AUC by +0.0193 and wins on 4/5 folds, which is not a consistent gain of at least +0.005. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw feature set into model selection.
```

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

TRAINING PARTITION: 2,250 rows | FEATURES: 23 | POSITIVE RATE: 22.1%
FEATURE SET: raw (23 features)
FAMILIES FROM THE SOLUTION PLAN: logistic_regression, random_forest, gradient_boosting (gradient_boosting runs as hist_gradient_boosting; the dummy baseline predicts the training prevalence, so its ROC-AUC is 0.5 by construction)
SCREEN: 3 stratified folds, identical for every model
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.7192 | 0.0224 | 0.7136 | 0.04 |
| random_forest | 0.7626 | 0.0098 | 0.7601 | 1.34 |
| hist_gradient_boosting | 0.7411 | 0.0248 | 0.7349 | 0.81 |

ALLOWED ACTIONS: logistic_regression | random_forest | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` has the highest stability-adjusted score (0.7601: mean 0.7626, std 0.0098), ahead of `hist_gradient_boosting` at 0.7349. The linear baseline scores 0.7136, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: [continue → WF-08] Carry `random_forest` into optimization; keep `hist_gradient_boosting` as the paired comparison when tuning.
```

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

TRAINING PARTITION: 2,250 rows | FEATURES: 23 | POSITIVE RATE: 22.1%
MODEL: random_forest | 3 stratified folds | training rows per fold: about 1,500
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0310 | 0.7626 | 0.1383 |
| sigmoid | 0.0167 | 0.7672 | 0.1362 |
| isotonic | 0.0240 | 0.7636 | 0.1373 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
```

**Answer**

```
DECISION: calibrate_sigmoid
WHY: Raw `random_forest` probabilities are off by an ECE of 0.0310, above the 0.03 bar. Sigmoid calibration fitted inside the training folds brings ECE to 0.0167 with ROC-AUC 0.7672 vs 0.7626 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Add sigmoid calibration inside the pipeline, lock it, and move to the final holdout check.
```

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

LOCKED PIPELINE: raw features → random_forest + sigmoid calibration
training rows: 2,250 | holdout rows: 750 (holdout positive rate 22.3%); the holdout is the part of the table the plan locked at the start
- 3-fold CV ROC-AUC on training: 0.7724 (std 0.0164)
- holdout ROC-AUC (scored once): 0.7508
- difference, CV minus holdout: +0.0216 (absolute 0.0216)
- largest feature shifts, training vs holdout (PSI): `X15` 0.023, `X13` 0.021, `X21` 0.021

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
```

**Answer**

```
DECISION: investigate_gap_without_shift
WHY: CV (0.7724) and holdout (0.7508) differ by +0.0216, outside the ±0.02 band, yet no feature shifts much (highest PSI 0.023). The holdout scored lower than CV without a population change, which points to choices fitted to CV noise (optimistic CV) or an unstable estimate on 750 rows, so the gap needs explaining before results are reported.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [continue → WF-10] Do not tune on the holdout. Put a bootstrap interval on the holdout score, review which earlier choices were made on CV, and report both numbers with the explanation.
```

**Input**

```
WORKFLOW STEP: WF-10 Knowledge capture
STAGE: report
Knowledge capture (WF-10): write the final report from the decisions above only. Every number must be one already shown in this run. Output JSON.
```

**Answer**

```
{
 "problem": {
  "goal": "find card holders who will default",
  "target": "defaulted_next_month",
  "prediction_moment": "at the end of each billing period"
 },
 "data": {
  "rows": 3000,
  "training_rows": 2250,
  "holdout_rows": 750,
  "deduplicated_before_split": false
 },
 "solution_revisits": [],
 "features": {
  "excluded": [],
  "held_for_confirmation": [],
  "recipe": "raw"
 },
 "model": "random_forest + sigmoid calibration",
 "results": {
  "cv_roc_auc": "0.7724",
  "cv_std": "0.0164",
  "holdout_roc_auc": "0.7508",
  "holdout_decision": "investigate_gap_without_shift"
 },
 "monitor": [],
 "limits": [
  "CV and holdout disagree beyond the ±0.02 band without a population shift; report both numbers with a bootstrap interval, not the higher one"
 ],
 "status": "complete"
}
```

### `end_to_end (labels_missing, stops at WF-10)`  (bank_marketing, public)

**Input**

```
WORKFLOW STEP: WF-01 Solution draft
STAGE: problem_understanding
Problem understanding: read the problem description and the file headers (names only, never values) and write the problem contract. Every known field says how it is known: 'stated' quotes the person's exact words, 'inferred' gives the reason. A field the text does not settle stays unknown and becomes an open question; a wrong value is worse than an unknown one. Output JSON.

DESCRIPTION:
We are a retail bank calling clients about a term deposit. We need to predict whether a client subscribes to a term deposit. Agents call the top clients on the list first. Each row is a client. The subscribed column says 1 when the client took the deposit. We score each client immediately before the call; the call's duration is known only after it ends. The team can make 300 calls a day.

FILES (headers only):
bank_marketing.csv: age, job, marital, education, default, balance, housing, loan, contact, day_of_week, month, duration, campaign, pdays, previous, poutcome, subscribed
```

**Answer**

```
{
 "goal": {
  "value": "a client subscribes to a term deposit",
  "status": "stated",
  "quote": "We need to predict whether a client subscribes to a term deposit."
 },
 "action": {
  "value": "agents call the top clients first",
  "status": "stated",
  "quote": "Agents call the top clients on the list first."
 },
 "unit": {
  "value": "client",
  "status": "stated",
  "quote": "Each row is a client."
 },
 "task": {
  "value": "binary",
  "status": "inferred",
  "reason": "subscribed is 1 or 0, so the answer is binary"
 },
 "target_column": {
  "value": "subscribed",
  "status": "stated",
  "quote": "The subscribed column says 1 when the client took the deposit."
 },
 "label_source": {
  "value": "column",
  "status": "inferred",
  "reason": "the label is a column of bank_marketing.csv"
 },
 "label_coverage": {
  "value": "complete",
  "status": "inferred",
  "reason": "the label is a column of the main table, so every row is labelled"
 },
 "prediction_moment": {
  "value": "immediately before the call",
  "status": "stated",
  "quote": "We score each client immediately before the call; the call's duration is known only after it ends."
 },
 "unknown": [
  "time_column",
  "horizon",
  "entities",
  "primary_metric"
 ],
 "constraints": [
  {
   "kind": "review_capacity",
   "text": "300 calls a day",
   "quote": "The team can make 300 calls a day."
  }
 ],
 "open_questions": [
  {
   "field": "entities",
   "text": "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?",
   "impact": 3
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-02 Source and lineage"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-02 Source and lineage
STAGE: contract_check
Contract check: before any analysis, check the real table against the problem contract. Checks, in order: target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); target_varies (more than one distinct value); time_column_present (only if the contract names one); entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); proceed when every check passes. Output JSON.

Check the contract above against the table.

DATA PROFILE:
TABLE: bank_marketing — 3,000 rows, 17 columns
| column | kind | distinct | blank |
|---|---|---|---|
| age | number | 65 | 0.0% |
| job | number | 12 | 0.0% |
| marital | number | 3 | 0.0% |
| education | number | 4 | 0.0% |
| default | number | 2 | 0.0% |
| balance | number | 1,780 | 0.0% |
| housing | number | 2 | 0.0% |
| loan | number | 2 | 0.0% |
| contact | number | 3 | 0.0% |
| day_of_week | number | 31 | 0.0% |
| month | number | 12 | 0.0% |
| duration | number | 741 | 0.0% |
| campaign | number | 30 | 0.0% |
| pdays | number | 265 | 0.0% |
| previous | number | 20 | 0.0% |
| poutcome | number | 4 | 0.0% |
| subscribed | text | 2 | 21.3% |

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
   "status": "fail",
   "detail": "21.3% of `subscribed` values are blank"
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
   "field": "label_coverage",
   "from": "complete",
   "to": "partial",
   "reason": "21.3% of rows have no label; unlabelled is not the same as negative"
  }
 ],
 "questions": [
  {
   "field": "label_coverage",
   "text": "Are the rows without a label unknown, or do they mean something (for example 'not checked')?",
   "impact": 2
  }
 ],
 "next": "Mark coverage partial, keep unlabelled rows out of training and evaluation, then run the data analysis.",
 "route": {
  "action": "back",
  "to": "WF-01 Solution draft",
  "why": "reopen the solution: the data corrects an inferred fact"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-01 Solution draft
STAGE: solution_revisit
Solution revisit (WF-01): the step that sent the work back found something the solution did not know. Record exactly that change, say what stands, and route the work back to where it continues. Output JSON.

SENT BACK FROM: WF-02 Source and lineage
```

**Answer**

```
{
 "change": {
  "label_coverage": {
   "from": "complete",
   "to": "partial",
   "reason": "21.3% of rows have no label; unlabelled is not the same as negative"
  }
 },
 "stands": "Every stated fact of the contract stands; only the inferred fact the data contradicted changes.",
 "data_rule": "Rows without a label leave the table before the split; they are neither training nor test rows.",
 "route": {
  "action": "continue",
  "to": "WF-03 Split design",
  "why": "the contract is corrected; design the split on the labelled rows"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature. If several thresholds are crossed, act on the first in this order: deduplicate_before_split, small_data_protocol, use_imbalance_aware_evaluation, impute_within_folds; the others are carried to the plan.

PROFILE:
- rows: 2,362 | features: 16 | rows per feature: 147.6
- positive rate: 11.4% (minority class 11.4%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 0.0%, `job` 0.0%, `marital` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: proceed
WHY: No threshold is crossed: duplicates 0.0%, minority class 11.4%, highest missing rate 0.0%, 2,362 rows (147.6 per feature). The data is ready for the split and the training-only analysis.
RULE: DCLAB-R01 (contract written, profile clean)
CONFIDENCE: high
NEXT: [continue → WF-04] Write the profile to the decision ledger, design the split, then profile the training partition.
```

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
STAGE: data_analysis
Data analysis: the measurements below were computed on the training partition only, checked against the problem contract (target, prediction moment, time column, entities). For each measurement, write the finding: its severity (info, warning or critical), what it shows, what it implies for the solution, and which plan decisions it affects. Output JSON. Rules used by the measurements: a class is tiny under 1% of rows or 30 rows; a positive class under 1% of rows is rare.

TABLE: bank_marketing — 2,362 rows, 17 columns (age, job, marital, education, default, balance, housing, loan, contact, day_of_week, month, duration, campaign, pdays, previous, poutcome, subscribed)
TRAINING PARTITION: a random 75% of rows (1,771 rows); the rest is the locked holdout

MEASUREMENTS:
- F-keys [grain] evidence: {"rows": 2362, "unique_columns": []}
- F-prevalence [target] evidence: {"positive_class": "1", "positive_rate": 0.11293, "positives": 200, "rows": 1771}
- F-partial-labels [target] evidence: {"positive_rate_lower_bound": 0.11293}
```

**Answer**

```
{
 "task": "binary",
 "target": "subscribed",
 "training_rows": 1771,
 "findings": [
  {
   "id": "F-keys",
   "severity": "info",
   "statement": "bank_marketing has 2362 rows; 0 column(s) identify rows: none.",
   "implication": "A column that identifies rows can only memorise: keep it out of the features.",
   "affects": [
    "features.identifiers"
   ]
  },
  {
   "id": "F-prevalence",
   "severity": "info",
   "statement": "The positive class (1) is 11.29% of training rows.",
   "implication": "Average precision or ROC AUC rank the cases; accuracy depends on the threshold.",
   "affects": [
    "metric.primary",
    "threshold.policy",
    "split.strategy"
   ]
  },
  {
   "id": "F-partial-labels",
   "severity": "warning",
   "statement": "The brief says the labels are a sample: unlabelled rows are not known negatives.",
   "implication": "Report precision as a lower bound; prefer methods that tolerate hidden positives (positive-unlabelled learning, anomaly scores) and never call unlabelled rows clean.",
   "affects": [
    "labels.coverage",
    "model.families",
    "metric.primary"
   ]
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-04 Train-only EDA",
  "why": "the findings go to the solution design"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-04 Train-only EDA
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
    "task": "binary",
    "labels": "partial"
   },
   "cites": [
    "F-partial-labels"
   ],
   "reason": "The problem is binary: the labels are a sample, so unlabelled rows are not known negatives. The brief says so (inferred).",
   "alternatives": [
    {
     "choice": "treat unlabelled rows as negatives",
     "why_not": "they include frauds nobody found: precision would be biased and the model taught that they are clean"
    }
   ]
  },
  {
   "key": "labels.coverage",
   "choice": "partial",
   "cites": [
    "F-partial-labels"
   ],
   "reason": "The brief says the labels are a sample: unlabelled is not negative.",
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
   "reason": "Positives are 11.29% of rows: ROC AUC ranks well at this balance.",
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
    "gradient_boosting",
    "pu_bagging",
    "isolation_forest",
    "ecod",
    "percentile_rules"
   ],
   "cites": [
    "F-partial-labels",
    "F-prevalence"
   ],
   "reason": "Screen linear and tree families on identical folds; with rare or partial labels, positive-unlabelled learning, anomaly detectors and rules join the screen, with a dummy baseline always in.",
   "alternatives": []
  },
  {
   "key": "threshold.policy",
   "choice": {
    "kind": "review_budget",
    "capacity": "300 calls a day"
   },
   "cites": [
    "F-prevalence"
   ],
   "reason": "The team can review 300 calls a day: alert on that many of the riskiest, chosen on out-of-fold scores.",
   "alternatives": [
    {
     "choice": "0.5",
     "why_not": "a probability threshold ignores how many cases the team can review"
    }
   ]
  }
 ],
 "risks": [],
 "would_change": [
  "A cost of errors or a review capacity would set the threshold and may change the primary metric."
 ],
 "workflow": [
  {
   "step": "WF-01 Solution draft",
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
   "step": "WF-02 Source and lineage"
  },
  {
   "step": "WF-03 Split design",
   "decisions": [
    "split.strategy"
   ]
  },
  {
   "step": "WF-04 Train-only EDA",
   "gate": "solution"
  },
  {
   "step": "WF-05 Leakage audit",
   "decisions": [
    "features.forbidden",
    "features.identifiers"
   ]
  },
  {
   "step": "WF-06 Feature ladder",
   "decisions": [
    "features.direction"
   ]
  },
  {
   "step": "WF-07 Algorithm screen",
   "decisions": [
    "model.families"
   ]
  },
  {
   "step": "WF-08 Conservative optimization"
  },
  {
   "step": "WF-09 Reliability challenge",
   "gate": "holdout"
  },
  {
   "step": "WF-10 Knowledge capture"
  }
 ],
 "route": {
  "action": "continue",
  "to": "WF-05 Leakage audit",
  "why": "after the solution gate"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-05 Leakage audit
STAGE: leakage_audit
Leakage audit: decide, column by column, what may enter a model that predicts at the contract's prediction moment. Exclude a column the solution plan forbids, a column the declared decision moment rules out, and an identifier; flag a column the plan holds for confirmation; keep a column with no leakage evidence. A strong score alone starts a review, it does not prove leakage. Columns not listed keep their plan status. A leak the plan did not know reopens the solution (back to WF-01). Output JSON.

DECISION MOMENT (dataset contract): Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
EVIDENCE: computed on the training partition only (1,771 rows); CV uses the plan's other usable columns as the base.
OTHER COLUMNS, not audited one by one: `age`, `job`, `marital`, `default`, `housing`, `loan`, `contact`, `day_of_week`, `campaign`, `pdays`, `previous`
COLUMNS:
- `duration` | number | plan says: not named | data dictionary: Last contact duration, in seconds. | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 34.0% of rows | univariate ROC-AUC 0.801 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.861, without 0.662, change +0.199
- `month` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 0.7% of rows | univariate ROC-AUC 0.624 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.662, without 0.655, change +0.007
- `poutcome` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 0.2% of rows | univariate ROC-AUC 0.620 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.662, without 0.639, change +0.023
- `balance` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 68.2% of rows | univariate ROC-AUC 0.613 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.662, without 0.640, change +0.023
- `education` | number | plan says: not named | data dictionary: no entry | missing 0.0% (positives 0.0%, negatives 0.0%) | unique 0.2% of rows | univariate ROC-AUC 0.553 | target copy: no | suspicious name: no | row-number pattern: no | 3-fold CV ROC-AUC with 0.662, without 0.649, change +0.013

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
{
 "columns": [
  {
   "column": "duration",
   "decision": "exclude",
   "why": "The declared decision moment rules `duration` out (last contact duration, in seconds): it describes what happens during or after the event being predicted. Its signal (univariate ROC-AUC 0.801, CV change +0.199) would not be available in production. The plan's checks on name and power did not catch it, so the solution must record it.",
   "rule": "DCLAB-R04 (post-outcome leakage)"
  },
  {
   "column": "month",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `month` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.624, CV change +0.007).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "poutcome",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `poutcome` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.620, CV change +0.023).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "balance",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `balance` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.613, CV change +0.023).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  },
  {
   "column": "education",
   "decision": "keep",
   "why": "Neither the plan nor the decision moment marks `education` as written after the prediction moment. The evidence shows no leakage pattern (univariate ROC-AUC 0.553, CV change +0.013).",
   "rule": "DCLAB-R04 (checked: no leakage pattern)"
  }
 ],
 "feature_set": [
  "age",
  "job",
  "marital",
  "education",
  "default",
  "balance",
  "housing",
  "loan",
  "contact",
  "day_of_week",
  "month",
  "campaign",
  "pdays",
  "previous",
  "poutcome"
 ],
 "not_raw_features": [],
 "next": "Build the feature matrix from the feature_set columns on the training partition and compare feature recipes.",
 "route": {
  "action": "back",
  "to": "WF-01 Solution draft",
  "why": "a leak changes the solution: `duration` must be recorded as forbidden"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-01 Solution draft
STAGE: solution_revisit
Solution revisit (WF-01): the step that sent the work back found something the solution did not know. Record exactly that change, say what stands, and route the work back to where it continues. Output JSON.

SENT BACK FROM: WF-05 Leakage audit
```

**Answer**

```
{
 "change": {
  "features.forbidden": {
   "add": [
    {
     "column": "duration",
     "reason": "ruled out by the declared decision moment; found by the leakage audit"
    }
   ]
  }
 },
 "stands": "Every other decision of the solution stands; the audit already kept these columns out of the feature set.",
 "route": {
  "action": "continue",
  "to": "WF-06 Feature ladder",
  "why": "the audit is complete; continue with the feature ladder"
 }
}
```

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

TRAINING PARTITION: 1,771 rows | FEATURES: 15 | POSITIVE RATE: 11.3%
MODEL USED FOR THE COMPARISON: hist_gradient_boosting | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 15 | 0.6728 | 0.0091 | — | — |
| log_transforms | 18 | 0.6728 | 0.0091 | +0.0000 | 0/5 |
| ratios | 21 | 0.6759 | 0.0222 | +0.0032 | 2/5 |
| interactions | 21 | 0.6558 | 0.0370 | -0.0169 | 2/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. The closest, `ratios`, changes ROC-AUC by +0.0032 and wins on 2/5 folds, which is not a consistent gain of at least +0.005. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw feature set into model selection.
```

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

TRAINING PARTITION: 1,771 rows | FEATURES: 15 | POSITIVE RATE: 11.3%
FEATURE SET: raw (15 features)
FAMILIES FROM THE SOLUTION PLAN: logistic_regression, random_forest, gradient_boosting, pu_bagging, isolation_forest, ecod, percentile_rules (gradient_boosting runs as hist_gradient_boosting; the dummy baseline predicts the training prevalence, so its ROC-AUC is 0.5 by construction)
SCREEN: 3 stratified folds, identical for every model
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.7065 | 0.0211 | 0.7012 | 0.04 |
| random_forest | 0.7195 | 0.0181 | 0.7150 | 1.17 |
| hist_gradient_boosting | 0.6562 | 0.0131 | 0.6529 | 0.74 |

ALLOWED ACTIONS: logistic_regression | random_forest | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` has the highest stability-adjusted score (0.7150: mean 0.7195, std 0.0181), ahead of `logistic_regression` at 0.7012. The linear baseline scores 0.7012, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: [continue → WF-08] Carry `random_forest` into optimization; keep `logistic_regression` as the paired comparison when tuning.
```

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

TRAINING PARTITION: 1,771 rows | FEATURES: 15 | POSITIVE RATE: 11.3%
MODEL: random_forest | 3 stratified folds | training rows per fold: about 1,180
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0249 | 0.7092 | 0.0899 |
| sigmoid | 0.1393 | 0.3091 | 0.1216 |
| isotonic | 0.0294 | 0.7007 | 0.0923 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
```

**Answer**

```
DECISION: keep_uncalibrated
WHY: Raw `random_forest` probabilities are already usable or calibration does not pay: raw ECE is 0.0249 (at or below the 0.03 bar). Adding a calibration layer would add complexity for no measured benefit.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Lock the uncalibrated model and move to the final holdout check.
```

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

LOCKED PIPELINE: raw features → random_forest
training rows: 1,771 | holdout rows: 591 (holdout positive rate 11.7%); the holdout is the part of the table the plan locked at the start
- 3-fold CV ROC-AUC on training: 0.7193 (std 0.0294)
- holdout ROC-AUC (scored once): 0.7286
- difference, CV minus holdout: -0.0092 (absolute 0.0092)
- largest feature shifts, training vs holdout (PSI): `month` 0.026, `balance` 0.026, `age` 0.021

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
```

**Answer**

```
DECISION: accept_result
WHY: CV ROC-AUC 0.7193 and holdout ROC-AUC 0.7286 differ by -0.0092, inside the ±0.02 band, so the CV estimate held up on untouched data. No feature shifted meaningfully between training and holdout.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [continue → WF-10] Report CV and holdout together with uncertainty, and do not tune anything on the holdout.
```

**Input**

```
WORKFLOW STEP: WF-10 Knowledge capture
STAGE: report
Knowledge capture (WF-10): write the final report from the decisions above only. Every number must be one already shown in this run. Output JSON.
```

**Answer**

```
{
 "problem": {
  "goal": "a client subscribes to a term deposit",
  "target": "subscribed",
  "prediction_moment": "immediately before the call"
 },
 "data": {
  "rows": 2362,
  "training_rows": 1771,
  "holdout_rows": 591,
  "deduplicated_before_split": false
 },
 "solution_revisits": [
  "WF-02 → WF-01: label coverage corrected to partial",
  "WF-05 → WF-01: duration recorded as forbidden"
 ],
 "features": {
  "excluded": [
   "duration"
  ],
  "held_for_confirmation": [],
  "recipe": "raw"
 },
 "model": "random_forest",
 "results": {
  "cv_roc_auc": "0.7193",
  "cv_std": "0.0294",
  "holdout_roc_auc": "0.7286",
  "holdout_decision": "accept_result"
 },
 "monitor": [],
 "limits": [],
 "status": "complete"
}
```

## data_understanding

### `deduplicate_before_split`  (spambase, du_deduplicate_before_split)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 2,010 | FEATURES: 57 | POSITIVE RATE: 39.7%

PROFILE:
- rows: 2,010 | features: 57 | rows per feature: 35.3
- positive rate: 39.7% (minority class 39.7%)
- exact duplicate rows (features + target): 16.6%
- highest missing rates: `word_freq_make` 0.0%, `word_freq_address` 0.0%, `word_freq_all` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: deduplicate_before_split
WHY: 16.6% of rows are exact duplicates, above the 1% threshold. If they are split randomly, copies of the same row land in both training and test folds, so the test score partly measures memory. Duplicates must be removed, or grouped so all copies stay in one fold, before any split.
RULE: DCLAB-R04 (duplicate/entity leakage) and DCLAB-R02 (split by the independence unit)
CONFIDENCE: high
NEXT: [continue → WF-04] Deduplicate (or define groups) before the split, re-profile, then profile the training partition.
```

### `impute_within_folds`  (spambase, du_impute_within_folds)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 3,296 | FEATURES: 57 | POSITIVE RATE: 39.5%

PROFILE:
- rows: 3,296 | features: 57 | rows per feature: 57.8
- positive rate: 39.5% (minority class 39.5%)
- exact duplicate rows (features + target): 0.1%
- highest missing rates: `word_freq_re` 58.5%, `word_freq_table` 38.8%, `word_freq_3d` 14.5%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: impute_within_folds
WHY: `word_freq_re` is missing in 58.5% of rows, above the 5% threshold. Filling it with statistics from the whole dataset would leak test-fold information into training, so imputation (and a missing-value indicator, since absence may carry meaning) must be fitted inside each training fold.
RULE: DCLAB-R03 (fit imputers on training partitions only) and DCLAB-R12 (missingness as a state)
CONFIDENCE: high
NEXT: [continue → WF-04] Plan in-fold imputation with missing indicators, design the split, then profile the training partition.
```

### `proceed`  (mushroom, du_proceed)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS USED: 1,847 | FEATURES: 22 | POSITIVE RATE: 47.9%

PROFILE:
- rows: 1,847 | features: 22 | rows per feature: 84.0
- positive rate: 47.9% (minority class 47.9%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `cap-shape` 0.0%, `cap-surface` 0.0%, `cap-color` 0.0%
- near-constant columns (one value in >99.5% of rows): 1

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: proceed
WHY: No threshold is crossed: duplicates 0.0%, minority class 47.9%, highest missing rate 0.0%, 1,847 rows (84.0 per feature). The data is ready for the split and the training-only analysis.
RULE: DCLAB-R01 (contract written, profile clean)
CONFIDENCE: high
NEXT: [continue → WF-04] Write the profile to the decision ledger, design the split, then profile the training partition.
```

### `small_data_protocol`  (adult, du_small_data_protocol)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 224 | FEATURES: 14 | POSITIVE RATE: 20.1%

PROFILE:
- rows: 224 | features: 14 | rows per feature: 16.0
- positive rate: 20.1% (minority class 20.1%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 0.0%, `workclass` 0.0%, `fnlwgt` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: small_data_protocol
WHY: There are only 224 rows (16.0 per feature). Single splits on data this small swing widely, so use repeated stratified CV, start with simple regularized models, and report wide uncertainty instead of single scores.
RULE: DCLAB-R16 (paired, stable comparisons) and DCLAB-R13 (start from a transparent baseline)
CONFIDENCE: high
NEXT: [continue → WF-04] Use repeated stratified CV with a linear baseline as the split design, then profile the training partition.
```

### `use_imbalance_aware_evaluation`  (adult, du_use_imbalance_aware_evaluation)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 2,276 | FEATURES: 14 | POSITIVE RATE: 2.4%

PROFILE:
- rows: 2,276 | features: 14 | rows per feature: 162.6
- positive rate: 2.4% (minority class 2.4%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 0.0%, `workclass` 0.0%, `fnlwgt` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: use_imbalance_aware_evaluation
WHY: The minority class is only 2.4% of rows, below the 10% threshold. Accuracy and even ROC-AUC can look good while the model misses most positives, so evaluation needs stratified folds, PR-AUC, and a threshold chosen from the cost of each error.
RULE: DCLAB-R18 (metrics and thresholds from the decision cost)
CONFIDENCE: high
NEXT: [continue → WF-04] Build the split and evaluation plan around the imbalance (stratified folds, PR-AUC, cost-based threshold), then profile the training partition.
```

## leakage_audit

### `exclude`  (adult, date_after_decision)

**Input**

```
WORKFLOW STEP: WF-05 Leakage audit
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS: 48,842 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 23.9% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.728

COLUMN: `record_date`
DATA DICTIONARY: no entry
EVIDENCE:
- type: date (stored as days relative to the decision moment)
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 12.1% of rows
- univariate ROC-AUC: 0.809
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- values after the decision moment: 15.1%
- 3-fold CV ROC-AUC with the column: 0.960 | without: 0.905 | change: +0.056

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: exclude
WHY: 15% of `record_date` values fall after the decision moment, and those late dates are concentrated in one class (univariate ROC-AUC 0.809). At prediction time these dates do not exist yet, so the column leaks the future.
RULE: DCLAB-R04 (future leakage)
CONFIDENCE: high
NEXT: [continue → WF-05] Remove `record_date` from the feature list, record the reason in the decision ledger, and continue the audit.
```

### `flag_for_review`  (credit_default, strong_signal_unknown_origin)

**Input**

```
WORKFLOW STEP: WF-05 Leakage audit
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS: 30,000 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 22.1% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.701

COLUMN: `prior_outcome_count`
DATA DICTIONARY: no entry
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 100.0% of rows
- univariate ROC-AUC: 0.982
- identical to the target (or its inverse): no
- name matches the suspicious-word list: yes
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.980 | without: 0.732 | change: +0.248

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: flag_for_review
WHY: `prior_outcome_count` is unusually strong (univariate ROC-AUC 0.982 vs 0.701 for the strongest real feature) and raises CV ROC-AUC by +0.248, but there is no data-dictionary entry saying when it is recorded. A strong score alone does not prove leakage, and it does not prove availability either, so a person with the source lineage has to confirm it.
RULE: DCLAB-R05 (high AUC starts a review, it does not prove leakage)
CONFIDENCE: medium
NEXT: [continue → WF-05] Keep `prior_outcome_count` out of all models until the data owner confirms when it is recorded; continue the audit meanwhile.
```

### `keep`  (credit_default, strong_but_available)

**Input**

```
WORKFLOW STEP: WF-05 Leakage audit
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS: 30,000 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 22.1% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.701

COLUMN: `index_c`
DATA DICTIONARY: Computed at the decision moment from `X9` and `X7`.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 0.9% of rows
- univariate ROC-AUC: 0.652
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.739 | without: 0.732 | change: +0.007

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: keep
WHY: `index_c` has moderate signal (univariate ROC-AUC 0.652), and its lineage is clear: it is computed at the decision moment from `X9` and `X7`, which are themselves available then. Predictive power from available inputs is the goal, not a leak.
RULE: DCLAB-R05 (high AUC alone does not prove leakage)
CONFIDENCE: high
NEXT: [continue → WF-05] Keep `index_c` and continue the audit with the next column.
```

### `rebuild_within_folds`  (mushroom, full_data_target_encoding)

**Input**

```
WORKFLOW STEP: WF-05 Leakage audit
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS: 8,124 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 48.2% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.920

COLUMN: `avg_outcome_by_stalk_root`
DATA DICTIONARY: Mean of the target per `stalk-root` group, calculated once using all rows, including evaluation rows.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 0.1% of rows
- univariate ROC-AUC: 0.711
- identical to the target (or its inverse): no
- name matches the suspicious-word list: yes
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 1.000 | without: 1.000 | change: +0.000

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: rebuild_within_folds
WHY: Encoding `stalk-root` by its outcome rate is a legitimate idea, but this version was computed on all rows, so every evaluation row's own label is inside its feature value. Here the visible effect is small (CV change +0.000) because each group is large, but the method is still wrong and the same mistake on a high-cardinality column produces large fake gains. The fix is to recompute the rate inside each training fold, not to drop the idea.
RULE: DCLAB-R04 (cross-split aggregate leakage)
CONFIDENCE: high
NEXT: [continue → WF-05] Drop this version of `avg_outcome_by_stalk_root`, recompute it out-of-fold inside each training split, and re-audit it.
```

## feature_engineering

### `adopt_interactions`  (adult, fe_sim_interaction_logistic_regression)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
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
| raw | 14 | 0.5755 | 0.0127 | — | — |
| log_transforms | 19 | 0.5775 | 0.0130 | +0.0020 | 3/5 |
| ratios | 20 | 0.5955 | 0.0137 | +0.0201 | 5/5 |
| interactions | 20 | 0.6759 | 0.0133 | +0.1004 | 5/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_interactions
WHY: `interactions` improves on raw by +0.1004 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. Other recipes that also qualified gained less. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw + interactions feature set into model selection and log each new feature's formula.
```

### `adopt_log_transforms`  (spambase, fe_real_logistic_regression)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 3,000 | FEATURES: 57 | POSITIVE RATE: 39.4%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 57 | 0.9693 | 0.0051 | — | — |
| log_transforms | 114 | 0.9798 | 0.0048 | +0.0105 | 5/5 |
| ratios | 63 | 0.9695 | 0.0052 | +0.0002 | 4/5 |
| interactions | 63 | 0.9690 | 0.0049 | -0.0003 | 1/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_log_transforms
WHY: `log_transforms` improves on raw by +0.0105 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 57 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw + log_transforms feature set into model selection and log each new feature's formula.
```

### `adopt_ratios`  (spambase, fe_sim_interaction_logistic_regression)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 57 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 57 | 0.5374 | 0.0230 | — | — |
| log_transforms | 114 | 0.5494 | 0.0217 | +0.0120 | 5/5 |
| ratios | 63 | 0.5661 | 0.0272 | +0.0288 | 5/5 |
| interactions | 63 | 0.5658 | 0.0292 | +0.0285 | 5/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_ratios
WHY: `ratios` improves on raw by +0.0288 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. Other recipes that also qualified gained less. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw + ratios feature set into model selection and log each new feature's formula.
```

### `keep_raw`  (breast_cancer, fe_real_hist_gradient_boosting)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 569 | FEATURES: 30 | POSITIVE RATE: 37.3%

MODEL USED FOR THE COMPARISON: hist_gradient_boosting | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 30 | 0.9956 | 0.0013 | — | — |
| log_transforms | 52 | 0.9956 | 0.0013 | +0.0000 | 0/5 |
| ratios | 36 | 0.9942 | 0.0020 | -0.0014 | 0/5 |
| interactions | 36 | 0.9936 | 0.0013 | -0.0020 | 1/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. The closest, `log_transforms`, changes ROC-AUC by +0.0000 and wins on 0/5 folds, which is not a consistent gain of at least +0.005. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw feature set into model selection.
```

## model_selection

### `extra_trees`  (spambase, ms_real_300)

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 300 | FEATURES: 57 | POSITIVE RATE: 40.3%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9266 | 0.0203 | 0.9215 | 0.04 |
| random_forest | 0.9421 | 0.0206 | 0.9370 | 0.90 |
| extra_trees | 0.9495 | 0.0164 | 0.9454 | 0.75 |
| hist_gradient_boosting | 0.9369 | 0.0183 | 0.9324 | 0.44 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: extra_trees
WHY: `extra_trees` has the highest stability-adjusted score (0.9454: mean 0.9495, std 0.0164), ahead of `random_forest` at 0.9370. The linear baseline scores 0.9215, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: [continue → WF-08] Carry `extra_trees` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `hist_gradient_boosting`  (spambase, ms_sim_linear_full)

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 4,000 | FEATURES: 57 | POSITIVE RATE: 50.0%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9272 | 0.0040 | 0.9262 | 0.12 |
| random_forest | 0.9334 | 0.0051 | 0.9322 | 1.65 |
| extra_trees | 0.9299 | 0.0035 | 0.9290 | 1.33 |
| hist_gradient_boosting | 0.9347 | 0.0039 | 0.9338 | 1.39 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: hist_gradient_boosting
WHY: `hist_gradient_boosting` and `random_forest` are effectively tied: stability-adjusted scores 0.9338 vs 0.9322 (within 0.002) and similar runtimes (1.39s vs 1.65s). Nothing here separates them meaningfully, so `hist_gradient_boosting` goes forward on its slightly higher score, as a low-confidence choice that later evidence (calibration, stability on new folds) can overturn.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: low
NEXT: [continue → WF-08] Carry `hist_gradient_boosting` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `logistic_regression`  (breast_cancer, ms_real_full)

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 569 | FEATURES: 30 | POSITIVE RATE: 37.3%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9958 | 0.0041 | 0.9947 | 0.03 |
| random_forest | 0.9917 | 0.0062 | 0.9901 | 0.90 |
| extra_trees | 0.9939 | 0.0050 | 0.9926 | 0.67 |
| hist_gradient_boosting | 0.9929 | 0.0060 | 0.9914 | 0.48 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: logistic_regression
WHY: `logistic_regression` has the highest stability-adjusted score (0.9947: mean 0.9958, std 0.0041), ahead of `extra_trees` at 0.9926. The linear baseline wins here, which is exactly why it is always screened: no family is best on every dataset.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: [continue → WF-08] Carry `logistic_regression` into optimization; keep `extra_trees` as the paired comparison when tuning.
```

### `random_forest`  (adult, ms_real_1500)

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 1,500 | FEATURES: 14 | POSITIVE RATE: 23.7%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.8471 | 0.0129 | 0.8439 | 0.03 |
| random_forest | 0.9015 | 0.0056 | 0.9001 | 1.05 |
| extra_trees | 0.8961 | 0.0012 | 0.8958 | 0.85 |
| hist_gradient_boosting | 0.8979 | 0.0081 | 0.8958 | 0.79 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` has the highest stability-adjusted score (0.9001: mean 0.9015, std 0.0056), ahead of `hist_gradient_boosting` at 0.8958. The linear baseline scores 0.8439, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: [continue → WF-08] Carry `random_forest` into optimization; keep `hist_gradient_boosting` as the paired comparison when tuning.
```

## optimization_reliability

### `accept_result`  (mushroom, opt_holdout_random)

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS USED: 4,000 | FEATURES: 22 | POSITIVE RATE: 48.2%

MODEL: hist_gradient_boosting (locked) | training rows: 2,992 | holdout rows: 1,008 (holdout positive rate 46.2%)
- 3-fold CV ROC-AUC on training: 1.0000 (std 0.0000)
- holdout ROC-AUC (scored once): 1.0000
- difference, CV minus holdout: +0.0000
- largest feature shifts, training vs holdout (PSI): `cap-color` 0.022, `stalk-color-above-ring` 0.014, `gill-color` 0.011

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: accept_result
WHY: CV ROC-AUC 1.0000 and holdout ROC-AUC 1.0000 differ by +0.0000, inside the ±0.02 band, so the CV estimate held up on untouched data. No feature shifted meaningfully between training and holdout.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [continue → WF-10] Report CV and holdout together with uncertainty, and do not tune anything on the holdout.
```

### `calibrate_isotonic`  (spambase, opt_calibration_logistic_regression)

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 900 | FEATURES: 57 | POSITIVE RATE: 36.4%

MODEL: logistic_regression | 3 stratified folds | training rows per fold: about 600
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0420 | 0.9550 | 0.0703 |
| sigmoid | 0.0744 | 0.9622 | 0.0735 |
| isotonic | 0.0260 | 0.9619 | 0.0626 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_isotonic
WHY: Raw `logistic_regression` probabilities are off by an ECE of 0.0420, above the 0.03 bar. Isotonic calibration fitted inside the training folds brings ECE to 0.0260 with ROC-AUC 0.9619 vs 0.9550 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Add isotonic calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `calibrate_sigmoid`  (credit_default, opt_calibration_hist_gradient_boosting)

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS USED: 900 | FEATURES: 23 | POSITIVE RATE: 23.4%

MODEL: hist_gradient_boosting | 3 stratified folds | training rows per fold: about 600
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.1319 | 0.7035 | 0.1767 |
| sigmoid | 0.0493 | 0.7202 | 0.1573 |
| isotonic | 0.0422 | 0.7214 | 0.1547 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_sigmoid
WHY: Raw `hist_gradient_boosting` probabilities are off by an ECE of 0.1319, above the 0.03 bar. Sigmoid calibration fitted inside the training folds brings ECE to 0.0493 with ROC-AUC 0.7202 vs 0.7035 raw, so ranking is kept while probabilities become trustworthy. Sigmoid is preferred over isotonic because the training folds have only about 600 rows (under 1,000), where isotonic tends to overfit.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Add sigmoid calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `investigate_distribution_shift`  (credit_default, opt_holdout_shifted)

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS USED: 4,000 | FEATURES: 23 | POSITIVE RATE: 22.1%

MODEL: hist_gradient_boosting (locked) | training rows: 3,000 | holdout rows: 1,000 (holdout positive rate 23.9%)
- 3-fold CV ROC-AUC on training: 0.7663 (std 0.0084)
- holdout ROC-AUC (scored once): 0.5930
- difference, CV minus holdout: +0.1733
- largest feature shifts, training vs holdout (PSI): `X16` 12.434, `X15` 4.403, `X17` 3.628

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_distribution_shift
WHY: The holdout ROC-AUC (0.5930) is 0.1733 below CV (0.7663), outside the ±0.02 band, and `X16` has a PSI of 12.434 (above 0.2): the holdout population is different from training. The CV number does not describe this population, so the shift must be explained before any claim is made.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [back → WF-03] Do not tune on the holdout. Slice results by the shifted feature, document the gap, and redesign the split so training and test reflect that shift (by time or by the shifted group); the spent holdout is reported, never reused.
```

### `investigate_gap_without_shift`  (credit_default, opt_holdout_random)

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS USED: 4,000 | FEATURES: 23 | POSITIVE RATE: 22.1%

MODEL: hist_gradient_boosting (locked) | training rows: 2,984 | holdout rows: 1,016 (holdout positive rate 21.9%)
- 3-fold CV ROC-AUC on training: 0.7255 (std 0.0188)
- holdout ROC-AUC (scored once): 0.7635
- difference, CV minus holdout: -0.0380
- largest feature shifts, training vs holdout (PSI): `X15` 0.018, `X5` 0.015, `X12` 0.013

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_gap_without_shift
WHY: CV (0.7255) and holdout (0.7635) differ by -0.0380, outside the ±0.02 band, yet no feature shifts much (highest PSI 0.018). The holdout scored higher than CV, which usually means an easier or luckier sample of 1,016 rows rather than a better model, so the higher number must not be reported as the expected performance.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [continue → WF-10] Do not tune on the holdout. Put a bootstrap interval on the holdout score, review which earlier choices were made on CV, and report both numbers with the explanation.
```

### `keep_uncalibrated`  (mushroom, opt_calibration_random_forest)

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS USED: 900 | FEATURES: 22 | POSITIVE RATE: 46.8%

MODEL: random_forest | 3 stratified folds | training rows per fold: about 600
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0155 | 0.9998 | 0.0073 |
| sigmoid | 0.0169 | 0.9997 | 0.0071 |
| isotonic | 0.0051 | 0.9998 | 0.0040 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: keep_uncalibrated
WHY: Raw `random_forest` probabilities are already usable or calibration does not pay: raw ECE is 0.0155 (at or below the 0.03 bar). Adding a calibration layer would add complexity for no measured benefit.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Lock the uncalibrated model and move to the final holdout check.
```

## code_then_decide

### `accept_result`  (credit_default, code_holdout)

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Final holdout check for hist_gradient_boosting on `data/public/credit_default` (target: whether a client defaults on next month's credit-card payment; stratified sample of 3,000 rows, seed 1; drop the contract's blocked columns: none); the holdout is the top quarter of rows by `X5` (a deliberately shifted holdout). Report 3-fold CV ROC-AUC on the training part (mean, std), the holdout ROC-AUC scored once, the difference CV minus holdout and its absolute value, and the three largest feature shifts (PSI) between training and holdout.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 1
MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),
}


def psi(a, b, bins=10):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if a.nunique() < 2 or len(b) < 20:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ha = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    hb = np.clip(np.histogram(b, edges)[0] / len(b), 1e-6, None)
    return float(np.sum((ha - hb) * np.log(ha / hb)))


X = pd.read_parquet("data/public/credit_default/X.parquet")
y = pd.read_parquet("data/public/credit_default/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=1)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
holdout = (X["X5"].rank(method="first", pct=True) > 0.75).to_numpy()
Xtr, ytr = X[~holdout].reset_index(drop=True), y[~holdout].reset_index(drop=True)
Xho, yho = X[holdout], y[holdout]

cv = []
for tr, te in StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(Xtr, ytr):
    m = MODELS["hist_gradient_boosting"]().fit(Xtr.iloc[tr], ytr.iloc[tr])
    cv.append(roc_auc_score(ytr.iloc[te], m.predict_proba(Xtr.iloc[te])[:, 1]))
cv = np.array(cv)
final = MODELS["hist_gradient_boosting"]().fit(Xtr, ytr)  # locked: every choice is made before this line
holdout_auc = roc_auc_score(yho, final.predict_proba(Xho)[:, 1])  # scored once
shifts = sorted(((c, psi(Xtr[c], Xho[c])) for c in X.columns), key=lambda t: -t[1])[:3]
print(json.dumps({
    "training_rows": len(ytr), "holdout_rows": int(len(yho)),
    "cv_roc_auc_mean": round(float(cv.mean()), 6), "cv_roc_auc_std": round(float(cv.std()), 6),
    "holdout_roc_auc": round(float(holdout_auc), 6),
    "cv_minus_holdout": round(float(cv.mean() - holdout_auc), 6),
    "abs_difference": round(float(abs(cv.mean() - holdout_auc)), 6),
    "largest_psi": {c: round(p, 4) for c, p in shifts},
}, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
EXECUTION OUTPUT:
```json
{
 "training_rows": 2250,
 "holdout_rows": 750,
 "cv_roc_auc_mean": 0.772624,
 "cv_roc_auc_std": 0.022024,
 "holdout_roc_auc": 0.757487,
 "cv_minus_holdout": 0.015137,
 "abs_difference": 0.015137,
 "largest_psi": {
  "X5": 12.3967,
  "X4": 0.5658,
  "X3": 0.2682
 }
}
```

Now decide.
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
```

**Answer**

```
DECISION: accept_result
WHY: CV ROC-AUC 0.7726 and holdout ROC-AUC 0.7575 differ by +0.0151, inside the ±0.02 band, so the CV estimate held up on untouched data. `X5` does shift (PSI 12.397), so it goes on the monitoring list, but it did not hurt performance here.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [continue → WF-10] Report CV and holdout together with uncertainty, and do not tune anything on the holdout.
```

### `adopt_log_transforms`  (credit_default, code_feature_ladder)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Feature ladder for `data/public/credit_default` (target: whether a client defaults on next month's credit-card payment; use a stratified sample of 3,000 rows, seed 4, and drop the columns the dataset's contract blocks: none). Compare the raw columns with three recipes on the SAME 5 stratified folds and the same model (logistic_regression): log1p of non-negative skewed columns (|skew| > 1, more than 10 distinct values), pairwise ratios and pairwise products of standardized values of `X17`, `X1`, `X19`. For each recipe report the feature count, mean and std ROC-AUC, the mean paired gain over raw and how many folds improved.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 4
MODELS = {
    "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),
}

X = pd.read_parquet("data/public/credit_default/X.parquet")
y = pd.read_parquet("data/public/credit_default/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=4)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
pick = [np.str_('X17'), np.str_('X1'), np.str_('X19')]
cont = [c for c in X.columns if X[c].nunique() > 10]
skewed = [c for c in cont if X[c].min() >= 0 and abs(X[c].skew()) > 1]
pairs = [(a, b) for i, a in enumerate(pick) for b in pick[i + 1:]]

def recipes(frame, stats):
    out = {"raw": frame}
    if skewed:
        out["log_transforms"] = frame.assign(**{f"log1p_{c}": np.log1p(frame[c]) for c in skewed})
    out["ratios"] = frame.assign(**{f"{a}_over_{b}": frame[a] / (frame[b].abs() + 1e-3) for a, b in pairs})
    z = {c: (frame[c] - stats[c][0]) / (stats[c][1] + 1e-9) for c in pick}
    out["interactions"] = frame.assign(**{f"{a}_x_{b}": z[a] * z[b] for a, b in pairs})
    return out

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
aucs, n_features = {}, {}
for tr, te in folds:
    stats = {c: (X.iloc[tr][c].mean(), X.iloc[tr][c].std()) for c in pick}  # training fold only
    train_sets, test_sets = recipes(X.iloc[tr], stats), recipes(X.iloc[te], stats)
    for name in train_sets:
        n_features[name] = train_sets[name].shape[1]
        m = MODELS["logistic_regression"]().fit(train_sets[name], y.iloc[tr])
        aucs.setdefault(name, []).append(roc_auc_score(y.iloc[te], m.predict_proba(test_sets[name])[:, 1]))

raw = np.array(aucs["raw"])
report = {}
for name, a in aucs.items():
    a = np.array(a)
    report[name] = {"features": n_features[name],
                    "mean_roc_auc": round(float(a.mean()), 6), "std": round(float(a.std()), 6),
                    "mean_paired_gain": None if name == "raw" else round(float((a - raw).mean()), 6),
                    "folds_improved": None if name == "raw" else int(((a - raw) > 0).sum())}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
EXECUTION OUTPUT:
```json
{
 "raw": {
  "features": 23,
  "mean_roc_auc": 0.718365,
  "std": 0.037503,
  "mean_paired_gain": null,
  "folds_improved": null
 },
 "log_transforms": {
  "features": 29,
  "mean_roc_auc": 0.741732,
  "std": 0.033099,
  "mean_paired_gain": 0.023368,
  "folds_improved": 5
 },
 "ratios": {
  "features": 26,
  "mean_roc_auc": 0.724316,
  "std": 0.035053,
  "mean_paired_gain": 0.005951,
  "folds_improved": 4
 },
 "interactions": {
  "features": 26,
  "mean_roc_auc": 0.719839,
  "std": 0.035683,
  "mean_paired_gain": 0.001474,
  "folds_improved": 3
 }
}
```

Now decide.
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
```

**Answer**

```
DECISION: adopt_log_transforms
WHY: `log_transforms` improves on raw by +0.0234 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw + log_transforms feature set into model selection and log each new feature's formula.
```

### `adopt_ratios`  (adult, code_feature_ladder)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Feature ladder for `data/public/adult` (target: whether a person earns more than $50K per year; use a stratified sample of 3,000 rows, seed 2, and drop the columns the dataset's contract blocks: none). Compare the raw columns with three recipes on the SAME 5 stratified folds and the same model (logistic_regression): log1p of non-negative skewed columns (|skew| > 1, more than 10 distinct values), pairwise ratios and pairwise products of standardized values of `hours-per-week`, `education`, `age`. For each recipe report the feature count, mean and std ROC-AUC, the mean paired gain over raw and how many folds improved.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2
MODELS = {
    "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),
}

X = pd.read_parquet("data/public/adult/X.parquet")
y = pd.read_parquet("data/public/adult/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=2)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
pick = [np.str_('hours-per-week'), np.str_('education'), np.str_('age')]
cont = [c for c in X.columns if X[c].nunique() > 10]
skewed = [c for c in cont if X[c].min() >= 0 and abs(X[c].skew()) > 1]
pairs = [(a, b) for i, a in enumerate(pick) for b in pick[i + 1:]]

def recipes(frame, stats):
    out = {"raw": frame}
    if skewed:
        out["log_transforms"] = frame.assign(**{f"log1p_{c}": np.log1p(frame[c]) for c in skewed})
    out["ratios"] = frame.assign(**{f"{a}_over_{b}": frame[a] / (frame[b].abs() + 1e-3) for a, b in pairs})
    z = {c: (frame[c] - stats[c][0]) / (stats[c][1] + 1e-9) for c in pick}
    out["interactions"] = frame.assign(**{f"{a}_x_{b}": z[a] * z[b] for a, b in pairs})
    return out

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
aucs, n_features = {}, {}
for tr, te in folds:
    stats = {c: (X.iloc[tr][c].mean(), X.iloc[tr][c].std()) for c in pick}  # training fold only
    train_sets, test_sets = recipes(X.iloc[tr], stats), recipes(X.iloc[te], stats)
    for name in train_sets:
        n_features[name] = train_sets[name].shape[1]
        m = MODELS["logistic_regression"]().fit(train_sets[name], y.iloc[tr])
        aucs.setdefault(name, []).append(roc_auc_score(y.iloc[te], m.predict_proba(test_sets[name])[:, 1]))

raw = np.array(aucs["raw"])
report = {}
for name, a in aucs.items():
    a = np.array(a)
    report[name] = {"features": n_features[name],
                    "mean_roc_auc": round(float(a.mean()), 6), "std": round(float(a.std()), 6),
                    "mean_paired_gain": None if name == "raw" else round(float((a - raw).mean()), 6),
                    "folds_improved": None if name == "raw" else int(((a - raw) > 0).sum())}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
EXECUTION OUTPUT:
```json
{
 "raw": {
  "features": 14,
  "mean_roc_auc": 0.849617,
  "std": 0.019785,
  "mean_paired_gain": null,
  "folds_improved": null
 },
 "log_transforms": {
  "features": 19,
  "mean_roc_auc": 0.849604,
  "std": 0.019389,
  "mean_paired_gain": -1.2e-05,
  "folds_improved": 2
 },
 "ratios": {
  "features": 17,
  "mean_roc_auc": 0.858355,
  "std": 0.01937,
  "mean_paired_gain": 0.008739,
  "folds_improved": 5
 },
 "interactions": {
  "features": 17,
  "mean_roc_auc": 0.848105,
  "std": 0.020956,
  "mean_paired_gain": -0.001512,
  "folds_improved": 1
 }
}
```

Now decide.
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
```

**Answer**

```
DECISION: adopt_ratios
WHY: `ratios` improves on raw by +0.0087 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 3 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw + ratios feature set into model selection and log each new feature's formula.
```

### `calibrate_isotonic`  (spambase, code_calibration)

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Calibration check for `random_forest` on `data/public/spambase` (target: whether an email is spam; stratified sample of 3,000 rows, seed 1; drop the contract's blocked columns: none): out-of-fold probabilities on 3 stratified folds, raw and with sigmoid and isotonic calibration fitted inside each training fold. Report ECE (10 bins), ROC-AUC and Brier for each, and the training rows per fold.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

SEED = 1
MODELS = {
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
}


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


X = pd.read_parquet("data/public/spambase/X.parquet")
y = pd.read_parquet("data/public/spambase/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=1)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
preds = {k: np.zeros(len(y)) for k in ("raw", "sigmoid", "isotonic")}
for tr, te in folds:
    preds["raw"][te] = MODELS["random_forest"]().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    for method in ("sigmoid", "isotonic"):  # calibration learns only from the training fold
        cal = CalibratedClassifierCV(MODELS["random_forest"](), method=method, cv=3).fit(X.iloc[tr], y.iloc[tr])
        preds[method][te] = cal.predict_proba(X.iloc[te])[:, 1]

report = {"training_rows_per_fold": int(len(y) * 2 / 3)}
for k, p in preds.items():
    report[k] = {"ece": round(ece(y, p), 6), "roc_auc": round(float(roc_auc_score(y, p)), 6),
                 "brier": round(float(np.mean((p - y.to_numpy()) ** 2)), 6)}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
EXECUTION OUTPUT:
```json
{
 "training_rows_per_fold": 2000,
 "raw": {
  "ece": 0.053784,
  "roc_auc": 0.980984,
  "brier": 0.049107
 },
 "sigmoid": {
  "ece": 0.016433,
  "roc_auc": 0.979271,
  "brier": 0.045885
 },
 "isotonic": {
  "ece": 0.007518,
  "roc_auc": 0.979094,
  "brier": 0.045732
 }
}
```

Now decide.
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
```

**Answer**

```
DECISION: calibrate_isotonic
WHY: Raw `random_forest` probabilities are off by an ECE of 0.0538, above the 0.03 bar. Isotonic calibration fitted inside the training folds brings ECE to 0.0075 with ROC-AUC 0.9791 vs 0.9810 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Add isotonic calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `calibrate_sigmoid`  (adult, code_calibration)

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Calibration check for `hist_gradient_boosting` on `data/public/adult` (target: whether a person earns more than $50K per year; stratified sample of 3,000 rows, seed 2; drop the contract's blocked columns: none): out-of-fold probabilities on 3 stratified folds, raw and with sigmoid and isotonic calibration fitted inside each training fold. Report ECE (10 bins), ROC-AUC and Brier for each, and the training rows per fold.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 2
MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),
}


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


X = pd.read_parquet("data/public/adult/X.parquet")
y = pd.read_parquet("data/public/adult/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=2)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
preds = {k: np.zeros(len(y)) for k in ("raw", "sigmoid", "isotonic")}
for tr, te in folds:
    preds["raw"][te] = MODELS["hist_gradient_boosting"]().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    for method in ("sigmoid", "isotonic"):  # calibration learns only from the training fold
        cal = CalibratedClassifierCV(MODELS["hist_gradient_boosting"](), method=method, cv=3).fit(X.iloc[tr], y.iloc[tr])
        preds[method][te] = cal.predict_proba(X.iloc[te])[:, 1]

report = {"training_rows_per_fold": int(len(y) * 2 / 3)}
for k, p in preds.items():
    report[k] = {"ece": round(ece(y, p), 6), "roc_auc": round(float(roc_auc_score(y, p)), 6),
                 "brier": round(float(np.mean((p - y.to_numpy()) ** 2)), 6)}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
EXECUTION OUTPUT:
```json
{
 "training_rows_per_fold": 2000,
 "raw": {
  "ece": 0.058304,
  "roc_auc": 0.897771,
  "brier": 0.109184
 },
 "sigmoid": {
  "ece": 0.019586,
  "roc_auc": 0.900769,
  "brier": 0.103225
 },
 "isotonic": {
  "ece": 0.016913,
  "roc_auc": 0.898926,
  "brier": 0.103683
 }
}
```

Now decide.
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
```

**Answer**

```
DECISION: calibrate_sigmoid
WHY: Raw `hist_gradient_boosting` probabilities are off by an ECE of 0.0583, above the 0.03 bar. Sigmoid calibration fitted inside the training folds brings ECE to 0.0196 with ROC-AUC 0.9008 vs 0.8978 raw, so ranking is kept while probabilities become trustworthy. Sigmoid is preferred because isotonic is better by less than 0.005 ECE (0.0169 vs 0.0196), not enough to justify the more flexible method.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Add sigmoid calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `deduplicate_before_split`  (spambase, code_presplit)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Profile `data/public/spambase` (X.parquet, y.parquet; target: whether an email is spam) BEFORE any split: rows, features, rows per feature, positive rate and minority share, exact duplicate rows (features + target), the three highest missing rates, and near-constant columns (one value in more than 99.5% of rows).
```

**Answer**

```
```python
import json

import pandas as pd


X = pd.read_parquet("data/public/spambase/X.parquet")
y = pd.read_parquet("data/public/spambase/y.parquet").iloc[:, 0].astype(int)

missing = X.isna().mean().sort_values(ascending=False).head(3)
report = {
    "rows": len(X),
    "features": X.shape[1],
    "rows_per_feature": round(len(X) / X.shape[1], 1),
    "positive_rate": round(float(y.mean()), 4),
    "minority_share": round(float(min(y.mean(), 1 - y.mean())), 4),
    "duplicate_share": round(float(pd.concat([X, y.rename("__y")], axis=1).duplicated().mean()), 4),
    "top_missing": {c: round(float(v), 4) for c, v in missing.items()},
    "near_constant_columns": int(sum(X[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.995 for c in X.columns)),
}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-03 Split design
EXECUTION OUTPUT:
```json
{
 "rows": 4601,
 "features": 57,
 "rows_per_feature": 80.7,
 "positive_rate": 0.394,
 "minority_share": 0.394,
 "duplicate_share": 0.085,
 "top_missing": {
  "word_freq_make": 0.0,
  "word_freq_address": 0.0,
  "word_freq_all": 0.0
 },
 "near_constant_columns": 0
}
```

Now decide.
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature. If several thresholds are crossed, act on the first in this order: deduplicate_before_split, small_data_protocol, use_imbalance_aware_evaluation, impute_within_folds.

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
```

**Answer**

```
DECISION: deduplicate_before_split
WHY: 8.5% of rows are exact duplicates, above the 1% threshold. If they are split randomly, copies of the same row land in both training and test folds, so the test score partly measures memory. Duplicates must be removed, or grouped so all copies stay in one fold, before any split.
RULE: DCLAB-R04 (duplicate/entity leakage) and DCLAB-R02 (split by the independence unit)
CONFIDENCE: high
NEXT: [continue → WF-04] Deduplicate (or define groups) before the split, re-profile, then profile the training partition.
```

### `hist_gradient_boosting`  (mushroom, code_screen)

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Algorithm screen for `data/public/mushroom` (target: whether a mushroom is poisonous; stratified sample of 3,000 rows, seed 2; drop the contract's blocked columns: none): logistic_regression, random_forest and hist_gradient_boosting on the SAME 3 stratified folds. Report mean and std ROC-AUC, the stability-adjusted score (mean − 0.25 × std) and fit+predict seconds.
```

**Answer**

```
```python
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2
MODELS = {
    "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),
}

X = pd.read_parquet("data/public/mushroom/X.parquet")
y = pd.read_parquet("data/public/mushroom/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=2)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
report = {}
for name, make in MODELS.items():
    start, aucs = time.time(), []
    for tr, te in folds:
        m = make().fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    aucs = np.array(aucs)
    report[name] = {"mean_roc_auc": round(float(aucs.mean()), 6), "std": round(float(aucs.std()), 6),
                    "stability_adjusted": round(float(aucs.mean() - 0.25 * aucs.std()), 6),
                    "seconds": round(time.time() - start, 2)}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
EXECUTION OUTPUT:
```json
{
 "logistic_regression": {
  "mean_roc_auc": 0.999124,
  "std": 9.7e-05,
  "stability_adjusted": 0.9991,
  "seconds": 0.07
 },
 "random_forest": {
  "mean_roc_auc": 1.0,
  "std": 0.0,
  "stability_adjusted": 1.0,
  "seconds": 1.12
 },
 "hist_gradient_boosting": {
  "mean_roc_auc": 0.999928,
  "std": 0.000102,
  "stability_adjusted": 0.999902,
  "seconds": 0.54
 }
}
```

Now decide.
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

ALLOWED ACTIONS: logistic_regression | random_forest | hist_gradient_boosting
```

**Answer**

```
DECISION: hist_gradient_boosting
WHY: `hist_gradient_boosting` and `random_forest` are effectively tied on stability-adjusted score (0.9999 vs 1.0000, within 0.002), so the clearly faster one goes forward: `hist_gradient_boosting` takes 0.54s vs 1.12s. A score gap this small is noise, while a speed gap this large is real.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: medium
NEXT: [continue → WF-08] Carry `hist_gradient_boosting` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `investigate_distribution_shift`  (bank_marketing, code_holdout)

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Final holdout check for hist_gradient_boosting on `data/public/bank_marketing` (target: whether a client subscribes to a term deposit; stratified sample of 3,000 rows, seed 1; drop the contract's blocked columns: duration); the holdout is the top quarter of rows by `job` (a deliberately shifted holdout). Report 3-fold CV ROC-AUC on the training part (mean, std), the holdout ROC-AUC scored once, the difference CV minus holdout and its absolute value, and the three largest feature shifts (PSI) between training and holdout.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 1
MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),
}


def psi(a, b, bins=10):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if a.nunique() < 2 or len(b) < 20:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ha = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    hb = np.clip(np.histogram(b, edges)[0] / len(b), 1e-6, None)
    return float(np.sum((ha - hb) * np.log(ha / hb)))


X = pd.read_parquet("data/public/bank_marketing/X.parquet")
y = pd.read_parquet("data/public/bank_marketing/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=1)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
X = X.drop(columns=['duration'])  # the dataset contract blocks these

holdout = (X["job"].rank(method="first", pct=True) > 0.75).to_numpy()
Xtr, ytr = X[~holdout].reset_index(drop=True), y[~holdout].reset_index(drop=True)
Xho, yho = X[holdout], y[holdout]

cv = []
for tr, te in StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(Xtr, ytr):
    m = MODELS["hist_gradient_boosting"]().fit(Xtr.iloc[tr], ytr.iloc[tr])
    cv.append(roc_auc_score(ytr.iloc[te], m.predict_proba(Xtr.iloc[te])[:, 1]))
cv = np.array(cv)
final = MODELS["hist_gradient_boosting"]().fit(Xtr, ytr)  # locked: every choice is made before this line
holdout_auc = roc_auc_score(yho, final.predict_proba(Xho)[:, 1])  # scored once
shifts = sorted(((c, psi(Xtr[c], Xho[c])) for c in X.columns), key=lambda t: -t[1])[:3]
print(json.dumps({
    "training_rows": len(ytr), "holdout_rows": int(len(yho)),
    "cv_roc_auc_mean": round(float(cv.mean()), 6), "cv_roc_auc_std": round(float(cv.std()), 6),
    "holdout_roc_auc": round(float(holdout_auc), 6),
    "cv_minus_holdout": round(float(cv.mean() - holdout_auc), 6),
    "abs_difference": round(float(abs(cv.mean() - holdout_auc)), 6),
    "largest_psi": {c: round(p, 4) for c, p in shifts},
}, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
EXECUTION OUTPUT:
```json
{
 "training_rows": 2250,
 "holdout_rows": 750,
 "cv_roc_auc_mean": 0.74388,
 "cv_roc_auc_std": 0.013609,
 "holdout_roc_auc": 0.663351,
 "cv_minus_holdout": 0.080529,
 "abs_difference": 0.080529,
 "largest_psi": {
  "job": 12.3373,
  "education": 0.1742,
  "age": 0.093
 }
}
```

Now decide.
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
```

**Answer**

```
DECISION: investigate_distribution_shift
WHY: The holdout ROC-AUC (0.6634) is 0.0805 below CV (0.7439), outside the ±0.02 band, and `job` has a PSI of 12.337 (above 0.2): the holdout population is different from training. The CV number does not describe this population, so the shift must be explained before any claim is made.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [back → WF-03] Do not tune on the holdout. Slice results by the shifted feature, document the gap, and redesign the split so training and test reflect that shift (by time or by the shifted group); the spent holdout is reported, never reused.
```

### `investigate_gap_without_shift`  (bank_marketing, code_holdout)

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
STAGE: optimization_reliability (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Final holdout check for hist_gradient_boosting on `data/public/bank_marketing` (target: whether a client subscribes to a term deposit; stratified sample of 3,000 rows, seed 4; drop the contract's blocked columns: duration); the holdout is a random quarter of the rows. Report 3-fold CV ROC-AUC on the training part (mean, std), the holdout ROC-AUC scored once, the difference CV minus holdout and its absolute value, and the three largest feature shifts (PSI) between training and holdout.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 4
MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),
}


def psi(a, b, bins=10):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if a.nunique() < 2 or len(b) < 20:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ha = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    hb = np.clip(np.histogram(b, edges)[0] / len(b), 1e-6, None)
    return float(np.sum((ha - hb) * np.log(ha / hb)))


X = pd.read_parquet("data/public/bank_marketing/X.parquet")
y = pd.read_parquet("data/public/bank_marketing/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=4)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
X = X.drop(columns=['duration'])  # the dataset contract blocks these

holdout = np.random.default_rng(SEED).random(len(X)) < 0.25
Xtr, ytr = X[~holdout].reset_index(drop=True), y[~holdout].reset_index(drop=True)
Xho, yho = X[holdout], y[holdout]

cv = []
for tr, te in StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(Xtr, ytr):
    m = MODELS["hist_gradient_boosting"]().fit(Xtr.iloc[tr], ytr.iloc[tr])
    cv.append(roc_auc_score(ytr.iloc[te], m.predict_proba(Xtr.iloc[te])[:, 1]))
cv = np.array(cv)
final = MODELS["hist_gradient_boosting"]().fit(Xtr, ytr)  # locked: every choice is made before this line
holdout_auc = roc_auc_score(yho, final.predict_proba(Xho)[:, 1])  # scored once
shifts = sorted(((c, psi(Xtr[c], Xho[c])) for c in X.columns), key=lambda t: -t[1])[:3]
print(json.dumps({
    "training_rows": len(ytr), "holdout_rows": int(len(yho)),
    "cv_roc_auc_mean": round(float(cv.mean()), 6), "cv_roc_auc_std": round(float(cv.std()), 6),
    "holdout_roc_auc": round(float(holdout_auc), 6),
    "cv_minus_holdout": round(float(cv.mean() - holdout_auc), 6),
    "abs_difference": round(float(abs(cv.mean() - holdout_auc)), 6),
    "largest_psi": {c: round(p, 4) for c, p in shifts},
}, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-09 Reliability challenge
EXECUTION OUTPUT:
```json
{
 "training_rows": 2264,
 "holdout_rows": 736,
 "cv_roc_auc_mean": 0.726268,
 "cv_roc_auc_std": 0.026335,
 "holdout_roc_auc": 0.749984,
 "cv_minus_holdout": -0.023716,
 "abs_difference": 0.023716,
 "largest_psi": {
  "day_of_week": 0.0207,
  "age": 0.0206,
  "contact": 0.0101
 }
}
```

Now decide.
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
```

**Answer**

```
DECISION: investigate_gap_without_shift
WHY: CV (0.7263) and holdout (0.7500) differ by -0.0237, outside the ±0.02 band, yet no feature shifts much (highest PSI 0.021). The holdout scored higher than CV, which usually means an easier or luckier sample of 736 rows rather than a better model, so the higher number must not be reported as the expected performance.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: [continue → WF-10] Do not tune on the holdout. Put a bootstrap interval on the holdout score, review which earlier choices were made on CV, and report both numbers with the explanation.
```

### `keep_raw`  (bank_marketing, code_feature_ladder)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: feature_engineering (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Feature ladder for `data/public/bank_marketing` (target: whether a client subscribes to a term deposit; use a stratified sample of 3,000 rows, seed 2, and drop the columns the dataset's contract blocks: duration). Compare the raw columns with three recipes on the SAME 5 stratified folds and the same model (logistic_regression): log1p of non-negative skewed columns (|skew| > 1, more than 10 distinct values), pairwise ratios and pairwise products of standardized values of `month`, `day_of_week`, `previous`. For each recipe report the feature count, mean and std ROC-AUC, the mean paired gain over raw and how many folds improved.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2
MODELS = {
    "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),
}

X = pd.read_parquet("data/public/bank_marketing/X.parquet")
y = pd.read_parquet("data/public/bank_marketing/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=2)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
X = X.drop(columns=['duration'])  # the dataset contract blocks these

pick = [np.str_('month'), np.str_('day_of_week'), np.str_('previous')]
cont = [c for c in X.columns if X[c].nunique() > 10]
skewed = [c for c in cont if X[c].min() >= 0 and abs(X[c].skew()) > 1]
pairs = [(a, b) for i, a in enumerate(pick) for b in pick[i + 1:]]

def recipes(frame, stats):
    out = {"raw": frame}
    if skewed:
        out["log_transforms"] = frame.assign(**{f"log1p_{c}": np.log1p(frame[c]) for c in skewed})
    out["ratios"] = frame.assign(**{f"{a}_over_{b}": frame[a] / (frame[b].abs() + 1e-3) for a, b in pairs})
    z = {c: (frame[c] - stats[c][0]) / (stats[c][1] + 1e-9) for c in pick}
    out["interactions"] = frame.assign(**{f"{a}_x_{b}": z[a] * z[b] for a, b in pairs})
    return out

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
aucs, n_features = {}, {}
for tr, te in folds:
    stats = {c: (X.iloc[tr][c].mean(), X.iloc[tr][c].std()) for c in pick}  # training fold only
    train_sets, test_sets = recipes(X.iloc[tr], stats), recipes(X.iloc[te], stats)
    for name in train_sets:
        n_features[name] = train_sets[name].shape[1]
        m = MODELS["logistic_regression"]().fit(train_sets[name], y.iloc[tr])
        aucs.setdefault(name, []).append(roc_auc_score(y.iloc[te], m.predict_proba(test_sets[name])[:, 1]))

raw = np.array(aucs["raw"])
report = {}
for name, a in aucs.items():
    a = np.array(a)
    report[name] = {"features": n_features[name],
                    "mean_roc_auc": round(float(a.mean()), 6), "std": round(float(a.std()), 6),
                    "mean_paired_gain": None if name == "raw" else round(float((a - raw).mean()), 6),
                    "folds_improved": None if name == "raw" else int(((a - raw) > 0).sum())}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
EXECUTION OUTPUT:
```json
{
 "raw": {
  "features": 15,
  "mean_roc_auc": 0.725618,
  "std": 0.044821,
  "mean_paired_gain": null,
  "folds_improved": null
 },
 "log_transforms": {
  "features": 18,
  "mean_roc_auc": 0.725069,
  "std": 0.044816,
  "mean_paired_gain": -0.000549,
  "folds_improved": 3
 },
 "ratios": {
  "features": 18,
  "mean_roc_auc": 0.72532,
  "std": 0.044306,
  "mean_paired_gain": -0.000298,
  "folds_improved": 2
 },
 "interactions": {
  "features": 18,
  "mean_roc_auc": 0.725008,
  "std": 0.044221,
  "mean_paired_gain": -0.00061,
  "folds_improved": 2
 }
}
```

Now decide.
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. The closest, `ratios`, changes ROC-AUC by -0.0003 and wins on 2/5 folds, which is not a consistent gain of at least +0.005. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: [continue → WF-07] Carry the raw feature set into model selection.
```

### `keep_uncalibrated`  (bank_marketing, code_calibration)

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
STAGE: optimization_reliability (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Calibration check for `random_forest` on `data/public/bank_marketing` (target: whether a client subscribes to a term deposit; stratified sample of 3,000 rows, seed 1; drop the contract's blocked columns: duration): out-of-fold probabilities on 3 stratified folds, raw and with sigmoid and isotonic calibration fitted inside each training fold. Report ECE (10 bins), ROC-AUC and Brier for each, and the training rows per fold.
```

**Answer**

```
```python
import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

SEED = 1
MODELS = {
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
}


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


X = pd.read_parquet("data/public/bank_marketing/X.parquet")
y = pd.read_parquet("data/public/bank_marketing/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=1)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
X = X.drop(columns=['duration'])  # the dataset contract blocks these

folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
preds = {k: np.zeros(len(y)) for k in ("raw", "sigmoid", "isotonic")}
for tr, te in folds:
    preds["raw"][te] = MODELS["random_forest"]().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    for method in ("sigmoid", "isotonic"):  # calibration learns only from the training fold
        cal = CalibratedClassifierCV(MODELS["random_forest"](), method=method, cv=3).fit(X.iloc[tr], y.iloc[tr])
        preds[method][te] = cal.predict_proba(X.iloc[te])[:, 1]

report = {"training_rows_per_fold": int(len(y) * 2 / 3)}
for k, p in preds.items():
    report[k] = {"ece": round(ece(y, p), 6), "roc_auc": round(float(roc_auc_score(y, p)), 6),
                 "brier": round(float(np.mean((p - y.to_numpy()) ** 2)), 6)}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-08 Conservative optimization
EXECUTION OUTPUT:
```json
{
 "training_rows_per_fold": 2000,
 "raw": {
  "ece": 0.014004,
  "roc_auc": 0.757582,
  "brier": 0.090163
 },
 "sigmoid": {
  "ece": 0.014345,
  "roc_auc": 0.747311,
  "brier": 0.090398
 },
 "isotonic": {
  "ece": 0.01045,
  "roc_auc": 0.748542,
  "brier": 0.090161
 }
}
```

Now decide.
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
```

**Answer**

```
DECISION: keep_uncalibrated
WHY: Raw `random_forest` probabilities are already usable or calibration does not pay: raw ECE is 0.0140 (at or below the 0.03 bar). Adding a calibration layer would add complexity for no measured benefit.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: [continue → WF-09] Lock the uncalibrated model and move to the final holdout check.
```

### `logistic_regression`  (breast_cancer, code_screen)

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Algorithm screen for `data/public/breast_cancer` (target: whether a tumour is malignant; stratified sample of 3,000 rows, seed 1; drop the contract's blocked columns: none): logistic_regression, random_forest and hist_gradient_boosting on the SAME 3 stratified folds. Report mean and std ROC-AUC, the stability-adjusted score (mean − 0.25 × std) and fit+predict seconds.
```

**Answer**

```
```python
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1
MODELS = {
    "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),
}

X = pd.read_parquet("data/public/breast_cancer/X.parquet")
y = pd.read_parquet("data/public/breast_cancer/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=1)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
report = {}
for name, make in MODELS.items():
    start, aucs = time.time(), []
    for tr, te in folds:
        m = make().fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    aucs = np.array(aucs)
    report[name] = {"mean_roc_auc": round(float(aucs.mean()), 6), "std": round(float(aucs.std()), 6),
                    "stability_adjusted": round(float(aucs.mean() - 0.25 * aucs.std()), 6),
                    "seconds": round(time.time() - start, 2)}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
EXECUTION OUTPUT:
```json
{
 "logistic_regression": {
  "mean_roc_auc": 0.994009,
  "std": 0.003536,
  "stability_adjusted": 0.993125,
  "seconds": 0.05
 },
 "random_forest": {
  "mean_roc_auc": 0.98999,
  "std": 0.006364,
  "stability_adjusted": 0.988399,
  "seconds": 1.12
 },
 "hist_gradient_boosting": {
  "mean_roc_auc": 0.989417,
  "std": 0.006687,
  "stability_adjusted": 0.987745,
  "seconds": 0.7
 }
}
```

Now decide.
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

ALLOWED ACTIONS: logistic_regression | random_forest | hist_gradient_boosting
```

**Answer**

```
DECISION: logistic_regression
WHY: `logistic_regression` has the highest stability-adjusted score (0.9931: mean 0.9940, std 0.0035), ahead of `random_forest` at 0.9884. The linear baseline wins here, which is exactly why it is always screened: no family is best on every dataset.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: [continue → WF-08] Carry `logistic_regression` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `proceed`  (credit_default, code_presplit)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Profile `data/public/credit_default` (X.parquet, y.parquet; target: whether a client defaults on next month's credit-card payment) BEFORE any split: rows, features, rows per feature, positive rate and minority share, exact duplicate rows (features + target), the three highest missing rates, and near-constant columns (one value in more than 99.5% of rows).
```

**Answer**

```
```python
import json

import pandas as pd


X = pd.read_parquet("data/public/credit_default/X.parquet")
y = pd.read_parquet("data/public/credit_default/y.parquet").iloc[:, 0].astype(int)

missing = X.isna().mean().sort_values(ascending=False).head(3)
report = {
    "rows": len(X),
    "features": X.shape[1],
    "rows_per_feature": round(len(X) / X.shape[1], 1),
    "positive_rate": round(float(y.mean()), 4),
    "minority_share": round(float(min(y.mean(), 1 - y.mean())), 4),
    "duplicate_share": round(float(pd.concat([X, y.rename("__y")], axis=1).duplicated().mean()), 4),
    "top_missing": {c: round(float(v), 4) for c, v in missing.items()},
    "near_constant_columns": int(sum(X[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.995 for c in X.columns)),
}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-03 Split design
EXECUTION OUTPUT:
```json
{
 "rows": 30000,
 "features": 23,
 "rows_per_feature": 1304.3,
 "positive_rate": 0.2212,
 "minority_share": 0.2212,
 "duplicate_share": 0.0012,
 "top_missing": {
  "X1": 0.0,
  "X2": 0.0,
  "X3": 0.0
 },
 "near_constant_columns": 0
}
```

Now decide.
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature. If several thresholds are crossed, act on the first in this order: deduplicate_before_split, small_data_protocol, use_imbalance_aware_evaluation, impute_within_folds.

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
```

**Answer**

```
DECISION: proceed
WHY: No threshold is crossed: duplicates 0.1%, minority class 22.1%, highest missing rate 0.0%, 30,000 rows (1304.3 per feature). The data is ready for the split and the training-only analysis.
RULE: DCLAB-R01 (contract written, profile clean)
CONFIDENCE: high
NEXT: [continue → WF-04] Write the profile to the decision ledger, design the split, then profile the training partition.
```

### `random_forest`  (adult, code_screen)

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
STAGE: model_selection (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Algorithm screen for `data/public/adult` (target: whether a person earns more than $50K per year; stratified sample of 3,000 rows, seed 2; drop the contract's blocked columns: none): logistic_regression, random_forest and hist_gradient_boosting on the SAME 3 stratified folds. Report mean and std ROC-AUC, the stability-adjusted score (mean − 0.25 × std) and fit+predict seconds.
```

**Answer**

```
```python
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2
MODELS = {
    "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, random_state=SEED),
}

X = pd.read_parquet("data/public/adult/X.parquet")
y = pd.read_parquet("data/public/adult/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=2)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y))
report = {}
for name, make in MODELS.items():
    start, aucs = time.time(), []
    for tr, te in folds:
        m = make().fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    aucs = np.array(aucs)
    report[name] = {"mean_roc_auc": round(float(aucs.mean()), 6), "std": round(float(aucs.std()), 6),
                    "stability_adjusted": round(float(aucs.mean() - 0.25 * aucs.std()), 6),
                    "seconds": round(time.time() - start, 2)}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-07 Algorithm screen
EXECUTION OUTPUT:
```json
{
 "logistic_regression": {
  "mean_roc_auc": 0.850011,
  "std": 0.007934,
  "stability_adjusted": 0.848028,
  "seconds": 0.04
 },
 "random_forest": {
  "mean_roc_auc": 0.901497,
  "std": 0.008219,
  "stability_adjusted": 0.899442,
  "seconds": 1.16
 },
 "hist_gradient_boosting": {
  "mean_roc_auc": 0.897955,
  "std": 0.005365,
  "stability_adjusted": 0.896613,
  "seconds": 0.76
 }
}
```

Now decide.
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

ALLOWED ACTIONS: logistic_regression | random_forest | hist_gradient_boosting
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` has the highest stability-adjusted score (0.8994: mean 0.9015, std 0.0082), ahead of `hist_gradient_boosting` at 0.8966. The linear baseline scores 0.8480, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: [continue → WF-08] Carry `random_forest` into optimization; keep `hist_gradient_boosting` as the paired comparison when tuning.
```

### `small_data_protocol`  (breast_cancer, code_presplit)

**Input**

```
WORKFLOW STEP: WF-03 Split design
STAGE: data_understanding (code)
Write the Python (pandas, scikit-learn) that computes the evidence this step needs, on the files named below, leakage-safe: everything that learns from data is fitted inside training folds only. Print the result as JSON. The code will be run and its output returned to you.

TASK: Profile `data/public/breast_cancer` (X.parquet, y.parquet; target: whether a tumour is malignant) BEFORE any split: rows, features, rows per feature, positive rate and minority share, exact duplicate rows (features + target), the three highest missing rates, and near-constant columns (one value in more than 99.5% of rows).
```

**Answer**

```
```python
import json

import pandas as pd


X = pd.read_parquet("data/public/breast_cancer/X.parquet")
y = pd.read_parquet("data/public/breast_cancer/y.parquet").iloc[:, 0].astype(int)

missing = X.isna().mean().sort_values(ascending=False).head(3)
report = {
    "rows": len(X),
    "features": X.shape[1],
    "rows_per_feature": round(len(X) / X.shape[1], 1),
    "positive_rate": round(float(y.mean()), 4),
    "minority_share": round(float(min(y.mean(), 1 - y.mean())), 4),
    "duplicate_share": round(float(pd.concat([X, y.rename("__y")], axis=1).duplicated().mean()), 4),
    "top_missing": {c: round(float(v), 4) for c, v in missing.items()},
    "near_constant_columns": int(sum(X[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.995 for c in X.columns)),
}
print(json.dumps(report, indent=1))
```
```

**Input**

```
WORKFLOW STEP: WF-03 Split design
EXECUTION OUTPUT:
```json
{
 "rows": 569,
 "features": 30,
 "rows_per_feature": 19.0,
 "positive_rate": 0.3726,
 "minority_share": 0.3726,
 "duplicate_share": 0.0,
 "top_missing": {
  "radius1": 0.0,
  "texture1": 0.0,
  "perimeter1": 0.0
 },
 "near_constant_columns": 0
}
```

Now decide.
Data understanding, before the split: profile the whole table and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature. If several thresholds are crossed, act on the first in this order: deduplicate_before_split, small_data_protocol, use_imbalance_aware_evaluation, impute_within_folds.

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
```

**Answer**

```
DECISION: small_data_protocol
WHY: There are only 569 rows (19.0 per feature). Single splits on data this small swing widely, so use repeated stratified CV, start with simple regularized models, and report wide uncertainty instead of single scores.
RULE: DCLAB-R16 (paired, stable comparisons) and DCLAB-R13 (start from a transparent baseline)
CONFIDENCE: high
NEXT: [continue → WF-04] Use repeated stratified CV with a linear baseline as the split design, then profile the training partition.
```

## code_review

### `fix_leak`  (adult, review_oversample_before_cv)

**Input**

```
WORKFLOW STEP: WF-06 Feature ladder
STAGE: code_review
Code review: read the snippet and decide whether it leaks information across the train/test boundary or tunes on data it later reports on. fix_leak = it does; give the fixed code. no_issue = it does not.

DATA: `data/public/adult` (target: whether a person earns more than $50K per year)

CODE:
```python
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X = pd.read_parquet("data/public/adult/X.parquet")
y = pd.read_parquet("data/public/adult/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=1)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
rng = np.random.default_rng(1)
minority = int(y.value_counts().idxmin())
extra = rng.choice(np.flatnonzero(y.to_numpy() == minority), size=int((y != minority).sum() - (y == minority).sum()))
X_bal = pd.concat([X, X.iloc[extra]], ignore_index=True)
y_bal = pd.concat([y, y.iloc[extra]], ignore_index=True)
model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(200, random_state=1, n_jobs=-1))
scores = cross_val_score(model, X_bal, y_bal, cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc")
print(json.dumps({"cv_roc_auc": round(scores.mean(), 4)}))
```

ITS OUTPUT:
```json
{"cv_roc_auc": 0.985}
```

ALLOWED ACTIONS: fix_leak | no_issue
```

**Answer**

```
DECISION: fix_leak
WHY: Minority rows are copied before the folds are made, so copies of the same row land in both training and test folds and the model is tested on rows it memorised. The test folds also no longer have the real class balance, so the score describes neither the model nor production.
FIX:
```python
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X = pd.read_parquet("data/public/adult/X.parquet")
y = pd.read_parquet("data/public/adult/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=1)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
rng = np.random.default_rng(1)
minority = int(y.value_counts().idxmin())
scores = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
    ytr = y.iloc[tr]
    pool = tr[ytr.to_numpy() == minority]
    extra = rng.choice(pool, size=int((ytr != minority).sum() - (ytr == minority).sum()))
    idx = np.concatenate([tr, extra])  # copies come from the training fold only
    model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(200, random_state=1, n_jobs=-1))
    model.fit(X.iloc[idx], y.iloc[idx])
    scores.append(roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
print(json.dumps({"cv_roc_auc": round(float(np.mean(scores)), 4)}))
```
RULE: DCLAB-R03 (resampling inside training folds only) and DCLAB-R18 (evaluate on the real class balance)
CONFIDENCE: high
NEXT: [continue → WF-06] Oversample only the training rows inside each fold, keep every test fold untouched, and re-run.
```

### `no_issue`  (mushroom, review_target_encoding_all_rows)

**Input**

```
WORKFLOW STEP: WF-05 Leakage audit
STAGE: code_review
Code review: read the snippet and decide whether it leaks information across the train/test boundary or tunes on data it later reports on. fix_leak = it does; give the fixed code. no_issue = it does not.

DATA: `data/public/mushroom` (target: whether a mushroom is poisonous)

CODE:
```python
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import make_column_transformer

X = pd.read_parquet("data/public/mushroom/X.parquet")
y = pd.read_parquet("data/public/mushroom/y.parquet").iloc[:, 0].astype(int)
if len(X) > 3000:  # a stratified sample keeps the class balance
    keep = y.groupby(y, group_keys=False).apply(lambda s: s.sample(frac=3000 / len(X), random_state=2)).index
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
cat = X.columns[X.nunique().argmax()]  # the column with the most distinct values, treated as a category
X[cat] = X[cat].astype(str)
others = [c for c in X.columns if c != cat]
prep = make_column_transformer((TargetEncoder(random_state=2), [cat]),
                               (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), others))
model = make_pipeline(prep, LogisticRegression(max_iter=3000))
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=2), scoring="roc_auc")
print(json.dumps({"cv_roc_auc": round(scores.mean(), 4)}))
```

ITS OUTPUT:
```json
{"cv_roc_auc": 0.9992}
```

ALLOWED ACTIONS: fix_leak | no_issue
```

**Answer**

```
DECISION: no_issue
WHY: Everything that learns from the data is fitted on training rows only, and the score is computed on rows that took no part in any choice, so the number is an honest estimate.
RULE: DCLAB-R04 (cross-split aggregate leakage) and DCLAB-R03 (encoders fitted inside training folds)
CONFIDENCE: high
NEXT: [continue → WF-05] Use the result as it is.
```

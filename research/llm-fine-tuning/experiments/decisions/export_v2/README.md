---
pretty_name: DCLab decisions v2 (every point with a set)
license: other
language:
- en
task_categories:
- text-generation
tags:
- decision-making
- data-science
- synthetic
configs:
- config_name: default
  data_files:
  - split: train
    path: train.chat.jsonl
  - split: validation
    path: val.chat.jsonl
---

# DCLab decisions v2 (every point with a set)

Decisions DCLab makes while building a model, each with its label: benchmark labels of the invented tables, image sets and written journeys.

- 33666 decisions from 181 groups (a group, one table or project, never spans two splits)
- by point: column.category_code 3620, column.group 6154, column.is_identifier 6154, column.semantic_role 5430, column.time 6154, feature.leakage_suspect 6154
- by split: dev 6882, train 26784
- labels by source: benchmark 33666

Each row's user turn holds the question, what is known about the decision's subject (a column's bands and name tokens, a table's, an image set's or a conversation's counts, flags and codes; never a value of the table) and the options; the assistant's turn the answer as JSON. The sealed partition is not here: it is read once, to compare a tuned model with the rules and the classical classifiers on the same gate.

## Limits

- Invented tables (22.1, 29.2), invented image sets (26.8) and written journeys (26.3): evidence about these kinds of decisions, not about real tables.
- Labels are benchmark labels, set when each table was invented or each journey written; a workflow decision's journey may accept several answers (the comparison counts each as right).
- The sealed partitions are not included: each is read once, to compare a tuned model with the rules and the classifiers.
- The workflow points have no sealed set left: their own (journeys_workflow_sealed) was read by DWM-001 to 004, and the sealed journey sets their rows come from were each read by their gate before they were played here (a set not read yet is not).

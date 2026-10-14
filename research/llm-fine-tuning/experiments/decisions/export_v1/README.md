---
pretty_name: DCLab decisions v1 (generated tables)
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

# DCLab decisions v1 (generated tables)

Decisions DCLab makes while building a model, each with its label: benchmark labels of the generated tables of 22.1.

- 33666 decisions from 181 groups (a group, one table or project, never spans two splits)
- by point: column.category_code 3620, column.group 6154, column.is_identifier 6154, column.semantic_role 5430, column.time 6154, feature.leakage_suspect 6154
- by split: dev 6882, train 26784
- labels by source: benchmark 33666

Each row's user turn holds the question, what is known about the column as bands (never a value of the table) and the options; the assistant's turn the answer as JSON. The sealed partition is not here: it is read once, to compare a tuned model with the rules and the classical classifiers on the same gate.

## Limits

- Invented tables with planted columns (22.1): evidence about these kinds of columns, not about real tables.
- Labels are benchmark labels, set when each table was invented.
- The sealed partition is not included: it is read once, to compare a tuned model with the rules and the classifiers.

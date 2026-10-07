# User study protocol: with and without DCLab (package 14.6)

**Status: draft for the owner's approval.** Nothing is recorded until the owner approves this page and the tasks in
`evaluation/OWNER_INPUTS.md` (package 14.1) are written. No participant is invited by a session of the assistant.

## Question

Does a data scientist finish a real modelling task correctly, and spend less time on it, with DCLab than without it?
Three to five people cannot prove that. The study is a first look: it records what happens, in numbers, and says where the
assistant helped, was corrected, or raised a false alarm that cost time.

## Design

- **Who:** three to five data scientists who did not build DCLab and have not seen the benchmark. Each takes part once.
- **What each does:** two tasks from package 14.1 (real tasks with a success criterion and a time budget), one **with**
  DCLab and one **without** (their usual notebook and tools). Which task goes with which condition, and which comes first,
  alternates between participants (P1: T1 with, T2 without; P2: T2 with, T1 without; P3: T1 without first, and so on) so a
  harder task is not always on one side.
- **Data:** the sources of 14.1 (HyperAck, churn) or public tables. The participant works on their own machine or a shared
  one with the data the owner has approved for the study.
- **Time budget:** the budget of each task; a task not finished in time is recorded as not completed.

## What is recorded (and what is not)

The harness `python -m dclab_rnd.agent_eval.study` writes one line per event to `evaluation/study/events.jsonl`:

| Event | Fields |
|---|---|
| `start` | pseudonym (P1…), task (T1…), condition (`with` / `without`), time |
| `correction` | whole seconds the participant spent correcting an assistant suggestion (the "with" condition only) |
| `citation` | whether a cited evidence record was correct (yes / no) |
| `severe-false-alarm` | that a warning was wrong enough to cost real time or to block good work |
| `finish` | completed (yes / no); the time since `start` is computed |

There is **no free-text field**: no cell value, column name, name or comment can reach the file, and the code refuses one.
The observer judges "correct citation" and "severe false alarm" against the rule records and the task's own facts, and
notes nothing else. The participant's name is kept only by the owner, apart from the file, if at all.

## Analysis (fixed before any session)

`python -m dclab_rnd.agent_eval.study analyze --output evidence/campaigns/benchmark_v2/study/STU-001.json` reports, per
condition: tasks completed with a 95% Wilson interval, median time to complete, total correction time, correct citations
with an interval, and severe false alarms. **No significance test is run.** A difference is read only against the spread
between tasks and people, and a result is never described as the assistant being ready for production.

## Consent text (read aloud, then confirmed by the participant)

"We are testing a tool that helps check machine-learning notebooks for mistakes such as leakage. You will do two short
tasks, one with the tool and one without. We record only how long each task takes, whether you finished it, how long you
spent correcting the tool, and whether its citations and warnings were right. We do not record your screen, your
name or any data value. You can stop at any time and ask for your sessions to be deleted, and their records are removed from the file."

## Before the first session (the owner)

1. Approve this protocol or change it.
2. Write the two or three tasks, with the success criterion and the time budget, in `evaluation/OWNER_INPUTS.md`.
3. Choose three to five participants and say whether their names are kept.
4. Say which data the participants may use.

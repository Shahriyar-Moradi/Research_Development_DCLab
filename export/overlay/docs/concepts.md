# Concepts

The few ideas behind dclabai, with one running example: **a phone company wants to know which customers will leave
next month, so its team can call them first.**

## The prediction moment

The moment a prediction is made, and so the moment its inputs must exist. For the phone company: *the first day of
each month, before the calls start.*

Everything else follows from it. A column written *after* that moment is not allowed, however useful it looks. A
"cancellation reason" column predicts leaving perfectly, because it is only filled in once a customer has left. Using
it would be **leakage**: the model learns the answer from the future and fails in real use.

## The problem card and the data card

- The **problem card** says what is predicted, for whom, at which moment, and what a mistake costs ("a call costs 5;
  a customer lost costs 300"). Home builds it from your words and a few questions.
- The **data card** describes your tables without holding their rows: each column's role (an identifier, a time, a
  possible outcome, a leak suspect), its type and its missing values.

Before anything is built, the two are checked against each other. If the problem asks about "next month" but the table
has no dates, the studio asks instead of guessing.

## The solution brief: a kind plus modifiers

Every problem is **one kind** (yes/no, a number, a rare event, a ranking, a forecast, matching records...) **plus zero
or more modifiers** (rare positives, groups of rows from the same customer, time order, noisy labels, a team that can
review only so many cases a day).

The kind and the modifiers together give the **recipe**: how to split the data, which score to report, which model
families to try, which leakage checks to run. For the phone company: a yes/no kind; "rare positives" (few customers
leave) makes the main score *average precision* instead of accuracy; "time order" makes the test set the latest month.

You approve the brief before anything runs. See [adding-capability.md](adding-capability.md) to add a kind or a modifier.

## Five stages, and the holdout read once

The engine runs five stages: **data**, **leakage audit**, **features**, **models**, **final**.

- A part of the data, the **holdout**, is set aside at the start and never looked at during the first four stages.
- All choices (features, model family, settings) are made on the rest, with cross-validation inside it.
- The **final** stage reads the holdout **once** and reports the score with its uncertainty (an interval).

Think of it as an exam: you study with practice questions, then sit the real exam once. Looking at the exam while
studying makes the grade meaningless.

## Rules decide, models advise

Splits, scores, thresholds and model choice are made by deterministic code: the same input gives the same result.
A language model, when you configure one, **reads and writes words**: it asks questions, explains results, proposes
changes. It never makes a choice inside a recipe, and every sentence it writes about your results is checked against
the numbers it cites.

## Agents, on your model or on their rules

The project chat has a small team: a problem analyst, a solution planner, a records reader, an evaluator, an
experimenter and a notebook coder. Each one acts through **tools** (show the models, propose a new version, write a
notebook cell).

- With **no model**, every agent follows its rules: you still get the tables, the plans and the runs.
- With **a model**, the agents use it only once it calls tools properly. `dclab doctor` tests this; a model that
  writes its tool calls as text instead is kept off, and the rules answer. You can override with
  `dclab models trust all`.

## Evidence

Some advice comes with a source: a **rule** ("fit every preprocessing step inside the training folds") or a measured
**pitfall** ("scaling before the split changed the score by at most 0.0001 here; still do it right"). The evidence
library lists them, and the agents cite them by id (for example `DCLAB-R04`).

## Your data stays here

Projects, chats and records live in your workspace folder (`~/.dclabai`, or `DCLAB_AGENT_HOME`). Nothing is sent
anywhere except the model requests you configure, and those carry summaries of your tables (with at most a few
example values per column), never whole rows. See
[privacy.md](privacy.md).

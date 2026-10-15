# Labelling DCLab's decisions, blind

Who labels: the owner first, reviewers later (the owner's choice, 2026-10-15). A blind label is the right answer to one
decision DCLab made, given by a person who does not see what DCLab answered. These labels are the only honest measure
of the rules and the decision models (22.4: four of the six classifiers wait for them), and the training labels the
decision sets need (`dclab_core/datasets.py`).

## Where

1. Start the product: `make dev` (restart it once after an update: it migrates the database).
2. Use DCLab as usual: upload tables, answer the questions, run the stages. Every decision is recorded in the ledger as
   it is made (`dclab_rnd/ledger/capture.py`); nothing extra is needed.
3. Open **Admin → Decisions**. The panel **Label a decision, blind** shows one decision at a time.

The owner and reviewers can label (`review` in `dclab_rnd/accounts/roles.py`); a reviewer's account is added on the
Admin page's Members tab when sign-in is on.

## What you see, and what you never see

You see the question (for example "Is the column an identifier (never a feature)?"), the column or table it is about,
and what was known when it was decided: bands such as the share of distinct values, missing values, the storage type
and the shape of the written values (`state`). You never see the answer DCLab gave, the value it used, or who decided
(`dclab_core.ledger.blind_view`).

## How to answer

- Answer what is right for this column in this problem, as an experienced data scientist would, from what the panel
  shows. Do not guess what DCLab answered.
- If what is shown is not enough to tell, press **Can't tell — next**. The decision stays unlabelled and counts as a
  gap. A gap is reported; a guess would quietly bias every number measured with it.
- One label per person per decision: a second answer is refused, because its weight would count twice.
- Think of the prediction moment for the leakage question ("Would the column be unknown at the prediction moment, or
  encode the target?"): a refund paid after a cancellation, a delivery's end time or a final fare are "yes".

## Why some decisions come more often

The draw is fixed and known before anyone labels (`dclab_core.ledger.blind_draw`): decisions where the answers
disagreed (a rule against a model or a person) are drawn with probability 0.6, those where a model abstained 0.3, the
others 0.1, and disagreements are shown first. Every label carries its weight (1 / its probability), so the estimates
correct for the over-sampling. The draw never depends on what you answered.

## How much is enough

About 30 labels per decision point give an interval narrow enough to compare a model with the rule (a Wilson interval
on 30 answers is about ±0.17 at the worst). Start with the points the classifiers answer: `column.is_identifier`,
`column.semantic_role`, `column.category_code`, `column.time`, `column.group` and `feature.leakage_suspect`. Then
`python -m dclab_rnd.ledger.shadow recorded` measures each classifier against your labels.

## What is never labelled blind

Points whose state would name the answer, or that are a person's own judgement: the pack, the report's sentences,
accepting a lesson, workflow approvals and stops, the agent's tool choice (`dclab_core.ledger.NOT_BLIND`).

# The roadmap: everything left, in one order

One page for every package still to build in DCLab R&D, in the order they will be built, grouped into milestones.
Each milestone says what the owner can do once it is done. The packages themselves are described in:

- `CAPTURE_AND_DECISION_DATA_PLAN.md` — Part 8 (capture everything, 19.4–19.13) and Part 11 (decision models of any
  kind, 22.0–22.9);
- `EXPERIENCE_AND_WORKFLOWS_PLAN.md` — Part 9 (from the problem to the pipeline, 20.2–20.10) and Part 10 (agents,
  orchestration and harness, 21.1–21.5);
- `PRODUCT_BUILD_PLAYBOOK.md` — the older packages still open (16.14, 16.15, 15.11).

Written 2026-10-13. Done before it: 20.0 (the guard), 19.1–19.3 (the record of every session), 20.1 (the problem
card). Every package is built the same way: a failing test, the change, the checks, an adversarial review and its
fixes, planted faults, its development record for the owner's SFT, a commit pushed.

**Progress (2026-10-14):** milestone A done (19.4–19.8); milestone B done (19.9–19.13); milestone C: 20.3, 21.3 and 20.4 done; 20.2 built, its gate failed twice (8 of 9 on two sealed sets) and waits for a sealed set written by someone else; milestone D done (22.0–22.3: two of six classifiers passed their sealed gate, semantic_role and leakage_suspect); milestone E next.

## The two goals it serves

1. **DCLab solves machine-learning problems the right way, end to end**: it understands the problem and the data,
   connects them, builds the solution workflow and the pipeline, with code for each step, starting from the owner's
   delivery-fraud challenge.
2. **Everything that happens becomes training data** for the owner's models, organized and findable: the session
   record, the decision ledger, and decision models of any kind (open-source fine-tuned such as Laya or Qwen,
   classical, hosted) in two uses: **for data scientists** (technical decisions asked while they work) and **inside
   DCLab's workflow** (the decisions that steer solving a problem).

## Milestones

| | Milestone | Packages | What the owner can do after it | Hours |
|---|---|---|---|---|
| **A** | **Every session and decision on the record** | 19.4 Sessions page and finder · 19.5 run lineage · 19.6 decision record contract · 19.7 every decision captured (technical and workflow) · 19.8 labels (planted truth, measurements, people, blind review) | open any session on a page and read it; see every decision DCLab made, from what, by whom, and whether it was right; trace any number back to its run | 16–19 |
| **B** | **Collected the correct way, ready to train on** | 19.9 quality signals and failure detectors · 19.10 consent, redaction, retention, deletion · 19.11 data cards and synthetic tables · 19.12 training and evaluation sets · 19.13 archive and catalog | switch training use on per workspace and project; `make datasets` builds the SFT and decision datasets the consent allows, each with its manifest; one catalog of every set | 14–17 |
| **C** | **DCLab understands the problem and the data together** | 20.3 data card · 21.3 files understood together (across drafts) · 20.4 the problem meets the data · 20.2 the understanding conversation | upload the challenge's two files anywhere: they are recognised and joined; a mismatch between the problem and the table becomes a question, not a wrong model | 13–15 |
| **D** | **First decision models, on generated data** | 22.0 one interface for every decision model · 22.1 generated benchmarks at scale · 22.2 classical classifiers · 22.3 gates fixed before the runs | thousands of labelled decisions per point from invented tables; a classifier per technical point with its accuracy beside the rule's | 12–15 |
| **E** | **Agents managed and coordinated** | 21.1 one harness for every agent · 21.2 the coordinator · 21.4 model management per role | one coordinator walks problem → files → analysis → workflow through gates, never builds while something blocks; each role's model, prompt and cost on one page | 12–14 |
| **F** | **The challenge solved** | 20.5 analysis by family · 20.6 the solution workflow · 20.7 anomaly detection without labels (with 16.15 C: projects of several tables) | the delivery-fraud challenge, labelled or not, from the first sentence to a ranked review list with honest numbers | 15–18 |
| **G** | **Pipeline and code** | 20.8 the pipeline of workflows · 20.9 code for each step | every project as a pipeline with lineage, and its code (notebook and package) that reproduces its numbers | 10–12 |
| **H** | **Decision models in use, both uses** | 22.4 shadow in the studio · 22.5 open-source models: export, fine-tune, compare · 22.6 a model per decision · 22.8 for data scientists · 22.9 inside the workflow | ask "is this a leak?" in the notebook and get a model's answer with its evidence; the coordinator uses a decision model where it beat the rules; a fine-tuned open-source model compared on the same gates | 18–22 |
| **I** | **Proof and adoption** | 20.10 the acceptance suite of conversations · 21.5 evaluation from real sessions · 22.7 adoption by the main product · 16.14 open halves (results and pitfalls by kind) · 16.15 B (segmentation, uplift, ranking in the engine) · 15.11 the real challenge data on the owner's machine | every family solved from a written conversation, measured; recorded sessions as regression tests; the main product's mapping tested | 16–20 |

**Total: about 126–152 working hours.** In calendar time, about 13 to 16 days of continuous work, and realistically
4 to 5 weeks with the owner's answers, real-use tests, reviews and machine time. Live model use: only in 20.2, 21.4,
22.5 and 20.10, each run capped at €0.5 under the owner's rule; fine-tuning runs (22.5) need a GPU.

## Order and why

A before everything: the owner's first priority, and every later package's tests and training data come through it.
B right after A: the consent switches must exist before any dataset leaves the workspace. C next: the challenge's
files and the morning's failure need it, and it feeds D's decision points. D early: generated data needs no consent
and gives the first decision models while the workflow is built. E before F: the coordinator walks the workflow F
builds. G after F, H after D and E (models need the ledger, the gates and the coordinator), I last.

```text
A ──► B ──► (training data can leave the workspace)
│
├──► C ──► E ──► F ──► G
│     └──────────┐
└──► D ─────────► H ──► I
```

## Decisions waiting on the owner

| Decision | Needed by | Proposal |
|---|---|---|
| Retention per class | B (19.10) | events 24 months, model text 12 months (built and off until set: DCLAB_RETENTION_EVENT_DAYS=730, DCLAB_RETENTION_MODEL_TEXT_DAYS=365) |
| Who reads session timelines and decisions | A (19.4) | owner and admin roles |
| Who labels blind | A (19.8) | the owner first, reviewers later |
| A GPU for fine-tuning, and which open-source models first | H (22.5) | one small model (Qwen 2.5 1.5B or Phi-3 mini) beside the one the owner names (e.g. Laya) |
| The main product's ADR 0008 changed to "only with consent" | I (22.7) | in that repository |
| Pretrained image weights, or a download approval | 16.12 (later) | the owner's choice |
| Rotate the OpenAI key | now | — |

## Status

Each milestone's status is kept here as it moves: **A — done** (19.4–19.8, 2026-10-14). **B — done** (19.9–19.13). **C — done** (20.3, 21.3, 20.4; 20.2 built, its gate failed twice and waits for an independent sealed set). **D — done** (22.0–22.3). **E — next.**

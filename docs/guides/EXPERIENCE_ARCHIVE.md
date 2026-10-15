# The experience archive: where everything a workspace keeps lives

Package 19.13. One layout for everything DCLab records about the work done in a workspace, and one catalog of the
datasets built from it. A workspace is one folder (`DCLAB_AGENT_HOME`, by default `agent_runs/`, never committed),
plus its rows in PostgreSQL when `DCLAB_DATABASE_URL` is set. With PostgreSQL, `make backup` takes both and
`make restore` puts both back, so a restored workspace lists the same catalog; without it, everything is in the folder
and a copy of the folder is the backup.

## What is kept, and where

| What | Files (no database) | PostgreSQL | Written by |
|---|---|---|---|
| The session record: every event of every visit (19.2) | `experience/YYYY/MM/DD/<visit>.jsonl` | `experience_event` | `dclab_rnd/experience`, every central writer |
| Run records: what each stage, pipeline and agent run read, was told and made (19.5) | inside each stage's record, `<project>/stages/<stage>.json` (`run`), and a `run.recorded` event | `stage_record` and `experience_event` | `studio/lineage.py` |
| Decisions and their labels (19.6–19.8) | `ledger/decisions/YYYY/MM/DD.jsonl`, `ledger/labels/YYYY/MM/DD.jsonl` | `decision_record`, `decision_label` | `ledger/capture.py`, `ledger/labels.py` |
| Data cards (19.11) | a draft's `data_card`, a project's `data.card` | inside the draft and project documents | `studio/data.py`, `draft/pipeline.py` |
| Consent (19.10) | `consent.json`; each project's `settings.consent` | the project documents | `consent.py`, the owner only |
| Datasets (19.12) | `datasets/<builder>/<name>/`: `rows.jsonl`, `sealed.jsonl`, `manifest.json` | (files only) | `make datasets` |
| Failure candidates reviewed (19.12) | `datasets/failures/reviews.json`; accepted ones in `.../accepted.jsonl` | (files only) | a reviewer |
| The audit: who did what, and every deletion's tombstone | `audit.jsonl` | `audit_event` | `audit.py` |

The files of the session record and the ledger are append-only; only a deletion or a retention sweep rewrites them
(`experience/retention.py`), under the folder's lock that every appender holds (`filelock.py`). On PostgreSQL the
same tables refuse an update or a delete except from those two functions (migration 0017).

## The datasets

One folder per set: `<builder>/<name>/` with

- `rows.jsonl`: the train and dev rows, each with its `split`;
- `sealed.jsonl`: a decision set's sealed partition, kept apart and never used for tuning (written only when there
  are sealed rows);
- `manifest.json`: a `dclab_core.datasets.DatasetManifest`: builder, version, when, the consent it was built under,
  the rows and splits, counts, every source by id (decisions, labels, events, conversations, runs, visits, drafts,
  projects, packages) and how it was split.

The builders are `decisions` (one set per decision point), `conversations`, `preferences`, `trajectories`,
`workflows`, `failures` and `development` (the development records, the owner's own: `development/manifest.json`,
pointing at `research/llm-fine-tuning/experiments/sft/out_dev`). A rebuild replaces a set's files in place; the
manifest says when it was built.

## The catalog

`python -m dclab_rnd.experience catalog` lists every set with its builder, rows, consent, version, date and its
sources counted by kind; `--lineage BUILDER/NAME` resolves one set's sources to the sessions (visits) they came from:
an event to its visit, a conversation, run, draft or project to every visit that touched it, a decision to what it
belonged to. The Admin page's **Datasets** tab shows the same (`GET /api/datasets`, `GET /api/datasets/lineage`),
for the owner only.

## What leaves

Nothing leaves a workspace without its owner's switch (`consent.py`): the datasets are built only from what the
consent export lets through (redacted text; never a table's values, a model prompt, a person's id or a request's id).
Since 23.1 every event and decision carries the level it was written under (`consent`: `operate`, `improve`,
`train`); a record written before a switch was turned on never leaves, however the switches stand later, and every
change of a switch is a `consent.changed` event in the record with the notice version shown.
Moving a dataset out of the workspace folder is the owner's act, outside DCLab.

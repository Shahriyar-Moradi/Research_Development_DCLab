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
| How a decision turned out (23.5) | `ledger/outcomes/YYYY/MM/DD.jsonl` | `decision_outcome` | `ledger/outcomes.py` |
| Data cards (19.11) | a draft's `data_card`, a project's `data.card` | inside the draft and project documents | `studio/data.py`, `draft/pipeline.py` |
| Consent (19.10) | `consent.json`; each project's `settings.consent` | the project documents | `consent.py`, the owner only |
| Datasets (19.12) | `datasets/<builder>/<name>/`: `rows.jsonl`, `sealed.jsonl`, `manifest.json` | (files only) | `make datasets` |
| Failure candidates reviewed (19.12) | `datasets/failures/reviews.json`; accepted ones in `.../accepted.jsonl` | (files only) | a reviewer |
| The audit: who did what, and every deletion's tombstone | `audit.jsonl` | `audit_event` | `audit.py` |
| Claude beside the system (phase 3, 25.2–25.5): the twin folder, both approaches side by side | `twin/index.jsonl`, `twin/sessions/YYYY/MM/DD/<session>/` (a Home session: `session.json`, `conversation.jsonl`, `ai_first.jsonl`, `system.jsonl`, `thinking.jsonl`, `steps.jsonl`, `picks.jsonl`), `twin/runs/<run>/` (a whole problem in two lanes, each a workspace folder; `steps.jsonl`, `expert_steps.jsonl`, Claude's expert record in `ai_first/design/`); the switch in `twin.json` | (files only, in both modes) | `dclab_rnd/twin` |

The files of the session record and the ledger are append-only; only a deletion or a retention sweep rewrites them
(`experience/retention.py`), under the folder's lock that every appender holds (`filelock.py`). On PostgreSQL the
same tables refuse an update or a delete except from those two functions (migrations 0017, 0018).

## Outside the workspace: live tests, backups, measurements and the build history

Everything below is on the owner's disk (never in the session scratchpad, which `/private/tmp` clears on a reboot) and,
except the committed results, kept out of git. A live test that calls a real model writes to a durable folder from the
start and is verified by checksum after.

| What | Where | Committed |
|---|---|---|
| A live test's whole workspace (drafts, Home conversations, Claude beside them, projects built, agent sessions, every model call with its cost, events, decisions) and the scripts that drove it | `agent_runs/live_tests/<date>_<topic>/` (`workspace/`, `driver/`, `README.md`) | no |
| Twin runs and Home twin sessions the Evaluation page reads | `agent_runs/twin/runs/`, `agent_runs/twin/sessions/` | no |
| A backup: the database (pg_dump) and the workspace folder, with SHA-256 hashes | `backups/<time>/` (`database.dump`, `files.tar.gz`, `manifest.json`), by `make backup`; `make restore` puts it back | no |
| Measurements: one immutable JSON per result, a correction is a new result that names the one it supersedes | `evidence/campaigns/<campaign>/results/` (twin: `twin_v1/`, harness: `harness_v1/`, agent eval: `agent_eval_v1/`) | yes |
| How DCLab is built and fixed: the curated records, and the raw session transcripts | `research/llm-fine-tuning/experiments/dev_trajectories/records/` (committed), `.../raw/` by `make dev-archive` (local only) | records only |
| Decision benchmarks (the rules' labelled decisions on invented tables, a sealed partition) | `research/llm-fine-tuning/experiments/decisions/` | the manifest; rows by `make decisions-gen` |

Everything here is on one disk. A second copy somewhere else (an external drive, or the private records repository of
`docs/guides/TESTERS.md`) is the owner's to make; `make backup TO=<that folder>` writes there directly.

## The datasets

One folder per set: `<builder>/<name>/` with

- `rows.jsonl`: the train and dev rows, each with its `split`;
- `sealed.jsonl`: a decision set's sealed partition, kept apart and never used for tuning (written only when there
  are sealed rows);
- `manifest.json`: a `dclab_core.datasets.DatasetManifest`: builder, version, when, the consent it was built under,
  the rows and splits, counts, every source by id (decisions, labels, events, conversations, runs, visits, drafts,
  projects, packages) and how it was split.

The builders are `decisions` (one set per decision point), `conversations`, `preferences`, `trajectories`,
`workflows`, `problem_maps` and `routes` (24.7: the words a person said and the map of parts first drawn, built from and corrected, with what the programme did against its best leaf; a sub-problem's state, candidate approaches, the one chosen and what became of it), `failures`, `outcomes` (23.5: each decision with the number that followed it), `twin` (25.7: what the system and Claude answered on the same inputs: `twin.steps`, `twin.preferences` where a truth chose one, `twin.expert` the expert's records with what became of them, and Claude's answers as a teacher's beside the rule's at the ledger's points, `twin.teacher.workflow` and `twin.teacher.technical`, the material of the decision models of either use) and `development` (the development records, the owner's own: `development/manifest.json`,
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
consent export lets through (redacted text; never a table's values, a person's id or a request's id; a model's prompts and replies only under
train, redacted, for the purposes allowed to keep them). The notice moved to 2026-10-16 with that change (23.10).
Since 23.1 every event and decision carries the level it was written under (`consent`: `operate`, `improve`,
`train`); a record written before a switch was turned on never leaves, however the switches stand later, and every
change of a switch is a `consent.changed` event in the record with the notice version shown.
Moving a dataset out of the workspace folder is the owner's act, outside DCLab.

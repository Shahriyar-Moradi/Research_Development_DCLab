# Development trajectories: how DCLab is built and fixed

The owner asked (2026-10-12) that everything done while building DCLab be kept for training their own model: each
package's request, plan, decisions and why, every finding (a reviewer's, a failing test's, a measurement's, the owner's
corrections) with the scenario that shows it, its cause, the fix and the test that proves it, and every measurement
against the gate fixed before it ran.

| Path | What | Committed |
|---|---|---|
| `records/<package>.json` | one curated trajectory per package (schema: `Trajectory` in `dclab_rnd/devlog.py`); written as each package is built, and backfilled for earlier work from the playbook, git history and the session transcripts | yes |
| `raw/` | the Claude Code session transcripts of this project, copied by `make dev-archive`: every step in full | **no** (gitignored: they may hold private data, such as the challenge data, which must never leave the machine) |
| `../sft/out_dev/` | the SFT examples built from the records by `make sft-dev` (chat and alpaca formats, a manifest) | yes |

The examples teach four things: `plan_package` (the request → the plan and the decisions with reasons), `review_change`
(the change → the defects, each with its scenario), `fix_finding` (a defect → its cause, the smallest fix, the test that
proves it) and `judge_measurement` (a gate fixed before the run and the numbers → the verdict and what follows) and `retrospective` (the request and the
outcome → the lessons the next package should keep).
Whole packages are held out for validation. The builder refuses absolute paths, long hashes and e-mail addresses; a
record holds names, decisions and aggregate numbers, never a row of anyone's data.

```bash
make sft-dev       # rebuild ../sft/out_dev/ from the records
make dev-archive   # refresh the local raw transcripts (not committed)
```

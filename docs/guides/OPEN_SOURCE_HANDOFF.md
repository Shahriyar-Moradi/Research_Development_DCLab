# dclabai (open-source edition): handoff, 2026-10-21

Read this first in a new session. It says where everything is, what is done, what is in progress, and what is left.

## 1. Where things are

| What | Where |
|---|---|
| Public repo | https://github.com/Shahriyar-Moradi/dclabai (public, Apache-2.0, fresh history, CI green on `28ee162`) |
| Private source (the only place to edit) | `R&D` repo, `main`. The public tree is **generated** from it, never edited by hand. |
| Worktree used for this work | `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/rnd-open-seams`, branch `open-seams` |
| The plan | `~/.claude/plans/async-skipping-firefly.md` |
| Memory note | `~/.claude/projects/-Users-shahriar-Desktop-Desktop-Work-MyStartUp-R-D/memory/open-source-dclab.md` |
| Development record (training data) | `research/llm-fine-tuning/experiments/dev_trajectories/records/open-source-dclabai.json` and `pandas-3-support.json` |
| Live test records | `agent_runs/live_tests/2026-10-21/open-edition/` (Claude and Ollama runs) |

The export machinery (all private, never published):
- `scripts/export_open.py`: builds the public tree, runs every check, writes `<out>.EXPORT_REPORT.md` and `.json`.
- `export/manifest.py`: the allowlist (what ships), the excluded paths, the test selection, the text rewrites.
- `export/forbidden.txt`: private words. **Never exported.** A hit anywhere fails the export.
- `export/overlay/`: files only the public repo has: `pyproject.toml`, `README.md`, `docs/`, `Dockerfile`,
  `docker-compose.yml`, `packages/npm/`, `.github/workflows/ci.yml` and `release.yml`, `scripts/run_tests.py`,
  LICENSE, NOTICE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, CHANGELOG.
- In the code: `dclab_rnd/extensions.py` (`optional()`, `edition()`, `load_plugins()`), `dclab_rnd/gates.py`
  (which agents use the model), and `# dclab:closed-begin` / `closed-end` blocks (58 today) that the export strips.

## 2. Commits: what is pushed and what is not

| Commit | Where | Pushed? |
|---|---|---|
| `f93741cc` and earlier (all open-source work up to the live-test fixes) | private `main` and `dev` | yes |
| `28ee162` "Safer model turns in the project chat" | public `main` | yes, CI green |
| **`878e3295` "Support pandas 3..."** | private branch `open-seams` only | **no** |

Careful: private `main` has moved on since then (`58ccfc48`, from another session). `878e3295` is not in `main`.
Bring it in with `git cherry-pick 878e3295` (or merge `open-seams`) once its checks pass (section 4).

## 3. What was done

**Seams in the private code** (the same code runs in both editions):
- Closed features sit behind `extensions.optional()` and degrade to "off".
- Measured switches go through `gates.py`.
- Contracts load lazily.
- Sample data comes from invented replicas, never the owner's data.

**The export** (`scripts/export_open.py`):
- Copies only what the allowlist names (from `git ls-files`), renames `dclab_rnd` → `dclabai`, and strips closed blocks.
- Scans for private words and secrets, then checks that the open tree never imports a closed module.
- Builds the UI.
- Ships 30 knowledge records. It drops 3 that name private data: DCLAB-R13, PIT-004, PIT-005.
- Reruns the rare_event (7 of 7) and anomaly (8 of 8) acceptance checks in the open tree.
- **Builds the wheel and scans it** too: 548 files, 0 findings.
- Writes a readable `EXPORT_REPORT.md`. The last export had **0 findings**: 637 files, 8.7 MB, 98 test files.

**The product as shipped:**
- `dclab` command: `start`, `init`, `doctor`, `models trust/status`, `migrate`, `users`, `worker`, `version`.
- Bring your own model: Ollama, any OpenAI-compatible URL, Claude, or none (the rules answer).
- `dclab doctor` probes whether the model calls tools properly. Only then do the agents use it.
- Admin's Models tab shows, for each agent, "model" or "rules" and why (`GET /api/models/agents`).
- Plugins: installed packages register through the entry point group `dclabai.plugins`. A broken plugin is skipped.
- Open theme (`--dc-*` tokens). The copied TripsSource kit is excluded.
- Docs: `quickstart`, `models`, `concepts`, `architecture`, `privacy`, `adding-capability`.
- Install paths: pip/uv, the npm launcher (`npx dclabai`), Docker (CI builds the image).

**Leak fixes, already done:** HyperAck, the confidential fraud challenge, the retail interview, Jev, and
main-product names are forbidden words. Every mention is removed or rewritten as invented content. Do **not** add a
"rename" for private data. Forbid the word and remove the content.

**Live checks on the installed package** (fresh Python 3.12, installed from the export):

| Check | Result |
|---|---|
| Install and `dclab start`; 5 product flows (upload, team, no data, synthetic, log file) | pass |
| PostgreSQL 16 with sign-in on: owner account, API token, signed-out request refused (401), 5 flows | pass |
| Notebook cells in a Jupyter kernel | code cells run; the holdout cell is refused (409) |
| Ollama `qwen2.5:1.5b` (passes the tool probe) in the project chat | found a bug, fixed (below) |
| Claude (cap €0.5; 21 requests, €0.11) in the project chat | good tool use; found a bug, fixed (below) |
| npm launcher, browser check (16 pages, 0 errors), public CI | pass |

**Bugs found live and fixed** (both pushed):
1. A small model that ran no tool wrote made-up prose, including "a new version is awaiting a confirmation". Now,
   when the rules would act and the model ran no tool, the rules answer.
2. Claude confirmed a problem change on a message that was not a yes ("build a new version with 5 folds"). Now a
   change is confirmed only when the person says yes.

**pandas 3 port** (`878e3295`, not pushed). On pandas 3, 92 of 97 open test files passed. Three causes:
1. Text columns are pandas 3's new string type, so `dtype == object` missed them. Fix: `dclab_core.times.is_text`,
   used in 9 places.
2. pandas 3 reads a time without a year ("Oct 1 14:32") as **year 1**, silently. Fix: `read_times` leaves such
   values missing, in all 10 places that parse mixed times. This one was a hidden data bug, not just a crash.
3. Copy-on-write hands out read-only arrays. Fix: 3 places copy before changing.

The 7 affected test files pass on pandas 2.3.3 and 3.0.6. dclab-core is now 0.100.1, with a changelog entry and a
development record.

## 4. In progress when this session ended

- The open edition's **whole suite on pandas 3** was running in the background. It had reached 15 of 98 files, with
  0 failures. Its output was in the old session's temporary folder, which may be gone. **Rerun it** (section 6).

## 5. What remains

**For Claude (next session), in order:**
1. Rerun the open suite on pandas 3 and fix any failure the same way: one shared helper, pandas 2 unchanged.
2. Run the full private gate on pandas 2: `make check-all`. The owner's rule allows it now that the plan is done.
   Fix what fails.
3. If both pass:
   - Cherry-pick `878e3295` into private `main`, keep `dev` fast-forwarded, and push.
   - In `export/overlay/pyproject.toml`, change `pandas>=2.0,<3` to `pandas>=2.0,<4` and drop the "not supported"
     comment.
   - Add a pandas 3 job to the public CI (`export/overlay/.github/workflows/ci.yml`).
   - Export and update the public repo (section 6). Check its CI.
4. Watch the public CI's Python 3.11 job. It failed once (run for `0ab1ba5`) and passed on the next commit. If it
   fails again, it is flaky: find out why.
5. Small leftovers (low risk):
   - Hidden twin-panel markup on Home.
   - Package numbers and "main product" mentions in docstrings.
6. Before a release:
   - Have a fresh-eyes agent audit the exported tree for leaks.
   - Run `dclab-reviewer` on the seams.

**For the owner (Claude cannot do these):**
1. Claim the names. As of 2026-10-21, `dclabai` is **free on PyPI, TestPyPI and npm**. Claim them soon.
2. Set up trusted publishing for PyPI and npm, and GHCR for the Docker image (`release.yml` is ready).
3. Read `EXPORT_REPORT.md` and look over the open theme in the browser.
4. Decide whether the fraud_programme replica is generic enough to ship.
5. Release in this order:
   - Tag `v0.1.0rc1` (TestPyPI, npm `next`, GHCR).
   - Test-install on a clean Mac and on Linux.
   - Then `v0.1.0`.
6. Local Docker Desktop is broken. Its disk image was damaged when the Mac's disk filled up. The owner chose to leave
   it; CI builds the image. If it is ever needed again: Docker Desktop, then Troubleshoot, then "Clean / Purge data".
   **That erases everything in Docker.**
7. `training_records.zip` (5.3 GB) is on the same disk as the originals, so it is not a backup yet. Move it to an
   external drive or the cloud. The 6 old snapshots are in the Trash; emptying it frees about 0.9 GB.

**Planned for v0.2, not started:** SQLite storage, opt-in sharing of traces, running tool calls that small models
write as text (behind a setting), a desktop app, making the public repo the upstream.

## 6. How to (commands)

The commands below use these two names. In a new shell, set them first:

```bash
SEAMS=/Users/shahriar/Desktop/Desktop/Work/MyStartUp/rnd-open-seams
PY="/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/.venv/bin/python"
```

Export (about 1 minute; `--no-wheel` skips the wheel scan):

```bash
cd "$SEAMS" && "$PY" scripts/export_open.py --out <SCRATCH>/dclabai-next
```

The open suite on pandas 3 (about 40 minutes on this Mac):

```bash
uv venv --python 3.12 <SCRATCH>/p3o && echo "pandas>=3" > <SCRATCH>/override.txt && uv pip install --python <SCRATCH>/p3o/bin/python --override <SCRATCH>/override.txt "<SCRATCH>/dclabai-next[notebook]"
```

```bash
cd <SCRATCH>/dclabai-next && env -u DCLAB_TEST_DATABASE_URL DCLAB_NO_LIVE_MODELS=1 <SCRATCH>/p3o/bin/python scripts/run_tests.py --jobs 4
```

Private tests (always `DCLAB_NO_LIVE_MODELS=1`, so the `.env` key is never used):

```bash
cd "$SEAMS" && DCLAB_NO_LIVE_MODELS=1 "$PY" tests/run_parallel.py test_core_times.py test_draft_data.py
```

To update the public repo, first clone it into a scratch folder:

```bash
git clone https://github.com/Shahriyar-Moradi/dclabai.git <SCRATCH>/dclabai-public
```

Then copy the export over the clone, keeping its `.git` folder:

```bash
rsync -a --delete --exclude .git <SCRATCH>/dclabai-next/ <SCRATCH>/dclabai-public/
```

Commit with the no-reply email, never the owner's real address, then push:

```bash
cd <SCRATCH>/dclabai-public && git add -A && git -c user.name="Shahriar Moradi" -c user.email="62621798+Shahriyar-Moradi@users.noreply.github.com" commit -m "..." && git push origin main
```

Check the public CI:

```bash
gh run list -R Shahriyar-Moradi/dclabai -L 3
```

## 7. Rules to keep

- Edit only the private source. The public tree is regenerated on every export.
- A new private word goes into `export/forbidden.txt`. Private content is removed, never renamed.
- `git add` explicit paths only. The owner keeps uncommitted work in the R&D tree. Never `git stash`.
- Tests never use a live key. Live model runs are capped at €0.5 each, and their records go to
  `agent_runs/live_tests/<date>/`.
- Check free disk before anything big. Stop below 2 GB. A local Docker build once grew about 7 GB and broke Docker.
- The Mac has 8 GB of memory. Run heavy jobs one after another, not together.
- Every fix gets a development record in `dev_trajectories/records/`, then `python -m dclab_rnd.devlog`.

## 8. A first message for the new session

> Continue the dclabai open-source work. Read `docs/guides/OPEN_SOURCE_HANDOFF.md` in the R&D repo, then do
> section 5 "For Claude" in order, starting with the pandas 3 run on the open suite.

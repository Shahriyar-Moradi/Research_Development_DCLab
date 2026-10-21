# dclabai (open-source edition): handoff, 2026-10-21 (evening)

Read this first in a new session. It says where everything is, what is done, what is in progress, and what is left.

## 1. Where things are

| What | Where |
|---|---|
| Public repo | https://github.com/Shahriyar-Moradi/dclabai (public, Apache-2.0). Latest: `5e4df29`, CI green (all 9 jobs, run 39651852173) |
| Private source (the only place to edit) | `R&D` repo, `main`. The public tree is **generated** from it, never edited by hand. |
| Worktree used for this work | `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/rnd-open-seams`, branch `open-seams` (equal to `main` after the push) |
| The plan | `~/.claude/plans/async-skipping-firefly.md` |
| Memory note | `~/.claude/projects/-Users-shahriar-Desktop-Desktop-Work-MyStartUp-R-D/memory/open-source-dclab.md` |
| Development records (training data) | `research/llm-fine-tuning/experiments/dev_trajectories/records/`: `open-source-dclabai.json`, `pandas-3-support.json`, `open-edition-review-fixes.json` |
| Live test records | `agent_runs/live_tests/2026-10-21/open-edition/` (Claude and Ollama runs) |

The export machinery (all private, never published):
- `scripts/export_open.py`: builds the public tree, runs every check, writes `<out>.EXPORT_REPORT.md` and `.json`.
- `export/manifest.py`: the allowlist (what ships), the excluded paths, the test selection, the text rewrites.
- `export/forbidden.txt`: private words. **Never exported.** A hit anywhere fails the export.
- `export/overlay/`: files only the public repo has: `pyproject.toml`, `README.md`, `docs/`, `Dockerfile`,
  `docker-compose.yml`, `packages/npm/`, `.github/workflows/ci.yml` and `release.yml`, `scripts/run_tests.py`,
  LICENSE, NOTICE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, CHANGELOG.
- In the code: `dclab_rnd/extensions.py` (`optional()`, `edition()`, `load_plugins()`), `dclab_rnd/gates.py`
  (which agents use the model), and `dclab:closed-begin` / `closed-end` blocks (70 today) that the export strips.

## 2. Commits: what is pushed

Everything is pushed. Nothing waits on a branch.

| Commit | Where | What |
|---|---|---|
| `3447c490` "Support pandas 3..." | private `main` and `dev` | the pandas 3 port (section 3) |
| `49f40b3a` "Fix what reviewing the installed open edition found" | private `main` and `dev` | the review fixes (section 3) |
| `5e4df29` "pandas 3 support, a step-by-step README, and fixes..." | public `main` | the export of `49f40b3a` |

The public commit also carries the other sessions' product work since `28ee162` (for example the Home team), because
the export publishes whatever private `main` holds. That export had 0 findings.

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
- **Builds the wheel and scans it** too: 551 files, 0 findings.
- Writes a readable `EXPORT_REPORT.md`. The last export had **0 findings**: 641 files, 8.8 MB, 100 test files.

**The product as shipped:**
- `dclab` command: `start`, `init`, `doctor`, `models trust/status`, `migrate`, `users`, `worker`, `version`.
- Bring your own model: Ollama, any OpenAI-compatible URL, Claude, or none (the rules answer).
- `dclab doctor` probes whether the model calls tools properly. Only then do the agents use it.
- Admin's Models tab and `dclab models status` show the same rows: for each agent, "model" or "rules" and why.
  Two roles also need their own switch: `DCLAB_CONVERSATION_READER=1` and `DCLAB_MAP_ROLES=1`.
- Plugins: installed packages register through the entry point group `dclabai.plugins`. A broken plugin is skipped.
- Open theme (`--dc-*` tokens). The copied TripsSource kit is excluded.
- Docs: `quickstart`, `models`, `concepts`, `architecture`, `privacy`, `adding-capability`.
- **Install: GitHub only for now.** The README says so and walks through it step by step (set up, run, a first
  project, what to do when something goes wrong). pip, npx and the Docker image return 404 until the first release.

**Leak fixes, already done:** HyperAck, the confidential fraud challenge, the retail interview, Jev, and
main-product names are forbidden words. Every mention is removed or rewritten as invented content. Do **not** add a
"rename" for private data. Forbid the word and remove the content.

**pandas 3** (`3447c490`, pushed):
1. Text columns are pandas 3's new string type, so `dtype == object` missed them. Fix: `dclab_core.times.is_text`.
2. pandas 3 reads a time without a year ("Oct 1 14:32") as **year 1**, silently. Fix: `read_times` leaves such
   values missing. This was a hidden data bug, not just a crash.
3. Copy-on-write hands out read-only arrays. Fix: 3 places copy before changing.
- The open suite passed **98 of 98 files on pandas 3.0.6**. The open edition now needs `pandas>=2.0,<4`.
- Public CI runs the newest pandas everywhere, plus one job that pins `pandas<3` so pandas 2 stays tested. On
  `5e4df29` all 9 jobs passed (31 minutes): pandas 3.0.6 on Python 3.11, 3.12 (Linux and macOS) and 3.13; pandas 2.3.3
  in the pinned job and on Python 3.10 (pandas 3 needs 3.11 or later).
- Some test files were very slow (`test_spatial` 32 min, `test_cascade` 47 min). The same file was as slow on
  pandas 2 under the same load: the cause was other jobs on the machine, not pandas 3.

**Review of the installed package** (`49f40b3a`, pushed). The owner installed from GitHub and ran it with Claude
Sonnet 5.5. Then a review covered an Intern run, all 17 pages, the server log, the CLI and the 5 product flows.
Found and fixed:
1. The README offered installs that all failed, with `# or:` comments that zsh passes to pip as package names.
2. The understanding team's panel showed on every Home draft in the open edition, and its button could only fail.
3. The Blueprint switch and the command palette opened blank private pages (Research lab, Benchmark, Policy model).
4. The reader's (and the map roles') status said "model" while their switch was off and the rules served.
5. Synthetic text could not carry the target: a synthetic sentiment table scored at chance (macro-F1 0.358).
   Now a text column takes effects on **whole words** its sentences contain (`"comment=problem": -2`), and the
   prompt says when to use them.
6. The plugin test broke alembic in the PostgreSQL pass (it faked entry points for every library). Pre-existing.

Gates on the pushed code: `check-all` passed part by part. The product flows passed when run alone; together they
had timed out under load. After the rebase, `check-quick`, `agent-eval` (AEV-051) and `harness-eval` (HRN-039,
17 of 17) passed. A `dclab-reviewer` pass found 2 problems in the fixes (items 4 and 5 above), both fixed.

## 4. In progress when this session ended

Nothing. All work is pushed.

## 5. What remains

**For Claude (next session):**
1. Real text features for multiclass targets. Today only binary targets get word TF-IDF (the text machine). A
   3-class sentiment task treats text as categories, so it stays weak even on real reviews.
2. With no model chosen in `dclab init`, an `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` already in the shell is used
   anyway. The docs say so now. Decide whether `dclab` should use only the model chosen in `dclab init`.
3. A step of `make check-all`, run in a worktree, rewrote `training_records/STORE_CATALOG.json` from the worktree's
   empty store. It would have deleted 3,156 lines. It was restored and not committed. Find the step, and make it never
   write the repository's copy outside the main checkout.
4. `make product-e2e` waits 40 seconds for its server. Under load that was too short once. Consider a longer wait.
5. Watch the public CI's Python 3.11 job. It failed once (run for `0ab1ba5`) and has passed since (also on
   `5e4df29`). If it fails again, find out why.
6. Small leftovers (low risk):
   - Hidden twin-panel markup on Home.
   - Package numbers and "main product" mentions in docstrings.
7. Before a release:
   - Have a fresh-eyes agent audit the exported tree for leaks.
   - Run `dclab-reviewer` on the seams.

**For the owner (Claude cannot do these):**
1. Claim the names: `dclabai` on **PyPI** and **npm**. They were free on 2026-10-21, and the public repo already uses
   the name.
2. Set up trusted publishing for PyPI and npm, and GHCR for the Docker image (`release.yml` is ready).
3. Read `EXPORT_REPORT.md` and look over the open theme in the browser.
4. Decide whether the fraud_programme replica is generic enough to ship.
5. Release in this order:
   - Tag `v0.1.0rc1` (TestPyPI, npm `next`, GHCR).
   - Test-install on a clean Mac and on Linux.
   - Then `v0.1.0`, and change the README's install back to pip, npx and Docker.
6. Local Docker Desktop is broken. Its disk image was damaged when the Mac's disk filled up. The owner chose to leave
   it; CI builds the image. If it is ever needed again: Docker Desktop, then Troubleshoot, then "Clean / Purge data".
   **That erases everything in Docker.**
7. `training_records.zip` (5.3 GB) is on the same disk as the originals, so it is not a backup yet. Move it to an
   external drive or the cloud.
8. The owner's conda env `ai12` had pandas 3.0.6 downgraded to 2.3.3 by a GitHub install of the old package. If other
   projects there need pandas 3: `pip install "pandas==3.0.6"` in `ai12`.

**Planned for v0.2, not started:** SQLite storage, opt-in sharing of traces, running tool calls that small models
write as text (behind a setting), a desktop app, making the public repo the upstream.

## 6. How to (commands)

The commands below use these two names. In a new shell, set them first:

```bash
SEAMS=/Users/shahriar/Desktop/Desktop/Work/MyStartUp/rnd-open-seams
PY="/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/.venv/bin/python"
```

Bring the worktree up to date first (`main` moves often; other sessions push to it):

```bash
cd "$SEAMS" && git fetch origin && git rebase origin/main
```

`make` breaks on the `&` in the R&D path. To run `make` targets in the worktree, link the environments first (git does
not ignore the links: never `git add` them, and remove them after):

```bash
cd "$SEAMS" && ln -s "/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/.venv" .venv && ln -s "/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/.venv-agent" .venv-agent
```

Export (about 1 minute; `--no-wheel` skips the wheel scan):

```bash
cd "$SEAMS" && "$PY" scripts/export_open.py --out <SCRATCH>/dclabai-next
```

The open suite on pandas 3 (40 to 70 minutes on this Mac, longer when other jobs run):

```bash
uv venv --python 3.12 <SCRATCH>/p3o && echo "pandas>=3" > <SCRATCH>/override.txt && uv pip install --python <SCRATCH>/p3o/bin/python --override <SCRATCH>/override.txt "<SCRATCH>/dclabai-next[notebook]"
```

```bash
cd <SCRATCH>/dclabai-next && env -u DCLAB_TEST_DATABASE_URL DCLAB_NO_LIVE_MODELS=1 <SCRATCH>/p3o/bin/python scripts/run_tests.py --jobs 4
```

A prompt change needs both scripted suites rerun (no model, no cost), then `"$PY" -m dclab_rnd.devlog`. Number the
`AEV-` file after the highest one in `main`:

```bash
cd "$SEAMS" && DCLAB_NO_LIVE_MODELS=1 "$PY" -m dclab_rnd.agent_eval --output evidence/campaigns/agent_eval_v1/results/AEV-0NN_judgment_v1_scripted.json
```

```bash
cd "$SEAMS" && DCLAB_NO_LIVE_MODELS=1 "$PY" -m dclab_rnd.agent_eval.regression
```

After `make check-all` in the worktree, check `git status`: restore `training_records/STORE_CATALOG.json` if it changed.

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
- A page left out of the open edition must take every way in with it: the menu, the palette, a switch, a saved setting
  and any sentence that points to it. Use closed blocks; the full edition must not change.
- A new private word goes into `export/forbidden.txt`. Private content is removed, never renamed.
- `git add` explicit paths only. The owner keeps uncommitted work in the R&D tree. Never `git stash`.
- Tests never use a live key. Live model runs are capped at €0.5 each, and their records go to
  `agent_runs/live_tests/<date>/`.
- Check free disk before anything big. Stop below 2 GB. A local Docker build once grew about 7 GB and broke Docker.
- The Mac has 8 GB of memory. Run heavy jobs one after another, not together. Other sessions often run long jobs
  (`lake selfplay`, `twin eval`): check `ps` before timing anything.
- Every fix gets a development record in `dev_trajectories/records/`, then `python -m dclab_rnd.devlog`. A finding's
  `source` is one of reviewer, test, measurement, owner, self; its `severity` one of behaviour, latent, gap, flaky,
  privacy, leakage; a measurement's numbers must appear in its own name or gate.

## 8. A first message for the new session

> Continue the dclabai open-source work. Read `docs/guides/OPEN_SOURCE_HANDOFF.md` in the R&D repo, then do
> section 5 "For Claude", starting with real text features for multiclass targets.

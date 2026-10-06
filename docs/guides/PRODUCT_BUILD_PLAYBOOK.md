# Product build playbook: the plan as small plans and prompts

The plan "from demo v1 to the working DCLab product" is divided here into **40 small plans**. Each one says what it
delivers, how to check it, and gives a prompt to paste into a new Claude Code session.

**Status in this repository (2026-10-05): 36 are done, 4 are open.** The open four need the owner: a key, an account
or a decision (section 8). Every done package names the commit that did it, so the playbook is also the map from the
plan to the code.

**Part 2 (sections 9 to 13) is the software foundation underneath**: a database, accounts, durable jobs, the model
gateway and agent runtime, and the infrastructure to run it for more than one person. **Of Part 2, packages 9.1, 9.2 and 11.4 are built.**
Today the product is a single-user program on one machine that keeps everything as files in a folder.

**خلاصه فارسی.** این سند برنامه‌ی «از دموی نسخه‌ی ۱ تا محصول واقعی DCLab» را به **۴۰ برنامه‌ی کوچک** تقسیم می‌کند.
هر برنامه می‌گوید چه چیزی تحویل می‌دهد، چطور بررسی می‌شود، و یک پرامپت آماده دارد که در یک نشست تازه‌ی Claude Code
جای‌گذاری می‌شود. در این مخزن **۳۶ برنامه انجام شده و ۴ برنامه باز است**؛ چهار مورد باز به تصمیم یا کلید شما نیاز
دارند (بخش ۸). برای بررسی کل مسیر یک فرمان کافی است: `make product-e2e`.

## How to use it

1. Open a new session in this repository.
2. Paste the **shared preamble** (next section), then the prompt of **one** package.
3. Work in order inside a phase. Phases 3 to 6 depend on phases 1 and 2.
4. A package is finished when its **Done when** checks pass and `make rd-check` passes.

The prompts are safe to run again: each session first runs the checks and builds only what is missing.

| Command | What it checks |
|---|---|
| `make rd-check` | every unit test, the evidence records, and that every generated file is current |
| `make product-e2e` | the four main flows on a temporary server: upload, no data, synthetic, log file (about 90 s, no model) |
| `make web` | rebuilds the product frontend into `dclab_rnd/agentic/static/app/` |
| `make notebook` | the product on http://127.0.0.1:8765 |

## Shared preamble (paste first, every time)

```text
You are working in the DCLab R&D repository. Read CLAUDE.md, then the package I name in
docs/guides/PRODUCT_BUILD_PLAYBOOK.md. Do only that package.

Rules for every package:
1. The UI is demo v1 (docs/product-demo, tag demo-v1, frozen; never edit it). The product frontend in
   dclab_rnd/agentic/web/src is a copy of it. Do not change layout, colours, components or wording unless the
   package says so. New pieces reuse existing components (.composer, .thread/.msg/.ask-card/.tool-call,
   .graph-svg, the data-source tabs, .pack-card, .stats).
2. Leakage is the top risk. Nothing unavailable at the prediction moment may be used. Cleaning before a project
   exists is structural and lossless. Analysis before the split is descriptive and never relates a column to the
   outcome. Imputation, scaling, encoding and selection stay inside training folds in the engine.
3. Code decides, a model advises. A model may propose JSON (a parse spec, a workflow, a pack, a synthetic
   schema); deterministic code validates it and applies it. A model never computes a split or a metric.
4. Every path that uses a model also works with no model configured. Tests use scripted clients, never a live key.
5. Synthetic data is labelled synthetic in storage, in the UI, in exports and in briefs.
6. Secrets stay on the server: never in drafts, events, logs, responses or the page.
7. Generated files are rebuilt by their command, never edited by hand: evidence/knowledge/**,
   research/*/INDEX.md, dclab_rnd/agentic/static/app/** (make web), docs/product-demo/index.html.
8. Other work may be uncommitted in this tree. Do not touch, stage or commit files this package did not need.

How to work:
- First run the package's "Done when" checks. If all pass, report that and stop. Otherwise build what is missing.
- Check the feature in the running product, not only with unit tests: start a server on a spare port with a
  temporary DCLAB_AGENT_HOME and use the API or the browser. `make product-e2e` covers the main flows.
- Finish with `make rd-check`. Commit only when it passes, with a message that explains why. Do not push.
- Report in clear English and Persian: what was built, how it was checked, and what is still open.
```

## What was asked, and where it is

| The owner asked | Packages |
|---|---|
| Commit the demo redesign and the test fix | 0.1 |
| Keep the demo as demo v1 | 0.2 |
| Same UI, UX and frontend, now fully working; change nothing unless asked | 1.1 to 1.5, rule 1 of the preamble |
| "Contract" becomes "solution" | 2.1 to 2.3 |
| Describe the problem in one sentence, or ask for any kind of model, in the "Draft the solution" form | 3.1, 3.9 |
| After the data arrives, structure it, clean it and prepare it in the background while the user stays on Home | 3.2 to 3.5 |
| A chat at the bottom asks a few questions to understand the real problem | 3.7 |
| The data analysis appears in the chat as parts become ready | 3.4, 3.5, 3.9 |
| The agent draws a solution workflow for this problem on Home and updates it from the chat | 3.6, 3.9 |
| "Let's build the solution" leads to the project | 3.10, 4.1 to 4.4 |
| No data: the chat still builds the workflow and asks for a sample later | 3.7, checked by 7.1 |
| "Connect data" instead of "Attach data", and a "Bring the data" card | 3.9, 6.1 to 6.3 |
| Synthetic data from a prompt, with no sample | 3.8 |
| On the next page: the preprocessing and a small analysis | 4.2 |
| Then the solution draft | 4.3 |
| Split and budget unchanged, then review and start | 4.4 |
| All other parts stay as they are | 5.1 to 5.6 (wired to real data, same look) |

## Status at a glance

| # | Package | Status | Commit |
|---|---|---|---|
| 0.1 | Commit the demo redesign and the test fix | Done | `d7937e9`, `8469de7` |
| 0.2 | Freeze the demo as v1 | Done | tag `demo-v1` |
| 1.1 | Copy the demo into the product frontend, with a build | Done | `b9abbad` |
| 1.2 | No inline styles or scripts (strict CSP) | Done | `b9abbad` |
| 1.3 | Serve the product at `/`, the old UI at `/classic` | Done | `b9abbad` |
| 1.4 | The page's data layer: API, live stream, polling | Done | `b9abbad`, `15c4958` |
| 1.5 | Serve the fonts from the app | Done | `93a9a65` |
| 2.1 | Rename contract to solution in the backend | Done | `db85ad2` |
| 2.2 | Rename the agent tools, keep the old names working | Done | `db85ad2` |
| 2.3 | Rename in the UI wording | Done | `db85ad2` |
| 3.1 | The draft: store, event log, routes, live stream | Done | `15c4958` |
| 3.2 | Turn any file into a table | Done | `15c4958` |
| 3.3 | Structural, lossless cleaning with a log | Done | `15c4958` |
| 3.4 | Descriptive analysis | Done | `15c4958` |
| 3.5 | The background pipeline and the upload and sample routes | Done | `15c4958`, `88acb12` |
| 3.6 | Pack detection and the solution workflow | Done | `15c4958`, `2899447` |
| 3.7 | The Home agent | Done | `15c4958`, `d9cf587`, `7ef1cea`, `2899447` |
| 3.8 | Synthetic data | Done | `15c4958`, `7ef1cea` |
| 3.9 | The Home page | Done | `15c4958`, `aeb1221` |
| 3.10 | Build the project from the draft | Done | `15c4958`, `7ef1cea` |
| 4.1 | The wizard's five steps and the Goal step | Done | `15c4958` |
| 4.2 | The Data step | Done | `15c4958` |
| 4.3 | The Solution draft step | Done | `15c4958`, `7ef1cea` |
| 4.4 | Split and budget, Review and start | Done | `15c4958`, `7ef1cea` |
| 5.1 | Workflow page | Done | `4e9b0d9`, `cbd79a2` |
| 5.2 | Solution & data page, with editing | Done | `6faef03`, `0678f0d` |
| 5.3 | Notebook, Leakage & features, Models, Reliability, Decision brief | Done | `6faef03`, `c13bb6f` |
| 5.4 | Intern page | Done | `6faef03`, `e5f66f7` |
| 5.5 | Evidence library | Done | `8997529` |
| 5.6 | Workspace and platform pages | Done | `1f96282`, `aeb1221`, `3695306` |
| 6.1 | Kaggle and Hugging Face | Done, not run against the live services | `96feb56` |
| 6.2 | Databases and cloud storage | Done | `96feb56` |
| 6.3 | Connector status, credentials tests, requirements | Done | `96feb56`, `ec3ae4d` |
| 7.1 | End-to-end flows | Done | `make product-e2e` |
| 7.2 | Visual, console and phone-width checks | Done | `02ca606` |
| 7.3 | Security checks | Done | `93a9a65` |
| 8.1 | Run every model path against a real model | **Open: needs the owner's yes** | |
| 8.2 | Run Kaggle and Hugging Face against the live services | **Open: needs accounts** | |
| 8.3 | Remove the old UI at `/classic` | **Open: needs a decision** | |
| 8.4 | Meter the euro budget | **Open: needs a design decision** | |

---

## Phase 0 — Commit and freeze demo v1

### 0.1 Commit the demo redesign and the test fix

- **Delivers:** two commits: the redesigned demo, and the intern test that no longer calls a live model.
- **Done when:** `git log --oneline -- docs/product-demo` shows the redesign; `tests/test_intern.py` patches the environment in `setUp`; `make rd-check` passes.

```text
Package 0.1. Commit the demo redesign (docs/product-demo/**) and the test fix (tests/test_intern.py) as two
commits. Leave every other modified or untracked file alone, including docs/recap/*. Run make rd-check first.
```

### 0.2 Freeze the demo as v1

- **Delivers:** the tag `demo-v1` and a note in the demo's README that it is the frozen reference.
- **Done when:** `git tag -l demo-v1` prints it; `docs/product-demo/README.md` says "Demo v1"; `python docs/product-demo/build.py --check` passes.

```text
Package 0.2. Tag the current demo commit as demo-v1 and add a short "Demo v1: frozen reference for development"
note to docs/product-demo/README.md. The demo stays buildable and its --check stays in make rd-check. From now on
nobody edits docs/product-demo. Ask me before pushing main, dev and the tag.
```

---

## Phase 1 — The product frontend (same UI, served for real)

### 1.1 Copy the demo into the product frontend, with a build

- **Delivers:** `dclab_rnd/agentic/web/src/` (shell, styles, core, one view file per page) and `web/build.py`, which writes separate files into `dclab_rnd/agentic/static/app/`.
- **Files:** `dclab_rnd/agentic/web/build.py`, `web/src/**`, `Makefile` (`web`, and `--check` inside `rd-check`).
- **Done when:** `python -m dclab_rnd.agentic.web.build --check` says the frontend is up to date; `tests/test_web_build.py` passes.

```text
Package 1.1. Copy docs/product-demo/src into dclab_rnd/agentic/web/src without changing its look. Write
dclab_rnd/agentic/web/build.py: it assembles index.html, app.css and one JavaScript file per view and per data
file into dclab_rnd/agentic/static/app/, with nothing inlined. Add --check (exit 1 when static/app is stale or
has extra files), a `make web` target, and put --check into make rd-check. Test it in tests/test_web_build.py.
```

### 1.2 No inline styles or scripts (strict CSP)

- **Delivers:** per-element styles written as `data-style="…"` and applied by `core.js` through one constructed stylesheet; the build fails on any inline `<script>`, `<style>`, `on…=` handler or `style=` attribute.
- **Done when:** `tests.test_web_build.BuildTests` passes, including the test that an inline style is caught; every page renders the same as demo v1.

```text
Package 1.2. The server's Content-Security-Policy is script-src 'self'; style-src 'self', so inline styles and
scripts are blocked. Replace every style="…" in the views and in strings built by JavaScript with data-style="…".
In core.js, turn each distinct data-style value into one rule of a constructed stylesheet
([data-style="…"]:not([hidden]){…}) and watch for new values with a MutationObserver. Add a global
[hidden]{display:none!important} rule. Make build.py refuse inline scripts, style blocks, event-handler attributes
and style attributes, and test that. Compare each page with demo v1 at 1440 px: nothing may look different.
```

### 1.3 Serve the product at `/`, the old UI at `/classic`

- **Delivers:** `/` returns the built app; `/classic` returns the earlier UI; the CSP allows only this app.
- **Done when:** `tests.test_web_build.ServeTests` passes.

```text
Package 1.3. In dclab_rnd/agentic/server.py serve dclab_rnd/agentic/static/app/index.html at / and move the
earlier UI to /classic (its files stay in static/). Keep the CSP strict: default-src 'self'; script-src 'self';
style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'. Test that /
serves the product, that /classic still serves the old UI, and that the CSP names no other host.
```

### 1.4 The page's data layer

- **Delivers:** `DC.api` (JSON, CSRF header from `/api/config`, one retry when the token is stale), `DC.stream` (server-sent events), `DC.poll`; the artifact-only store (`window.claude`) removed; the blueprint layer and the tours kept.
- **Done when:** the Home page loads `/api/workspace` with no console error; a write after a server restart succeeds on the retry.

```text
Package 1.4. Add the data layer to dclab_rnd/agentic/web/src/core.js: DC.api(path, opts) sends JSON with the
X-DCLab-Token header taken from GET /api/config and retries once after a 403 on a write (the server restarted);
DC.stream(path, onEvent) wraps EventSource; DC.poll repeats a call. Remove the demo-only window.claude storage
from the Decisions store (keep localStorage). Keep the blueprint layer and the tours. A page that is not wired to
the server yet keeps its sample data and the "Sample data" pill; a wired page calls DC.markSample(false).
```

### 1.5 Serve the fonts from the app

- **Delivers:** Inter, JetBrains Mono and Vazirmatn as variable woff2 files under `web/src/fonts/` (SIL Open Font License, licence texts beside them); no request to any other host.
- **Done when:** `tests.test_web_build` passes (no outside host in the page, the stylesheet or the CSP; every `@font-face` file is shipped).

```text
Package 1.5. Self-host the three fonts so a page load contacts no other host and the UI works offline. Put
variable woff2 files and their OFL licence texts in dclab_rnd/agentic/web/src/fonts, declare them with @font-face
under the same family names (the look must not change), make build.py copy and --check binary files, and remove
the Google hosts from the CSP. Ask me before downloading any font file. Test that nothing loads from another host.
```

---

## Phase 2 — Rename "contract" to "solution"

### 2.1 Backend

- **Delivers:** `studio/solution.py` with the `Solution` model; the project key `solution`; routes `/solution*` with `/contract*` kept as hidden aliases; the gate `solution` and the policy `require_solution_signoff`, with the old names mapped.
- **Done when:** `tests/test_solution_rename.py`, `tests/test_studio.py` and `tests/test_graph.py` pass; an old `project.json` with a `contract` key opens and is rewritten.

```text
Package 2.1. Rename the prediction contract to the solution in the backend. studio/contract.py becomes
studio/solution.py (keep contract.py as a thin shim). ProjectStore migrates old project.json files on read
(contract -> solution, old gate and policy names). In studio/graph.py WF-01 is "Solution draft", the gate is
"solution", the policy key is require_solution_signoff, and LEGACY_MOVES, LEGACY_GATES and LEGACY_POLICY map the
old names. Routes become /solution and /solution/proposal with the /contract routes kept as aliases hidden from
the schema. Update engine, export, SFT and the tests, and add a migration test.
Do not rename research records: rule text such as DCLAB-R01 "prediction contract", campaign results, the SFT
corpus and anything under evidence/ stay as they are.
```

### 2.2 Agent tools

- **Delivers:** `propose_solution` and `set_solution` for the intern and the MCP server; `propose_contract` and `set_contract` still accepted.
- **Done when:** `tests/test_intern.py` and `tests/test_mcp.py` pass, including a call by an old name.

```text
Package 2.2. Rename the intern and MCP tools propose_contract and set_contract to propose_solution and
set_solution (dclab_rnd/intern/tools.py, dclab_rnd/mcp_server.py). Accept the old names through a LEGACY_TOOLS
map so existing Chat UI setups keep working. Update the intern's standard plan and the tests.
```

### 2.3 UI wording

- **Delivers:** "Draft the solution", "Solution draft", "Solution & data", the WF-01 node "Solution", across the 19 views, `features.js` and `data.js`.
- **Done when:** a search for "contract" in `web/src` finds only research wording (for example "Decision-time contract" quoted from a dataset record) and internal CSS class names.

```text
Package 2.3. Reword the product frontend (dclab_rnd/agentic/web/src views, features.js, data.js): "contract"
becomes "solution" wherever it names the product's object ("Draft the solution", "Solution draft",
"Solution & data", the WF-01 node "Solution"). Keep quotes from research records as written. Internal CSS class
names may stay. Rebuild with make web and check every page for leftover wording.
```

---

## Phase 3 — Home: draft, data pipeline, chat and solution workflow

### 3.1 The draft: store, event log, routes, live stream

- **Delivers:** `dclab_rnd/draft/store.py` (one folder per draft: `draft.json`, `events.jsonl`, `data/`), create/read/update/delete routes, and `GET /api/drafts/{id}/events` as a resumable event stream.
- **Done when:** `tests.test_draft_flow.StoreTests` and `ApiTests.test_upload_and_events_stream` pass.

```text
Package 3.1. A draft is the workspace before a project exists. Write dclab_rnd/draft/store.py: DraftStore under
DCLAB_DRAFT_HOME with atomic saves, update(fn) under a per-draft lock, and an append-only event log with sequence
numbers. A draft holds the problem, the pack, messages, questions, assets, the cleaning log, the analysis, the
workflow, what the chat understood, and its status. In dclab_rnd/draft/api.py add POST/GET/PATCH/DELETE
/api/drafts and GET /api/drafts/{id}/events as server-sent events that replay from Last-Event-ID. Writes need the
CSRF token; a missing or malformed JSON body is a 422, never a 500.
```

### 3.2 Turn any file into a table

- **Delivers:** `draft/structure.py`: CSV with any delimiter, Excel, Parquet, JSON, nested JSON, JSON lines, web-server and key=value logs, plain text; a model may propose a parse pattern that code checks before using.
- **Done when:** `tests/test_draft_data.py` passes (it holds a fixture for every format, a good and a bad model pattern, and the no-model path).

```text
Package 3.2. Write dclab_rnd/draft/structure.py: to_table(path, client=None) returns a DataFrame and a record of
how the file was read (format, parser, parse rate, notes). Detect the format from the content, not the extension.
Built-in readers cover delimited text, Excel, Parquet, JSON (the largest list of objects), JSON lines, combined
access logs, key=value logs, timestamp-level-message logs and plain text. Only when no built-in reader fits may a
model be asked once for a pattern; code applies it to a sample, measures the parse rate and refuses a poor one.
Unusable or oversized files raise StructureError with a message a user can act on.
```

### 3.3 Structural, lossless cleaning with a log

- **Delivers:** `draft/clean.py`: names, missing-value markers, trimming, numbers and dates written as text, yes/no columns, empty rows and columns, exact duplicates, category codes; every step logged with counts.
- **Done when:** the cleaning tests in `tests/test_draft_data.py` pass; the input frame is never changed.

```text
Package 3.3. Write dclab_rnd/draft/clean.py: clean(frame) returns a new frame and a log. Only structural,
lossless steps: tidy column names, unify missing-value markers, trim text, read numbers, dates and yes/no values
written as text, drop empty rows and columns, drop exact duplicate rows (counted). No imputation, scaling,
encoding or outlier removal: those belong inside training folds. Each step that changed something is one log
entry with a title, a detail and counts. Also detect numeric columns that are really category codes.
```

### 3.4 Descriptive analysis

- **Delivers:** `draft/analyze.py`: size, kinds, missing values, distributions, top values, time coverage, feature-to-feature correlations, and plain findings; nothing relates a column to the outcome.
- **Done when:** the analysis tests in `tests/test_draft_data.py` pass, including the one that leaves the target and outcome-like columns out of the correlations.

```text
Package 3.4. Write dclab_rnd/draft/analyze.py: analyze(frame, target=None) returns plain JSON for the page's
charts: a summary, studio.data.profile_table's profile, per-column distributions (histograms and quantiles, top
values, time coverage), Spearman correlations between features only, and short findings (many missing values,
duplicates, likely identifiers, near-constant columns, possible category codes, names that suggest a value known
only after the outcome). This runs before any split, so it must never relate a column to the target: leave the
target and every outcome-like column out of the correlations. Describe large tables from a seeded sample.
```

### 3.5 The background pipeline and the upload and sample routes

- **Delivers:** `draft/pipeline.py` (structure, clean, analyze, as a background job that writes events) and the routes `PUT /api/drafts/{id}/data?filename=` and `POST /api/drafts/{id}/data/sample`.
- **Done when:** `tests.test_draft_flow.ApiTests.test_full_flow_sample_then_build` passes; `make product-e2e` passes `upload` and `log-file`.

```text
Package 3.5. Write dclab_rnd/draft/pipeline.py: new_asset registers a data source on the draft; run(store,
draft_id, asset_id, agent, client) moves it through structuring, cleaning and analysing to ready or failed,
emitting a "pipeline" event at each step and an "analysis" event at the end, then calls agent.data_ready. Save the
cleaned table as Parquet in the draft. Add the upload route (raw body, sanitised file name, size limit) and the
studied-sample route. Both return at once and run the pipeline with asyncio.to_thread. A failure is reported to
the user in plain words and never crashes the job.
```

### 3.6 Pack detection and the solution workflow

- **Delivers:** `draft/pack.py` (nine packs; the user's choice wins, then the problem's words, then the data) and `draft/workflow.py` (a workflow built on the blocks WF-01 to WF-10, gates added by code, a model's proposal validated).
- **Done when:** `tests.test_draft_flow.PackTests` and `WorkflowTests` pass.

```text
Package 3.6. Write dclab_rnd/draft/pack.py: PACKS (key, name, description, maturity) and detect(problem,
analysis, chosen). A pack the user chose always wins. A horizon such as "next month" means forecasting only when
the sentence does not ask "which ones" (cancel, churn, leave). Write dclab_rnd/draft/workflow.py: template(pack,
understanding) returns the ten steps with labels for this problem; validate() accepts a model's proposal only if
every step maps to a block WF-01..WF-10 in order and the required blocks are present; gates and revisit loops are
added by code, never by the model. Each change makes a new version.
```

### 3.7 The Home agent

- **Delivers:** `draft/chat.py`: at most four questions about the real problem, a summary with "Let's build the solution", a model loop with tools when a model is configured, a scripted fallback otherwise, replies streamed as they are written, one turn at a time per draft.
- **Done when:** `tests.test_draft_flow.AgentTests` passes; `make product-e2e` passes `no-data`.

```text
Package 3.7. Write dclab_rnd/draft/chat.py: HomeAgent(store, client, on_request). It asks about the outcome, the
prediction moment, what happens with each prediction and what an error costs, and, when there is no data, whether
the user can share a sample, connect a source or wants simulated data. At most four questions, one at a time.
When data becomes ready it describes it and asks for the outcome column again with columns to choose from: the
source's own outcome column first, then the columns the problem sentence names (compare names only).
With a model: a short loop with the tools ask_user, record, set_pack, propose_workflow, request_data,
simulate_data, get_profile and get_analysis. The model sees column summaries and findings, never rows or cell
values. Its workflow proposal is validated by code. Add ChatClient.stream() to dclab_rnd/intern/llm.py and send
replies as "token" events in chunks; the final "chat" event replaces the live bubble.
Without a model, or after a model error: the same questions from a deterministic script.
Agent turns for one draft must not interleave (the data-ready turn and a user message run in different threads).
```

### 3.8 Synthetic data

- **Delivers:** `draft/synthetic.py`: a validated schema (from a model, or a built-in template the user picks), seeded generation, the label "synthetic" kept from the draft to the exports.
- **Done when:** `tests/test_draft_synthetic.py` passes; `make product-e2e` passes `synthetic`.

```text
Package 3.8. Write dclab_rnd/draft/synthetic.py: a pydantic SyntheticSpec (columns with types and
distributions, a target with a type, a rate and effects), generate(spec) with a seed and a row cap, and built-in
templates. With a model, it designs the spec from the description as JSON; code validates it, retries once, then
falls back to a template. Without a model the user picks a template in the Synthetic tab, and the chat says the
rows come from a template and do not follow the description. The model receives the problem and the answers,
never data. The asset is stored with synthetic: true, its spec and its outcome column, and the project, the
notebook, the report and the brief all say the data is synthetic.
```

### 3.9 The Home page

- **Delivers:** the composer that starts a draft and keeps the user on Home; "Connect data" and the "Bring the data" card (Upload, Studied samples, Kaggle, Hugging Face, Warehouse, Synthetic); the chat thread with pipeline progress and analysis cards; the solution workflow card; the optional pack picker; the Projects and Activity tabs and the "Needs you" list on real data.
- **Files:** `web/src/views/home.html`, `web/src/core.js` (`DC.graph`, `DC.connectors`, `DC.synthetic`).
- **Done when:** in the browser, one sentence plus an uploaded file leads to cleaning steps, an analysis card, a question with the right outcome column first, and a redrawn workflow, with no console error.

```text
Package 3.9. Wire dclab_rnd/agentic/web/src/views/home.html. "Draft the solution" creates a draft and stays on
Home. "Attach data" becomes "Connect data" and opens a "Bring the data" card under the composer: the wizard's
data-source tabs plus a Synthetic tab. The chat appears at the bottom and reuses the intern's thread markup; it
shows the pipeline's steps as a tool-call block, the analysis as a card built from .stats and the existing
charts, questions as ask-cards whose options answer with one click, and model replies as they are written
(insert streamed text as text, never as HTML). Move the project graph renderer into DC.graph with automatic
layout and draw the solution workflow with it; redraw on every "workflow" event. Keep the optional pack cards.
"Let's build the solution" opens the wizard with the draft. Fill the stats, the Projects and Activity tabs and
"Needs you" from the server. Remember the open draft across reloads.
```

### 3.10 Build the project from the draft

- **Delivers:** `POST /api/drafts/{id}/build`: a project with the cleaned table, the solution, the settings, the pack, the workflow and what the chat established; the solution saved through the workflow validator.
- **Done when:** `ApiTests.test_full_flow_sample_then_build` passes; a draft without data builds a plan-only project.

```text
Package 3.10. Add build_project to dclab_rnd/draft/api.py: create the project, copy the cleaned Parquet file,
attach it, carry over the synthetic flag, the settings (rows, quick mode, folds) and the budget, keep a copy of
the draft's pack, workflow, cleaning log and answers, and save the solution through studio.graph.check and
graph.log so the transition log starts with an allowed set_solution. Building twice returns the same project.
A draft with no data still builds: a plan the user can add data to later.
```

---

## Phase 4 — The wizard

### 4.1 Five steps and the Goal step

- **Delivers:** Goal, Data, Solution draft, Split and budget, Review and start; the "DCLab understood" sheet filled from the chat.
- **Done when:** the wizard opens from Home with the draft loaded and shows five steps.

```text
Package 4.1. In dclab_rnd/agentic/web/src/views/new.html make the wizard five steps: Goal, Data, Solution draft,
Split and budget, Review and start. Domain pack is no longer a step. Keep the stepper and the step markup as they
are. The wizard works on the draft from Home (or creates one). The Goal step shows the problem and a "DCLab
understood" sheet filled from the chat's answers, each line marked as known from the chat or still open.
```

### 4.2 The Data step

- **Delivers:** the draft's data sources with their status, the cleaning log, the small analysis, the "Bring the data" card to add or replace data, and the real profile in the "Data profile" tab.
- **Done when:** data brought on Home appears ready here; data brought here runs the same pipeline with visible progress.

```text
Package 4.2. Wire the wizard's Data step: list the draft's assets with rows, columns, how the file was read and
a "synthetic" pill where it applies; show the cleaning log and the summary of the analysis; offer the same
"Bring the data" tabs as Home (ids prefixed wz-), which run the same pipeline here and refresh from the event
stream; fill the "Data profile" tab with the real profile. "Next" stays disabled until a table is ready.
```

### 4.3 The Solution draft step

- **Delivers:** the proposal (target, task, prediction moment, forbidden columns with proof, identifiers, time and group column, metric), the detected pack as changeable cards, the solution workflow, and Accept.
- **Done when:** the untouched defaults save without an error for a table with a unique timestamp; an empty prediction moment is refused with a clear message.

```text
Package 4.3. Wire the wizard's Solution draft step to POST /api/drafts/{id}/solution/proposal and PUT
/api/drafts/{id}/solution. Preselect the outcome from the chat, then the source's own outcome, then the first
candidate. Show the column audit's flagged columns as ticked "forbidden" lines with their reason and proof chips.
A real prediction moment (from the chat or a studied sample) is a value; guidance is shown beside an empty box
and is never saved as the moment. A timestamp that is only "unique" is the time key, not a forbidden candidate.
The task is detected from the target; the Task box applies only when the user asks for the audit again. Show the
detected pack with the pack cards so it can be changed, and the solution workflow. Accept saves and moves on.
```

### 4.4 Split and budget, Review and start

- **Delivers:** the unchanged controls, now real: rows, quick mode and folds reach the engine; the split shown is the one the solution implies; the review summarises what will run; "I drive" opens the project, "Intern drives" starts an intern session.
- **Done when:** `tests.test_draft_flow` passes (folds reach the cross-validation; the stored split follows the solution); `make product-e2e` shows `StratifiedKFold(5` and `TimeSeriesSplit(5)`.

```text
Package 4.4. Wire Split and budget and Review and start without changing how they look. PUT
/api/drafts/{id}/settings stores rows, quick mode, folds, where it runs and the budget. The solution decides the
split (a time column, then a group column, otherwise stratified): show that choice, disable the others and say
how to change it. The folds must reach studio.engine's cross-validation and the exported notebook. Only "this
machine" can be chosen until sandboxes exist; say so. The review lists the problem, pack, data, solution, split
and budget as stored. "I drive" builds the project and opens the Workflow page; "Intern drives" builds it and
starts an intern session with the budget.
```

---

## Phase 5 — The project pages on real data

### 5.1 Workflow page

- **Delivers:** the graph, the log and the validator from the server; gate approvals; the project switcher with real projects; run a stage or all stages.
- **Done when:** after `make product-e2e`, opening `#project` shows every stage completed and a transition log of allowed moves.

```text
Package 5.1. Wire dclab_rnd/agentic/web/src/views/project.html to GET /api/projects/{id}, GET .../graph,
POST .../graph/check, the transitions and the approvals. Keep the layout code on the page. The switcher lists
real projects before the samples. Add "run this stage" and "run all" with progress, and DC.currentProject so
every project page opens the same project. With no project, keep the sample and the "Sample data" pill.
```

### 5.2 Solution & data page

- **Delivers:** the saved solution, the data and the column audit; an editor that saves through the validator and asks before clearing completed stages.
- **Done when:** editing the target saves, logs `set_solution allowed`, and works for a project built from an upload.

```text
Package 5.2. Wire dclab_rnd/agentic/web/src/views/solution.html to the open project. Add "Edit the solution" in
the page's drawer: it saves with PUT /api/projects/{id}/solution, fetches the task and metric from the proposal
when the target changes, keeps the user's edits when they ask for the audit again, and asks for confirmation
when completed stages would be cleared. The proposal route must work for projects whose suggestion holds only
what the chat established.
```

### 5.3 Notebook, Leakage & features, Models, Reliability, Decision brief

- **Delivers:** each page reads the stage records of the open project; the exports work.
- **Done when:** `tests/test_project_review.py` and `tests/test_studio.py` pass; the exported notebook reads the project's data file by its type and uses the recorded folds.

```text
Package 5.3. Wire notebook.html, audit.html, models.html, reliability.html and brief.html to the open project's
stage records, notes and exports (use the read-only GET /api/projects/{id}/review; the notebook export logs a
move, so do not call it to display a page). Every number on these pages comes from a record, with its interval
where the record has one. A page with no completed stage says what to run next. Synthetic data is labelled on the
brief and in both exports. Never call a model production-ready.
```

### 5.4 Intern page

- **Delivers:** intern sessions from the server, live; the standard plan when no model is configured.
- **Done when:** `tests/test_intern.py` passes; "Intern drives" from the wizard lands on a running session.

```text
Package 5.4. Wire dclab_rnd/agentic/web/src/views/intern.html to /api/intern/sessions and its live stream:
the session list, the steps with their tool calls, the budget used, and failures shown as failures. Without a
model the intern runs its standard plan; say which mode is running.
```

### 5.5 Evidence library

- **Delivers:** the live evidence index for every chip, drawer and search; an Ask tab answered from the index.
- **Done when:** `tests/test_pages_evidence.py` passes; the page shows the same number of records as `evidence/knowledge/rag/records.jsonl` has lines.

```text
Package 5.5. Add dclab_rnd/agentic/pages/evidence.py: GET /api/evidence returns the index as the page needs it;
POST /api/evidence/ask searches it and, with a model, keeps only the sentences of the answer that cite a
retrieved record and whose numbers appear in that record; without a model it returns the closest records and
says when the matches are too weak to answer. Load the live records in core.js after the first paint.
```

### 5.6 Workspace and platform pages

- **Delivers:** Compute & jobs, Home's Activity tab, Research lab, Benchmark, Policy model, Domain packs, Integrations and Admin on real data, with plain empty states for what does not exist yet.
- **Done when:** `tests/test_pages_ops.py`, `test_pages_lab.py`, `test_pages_learn.py` and `test_pages_platform.py` pass.

```text
Package 5.6. One route module per group in dclab_rnd/agentic/pages (ops, lab, learn, platform), each with
register(app, ctx), reading only files the product already writes. Compute lists real stage runs, intern
sessions, data pipelines and research runs. Lab and Benchmark read the registry and the campaigns. Policy model
reads the SFT corpus and the transition logs. Domain packs counts real usage. Integrations lists the real tools,
routes and connectors. Admin shows the invariants, the gate switches, the limits, what a model may see, and an
audit log of every validated move, approval and switch change. Where nothing exists (GPU jobs, spend, a review
queue, training runs) the page says so. No invented number anywhere.
```

---

## Phase 6 — Connectors

### 6.1 Kaggle and Hugging Face

- **Delivers:** search and import from Kaggle; import from the Hugging Face Hub (dataset id, revision, config, split); licence, version and SHA-256 recorded.
- **Done when:** the Kaggle and Hugging Face tests in `tests/test_connectors.py` pass (they use fakes). Not yet run against the live services: see 8.2.

```text
Package 6.1. Write dclab_rnd/connectors/kaggle.py and hf.py. Kaggle uses KAGGLE_USERNAME and KAGGLE_KEY from the
server's environment; Hugging Face uses huggingface_hub with an optional HF_TOKEN. Each import downloads one
table into the draft's data folder with a size cap, records the source, licence, version or revision and the
file's SHA-256, registers an asset and runs the same pipeline as an upload. Errors become plain messages; a
refused credential never echoes the key. Add the search and import routes and wire the tabs on Home and in the
wizard. Test with fakes; never call the live services from tests.
```

### 6.2 Databases and cloud storage

- **Delivers:** read-only table or query import through a connection the server defines (`DCLAB_DB_<NAME>`); one object from `s3://` or `gs://`.
- **Done when:** the database and cloud tests in `tests/test_connectors.py` pass; the draft JSON holds the connection's name, never its URL.

```text
Package 6.2. Write dclab_rnd/connectors/db.py and cloud.py. A database connection is a SQLAlchemy URL in the
server's environment as DCLAB_DB_<NAME>; the page only ever names it. Allow a table pick or one read-only SELECT,
row-capped, and refuse anything that writes. cloud.py reads one CSV or Parquet object from s3:// (boto3) or gs://
(google-cloud-storage or gcsfs) with a size cap. Both produce the same asset record and go through the pipeline.
The connection string and any credential must never reach a draft, an event, a log line or an error message.
```

### 6.3 Connector status, credentials tests, requirements

- **Delivers:** `GET /api/connectors` (what is configured, by name only), the tests that no secret leaks, and the libraries in `requirements/base.txt`.
- **Done when:** `tests/test_connectors.py` passes; `requirements/base.txt` lists pyarrow, openpyxl, huggingface-hub, SQLAlchemy, boto3 and google-cloud-storage.

```text
Package 6.3. Add connectors.status() and GET /api/connectors: which sources are configured and which library is
missing, without any secret. Import each connector library lazily so the product starts without it and the page
names the missing one. List the libraries in requirements/base.txt. Test that keys, tokens and connection
strings never appear in drafts, events, responses or error messages.
```

---

## Phase 7 — Verification

### 7.1 End-to-end flows

- **Delivers:** `scripts/product_e2e.py` and `make product-e2e`: upload (the Telco CSV), no data, synthetic, log file, all without a model.
- **Done when:** `make product-e2e` ends with "all flows passed".

```text
Package 7.1. Run make product-e2e. For every flow that fails, find the cause in the product (not in the script),
fix it and add a unit test that would have caught it. Then drive the same flows in the browser once: Home, the
chat, the wizard's five steps with their untouched defaults, the Workflow page. Report each defect you found.
The scores the script prints show that the flow runs; they are not results (quick mode, one holdout).
```

### 7.2 Visual, console and phone-width checks

- **Delivers:** every page compared with demo v1 at 1440 px in light and dark mode; no console error on any of the 19 pages, on a workspace with projects and on an empty one; no sideways scroll at 375 px.
- **Done when:** the three checks are reported page by page.

```text
Package 7.2. Start the product on a spare port twice: once with a workspace that has projects (run make
product-e2e against it with --base) and once with an empty one. Load all 19 pages on both and report any console
or Content-Security-Policy error. Compare Home, the wizard and the project pages with demo v1 at 1440 px in light
and dark mode: only the agreed additions may differ. At 375 px no page may scroll sideways. Fix what you find
without changing the design.
```

### 7.3 Security checks

- **Delivers:** the strict CSP, the CSRF token on every write, the host check, sanitised upload names, and no secret in any draft, event or response.
- **Done when:** `tests/test_web_build.py`, `tests/test_agentic.py`, `tests/test_connectors.py` and `tests/test_draft_flow.py` pass.

```text
Package 7.3. Confirm with tests, and add any that are missing: the CSP names no outside host and allows no
inline code; every write route refuses a request without the CSRF token; a request with another Host header is
refused; an upload's file name cannot leave the draft's data folder; no key, token or connection string appears
in a draft, an event, a response or an error message; what a model may read about the data is summaries and
findings, never rows or cell values (the Home agent), and the Admin page states this accurately.
```

---

## 8. Open packages (they need the owner)

### 8.1 Run every model path against a real model

No path that uses a model has run against a real one. The tests use scripted clients, as the plan requires. Open
paths: the Home agent's loop and its streamed replies, the parse pattern for unusual files, the synthetic schema, the
workflow proposal, the Evidence library's answers, the intern.

- **Tried on 2026-10-05:** the OpenAI account behind the key in `.env` has no credit (`insufficient_quota`), so all 7 requests were refused and no tokens were used. The fallbacks worked. A run against a small local model (`qwen2.5-coder:1.5b` in Ollama) showed that streaming works (34 chunks, none dropped) and exposed one defect: **a model that answers in prose and never calls a tool leaves the draft with nothing recorded, no question and no summary** (package 11.4).
- **Needs:** credit on the account, or another OpenAI-compatible endpoint in `.env`. It sends the problem sentence, the answers and column summaries to the model's provider.
- **Done when:** each path has run once on synthetic data, the validators accepted or correctly refused what the model proposed, and anything that broke has a fix and a test.

```text
Package 8.1. I allow a live check with the configured model, on synthetic data only. Start the product with the
key from .env. Create a draft, generate synthetic data from a description (the model designs the table), hold a
short conversation, and watch the replies stream. Upload an unusual log file so the model must propose a parse
pattern. Ask the Evidence library one covered and one uncovered question. Run one intern session with a small
budget. For each path report what the model proposed, what the code accepted or refused, and any defect. Fix
defects with a scripted-client test. Report the number of requests and tokens used. Send no real data.
```

### 8.2 Run Kaggle and Hugging Face against the live services

- **Needs:** a Kaggle key in `.env` (`KAGGLE_USERNAME`, `KAGGLE_KEY`); Hugging Face works without a token for public datasets. Each import downloads a file, so the owner approves the dataset and its size first.
- **Done when:** one small public dataset from each service is imported through the page, with its licence and SHA-256 recorded.

```text
Package 8.2. With my approval for each download, import one small public dataset from Hugging Face and one from
Kaggle through the "Bring the data" card. Tell me the dataset, its licence and its size before downloading.
Check the asset's recorded source, licence, version or revision and SHA-256, and that no credential appears in
the draft. Fix what the live services do differently from the fakes in tests/test_connectors.py.
```

### 8.3 Remove the old UI at `/classic`

The old UI can still start LLM research campaigns (`POST /api/runs`), which the new Research lab page only displays.
Removing `/classic` before that exists in the new UI would remove a feature.

- **Needs:** a decision: build the campaign designer in the new UI first, or drop campaign starts from the web UI (they stay available from the command line).

```text
Package 8.3. List what /classic can do that the product frontend cannot (start with starting and resuming
research campaigns). For each item either build it in the product frontend with existing components or confirm
with me that it is dropped. Then remove the /classic route and the old files under dclab_rnd/agentic/static
(css, js, index.html), keep static/app, and update the tests and CLAUDE.md.
```

### 8.4 Meter the euro budget

The wizard stores a call, minute and euro budget. The intern enforces calls and minutes. Nothing meters money, so
the euro cap is not enforced and the pages say so.

- **Needs:** a decision on what counts as spend (model tokens at a price list, compute time at a rate, or both).

```text
Package 8.4. Design and build spend metering: record tokens per model request and seconds per stage run with the
project, turn them into euros with a price table kept in one place, show the running total on Compute and Admin,
and stop a run that would pass the project's cap with a clear message. Until a price is configured, keep saying
that money is not metered.
```

---

## Where the build differs from the plan's text

- **`draft/pipeline.py`** does the work the plan called `ingest.py`.
- **Kaggle** is reached through its REST API with the standard library, not the `kaggle` package; **Hugging Face** uses `huggingface_hub` without the `datasets` package. Fewer dependencies, same result.
- **Google Cloud Storage** uses `google-cloud-storage`, or `gcsfs` when that is what is installed.
- **The top-bar pill** on a wired page reads "Your data" instead of disappearing.
- **The split** is not a free choice in the wizard. The solution decides it, because a random split on rows that have a declared time column answers a different question (PIT-005, DCLAB-R02).
- **The Home agent** receives column summaries in its state and can page through the rest with `get_profile`; it never receives rows or cell values.
- **Beyond the plan's first version:** package 5.6 wired the pages the plan left for later.

---

# Part 2 — The software foundation

**What exists today, plainly.** One FastAPI process on `127.0.0.1`. Projects, drafts and intern sessions are JSON
files and Parquet tables in a workspace folder; the research runs use one SQLite file. Jobs are tasks inside the
server process and are lost when it restarts. There are no accounts: whoever opens the page is the owner. Each
feature builds its own model client. There is no container, no deployment and no monitoring.

That is a sound design for one person on one laptop. It does not support several people, a server, or work that
must survive a restart. Part 2 builds that, without changing the UI or the science.

**Decided by the owner on 2026-10-05:**

| Decision | Choice | What it means for the packages |
|---|---|---|
| Who uses it | One team of about 1,000 users | Accounts, roles and workspaces are required (10.2). Several app instances run behind a load balancer, so nothing may live only in one process: jobs, locks, live events and the model pause all move to the database (10.3, 10.5, 11.1). Every list is paged and indexed. Sign-in through the company's identity provider (OIDC) is the default. |
| Database | PostgreSQL only | No SQLite path. Tests run against a real PostgreSQL (a local server or a container). A connection pool is required at this size (9.2). |
| Where it runs | Locally during development, with Docker as an option; a cloud service after development | `make dev` runs the backend, the worker and the frontend on the developer's machine against a local PostgreSQL (12.0). Docker Compose gives the same thing in containers (12.1). The cloud deployment is its own package (12.6) and uses managed PostgreSQL and object storage. |

| # | Package | Status |
|---|---|---|
| 9.1 | A storage interface in front of the three stores | Done (`dclab_rnd/storage`, `tests/test_storage_interface.py`) |
| 9.2 | The database schema and migrations | Done (`dclab_rnd/storage/{models,db,postgres}.py`, `migrations/`) |
| 9.3 | File storage for tables and artifacts | Done (`dclab_rnd/storage/files.py`, `tests/test_file_storage.py`): put, open, path, exists, delete; local folder, or S3-compatible with `DCLAB_FILES_URL=s3://bucket/prefix` (boto3 lazily, credentials from its default chain only); each file recorded with key (its path under the workspace folder), size, SHA-256, content type and backend (`.dclab_files.json`, or the `stored_file` table, migration 0008); project data, draft uploads and cleaned tables, and the intern's exports are recorded; every read of a project's table checks the hash (the engine, the Solution routes, the intern's tools); a workspace folder carries its identity in `.dclab_workspace`, so a moved folder is the same PostgreSQL workspace. Open: draft reads (the pipeline's `clean.parquet`) and connectors' downloads are recorded but not checked on read; the HTTP exports are built in memory, not stored; a workspace moved before its marker existed gets a new id (run it once in place first); a copied workspace folder shares the original's PostgreSQL rows (its identity travels with it), so copy only to move |
| 9.4 | Move an existing workspace into the database | Done (`dclab_rnd/storage/migrate.py`, `python -m dclab_rnd.storage migrate SOURCE --to TARGET`, `tests/test_storage_migrate.py`): copies projects (document, stage records, transitions, activity), drafts with their events, intern sessions, the data and export files through 9.3's file storage (the bucket when DCLAB_FILES_URL names one), and the agent traces, model usage, output checks, lessons, routing and shadow log, with their own ids and timestamps (lists keep their order); the folders may not contain each other and the source is only read; nothing the target holds is copied again (a file the database workspace changed since is kept; twice duplicates nothing; an id another workspace owns, lessons included, is a reported conflict); each project, draft and session is one transaction, and a failure is reported by type and id while the run goes on; then every count, log and file is verified (fewer in the database is a difference, more is a note) and a report printed (exit 1 on a difference, a failure or a conflict). Checked on a workspace built by the four product-e2e flows (3 projects, 4 drafts, 9 files, verified; a second run copied nothing) and in the test, where every page lists the same items in the same order from the files and from the database. Open: an append-only log is copied only into a workspace that has none of it (a log partly copied by an interrupted run must be emptied first; verify says so); the legacy research campaign runs (`research.sqlite3`, the Jobs page) are not moved; a source document that is not valid JSON stops the run before anything is copied, and the message names the error, not the file; one log table that fails skips the logs after it for that run (reported, and verify names the missing rows); a one-way move, not a sync |
| 10.1 | Settings, routers and typed request and response models | Open |
| 10.2 | Accounts, workspaces and roles | Open |
| 10.3 | Durable jobs and workers | Open |
| 10.4 | One audit trail in the database | Open |
| 10.5 | Live events and locks that work across several app instances | Open |
| 10.6 | Rate limits and per-user quotas | Open |
| 11.1 | The model gateway | Done by A1.1–A1.3 of `AGENTIC_FOUNDATION_PLAN.md` (`dclab_rnd/models/`): no module outside `dclab_rnd/models/` creates a model client; tiers, purposes, retries, prices, project and workspace caps, the usage log; A6.4 added routing and shadow |
| 11.2 | One agent runtime and tool registry | Done by A2.1–A2.4 (`dclab_rnd/agents/`): one registry for the intern, `/mcp` and the Home agent, the validator in front of every write, traces and replay |
| 11.3 | An evaluation suite for the agents | Done by A4.1–A4.3 (`dclab_rnd/agent_eval/`, `make agent-eval`, `make agent-eval-live`): planted traps, scripted runs in rd-check, live runs behind a flag and caps, the prompt gate |
| 11.4 | Progress with a model that does not call tools | Done |
| 12.0 | Run everything locally with one command | Open |
| 12.1 | Containers | Open |
| 12.2 | Configuration and secrets | Open |
| 12.3 | Continuous integration | Open |
| 12.4 | Logs, health and metrics | Open |
| 12.5 | Backups and retention | Open |
| 12.6 | Cloud deployment | Open (after development) |
| 13.1 | A typed API client and consistent loading and error states | Open |
| 13.2 | Browser tests of the main flows | Open |

## 9. Database

### 9.1 A storage interface in front of the three stores

- **Delivers:** one small interface each for projects, drafts and intern sessions, with today's file stores as the first implementation. Nothing else changes.
- **Done when:** every route and module reaches storage only through the interface; `make rd-check` and `make product-e2e` pass unchanged.

```text
Package 9.1. ProjectStore (dclab_rnd/studio/store.py), DraftStore (dclab_rnd/draft/store.py) and SessionStore
(dclab_rnd/intern) are used directly all over the code. Define a Protocol for each in dclab_rnd/storage/ with
exactly the methods callers use today (get, save, update under a lock, list, delete, the append-only logs, the
data folder). Make the file stores implement them and make every caller depend on the Protocol. Change no
behaviour: make rd-check and make product-e2e must pass without edits to their expectations.
```

### 9.2 The database schema and migrations

- **Delivers:** SQLAlchemy models and Alembic migrations: workspaces, users, projects, drafts, stage records, transitions, approvals, events, intern sessions, jobs, model requests; a database implementation of the 9.1 interfaces.
- **Done when:** the whole test suite and `make product-e2e` pass against PostgreSQL; the projects list stays fast with 10,000 projects (measured, with the query plan).

```text
Package 9.2. Implement the storage Protocols from 9.1 on a database with SQLAlchemy 2 and Alembic. Tables:
workspace, user, membership, project, draft, asset, stage_record, transition, approval, event, intern_session,
job, model_request. Keep documents that are read whole (a stage record, a solution, a workflow) as JSON columns;
make columns of what is filtered or joined (ids, workspace, status, timestamps, kind). Append-only logs
(transitions, events) get a sequence per parent and are never updated. An immutable stage record stays immutable.
PostgreSQL only: DCLAB_DATABASE_URL points at it, with a connection pool sized for several app instances (and
PgBouncer in front in production). Index every column a list filters or sorts on and page every list: one team
of about 1,000 users will hold thousands of projects. Tests run against a real PostgreSQL in a temporary
database, never a mock. No raw SQL built from request input.
```

**What 9.2 built, and what it left for later packages.**
- Ten tables: `workspace`, `app_user`, `membership` (ready for 10.2), `project`, `stage_record`, `activity`, `transition`, `draft`, `draft_event`, `intern_session`. A document read whole is JSONB; what a list filters or sorts on is a column with an index.
- `DCLAB_DATABASE_URL` switches the whole product to PostgreSQL (`dclab_rnd.storage.open_stores`); without it the file stores run, so both stay tested. One folder (`DCLAB_AGENT_HOME`) is one workspace until accounts arrive.
- Row locks make `update` safe across app instances; draft event numbers come from an UPDATE of the draft row; an agent turn is a PostgreSQL advisory lock.
- Measured: with 10,000 projects, a page of 50 takes about 1 ms and uses the workspace index; reading one project takes 0.5 ms.
- The whole suite (343 tests) and `make product-e2e` pass on both backends with the same scores.
- **Not done yet:** the `/api/projects` route still returns every project of the workspace (package 10.1 gives it paging); tables and exports are still files (9.3); old workspaces are not imported (9.4); the research-run store (`agentic/store.py`) is still its own SQLite file.

### 9.3 File storage for tables and artifacts

- **Delivers:** uploaded files, cleaned Parquet tables and exports behind one interface: a local folder now, S3-compatible storage later; the database holds the path, size and SHA-256.
- **Done when:** a project's data can be read after the workspace folder is moved; the hash is checked on read.

```text
Package 9.3. Tables and artifacts do not belong in the database. Add dclab_rnd/storage/files.py with put, open,
delete and exists, a local-folder implementation and an S3-compatible one (boto3, lazily imported). Record each
file in the database with its key, size, SHA-256 and content type, and verify the hash when a stage loads data.
Keep the sanitised-name and size-limit rules. Never store credentials in a file record.
```

### 9.4 Move an existing workspace into the database

- **Delivers:** `python -m dclab_rnd.storage migrate`, which copies a file workspace into the database and verifies it.
- **Done when:** a workspace created by `make product-e2e --base` migrates, and every page shows the same projects, records and logs.

```text
Package 9.4. Write a one-way, repeatable migration from a file workspace to the database: projects with their
solution, stage records, transitions, approvals and activity; drafts with their events; intern sessions. Copy
files through the 9.3 interface. Verify counts and the SHA-256 of every data file, print a report, and change
nothing in the source folder. Running it twice must not duplicate anything.
```

## 10. Backend

### 10.1 Settings, routers and typed models

- **Delivers:** one settings object read once; routes grouped in routers per area; pydantic models for every request and response.
- **Done when:** `server.py` only builds the app; the OpenAPI schema describes every route; a bad request body is a 422 everywhere.

```text
Package 10.1. dclab_rnd/agentic/server.py holds most routes as closures and reads os.environ in many places.
Introduce dclab_rnd/settings.py (pydantic-settings: database URL, workspace folder, model endpoint and tiers,
limits, feature switches) and split the routes into routers (projects, drafts, intern, evidence, pages, admin)
that receive their dependencies through FastAPI Depends. Give every route pydantic request and response models.
Keep every path, status code and payload as it is: the frontend and the tests must not need changes.
```

### 10.2 Accounts, workspaces and roles

- **Delivers:** sign-in, sessions, workspaces with members, and the roles the Admin page already describes (owner, data scientist, reviewer, viewer) enforced on every route; gate approvals record who approved.
- **Done when:** a viewer cannot run a stage or approve a gate; a user never sees another workspace; a test covers every write route.

```text
Package 10.2. Add accounts and workspaces. Sign-in with a password hashed with argon2 and a server-side session
in an HttpOnly, SameSite=Lax cookie; keep the CSRF token on writes. Every project, draft and session belongs to a
workspace and every query filters by it. Enforce roles in one dependency: owner (everything), data scientist
(projects, runs), reviewer (approve gates, read), viewer (read). A gate approval records the user. For a team of
about 1,000, sign-in through the company's identity provider (OIDC) is the main path, with groups mapped to
roles; password sign-in stays for local development. Sessions live in the database, not in process memory.
Write one test per write route that the wrong role and the wrong workspace are refused.
```

### 10.3 Durable jobs and workers

- **Delivers:** stage runs, data pipelines, intern sessions and simulations as rows in a job table, run by a worker process, with progress, cancel, retry and recovery after a restart.
- **Done when:** killing the server mid-run and starting it again resumes or cleanly fails the job; "Stop" works on the Compute page.

```text
Package 10.3. Jobs are asyncio tasks inside the server and die with it. Add a job table (kind, payload, status,
progress, started, finished, error, attempts, cancel requested) and a worker process (python -m dclab_rnd.worker)
that claims jobs with SELECT … FOR UPDATE SKIP LOCKED, so any number of workers can run. Move stage runs,
the draft pipeline, simulations and intern sessions onto it. A job writes progress events the page already
streams. Add cancel (checked between steps) and wire the Compute page's Stop button. On start, a job left
"running" by a dead worker is marked interrupted and can be retried. A stage stays deterministic: same seed, same
record.
```

### 10.4 One audit trail

- **Delivers:** every validated move, approval, policy change, sign-in and data import as one append-only table with who, when and what.
- **Done when:** the Admin audit tab reads one query; an entry cannot be changed or deleted through the application.

```text
Package 10.4. The audit log is assembled from transition and activity files per project. Write every governed
action (validated moves allowed or refused, gate approvals and their use, policy switches, solution changes, data
imports with their source and hash, sign-ins, role changes) to one append-only audit table with the user, the
workspace, the time and a JSON detail. The application has no update or delete on it. Point the Admin audit tab
at it with paging and filters.
```

### 10.5 Live events and locks across several app instances

- **Delivers:** the chat and pipeline streams, the per-draft agent turn lock and the model pause working when requests land on different app instances.
- **Done when:** with two app instances and one worker, a page connected to one instance shows events produced through the other, and two messages to one draft never interleave.

```text
Package 10.5. Several things assume one process: the event stream tails a file, the agent turn lock and the
store locks are threading locks, and the model pause is a module variable. Move them to PostgreSQL: events are
rows with a sequence per draft, the stream wakes on LISTEN/NOTIFY and replays from Last-Event-ID; the per-draft
turn lock is an advisory lock; the model pause is a row with an expiry. Test with two app processes and one
worker against the same database.
```

### 10.6 Rate limits and per-user quotas

- **Delivers:** limits per user and per workspace on requests, uploads, running jobs and model spend, stored in the database and shown on the Admin page.
- **Done when:** a user over a limit gets a 429 with a clear message and the others are unaffected.

```text
Package 10.6. With about 1,000 users one person must not be able to starve the rest. Add limits per user and per
workspace: requests per minute, upload bytes per day, jobs running at once, and model spend per month (from the
gateway's usage log). Enforce them in one dependency and in the job queue, return 429 with what was exceeded and
when it resets, and show usage against limits on the Admin page. Limits are settings with safe defaults.
```

## 11. AI and agents

The full design of the AI and agent layer (principles, architecture, and phases A1 to A6 with a prompt per package)
is in [AGENTIC_FOUNDATION_PLAN.md](AGENTIC_FOUNDATION_PLAN.md). Packages 11.1 to 11.3 below are its phases A1, A2 and
A4 in short form.

### 11.1 The model gateway

- **Delivers:** one place every model request goes through: provider and model per tier (cheap, standard, strong), retries with backoff, the pause after a hard refusal, a per-project and per-workspace budget, and a usage log (tokens, cost, latency, purpose).
- **Done when:** no module creates a model client by itself; the Compute and Admin pages show real spend; a request over budget is refused with a clear message.

```text
Package 11.1. Model clients are created in several places (intern/llm.py, studio/agent.py, the campaign loop).
Build dclab_rnd/models/gateway.py: complete() and stream() take a purpose ("home_agent", "parse_pattern",
"synthetic_schema", "evidence_answer", "intern") and a tier; settings map each tier to an OpenAI-compatible
endpoint and model, so a small local model can serve cheap purposes. The gateway owns retries, timeouts, the
safe failure reasons and the pause (already in intern/llm.py), counts tokens, prices them from one table, checks
the project and workspace budget before sending, and writes a model_request row (purpose, model, tokens, cost,
latency, outcome; never the prompt text unless a debug setting is on). Route every existing caller through it.
This also delivers package 8.4.
```

### 11.2 One agent runtime and tool registry

- **Delivers:** the Home agent, the intern and the campaign agent on one loop: a tool registry with JSON schemas, the validator in front of every tool that changes something, a step and time budget, and a stored trace of each step.
- **Done when:** a new tool is registered once and is available to the intern and over MCP; every agent step can be replayed from its trace.

```text
Package 11.2. There are three agent loops (draft/chat.py, intern/loop.py, agentic/worker.py). Extract one
runtime in dclab_rnd/agents/: a Tool (name, description, JSON schema, handler, whether it changes state), a
registry shared by the intern and the MCP server, and a loop that asks the gateway, validates arguments against
the schema, sends every state-changing tool through studio.graph.check, enforces the step and minute budget, and
stores a trace (state summary, tool, arguments, verdict, result summary). Rebuild the three agents on it with
their behaviour unchanged. The model still never computes a split or a metric.
```

### 11.3 An evaluation suite for the agents

- **Delivers:** scripted cases that every agent must pass (planted leaks, a refused move, a missing prediction moment, an invalid workflow) and an optional live run that scores a real model on the same cases.
- **Done when:** `make agent-eval` reports pass and fail per case with a scripted model; the live run reports valid moves, leaks caught, unsafe actions, tokens and cost.

```text
Package 11.3. Build the judgment suite the Benchmark page describes. Each case is a small table with a planted
trap (a post-outcome column, an identifier that predicts the target, a random split on timestamped rows) and the
expected behaviour. Run the standard plan and any configured model on the cases through the 11.2 runtime and
score: valid moves, leaks caught, citations that exist, unsafe actions attempted, tokens and cost. The scripted
run is part of make rd-check; the live run needs an explicit flag and my permission each time. Report scores
with their uncertainty and never as proof of production readiness.
```

### 11.4 Progress with a model that does not call tools

- **Delivers:** the Home conversation keeps moving when a model answers in prose: the pending question is answered from the user's reply by code, and the next question or the summary follows.
- **Done when:** a scripted model that never calls a tool still leads to a recorded outcome, moment and action, and a summary.

```text
Package 11.4. Found in the live check: a small model replied in prose and never called record or ask_user, so
nothing was recorded, no question was asked and no summary came. In HomeAgent.model_turn, when a turn ends with
no tool call and a question is pending, record the user's reply as that question's answer in code, and when no
question is pending and something is still unknown, ask the next scripted question after the model's text. The
user must always have a next step. Test it with a scripted client that only ever returns text.
```

## 12. Infrastructure

### 12.0 Run everything locally with one command

```text
Package 12.0. Development happens on the developer's machine without containers. Add make dev: it checks that a
local PostgreSQL is reachable (and says how to start one), creates the dclab_dev database and runs the
migrations when needed, then starts the API with reload, one worker, and a watcher that rebuilds the frontend
when a file under dclab_rnd/agentic/web/src changes. Add make db-reset for a clean database and make test-db for
the temporary database the tests use. Document it in the README in ten lines.
```

### 12.1 Containers

```text
Package 12.1. Add a Dockerfile (multi-stage, a non-root user, the built frontend copied in, no key baked in) and
a docker-compose.yml with three services: app, worker and PostgreSQL with a named volume, plus a volume for file
storage. The app listens on 0.0.0.0 only inside the container and is published on 127.0.0.1 by default. Add
make up, make down and make logs. Done when make up starts a product that passes make product-e2e --base.
```

### 12.2 Configuration and secrets

```text
Package 12.2. Document every setting in .env.example with a safe default. Keys and connection strings come from
the environment or Docker secrets, never from a committed file or an image layer. Fail at start with a clear
message when a required setting is missing or the database cannot be reached. Add the allowed host names as a
setting for the host check. Scan the repository and the image for committed secrets and report what you find.
```

### 12.3 Continuous integration

```text
Package 12.3. Extend .github/workflows: on every push run make rd-check and make product-e2e against a PostgreSQL
service container, and build the image on main. No job may use a model key. Cache dependencies. A failing check
blocks the merge.
```

### 12.4 Logs, health and metrics

```text
Package 12.4. Add structured JSON logs with a request id (never a secret, a prompt or a data value), GET /healthz
(process up) and GET /readyz (database and file storage reachable), and counters for requests, job durations,
failures and model usage in Prometheus format on a port that is not public. Show job failures on the Compute page.
```

### 12.5 Backups and retention

```text
Package 12.5. Add make backup and make restore: a database dump plus the file storage, restorable on a clean
machine, with a test that restores into a temporary database and compares counts. Add the retention rule the
Admin page promises: raw uploads can be deleted after N days while the cleaned table, its hash and every record
stay. Deleting a project removes its files and leaves an audit entry.
```

### 12.6 Cloud deployment (after development)

```text
Package 12.6. Deploy the same image to a cloud: managed PostgreSQL with backups and point-in-time recovery,
object storage for tables and artifacts (9.3), two or more app instances behind a load balancer with TLS, one or
more workers that scale with the job queue, secrets in the cloud's secret manager, and logs and metrics in its
monitoring. Write it as infrastructure code (Terraform or the provider's equivalent) with a staging and a
production environment. Ask me which provider and region before writing anything, and never apply a change to a
cloud account without my explicit yes.
```

## 13. Frontend

The UI stays demo v1. These two packages make it sturdier without changing how it looks.

### 13.1 A typed API client and consistent states

```text
Package 13.1. Generate one API client from the server's OpenAPI schema (10.1) into
dclab_rnd/agentic/web/src/api.js and make every view use it instead of path strings. Give every view the same
three states with existing components: loading, empty (what to do next) and failed (what happened, try again).
A signed-out response sends the user to sign-in. Change no layout, colour or wording that already exists.
```

### 13.2 Browser tests of the main flows

```text
Package 13.2. Add Playwright tests that drive the real page: one sentence and an uploaded file on Home, the
chat's questions answered by clicking options, the wizard's five steps with their defaults, a run on the
Workflow page, and the brief. Assert no console or CSP error on all 19 pages, no sideways scroll at 375 px, and
that the stats on screen equal the API's numbers. Run them in CI against the container from 12.1.
```

## Suggested order for Part 2

1. **9.1** and **10.1** (they move code without changing behaviour), then **12.0** (local development on PostgreSQL).
2. **9.2, 9.3, 9.4** (the database), then **10.3** (jobs), **10.4** (audit) and **10.5** (several instances).
3. **10.2** (accounts and roles) and **10.6** (limits): required before more than one person uses it.
4. **11.1** and **11.2** (the AI layer), then **11.3**.
5. **12.1 to 12.5**, **13.1** and **13.2**; **12.6** (cloud) last.


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
| 8.1 | Run every model path against a real model | Done with a local model (2026-10-07; `scripts/model_paths.py`, `make model-paths`); the run with OpenAI (`make model-paths ARGS="--configured --cap-eur 2"`) waits for credit on the account behind the key in `.env` | Every path ran against qwen2.5-coder:1.5b in Ollama on made-up data (14 requests, about 17,400 tokens in and 2,400 out, no cost): the Home chat streamed its replies, asked, and recorded the target, the prediction moment, the action and the data plan; asked to draw the solution's steps, this model never proposed a workflow, so the workflow check has not yet seen a model's proposal (it stays the pack's template; to be seen with OpenAI); the model's synthetic table design was refused by the validator and the code used a template, saying so (the template picked from the words was a poor match: churn for deliveries); its log parse pattern was unusable, so the file was read as plain text and the notes say 20 lines were sent; the evidence answer to a covered question was wrongly "not covered" (the small model's judgment) and the answer to an uncovered one was dropped by the citation check. One defect, fixed with tests: a model that writes its tool calls as text (a ```json fence, a <tools> tag, a bare object) ran nothing yet ended "completed"; in a run where no tool has run the runtime now tells it once (the prompt `tools_as_text.md`) and then stops (`tools_as_text`), the intern fails the session with the reason, and so does an intern first turn answered in prose without any tool call (the live judgment suite still scores both), and the Home agent drops every streamed bubble that was not an answer while its script carries on; a report that quotes a call after real work stays an answer. The intern path runs on a project built from a made-up table; a path whose model requests all failed is reported as a failure, not as a refusal |
| 8.2 | Run Kaggle and Hugging Face against the live services | **Hugging Face done, Kaggle open** (2026-10-07). With the owner's approval of this one download, `scikit-learn/iris` was imported through the draft route the Bring-the-data card calls (`POST /api/drafts/{id}/data/hf`, on a temporary server, no token): licence cc0-1.0, revision main (commit `0bda0ce8…`), one file `Iris.csv`, 5,107 bytes, SHA-256 `600ac44f…` recorded on the asset with its source URL, 150 rows × 6 columns, cleaned and analysed in 3 seconds; no credential appears in the draft. The live service behaved as the fakes in `tests/test_connectors.py` did: nothing needed fixing. Kaggle needs `KAGGLE_USERNAME` and `KAGGLE_KEY` (none is configured), and a dataset to approve | The import went through the API, not by clicking the card |
| 8.3 | Remove the old UI at `/classic` | Done (2026-10-07) | The owner decided: drop the old UI; research campaigns stay on the command line (`python -m dclab_rnd.agentic run|resume`). An inventory of what only the old UI did: starting, pausing and resuming campaigns (now the command line; Ctrl+C pauses a run, and Compute shows a stopped run's resume command), the Knowledge view of campaign lessons (`/api/knowledge` stays for API users), and four things now built in the product with its existing components: deleting a project (Workflow page) and an intern session (Sessions tab), each with a confirm dialog; a link to the field guide `/guide` (Evidence); and each research trial's files (Compute's job detail). Removed: the `/classic` route, its 16 files in `agentic/static/{index.html,css,js}`, its entry among the public page shells, Compute's "Open in the classic UI" link and the stale Playwright script that drove the old page; the docs and texts that pointed to it now name the command line. The `/api/runs` routes stay (the Jobs page's export link, the command line's runs and API users read them); tests check that `/classic` and its files answer 404 and that the research map is on the Lab page |
| 8.4 | Meter the euro budget | Done (built in A1.2, 10.6 and 11.x; recorded 2026-10-07) | The owner decided: model spend at a price list; compute on this machine is not billed. Every model request of the product (Home agent, intern, evidence, parsing, synthetic, explanations) goes through the gateway, which records its tokens with the project (`models/usage.py`), turns them into euros with the one price table (`models/prices.py`, or a file named by `DCLAB_PRICES_FILE`), shows the month's total on Compute and Admin, and refuses before sending a request that would pass the project's wizard budget, an intern session's cap, the workspace's `DCLAB_WORKSPACE_MONTHLY_EUR` or a person's monthly limit, saying which and what is left (the legacy research campaign's own client is stopped only once a cap is reached: `check_external`). Open: no price is configured (none is ever guessed), so until the owner adds the model's checked price to `models/prices.py` or to a JSON file named by `DCLAB_PRICES_FILE` the pages say money is not metered and euro caps cannot stop remote requests; local models cost nothing |

Part 2 (packages 9–13, the software foundation) is done; its table is in Part 2. Part 3 (packages 14.x, proving it works: a frozen benchmark, live model checks, a user study, staging) starts at the end of this file. Part 4 (packages 15.x, rare events with partial labels: what a fraud challenge taught, built so DCLab solves such problems itself) follows it, then Part 5 (packages 16.x: a problem-kind interface and many more kinds of problems, up to solving a whole challenge from its brief) and Part 6 (packages 17.x: understand the problem, analyse the data for decisions, plan, review and report, as a portable core the main product can adopt; built first, after 15.1), and Part 7 (packages 18.x: the agentic layer, model roles that understand, investigate, plan, engineer features, criticise, improve and report, each checked by code and measured before it is on by default).

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
| 10.1 | Settings, routers and typed request and response models | Done (`dclab_rnd/settings.py`, `dclab_rnd/agentic/{services,api_models}.py`, `agentic/routers/`, `tests/test_api_contract.py`): the server's settings are one pydantic-settings object read once per app (the workspace folder, database and file storage, upload limit, the campaign's model, the no-live-models switch); the routes live in routers per area (core, research runs, projects, intern and lessons; the drafts in `draft/api.py`; each page in `agentic/pages`), taking the stores, the gateway and the running jobs through `Depends(services)`; `server.py` only builds the app; every route has request and response models (documenting, never trimming: `extra="allow"` and `exclude_unset`), so the OpenAPI schema describes every route (87 operations; the two old "contract" aliases and the favicon stay hidden) and a body that is not JSON, or not an object, is a 422 everywhere. A write to `/api/` without a JSON content type is still read as JSON (the routes always did; `curl -d` sends a form type), the uploads excepted. A bad value of a mirrored setting (`DCLAB_PORT`, `DCLAB_WORKSPACE_MONTHLY_EUR`) falls back to its default, never stopping the server. Paths, status codes and payloads are unchanged (the suite and the four e2e flows pass as before; a test calls every GET route with real ids and finds no 5xx, and checks that no response gained a key; the one test that grepped `server.py` for the upload limit now checks the limit the route enforces). Open: the model tiers and the database URL are still read by `models.settings` and `storage.db` on their own (per request, for keys and routing, and for command-line tools that build no app); the response models name the main fields, not every nested one |
| 10.2 | Accounts, workspaces and roles | Done. Part A (`dclab_rnd/accounts/`, `dclab_rnd/context.py`, `agentic/{pool,current}.py`, migration 0012, `tests/test_accounts.py`): sign-in only when configured (the owner's choice, 2026-10-07): with `DCLAB_AUTH` unset the app is today's single owner on this machine; `DCLAB_AUTH=password` (or `oidc`, part B) needs PostgreSQL and turns on accounts, server-side sessions (a table; the cookie is HttpOnly and SameSite=Lax, only the token's SHA-256 is stored; 12 hours idle, 7 days at most) with a request token per session on writes, and personal API tokens (Bearer; only their hash stored). Roles are the plan's four, shown with the Admin page's names (Owner, ML engineer, Reviewer, Business viewer); one check on every route, added by the shared Router from the route's method and path, so no route is left out: read for every role, write for ML engineers and owners, a stage's choice also for reviewers, gates and lesson reviews for reviewers and owners, gate switches, the training opt-in and members for owners. One server serves several workspaces: each has its own Services (stores, gateway, lessons, jobs, audit; its files in `<home>/workspaces/<id>/`), chosen per request from the session; the draft routes and the pages reach it through stand-ins. A gate approval records the user (and the audit names them; jobs carry who started them). `python -m dclab_rnd.accounts add --email … --role owner` makes the first owner. Checked by a test that calls every write route as a member whose role may not (403) and as a member of another workspace with this workspace's ids (never a success; nothing changes), and that a viewer cannot run a stage or approve a gate and a user never sees another workspace. With accounts on, every workspace has its worker (in the server, or `python -m dclab_rnd.worker`, which picks up a new workspace within a minute), the page shells are open (they hold no data; every API call they make is checked) and FastAPI's schema and docs pages are not served; removing a member ends their sessions and tokens in that workspace. Open: the owner can add an existing user to their workspace without that user's consent (a team's owner manages its members). Part B done (`dclab_rnd/accounts/oidc.py`, `tests/test_oidc.py`): `DCLAB_AUTH=oidc` signs in through the company's identity provider: OpenID Connect discovery, the authorization code with PKCE, a signed ten-minute cookie for the state, nonce and verifier (`DCLAB_SECRET_KEY`), and the ID token checked against the provider's published keys (signature, issuer, audience, expiry, nonce); `DCLAB_OIDC_ROLES` maps groups to roles (`group=role`, or `group=workspace:role`), the highest role a person's groups give is theirs, and each sign-in sets their memberships in the mapped workspaces from their groups (leaving a group takes the role away); a person none of whose groups is mapped is refused; a user is found by the provider's subject (one account per subject, a unique index), and an existing account is linked by email only when the provider verified that email and the account is not linked to another subject (an email a person can type at the provider is not proof); the membership changes are one transaction, and sign-ins and the roles groups set are in the audit. A mapping that names a workspace that does not exist is refused before anyone is sent to the provider; a provider that is down or answers wrongly gives the refusal page, not an error; the token's issuer is compared exactly as the provider publishes it; behind HTTPS the callback address must be set. Checked against a fake provider: a forged or expired state, a token for another client, a signature by another key, an unverified email that names an existing account and a second identity with the same verified email are refused; leaving a group takes the role away. Open: a group change reaches DCLab at the person's next sign-in (sessions last 12 hours idle); the provider's keys are fetched at each sign-in (no cache); a sign-in that removes a workspace's last owner is allowed (the groups decide). Part C done: a sign-in page (password, or "Sign in with your company account") shown before anything of the workspace loads and again when a session ends; the workspace switcher, the avatar and Home's greeting name the signed-in person and their workspace; the Admin page has a Members tab (the owner adds members, changes roles, removes them), an API tokens tab (made, shown once, revoked) and the roles table marked enforced; a project's team lists the workspace's members; /mcp takes an API token and acts in that token's workspace with its role (a viewer's token is refused). Without accounts nothing of this shows: the switcher says "This machine · one owner" |
| 10.3 | Durable jobs and workers | Done (`dclab_rnd/jobs/` store, worker, handlers, live view; `dclab_rnd/worker.py`; `agentic/routers/jobs.py`; migration 0009; `tests/test_jobs.py`): stage runs, draft data pipelines, synthetic data and intern turns are rows in one job table (kind, key, payload, status, progress, created/started/finished, heartbeat, worker, error, attempts, cancel requested), in `jobs.json` or the `job` table, with one queued or running job per key (a unique index, so two app instances cannot start the same project's stages); a worker claims with `SELECT … FOR UPDATE SKIP LOCKED` (two workers never run the same job, tested) and runs up to `DCLAB_WORKER_THREADS` at a time; the server runs one itself unless `DCLAB_WORKER=external`, and `python -m dclab_rnd.worker` runs one on its own (`make dev` starts it, with reload); `?wait=true` claims the job at once and runs it in the request, as before. A job calls `checkpoint()` between its steps (between stages, the pipeline's steps, the intern's tool calls, synthetic design and generation): Stop on the Compute page (`POST /api/jobs/{id}/stop`) ends it there as cancelled, the step it is on finishes (a stage stays deterministic), and its domain is put back (stages pending, the asset failed with the reason, the session stopped); a worker beats every 5 s, and a running job whose heartbeat is older than `DCLAB_JOB_STALE_SECONDS` (120), or whose worker was a process of this machine that has ended (a server killed and started again at once), is marked interrupted on start or by any worker (only while its heartbeat is still stale, and a worker whose job was taken stops at its next checkpoint and writes nothing more), its domain settled (the stage it was on failed, the rest pending), and Retry queues it again after the checks a new request passes. Checked by hand on files and PostgreSQL: a server killed with SIGKILL during a stage came back with the job interrupted and the stage failed, and Retry ran all five stages; two API instances and one worker process on one database: the second instance refused the run the first had queued, and the worker process ran it. Job responses leave out the free text a payload carries (a message, a description for a model); a request that loses the race for a project to another instance puts its stages back. `make dev` runs the worker under `watchfiles` when installed (without reload otherwise) and says loudly when it stops. Open: the Home agent's turns stay asyncio tasks in the instance that took the message (10.5 moves the turn lock); a stop waits for the step it is on (a long stage is not cut); a retried run re-runs the stages it was asked for, completed ones included; a worker serves the one workspace its folder names (accounts and workspaces come with 10.2); the file store is for one machine; jobs are not copied by `storage migrate` (they are transient); the legacy research campaign keeps its own runs |
| 10.4 | One audit trail in the database | Done (`dclab_rnd/audit.py`, migration 0010, `tests/test_audit.py`, `tests/test_pages_platform.py`): one append-only log per workspace (`audit.jsonl`, or the `audit_event` table, whose trigger refuses UPDATE and DELETE; no cascade from projects, so the audit outlives what it records) with when, who (actor, the name shown, a user id once accounts exist), the project or draft, the move, the outcome and a JSON detail. Written where each governed action happens: every validated move allowed, refused or waiting (`graph.log`, so the engine, the intern's guard, the person's routes and `build_project` all write it; the Home agent's refused draft moves too, without the model's arguments), gate approvals and their use by the final stage, gate switches and the training opt-in, model routing changes, solutions saved (project, draft, intern, wizard; with the solution hash), data imports with their source and SHA-256 (a project's table from an upload, a sample or a draft; a draft's upload, sample, connector or synthetic file), lesson reviews and memory-note removals. The Admin audit tab reads one query, with paging and filters (kind, outcome, people or agents, project). A workspace from before 10.4 has its history (moves, approvals, switches, routing) imported once, on the first start, under a lock; `storage migrate` copies and verifies the audit like the other logs. Open: sign-ins and role changes are reserved kinds until accounts exist (10.2), and `who` is the declared role until then; writing the audit is best effort for the action (a failure is reported on stderr, the action stands); the per-project transition and activity logs stay (trajectories, the ops page and the agents read them); a store made outside a workspace (a script, the agent eval's scratch projects) writes no audit; the earlier history is imported only into an empty audit, so a command-line tool that writes the first row on a pre-10.4 workspace skips it; on files the Admin tab reads the whole log per request |
| 10.5 | Live events and locks that work across several app instances | Done (`dclab_rnd/storage/notify.py`, `dclab_rnd/models/pause.py`, migration 0011, `tests/test_multi_instance.py`): on PostgreSQL a draft's events were already rows numbered per draft (9.2), the agent turn lock an advisory lock and the store writes row-locked; now each event also sends `pg_notify` in the transaction that writes it, and one listener thread per server process wakes the streams waiting on that draft (they still read the rows and replay from Last-Event-ID; a missed notice only delays a stream to its 2-second poll; on files the stream polls as before). "The agent is still answering" (409) asks the turn lock, which every instance shares, as well as this instance's own queued turn (two messages at once to one instance: one is refused), so a message sent to another instance is refused or queued the same way; a turn's user message and its answer are written under the lock, so two messages never interleave. The model pause after a refusal that stays is a `model_pause` row with an expiry, read before each request by every instance and worker; its key is the endpoint, the model and a hash of the API key (a new key is not paused by the old one's refusal). Checked by a test with two server processes and one worker process on one database: a stream on B shows the agent's turns and a message sent through A, and the pipeline events of a file A queued and the worker ran; messages sent at once to A and B are answered one after the other. Open: a message sent while another instance's turn is about to take the lock can pass the check and then wait for that turn (it is not refused, but it never interleaves); the file stores stay one-machine (their locks are a thread lock and a file lock); the listener needs a direct connection to PostgreSQL (behind a pooler in transaction mode LISTEN delivers nothing and streams fall back to the 2-second poll), and psycopg 3.2 or later |
| 10.6 | Rate limits and per-user quotas | Done (`dclab_rnd/limits.py`, migration 0014, `tests/test_limits.py`): with accounts on, four limits per person and per workspace, settings with safe defaults (0 switches one off; a bad value keeps the default): API requests a minute (600 / 3000), bytes brought in a day by uploads and connector imports (2 GB / 10 GB), jobs queued or running at once (4 / 16) and model spend a month on priced models (€25 a person; the workspace's is the gateway's monthly cap). Requests and bytes are counters in PostgreSQL (`usage_counter`, a fixed window, incremented in one statement, so every instance counts the same; a refused request is not counted); jobs are counted from the job table when one is queued; model spend from the usage log, which now records the person each request was for (the gateway holds an estimate per person as it does per project). Over the requests, bytes or jobs limit is a 429 that says which limit, whose (yours or the workspace's), how much was allowed and used, and when it resets (Retry-After), shown on the page in words; the others are unaffected. A job is refused before anything is written (a stage left pending, no draft asset stored, no bytes charged) and counted under a lock per workspace, so requests at once never take more than the limit; a retry counts like a new job. Over the model budget, the request is refused by the gateway with the same words ("you have used your monthly model budget"), which the agent shows in its turn; a person's spend counts every workspace they work in. The Admin Budgets pane shows each limit with what you and the workspace have used. Open: without accounts (one owner on a machine) nothing is limited; a fixed window lets a burst of up to twice a limit straddle a minute's edge; unpriced models are not counted in euros; a person's in-flight model estimate is held per workspace (their spent euros count every workspace); old counter windows are deleted now and then (two days kept) |
| 11.1 | The model gateway | Done by A1.1–A1.3 of `AGENTIC_FOUNDATION_PLAN.md` (`dclab_rnd/models/`): no module outside `dclab_rnd/models/` creates a model client; tiers, purposes, retries, prices, project and workspace caps, the usage log; A6.4 added routing and shadow |
| 11.2 | One agent runtime and tool registry | Done by A2.1–A2.4 (`dclab_rnd/agents/`): one registry for the intern, `/mcp` and the Home agent, the validator in front of every write, traces and replay |
| 11.3 | An evaluation suite for the agents | Done by A4.1–A4.3 (`dclab_rnd/agent_eval/`, `make agent-eval`, `make agent-eval-live`): planted traps, scripted runs in rd-check, live runs behind a flag and caps, the prompt gate |
| 11.4 | Progress with a model that does not call tools | Done |
| 12.0 | Run everything locally with one command | Done (`scripts/dev.py`, `make dev`, `make db-reset CONFIRM=yes`, `make test-db`, README "Develop locally", `tests/test_dev_script.py`): `make dev` checks that a local PostgreSQL answers (and says how to start one), creates and migrates `dclab_dev` when needed, serves the API with reload and one worker, and rebuilds the frontend when a file under `agentic/web/src` changes (a polling watcher: no new dependency); `make db-reset` drops and recreates it only with `CONFIRM=yes`; `make test-db` creates and migrates `dclab_test`; the database comes from the shell or `.env` (as the server reads it) and must be named `dclab_<letters, digits, _>` on this machine, or it is refused before any connection: the URL's host, its `host`/`hostaddr` query and `PGHOST`/`PGHOSTADDR` are all read, a service file is refused, and the server reached must answer from a socket or loopback before anything is created or dropped (`--remote` to mean another host); the frontend watcher survives a failing build and a file that vanishes mid-scan. Checked by hand on a scratch database (created, migrated, served, the frontend rebuilt on a change). Open: a tunnel forwarding a local port to another server passes the host checks (only the `dclab_*` name and `CONFIRM=yes` stand there); the API's reload watches `dclab_rnd` only (a change in `general_pipeline` needs a restart) |
| 12.1 | Containers | Done (`Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements/product.txt`, `make up`/`down`/`logs`): a multi-stage image (wheels built in the first stage, a user without privileges, the built frontend, no key, no .env in any folder), the app, a worker and PostgreSQL with named volumes, published on 127.0.0.1; the image installs only what the server, the worker and the five stages use (no PyTorch or TabPFN; the CPU build of XGBoost). Proven by CI's containers job (run 37547513100): `docker compose up --build --wait` starts the app, the worker and PostgreSQL, the four end-to-end flows pass against it (`product_e2e.py --base`, the same results as on a laptop), and the image's 1,871 files hold no secret. Open: not built on the owner's machine, whose Docker disk is full; the image is about 2 GB with LightGBM, XGBoost (CPU) and scikit-learn |
| 12.2 | Configuration and secrets | Done (`.env.example`, `dclab_rnd/secret_files.py`, `dclab_rnd/settings.py` `problems`, `scripts/scan_secrets.py`, `make scan-secrets`, `tests/test_config.py`): every setting the code and the scripts read (DCLAB_, OPENAI_, HF_, KAGGLE_ and AWS_ names, in the tracked code, the scripts and the Docker files) is in `.env.example` with its safe default, and a test fails when one is added without it. A key or a connection string comes from the environment, or from a file named by `NAME_FILE` (Docker secrets, a platform's secret store), read when the package is imported so the server, the worker and every command-line tool see it; only the product's own secrets are read this way (a fixed list, `DCLAB_DB_<NAME>`, `DCLAB_TIER_<TIER>_API_KEY`), so AWS's own `_FILE` settings keep their meaning. No key enters an image: `.dockerignore` excludes every `.env` in any folder (a review found `dclab_rnd/agentic/.env` would otherwise have been copied; no image had been built). The server and the worker refuse to start and say why when: a secret file cannot be read; `DCLAB_AUTH` is not password, oidc or none (a typo must not leave a server without sign-in); other host names are allowed without sign-in; accounts are on without PostgreSQL; the database does not answer; its schema is behind the code ("run python -m dclab_rnd.storage upgrade"); sign-in through the identity provider lacks a setting; or a numeric setting (ports, threads, retries, pools, budgets, every limit) is not a number; never echoing a URL. `DCLAB_ALLOWED_HOSTS` adds a proxy's host names (taken as written: scheme, port and capitals are stripped; a wildcard only as *.domain). The scan reports where, the kind and a hash of the value, never the value; a placeholder hides only itself, not its line; the 1,875 tracked files and every line added in the history's 117 commits hold no secret apart from two made-up test values, allow-listed by their hash. Open: the image's own scan runs in CI (12.3), as this machine's Docker disk is full; the scan is patterns, not entropy (base64-encoded keys are not found) |
| 12.3 | Continuous integration | Done (`.github/workflows/ci.yml`, replacing rd-knowledge.yml, which had failed at its Python setup on every push): on every push and pull request, and nightly, `make rd-check`, `make test-pg` against a PostgreSQL service container, `make product-e2e` and `make scan-secrets`; on main, the containers job (12.1); no job uses a model key (an empty OPENAI_API_KEY and DCLAB_NO_LIVE_MODELS=1); pip is cached by requirements/base.txt, with CPU-only PyTorch. Its first runs found what local runs could not: mcp 2.x installed fresh (pinned below 2), a service container's own address (the development script's database test), a test file not yet scanned, and a 30-second wait too short on a busy runner; run 37547513100 passes both jobs. Open: blocking the merge needs main protected with these two checks required (Settings > Branches: the owner's setting); the dependencies are not locked (a new release can still break CI; a lock file would make it repeatable) |
| 12.4 | Logs, health and metrics | Done (`dclab_rnd/observe.py`, `tests/test_observe.py`): with `DCLAB_LOG_FORMAT=json` (the containers set it) one JSON line per request and per job that ends badly (plain text outside the containers logs only server errors, with the same fields), with a request id the response carries back (`X-Request-ID`; a proxy's id is kept when it is a plain token), the route's pattern (never the ids in it, a query string, a body, a prompt or a secret), the status, the time and the person's and workspace's ids; `GET /healthz` (the process is up) and `GET /readyz` (the database answers and the file storage can be written; 503 names what failed, never a URL; at most one probe at a time, 5 s at most), the compose health check uses it; a 500 carries its request id; Prometheus counters on `DCLAB_METRICS_PORT` (off unless set; the app on 9464 and the worker on 9465 in the containers, never published): requests by route and status with their time, jobs by kind and ending with their time, failures, and model requests, tokens and euros by purpose; the Compute page lists the latest failed or interrupted jobs with their reason and Retry. Open: each process serves its own counters (Prometheus adds them); the counters start at zero when a process restarts; a load balancer probing with an IP as its Host name needs that name in DCLAB_ALLOWED_HOSTS; a streamed answer's time is the time to its first byte; uvicorn's own error lines and a job's traceback stay plain text beside the JSON lines |
| 12.5 | Backups and retention | Done (`dclab_rnd/storage/backup.py`, `dclab_rnd/retention.py`, `tests/test_backup_retention.py`): `make backup` writes `backups/<time>/` with the database (`pg_dump`, custom format, no owners or grants, from the same snapshot as the row counts), the workspace folder as `files.tar.gz` (linked folders followed) and a manifest of the schema, the rows per table, the files and both archives' SHA-256; `make restore FROM=… HOME_DIR=…` checks the archives first, goes only into an empty database and an empty folder, and compares the rows and files after (the test restores into a scratch database and opens the restored project's table through its recorded hash); `DCLAB_RETENTION_RAW_DAYS=N` deletes a draft's raw upload N days after it became a cleaned table (a name uploaded twice follows its newest upload; never a failed one, one still being processed, or one whose content is not the recorded one; the file is moved aside atomically, checked, then removed, so two sweeps delete and audit it once; each worker sweeps hourly on its own thread, `python -m dclab_rnd.retention --dry-run` lists), keeping the cleaned table, its SHA-256, the asset's record and the audit, with an audit entry of kind retention per file; deleting a project or a draft removes its files and leaves an audit entry of kind deletion (a draft's delete now waits for a write in progress, which used to bring a deleted draft back); the Admin privacy list states the rule. Open: with files in a bucket (DCLAB_FILES_URL) the bucket's own versioning keeps them, the archive holds only the folder; the files are archived just after the dump, so a file written in between can be in the archive without a row and a file deleted in between is listed as vanished (stop the app and the workers for an exact pair); a backup that fails half way is removed, and one inside the workspace folder is refused; a failed asset's raw file stays until its draft is deleted; the days count from when the file was received; a moved-aside file that cannot be put back (another file took its name) stays as a hidden .retiring- file; the identical bytes uploaded again in the instant of a deletion are deleted too (its asset fails and asks for the file again); a superseded asset's cleaned table was already replaced by the newer one, so its raw file going leaves only its records; backups are not encrypted or sent off the machine (the operator's storage does that) |
| 12.6 | Cloud deployment | Done as code, not applied (`deploy/`, `deploy/README.md`, `tests/test_deploy.py`, `make deploy-check`; the owner chose AWS and Google Cloud, region later): OpenTofu/Terraform for both clouds, each one module with a staging and a production environment (region a variable without a default). AWS: ECS Fargate app (2+ tasks) behind an ALB with TLS 1.2/1.3, Fargate workers scaled on the job queue (CloudWatch alarms: jobs waiting two minutes add workers, fifteen minutes with nothing waiting or running removes one; a worker running a job holds ECS scale-in protection, so scaling in never stops it; a deployment starts the new workers beside the old, which drain: they claim nothing new, finish their jobs and then end, so after a rollback the service simply starts its current version again), RDS PostgreSQL 16 with backups and point-in-time recovery (35 days and Multi-AZ in production), S3 (versioned) for the files of record and EFS for the shared workspace folder, Secrets Manager, CloudWatch logs, metrics through an OpenTelemetry sidecar, alarms. Google Cloud: Cloud Run app (2+) behind a global HTTPS load balancer with a managed certificate, a Cloud Run worker pool sized by the same rule (one worker, holding an advisory lock, sets the pool's instance count through the Cloud Run Admin API), Cloud SQL PostgreSQL 16 on a private address with point-in-time recovery (regional in production), a versioned bucket mounted as the workspace folder, Secret Manager, Cloud Logging, managed Prometheus, alert policies. In the app: each worker publishes the jobs waiting and active (DCLAB_QUEUE_METRIC: CloudWatch or Cloud Monitoring) and Prometheus gauges, and on Google Cloud applies the rule (DCLAB_WORKER_POOL); workers migrate before they start, as the app does; the research campaign's SQLite file can live off the shared folder (DCLAB_RESEARCH_DB); the first owner is made by a one-off task or job whose password comes from a secret; the migration on start takes an advisory lock (several instances start together); /healthz and /readyz pass the host check (a load balancer probes by IP). Validated with tofu validate for the four environments; a test checks that every setting the code gives the app is documented, that no secret is written in it and that production keeps its protections. Open: nothing has been planned against a real account (the first plan may need quotas or organisation-policy changes); on Google Cloud the worker pool is BETA and the call that sizes it is tested against a stand-in only; CI does not run make deploy-check yet (the owner asked to leave CI for now); TLS certificates and DNS are the owner's |
| 13.1 | A typed API client and consistent loading and error states | Done (`dclab_rnd/agentic/web/client.py`, `src/api.js`, `tests/test_web_client.py`): `make web` generates `DC.client` from the server's OpenAPI schema, one function per route named after its Python function (`DC.client.getProject(id)`), with JSDoc types for every model, path parameters encoded and required, and an unknown query parameter refused at run time; `DC.client.path` and `DC.client.href` give paths for the event stream and download links; the build's check fails when a route changes and api.js was not rebuilt; every page calls the client (98 calls; a test fails on a path string and on a call to a route the server does not have). `DC.states` gives loading, empty and failed (what happened, Try again) from the existing `.empty` component; every page's load shows Loading… when slow and, when a read failed, the existing warning callout with the reason and Try again; a read whose failure the page shows itself, or an expected miss, is quiet; a signed-out answer goes to sign-in. Checked in the browser on all 19 pages (no console error) and with a failing read. Open: the sign-in form and /api/config are fetched before the client exists (they bootstrap it); empty states keep their existing wording; pages that fall back to their sample still do, now under the warning; a poll of another page that fails while a page loads can still be counted against it. Generating the client builds a server on a scratch folder on files (no workspace, no database: the build works with the database stopped), and importing the server no longer builds its app (made on first use) |
| 13.2 | Browser tests of the main flows | Done (`scripts/browser_e2e.py`, `make browser-test`, `requirements/browser.txt`; CI's containers job runs it against the container from 12.1 and keeps the screenshots of a failing check): Chromium types one sentence and uploads the Telco CSV on Home, answers the chat's questions by clicking their options, goes through the wizard's five steps with their defaults, starts the run on the Workflow page (clicking the next stage if the run stops before one) and checks the brief's headline against the stage record and its report download; the numbers on Home, Compute, Integrations and Packs equal the API's (read over HTTP, not through the page); all 19 pages at 1280 px show no console error, uncaught error, CSP violation or failed-load warning; all 19 at 375 px fit without sideways scroll (it found one: a long path in a footnote's code on Benchmark, which now wraps; and Home's project count kept the sample's 8 on an empty workspace, now fixed). Each page is read when its own load has ended (the 13.1 tracker), not after a fixed wait; the app's first load counts; checked on files and on PostgreSQL. About 90 s, no model. Open: not part of make check-all (it needs Playwright and Chromium; CI runs it on main); a model-driven chat is not covered (no live key in tests); the page-state check counts only what the 13.1 tracker sees |

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


---

# Part 3 — Prove it works

**What exists, plainly.** Parts 1 and 2 built the product and its foundation, and they are tested: about 750 tests,
the end-to-end flows on a temporary server, browser tests on 19 pages, the scripted judgment suite of 20 seeded
cases (`make agent-eval`), and every model path once against a small local model (8.1). None of that shows that
the agent makes good decisions on cases it has never seen, that it helps a data scientist, or that it runs for real
users. `docs/SYSTEM_EVALUATION.md` names the four questions, and the release decision is an AND gate over them:

| Question | Measured by | Package |
|---|---|---|
| Does the software obey its rules? | invariant violations: the tests, the validator, the audit | done (Parts 1 and 2) |
| Is the scientific evidence internally valid? | reconstructed metrics, splits and provenance | done for the archive (`evaluation/`), kept by `make rd-check` |
| Does the agent choose good actions? | completion and safe decisions on unseen cases | 14.3, 14.4, 14.5 |
| Does it help a data scientist? | tasks completed, time, corrections, trust | 14.6 |

A safety failure is never averaged into a good score for writing or speed. 20 seeded cases, or 579 registry
records, are not that many independent tests of judgment.

**Decided by the owner on 2026-10-07:** the next phase is this evaluation. First write Part 3, then build the frozen
benchmark (14.3), then continue with what does not wait for the owner.

| # | Package | Status |
|---|---|---|
| 14.1 | The owner's inputs: contracts, reviewers, tasks, model access | Prepared (2026-10-07), open on the owner's answers: `evaluation/OWNER_ANSWERS.md` has one block per question (the file's own columns listed, so the answers can use their words), and `python -m dclab_rnd.agent_eval.owner_inputs status` lists what is open and checks each `evaluation/contracts/*.json` loads as a `studio.solution` Solution that names only columns its data has (`tests/test_owner_inputs.py`). Nothing is filled in for the owner: 10 of 11 questions are open; the provider check is recorded from the live runs of 14.2 and marked for the owner to confirm. The two contracts are not written: the prediction moments are the owner's to say |
| 14.2 | Every live check against the production model | Done (2026-10-07, gpt-6-luna, about €0.35 in all): the first run failed every tool request (AEV-012 kept: OpenAI refuses function tools on chat completions for this model unless reasoning_effort is "none"; `models/client.py` now learns that from the refusal, retries once and sends it from the start, with a scripted-client test), and the specialists could not start in `.venv-agent` (the usage log imported SQLAlchemy without a database; `storage/url.py`, with a test). Then: `model-paths` all five paths ran, 21 requests, none failed, €0.005 (the synthetic design, the log pattern at parse rate 1.0, the Home chat with a 10-node workflow the check accepted, cited evidence and an honest "not covered", the intern's five tools); the five specialist cases passed their hard checks, one request each (`evaluation/live_specialists_2026-10-07_gpt-6-luna.json`); the judgment suite, 2 repeats (AEV-013): leaks caught 0.96 (95%: 0.47–1.0), false alarms 0.28 (0–0.59), valid moves and citations 1.0, 547 requests, €0.34. Every request with tools went with reasoning_effort "none" (the scores are for the model without reasoning on tool turns; later live results say so). Open: nearly every intern run ended on its step budget rather than finishing, and input tokens dominate the cost (3.25 M in, 31 k out); two repeats are too few for tight intervals |
| 14.3 | A frozen benchmark of 100 or more unseen cases | Done (2026-10-07; `dclab_rnd/agent_eval/benchmark.py`, `benchmark_sealed.json`, `tests/test_benchmark.py`, `make benchmark SPLIT=dev|test`, results in `evidence/campaigns/benchmark_v2/`): 108 cases, a dev set of 38 (3 bases) and a sealed test set of 70 (6 bases: 4 new generators and the diabetes and wine tables shipped with scikit-learn), no base in both; the test fingerprints are committed and a test fails if one changes; a run on the test set must be recorded. Scripted policies on the sealed set (BEN-002): the audit's defaults keep 28/38 leaks out (95%: 58–85%) with 18/70 false alarms; the standard plan (no model) 7/38 (9–33%) with 12/70 false alarms; the bad policy 0/38. Open: the labels are the author's until 14.4; cases from one base are not independent; in the missing_is_outcome family the audit's catches come from two telling names (exit_rating, delivery_rating_given), not the missingness, kept until a new version because the sealed set has been scored |
| 14.4 | Expert review of the benchmark's labels | Kit built (2026-10-07), open on two reviewers: `python -m dclab_rnd.agent_eval.review export` writes a blank form per reviewer for a stratified sample of dev and test cases (one row per column: name, kind, missing, distinct, three values; the reviewer fills keep_out, usable, time or group) under anonymous references, with a key only the owner keeps; `review agreement` reads the two returned forms and reports agreement and Cohen's kappa between the reviewers and with the benchmark, every disagreement, and the cases both confirm (`tests/test_label_review.py`). A changed label on the sealed set makes a new benchmark version, never an edit |
| 14.5 | Baselines on the benchmark, with the gate written first | Done (2026-10-07; `dclab_rnd/agent_eval/{gate,baselines}.py`, `tests/test_{gate,baselines}.py`, `prompts/base_audit.md`, `agent_eval.live --split test`; results BEN-002 to BEN-005 in `evidence/campaigns/benchmark_v2/results/`): gate_v1 was committed (48acdb4) before the base-model and agent runs; the scripted runs (BEN-002) were recorded earlier, and the thresholds come from the package text, not from them. **No policy passes the AND gate on the sealed 70 cases.** Leaks kept out of 38 (95% interval) / false alarms of 70 (of the 32 pure controls): standard plan 7 (9–33%) / 12 (7); audit rules 28 (58–85%) / 18 (13); bad policy 0 / 0 and a move the validator refuses (valid moves 0.75); prompted base model, names and summaries only, no tools (gpt-6-luna, €0.0097) 33 (73–94%) / 6 (3); the current agent, the intern on gpt-6-luna with its tools and 16 steps (€0.58, 976 requests) **35 (79–97%) / 26 (21)**, valid moves 1.0, citations 1.0. The agent keeps the most leaks out and fails on false alarms: it forbids clean columns on 37% of cases (for example a real wine feature), and 53 of its 70 runs ended on the step budget, not by finishing. The base model is the closest to the gate on false alarms and short on leaks. Open: the labels are the author's until 14.4; one repeat per case; the base model's prompt names kinds of leak in general words (a hinted baseline); the gate counts false alarms over the 70 cases with a clean column, and the report also gives the 32 pure controls |
| 14.6 | A user study: with and without DCLab | Protocol, harness and analysis built (2026-10-07), open on the owner's approval and 3–5 participants: `docs/USER_STUDY_PROTOCOL.md` (a crossover design with the order alternated, what is recorded, the consent text, the analysis fixed first) and `python -m dclab_rnd.agent_eval.study start\|correction\|citation\|severe-false-alarm\|finish\|analyze`: every event is an enumerated value or a number, there is no free-text field (the code refuses one), and the analysis is descriptive with Wilson intervals and no significance test (`tests/test_user_study.py`). The tasks come from 14.1 |
| 14.7 | Staging in a cloud, smoke-tested | Runbook and check built (2026-10-07), open on the owner's yes, region, account and domain: `deploy/STAGING.md` gives the order (decide, `tofu plan` with the cost, apply only after the owner's yes, flows, backup and restore drill, destroy) and `make staging-check URL=…` runs the end-to-end flows and the browser checks against the address. Staging asks for sign-in, so both scripts take an API token of a test member in `DCLAB_E2E_TOKEN` (the header is tested; the checks have never run against a signed-in server, so the first run may need a fix), and with a model key in staging the flows send model requests. Nothing was planned or applied: it needs the provider's credentials |
| 14.8 | The policy model's gate (A6.3) | Waiting: 14.3–14.6 first, then enough opted-in projects |

## 14. Evaluation

### 14.1 The owner's inputs

- **Delivers:** the answers in `evaluation/OWNER_INPUTS.md` turned into files: the prediction contracts for HyperAck and churn as solutions (`evaluation/contracts/*.json`, checked by `studio.solution`), the reviewers' names, two or three real tasks with a success criterion and a time budget, and the provider check.
- **Needs:** the owner's answers. Nothing is guessed: an unanswered question stays open in the file.
- **Done when:** each contract loads as a valid solution, and the HyperAck and churn projects can be re-run under it.

```text
Package 14.1. Read evaluation/OWNER_INPUTS.md and the owner's answers. Write each prediction contract as a
solution file under evaluation/contracts/ (target, prediction moment, forbidden columns with reasons, time and
group columns, metric) and check it with studio.solution. Record the reviewers, the tasks and the provider check in
evaluation/OWNER_INPUTS.md. Leave every unanswered question visibly open; never fill one in.
```

### 14.2 Every live check against the production model

- **Delivers:** `make model-paths ARGS="--configured --cap-eur 2"` (8.1 on the main model, including a workflow proposal), the five fixed specialist cases in `evaluation/live_cases_v1.json`, and `make agent-eval-live ARGS="--yes --cap-eur 2"`, each with its requests, tokens and euros.
- **Needs:** credit on the account (8.1 found none), and the model's checked price (8.4) so the caps can stop spending.
- **Done when:** each check has run, every provider or executor failure stays in the result (never replaced by a mock), and anything that broke has a fix and a test.

```text
Package 14.2. With the owner's yes and a cap, run make model-paths ARGS="--configured --cap-eur 2", the five live
specialist cases and make agent-eval-live ARGS="--yes --cap-eur 2". Synthetic or public data only. Report, per
path and case, what the model proposed and what the code accepted or refused, with requests, tokens and euros.
Keep every failure in the record. Fix defects with a scripted-client test.
```

### 14.3 A frozen benchmark of 100 or more unseen cases

- **Delivers:** `dclab_rnd/agent_eval/benchmark.py` (`make benchmark`): 100 or more cases, each a seeded table with a prediction moment, at most one planted trap, and the decisions a careful data scientist would accept (columns to keep out, columns that must stay usable, the time or group column to declare, rows that repeat). The tables come from base generators that no prompt was tuned on and from real public tables that ship with scikit-learn (no download); the traps are the families of `docs/SYSTEM_EVALUATION.md` (a post-outcome feature under a telling or a neutral name, a noisy copy of the target, an identifier in target order, the target in another unit or as a sum, a value missing exactly when the outcome happened, entity leakage across a split, a time-ordered table split at random, repeated rows) and clean controls (constant columns, honest strong drivers, informative missingness known at the start, a harmless "post" name). The cases are split into a **dev** set (about 40, for building and tuning) and a **sealed test** set (60 or more): the test set's fingerprints are committed, a test fails if one changes, and running a policy on it is recorded.
- **Done when:** 100 or more cases (60 or more sealed), the same on every machine; the scripted policies (standard plan, audit, bad) scored on both sets, per family, with Wilson intervals; the sealed set never used to change a prompt or a rule (written in the module and checked by a test that the dev and test seeds do not overlap).

```text
Package 14.3. Build a frozen benchmark next to the judgment suite (dclab_rnd/agent_eval). Generate 100 or more
cases from seeded base tables: new generators (claims, loans, readmissions, demand, sensors, subscriptions) and
the scikit-learn tables that ship with the library (no download). Plant at most one trap per case from the families
in docs/SYSTEM_EVALUATION.md, and include clean controls. Each case states its prediction moment and the decisions
that are acceptable. Split into dev and a sealed test set; commit the test fingerprints; a test fails when a case
changes. Score the scripted policies with the existing score() on both sets, per family, with Wilson intervals,
and store the result as evidence. Never tune a prompt or a rule on the sealed set.
```

### 14.4 Expert review of the benchmark's labels

- **Delivers:** two reviewers label a sample of cases (dev and test) without seeing the expected decisions; the agreement rate between them and with the benchmark; labels corrected on dev; the sealed set frozen after review (a new version if a test label changes, never an edit).
- **Done when:** the sample is reviewed, disagreements are resolved and recorded, and the benchmark's version says which labels were reviewed.

### 14.5 Baselines on the benchmark, with the gate written first

- **Delivers:** the thresholds of the AND gate written before any run (for example: no unsafe move; leaks kept out on at least 90% of leakage cases with the lower 95% bound above 80%; false alarms on at most 10% of controls); then the standard plan, the audit reference, a prompted base model and the current agent run on the sealed set with identical tools and budgets.
- **Done when:** a result per policy and family with intervals, every failure kept, the gate's verdict stated as evidence about these policies on these cases, never as a production claim.

### 14.6 A user study: with and without DCLab

- **Delivers:** a protocol (the tasks from 14.1, consent, what is recorded), a harness that times each task and records corrections without keeping cell values, and an analysis script: tasks completed, time, correction time, citation correctness and severe false alarms, with and without the assistant.
- **Needs:** 3–5 data scientists; the tasks from 14.1.
- **Done when:** the protocol is approved by the owner, the harness is tested, and the first sessions are recorded.

### 14.7 Staging in a cloud, smoke-tested

- **Delivers:** `tofu plan` for the chosen cloud's staging (12.6) with a cost estimate; after the owner's yes, `apply`; then `make product-e2e` and the browser tests (13.2) against the staging address, and a backup and restore drill (12.5).
- **Needs:** the region, the account, a domain, and the owner's explicit yes before any apply.
- **Done when:** staging passes the flows and the browser tests, and the plan, the cost and the drill are recorded. Production is a separate yes.

### 14.8 The policy model's gate (A6.3)

- **Delivers:** nothing is trained until 14.3–14.6 have results and A6.2's threshold is met (50 distinct trajectories from 10 opted-in projects); the sealed benchmark never enters a training set.
- **Done when:** the owner approves the compute and the cost, and the trained model is scored on the sealed set against the 14.5 baselines.

## Suggested order for Part 3

1. **14.1** (the owner) and **14.3** (now), side by side.
2. **14.5** with the scripted policies, then **14.4** (the reviewers).
3. **14.2** when there is credit, then **14.5** with the live models.
4. **14.6**, then **14.7**; **14.8** last.

# Part 4 — Rare events with partial labels (fraud and abuse)

**Why this Part.** On 2026-10-07 the owner gave a hiring challenge as a probe of what DCLab can do: detect fraud by
bikers and customers in a city's delivery rides, with a small sample of known fraudulent rides in a second file. A
complete hand solution was built outside this repository (`../Snap_fruad/`, with its own git history). Its lessons,
in the order they mattered:

1. **The labels are the hardest part.** The sample joins to the ride table on the order, not on (customer_id,
   biker): the table has two customer ids, so the coarse join turned 94 known frauds into 3,416. The rest of the
   table is *unlabelled*, not clean (a positive-unlabelled problem), and 8 sample rows are not in the table at all.
   One column means different things in the two files (the receiver's phone in the sample, the sender's in the table).
2. **Rows cluster on entities.** 94 frauds sit on 53 bikers; a random split lets a model recognise the biker. Every
   honest number came from folds grouped by biker and a cluster bootstrap over bikers.
3. **The signal is behaviour, not columns.** A fake ride ends where it starts (median 26 m against 323 m) and the same
   biker returns to the same drop-off square. Entity and place counts, repeat shares and shrunk rates separated
   frauds (single-feature AUC up to 0.90); raw columns did not (0.76 at best).
4. **Compare families, choose by a rule written first.** Rules, seven anomaly detectors, seven classifiers (with PU
   bagging), hybrids and a blend on identical folds and resamples. Average precision 0.081–0.095 for the top group
   (about 115–135 times chance), ECOD 0.050 with no label at all; PU bagging chosen as tied with the top and built
   for partial labels. Recall in the top 0.5 % of rides is the number an analyst plans with.
5. **Say what the data cannot show.** A train-on-past check fell to AP 0.010, because the later frauds are of a kind
   the table cannot see (no receiver phone). Precision is a lower bound.
6. **Ship it.** A scoring API (Flask + Gunicorn) in Docker, the same features in training and serving, tests on
   synthetic rides.

**What DCLab has and lacks today** (read from the code on 2026-10-12): it has average precision as the default
metric below 10 % positives, grouped CV (`StratifiedGroupKFold`), a cluster bootstrap, majority and chance baselines,
static detection of oversampling and preprocessing before the split, and an "Imbalanced and fraud" pack. It lacks:
a second file of labels (a project has one table), any notion of partial labels (unlabelled rows count as negatives),
a check of the join key, entity behaviour features (all derivations are row-wise), anomaly detectors and PU learning
(none in the code), alert-budget metrics, a split that is both time-ordered and grouped (time wins and entities leak),
a pack that changes what the engine does (the imbalanced pack only relabels workflow steps), copilot checks for the
mistakes above, measured evidence for them, benchmark cases of this kind, and an export to a scoring service.

**Goal of the Part.** Given a ride table and a sample file of known frauds, DCLab reaches the hand solution's
decisions on its own: the right join, partial labels declared, folds grouped by the entity, behaviour features
fitted inside folds, rules, anomaly and supervised families compared on an alert budget, a model chosen by a rule
with intervals, the limits said, and a scoring service exported.

**The challenge's data is confidential** ("must not be disclosed or distributed"): it never enters this repository,
a campaign, a prompt, the SFT corpus or a benchmark. Everything here is built and tested on a synthetic replica
(15.1). The real data is used once, on the owner's machine, for 15.11, and only aggregate numbers are recorded.

| # | Package | Status |
|---|---|---|
| 15.1 | A synthetic replica of the challenge and its acceptance test | Done (2026-10-12; `dclab_rnd/replicas/delivery_fraud.py`, `tests/test_fraud_acceptance.py`): a seeded ride table (about 16,600 rows, 3 accounts, 240 bikers, cancelled rides without a biker, multi-drop orders, order ids rising with time) and a sample file of known frauds, all invented (vocabulary and formats included). Planted: fake trips that end metres from the pick-up at a spot the biker repeats, and self-deliveries the table cannot show. Seeds 0-3: 89-103 known fraud rows (0.54-0.63 %) on 37-40 bikers (the real case: 94 on 53); the (account, biker) join marks 18-24 times as many rows as the order join; 5 sample rows are not in the table; `sender_phone` holds the receiver's phone in the sample and about 90 % of its fares disagree. The hand method recovers it: behaviour features with folds grouped by biker reach AP 0.47-0.54 over seeds 0-2, against 0.05-0.12 for the raw columns and 0.27-0.31 for raw columns plus the biker id on a random split (the leak). Baseline of the engine as it was (labels joined by hand, seed 0): holdout AP 0.277 (95 % cluster bootstrap 0.086-0.501; chance 0.006), recipe selected_mi, extra trees. Seven acceptance tests are expected failures, each naming its package and each checked to fail for its own reason; one is a real misdetection: "Which rides should our quality team review first?" picks the text pack (the word review), for 15.6. The end-to-end test asks for the decisions (label key, partial labels, folds by biker, the behaviour recipe, the families, a stated rule, recall at the budget above the budget, precision as a lower bound, the disagreeing column as a stated limit) and AP above 0.35 on the holdout; it does not ask the interval's lower end to clear chance, because a holdout grouped by biker holds 4-11 fraud bikers and a good model's lower end fell below chance on one of five simulated holdouts. Differs from the text: the replica is its own module (first `dclab_rnd/replicas/`, moved to `dclab_core/replicas/` in 17.3), because the spec generator of `draft/synthetic.py` draws one table from a linear predictor and cannot express two files or behaviour; it has 3 accounts, not 2 (a third small account keeps the join from being a plain two-way split); Home does not offer it yet (15.6). Open: the end-to-end test names a ProjectStore on files, so 15.2 must add its own PostgreSQL check |
| 15.2 | A label file: join, key check, partial labels | Not started |
| 15.3 | Alert-budget metrics and a time-and-group split | Not started |
| 15.4 | Entity behaviour features, fitted inside folds | Not started |
| 15.5 | Rules, anomaly detectors and PU learning in the model screen | Not started |
| 15.6 | The fraud pack reaches the engine; Home asks the right questions | Not started |
| 15.7 | Copilot checks and measured pitfalls for these mistakes | Not started |
| 15.8 | Evidence: the FRD track's comparison of families | Not started |
| 15.9 | Benchmark v3: rare-event cases (dev first, a new sealed version) | Not started |
| 15.10 | Export a finished project as a scoring service | Not started |
| 15.11 | Acceptance on the real challenge, on the owner's machine | Waiting: 15.1–15.10 |

## 15. Rare events with partial labels

### 15.1 A synthetic replica of the challenge and its acceptance test

- **Delivers:** a seeded generator `delivery_fraud` in `draft/synthetic.py` that writes two files: a ride table
  (terminals of orders, bikers, two customer accounts, sender phones, coordinates, fares, cancelled rides without a
  biker) and a sample of known frauds (a minority of the planted ones, joined by order id, with the receiver's phone
  under the name the table uses for the sender's, and a few rows whose orders are not in the table). Planted
  behaviour: fake rides that end within metres of the pick-up at a repeated drop-off, self-dealing phones, frauds
  concentrated on a few bikers, and a kind of fraud that the table cannot show. Plus
  `tests/test_fraud_acceptance.py`: the decisions DCLab must reach on the replica (join on the order; the pair join
  refused with the inflation it causes; labels declared partial; folds grouped by biker; behaviour features chosen
  over raw columns; average precision and recall at the budget above the chance and majority baselines; a model chosen
  by the selection rule with intervals; the limit about the unseen kind of fraud stated). The test is marked
  expected-to-fail and each later package turns one part of it green.
- **Done when:** the generator is deterministic per seed and validated (the planted signal is recoverable by the hand
  solution's method, recorded as numbers in the test), and the acceptance test runs and fails for the known reasons.

```text
Package 15.1. Read Part 4's lessons. Add a seeded delivery_fraud generator to draft/synthetic.py that writes a ride
table and a separate sample of known frauds with the traps listed in 15.1 (coarse key that inflates labels, partial
labels, rows missing from the table, a column whose meaning differs between the files, entity clustering, a fraud
kind the table cannot show). No confidential data: invent everything. Write tests/test_fraud_acceptance.py with the
decisions a careful data scientist reaches on it, marked xfail per part, and a test that the planted signal is
recoverable. Do not change the engine in this package.
```

### 15.2 A label file: join, key check, partial labels

- **Delivers:** a project can hold a table and a label file (`attach_labels`; the draft's second asset is kept by
  `build_project`). The join key is chosen and checked by code: for each candidate key, how many table rows one label
  row would mark; a key that marks more than one row per label (or more rows than the sample has) is refused with the
  inflation as the reason, unless the person states that the label is per entity. Label rows not found in the table
  are counted and kept apart. Columns with the same name are compared and flagged when their values do not match
  (the phone case). `Solution` gains `labels: complete | partial` (default complete; partial when labels come from a
  sample or the person says so) and the label provenance; `prepare()` no longer treats a missing label as a negative
  without that declaration. A new graph move `attach_labels` and an intern tool `review_labels` (read) that reports
  the key check.
- **Done when:** the replica's labels are joined on the order, the pair join is refused with "94 → 3,416"-style
  numbers, partial labels reach the solution and the stage record, the moves are logged, and the PostgreSQL store
  holds the label file too (`make check-all`).

```text
Package 15.2. Let a project hold a label file next to its table. Implement the join-key check in code (rows marked
per label row, labels not found, same-named columns that disagree), refuse an inflating key with the numbers, and
add Solution.labels = complete|partial with provenance. Add the attach_labels move to studio/graph.py and a read tool
review_labels to the project scope. Keep the draft's second asset in build_project. File and PostgreSQL stores both.
Turn the label parts of tests/test_fraud_acceptance.py green.
```

### 15.3 Alert-budget metrics and a time-and-group split

- **Delivers:** `recall_at_budget`, `precision_at_budget` and AP lift in `binary_metrics`; `Solution.alert_budget`
  (a share of rows an analyst can review, asked in words like "about 70 a day"); the budget metric selectable as the
  primary metric. A split that is time-ordered *and* keeps entities apart (forward folds where an entity seen in a
  training fold is removed from the later test fold, with the number removed reported), used when both a time and a
  group column are declared, instead of silently dropping the group. With partial labels, every precision is reported
  as a lower bound.
- **Done when:** the metrics have tests against hand-computed values, the split never puts an entity on both sides
  (tested), and the replica's final stage reports recall at the budget with a cluster-bootstrap interval.

```text
Package 15.3. Add recall/precision at an alert budget and AP lift to the engine's binary metrics, a Solution field
alert_budget, and the budget metric as a primary-metric choice. Add a split for time plus group (no entity on both
sides; report what was removed) and use it when both columns are declared. Mark precision as a lower bound when the
labels are partial. Tests with hand-computed values and an entity-overlap test.
```

### 15.4 Entity behaviour features, fitted inside folds

- **Delivers:** a recipe `entity_behaviour` in `expansion/runner.py`: for the declared entity columns (the group
  column and others named in the solution) and for place cells derived from coordinates, counts per entity, per
  entity and day, per entity and place, per place and route; repeat counts and shares; rates shrunk towards the
  training mean. The count tables are fitted on the training fold only. Two modes follow the prediction moment: an
  *audit window* (all rides of the window are known when scoring, labels never) and *at the event* (only earlier
  rows); the moment decides which, and the choice is written in the stage record. The feature ladder compares it with
  the row-wise recipes as usual.
- **Done when:** a test shows no test-fold row changes a training-fold feature, the at-the-event mode uses no later
  row (tested), and on the replica the ladder picks the behaviour recipe for a measured reason.

```text
Package 15.4. Add an entity_behaviour recipe to the feature ladder: count tables per entity, entity-day,
entity-place, place and route (place = a grid cell from coordinates when present), repeat shares and shrunk rates,
fitted on training folds only. Two modes chosen from the prediction moment: audit window or past rows only; record
the mode. Tests for fold isolation and for no future rows in the at-the-event mode.
```

### 15.5 Rules, anomaly detectors and PU learning in the model screen

- **Delivers:** for binary rare-event tasks, three more families in `MODEL_FAMILIES`: a percentile rule score
  (thresholds from the unlabelled training rows), unsupervised detectors (Isolation Forest and ECOD by default; LOF
  and others available) fitted on the training fold without labels and without any resampling, and PU bagging when
  the labels are partial. Detectors are ranked on the same folds and metrics as classifiers. The selection rule for
  this task is written in the stage record before the screen runs: the best bootstrap mean on the primary metric,
  and a simpler family when the paired interval includes zero.
- **Done when:** the screen on the replica lists all families with intervals, a resampling step can never precede a
  detector (tested), and the choice cites the rule.

```text
Package 15.5. Add rule, anomaly (IsolationForest, ECOD; LOF optional) and PU-bagging families to the model screen
for binary rare-event tasks. Detectors never see labels or resampled rows. Write the selection rule (best bootstrap
mean; a simpler family wins when the paired interval includes zero) into the record before the screen. Tests on the
replica and a test that no resampler precedes a detector.
```

### 15.6 The fraud pack reaches the engine; Home asks the right questions

- **Delivers:** the imbalanced pack sets engine defaults (primary metric, alert budget, the anomaly and PU families,
  the behaviour recipe, the time-and-group split) through one place the engine reads, not only workflow labels. Home
  asks, in the pack's words, what is scored (a ride, a transaction), who could commit the fraud (the entity), whether
  the labels are complete or a sample, whether they come in a separate file, and how many alerts a day the team can
  review. Data signals pick the pack when the positive rate is below 1 % (promised in `detect`, not built).
- **Done when:** a draft of the replica, started with "find fraudulent rides of bikers", reaches a solution with the
  entity, partial labels, the label file and the budget, using no more than the allowed questions, and the built
  project runs with the pack's defaults (scripted model and the no-model fallback both tested).

```text
Package 15.6. Make the imbalanced pack change engine defaults through one function the engine reads. Extend the Home
agent's questions for this pack: scored unit, entity, label coverage, label file, review budget; keep MAX_QUESTIONS.
Pick the pack from data when positives are under 1 %. Tests with a scripted model and with the fallback.
```

### 15.7 Copilot checks and measured pitfalls for these mistakes

- **Delivers:** new copilot detectors: a label made by `isin`/`merge` on a key that is not unique in the table and then
  assigned to a target; ROC-AUC or F1 computed on hard predictions at a rare prevalence; SMOTE followed by CV on the
  resampled set, or SMOTE in a CV loop without an imbalanced-learn pipeline; an anomaly detector fitted on
  resampled data; a random split when an id-like column repeats across rows (with the column read from the
  notebook). Each cites a rule and a measured pitfall: PIT-007 label inflation by a coarse key, PIT-008 random split
  on entity-clustered rows, PIT-009 anomaly detection on oversampled data, PIT-010 threshold metrics at rare
  prevalence, run on the replica and a public table.
- **Done when:** the two original challenge notebooks' *patterns* (rewritten on the replica, never the real data)
  raise each finding, a corrected notebook raises none of medium or high severity, and the PIT records are in the
  evidence index after `make knowledge`.

```text
Package 15.7. Add the copilot detectors listed in 15.7 to copilot/analyzer.py, each with a rule and a precedent
query. Measure PIT-007..PIT-010 in pitfalls.py on the replica and a public table (wrong way vs right way). Add an
example leaky notebook and a clean one on the replica; tests that the leaky one raises each finding and the clean one
raises none of medium or high severity. make knowledge.
```

### 15.8 Evidence: the FRD track's comparison of families

- **Delivers:** the `anomaly-and-fraud-detection` track scaffolded (`make new-track`) and FRD-002 run as a campaign:
  rules, anomaly detectors, classifiers, PU bagging and hybrids on the replica (several seeds and label shares) and on
  the public card-fraud table, with grouped or time splits, cluster-bootstrap intervals and paired tests. FRD-003: the
  alert budget against the expected cost per 1,000 events. Claims are written with their limits and become evidence
  records the agents can cite.
- **Done when:** the campaign files are immutable results with provenance, `make rd-sync knowledge research-index` is
  clean, and an agent asked "which method for rare fraud with few labels?" cites FRD records.

```text
Package 15.8. Scaffold the anomaly-and-fraud-detection track and run FRD-002 (families compared on the replica over
seeds and label shares, and on the public card-fraud table) and FRD-003 (budget vs cost) as an evidence campaign,
with intervals, paired tests and limits. Register the records, rebuild the knowledge, and test that a search for the
question cites them.
```

### 15.9 Benchmark v3: rare-event cases

- **Delivers:** new rare-event bases and traps for the frozen benchmark (label inflation by key, partial labels,
  entity clustering, oversampling before an anomaly detector), dev cases first. Because adding a trap changes the
  sealed set, this is **benchmark_v3** with new fingerprints and its own campaign; v2 and its results stay as they are.
- **Done when:** v3 is frozen, the scripted policies are scored on dev and once on the sealed set, and no prompt or rule
  was changed after a sealed-set run.

```text
Package 15.9. Add rare-event bases and the traps in 15.9 to agent_eval/benchmark.py as benchmark_v3 (new VERSION,
new sealed fingerprints, new campaign folder; v2 untouched). Build on dev; score the scripted policies on dev, then
once on the sealed set, recorded. Never tune on the sealed set.
```

### 15.10 Export a finished project as a scoring service

- **Delivers:** next to the notebook export, a service export: a small package with the fitted feature builder and
  model, a Flask app served by Gunicorn (health, model, predict for one, many or a CSV), a Dockerfile without data, and
  tests generated from the project's synthetic rows. With the behaviour recipe, the service adds a batch's counts to
  the stored ones without counting a known row twice, so training and serving compute the same features.
- **Done when:** the export of the replica project builds, its own tests pass, the Docker image runs in the end-to-end
  flows, and the export refuses a project whose final stage is not approved.

```text
Package 15.10. Add a service export for finished projects: model and feature builder, Flask + Gunicorn app,
Dockerfile with no data, generated tests. Same features in training and serving (test it). Refuse unapproved
projects. Add the flow to make product-e2e.
```

### 15.11 Acceptance on the real challenge, on the owner's machine

- **Needs:** 15.1–15.10, and the owner running it locally with the challenge's files.
- **Delivers:** DCLab run on the real files from a draft, with the owner answering Home's questions; its decisions and
  numbers compared with the hand solution in `../Snap_fruad/` (join, entity, split, features, chosen family, average
  precision and recall at the budget with intervals). Only aggregate numbers are recorded; no row, id or file of the
  challenge enters the repository.
- **Done when:** each decision matches or the difference is explained, and the result is recorded as evidence about
  this case only, never as a general claim.

## Suggested order for Part 4

1. **15.1** first: the replica and the acceptance test define "done" for everything else.
2. **15.2** and **15.3** (labels, metrics, split): without them every later number is wrong.
3. **15.4** and **15.5** (features and families), then **15.6** (the pack and Home).
4. **15.7** and **15.8** (checks and evidence) can run beside 3.
5. **15.9**, then **15.10**; **15.11** last, with the owner.

# Part 5 — Many kinds of problems

**Why this Part.** The owner, 2026-10-12: the fraud challenge is only an example; DCLab should solve many kinds of
machine-learning problems and challenges. Today the engine knows three tasks (`Solution.task`: binary, multiclass,
regression; text tables as a variant of binary). Nine packs exist (`draft/pack.py`), but a pack only changes Home's
first question and the workflow's labels: no pack reaches the engine. Each new kind of problem would today mean edits
scattered across `solution.py`, `engine.py`, `expansion/datasets.py`, `expansion/runner.py`, `draft/*`, the copilot and
the benchmark.

**The idea.** A *problem kind* is one module that says everything DCLab must know to solve that kind well, and the
engine, Home, the intern, the copilot, the benchmark and the export read it through one interface. A kind is only
called supported when its own acceptance test passes on a synthetic replica of a realistic case, as 15.1 sets up
for fraud. The ten workflow steps (WF-01…WF-10), the graph validator, the gates, the evidence and the cited explanations
stay the same for every kind: they are what makes DCLab careful, whatever the problem.

What a kind contributes (the interface of 16.1):

| Part | Examples |
|---|---|
| detection | words in the problem, signals in the data (no target column → clustering; a censoring flag → survival) |
| questions | the unit, the entity, the horizon, label coverage, the decision and its cost, in the kind's words |
| solution fields | horizon, censoring column, treatment column, alert budget, entity, label files |
| data assembly | joins, label files, event tables aggregated to an entity as of a cutoff |
| split | stratified, grouped, time, time and group, rolling origin, spatial blocks, by user |
| metrics and baselines | AP and recall at a budget; MASE against seasonal naive; C-index against Kaplan-Meier; NDCG@k against popularity |
| feature recipes | lags and windows, entity behaviour, text, embeddings |
| model families | classifiers, regressors, forecasters, survival models, rankers, detectors, clusterers, uplift learners |
| checks | leakage traps of that kind for the audit, the copilot and the benchmark |
| reliability | what the final stage must challenge (drift by time, calibration, subgroup gaps, censoring) |
| export | notebook and scoring service (15.10) with the kind's inputs |
| evidence | its research track and the claims agents may cite |

**Order with Part 4.** 15.1 (the fraud replica) comes first. Then 16.1 and 16.2 build the interface and the acceptance
harness, so 15.2–15.10 are built *as the first new kind* rather than as special cases; 16.3 onwards add kinds, and
16.13 lets a whole challenge (a brief plus files) be solved end to end.

| # | Package | Status |
|---|---|---|
| 16.1 | The problem-kind interface; today's tasks moved onto it | Not started |
| 16.2 | The kind acceptance harness and an honest capability matrix | Not started |
| 16.3 | Forecasting (horizons, rolling-origin backtests) | Not started |
| 16.4 | Time to event (censoring) | Not started |
| 16.5 | Ranking and recommendation | Not started |
| 16.6 | Segmentation (no target) | Not started |
| 16.7 | Uplift: who responds to an action | Not started |
| 16.8 | Several tables: entity targets from event tables | Not started |
| 16.9 | Decisions on top of a model: thresholds, prices, budgets | Not started |
| 16.10 | Places: spatial splits and features | Not started |
| 16.11 | Sequences of events and sensor anomalies | Not started |
| 16.12 | Images and long text through pretrained embeddings | Not started |
| 16.13 | Challenge mode: from a brief and files to every deliverable | Not started |
| 16.14 | Agents and evidence per kind | Not started |

## 16. Many kinds of problems

### 16.1 The problem-kind interface; today's tasks moved onto it

- **Delivers:** `dclab_rnd/kinds/` with a typed `ProblemKind` protocol (the parts in the table above) and a registry
  (`kinds.get(key)`, `kinds.detect(problem, profile)`); binary, multiclass, regression and text-tabular moved onto it
  with no change in behaviour; the engine, `Solution`, `prepare_bundle`, the runner and Home ask the kind instead of
  branching on `solution.task`; packs map to kinds (`imbalanced` → the rare-event kind of Part 4). `Solution` keeps its
  fields and gains a `kind` and a typed `kind_fields` block validated by the kind.
- **Done when:** every existing test passes unchanged (behaviour identical, checked by re-running the HyperAck and
  churn projects and comparing stage records), `/api/kinds` lists the kinds, and adding a kind touches only its module
  and its tests (a test registers a toy kind and runs it through all five stages).

```text
Package 16.1. Introduce dclab_rnd/kinds/: a ProblemKind protocol (detect, questions, solution fields, data assembly,
split, metrics and baselines, recipes, model families, checks, reliability, export, evidence track) and a registry.
Move binary, multiclass, regression and text-tabular onto it without changing behaviour: compare the stage records of
the HyperAck and churn projects before and after. The engine, Solution, prepare_bundle, the runner and Home read the
kind. Add /api/kinds and a toy kind in tests that runs through all five stages with no edit outside its module.
```

### 16.2 The kind acceptance harness and an honest capability matrix

- **Delivers:** each kind ships a replica generator (seeded, invented data shaped like a real case, with that kind's
  traps) and an acceptance test of the decisions a careful data scientist reaches on it, like 15.1. `make kinds-check`
  runs them all; the Domain packs page and `/api/kinds` show a capability matrix generated from the results (a kind is
  "supported" only when its acceptance passes; otherwise it says what is missing), replacing the hand-written maturity
  labels.
- **Done when:** the matrix is generated, not edited; the existing kinds pass their acceptance; a failing acceptance
  turns its row from supported to partial in a test.

```text
Package 16.2. Add a per-kind replica generator and acceptance test, make kinds-check, and a capability matrix built
from the results for /api/kinds and the Domain packs page. Remove the hand-written maturity labels in favour of it.
```

### 16.3 Forecasting (horizons, rolling-origin backtests)

- **Delivers:** a forecasting kind: a series id, a time index, a horizon and a forecast moment; rolling-origin backtests
  with a gap; lag and window features computed only from the past of each origin; baselines naive, seasonal naive and
  moving average; MASE, sMAPE and quantile loss; global gradient-boosted models with lags and a classical model per
  series where the series is long enough; prediction intervals by backtest residuals; checks for future covariates
  that are not known at the forecast moment.
- **Done when:** the replica (demand per store and product with promotions known ahead and weather not) is solved:
  backtests never see the future (tested), every model is compared with seasonal naive, and the leaky covariate is kept out.

```text
Package 16.3. Add the forecasting kind: series id, time index, horizon, forecast moment; rolling-origin backtests with
a gap; past-only lags and windows; naive and seasonal-naive baselines; MASE, sMAPE, quantile loss; a global boosted
model and a per-series classical model; intervals from backtest residuals; a check for covariates unknown at the
moment. Replica: store-product demand with a known-ahead promotion and a leaky same-day covariate.
```

### 16.4 Time to event (censoring)

- **Delivers:** a survival kind: a duration and an event flag (censoring), a start moment; Kaplan-Meier and Cox
  baselines, gradient-boosted survival and discrete-time classification; C-index, integrated Brier score and
  time-dependent AUC at named horizons; checks for durations measured past the moment and for treating censored rows
  as events.
- **Done when:** the replica (subscriptions with right-censoring) is solved, a model that drops censored rows is shown
  to be biased against the honest one, and survival curves per group are in the final record.

```text
Package 16.4. Add the time-to-event kind (duration, event flag, start moment): Kaplan-Meier and Cox baselines,
boosted survival and discrete-time models, C-index, integrated Brier score, time-dependent AUC; checks for censored
rows treated as events or dropped. Replica: subscriptions with right-censoring.
```

### 16.5 Ranking and recommendation

- **Delivers:** a ranking kind: users, items and interactions (implicit or explicit) with timestamps; splits by time
  per user or by held-out users; popularity and recently-popular baselines; matrix factorisation and a gradient-boosted
  ranker on candidate features; NDCG@k, recall@k, coverage and novelty; checks for interactions after the
  recommendation moment and for items that did not exist yet.
- **Done when:** the replica (a shop's views and purchases) is solved, every model is compared with popularity, and a
  future-interaction leak is caught.

```text
Package 16.5. Add the ranking kind: user-item interactions with time; per-user time splits or held-out users;
popularity baselines; matrix factorisation and a learning-to-rank model on candidates; NDCG@k, recall@k, coverage;
checks for future interactions and not-yet-existing items. Replica: a shop's views and purchases.
```

### 16.6 Segmentation (no target)

- **Delivers:** a segmentation kind for problems without an outcome: scaling and encoding choices recorded; k-means,
  Gaussian mixtures and HDBSCAN; the number of segments chosen by stability (bootstrap agreement) and silhouette, not by
  one run; each segment described in the data's own columns; an honest statement that no ground truth exists, and an
  optional check against a business outcome if one is named.
- **Done when:** the replica (customers with three planted behaviours and noise) is recovered with stability
  intervals, and a too-high number of segments is rejected for instability.

```text
Package 16.6. Add the segmentation kind: recorded preprocessing, k-means, mixtures and HDBSCAN, the number of segments
by bootstrap stability and silhouette, segment profiles in the data's columns, and the "no ground truth" limit in the
record. Replica: customers with planted behaviours.
```

### 16.7 Uplift: who responds to an action

- **Delivers:** an uplift kind: a treatment column, an outcome, and whether assignment was randomised; T- and
  X-learners and uplift trees; Qini and AUUC with intervals; propensity overlap checks; a refusal to call effects causal
  when assignment was not randomised and overlap is poor, said in the record.
- **Done when:** the replica (a randomised campaign with a planted responsive segment) recovers the segment, and an
  observational variant raises the overlap warning.

```text
Package 16.7. Add the uplift kind: treatment, outcome, randomised or not; T- and X-learners, uplift trees; Qini and
AUUC with intervals; propensity overlap; no causal wording without randomisation or overlap. Replicas: randomised and
observational campaigns.
```

### 16.8 Several tables: entity targets from event tables

- **Delivers:** a project can hold several related tables (an entity table, event tables, label tables) with declared
  keys; the target and features are built per entity *as of a cutoff time* (counts, recencies, sums and trends over
  windows before the cutoff), with the cutoff per row recorded; generalises 15.2 (label files) and 15.4 (behaviour
  counts); checks for events after the cutoff and for one-to-many joins that duplicate rows.
- **Done when:** the replica (customers, orders, support tickets; churn in the 30 days after a cutoff) is solved with
  no event after a row's cutoff used (tested), and a duplicating join is refused with the numbers.

```text
Package 16.8. Let a project hold related tables with keys. Build per-entity targets and window features as of a
per-row cutoff, record the cutoffs, and refuse joins that duplicate rows. Generalise 15.2 and 15.4 onto it. Replica:
customers, orders and tickets with a churn window after the cutoff; a test that no later event is used.
```

### 16.9 Decisions on top of a model: thresholds, prices, budgets

- **Delivers:** a decision layer for any kind that predicts: the person states the action, its value and its cost
  (Home already asks what a wrong prediction costs), and DCLab chooses the threshold, the review budget or the price
  that maximises expected value on out-of-fold predictions, with an interval and a stated assumption about how the
  outcome responds to the action. Price optimisation uses a model of acceptance by price (as in the challenge's first
  part) and warns when prices in the data were not varied enough to learn a response.
- **Done when:** the fraud replica gets a budget from the review cost, a pricing replica gets a price curve with an
  interval, and a pricing replica with fixed prices gets the "cannot learn the response" warning.

```text
Package 16.9. Add a decision layer: from the stated value and cost of an action, choose a threshold, review budget or
price maximising expected value on out-of-fold predictions, with intervals and the assumptions written. For prices,
model acceptance by price and warn when prices did not vary. Replicas: fraud review budget; ride pricing with and
without price variation.
```

### 16.10 Places: spatial splits and features

- **Delivers:** a spatial part usable by any kind with coordinates: grid or H3-style cells, distances, neighbourhood
  counts fitted inside folds, and spatial block cross-validation when places cluster; a check for random splits on
  spatially autocorrelated data; ties the `maps` pack to it.
- **Done when:** a replica with spatial autocorrelation shows the random split's optimism against spatial blocks, and
  the engine picks blocks for it.

```text
Package 16.10. Add spatial features (cells, distances, fold-fitted neighbourhood counts) and spatial block CV for any
kind with coordinates, a check for random splits on autocorrelated places, and wire the maps pack. Replica with
spatial autocorrelation.
```

### 16.11 Sequences of events and sensor anomalies

- **Delivers:** a kind for streams (machine sensors, logs, sessions): windows over the stream, features from each
  window's past, labels that may be sparse or delayed; detectors from 15.5 applied per window and supervised models
  when labels exist; event-based metrics (detection delay, alerts per day, recall of incidents rather than of rows);
  splits by time and machine.
- **Done when:** the replica (machines with planted failures and their warning signs) reports detection delay and
  incident recall, with no window using the future.

```text
Package 16.11. Add a stream kind: windows, past-only window features, sparse or delayed labels, detectors and
classifiers, incident-level metrics (delay, recall of incidents, alerts per day), splits by time and machine.
Replica: machines with planted failures.
```

### 16.12 Images and long text through pretrained embeddings

- **Delivers:** images and long text turned into features by a pretrained encoder run locally and cached (no training
  of large networks on this machine), then any tabular kind on top; splits by scene, document source or author;
  checks for near-duplicate images or texts across the split. Fine-tuning stays in the `llm` pack's own flow.
- **Done when:** an image replica (synthetic shapes with a planted label) and a text replica are solved with the
  embeddings cached by content hash, and near-duplicates across folds are found.

```text
Package 16.12. Add embedding features for images and long text from a local pretrained encoder, cached by content
hash; grouped splits by source; a near-duplicate check across folds. Replicas for images and text. No large-network
training on this machine.
```

### 16.13 Challenge mode: from a brief and files to every deliverable

- **Delivers:** a challenge can be started from a brief (a PDF or text) and its files. DCLab reads the brief into a
  requirement list with the rubric's points when given (models asked for, report sections, API, Docker, source
  control, tests), detects the kinds involved (a challenge can hold several parts), builds one project per part, and
  produces the deliverables: the notebook per part, the report as a PDF with the sections the rubric names, the
  scoring service (15.10), tests, and a git repository; plus a coverage table mapping every requirement to the file
  that meets it, or saying it is not met and why. Confidentiality words in the brief ("must not be disclosed")
  mark the data as private: it stays out of exports, prompts and training data.
- **Done when:** a written mock challenge (two parts: a classification with a pricing question and a fraud part, on
  replicas) is solved end to end with every rubric line covered or explained, and a test proves the private data did
  not enter any export or model request.

```text
Package 16.13. Add challenge mode: parse a brief into requirements (and rubric points), detect the kinds of each
part, build a project per part, and produce notebooks, a PDF report with the named sections, the service export,
tests and a git repository, plus a requirement-coverage table. Respect confidentiality words: private data never
leaves the workspace. Acceptance: a two-part mock challenge on replicas.
```

### 16.14 Agents and evidence per kind

- **Delivers:** each kind has a research track with a first campaign (as 15.8 for fraud), its pitfalls measured, its
  traps in the benchmark (a new benchmark version when the sealed set changes), and its tools in the registry (the
  intern can run a kind's checks); lessons are filed by kind; SFT trajectories carry the kind, and no kind's sealed
  cases ever enter a training set.
- **Done when:** for each supported kind an agent asked a method question cites that kind's evidence, and the
  benchmark reports results per kind with intervals.

```text
Package 16.14. For each supported kind: scaffold its research track and first campaign, measure its pitfalls, add its
traps to a new benchmark version, register its tools in the agent registry, file lessons and SFT trajectories by kind.
Test that method questions cite the kind's evidence.
```

# Part 6 — Understand, analyse, decide: a portable solving core

**Why this Part.** The owner, 2026-10-12: before anything kind-specific, DCLab must first *understand the problem or
challenge as it is described*, then *analyse the data* for the decisions it implies, for binary, multiclass and
regression alike, and solve problems better by improving **the solution and the solution workflow**, not the code
around it. And: **what is built must be adoptable by DCLab's main product, not only by this R&D project.**

**What "the solution workflow" means here.** A careful data scientist works in this order, and every step leaves a
decision with its reason:

1. **Understand**: the goal and the action a prediction feeds; what one scored row is; the target, or that there is
   none; where the labels come from and whether they are complete; the prediction moment; the entities a row belongs
   to; constraints (confidentiality, latency, review capacity); the deliverables and how they are judged.
2. **Analyse for decisions**: what a row really is (grain, duplicates, several rows per entity), which keys join the
   files, how the target behaves (prevalence; class sizes; skew, caps and zeros), time (range, drift), how the target
   concentrates on entities, what looks like leakage, what is missing and whether missingness speaks. Each finding
   ends in an *implication* ("150 rows per biker and labels on a few bikers → folds grouped by biker").
3. **Plan**: framing, split, metric (from the cost of a wrong decision), baselines, feature directions, model
   families, the risks to check, and what would change the plan, each with its reason and the alternatives set aside.
4. **Run** the plan through the ten workflow steps (the engine as it is, then Parts 4 and 5).
5. **Review and improve**: error analysis (by segment, class or residual), hypotheses, the next experiments ranked by
   their expected value, a bounded number of rounds, the holdout untouched until the end.
6. **Report**: the problem, the findings, the decisions and their reasons, the results with their uncertainty, the
   limits and the next steps.

Today DCLab does part of 1 (Home asks up to four questions: target, moment, action and cost, data), part of 2 (a
profile and the leakage audit, not decision-oriented) and 4; 3 is `Solution.propose()` heuristics; 5 and 6 are thin.

**Adoptable by the main product: the most important constraint.** The main product lives in its own repository
(github.com/Shahriyar-Moradi/DCLab) and stays there: nothing of it is copied into this one. It was read on 2026-10-12
(through `gh`, read only): a Python 3.12 FastAPI backend (`apps/api/app`: `api/` transport, `services/`
orchestration, `domain/` contracts, `engine/` pure ML on DataFrames and typed plans) and a Next.js front end. Its
engine already has planners (`engine/modeling/`: `ProblemProfile`, `MetricPlan`, `ValidationPlan`, `HoldoutPlan`,
`Objective`), a versioned `ProblemSpec` (`services/problem_spec_service.py`), an `ExperimentPlan`
(`domain/run_plans.py`) with one governed decision point per field, and `engine/investigate/` trust checks. Its rules:
one training path, statistics on the training partition only, the holdout scored once, every model call through its
gateway with a deterministic fallback, compact evidence only (never rows), and "no new parallel system for an
existing concern". `docs/DCLAB_MASTER_CONTEXT.md` adds the bar for R&D work to enter it: reproducible evidence,
defined user value, an integration design, a security and cost review, and stable interfaces.

So the solving logic is built as **a pure library**, `dclab_core/`, at this repository's root, which the main product
installs with pip (pinned) the way it packages `packages/dclab_client`:

- it depends only on numpy, pandas, scikit-learn and pydantic, runs on Python 3.10-3.13, and never imports
  `dclab_rnd` (a test fails if it does); it never trains a model, calls a model provider or touches a database;
- it takes a DataFrame and an explicit training index, and never computes a statistic on rows outside it;
- every output (problem brief, data findings, solution plan, decisions, review, report) is a frozen pydantic model with
  a version string (`dclab_core.problem_brief.v1`) and a reason, with an exported JSON Schema;
- a model is used only through an injected `Advisor` protocol (`ask(purpose, evidence) -> validated answer or None`);
  `None` means the deterministic path, which always works; evidence passed to it is aggregates, never rows;
- outputs are shaped to map onto the main product's contracts (`ProblemSpec`, `ProblemProfile`, `MetricPlan`,
  `HoldoutPlan`, `ValidationPlan`, `ExperimentPlan` and its decision points, `Objective`), with the mapping written and
  tested here from field lists, so the main product adopts proposals through its own governed decisions, never a
  second pipeline;
- the R&D studio is one consumer (an adapter in `dclab_rnd`); the main product is the other.

The solving parts of Parts 4 and 5 (the label review, alert-budget metrics, behaviour recipes, model families, the
problem-kind interface) are built inside `dclab_core` for the same reason, with the studio calling them.

| # | Package | Status |
|---|---|---|
| 17.1 | The portable core and its contracts | Done (2026-10-12; `dclab_core/` with its own `pyproject.toml`, `contracts.py`, `evidence.py`, `advisor.py`, `schemas.py` and `schemas/`, `dclab_rnd/core_adapter.py`, `tests/test_core_contracts.py`, `make core-schemas`): six versioned contracts (problem brief, data findings, solution plan, decision, review, report), frozen, refusing unknown fields and NaN, read only with their version (`Versioned.read`); typed facts (stated with a quote from the text or a person, inferred with a reason, unknown without a value), with the same rules in the published JSON Schemas (checked with jsonschema). The brief covers the main product's ProblemSpec fields (objective, task type, target column, unit, prediction time column, horizon, primary metric, constraints, success criteria) plus a target definition for labels kept in another file. Evidence is a summary by one rule for findings and for the advisor (no frames, arrays, sets, bytes, long lists, big mappings or tables of records; numpy scalars become numbers). The advisor is injected; `consult` checks the evidence, returns None on abstain, failure or an invalid answer, and reports failures to `on_error`. Each version's shape is frozen in `schemas/baseline/`: changing a contract without moving its version fails a test. Boundary: the AST test refuses dclab_rnd, any third-party module beyond numpy, pandas, scikit-learn and pydantic, and the standard library's database, network, process and dynamic-import modules; a subprocess checks what importing the core loads; the code parses as Python 3.10. The wheel was built and installed in a clean Python 3.12 environment (numpy 2.5, pandas 3.0, scikit-learn 1.9, pydantic 2.14) and used from an unrelated folder. The studio adapter turns a project into a brief and a plan and back without loss (forbidden columns included) and calls a field the person's statement only when the solution is signed. Reviewed (13 findings, all fixed with tests). Differs from the text: the model protocol is `Advisor`, not `ModelClient`, and the pydantic floor is 2.10. Open: nothing reads these contracts yet in the main product (17.8) |
| 17.2 | Understand the problem: from a description to a problem brief | Done (2026-10-12; `dclab_core/understand.py`, `tests/test_core_understand.py`, Home's `read_brief` in `draft/chat.py`, the intern's `understand_problem` tool and the compact brief in `describe_data`): a description and its files' column names give a ProblemBrief (goal, action, unit, task, target column or definition, label source and coverage, moment, horizon, entities, time column, primary metric, constraints, deliverables with rubric points, success criteria, open questions ranked by impact). The four written briefs (a fraud challenge with a rubric, churn, a delivery-time regression, ticket routing) give the expected briefs with and without a scripted advisor. The rules are narrow and give up rather than guess, because a wrong moment or target is a leak: a moment inside the outcome's clause, a bare 'at the moment' or a 'when' clause whose verb does not start the process is never read; a column is the target only when the sentence binds it; a review capacity needs a review verb and an error cost an error word; the task is read from the goal sentence first. Seven adversarial sentences from the review are negative tests. An advisor's proposal is kept only for an unknown field, with whole quoted words (Home's rule), a quote that supports the value, and a valid value (a task type, an existing column); a failing or wrong advisor changes nothing. Long pasted text is clipped (sentences split at real breaks, soft wraps joined; lists cut to the contract's limits), never an error. Home keeps the brief on the draft (rebuilt when the data is ready) and takes from it only a bound target and the action in the user's words; the moment stays with Home's own rules. Reviewed (8 findings, all fixed with tests). Open: Home does not yet ask the brief's label-coverage and entity questions (15.6); the intern's model prompt still names describe_data first, which now carries the compact brief; during this package some tests were run directly without DCLAB_NO_LIVE_MODELS=1 and made about 50 live requests (about EUR 0.03) |
| 17.3 | Initial data analysis for decisions, for every task | Done (2026-10-12; `dclab_core/analyse.py`, `dclab_core/replicas/{ticket_routing,delivery_time}.py`, `tests/test_core_analyse.py`): tables, a brief and the host's training rows give DataFindings, each with evidence (numbers and names) and an implication naming the decision points it bears on. Anything that reads the target is computed on the training rows only (target behaviour, conflicting labels, entities, drift, leakage, missingness); structure without the target on all rows (duplicates, identifiers, constant and impossible values), each finding saying which. Every claim about the target is measured out of fold against chance (two seeded halves): a column's single-column power (AUC; gain over the majority normalised to 0-1; |Spearman|), an entity's power to tell the target, so shuffled labels raise nothing (tested). A name alone never forbids: a post-outcome name without history words is a question (info) unless the column also predicts the target. Regression: a cap only as a spike at the maximum, the tail by skew or the quantile tail ratio (robust to the cap), a target recomputed from two timestamps (the later one named). Label files: the best key, pairs of id-like keys, rows marked per label, ids compared as text (no float rounding), and same-named columns that disagree. The three replicas' traps are named with their implications and hand-checked numbers (multiclass: tiny classes, a label copy, duplicates, a late class; regression: the cap, the tail, the recomputable target, a post-delivery rating, negative distances, missingness in storms, restaurants; binary: rarity, partial labels, the join and its inflating pairs, the disagreeing column, frauds on bikers). 200,000 rows by 48 columns: about 7 s. Reviewed (14 findings, all fixed with tests: train-only conflicts, normalised multiclass power, out-of-fold entity concentration, honest history names, decimal columns are not identifiers, 1/2 labels, long ids, discrete scales, multiclass missingness, late positives, pandas 3 string dtype, odd names and long lists never raise, vectorised modes). The replicas moved to `dclab_core/replicas/` (15.1's included) so the main product can run them. Open: the studio does not call `analyse` yet except through 17.4's plan |
| 17.4 | The solution plan: decisions with reasons and alternatives | Done (2026-10-12; `dclab_core/plan.py`, `tests/test_core_plan.py`, `core_adapter.solution_fields_from_plan`, the plan in `studio.solution.propose`, the intern's `adopt_plan`): a brief and the findings give one Decision per point (framing, label coverage and join, split, primary and secondary metric, baseline, columns kept out, columns to confirm, identifiers, feature directions, cleaning, families, threshold policy), each citing the findings it rests on with the alternatives set aside, plus risks and what would change the plan. On the three replicas the plans reach the acceptance decisions (fraud: partial labels, join on the order with the inflating keys refused, folds by biker, average precision with recall at the budget, PU and anomaly families; tickets: a forward split because a class appears late, macro F1, resolved_by kept out and first_response_minutes to confirm; delivery: folds by restaurant, MAE with the share within tolerance, the cap, the recomputed delivery time and the rating kept out). The grouping entity is the one the person names, else the one that tells the target out of fold, with at least 10 values (account_id, 3 values, is a refused alternative). `validate` refuses a critical leak not kept out, a random split against an entity, a metric of another task, an inflating label key, unknown citations and malformed choices; only a person's override sets a finding aside, a model's is checked like any proposal. `propose()` adds the compact plan (about 2 KB, no row values) read on a seeded 80 % of rows, logs when the core cannot plan, and leaves its own fields unchanged; the intern sees it and adopts it through the set_solution move (graph check, log, memory, audit), refused when an override ignores a finding. The judgment suite was re-run for the two new project tools (AEV-015, scores identical to AEV-014). Reviewed (11 findings, all fixed with tests). Open: the engine still reads the solution, not the plan's secondary metrics, cleaning or directions (17.5); there is no person-facing route to override one decision yet |
| 17.5 | A better workflow per task: binary, multiclass, regression | Done in part (2026-10-12; `dclab_core/assess.py`, the engine's final stage records `task_assessment`, `tests/test_core_assess.py`): binary: the alert threshold from the adopted plan's `threshold.policy` (a review capacity as a share of rows, "0.5%" or "200 a day" over the training span; stated costs when both errors have a number; otherwise the F1-best one, said so, with a note to ask), chosen over every distinct out-of-fold score and read once on the holdout; calibration (Brier, ECE); precision marked a lower bound when the plan says the labels are partial. Multiclass: per-class precision and recall, the most confused pairs, the tiny classes, top-k accuracy, macro recall over every class. Regression: intervals from out-of-fold residuals with their holdout coverage (0.81 for 80% on the delivery replica), error by prediction range and by segment (columns chosen on training rows), the cap from training rows (`analyse.target_cap`, shared, never a discrete scale), whether errors grow with the prediction. The tickets and delivery replicas' records carry them; a binary engine test checks the plan's budget and the partial-label note. Fixed on the way: a training fold without a class crashed XGBoost in the engine (`runner.fit_and_predict` fits codes of the classes present) and gave wrong columns in the exported notebook (`export.MULTICLASS_HELPERS`). Open, moved to 17.6 as experiments the review proposes from these measured weaknesses: choosing a target transform, merging tiny classes, and the improvement margins on the multiclass and regression replicas (17.5 measures, it does not change the model); top-k is reported whatever the action |
| 17.6 | Review and improve: error analysis and the next experiments | Not started |
| 17.7 | The solution report, from the records | Not started |
| 17.8 | The adoption kit for the main product | Not started |

## 17. Understand, analyse, decide

### 17.1 The portable core and its contracts

- **Delivers:** `dclab_core/` with its own `pyproject.toml` (installable as a wheel), `contracts.py` (frozen pydantic
  models with a version string: `ProblemBrief`, `DataFindings`, `SolutionPlan`, `Decision`, `Review`,
  `SolutionReport`), `make core-schemas` writing their JSON Schemas under `dclab_core/schemas/` (generated, checked
  fresh by `make rd-check`), the `Advisor` protocol, and boundary tests: importing `dclab_core` imports nothing from
  `dclab_rnd`, it runs with only its own requirements, and no module uses syntax beyond Python 3.10 or below 3.13.
  The studio gets an adapter that maps a project to and from these artefacts.
- **Done when:** the schemas are generated and fresh, the boundary test passes, a round trip project → artefacts →
  project loses nothing, and a breaking change to a contract fails a test unless its version moves.

```text
Package 17.1. Create dclab_core/ at the repository root as an installable library (its own pyproject): frozen,
versioned pydantic contracts (ProblemBrief, DataFindings, SolutionPlan, Decision, Review, SolutionReport), generated
JSON Schemas (make core-schemas, freshness in rd-check), an Advisor protocol for model help with a deterministic
fallback, and tests that dclab_core never imports dclab_rnd and needs only numpy, pandas, scikit-learn and pydantic.
Add the studio adapter (project <-> artefacts) with a round-trip test.
```

### 17.2 Understand the problem: from a description to a problem brief

- **Delivers:** `dclab_core.understand`: from a description (a sentence, a page, a challenge's PDF text) and the file
  names, a `ProblemBrief`: goal and action, scored unit, target (or none), label source (a column, another file,
  none) and coverage (complete or a sample), prediction moment, entities, time, constraints (confidentiality words,
  latency, review capacity), deliverables and the rubric with its points, success criteria, and the open questions,
  ranked by how much the answer changes the plan. Each field records its status (stated, inferred, unknown) and, when
  stated, the quote it came from; a model may propose fields, kept only when the quote is in the text (Home's rule).
  Home and the intern start from the brief; Home asks the top open questions.
- **Done when:** four written briefs (the delivery-fraud challenge in invented words, a churn brief, a delivery-time
  regression brief, a ticket-routing multiclass brief) give the expected briefs without a model and with a scripted
  one, no field is filled without a quote or a stated inference, and the fraud brief lists the API, Docker and report
  as deliverables with their points.

```text
Package 17.2. Add dclab_core.understand: description and file names -> ProblemBrief, every field with status and
quote, open questions ranked by impact on the plan, rubric and deliverables when present, confidentiality words
noted. Deterministic first; a ModelClient may propose, kept only with a quote. Home and the intern start from it.
Tests: four written briefs (fraud, churn, delivery-time regression, ticket routing), with and without a scripted model.
```

### 17.3 Initial data analysis for decisions, for every task

- **Delivers:** `dclab_core.analyse`: from the tables (one or more) and the brief, `DataFindings`, each finding with
  its evidence (numbers, never row values) and its *implication* for the plan. Common: grain and duplicates, keys and
  joins across files (with what a join would mark, as in 15.2), rows per entity and target concentration by entity,
  time range and drift, leakage suspects (the existing audit), missingness and whether it predicts the target,
  types, cardinalities and impossible values. By task: binary (prevalence, rare-event regime, label coverage);
  multiclass (class sizes, tiny classes, classes to merge or a hierarchy, confusable classes); regression (skew, caps,
  zeros, outliers, units, censoring hints). Computed on the training part where it would steer choices (WF-04,
  train-only EDA). New replicas: ticket routing (multiclass: rare classes, a leaky "resolved_by" column, duplicated
  tickets) and delivery time (regression: heavy tail, a cap at 60 minutes, a leaky "actual_arrival").
- **Done when:** on the three replicas the findings name the traps each one plants, with the right implication, and the
  numbers match hand computations in the tests.

```text
Package 17.3. Add dclab_core.analyse: tables + brief -> DataFindings (evidence and implication per finding), common
checks and per-task checks for binary, multiclass and regression, train-only where it steers choices. Add the
ticket-routing (multiclass) and delivery-time (regression) replicas with their traps. Tests: each replica's findings
name its traps with the right implication and hand-checked numbers.
```

### 17.4 The solution plan: decisions with reasons and alternatives

- **Delivers:** `dclab_core.plan`: brief + findings -> `SolutionPlan`: framing (for example regression or a threshold
  classification; flat or hierarchical classes; partial labels), split, primary metric from the decision's cost
  (binary: AP, recall at a budget, cost; multiclass: macro-F1, top-k, cost matrix; regression: MAE, RMSE, a quantile
  loss, error in percent), baselines, feature directions, model families, risks to check, and "what would change this
  plan". Each decision is a `Decision` (choice, reason citing findings or evidence records, alternatives with why not).
  `Solution.propose()` and the engine read the plan; a person can override any decision with a reason, and the
  override is a recorded move.
- **Done when:** on the three replicas and the fraud replica the plan's decisions match what the acceptance tests
  expect, every decision cites a finding, and an override reaches the engine and the log.

```text
Package 17.4. Add dclab_core.plan: brief + findings -> SolutionPlan of Decisions (choice, reason with citations,
alternatives). Metrics from decision cost per task. Make Solution.propose() and the engine use it; overrides are
recorded moves. Tests on the four replicas.
```

### 17.5 A better workflow per task: binary, multiclass, regression

- **Delivers:** what a careful data scientist adds per task, as stage evidence the engine records: binary
  (a threshold from the stated cost or budget, calibration, precision as a lower bound with partial labels);
  multiclass (per-class metrics, the confusion matrix with the most confused pairs, a class-merge or hierarchy check
  for tiny classes, top-k when the action allows several); regression (target transform chosen by the residuals,
  prediction intervals from out-of-fold residuals, error by segment and range, behaviour at caps). Each addition comes
  from the plan, never on by default without a reason.
- **Done when:** each replica's final record carries its task's additions with numbers, and the multiclass and
  regression replicas each improve on the engine as it was by the margin their acceptance tests set.

```text
Package 17.5. Add per-task workflow steps to the engine from the plan: binary thresholds and calibration; multiclass
per-class metrics, confusion pairs, merge or hierarchy checks, top-k; regression target transforms, residual-based
intervals, error by segment. Record them as stage evidence. Acceptance on the multiclass and regression replicas.
```

### 17.6 Review and improve: error analysis and the next experiments

- **Delivers:** `dclab_core.review`: after the models stage, error analysis (where the model fails: segments,
  classes, ranges, entities), hypotheses that follow from it, and the next experiments ranked by expected gain and
  cost; the engine runs at most N rounds on training folds only, each round a recorded decision; the holdout stays
  untouched until the final stage (the existing gate). A model may propose hypotheses; code checks that each one cites
  a measured weakness.
- **Done when:** on the fraud replica a round finds the behaviour features (or on the regression replica the cap)
  from the error analysis, the improvement is measured on folds, and the holdout is used once.

```text
Package 17.6. Add dclab_core.review: error analysis by segment, class, range and entity; hypotheses citing measured
weaknesses; next experiments ranked by expected gain and cost; at most N recorded rounds on training folds; holdout
once. Tests on the fraud and regression replicas.
```

### 17.7 The solution report, from the records

- **Delivers:** `dclab_core.report`: a `SolutionReport` and its rendering (Markdown, HTML, PDF): the problem as
  understood, the data findings, the decisions and reasons, the results with intervals, the limits (including what
  the data cannot show), the next steps, and the deliverables map when a rubric exists. Every number comes from a
  record; the prose goes through the citation check.
- **Done when:** the report of each replica project renders, every number in it is traced to a record by a test, and
  no sentence calls a model production-ready.

```text
Package 17.7. Add dclab_core.report: SolutionReport from the records, rendered to Markdown, HTML and PDF; every number
traced to a record; prose through the citation check. Tests on the replica projects.
```

### 17.8 The adoption kit for the main product

- **Delivers:** `dclab_core` built as a versioned wheel with a changelog; `dclab_core.adapters.main_product`: pure
  functions mapping each output onto the main product's contracts by their field lists (`ProblemSpec` content,
  `ProblemProfile`, `MetricPlan`, `HoldoutPlan`, `ValidationPlan`, `ExperimentPlan` per decision-point key,
  `Objective`), without importing the main product; `docs/guides/CORE_INTEGRATION.md`: where each call goes in the
  main product (the problem-spec service and wizard; the load-profile stage after the holdout lock; the experiment
  planner and investigator agents as proposals through governed decision points; the experiment review and
  improve decision points; the technical report), the `Advisor` to implement over its gateway, what never crosses
  (rows, holdout labels), and the tests it should add on its side (holdout isolation, the benchmark harness).
- **Done when:** the wheel builds and installs in a clean Python 3.12 environment with only its dependencies, the
  mapping tests pass against the field lists recorded from the main product's code (with the commit read), and the
  guide names every call site. Wiring it into the main product is done in that repository, as its own change.

## Suggested order for Parts 4, 5 and 6

1. **15.1** (done: the fraud replica and its acceptance test).
2. **Part 6 first:** **17.1** (the core and contracts), **17.2** (understand), **17.3** (analyse, with the multiclass
   and regression replicas), **17.4** (plan), **17.5** (per-task workflow), **17.6** (review and improve), **17.7**
   (report). These raise the quality of every solution, whatever the kind.
3. **16.1** and **16.2** inside `dclab_core` (problem kinds and their acceptance harness).
4. **15.2-15.6**, **15.10**, then **16.13** (challenge mode): DCLab solves a challenge like the owner's end to end.
5. **15.7-15.9** beside 4; the other kinds of Part 5 by the owner's priority; **15.11** with the owner.
6. **17.8** after 17.4 (the first outputs worth adopting), then kept current with every package: each new output
   gets its mapping onto the main product's contracts in the same package.


# Part 7 — The agentic layer: models that understand, investigate, plan, critique and improve

**Why this Part.** The owner, 2026-10-12: involve more AI, agentic AI and LLMs inside the workflow, to understand
problems better, build better workflows, decide better and do more, across the variety of machine-learning problems.

**How, without losing what makes DCLab careful.** Models are strong at meaning and judgment: reading a brief, knowing
what a column of a payments table usually means and when it is written, recalling which features a domain uses,
criticising a plan, explaining a result. Code is strong at what must be exact: statistics, splits, fits, metrics and
the selection rules. So every agent here *proposes*, and code *checks and executes* (the main product's rule as
well: "agents propose; deterministic services validate"). Each agent:

- is a role in `dclab_core/agents/` with a typed input (aggregates and names, never rows), a typed output, a validator
  (quotes in the person's words, citations to findings or evidence records, values valid for the task) and a
  deterministic fallback, so DCLab works with the model off and degrades to the rules when it abstains or fails;
- has its prompt as a versioned file in `dclab_core/prompts/` (hash-checked, as in A4.3), so the main product runs the
  same prompt through its own gateway (the `Advisor` protocol of 17.1);
- records each proposal as a `Decision` with `source="advisor"`, kept, changed or refused by code or a person, and
  traced (the agent-step traces of A2);
- is on by default only after it is *measured*: on the replicas and the benchmark it must beat or match the
  deterministic path on the quality that matters (traps caught, decisions right, score gained) with no more unsafe
  moves, at a known cost per solved problem.

The roles, at the workflow's steps:

| Role | Step | Proposes | Checked by |
|---|---|---|---|
| Problem analyst | understand (WF-01) | brief fields the rules left open, the clarifying questions that matter most, the domain framing | `understand`: whole quoted words that support the value |
| Data investigator | analyse (WF-04, WF-05) | when each column is written relative to the moment, leaks to test, entities, domain meanings | each claim cites a column and becomes a check code runs (`analyse`, the leakage audit) |
| Solution planner | plan (WF-01, WF-03) | framing, split, metric from the cost of errors, families, risks | the plan's validator: decisions valid for the task and the findings (no random split against an entity finding without a reason) |
| Feature engineer | features (WF-06) | feature recipes in a small language (counts per entity and place, ratios, distances, time since, rolling windows) | compiled and fitted inside the training folds by code; kept only when cross-validation says so |
| Critic | review (WF-07 to WF-09) | weaknesses, hypotheses, the next experiments | `critic_gate`: every claim against the recorded numbers; hypotheses must cite a measured weakness |
| Experimenter | improve loop | which experiment to run next, when to stop | budgets (steps, euros), training folds only, holdout once |
| Reporter | report (WF-10) | the prose of the report and the explanations | `cited.py`: every sentence cites a record, every number is in it, no production-ready claims |
| Challenge coordinator | challenge mode (16.13) | the parts of a brief, their kinds, the requirement map | the requirement coverage table, checked against the files produced |

| # | Package | Status |
|---|---|---|
| 18.1 | The agent contract in dclab_core: roles, prompts as files, a host runner | Done (2026-10-12; `dclab_core/agents/`, `dclab_core/prompts/` with `HASHES.json`, `tests/test_core_agents.py`): a Role names its purpose and prompt file, builds compact evidence (checked: no rows), parses an answer into a validated output and has a deterministic fallback; `run_role` returns the output and whether it came from the advisor or the fallback and why (no advisor, abstained, failed, invalid), identical to the fallback in every case but an accepted answer. The problem analyst of 17.2 now runs as the first role with its pinned prompt; prompts ship in the wheel. Open: the R&D host's Advisor over the gateway and the live measurement (18.2) |
| 18.2 | The problem analyst, measured | Done (2026-10-12; the `problem_analyst` purpose (the description and file and column names, never values; its data-limit test), `dclab_rnd/core_advisor.py` (the gateway Advisor: one capped client per purpose; a reply cut at the token limit or without JSON is a failed request, never an abstention), `dclab_rnd/agent_eval/analyst/` (24 written briefs of every kind, dev 12 and sealed 12, fingerprinted; three readers: the rules alone, the rules with the analyst, a plain prompted model; every field scored right, wrong, missed or abstained), gate analyst_gate_v1 fixed before any run, `tests/test_analyst_measure.py`). Sealed set, one run (ANA-004, gpt-6-luna, €0.0074): rules alone 15 right and 4 wrong of 96 fields; with the analyst 72 right, 4 wrong, none new (its 4 errors are the rules' own); a plain prompted model 71 right, 1 wrong. Gate: pass. All four runs cost €0.0295. Tuned on dev before the sealed run: the scorer (a moment's "before …" clause, plurals, formatting) and the prompt's definition of the horizon (ANA-001 to ANA-003). One sealed gold was made stricter before its run (s06: "before the next review" is the end of the outcome window, not the moment). Limits: 12 sealed briefs I wrote, one model, one run each; the free-text golds accept any on-topic phrase, so goal and action are lenient. Open: the analyst cannot correct a rule's wrong value (only unknown fields are offered), which is where every remaining error sits; the next step is a disagreement the analyst raises as an open question, never a silent override. Home and the intern do not use it yet (switching it on is the owner's decision, with this result) |
| 18.3 | The data investigator, measured | Not started |
| 18.4 | The solution planner, measured | Not started |
| 18.5 | The feature engineer and its feature language | Not started |
| 18.6 | The critic and the improve loop | Not started |
| 18.7 | The reporter | Not started |
| 18.8 | The multi-agent workflow over WF-01 to WF-10, with gates and budgets | Not started |
| 18.9 | The agentic layer judged: agents against rules and a plain model, per role | Not started |

## 18. The agentic layer

### 18.1 The agent contract in dclab_core

- **Delivers:** `dclab_core/agents/`: a `Role` (name, purpose, input schema of aggregates, output schema, validator,
  fallback, budget), `run_role(role, advisor, inputs)` returning a validated output or the fallback with the reason,
  and the roles' prompts as files under `dclab_core/prompts/` with a hash test; a scripted advisor for tests; the R&D
  host runs roles through the gateway (`dclab_rnd.models`), the main product through its own (17.8).
- **Done when:** a role runs with a scripted advisor, with none, and with one that fails or answers badly, and the
  outputs are identical to the fallback in the last two cases; a changed prompt fails the hash test until recorded.

### 18.2 The problem analyst, measured

- **Delivers:** the analyst's prompt and wiring into `understand` (the hook exists); 24 written briefs of every kind
  (a dev set of 12 and a sealed set of 12) with the expected fields; a measurement of field accuracy and of *wrong*
  fields, with the rules alone, with the analyst, and with a plain prompted model.
- **Done when:** with the analyst, more fields are right and none is wrong that the rules left unknown (the gate is
  written before the run), measured on the sealed set once, with its cost.

### 18.3 The data investigator, measured

- **Delivers:** a role that reads `DataFindings` and the column names and kinds (never values), and proposes for each
  column when it is written relative to the moment, suspected leaks, entities and meanings; each proposal becomes a
  check that `analyse` or the leakage audit runs. Measured on the replicas and benchmark v3: planted traps found and
  clean columns falsely accused, with and without the role.
- **Done when:** it finds more planted traps than the rules alone without raising false alarms above the gate.

### 18.4 The solution planner, measured

- **Delivers:** a role that proposes the `SolutionPlan`'s decisions from the brief and the findings, each citing them;
  the plan validator (17.4) refuses invalid ones; measured against the replicas' acceptance decisions.
- **Done when:** its plans pass the validator and match the acceptance decisions at least as often as the rule-based
  planner, with every refusal recorded.

### 18.5 The feature engineer and its feature language

- **Delivers:** a small feature language (group counts and shares per entity, place and time window; ratios;
  distances between coordinate pairs; time since an entity's previous row; rolling statistics over past rows) that code
  compiles into fold-safe transformers (15.4's behaviour recipe becomes a set of programs in it); a role that proposes
  programs from the brief, the findings and domain knowledge; the feature ladder screens them.
- **Done when:** on the fraud replica the engineer proposes the repeated-drop-off and short-trip features without
  being told, and the ladder keeps them for a measured gain; no proposal can read the target or the holdout.

### 18.6 The critic and the improve loop

- **Delivers:** a critic that reads the stage records and the error analysis (17.6) and proposes hypotheses and
  experiments; an experimenter that runs the most promising within budgets on training folds; every claim through
  `critic_gate`.
- **Done when:** on the replicas the loop improves the cross-validated score over one pass, the holdout is used once,
  and every kept change cites the measured weakness it fixed.

### 18.7 The reporter

- **Delivers:** the report's prose (17.7) written by a model from the records and kept sentence by sentence only when
  `cited.py` passes it; the deterministic report when nothing passes.
- **Done when:** the replicas' reports are written with the model on and off, and no kept sentence fails the check.

### 18.8 The multi-agent workflow

- **Delivers:** the roles placed on the workflow graph's nodes (WF-01 to WF-10), passed the artefacts of the earlier
  nodes, under the graph validator, with a person's approval at the gates, budgets per project, and the conversation
  (Home and the intern) as the front that asks the person what only they can say.
- **Done when:** the fraud replica and a churn and a regression replica run end to end with all roles on, every move
  validated and traced, within a set budget, and the person asked only what the roles could not settle.

### 18.9 The agentic layer judged

- **Delivers:** a comparison per role and for the whole workflow: rules only, a plain prompted model, and the roles,
  on the replicas and the sealed benchmark, with intervals, unsafe moves, and euros per solved problem; the gate
  written before the runs.
- **Done when:** each role is on by default only where it passed its gate; the others stay available and off.

## Suggested order with Part 7

Part 7 is built beside Part 6, each role right after the deterministic step it advises: **18.1** after 17.3;
**18.2** now (the hook exists in 17.2); **18.3** with 17.3; **18.4** after 17.4; **18.5** with 15.4; **18.6** with
17.6; **18.7** with 17.7; **18.8** and **18.9** once the roles exist. Live runs go through the gateway with caps and
the owner's spending limits; tests use scripted advisors only.

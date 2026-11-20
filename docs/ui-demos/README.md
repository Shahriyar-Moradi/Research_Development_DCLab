# DCLab UI demos: versions

Each version of the product's UI design is kept here as a whole, standalone page. A new version is a new
folder (`v4/`, `v5/`, …); an old version is never edited, so any two can be compared side by side. A demo
changes no product code: the code follows a version only after the owner approves it.

| Version | Date | What it is | Files | Online copy | Code |
|---|---|---|---|---|---|
| v1 | 2026-10 | The first product demo, the frozen reference | `../product-demo/index.html` (tag `demo-v1`) | — | the product before the redesign |
| v2 | 2026-11-19 | Same layout, fixed: one name per page, a one-line project bar never made up, one main button, sample marked, Home's reply box opaque | `v2/index.html` (clickable, before/after), `v2/review.html` (the ten problems) | [demo](https://claude.ai/artifact/SmaSQYpaaP7YF7c2wcddsN) · [review](https://claude.ai/artifact/MDpwiCbBgE26nnZNKQyMNZ) | built on branch `ui-redesign`, commit `87b03509`, not merged |
| v3 | 2026-11-20 | Minimal and chat-first: sidebar of projects and recent chats, Library and Settings at the foot; no box above the chat; no machine facts in the top bar; a project is a page with tabs; a new Features tab | `v3/index.html` | [demo](https://claude.ai/artifact/YBGBRV8Hp4cN8xhQw9yQSA) | not built |
| v4 | 2026-11-20 | Three looks for one screen (the chat of a project), after the owner rejected v3's cream-and-terracotta look: A Bench (light, steps as a scale on the right), B Notebook (no side panels, turns as cells, yellow marks what needs an answer), C Control room (dark, a meter of passed checks) | `v4/index.html` | — | not built |
| v5 | 2026-11-20 | The owner rejected v4 too and gave a reference: a dashboard with a profile-topped sidebar, soft white cards, pill segmented tabs, green buttons and a grey-framed table. v5 applies that style to every page (Home, Chat, a project with Overview, Data, Features, Models and Report, Library, Settings), on a warmer background | `v5/index.html` | — | not built |
| v6 | 2026-11-20 | v5's style kept. Chat is Home; the dashboard moved into each project (Dashboard tab); a project has five tabs (Dashboard, Workflow, Data, Models, Brief), each with a page picker for its views; the owner's example pages (More tools, Workflow agents, Project overview, Experiment results) in that style; Library hidden, not deleted; a "Where everything went" page maps every page of today's product to its new place | `v6/index.html` | — | not built |

## v3 in short

- **Removed:** the Data · Plan · Run · Evidence box above the chat; `exp/churn`, `Py3.11 · DuckDB` and the search box in
  the top bar; the sidebar's group headers; the Developer / Data science switch in the sidebar.
- **Moved:** Compute & jobs → Settings → Jobs & usage. Workflow DAG, Datasets, Notebook, Leakage, Models → tabs of a
  project (the graph stays, as "Steps" on the Overview). Evidence library, Research lab, Benchmark, Policy model → one
  Library entry. Intern → "Let DCLab run it" inside the chat. Domain packs, Integrations, Admin → Settings.
- **Added:** a Features tab (each feature, how it is made, from which columns, known in time or not, kept or dropped
  and why, its importance, and the recipes compared); a Claude-style chat (centred column, solid composer at the
  bottom, questions answered by a click, the problem card in a side panel).
- **Numbers** in v3 are examples shaped like the Telco churn project, not measured results.

## Adding a version

1. Copy the last version's folder to the next number and change the copy.
2. Put the version and date in the page (v3: the "What changed" button, bottom right).
3. Add a row to the table above: what changed, and where the code stands.

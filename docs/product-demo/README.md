# DCLab R&D product demo

**Demo v1 — frozen reference for development.** This folder is the agreed UI and UX of the product, tagged `demo-v1` in git. The working product is built from a copy of these sources; do not edit this folder to change the product. It stays buildable and `make rd-check` keeps checking that `index.html` matches `src/`.

A clickable demo of the final DCLab R&D product, built to decide what to build next. It is UI only: nothing runs a model, and no backend is needed.

Open `docs/product-demo/index.html` in a browser. It is one self-contained file. A published copy, private to the owner, is at https://claude.ai/artifact/Vky86ZeEg2z5gqeapBXwtr. There, the Must, Later and Cut decisions save to a shared store that Claude can read.

## What is inside

19 pages, grouped in the sidebar like the product:

| Group | Pages |
|---|---|
| Workspace | Home · New project · Intern · Compute & jobs |
| Project (Term-deposit calls) | Workflow · Contract & data · Notebook · Leakage & features · Models · Reliability · Decision brief |
| Knowledge | Evidence library · Research lab · Benchmark · Policy model |
| Platform | Domain packs · Integrations · Admin · Blueprint & roadmap |

Every page opens on an **Overview** tab: three or four headline numbers and the one card that matters most. The details sit in the other tabs at the top of the page. A tab has its own link (`#models/tuning`), and ⌘K finds tabs as well as pages.

Across every page:

- **Roles.** The *View as* menu in the top bar switches between developer, business, researcher and admin views.
- **Guided tours.** Four tours walk through the product (sidebar → Guided tours). A tour opens the right tab by itself.
- **Blueprint layer.** The *Blueprint layer* switch in the sidebar outlines every feature on the page as built, partial, to build or research, with where it lives in the code.
- **Decisions.** Mark each feature *Must*, *Later* or *Cut*, in the side card or on the Blueprint page. "Copy decisions as Markdown" exports them.
- **Search.** Ctrl/⌘ K searches pages, tabs, actions and the evidence records.

## What is real and what is sample data

Real, taken from this repository when the demo was built:

- the 133 evidence records (rules, workflow blocks, experiments, leakage precedents, pitfalls, findings, dataset cards), which the Evidence page searches;
- the copilot's 10 findings on `dclab_rnd/copilot/examples/leaky_bank_marketing.ipynb`;
- every number shown next to a record chip such as `EXP-010`; the main project, *Term-deposit calls*, follows the real bank-marketing chain EXP-006 to EXP-010;
- the registry champions (`python -m dclab_rnd status`), the blind auditor replay, the SFT v3 manifest and one real training example;
- the train-only EDA rates, the leaky notebook's own printed score (accuracy 0.9689), and the honest score after the copilot's fixes (ROC-AUC 0.7577, AP 0.3454 on 9,043 test rows). All three were measured by running the code while building this demo.

Everything else is sample data: people, jobs, spend, slices, the planned benchmark leaderboard, and the vision and maps packs. Pages say so where it matters.

## Rebuild

```bash
make product-demo              # rebuild index.html from src/
make product-demo REFRESH=1    # also re-extract records, copilot review and SFT example
```

Sources live in `src/`:

- `shell.html`: the layout (sidebar, top bar, drawer, modal, palette).
- `styles.css`: the design system.
- `core.js`: router, page tabs, roles, blueprint layer, decisions, tours, search and charts.
- `features.js`: the feature registry with status and code paths.
- `data.js`: sample data and tours.
- `data/*.json`: data extracted from the repository.
- `views/*.html`: one file per page. A page lists its tabs as `<div class="pane" data-pane="id" data-label="Label">` blocks directly inside its `<section>`; the first one is the Overview.

`build.py --artifact OUT.html` also writes the skeleton-free version used for publishing.

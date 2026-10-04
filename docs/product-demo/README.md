# DCLab R&D product demo

A clickable demo of the final DCLab R&D product, built to decide what to build next. It is UI only: nothing runs a model, and no backend is needed.

Open `docs/product-demo/index.html` in a browser. It is one self-contained file. A published copy, private to the owner, is at https://claude.ai/artifact/Vky86ZeEg2z5gqeapBXwtr. There, the Must, Later and Cut decisions save to a shared store that Claude can read.

## What is inside

19 pages, grouped like the product:

| Group | Pages |
|---|---|
| Build | Home · New project · Workflow graph · Contract & data · Notebook · Leakage & features · Models · Reliability & holdout · Decision brief · Intern · Compute & jobs |
| Know | Evidence library · Research lab · Judgment benchmark |
| Learn | Policy model |
| Platform | Domain packs · Integrations · Admin |
| Plan | Blueprint (gap map, roadmap, decisions) |

Across every page:

- **Roles.** The top bar switches between developer, business, researcher and admin views.
- **Guided tours.** Four tours walk through the product: developer, business, researcher, admin.
- **Blueprint layer.** The *Blueprint* switch outlines every feature on the page as built, partial, to build or research, with where it lives in the code.
- **Decisions.** Mark each feature *Must*, *Later* or *Cut*, in the side card or on the Blueprint page. "Copy decisions as Markdown" exports them.
- **Search.** Ctrl/⌘ K searches pages, actions and the evidence records.

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

- `shell.html`: the layout.
- `styles.css`: the design system.
- `core.js`: router, roles, blueprint layer, decisions, tours, search and charts.
- `features.js`: the feature registry with status and code paths.
- `data.js`: sample data and tours.
- `data/*.json`: data extracted from the repository.
- `views/*.html`: one file per page.

`build.py --artifact OUT.html` also writes the skeleton-free version used for publishing.

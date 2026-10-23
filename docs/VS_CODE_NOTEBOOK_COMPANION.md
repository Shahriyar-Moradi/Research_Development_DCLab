# DCLab Notebook Companion: local VS Code preview

This is the first usable notebook-facing slice of DCLab R&D. It adds a short
status item **beneath each Python code cell** in VS Code notebooks. Clicking a
suggestion opens a read-only proof panel with the exact DCLab rule, its source
references, a recommended next check and limits on what the evidence proves.

It does **not** run a cell, train a model, edit a notebook, send notebook text
to OpenAI, or claim a prediction feature is leaky from its name alone. It is a
local static companion, not yet the full conversational agent or a validated
production assistant.

## Try it in VS Code now

1. Open `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/vscode_notebook_companion`
   as a folder in VS Code.
2. Press **F5** to launch an Extension Development Host.
3. In that new window, open the DCLab R&D folder, then open
   `part1_hyper_ack_classification.ipynb` or
   `Cluad_R&D_aritfacts_sptember22/demo_churn_model.ipynb`.
4. Wait briefly for the local review. Look beneath the code cells for DCLab
   comments. Click one to **Show proof**. The Command Palette also has
   **DCLab: Review Active Notebook** for an explicit refresh.

If the extension cannot find the R&D repository, set
`dclabNotebook.repoPath` to its absolute path. It uses the repository's
`.venv/bin/python` by default; override `dclabNotebook.pythonPath` if needed.
The workspace must be trusted to run this local reviewer. VS Code's Jupyter
extension must be installed so `.ipynb` files open as notebooks.

To inspect the same review as JSON without VS Code:

```bash
cd '/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D'
.venv/bin/python -m dclab_rnd.notebook_assist \
  --notebook part1_hyper_ack_classification.ipynb \
  --output output/evaluation/NEW_HYPERACK_NOTEBOOK_REVIEW.json
.venv/bin/python -m dclab_rnd.notebook_assist \
  --notebook 'Cluad_R&D_aritfacts_sptember22/demo_churn_model.ipynb' \
  --output output/evaluation/NEW_CHURN_NOTEBOOK_REVIEW.json
```

Choose new output names each time; reports are immutable snapshots. The VS
Code extension reviews unsaved in-memory cell text through stdin, so it does
not need to save or rewrite your notebook.

## Evidence levels and current limits

- **Code-order observation:** a fit before a split is visible in top-level
  code. It still needs a clean-kernel run to verify actual notebook behavior.
- **Cross-cell review:** a fitted dataframe feature precedes a later split;
  execution order and lineage are not observed, so the finding asks for a
  check rather than declaring a measured leak.
- **Availability question:** a selected field such as `cancellation_date`
  requires a prediction-time contract. Its name alone is not proof.
- **Recorded experiment analogue:** shown only for an exact same-dataset,
  same-feature claim such as bank-marketing `duration` (`EXP-007-C2`). Other
  suggestions show methodological rules, not an invented StackOverflow-like
  precedent.
- **No rule matched:** a cell is not certified correct. This initial engine
  only covers a few high-value checks.

This output is deliberately separate from SLM training data. Every finding
keeps the cell fingerprint, rule IDs, uncertainty and exact-experiment flag;
expert feedback and source-grouped evaluation must precede training or
product-wide claims.

On the seven **existing development checks** that exposed the old prototype's
misses, the new engine passes 7/7 signals and 4/4 predeclared proof-rule checks;
its availability wording passes the cautious-language check. The machine-readable
record is `output/evaluation/notebook_companion_development_2026-09-26.json`.
Because these same cases guided implementation, these numbers show regression
repair only—not accuracy on new notebooks or real user satisfaction.
The Python and JavaScript checks passed locally, but a VS Code Extension
Development Host exited during this run, so the beneath-cell UI still needs
an editor smoke test before it can be called verified.

## Next product steps

1. Have a data scientist review the findings on fresh, source-held-out
   notebooks; measure severe false alarms, missed hazards, usefulness and
   time-to-correction. The current HyperAck/churn cases are development tests.
2. Add an explicit accept/reject/needs-context feedback action and store
   human labels separately from the immutable R&D evidence.
3. Add a read-only evidence-search/chat tool: filter structured records by
   dataset, stage and issue; then rank relevant claims and cite exact paths.
   Use an LLM only to explain retrieved evidence after its citations pass
   validation. Do not put all experiment text into every prompt.
4. Add controlled code actions only for fixes tested by R&D and approved by
   the user; keep splits, metrics and experiments in deterministic tools.
5. Promote the same engine and evidence contract into the DCLab product only
   after blinded editor-task evaluation and independent confirmation.

The boundary follows [official OpenAI documentation on tool
calling](https://developers.openai.com/api/docs/guides/function-calling) and
[agent safety](https://developers.openai.com/api/docs/guides/agent-builder-safety):
the model may request a narrow, structured action, while application code
checks the action and owns side effects. The chosen VS Code cell status item is
documented in the [official VS Code Notebook API](https://code.visualstudio.com/api/references/vscode-api).

# Architecture

```
browser ── FastAPI server (dclabai/agentic) ── jobs (dclabai/jobs) ── worker (stages, pipelines, improvement runs)
              │                                      │
              ├─ Home: drafts and the Home agent (dclabai/draft)        ├─ the five-stage engine (dclabai/studio, dclabai/kinds)
              ├─ the project chat's team (dclabai/draft/orchestrator)   ├─ notebook cells in a Jupyter kernel (dclabai/studio/kernels)
              ├─ the intern (dclabai/intern)                            └─ storage: local files or PostgreSQL (dclabai/storage)
              └─ the model gateway (dclabai/models): OpenAI-compatible or Anthropic, budgets, limits, usage
dclab_core: the portable library under it all (problem and data cards, the fit check, recipes and modifiers, leakage rules,
             evaluation, reports, lineage), importing only numpy, pandas, scikit-learn and pydantic.
```

- **Agents** run in one harness (`dclabai/agents`): a model loop with registered tools, each write passing its scope's
  guard (the project graph's validator, the draft guard, the chat's guard); every step is traced.
- **Rules first:** every agent has deterministic rules; a model serves where it is configured and trusted.
- **Records:** every stage writes a record with its evidence; every decision is kept in the workspace's ledger and audit
  trail; a report's numbers trace back to their run.
- **Problems as recipes:** a kind of problem plus modifiers, resolved into choices that each name the part and the reason
  (see adding-capability.md).

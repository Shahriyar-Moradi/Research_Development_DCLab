# dclabai

**An agentic data-science studio that builds models you can trust.** Describe a problem in your own words, bring a table (or
let it make a synthetic one), and dclabai's agents and engine build a leakage-safe model, evaluate it honestly on held-back
cases read once, and explain it, with every choice recorded and every number traced to its run.

Run it with **your own model** (a local open model through Ollama, any OpenAI-compatible endpoint, or Claude) or with **no
model at all**: every agent has deterministic rules to fall back on.

- **Home:** a conversation that turns a problem into a problem card, checks it against your data, and builds the project.
- **Five leakage-safe stages:** data, leakage audit, features, models, final (the holdout read once), each with its
  evidence and the rule it applied.
- **A team in the project's chat:** a problem analyst (change the problem, see what changes), a solution planner (build a new
  version), a records reader, an evaluator (precision, recall, F1, confusion matrix), an experimenter (improve the model on
  the training folds only), a notebook coder (cells run in a Jupyter kernel on the training rows).
- **Many kinds of problems:** classification, regression, rare events, anomalies, forecasting, ranking, uplift, matching,
  time to event, segmentation and more, each a recipe of a kind plus modifiers (rare positives, groups, time order, noisy
  labels, review capacity).
- **Code export:** the whole pipeline as a script and a notebook that reproduce the holdout number.
- **Local first:** projects, chats and records stay on your machine; PostgreSQL and accounts are optional.

## Install

```bash
pip install dclabai            # or: uv tool install dclabai
dclab init                     # choose your model (or none)
dclab start                    # opens the studio in your browser
```

With npm (it installs and runs the Python package for you):

```bash
npx dclabai
```

With Docker:

```bash
docker run -p 127.0.0.1:8765:8765 -v dclabai:/workspace ghcr.io/shahriyar-moradi/dclabai
```

Optional extras: `pip install "dclabai[all]"` (Claude, PostgreSQL, notebook cells, MCP, PDF reports).
On macOS, LightGBM needs OpenMP: `brew install libomp` (`dclab doctor` checks it).

## Your model

| Model | How |
|---|---|
| Local, open (Ollama) | `ollama pull qwen2.5:7b`, then `dclab init` (it finds Ollama on localhost) |
| Any OpenAI-compatible endpoint | `dclab init` → base URL, key, model name |
| Claude | `pip install "dclabai[anthropic]"`, `dclab init` → your Anthropic key |
| None | skip `dclab init`: the agents follow their rules |

`dclab doctor` checks your setup, including whether your model calls tools natively (the agents need that; a model that
cannot is used for prose only). See [docs/models.md](docs/models.md).

## Documentation

- [Quickstart](docs/quickstart.md)
- [Concepts](docs/concepts.md): the prediction moment, the recipe, the holdout read once
- [Models](docs/models.md)
- [Architecture](docs/architecture.md)
- [Adding a kind of problem or a modifier](docs/adding-capability.md)
- [Privacy](docs/privacy.md)

## Contributing

Issues and pull requests are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

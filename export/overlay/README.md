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

## Set up (once)

For now, dclabai installs from this GitHub repository only. Copy each command on its own, paste it into a terminal and
press Enter.

**Step 1. Check Python.** You need Python 3.10 or later.

```bash
python3 --version
```

**Step 2. On a Mac only: install OpenMP.** The model library LightGBM needs it. (This needs [Homebrew](https://brew.sh).)

```bash
brew install libomp
```

**Step 3. Make a separate folder for dclabai's Python packages** (a "virtual environment"). Then dclabai never changes
the packages your other projects use.

```bash
python3 -m venv ~/.dclabai-env
```

**Step 4. Install dclabai from GitHub.** This takes a few minutes. `[all]` adds Claude, PostgreSQL, notebook cells, MCP
and PDF reports.

```bash
~/.dclabai-env/bin/pip install "dclabai[all] @ git+https://github.com/Shahriyar-Moradi/dclabai.git"
```

**Step 5. Choose the AI model the agents use.**

```bash
~/.dclabai-env/bin/dclab init
```

It asks you to pick a number. Type the number, then press Enter (Enter alone picks the default shown in brackets):

| Type | Model | What you need | Cost |
|---|---|---|---|
| `1` | Ollama, on your own computer | [Ollama](https://ollama.com) running, and a model that can call tools, for example `ollama pull qwen2.5:7b` | free |
| `2` | Any OpenAI-compatible service | its URL, your key and the model's name | the service's prices |
| `3` | Claude | your Anthropic key and a model name, for example `claude-sonnet-5-5` | Anthropic's prices |
| `4` | No model | nothing: every agent follows its built-in rules | free |

Your key is hidden while you type it. It is saved only on your computer, in `~/.config/dclabai/.env`, readable by you
alone.

**Step 6. Check that everything works.**

```bash
~/.dclabai-env/bin/dclab doctor
```

Every line should start with `ok`. The last lines show whether your model can call tools: the agents use the model only
if it can.

## Run it (every time)

**Start the studio:**

```bash
~/.dclabai-env/bin/dclab start
```

It opens in your browser at http://127.0.0.1:8765. Keep the terminal open while you work.

**Stop it:** go back to the terminal and press `Ctrl` + `C`.

**Get the latest version** (from time to time):

```bash
~/.dclabai-env/bin/pip install --force-reinstall --no-deps "dclabai[all] @ git+https://github.com/Shahriyar-Moradi/dclabai.git"
```

## Your first project, step by step

1. **Describe your problem.** On **Home**, write one sentence in the box, for example *"Predict which customers will
   cancel their subscription next month"*. Under the box, choose who will run the stages later: **Intern drives** (an
   agent runs them for you) or **I drive · notebook** (you run them yourself). Click **Draft the solution**.
2. **Bring your data, or ask for invented data.**
   - Your own table: click **Connect data** and upload a CSV, Excel or Parquet file.
   - No data you can share? Ask in the chat: *"make synthetic data for this"*. dclabai invents a table with the same
     kind of columns. It is always labelled synthetic: use it to try the steps, not to judge real performance.
3. **Answer a few questions** in the chat: what you want to predict, and *when* you would make the prediction. The
   "when" matters most: dclabai uses only what is known at that moment, so the model cannot cheat with future
   information (this is called *leakage*).
4. **Check what it understood.** The panel *What DCLab understood* shows each answer and where it came from. Click
   **Correct** on anything that is wrong.
5. **Build it.** When the answers look right, click **Build this chat as a project** (or say *"build it"*).
6. **Watch the five stages** on the project's **Workflow** page: understand the data, audit leakage, compare feature
   recipes, compare models, and a final test on held-back rows that were never used before (read once).
7. **Read the results.**
   - **Evaluation:** the final score with its range (for example *0.85, from 0.81 to 0.88*). A range that sits close to
     chance means the model learned little.
   - **Decision brief:** a plain summary of what was built, the risks, and what to do next.
8. **Keep talking to the project.** In the project's chat, ask *"show the models"*, *"how good is it?"*, *"try to improve
   it"* or *"make a plot of …"*. To change the problem itself, describe the change; dclabai asks before it builds a new
   version.
9. **Take the work with you.** On **Decision brief**, click **Export report (.md)**. On **Notebook**, click
   **Download .ipynb**. For developers, the whole pipeline as code that reproduces the final score is at
   `http://127.0.0.1:8765/api/projects/<project id>/export/code`.

## If something goes wrong

| What you see | Why | What to do |
|---|---|---|
| `zsh: command not found: dclab` | The command lives in the dclabai environment | Type the full path: `~/.dclabai-env/bin/dclab` |
| `Invalid requirement: '#'` or `unknown sort specifier` | A comment (`# ...`) was pasted after the command | Paste each command alone, without anything after it |
| `address already in use` | The studio is already running | Open http://127.0.0.1:8765, or stop the other one with `Ctrl` + `C` |
| The studio does not open, or waits forever (macOS) | macOS asks for permission to read a folder | Look for a popup and click **Allow** |
| `dclab doctor` shows a model you did not choose | An `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set in your shell; without a choice in `dclab init`, it is used | Run `dclab init` and pick the model you want (or `4`) |
| A LightGBM error on a Mac | OpenMP is missing | `brew install libomp` |
| Answers are slow or poor with Ollama | Small local models are weak | Use a bigger model (`qwen2.5:7b` or larger), or Claude |

More about models, budgets and which agents use them: [docs/models.md](docs/models.md).

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

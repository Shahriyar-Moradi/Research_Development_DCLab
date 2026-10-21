# Quickstart

## 1. Install

For now, dclabai installs from GitHub only. Use Python 3.10 or later, in an environment of its own. On macOS, run
`brew install libomp` first (LightGBM needs it).

```bash
python3 -m venv ~/.dclabai-env
```

```bash
~/.dclabai-env/bin/pip install "dclabai[all] @ git+https://github.com/Shahriyar-Moradi/dclabai.git"
```

The commands below assume `~/.dclabai-env/bin` is on your `PATH` (or type the full path).

## 2. Choose a model (or none)

```bash
dclab init
```

It finds Ollama on your machine, or asks for an OpenAI-compatible URL and key, or a Claude key, and writes
`~/.config/dclabai/.env`. Check the result with `dclab doctor`. Skip it to run every agent on its rules, but note that an
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` already set in your shell is used when no model is chosen here: `dclab doctor`
shows which model and endpoint answer.

## 3. Start the studio

```bash
dclab start
```

It opens http://127.0.0.1:8765. On **Home**, describe a problem ("which subscribers will cancel before their renewal?"),
bring a table (CSV, Excel, Parquet) or ask for synthetic data, answer the few questions, and say **build it**. The five
stages run with the leakage audit first and the holdout read once; open the project to follow them, see the evaluation and
the report, or keep talking in its chat: ask for the models, the evaluation, an improvement run, a plot, or change the
problem and build a new version.

## 4. Teams (optional)

For several people, run with PostgreSQL and accounts (the `[all]` install includes the PostgreSQL driver):

```bash
export DCLAB_DATABASE_URL=postgresql+psycopg://user@localhost/dclabai DCLAB_AUTH=password
dclab migrate && dclab users add --username you --role owner && dclab start
```

# Quickstart

## 1. Install

```bash
pip install dclabai          # Python 3.10+; or: uv tool install dclabai, or: npx dclabai, or Docker (see the README)
```

## 2. Choose a model (or none)

```bash
dclab init
```

It finds Ollama on your machine, or asks for an OpenAI-compatible URL and key, or a Claude key, and writes
`~/.config/dclabai/.env`. Skip it to run every agent on its rules. Check the result with `dclab doctor`.

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

For several people, run with PostgreSQL and accounts:

```bash
pip install "dclabai[postgres]"
export DCLAB_DATABASE_URL=postgresql+psycopg://user@localhost/dclabai DCLAB_AUTH=password
dclab migrate && dclab users add --username you --role owner && dclab start
```

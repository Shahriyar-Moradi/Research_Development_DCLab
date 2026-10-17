# DCLab's training records

Everything DCLab captured for training its own models, kept on this machine (the owner's choice, 2026-10-17), one dated
snapshot per `make archive`, never deleted by a rebuild. Each snapshot is read-only once written.

- `snapshots/<time>/workspace/`: the development workspace's database dump and folder (decision ledgers, experience logs,
  drafts, projects, live tests, twin runs, testers' records), with their own manifest.
- `snapshots/<time>/lake_bronze/`: the training lake's raw inputs: verified self-play, every other model's answers on every
  decision (`answers/<answerer>/<point>/<split>.jsonl`), the inputs' hashes.
- `snapshots/<time>/openml_profiles/`: public tables' profiles (bands only).
- `snapshots/<time>/decision_sets/`: the generated decision sets (their sealed partitions included: never train or tune on
  them) and the workflow decisions from journeys.
- `snapshots/<time>/MANIFEST.json`: every file with its size and SHA-256, counted by kind.
- `INDEX.json`: every snapshot, its counts and its manifest's hash.

What may train a model is said in each record (`use`, `origin`, `split`); Claude's and other hosted models' answers are
evaluation only until their terms say otherwise.

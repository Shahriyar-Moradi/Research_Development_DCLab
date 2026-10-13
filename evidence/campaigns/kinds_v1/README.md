# kinds_v1: each problem kind's decisions on its replica

Package 16.14 of `docs/guides/PRODUCT_BUILD_PLAYBOOK.md`. One result per problem kind (`results/KND-<n>_<kind>.json`),
made by `python -m dclab_rnd.kinds.campaign`: the kind's acceptance (`dclab_rnd.kinds.acceptance.run`) on seeds 0, 1
and 2 of its replica, with each decision a careful data scientist reaches on that kind of problem, the seeds it held and
failed on, and the share of seeds on which every decision held with a 95% Wilson interval over seeds.

- The evidence index reads the results as records of type `kind` (`KND-*`), tied to the kind's research track
  (`dclab_rnd/kinds/*.py`, `evidence_track`), so a method question about a kind is answered from its own decisions; the
  intern and `/mcp` read a kind's checks with the `kind_checks` tool.
- Results are never overwritten: a rerun writes new ids, and the newest result of a kind is the one agents read first.
- Decisions a kind names as open (`open_decisions`: waiting on the owner, such as pretrained weights) are listed and
  never counted as failures.

Limits: replicas are invented data with planted traps, three seeds each. The campaign says what a kind decides, not how
well it does on real data, and it is not a benchmark: no kind has a sealed set, and nothing is tuned on these runs.

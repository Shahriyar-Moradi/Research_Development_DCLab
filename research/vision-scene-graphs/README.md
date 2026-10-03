# Computer vision and scene graphs

**Status: proposed research; no scene-graph implementation or result is currently established by this repository.** The goal is for a vision-language assistant to describe objects and their relationships with evidence, not just list detected labels.

## Representation

- Nodes represent tracked objects or regions: person, bottle, cup, tool, road user.
- Edges represent candidate relations: holding, above, touching, approaching, inside.
- Node/edge attributes store boxes or masks, appearance, time, confidence, and evidence frames.
- A scene graph is a data representation; a detector, relation model, GNN, or transformer may predict it.

## First prototype

Choose a narrow short-video task such as person–object interaction. Detect/track objects, predict a small fixed vocabulary of relations, and return time-stamped triples with supporting frames and uncertainty. Compare against object labels alone and a video-language-model baseline. Include hard negatives where objects are nearby but not interacting. Measure object/relation precision and recall, event accuracy, temporal localization, calibration, latency, and unsupported claims.

See the [temporal GNN track](../temporal-gnn/README.md) for message passing and sequence reasoning. Do not treat a plausible graph as ground truth; relation annotations and evaluation data are needed.

# Temporal GNNs and graph transformers

**Status: proposed research.** This track tests whether explicitly passing information between object nodes improves interaction or event predictions across frames.

## What would be learned

A graph model can update each object's representation using messages from connected objects and edge features. A relation head can classify edges; a node head can classify object states; a graph/temporal head can classify events or predict a sequence. The GNN does not discover the intended meaning of an edge without a target, training signal, or defined unsupervised objective.

## Experimental comparison

Use the same detector/tracker and video splits for: (A) geometry/rule baseline, (B) pairwise relation classifier, (C) spatial GNN, and (D) temporal GNN or graph transformer. Compare with video-only and video-plus-graph VLM answers. Hold out complete videos, people, or environments. For online prediction, prohibit future frames.

Measure relation macro-F1 and rare-relation recall, event accuracy/F1, temporal localization, graph consistency, calibration, latency, and performance on unseen object-relation combinations. Inspect graph errors and occlusion failures. A gain on common relations alone may reflect dataset frequency rather than better reasoning.

The scene-graph setup lives in [`../vision-scene-graphs/README.md`](../vision-scene-graphs/README.md). This track remains an idea until data, code, and measured results are added.

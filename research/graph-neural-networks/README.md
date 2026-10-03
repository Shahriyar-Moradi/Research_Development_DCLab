# Graph neural networks

**Status:** planned

## Research question

When do graph neural networks (GCN, GraphSAGE, GAT) beat strong tabular models that use hand-made graph features, and how do we avoid graph-specific leakage?

## Why it matters for DCLab

Many DCLab use cases are relational: customers and merchants, riders and orders, users and items. A graph view could add signal that tables miss.

## Leakage traps specific to this field

- **Transductive leakage:** test nodes' features and edges are visible during training message passing.
- **Edge leakage:** edges created after the prediction moment (for example, a refund link) reveal the label.
- **Neighbor label leakage:** aggregating neighbors' labels includes the target node's own label through cycles.
- **Random node splits on temporal graphs:** use time-based splits for evolving graphs.

## First experiments

| ID | Question |
|---|---|
| GNN-001 | Baseline: tabular gradient boosting with degree, PageRank and neighbor-aggregate features vs GraphSAGE on a citation graph (Cora, ogbn-arxiv) |
| GNN-002 | Leakage audit: measure inflation from transductive vs inductive training, and from edges created after the prediction moment |
| GNN-003 | Fraud as a graph: Elliptic Bitcoin transactions, time-ordered split, GNN vs boosting with graph features |

## Candidate datasets

Cora/CiteSeer (Planetoid), ogbn-arxiv (Open Graph Benchmark), Elliptic Bitcoin dataset. Libraries: PyTorch Geometric or DGL.

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=graph-neural-networks TITLE="Graph neural networks" PREFIX=GNN
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.

# DCLab research tracks

This folder makes the separate R&D ideas easier to explore without moving or replacing the experiments, notebooks, and reports already in the repository. Each track has a short charter: the problem, research questions, a feasible first test, how to evaluate it, and links to existing material.

## Read the status accurately

- **Existing evidence** means code, experiment artifacts, a report, or an evaluation already exists. Read its limits; this does not imply the idea is validated for every use.
- **Implemented, limited scope** means there is a working prototype with stated boundaries.
- **Proposed research** means the idea has been discussed and needs a prototype and evaluation. A charter is not an implementation or a positive result.

The CV, scene-graph, GNN, graph-transformer, autonomous-driving, and cross-industry tracks are **proposed research**. This repository's strongest existing experiment work is tabular classification and the agentic research studio; it does not currently establish autonomous-driving or scene-graph performance.

## Tracks

| Track | Main question | Starting status |
|---|---|---|
| [Tabular data science and ML](tabular-ml/README.md) | Which EDA, leakage controls, features, model families, and optimization steps hold up under sound evaluation? | Existing experiment suites and field guide |
| [Agentic R&D studio](agentic-rd/README.md) | Can bounded research agents plan tests and retain trustworthy, replayable evidence? | Implemented prototype; requires continuing system evaluation |
| [Notebook assistant](notebook-assistant/README.md) | Can useful, evidence-linked help appear beside notebook cells without taking control away from the engineer? | Proposed extension; notebook integration is not in this main-branch snapshot |
| [Evidence, retrieval, and SLM learning](evidence-and-slm/README.md) | How should verified experiments, rules, and counterexamples ground answers and later training? | SFT preparation exists; retrieval and quality gates need evaluation |
| [Focused workflow model](workflow-model/README.md) | Can a small model reliably follow a narrow, versioned workflow and choose the next valid step? | Research proposal plus early training data |
| [Workflow-sized generation](workflow-actions/README.md) | Can the model generate a complete useful action or code block with fewer fragile intermediate steps? | Hypothesis; needs controlled token/action-unit experiments |
| [Computer vision and scene graphs](vision-scene-graphs/README.md) | Can visual objects and their relationships be represented and grounded clearly? | Proposed research |
| [Temporal graphs and GNNs](temporal-gnn/README.md) | Does graph reasoning improve recognition of interactions and event sequences over simpler baselines? | Proposed research |
| [Autonomous driving and road maps](driving-maps/README.md) | Can actor, lane, and map graphs support safer, evidence-grounded predictions in a bounded driving scenario? | Proposed, safety-critical research |
| [Workflow transfer to other industries](cross-industry-workflows/README.md) | Which parts of DCLab's workflow engine transfer, and which must be domain-specific? | Proposed research |
| [Evaluation and trust](evaluation-and-trust/README.md) | What measurable evidence shows that the assistant is correct, useful, and safe within a defined scope? | Evaluation framework and pilot artifacts exist |

## Recommended order

1. Strengthen the current tabular workflow and DCLab notebook assistant, where local data and prototypes already exist.
2. Define reliable evidence records and evaluate retrieval, recommendations, and user-visible claims.
3. Evaluate workflow-following models on held-out tasks before investing in fine-tuning.
4. Prototype scene graphs on short, labeled videos; compare against video-only and pairwise baselines before adding a GNN or graph transformer.
5. Study driving and other industries only after the relevant domain data, scenario coverage, and safety evaluation are available.

## Research record rules

For a new experiment, create a dated, immutable run record under the appropriate track's `experiments/` area only when an actual experiment is ready. Keep proposals separate from measured outcomes. Record the question, data source and license, split strategy, method, code/data fingerprints, metrics, failures, uncertainty, counterevidence, and next question. Never label a result production-ready on benchmark performance alone.

The detailed product context is in [`../docs/DCLAB_MASTER_CONTEXT.md`](../docs/DCLAB_MASTER_CONTEXT.md); run commands are in the [main README](../README.md#dclab-rd-paths).

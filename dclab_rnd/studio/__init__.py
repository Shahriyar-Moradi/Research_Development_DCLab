"""The DCLab notebook: build a model from your own data, stage by stage, with evidence.

A *project* is one dataset plus a prediction contract. The engine runs the five
DCLab stages on it (understand the data, audit leakage, climb the feature ladder,
screen algorithms, tune and confirm once on the locked holdout). Deterministic
code owns every split, metric and selection rule; the agent explains each result,
cites the R&D evidence behind it and proposes the next step. The person approves.

Modules:

- ``store``    projects on disk (one folder per project)
- ``data``     load and profile a table; built-in sample datasets
- ``contract`` the prediction contract and the heuristics that propose one
- ``engine``   the stage executors, built on the expansion campaign's primitives
- ``agent``    evidence-cited notes for every stage, and answers to questions
- ``export``   a runnable notebook and a report for the finished project
"""

from .store import ProjectStore  # noqa: F401

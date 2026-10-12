"""Replicas: seeded, invented data shaped like a real kind of problem, with that kind's traps planted.

A replica is how DCLab proves it can solve a kind of problem without anyone's private data (Part 4 and Part 5 of
docs/guides/PRODUCT_BUILD_PLAYBOOK.md): each one comes with an acceptance test of the decisions a careful data
scientist reaches on it. Everything in a replica is synthetic and labelled so (``LABELS``); its scores
say whether DCLab decides well, never how a model would do on real data.

They live in dclab_core, not in the R&D package, so DCLab's main product can run the same acceptance cases against the
library it adopts. Each generator is pure numpy and pandas and deterministic per seed.
"""

LABELS = {
    "synthetic": True,
    "warning": "Synthetic data generated for an acceptance test. Use it to check decisions, not to report real performance.",
}

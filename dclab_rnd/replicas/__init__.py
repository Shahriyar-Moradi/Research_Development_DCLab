"""Replicas: seeded, invented data shaped like a real kind of problem, with that kind's traps planted.

A replica is how DCLab proves it can solve a kind of problem without anyone's private data (Part 4 and Part 5 of
docs/guides/PRODUCT_BUILD_PLAYBOOK.md): each one comes with an acceptance test of the decisions a careful data
scientist reaches on it. Everything in a replica is synthetic and labelled so (``draft.synthetic.LABELS``); its scores
say whether DCLab decides well, never how a model would do on real data.
"""

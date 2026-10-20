"""dclab_core: how DCLab understands a problem, analyses its data for decisions and plans a solution.

A pure library, shared by the DCLab R&D studio and DCLab's main product (docs/guides/PRODUCT_BUILD_PLAYBOOK.md,
Part 6). It depends only on numpy, pandas, scikit-learn and pydantic; it never trains a model, calls a model provider
or touches a database: the host does those, and adopts what this library proposes through its own governed decisions.

Every output is a frozen, versioned contract (``dclab_core.contracts``) with a published JSON Schema
(``dclab_core/schemas/``). Model help, when the host offers it, goes through ``dclab_core.advisor``: aggregates in,
a validated answer or nothing out, and the deterministic path always works.
"""

__version__ = "0.100.0"

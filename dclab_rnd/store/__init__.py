"""DCLab's store (the storing phase, ``docs/guides/STORING_PHASE_PLAN.md``): everything kept for training the eight model
types and for learning how to build them and serve businesses, in three tiers (``raw``, ``records``, ``sets``).

- ``layout``: every path, in one place.
- ``store``: append-only keeping, versions, hashes, read-back.
- ``catalog``: ``CATALOG.json``, ``README.md`` and the schemas.

The contracts are ``dclab_core.store`` and ``dclab_core.store_rules``. ``python -m dclab_rnd.store status|catalog``.
"""

from .layout import Layout
from .store import Store

__all__ = ["Layout", "Store"]

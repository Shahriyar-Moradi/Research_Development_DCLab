"""Optional parts of the studio (the open edition, 2026-10-21; the plan: export/manifest.py).

The same code runs in two editions: this repository (everything) and the open-source tree built by
scripts/export_open.py, which leaves out the closed parts (the store and its training machinery, the measurement suites,
the twin lane, the orchestration spike, the research pages). Code that reaches a closed part asks for it here and goes on
without it when the edition does not have it:

    hooks = optional("store.hooks")
    if hooks is not None:
        hooks.keep_before(...)

``optional(name)`` takes a module relative to this package ("store.hooks") or a dclab_core module ("dclab_core.datasets").
A module that is missing returns None; a module that exists but fails to import raises, so a real bug is never hidden.
The export's import check accepts a closed module only when it is named here, inside optional().
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType
from typing import Optional

PACKAGE = __name__.rsplit(".", 1)[0]  # dclab_rnd here, dclabai in the open tree


@functools.lru_cache(maxsize=None)
def optional(name: str) -> Optional[ModuleType]:
    """The module, or None when this edition does not have it."""
    full = name if name.startswith("dclab_core.") else f"{PACKAGE}.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (full == missing or full.startswith(missing + ".")):
            return None
        raise


def available(name: str) -> bool:
    return optional(name) is not None


def edition() -> str:
    """"full" in this repository, "open" in the open-source tree (the store is the mark of the full edition)."""
    return "full" if available("store") else "open"


def load_plugins(app) -> list[str]:
    """Plugins: installed packages that declare an entry point in the group ``<package>.plugins`` (``dclabai.plugins``
    in the open edition), each a function ``register(app)`` that adds routes (with ``agentic.api_models.Router``, so the role checks apply), pages'
    data or job handlers. A plugin that
    fails is logged and left out: it never stops the studio from starting. Returns the names of the plugins loaded."""
    import logging
    from importlib.metadata import entry_points

    loaded = []
    for point in entry_points(group=f"{PACKAGE}.plugins"):
        try:
            point.load()(app)
            loaded.append(point.name)
        except Exception as exc:  # noqa: BLE001 - one broken plugin must not take the studio down
            logging.getLogger(__name__).warning("plugin %s not loaded: %s: %s", point.name, type(exc).__name__, str(exc)[:200])
    return loaded


__all__ = ["optional", "available", "edition", "load_plugins"]

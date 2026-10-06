"""One agent runtime for DCLab (phase A2, docs/guides/AGENTIC_FOUNDATION_PLAN.md).

- ``Tool`` and ``Registry`` (registry.py): every tool any agent can call, registered once with its schema and effect.
- ``Policy`` and ``run()`` (runtime.py): the loop every agent runs on.
- ``Tracer``, ``open_traces`` and ``replay`` (traces.py): a row per step of every run, and its replay.
- ``default_registry()``: the process-wide registry. The intern and MCP clients read its "project" tools, the Home
  agent its "draft" tools; adding a tool is one ``register`` call and every reader of that scope lists it.
"""

from __future__ import annotations

import functools

from .registry import EFFECTS, SCOPES, Registry, Tool
from .runtime import Policy, RunResult, Step, run
from .traces import FileTraces, PgTraces, Tracer, open_traces, replay


# Where each group of tools is registered. The campaign group imports no table library, so the campaign's own
# environment (.venv-agent, no pandas) builds a registry of that group alone.
GROUPS = {"project": "dclab_rnd.intern.tools", "draft": "dclab_rnd.draft.chat", "session": "dclab_rnd.intern.loop",
          "campaign": "dclab_rnd.agentic.campaign_tools"}


def build_registry(groups: tuple[str, ...] = tuple(GROUPS)) -> Registry:
    """A registry with the given groups of DCLab tools (all of them by default)."""
    import importlib

    registry = Registry()
    for group in groups:
        importlib.import_module(GROUPS[group]).register(registry)
    return registry


@functools.cache
def default_registry() -> Registry:
    return build_registry()


@functools.cache
def registry_of(*groups: str) -> Registry:
    """A process-wide registry of some groups only (the campaign worker's environment has no pandas)."""
    return build_registry(tuple(groups))


__all__ = ["EFFECTS", "SCOPES", "FileTraces", "PgTraces", "Policy", "Registry", "RunResult", "Step", "Tool", "Tracer",
           "build_registry", "default_registry", "open_traces", "registry_of", "replay", "run"]

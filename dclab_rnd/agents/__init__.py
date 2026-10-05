"""One agent runtime for DCLab (phase A2, docs/guides/AGENTIC_FOUNDATION_PLAN.md).

- ``Tool`` and ``Registry`` (registry.py): every tool any agent can call, registered once with its schema and effect.
- ``Policy`` and ``run()`` (runtime.py): the loop every agent runs on.
- ``default_registry()``: the process-wide registry. The intern and MCP clients read its "project" tools, the Home
  agent its "draft" tools; adding a tool is one ``register`` call and every reader of that scope lists it.
"""

from __future__ import annotations

import functools

from .registry import EFFECTS, SCOPES, Registry, Tool
from .runtime import Policy, RunResult, Step, run


def build_registry() -> Registry:
    """A registry with every DCLab tool: the intern's evidence and project tools, and the Home agent's draft tools."""
    from dclab_rnd.draft import chat as home_tools
    from dclab_rnd.intern import tools as project_tools

    registry = Registry()
    project_tools.register(registry)
    home_tools.register(registry)
    return registry


@functools.cache
def default_registry() -> Registry:
    return build_registry()


__all__ = ["EFFECTS", "SCOPES", "Policy", "Registry", "RunResult", "Step", "Tool", "build_registry", "default_registry", "run"]

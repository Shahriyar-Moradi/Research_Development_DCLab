"""API routes behind the product's workspace and platform pages.

Each module here serves one group of pages and exposes ``register(app, ctx)``:

- ``ops``       Compute & jobs, and Home's Activity tab (what ran, what runs now)
- ``lab``       Research lab and Benchmark (registry, campaigns, critic gate, auditor replay)
- ``learn``     Policy model and Domain packs (SFT corpus, project-derived examples, packs)
- ``platform``  Integrations and Admin (MCP tools, connectors, routes, policies, audit log)
- ``evidence``  Evidence library (the live evidence index, and answers checked against it)

``ctx`` carries the server's stores so a module never builds its own. A module that is
missing is skipped, so groups can land one at a time.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

MODULES = ("ops", "lab", "learn", "platform", "evidence")


@dataclass
class Context:
    store: Any                 # agentic.store.Store: the legacy research runs (SQLite)
    projects: Any              # studio.ProjectStore
    drafts: Any                # draft.store.DraftStore
    intern_sessions: Any       # intern.SessionStore
    jobs: Any = field(default_factory=dict)           # project stage jobs by project id (jobs.live.Live: the job table; a dict in tests)
    intern_jobs: Any = field(default_factory=dict)    # intern sessions working now
    draft_jobs: Any = field(default_factory=dict)     # draft pipelines, synthetic data and the Home agent's turns
    models: Any = None                                # dclab_rnd.models.Gateway: every model request goes through it
    job_store: Any = None                             # jobs.Jobs: what Stop and Retry act on (package 10.3)
    audit: Any = None                                 # audit.AuditLog: the workspace's audit trail (package 10.4)


def register_all(app, ctx: Context) -> list[str]:
    loaded = []
    for name in MODULES:
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{name}":
                continue
            raise
        module.register(app, ctx)
        loaded.append(name)
    return loaded

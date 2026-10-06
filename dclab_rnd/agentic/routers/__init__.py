"""The API's routers, one per area (package 10.1). Each takes its dependencies through ``Depends(services)``."""

from . import core, intern, projects, runs

ROUTERS = (core.router, runs.router, projects.router, intern.router)

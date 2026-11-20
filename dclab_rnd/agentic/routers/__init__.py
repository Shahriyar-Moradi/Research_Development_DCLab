"""The API's routers, one per area (package 10.1). Each takes its dependencies through ``Depends(services)``."""

from . import core, intern, jobs, memory, projects, runs, team_threads, teams

ROUTERS = (core.router, runs.router, projects.router, intern.router, jobs.router, memory.router, teams.router, team_threads.router)

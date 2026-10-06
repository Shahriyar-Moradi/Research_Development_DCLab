"""The product server: builds the app (package 10.1). Loopback only; API keys stay in the server environment.

The routes live in routers per area (``agentic/routers``: core, runs, projects, intern; ``draft/api.py``; the pages in
``agentic/pages``), each taking what it needs through ``Depends(services)``; the settings are read once
(``dclab_rnd/settings.py``). This module only puts them together: the request token and the security headers, the
lifespan that interrupts runs a restart cut and stops running jobs, the MCP endpoint and the static files.
"""

import asyncio
from contextlib import asynccontextmanager
import secrets

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog import ROOT
from .current import Current
from .pool import Pool
from .. import context, limits
from ..accounts import api as accounts_api
from ..accounts.guard import identify
from ..accounts.principal import ANONYMOUS, LOCAL_OWNER, reset_current, set_current
from ..accounts.roles import allows
from .routers import ROUTERS
from .. import mcp_server
from ..intern.tools import Toolbox
from ..settings import Settings
from . import pages
from ..draft import api as draft_api

load_dotenv(ROOT / ".env", override=False)
STATIC = ROOT / "dclab_rnd" / "agentic" / "static"
# No inline scripts or styles anywhere, and nothing is loaded from outside this app: the fonts are served from /static/app/fonts.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; "
       "font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")


def create_app(home=None) -> FastAPI:
    settings = Settings.load(home)  # read once for this app
    if settings.auth != "none" and not settings.database_url:
        raise RuntimeError("DCLAB_AUTH needs PostgreSQL: accounts and sessions live in the database (set DCLAB_DATABASE_URL)")
    pool = Pool(settings)  # one Services per workspace this server serves (package 10.2); the first is its own folder's
    s = pool.default
    # one MCP endpoint for every workspace: its toolbox reaches the projects of the workspace the request's token names (10.2)
    s.mcp = mcp_server.session_manager(Toolbox(Current("projects", s))) if mcp_server.available() else None

    @asynccontextmanager
    async def lifespan(app):
        for run in s.store.list():
            if run["status"] in ("queued", "running", "pausing"):
                s.store.update(run["id"], status="interrupted", phase="Server restarted; resume explicitly")
        await asyncio.to_thread(pool.start_all)  # recovers jobs a dead worker left running, then runs queued ones (10.3), in every workspace (10.2)
        if s.mcp is None:
            yield
        else:
            async with s.mcp.run():  # the MCP transport lives as long as the server
                yield
        for each in pool.all():  # running jobs stop at their next checkpoint, interrupted and retryable
            await asyncio.to_thread(each.worker.stop)
        active = [t for each in pool.all() for t in (*each.tasks.values(), *each.draft_tasks.values())]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    # with accounts on, the schema and the docs pages are not served: they are FastAPI's own routes, outside the role check
    docs = {} if settings.auth == "none" else {"openapi_url": None, "docs_url": None, "redoc_url": None}
    app = FastAPI(title="DCLab notebook", lifespan=lifespan, **docs)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.state.services = s
    app.state.pool = pool
    app.state.settings = settings
    app.state.store, app.state.tasks, app.state.projects, app.state.drafts, app.state.models = s.store, s.tasks, s.projects, s.drafts, s.gateway

    @app.middleware("http")
    async def json_by_default(request: Request, call_next):
        # The routes read their bodies as JSON whatever content type the client sent (curl -d sends a form type); typed
        # bodies keep that: a write to /api/ without a JSON content type is read as JSON. The uploads (…/data) take raw bytes.
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.url.path.startswith("/api/") and not request.url.path.endswith("/data"):
            kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if kind in ("", "text/plain", "application/x-www-form-urlencoded"):
                headers = [(k, v) for k, v in request.scope["headers"] if k != b"content-type"] + [(b"content-type", b"application/json")]
                request.scope["headers"] = headers
        return await call_next(request)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        # Who is asking and in which workspace (package 10.2): the session cookie or an API token, or the local owner
        # without sign-in. A browser's write carries its request token (the session's, or the server's without
        # accounts); an API token is not a cookie, so it needs none. /mcp is JSON-RPC for MCP clients.
        if request.url.path.startswith("/static/"):
            who = LOCAL_OWNER if settings.auth == "none" else ANONYMOUS  # files of the frontend: no data, no check
        else:
            who = await asyncio.to_thread(identify, dict(request.headers), dict(request.cookies), settings, s.workspace_id)
        if request.url.path.startswith("/mcp") and settings.auth != "none":
            if not who.signed_in or who.via != "token":
                return JSONResponse({"detail": "MCP clients send an API token: Authorization: Bearer dclab_…"}, status_code=401)
            if not allows(who.role, "write"):
                return JSONResponse({"detail": f"A {who.public()['role_label']} may not use the MCP tools"}, status_code=403)
        elif request.method not in ("GET", "HEAD", "OPTIONS") and not request.url.path.startswith("/mcp") and who.via != "token":
            expected = who.csrf if who.via == "session" else s.csrf
            if not secrets.compare_digest(request.headers.get("x-dclab-token", ""), expected or ""):
                return JSONResponse({"detail": "Missing local UI request token"}, status_code=403)
        try:
            workspace = await asyncio.to_thread(pool.get, who.workspace_id) if who.signed_in else s  # opening one is slow: not on the loop
        except KeyError:
            return JSONResponse({"detail": "This workspace no longer exists"}, status_code=404)
        if who.user_id and settings.auth != "none" and (request.url.path.startswith("/api/") or request.url.path.startswith("/mcp")):
            try:  # requests a minute, per person and per workspace (package 10.6): one person cannot starve the rest
                await asyncio.to_thread(limits.charge, "requests", 1, who.user_id, workspace.workspace_id)
            except limits.LimitExceeded as over:
                return JSONResponse({"detail": over.body}, status_code=429, headers={"Retry-After": str(over.body["retry_after"])})
        principal_token, services_token = set_current(who), context.set_services(workspace)
        try:
            response = await call_next(request)
        finally:
            context.reset_services(services_token)
            reset_current(principal_token)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = CSP
        return response

    for router in ROUTERS:
        app.include_router(router)
    app.include_router(accounts_api.router)
    # the draft routes and the pages hold stand-ins: each use reaches the request's workspace (agentic/current.py)
    c = {name: Current(name, s) for name in ("store", "projects", "drafts", "intern_sessions", "gateway", "draft_tasks", "traces", "jobs",
                                             "intern_jobs", "draft_jobs", "job_store", "audit")}
    draft_api.register(app, c["drafts"], c["projects"], c["gateway"], c["draft_tasks"], c["traces"], services=Current(None, s))
    pages.register_all(app, pages.Context(store=c["store"], projects=c["projects"], drafts=c["drafts"], intern_sessions=c["intern_sessions"],
                                          models=c["gateway"], jobs=c["jobs"], intern_jobs=c["intern_jobs"], draft_jobs=c["draft_jobs"],
                                          job_store=c["job_store"], audit=c["audit"]))
    if s.mcp is not None:
        app.mount("/mcp", app=s.mcp.handle_request, name="mcp")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=Settings.load().port)

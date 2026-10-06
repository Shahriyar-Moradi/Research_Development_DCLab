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
from .services import Services
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
    s = Services(settings)
    s.mcp = mcp_server.session_manager(Toolbox(s.projects)) if mcp_server.available() else None

    @asynccontextmanager
    async def lifespan(app):
        for run in s.store.list():
            if run["status"] in ("queued", "running", "pausing"):
                s.store.update(run["id"], status="interrupted", phase="Server restarted; resume explicitly")
        s.worker.start()  # recovers jobs a dead worker left running, then runs queued ones (package 10.3)
        if s.mcp is None:
            yield
        else:
            async with s.mcp.run():  # the MCP transport lives as long as the server
                yield
        await asyncio.to_thread(s.worker.stop)  # running jobs stop at their next checkpoint, interrupted and retryable
        active = list(s.tasks.values()) + list(s.draft_tasks.values())
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    app = FastAPI(title="DCLab notebook", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.state.services = s
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
        # /mcp is JSON-RPC for local MCP clients (Chat UI, Claude Desktop…); the host check still applies to it.
        if request.method not in ("GET", "HEAD", "OPTIONS") and not request.url.path.startswith("/mcp"):
            token = request.headers.get("x-dclab-token", "")
            if not secrets.compare_digest(token, s.csrf):
                return JSONResponse({"detail": "Missing local UI request token"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = CSP
        return response

    for router in ROUTERS:
        app.include_router(router)
    draft_api.register(app, s.drafts, s.projects, s.gateway, s.draft_tasks, s.traces, services=s)
    pages.register_all(app, pages.Context(store=s.store, projects=s.projects, drafts=s.drafts, intern_sessions=s.intern_sessions, models=s.gateway,
                                          jobs=s.jobs, intern_jobs=s.intern_jobs, draft_jobs=s.draft_jobs, job_store=s.job_store))
    if s.mcp is not None:
        app.mount("/mcp", app=s.mcp.handle_request, name="mcp")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=Settings.load().port)

"""The product server's settings, read once when an app is built (package 10.1).

Everything the server itself used to read from ``os.environ`` here and there is one typed object on
``app.state.services.settings``: where the workspace lives, which database and file storage it uses, the upload limit,
the model the research campaign defaults to, and the switches that keep tests away from live models.

What stays where it is, on purpose: the model tiers and their keys are read by ``dclab_rnd.models.settings`` on each
request (a key is never copied into an object a page could serialise, and the routing can change a tier while the
server runs); the database URL is read by ``storage.db`` for command-line tools that build no app. This object
mirrors those values for the pages, without the secrets.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_MAX_BYTES = 200 * 1024 * 1024  # the project upload route; the draft route (draft/api.py MAX_UPLOAD) and the connectors (connectors.MAX_BYTES) keep the same limit in their own constants


def load_secret_files() -> list[str]:
    """``NAME_FILE`` secrets (``dclab_rnd/secret_files.py``); an unreadable one raises RuntimeError."""
    from . import secret_files

    return secret_files.load(strict=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", populate_by_name=True)

    agent_home: Path = Field(ROOT / "agent_runs", alias="DCLAB_AGENT_HOME", description="the workspace folder")
    database_url: str = Field("", alias="DCLAB_DATABASE_URL", description="PostgreSQL for projects, drafts and sessions; files when empty")
    files_url: str = Field("", alias="DCLAB_FILES_URL", description="s3://bucket/prefix for tables and artifacts; the workspace folder when empty")
    ml_python: Path = Field(ROOT / ".venv/bin/python", alias="DCLAB_ML_PYTHON", description="the interpreter the research campaign runs experiments with")
    openai_model: str = Field("gpt-5.6-terra", alias="OPENAI_MODEL", description="the research campaign's default model")
    openai_key_set: bool = Field(False, description="whether OPENAI_API_KEY is set (the key itself is never kept here)")
    no_live_models: bool = Field(False, alias="DCLAB_NO_LIVE_MODELS", description="tests and the end-to-end flows: no request reaches a live model")
    workspace_monthly_eur: float | None = Field(None, alias="DCLAB_WORKSPACE_MONTHLY_EUR")
    port: int = Field(8765, alias="DCLAB_PORT")
    allowed_hosts: str = Field("", alias="DCLAB_ALLOWED_HOSTS", description="host names the server answers to besides 127.0.0.1 and localhost, comma separated")
    auth: str = Field("none", alias="DCLAB_AUTH", description="none: one owner on this machine; password or oidc: accounts, sessions and roles (PostgreSQL)")
    cookie_secure: bool = Field(False, alias="DCLAB_COOKIE_SECURE", description="send the session cookie over HTTPS only (set it behind TLS)")
    worker: str = Field("inline", alias="DCLAB_WORKER", description="inline: the server runs jobs too; external: only `python -m dclab_rnd.worker` does")
    worker_threads: int = Field(4, alias="DCLAB_WORKER_THREADS", description="jobs one worker runs at a time")

    @field_validator("auth", mode="before")
    @classmethod
    def _auth(cls, value):
        value = str(value or "").strip().lower()
        return value if value in ("password", "oidc") else "none"

    @field_validator("cookie_secure", mode="before")
    @classmethod
    def _secure(cls, value):
        return str(value).strip() == "1" if value not in (None, True, False) else bool(value)

    @field_validator("worker", mode="before")
    @classmethod
    def _worker(cls, value):
        return "external" if str(value or "").strip().lower() == "external" else "inline"

    @field_validator("worker_threads", mode="before")
    @classmethod
    def _threads(cls, value):
        try:
            return max(1, min(int(value), 64))
        except (TypeError, ValueError):
            return 4

    @field_validator("no_live_models", mode="before")
    @classmethod
    def _flag(cls, value):
        return str(value).strip() == "1" if value not in (None, True, False) else bool(value)

    @field_validator("port", "workspace_monthly_eur", "agent_home", "ml_python", mode="before")
    @classmethod
    def _empty_is_unset(cls, value, info):
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].default
        return value

    @field_validator("port", mode="before")
    @classmethod
    def _port(cls, value):
        try:
            return int(value)
        except (TypeError, ValueError):  # a bad value never stops the server from starting; the default serves
            return 8765

    @field_validator("workspace_monthly_eur", mode="before")
    @classmethod
    def _cap(cls, value):
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):  # the gateway reports a bad cap on its own (dclab_rnd/models/gateway.py)
            return None

    def hosts(self) -> list[str]:
        """What the host check lets through: this machine, the test client, and the names set in DCLAB_ALLOWED_HOSTS
        (as a person would write them: a scheme, a port and capitals are taken off, as the check compares names only)."""
        import re

        extra = []
        for host in (h.strip() for h in self.allowed_hosts.split(",") if h.strip()):
            host = re.sub(r"^[a-z]+://", "", host.lower()).split("/")[0]
            host = host if host.startswith("[") else host.split(":")[0]  # [::1]:8443 keeps its brackets
            if host and ("*" not in host or (host.startswith("*.") and "*" not in host[2:])):
                extra.append(host)
        return ["127.0.0.1", "localhost", "testserver", *extra]

    def problems(self) -> list[str]:
        """What stops this server from starting, in words (package 12.2): checked once, before anything is served."""
        import os

        from . import secret_files

        out = list(secret_files.ERRORS)
        raw_auth = (os.environ.get("DCLAB_AUTH") or "").strip().lower()
        if raw_auth not in ("", "none", "password", "oidc"):  # a typo must not leave a server without sign-in
            out.append(f"DCLAB_AUTH is password, oidc or none, not {raw_auth[:20]!r}")
        if self.allowed_hosts.strip() and self.auth == "none":
            out.append("DCLAB_ALLOWED_HOSTS serves this server under other names, but without sign-in anyone who reaches it is the owner: "
                       "set DCLAB_AUTH (password or oidc)")
        for host in (h.strip() for h in self.allowed_hosts.split(",") if h.strip()):
            if "*" in host and not (host.startswith("*.") and "*" not in host[2:]):
                out.append(f"DCLAB_ALLOWED_HOSTS: {host[:40]!r} — a wildcard is only allowed as *.domain")
        if self.auth != "none" and not self.database_url:
            out.append("DCLAB_AUTH needs PostgreSQL: accounts and sessions live in the database (set DCLAB_DATABASE_URL)")
        if self.database_url:
            from .storage import db

            problem = db.reachable(self.database_url)
            if problem:
                out.append(f"the database in DCLAB_DATABASE_URL does not answer ({problem.split(':')[0]}); start it, or check the URL")
            else:
                at, newest = db.current(self.database_url), db.head()
                if at != newest:
                    out.append(f"the database's schema is at {at or 'nothing'}, this code needs {newest}: run python -m dclab_rnd.storage upgrade")
        if self.auth == "oidc":
            from .accounts import oidc

            try:
                oidc.config()
            except oidc.OidcError as exc:
                out.append(str(exc))
        numbers = [("DCLAB_MODEL_RETRIES", int), ("DCLAB_MODEL_BACKOFF", float), ("DCLAB_DB_POOL_SIZE", int), ("DCLAB_DB_MAX_OVERFLOW", int),
                   ("DCLAB_PORT", int), ("DCLAB_WORKER_THREADS", int), ("DCLAB_JOB_STALE_SECONDS", float), ("DCLAB_MCP_PORT", int),
                   ("DCLAB_WORKSPACE_MONTHLY_EUR", float), ("DCLAB_LIMIT_USER_MONTHLY_EUR", float)]
        numbers += [(n, float) for n in sorted(os.environ) if n.startswith("DCLAB_LIMIT_") and n != "DCLAB_LIMIT_USER_MONTHLY_EUR"]
        for name, cast in numbers:
            raw = os.environ.get(name)
            if raw not in (None, ""):
                try:
                    cast(raw)
                except ValueError:
                    out.append(f"{name} must be a number, not {raw[:20]!r}")
        return out

    @classmethod
    def load(cls, home: Path | str | None = None) -> "Settings":
        """Read the environment once; ``home`` overrides the workspace folder (tests, the e2e server)."""
        import os

        load_secret_files()
        values = {"openai_key_set": bool(os.environ.get("OPENAI_API_KEY"))}
        if not os.environ.get("DCLAB_WORKSPACE_MONTHLY_EUR", "").strip():
            values["workspace_monthly_eur"] = None
        settings = cls(**values)
        if home is not None:
            settings = settings.model_copy(update={"agent_home": Path(home)})
        return settings

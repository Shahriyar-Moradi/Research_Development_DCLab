"""Tests never reach a live model unless a test asks for one (the owner's decision D3, 2026-10-23).

Until now one environment switch (``DCLAB_NO_LIVE_MODELS=1``) kept tests away from real models, and every entry point had to
remember to set it: four times one did not, and a test spent on the owner's key (the last, bae85e29: an API test's intern
was planned by a live model). This guard sits where a real provider client is created (``AnthropicClient.client``,
``ChatClient.client``), so no path around the gateway, the twin lane or a script can skip it.

The owner's providers (2026-10-25): ``DCLAB_BLOCK_PROVIDERS=openai`` in ``.env`` blocks a provider on this machine, its key
kept: no client to it is created, and a tier that points at it is unavailable (the rules serve). The owner uses Claude
only. A live process that never read ``.env`` (a command-line run) reads the owner's model settings from it all the
same (``owner_value``), so it reaches the same model as the server, never a key that happens to sit in the shell.

A process is a test when the test runner marked it (``tests/run_parallel.py`` sets ``DCLAB_TEST_RUN=1``), when it was
started as ``python -m unittest`` or under pytest, or when a test case is on the call stack. The marks hold in every thread:
the API server's background turns have no test frame on their stack. A test that means to reach a model (a fake server on
localhost, a measured live run) says so with ``DCLAB_ALLOW_LIVE_MODELS_IN_TESTS=1``. A client a test injects itself (a
scripted fake) is never touched: only creating a real SDK client is refused.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

ALLOW = "DCLAB_ALLOW_LIVE_MODELS_IN_TESTS"
MARK = "DCLAB_TEST_RUN"
TEST_FRAMES = ("/unittest/case.py", "/unittest/main.py", "/_pytest/")


BLOCK = "DCLAB_BLOCK_PROVIDERS"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
BLOCKED_HOSTS = {"openai": ("openai.com",), "anthropic": ("anthropic.com",), "openrouter": ("openrouter.ai",)}  # a provider by its hosts: a local server is never blocked


class LiveModelInTest(RuntimeError):
    """A test tried to create a real model client without asking for one."""


class ProviderBlocked(RuntimeError):
    """The owner blocked this provider on this machine (``DCLAB_BLOCK_PROVIDERS``): no request is sent to it."""


_ENV_CACHE: dict[str, Any] = {}


def _dotenv() -> dict[str, str]:
    """The owner's .env as values (read again when it changes), never loaded into the process."""
    try:
        stamp = ENV_FILE.stat().st_mtime
    except OSError:
        return {}
    if _ENV_CACHE.get("stamp") != stamp:
        try:
            from dotenv import dotenv_values

            parsed = {k: str(v or "") for k, v in dotenv_values(ENV_FILE).items()}
        except ImportError:
            parsed = {}
            for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    parsed[k.strip()] = v.strip().strip("'\"")
        _ENV_CACHE.update(stamp=stamp, values=parsed)
    return dict(_ENV_CACHE.get("values") or {})


def owner_value(name: str) -> str:
    """A model setting: the process's own value first (an empty one counts: the test runner blanks the owner's), else the
    owner's .env, never inside a test (a test sets the model settings it reads itself)."""
    if name in os.environ:
        return os.environ[name]
    if in_test():
        return ""
    return _dotenv().get(name, "")


def blocked(endpoint: str) -> Optional[str]:
    """The blocked provider ``endpoint`` belongs to, or None."""
    names = {n.strip().lower() for n in owner_value(BLOCK).split(",") if n.strip()}
    host = (urlparse(str(endpoint or "")).hostname or "").lower()
    for name in names:
        if host and any(host == h or host.endswith("." + h) for h in BLOCKED_HOSTS.get(name, ())):
            return name
    return None


def in_test() -> bool:
    if os.environ.get(MARK) == "1" or "_pytest" in sys.modules or "unittest" in os.path.basename(str(sys.argv[0] if sys.argv else "")):
        return True
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename.replace("\\", "/")
        if any(marker in name for marker in TEST_FRAMES):
            return True
        frame = frame.f_back
    return False


def check(provider: str, endpoint: str = "") -> None:
    """Refuse to create a real ``provider`` client inside a test, unless the test asked for one, and anywhere for a
    provider the owner blocked."""
    who = blocked(endpoint)
    if who:
        raise ProviderBlocked(f"{who} is blocked on this machine ({BLOCK} in .env): no request is sent to {urlparse(endpoint).hostname}")
    if os.environ.get(ALLOW) == "1" or not in_test():
        return
    where = f" at {endpoint}" if endpoint else ""
    raise LiveModelInTest(f"a test tried to reach a live {provider} model{where}; tests use scripted transports "
                          f"(set {ALLOW}=1 in the test that means to reach one)")

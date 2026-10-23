"""Tests never reach a live model unless a test asks for one (the owner's decision D3, 2026-10-23).

Until now one environment switch (``DCLAB_NO_LIVE_MODELS=1``) kept tests away from real models, and every entry point had to
remember to set it: four times one did not, and a test spent on the owner's key (the last, bae85e29: an API test's intern
was planned by a live model). This guard sits where a real provider client is created (``AnthropicClient.client``,
``ChatClient.client``), so no path around the gateway, the twin lane or a script can skip it.

A process is a test when the test runner marked it (``tests/run_parallel.py`` sets ``DCLAB_TEST_RUN=1``), when it was
started as ``python -m unittest`` or under pytest, or when a test case is on the call stack. The marks hold in every thread:
the API server's background turns have no test frame on their stack. A test that means to reach a model (a fake server on
localhost, a measured live run) says so with ``DCLAB_ALLOW_LIVE_MODELS_IN_TESTS=1``. A client a test injects itself (a
scripted fake) is never touched: only creating a real SDK client is refused.
"""

from __future__ import annotations

import os
import sys

ALLOW = "DCLAB_ALLOW_LIVE_MODELS_IN_TESTS"
MARK = "DCLAB_TEST_RUN"
TEST_FRAMES = ("/unittest/case.py", "/unittest/main.py", "/_pytest/")


class LiveModelInTest(RuntimeError):
    """A test tried to create a real model client without asking for one."""


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
    """Refuse to create a real ``provider`` client inside a test, unless the test asked for one."""
    if os.environ.get(ALLOW) == "1" or not in_test():
        return
    where = f" at {endpoint}" if endpoint else ""
    raise LiveModelInTest(f"a test tried to reach a live {provider} model{where}; tests use scripted transports "
                          f"(set {ALLOW}=1 in the test that means to reach one)")

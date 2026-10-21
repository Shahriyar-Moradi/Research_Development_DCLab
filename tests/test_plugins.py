"""Plugins: an installed package's entry point registers routes; a broken one is left out and the studio still starts."""

import importlib.metadata
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import extensions  # noqa: E402

REAL_ENTRY_POINTS = importlib.metadata.entry_points


def only_ours(points):
    """The studio's plugin group gets ``points``; every other library keeps its real entry points (alembic reads its own
    group when it is first imported, which a database run does while the studio starts)."""
    def entry_points(*args, **kwargs):
        if kwargs.get("group") == f"{extensions.PACKAGE}.plugins":
            return points
        return REAL_ENTRY_POINTS(*args, **kwargs)
    return entry_points


class Point:
    def __init__(self, name, register):
        self.name, self._register = name, register

    def load(self):
        return self._register


def good(app):
    from dclab_rnd.agentic.api_models import Router

    router = Router()

    @router.get("/api/plugin/hello")
    async def hello():
        return {"hello": "from a plugin"}
    app.include_router(router)


def broken(app):
    raise RuntimeError("this plugin is broken")


class PluginTests(unittest.TestCase):
    def test_a_plugin_adds_its_route_and_a_broken_one_is_left_out(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        points = [Point("good", good), Point("broken", broken)]
        with mock.patch("importlib.metadata.entry_points", side_effect=only_ours(points)), self.assertLogs(extensions.__name__, "WARNING") as logs:
            app = create_app(Path(tempfile.mkdtemp()))
        self.assertEqual(app.state.plugins, ["good"])
        self.assertIn("broken", logs.output[0])
        self.assertEqual(TestClient(app).get("/api/plugin/hello").json(), {"hello": "from a plugin"})

    def test_without_plugins_nothing_is_loaded(self):
        with mock.patch("importlib.metadata.entry_points", side_effect=only_ours([])):
            self.assertEqual(extensions.load_plugins(object()), [])


if __name__ == "__main__":
    unittest.main()

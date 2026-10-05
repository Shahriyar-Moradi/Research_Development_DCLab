"""The product frontend (dclab_rnd/agentic/web) builds into CSP-safe static files and is served at /."""
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agentic.web import build  # noqa: E402


class BuildTests(unittest.TestCase):
    def test_committed_build_is_current(self):
        self.assertEqual(build.main(["--check"]), 0, "run: python -m dclab_rnd.agentic.web.build")

    def test_nothing_the_csp_would_block(self):
        files = build.assemble()
        self.assertEqual(build.problems(files), [])
        page = files["index.html"]
        self.assertIn('<link rel="stylesheet" href="/static/app/app.css">', page)
        self.assertEqual(len(re.findall(r"<script src=", page)), len(re.findall(r"<script", page)))
        self.assertIn('id="view-home"', page)
        self.assertIn("/static/app/js/core.js", page)

    def test_inline_style_attributes_are_caught(self):
        files = {"index.html": "<div class=\"app\"></div>", "js/views/x.js": "el.innerHTML = '<b style=\"color:red\">x</b>';"}
        self.assertTrue(any("inline style" in p for p in build.problems(files)))
        files["js/views/x.js"] = "el.innerHTML = '<b data-style=\"color:red\">x</b><text font-style=\"italic\">';"
        self.assertEqual(build.problems(files), [])


class ServeTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))

    def test_root_serves_the_product_and_classic_keeps_the_old_ui(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="view-home"', page.text)
        csp = page.headers["Content-Security-Policy"]
        self.assertIn("script-src 'self';", csp)
        self.assertNotIn("unsafe-inline", csp)
        for asset in ("/static/app/app.css", "/static/app/js/core.js", "/static/app/js/views/home.js"):
            self.assertEqual(self.client.get(asset).status_code, 200, asset)
        self.assertIn('id="map-view"', self.client.get("/classic").text)


if __name__ == "__main__":
    unittest.main()

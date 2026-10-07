"""Research map: the structured records behind research/<track>/INDEX.md and the Studio's Research map page."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import research_map  # noqa: E402


class ResearchMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = research_map.research_map()

    def test_every_track_readme_status_is_read_in_both_forms(self):
        self.assertEqual(research_map.readme_status("**Status:** active · code lives in `dclab_rnd/`"), "active · code lives in `dclab_rnd/`")
        self.assertEqual(research_map.readme_status("**Status: proposed research.** The idea is to test…"), "proposed research")
        self.assertEqual(research_map.readme_status("# Title\n\nNo status line."), "")
        missing = [name for name, track in self.map["tracks"].items() if not track["status"]]
        self.assertEqual(missing, [], "a track README declares a status the map does not read")

    def test_generated_files_are_current(self):
        stale = [p for p, text in research_map.render_all().items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        self.assertEqual(stale, [], "run: make research-index")

    def test_every_track_is_in_exactly_one_theme_and_has_every_section(self):
        placed = [t for theme in self.map["themes"] for t in theme["tracks"]]
        self.assertEqual(sorted(placed), sorted(self.map["tracks"]))
        self.assertEqual(len(placed), len(set(placed)))
        for track in self.map["tracks"].values():
            for key in ("idea", "champion", "experiments", "notebooks", "evaluation", "reports", "related", "counts"):
                self.assertIn(key, track)

    def test_records_point_at_real_files_and_tracks(self):
        files = set(research_map.tracked_files())
        for track in self.map["tracks"].values():
            for row in track["champion"]["rows"]:
                self.assertIn(row["source"], files)
            for group in track["notebooks"]:
                self.assertTrue(all(f in files and f.endswith(".ipynb") for f in group["files"]))
            self.assertTrue(all(r in self.map["tracks"] for r in track["related"]))
        for a, b in self.map["edges"]:
            self.assertIn(a, self.map["tracks"])
            self.assertIn(b, self.map["tracks"])

    def test_leaky_tabpfn_champion_stays_flagged(self):
        rows = self.map["tracks"]["tabular-foundation-models"]["champion"]["rows"]
        self.assertTrue(any("⚠" in r["dataset"] for r in rows if "hyper" in r["dataset"].lower()))

    def test_preview_reads_tracked_text_only(self):
        self.assertEqual(research_map.preview("research/README.md")["kind"], "md")
        folder = research_map.preview("research/churn-prediction")
        self.assertEqual(folder["kind"], "folder")
        self.assertIn("research/churn-prediction/README.md", folder["entries"])
        for bad in ("", "/etc/passwd", "../.env", ".env", "research/../.env", "dclab_rnd/tools.py", "research/nope.md"):
            with self.assertRaises(ValueError, msg=bad):
                research_map.preview(bad)


class ResearchApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))

    def test_map_and_preview_endpoints(self):
        data = self.client.get("/api/research").json()
        self.assertEqual(data["totals"]["tracks"], len(data["tracks"]))
        self.assertEqual(self.client.get("/api/research/file", params={"path": "research/README.md"}).status_code, 200)
        self.assertEqual(self.client.get("/api/research/file", params={"path": "../.env"}).status_code, 404)

    def test_the_research_map_is_on_the_product_s_lab_page(self):
        page = self.client.get("/").text  # the earlier UI at /classic was removed (package 8.3)
        self.assertIn('id="view-lab"', page)
        self.assertEqual(self.client.get("/static/app/js/views/lab.js").status_code, 200)
        self.assertEqual(self.client.get("/classic").status_code, 404)


if __name__ == "__main__":
    unittest.main()

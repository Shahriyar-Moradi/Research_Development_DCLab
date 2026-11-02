"""A run on any store but the owner's never replaces the committed ``training_records/STORE_CATALOG.json``.

In a cloud session the default store folder (``training_records/store``, a gitignored clone of ``dclab-records``) is absent;
starting the server or running a store pass there made an empty store and wrote its empty view over the owner's catalog
(about 21,000 lines gone, twice). The copy is written only for a default-rooted store that is a git clone, or where a caller says.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.store import catalog, coverage, layout as layout_module  # noqa: E402


class CatalogCopyTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.default = tmp / "store"  # stands in for training_records/store
        self.copy = tmp / "STORE_CATALOG.json"
        self.copy.write_text('{"owner": "real"}\n', encoding="utf-8")
        self.patches = [mock.patch.object(layout_module, "DEFAULT", self.default), mock.patch.object(catalog, "COPY", self.copy),
                        mock.patch.object(layout_module, "CATALOG_COPY", self.copy)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_an_empty_default_store_leaves_the_committed_copy_alone(self):
        layout = layout_module.Layout(self.default)  # no clone here, as on a fresh checkout
        catalog.write(layout)
        coverage.write(layout, coverage.measure(layout, home=Path(tempfile.mkdtemp())))
        self.assertEqual(json.loads(self.copy.read_text(encoding="utf-8")), {"owner": "real"})

    def test_another_store_leaves_it_alone(self):
        other = layout_module.Layout(Path(tempfile.mkdtemp()))
        (other.root / ".git").mkdir()
        catalog.write(other)
        self.assertEqual(json.loads(self.copy.read_text(encoding="utf-8")), {"owner": "real"})

    def test_the_owners_clone_still_rebuilds_the_copy(self):
        (self.default / ".git").mkdir(parents=True)
        layout = layout_module.Layout(self.default)
        catalog.write(layout)
        held = json.loads(self.copy.read_text(encoding="utf-8"))
        self.assertIn("kinds", held)
        self.assertNotIn("owner", held)
        self.assertNotIn("built_at", held)


if __name__ == "__main__":
    unittest.main()

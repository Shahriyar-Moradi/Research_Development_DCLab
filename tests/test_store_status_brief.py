"""``store status --brief``: the store's counts short enough for a chat to read (a chat ran ``store status`` on the owner's
store and got 840 KB, the workspaces' names); the full status is unchanged."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class BriefStatusTest(unittest.TestCase):
    def test_brief_counts_workspaces_and_the_full_status_names_them(self):
        from dclab_rnd.store import __main__ as cli

        many = {"entries": 3, "sources": 2, "bytes": 900, "workspaces": [f"w{i}" for i in range(500)], "last": "2026-10-26T01:00:00+00:00"}
        none = {"entries": 0, "sources": 0, "bytes": 0, "workspaces": [], "last": None}
        root = tempfile.mkdtemp()
        from dclab_rnd.store.store import Store

        with mock.patch.object(Store, "counts", return_value={"sessions": many, "decisions": none}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(["status", "--brief", "--root", root])
            brief = json.loads(out.getvalue())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(["status", "--root", root])
            full = json.loads(out.getvalue())
        self.assertEqual(brief["kinds"]["sessions"]["workspaces"], 500)
        self.assertEqual(brief["total"], {"entries": 3, "bytes": 900, "last": "2026-10-26T01:00:00+00:00"})
        self.assertLess(len(json.dumps(brief)), 600)
        self.assertEqual(len(full["kinds"]["sessions"]["workspaces"]), 500, "the full status is unchanged")


if __name__ == "__main__":
    unittest.main()

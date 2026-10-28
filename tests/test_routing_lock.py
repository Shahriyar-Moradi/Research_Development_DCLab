"""22.6: the routing document's read-change-write is one step across processes. Four processes add to one counter at
once through FileRouting.change; none of their changes is lost (a read, then a write, lost some before)."""

import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.models.routing import FileRouting, atomic  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ADD = ("import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from dclab_rnd.models.routing import FileRouting; "
       "store = FileRouting(Path(sys.argv[2])); "
       "[store.change(lambda doc: {**doc, 'count': int(doc.get('count') or 0) + 1}) for _ in range(int(sys.argv[3]))]")


class RoutingLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_no_change_is_lost_across_processes(self):
        path = self.tmp / "model_routing.json"
        workers = [subprocess.Popen([sys.executable, "-c", ADD, str(ROOT), str(path), "25"]) for _ in range(4)]  # four processes at once
        codes = [w.wait(timeout=120) for w in workers]
        self.assertEqual(codes, [0, 0, 0, 0])
        self.assertEqual(FileRouting(path).get().get("count"), 100)

    def test_a_change_that_returns_nothing_writes_nothing(self):
        path = self.tmp / "model_routing.json"
        store = FileRouting(path)
        self.assertEqual(atomic(store, lambda doc: None), {"purposes": {}, "history": []})
        self.assertFalse(path.exists())

    def test_a_store_without_change_is_read_and_written_in_turn(self):
        class Plain:
            def __init__(self):
                self.doc = {"purposes": {}, "history": []}

            def get(self):
                return dict(self.doc)

            def save(self, doc):
                self.doc = doc
                return doc

        plain = Plain()
        self.assertEqual(atomic(plain, lambda doc: {**doc, "x": 1})["x"], 1)


if __name__ == "__main__":
    unittest.main()

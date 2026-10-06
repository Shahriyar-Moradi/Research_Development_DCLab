"""A project page read while a stage runs sees the old record or the new one, never half of it."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.studio.store import ProjectStore  # noqa: E402


class AtomicWriteTests(unittest.TestCase):
    def test_readers_never_see_a_half_written_record_or_project(self):
        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("Race", "general", "Predict churn")["id"]
        big = {"rows": [{"i": i, "text": "x" * 200} for i in range(3000)]}  # large enough that a plain write is seen in pieces
        store.write_stage(pid, "data", big)
        done, errors = threading.Event(), []

        def writer():
            try:
                for n in range(30):
                    store.write_stage(pid, "data", {**big, "n": n})
                    store.save({**store.get(pid), "note": n})  # two writers: a shared temporary name once lost a save
            except Exception as error:  # noqa: BLE001
                errors.append(repr(error))
            finally:
                done.set()

        def reader():
            while not done.is_set():
                try:
                    store.read_stage(pid, "data")
                    store.get(pid)
                except (json.JSONDecodeError, FileNotFoundError) as error:
                    errors.append(repr(error))
        threads = [threading.Thread(target=writer), threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(p.name for p in store.directory(pid).rglob("*.tmp")), [])  # no temporary file left behind


if __name__ == "__main__":
    unittest.main()

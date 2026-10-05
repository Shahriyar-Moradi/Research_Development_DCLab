"""The storage interface (dclab_rnd/storage): the file stores satisfy it, and nothing reaches around it."""
import inspect
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import storage  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.intern.sessions import SessionStore  # noqa: E402
from dclab_rnd.studio.store import ProjectStore  # noqa: E402

PAIRS = ((storage.Projects, ProjectStore), (storage.Drafts, DraftStore), (storage.Sessions, SessionStore))
# The stores themselves, and the research-run code whose own SQLite store (agentic/store.py) is not part of this interface.
OWNERS = ("dclab_rnd/studio/store.py", "dclab_rnd/draft/store.py", "dclab_rnd/intern/sessions.py", "dclab_rnd/agentic/store.py",
          "dclab_rnd/agentic/clean_export.py", "dclab_rnd/agentic/engine.py", "dclab_rnd/agentic/archive.py", "dclab_rnd/agentic/worker.py")
INTERNALS = re.compile(r"\b(projects|drafts|intern_sessions|sessions|self\.projects|self\.drafts|ctx\.projects|ctx\.drafts|ctx\.intern_sessions)"
                       r"\.(home|directory|stage_path|path|_locks|_turns|lock)\b")
WIRING = re.compile(r"(ProjectStore|SessionStore|DraftStore)\(")  # the one place stores are built from folders


class StorageInterfaceTests(unittest.TestCase):
    def test_file_stores_satisfy_their_protocols_with_matching_parameters(self):
        for protocol, store in PAIRS:
            instance = store(Path(tempfile.mkdtemp()))
            self.assertIsInstance(instance, protocol)
            for name, declared in inspect.getmembers(protocol, inspect.isfunction):
                if name.startswith("_"):
                    continue
                wanted = [p for p in inspect.signature(declared).parameters if p != "self"]
                got = [p for p in inspect.signature(getattr(store, name)).parameters if p != "self"]
                self.assertEqual(got[:len(wanted)], wanted, f"{store.__name__}.{name}")

    def test_no_code_outside_the_stores_reaches_into_their_folders(self):
        found = []
        for path in sorted((ROOT / "dclab_rnd").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in OWNERS:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if INTERNALS.search(line) and not WIRING.search(line):
                    found.append(f"{rel}:{number}: {line.strip()[:110]}")
        self.assertEqual(found, [], "use a method of dclab_rnd.storage instead")

    def test_the_methods_added_for_the_interface(self):
        projects, drafts = ProjectStore(Path(tempfile.mkdtemp()) / "projects"), DraftStore(Path(tempfile.mkdtemp()))
        project = projects.create("Churn")
        self.assertFalse(projects.has_stage(project["id"], "data"))
        projects.write_stage(project["id"], "data", {"stage": "data"})
        self.assertTrue(projects.has_stage(project["id"], "data"))
        self.assertTrue(projects.export_dir(project["id"]).is_dir())
        self.assertTrue(projects.location())
        draft = drafts.create("Predict which customers cancel")
        before = drafts.events_version(draft["id"])
        drafts.emit(draft["id"], "status", {"note": "x"})
        self.assertNotEqual(drafts.events_version(draft["id"]), before)
        with drafts.turn(draft["id"]):
            with drafts.turn(draft["id"]):  # re-entrant in one thread
                held = []
                other = threading.Thread(target=lambda: held.append(drafts.turn(draft["id"]).acquire(timeout=0.2)))
                other.start()
                other.join()
        self.assertEqual(held, [False])  # another thread waits for the turn


if __name__ == "__main__":
    unittest.main()

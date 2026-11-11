"""The storage interface (dclab_rnd/storage): the file stores satisfy it, and nothing reaches around it."""
import inspect
import re
import sys
import tempfile
import threading
import subprocess
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

    def test_project_effect_and_solution_locks_are_distinct_and_reentrant(self):
        with tempfile.TemporaryDirectory() as folder:
            first = ProjectStore(Path(folder))
            second = ProjectStore(Path(folder))
            pid = first.create("Task locks")["id"]
            entered = threading.Event()

            def effect():
                with second.turn(pid):
                    second.save(second.get(pid))
                    entered.set()

            with first.turn(pid, purpose="solution"):
                with second.turn(pid, purpose="solution"):
                    thread = threading.Thread(target=effect)
                    thread.start()
                    self.assertTrue(entered.wait(2))  # graph worker can take its separate effect lock
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_same_id_in_two_workspaces_keeps_both_process_locks(self):
        from dclab_rnd.studio import store as stores
        if stores.fcntl is None:
            self.skipTest("POSIX process lock probe")
        with tempfile.TemporaryDirectory() as folder:
            first, second = ProjectStore(Path(folder) / "a"), ProjectStore(Path(folder) / "b")
            project = first.create("A")
            pid = project["id"]
            second.directory(pid).mkdir()
            second.save(dict(project))
            probe = "import fcntl,sys\nf=open(sys.argv[1],'a')\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nsys.exit(1)"
            with first.turn(pid), second.turn(pid):
                for store in (first, second):
                    result = subprocess.run([sys.executable, "-c", probe, str(store.directory(pid) / ".project.lock")], timeout=3)
                    self.assertEqual(result.returncode, 0)

    def test_task_mode_refuses_thread_only_locks_without_changing_legacy_fallback(self):
        from unittest.mock import patch
        from dclab_rnd.tasks.dispatch import TaskDispatcher
        with tempfile.TemporaryDirectory() as folder:
            store = ProjectStore(Path(folder))
            pid = store.create("Unavailable process lock")["id"]
            with patch("dclab_rnd.studio.store.fcntl", None):
                with store.turn(pid):
                    store.save(store.get(pid))
                with self.assertRaises(RuntimeError):
                    TaskDispatcher(store, None, enabled=True).dispatch(pid, "data")


if __name__ == "__main__":
    unittest.main()

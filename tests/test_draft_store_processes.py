"""The file draft store across processes (review of 2026-11-22).

A team delivery handled by the worker (``python -m dclab_rnd.worker``) writes the Home chat's draft (``lead.to_home_chat``
→ ``drafts.update``) while a Home turn in the API process writes it too. The draft's lock was a thread lock, held in one
process only, so the two read-modify-writes could overwrite each other. Two processes update one draft many times here:
no update is lost, and every event keeps its own sequence number. PostgreSQL drafts lock their row and are not tested here.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft.store import DraftStore  # noqa: E402

N = 150
HAMMER = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from dclab_rnd.draft.store import DraftStore

store, draft_id, who, n = DraftStore(Path(sys.argv[2])), sys.argv[3], sys.argv[4], int(sys.argv[5])
for i in range(n):
    store.update(draft_id, lambda d: d["messages"].append({"who": who, "i": i}))
    store.emit(draft_id, "chat", {"who": who, "i": i})
"""


class DraftStoreAcrossProcessesTests(unittest.TestCase):
    def test_two_processes_updating_one_draft_lose_no_update(self):
        home = Path(tempfile.mkdtemp()) / "drafts"
        store = DraftStore(home)
        draft_id = store.create("Who should Sales call?")["id"]
        env = {**os.environ, "DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DATABASE_URL": ""}
        procs = [subprocess.Popen([sys.executable, "-c", HAMMER, str(ROOT), str(home), draft_id, who, str(N)], env=env,
                                  stderr=subprocess.PIPE, text=True) for who in ("api", "worker")]
        for proc in procs:
            _, err = proc.communicate(timeout=240)
            self.assertEqual(proc.returncode, 0, err)
        messages = store.get(draft_id)["messages"]
        for who in ("api", "worker"):
            self.assertEqual([m["i"] for m in messages if m["who"] == who], list(range(N)), who)  # every update kept, in order
        seqs = [e["seq"] for e in store.events(draft_id)]
        self.assertEqual(seqs, list(range(1, 2 * N + 1)))  # one number per event, none taken twice

    def test_a_deleted_draft_is_not_made_again_by_a_late_update(self):
        store = DraftStore(Path(tempfile.mkdtemp()) / "drafts")
        draft_id = store.create("x")["id"]
        store.delete(draft_id)
        with self.assertRaises(KeyError):
            store.update(draft_id, lambda d: None)
        self.assertFalse(store.directory(draft_id).exists())

    def test_a_draft_deleted_between_the_check_and_the_lock_is_a_missing_draft(self):
        """Reviewer round 2: another process deletes the folder after is_dir(): opening its lock file raised FileNotFoundError."""
        from unittest import mock

        store = DraftStore(Path(tempfile.mkdtemp()) / "drafts")
        draft_id = store.create("x")["id"]
        store.delete(draft_id)
        with mock.patch.object(Path, "is_dir", return_value=True):  # the folder was there when it was checked
            with self.assertRaises(KeyError):
                store.update(draft_id, lambda d: None)
        self.assertFalse(store.directory(draft_id).exists())


if __name__ == "__main__":
    unittest.main()

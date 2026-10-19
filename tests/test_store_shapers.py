"""Package 32.8: the shapers, raw into records.

Every raw entry has its records or a recorded reason why not; a record holds no secret, no person's contact in its words and
no key that quotes a table's cells, and its ids and hashes are left as they are; each record carries its provenance (its
deleted project marked), its capture rule's findings and its split; ``records/`` rebuilds byte for byte from ``raw/``.
"""

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class ShapersTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        self.store = Store(self.layout)
        at = "2026-10-20T10:00:00+00:00"
        decision = {"schema_version": "dclab_core.decision_record.v2", "id": "dg1234567890abcdef1234", "point": "column.is_identifier", "at": at,
                    "project": "p1", "state": {"rows": "1k-10k"}, "answers": [{"by": "rule", "value": True, "reason": "mail jane.roe@example.org"}],
                    "used": "rule", "value_used": True}
        label = {"schema_version": "dclab_core.decision_label.v2", "id": "dl1", "record": "dg1234567890abcdef1234", "value": True,
                 "source": "user_decision_exposed", "at": at}
        self.store.keep_log("decisions", "dev", "ledger/decisions/2026/10/20.jsonl", (json.dumps(decision) + "\n").encode())
        self.store.keep_log("decisions", "dev", "ledger/labels/2026/10/20.jsonl", (json.dumps(label) + "\n").encode())
        event = {"schema_version": "dclab_core.experience_event.v9", "id": "ev1", "at": at, "kind": "chat.person", "actor": "person",
                 "ids": {"visit": "v1", "project": "p1"}, "payload": {"text": "call +44 7700 900123, key sk-ant-api03-" + "x" * 30,
                                                                  "preview": ["Q7Kx@example.org"]}}
        self.store.keep_log("sessions", "dev", "experience/2026/10/20/v1.jsonl", (json.dumps(event) + "\n").encode())
        self.store.keep("students", "dev", "students/model.joblib", b"\x80fitted")
        self.store.keep("deletions", "dev", "deletions/project-p1.json", json.dumps({"what": "project", "id": "p1"}).encode())

    def shaped(self, kind):
        from dclab_rnd.store import shapers

        return list(shapers.records(self.layout, kind))

    def test_every_entry_is_shaped_or_says_why_and_no_secret_contact_or_cell_leaves(self):
        from dclab_rnd.store import shapers

        report = shapers.shape(self.layout)
        self.assertGreaterEqual(report["records"], 3)
        decisions = {r.type: r for r in self.shaped("decisions")}
        self.assertEqual(set(decisions), {"decision", "label"})
        d, lab = decisions["decision"], decisions["label"]
        self.assertEqual(d.body["id"], "dg1234567890abcdef1234")  # ids are left as they are
        self.assertEqual(d.body["group"], "project:p1")
        self.assertEqual(lab.body["group"], "project:p1")  # a label sealed with its decision
        self.assertEqual(d.split, lab.split)
        self.assertTrue(d.provenance.deleted)  # its project was deleted in the product; the store kept it
        self.assertNotIn("jane.roe", json.dumps(d.body))
        self.assertEqual(lab.flags, [])  # a person's label: confirmed in the store's words
        everything = b"".join(p.read_bytes() for p in (self.layout.root / "records").rglob("*.gz"))
        text = b"".join(gzip.decompress(p.read_bytes()) for p in (self.layout.root / "records").rglob("*.gz"))
        for gone in (b"7700 900123", b"sk-ant-api03-", b"Q7Kx@example.org"):
            self.assertNotIn(gone, text)
        unshaped = self.layout.unshaped("students", "dev").read_text(encoding="utf-8")
        self.assertIn("a binary file", unshaped)  # the fitted model: kept raw, and why it is not shaped
        self.assertTrue(everything)

    def test_records_rebuild_byte_for_byte_and_an_unchanged_kind_is_not_shaped_again(self):
        from dclab_rnd.store import shapers

        shapers.shape(self.layout)

        def digest():
            return {p.relative_to(self.layout.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.layout.root / "records").rglob("*.gz")}

        first = digest()
        again = shapers.shape(self.layout)
        self.assertIn("sessions", again["unchanged"])
        shapers.shape(self.layout, force=True)
        self.assertEqual(digest(), first)


    def test_the_review_cases_never_leak_and_every_record_keeps_its_links(self):
        import gzip as gz
        from dclab_rnd.store import shapers

        at = "2026-10-20T10:00:00+00:00"
        run = gz.compress((json.dumps({"kind": "unit", "output": "env key sk-ant-api03-" + "Q" * 30}) + "\n").encode())
        self.store.keep("tests", "rnd", "test_runs/2026-10-20/r1.jsonl.gz", run)  # a gzip source: opened before the credentials are removed
        transcript = [{"type": "user", "sessionId": "s-1", "timestamp": at, "uuid": "u1", "toolUseResult": {"stdout": "Author: Jane Roe <jane.roe@example.org>\n+44 7700 900123"}},
                      {"type": "user", "sessionId": "s-1", "timestamp": at, "uuid": "u2",
                       "toolUseResult": {"stdout": "email,amount\nQ7Kx@example.org,10\nZ9@example.org,12\nY8@example.org,9"}}]
        raw = "".join(json.dumps(t) + "\n" for t in transcript) + "not json at all Q7Kx@example.org\n"
        self.store.keep_log("agents", "rnd", "claude_code/main/s1.jsonl", raw.encode())
        db_decision = {"id": "dgdb0000000000000000001", "project_id": "p7", "doc": {"schema_version": "dclab_core.decision_record.v2", "point": "column.time",
                                                                                  "at": at, "state": {}, "answers": [], "used": "rule", "value_used": False}}
        db_label = {"id": "dldb1", "record_id": "dgdb0000000000000000001", "doc": {"schema_version": "dclab_core.decision_label.v2", "point": "column.time",
                                                                               "value": False, "source": "benchmark", "at": at}}
        self.store.keep_log("decisions", "pg", "db/decision_label.jsonl", (json.dumps(db_label) + "\n").encode())  # sorts before its decision
        self.store.keep_log("decisions", "pg", "db/decision_record.jsonl", (json.dumps(db_decision) + "\n").encode())
        on_draft = {"schema_version": "dclab_core.decision_record.v2", "id": "dgdraft00000000000000001", "point": "column.time", "at": at,
                    "draft": "d1", "state": {}, "answers": [], "used": "rule", "value_used": False, "ids": {"request": "3dbb964f0ff6"},
                    "context": {"project": {"id": "p1"}}}
        self.store.keep("conversations", "dev", "drafts/d1/draft.json", json.dumps({"id": "d1", "project_id": "p1"}).encode())
        self.store.keep_log("decisions", "dev", "ledger/decisions/2026/10/21.jsonl", (json.dumps({"project": {"id": "p1"}, "x": 1}) + "\n" + json.dumps(on_draft) + "\n").encode())
        self.store.keep("deletions", "dev", "deletions/session-s-1.json", json.dumps({"what": "session", "id": "s-1"}).encode())
        shapers.shape(self.layout)
        text = b"".join(gzip.decompress(p.read_bytes()) for p in (self.layout.root / "records").rglob("*.gz"))
        for gone in (b"sk-ant-api03-", b"jane.roe@example.org", b"7700 900123", b"Q7Kx@example.org", b"Z9@example.org", b"not json at all"):
            self.assertNotIn(gone, text, gone)
        agents = list(shapers.records(self.layout, "agents"))
        self.assertEqual({r.body.get("session") for r in agents if r.body.get("session")}, {"s-1"})
        self.assertTrue(all(r.at == at for r in agents if r.body.get("session")))  # when it happened, not when it was kept
        self.assertTrue(all(r.split for r in agents if r.body.get("session")))
        self.assertTrue(all(r.provenance.deleted for r in agents if r.body.get("session")))  # its session was deleted
        decisions = {r.body.get("id"): r for r in shapers.records(self.layout, "decisions") if r.body.get("id")}
        self.assertEqual(decisions["dgdb0000000000000000001"].type, "decision")
        label = next(r for r in shapers.records(self.layout, "decisions") if r.body.get("id") == "dldb1")
        self.assertEqual((label.type, label.body["group"]), ("label", "project:p7"))
        self.assertEqual(label.split, decisions["dgdb0000000000000000001"].split)
        drafted = decisions["dgdraft00000000000000001"]
        self.assertEqual(drafted.body["group"], "project:p1")  # a draft's decision sealed with the project it became
        self.assertEqual(drafted.body["ids"]["request"], "3dbb964f0ff6")  # an id left as it is

    def test_a_new_shaper_version_rebuilds_and_removes_stale_files(self):
        from dclab_rnd.store import shapers

        shapers.shape(self.layout)
        stray = self.layout.records_of("sessions") / "dev" / "1999-01.jsonl.gz"
        stray.write_bytes(gzip.compress(b"{}\n"))
        state = self.layout.shaped_state("sessions")
        doc = json.loads(state.read_text(encoding="utf-8"))
        doc["version"] = "shapers-0"
        state.write_text(json.dumps(doc), encoding="utf-8")
        shapers.shape(self.layout)
        self.assertFalse(stray.exists())  # rebuilt beside the old and swapped in: nothing stale survives


if __name__ == "__main__":
    unittest.main()

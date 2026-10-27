"""35.2: the memory store (dclab_rnd.memory): the same calls on the workspace file and on PostgreSQL, notes and
lessons read where they live, a deleted project's items retired and kept."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dclab_core import memory as m  # noqa: E402
from dclab_rnd import memory as store_mod  # noqa: E402
from dclab_rnd.memory import readers  # noqa: E402

AT = "2026-10-28T09:00:00+00:00"
CHAIN = [("global", "dclab"), ("workspace", "main"), ("person", "owner"), ("project", "p1")]


def item(text, scope="project", scope_id="p1", kind="fact", **kw):
    return m.new(kind=kind, scope=scope, scope_id=scope_id, text=text, source="test:1", who=kw.pop("who", "person:owner"), at=kw.pop("at", AT), **kw)


class Behaviour:
    """The calls every store answers alike."""

    def make(self):  # pragma: no cover - each store's test case
        raise NotImplementedError

    def test_written_versions_come_back_in_order_and_the_last_is_the_state(self):
        s = self.make()
        a = s.write(item("Project fact a."))
        s.write(m.counted(a, shown=2))
        b = s.write(item("Project fact b."))
        self.assertEqual([v.id for v in s.log()], [a.id, a.id, b.id])
        state = {i.id: i for i in store_mod.latest(s.log())}
        self.assertEqual(state[a.id].counters.shown, 2)

    def test_retirement_is_final(self):
        s = self.make()
        a = s.write(item("Project fact to retire."))
        s.write(m.retire(a))
        s.write(a)  # an old copy written again never brings it back
        self.assertEqual(store_mod.latest(s.log())[0].status, "retired")
        self.assertEqual(store_mod.view(s, CHAIN), [])

    def test_a_scope_retired_and_its_versions_kept(self):
        s = self.make()
        mine, other = s.write(item("Project fact one.")), s.write(item("Another project's fact.", scope_id="p2"))
        self.assertEqual(store_mod.retire_scope(s, "project", "p1"), 1)
        self.assertEqual(store_mod.retire_scope(s, "project", "p1"), 0)  # twice is once
        state = {i.id: i.status for i in store_mod.latest(s.log())}
        self.assertEqual(state, {mine.id: "retired", other.id: "active"})
        self.assertEqual(len(s.log()), 3)  # nothing deleted

    def test_retired_words_are_refused_not_stored_unseen(self):
        s = self.make()
        a = store_mod.remember(s, item("Answer in simple words."))
        s.write(m.retire(a))
        with self.assertRaises(store_mod.Refused):
            store_mod.remember(s, item("Answer in simple words."))

    def test_a_protected_name_is_refused_at_the_door(self):
        s = self.make()
        with self.assertRaises(store_mod.Refused):
            store_mod.remember(s, item("Group the folds by order_ids.", scope="global", scope_id="dclab", kind="know_how"), ["order_id"])
        self.assertEqual(s.log(), [])


class FileStoreTest(Behaviour, unittest.TestCase):
    def make(self):
        return store_mod.FileMemory(Path(tempfile.mkdtemp()))

    def test_a_damaged_line_is_skipped_never_fatal(self):
        s = self.make()
        s.write(item("Project fact before the crash."))
        with s.path.open("a", encoding="utf-8") as handle:  # a crash cut the line: no line break after it
            handle.write('{"schema_version": "dclab_core.memory_item.v1", "id": "M-0')
        s.write(item("Project fact after the crash."))
        self.assertEqual([i.text for i in s.log()], ["Project fact before the crash.", "Project fact after the crash."])
        self.assertEqual(s.unreadable, 1)

    def test_reading_or_deleting_never_makes_a_folder(self):
        from dclab_rnd.experience import retention

        home = Path(tempfile.mkdtemp())
        self.assertEqual(store_mod.FileMemory(home).log(), [])
        retention.forget(project="pX", log=SimpleNamespace(home=home), ledger=None)
        self.assertFalse((home / "memory").exists())


class PgStoreTest(Behaviour, unittest.TestCase):
    def setUp(self):
        import pgtest

        self.url = pgtest.require()
        pgtest.empty(self.url)
        from dclab_rnd.storage import db

        self.home = Path(tempfile.mkdtemp())
        self.wid = db.workspace_for(self.home, self.url)

    def make(self):
        return store_mod.PgMemory(self.wid, self.url)

    def test_the_table_refuses_an_edit_or_a_delete(self):
        import sqlalchemy as sa

        s = self.make()
        s.write(item("Project fact kept forever."))
        for statement in ("update memory_item_version set status = 'active'", "delete from memory_item_version"):
            with self.assertRaises(Exception), s.engine.begin() as c:
                c.execute(sa.text(statement))
        self.assertEqual(len(s.log()), 1)

    def test_another_workspace_sees_nothing(self):
        from dclab_rnd.storage import db

        self.make().write(item("Project fact of the first workspace."))
        other = store_mod.PgMemory(db.workspace_for(Path(tempfile.mkdtemp()), self.url), self.url)
        self.assertEqual(other.log(), [])


class ReadersTest(unittest.TestCase):
    def project(self):
        return {"id": "p1", "memory": [
            {"id": "n1", "at": "2026-10-20T08:00:00+00:00", "kind": "decision", "decision": "Target is churn within 30 days", "reason": "the owner said so",
             "who": "human", "move": "save_solution", "removed": None},
            {"id": "n2", "at": "2026-10-21T08:00:00+00:00", "kind": "correction", "decision": "Metric is recall at the budget", "reason": "not accuracy",
             "who": "human", "move": "save_solution", "removed": None},
            {"id": "n3", "at": "2026-10-22T08:00:00+00:00", "kind": "correction", "decision": "Split by time", "reason": "drift",
             "who": "agent", "move": "approve_stage", "removed": None},
            {"id": "n4", "at": "2026-10-23T08:00:00+00:00", "kind": "decision", "decision": "Old note", "reason": "removed later",
             "who": "human", "move": "save_solution", "removed": {"by": "human"}},
            {"id": "n5", "at": "2026-10-24T08:00:00+00:00", "kind": "decision", "decision": "Line one\nline\u2028two", "reason": "",
             "who": "human", "move": "save_solution", "removed": None}]}

    def test_notes_are_facts_of_their_project_read_the_same_every_time(self):
        items = readers.from_project(self.project())
        self.assertEqual([i.id for i in items], [i.id for i in readers.from_project(self.project())])
        by_source = {i.source.split(":")[-1]: i for i in items}
        self.assertEqual(by_source["n1"].text, "Target is churn within 30 days. the owner said so")
        self.assertTrue(by_source["n2"].correction)
        self.assertFalse(by_source["n3"].correction)  # an agent's note is never a person's correction
        self.assertEqual(by_source["n3"].who, "agent:agent")
        self.assertNotIn("n4", by_source)  # a removed note is not read
        self.assertEqual(by_source["n5"].text, "Line one line two")
        shown = store_mod.view(None, CHAIN, items)
        self.assertEqual(shown[0].id, by_source["n2"].id)  # the person's correction first

    def test_words_removed_then_written_again_show(self):
        project = self.project()
        project["memory"].append({**project["memory"][3], "id": "n6", "removed": None, "at": "2026-10-25T08:00:00+00:00"})
        again = [i for i in readers.from_project(project) if i.source.endswith(":n6")][0]
        s = store_mod.FileMemory(Path(tempfile.mkdtemp()))
        s.write(m.retire(again))  # even a retired stored copy of the same words never hides the live note
        self.assertIn(again.id, [i.id for i in store_mod.view(s, CHAIN, readers.from_project(project))])

    def test_an_author_named_like_a_user_is_not_a_person(self):
        self.assertEqual(readers.author("username"), "agent:username")
        self.assertEqual(readers.author("human"), "person:human")

    def test_only_accepted_lessons_are_know_how(self):
        lessons = [{"id": "L1", "status": "accepted", "claim": "Group the folds by the repeated entity.", "updated": AT, "review": {"by": "reviewer"}},
                   {"id": "L2", "status": "proposed", "claim": "A proposal not yet reviewed.", "updated": AT}]
        got = readers.from_lessons(lessons, "main")
        self.assertEqual([(i.kind, i.scope, i.scope_id, i.who) for i in got], [("know_how", "workspace", "main", "person:reviewer")])

    def test_a_long_or_empty_text(self):
        self.assertEqual(len(readers.line("word " * 200)), m.TEXT)
        self.assertEqual(readers.line(" ​\n "), "")


class DeletionTest(unittest.TestCase):
    def test_deleting_a_project_retires_its_items_and_keeps_them(self):
        from dclab_rnd.experience import retention

        home = Path(tempfile.mkdtemp())
        s = store_mod.open_memory(home)  # the workspace's own store: the file, or the database when the suite runs on PostgreSQL
        a = s.write(item("A fact of the project being deleted."))
        s.write(item("A fact of another project.", scope_id="p2"))
        out = retention.forget(project="p1", log=SimpleNamespace(home=home), ledger=None)
        self.assertEqual(out["memory"], 1)
        state = {i.id: i.status for i in store_mod.latest(store_mod.open_memory(home).log())}
        self.assertEqual(state[a.id], "retired")
        self.assertEqual(len(store_mod.open_memory(home).log()), 3)


if __name__ == "__main__":
    unittest.main()

"""35.6: memory upkeep. Near-duplicates merge into the newest, a line shown many times and never cited retires, a line
marked harmful twice retires, two lines that disagree are listed for a person and never settled by code; every change
is an event, and nothing is deleted."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import memory as m  # noqa: E402
from dclab_rnd import experience  # noqa: E402
from dclab_rnd import memory as store_mod  # noqa: E402
from dclab_rnd.memory import upkeep as up  # noqa: E402
from dclab_rnd.memory import writing as w  # noqa: E402


class UpkeepTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = store_mod.open_memory(self.home)
        self.log = experience.open_log(self.home)
        env = mock.patch.dict(os.environ, {"DCLAB_MEMORY": ""})
        env.start()
        self.addCleanup(env.stop)

    def fact(self, text, at, scope_id="p1", who="person:owner"):
        return w.record_fact(self.store, scope="project", scope_id=scope_id, text=text, source="test:1", who=who, at=at)

    def state(self):
        return {i.id: i for i in store_mod.latest(self.store.log())}

    def test_near_duplicates_merge_into_the_newest(self):
        old = self.fact("The target is churn within 30 days of the snapshot.", "2026-10-01T00:00:00+00:00")
        new = self.fact("The target is the churn within 30 days of the snapshot", "2026-10-20T00:00:00+00:00")
        plan = up.plan(self.store, self.log)
        self.assertEqual([(d["item"], d["into"]) for d in plan["merge"]], [(old.id, new.id)])  # only filler words differ
        up.apply(self.store, plan, self.log)
        self.assertEqual((self.state()[old.id].status, self.state()[old.id].retired_by), ("retired", new.id))
        self.assertEqual(self.state()[new.id].status, "active")

    def test_a_line_shown_many_times_and_never_cited_retires(self):
        stale = self.fact("An old note nobody uses.", "2026-10-01T00:00:00+00:00", who="agent:home")
        used = self.fact("A note that answers.", "2026-10-01T00:00:00+00:00", who="agent:home")
        for _ in range(up.RETIRE_AFTER):
            experience.record("memory.shown", "agent", {"items": [stale.id, used.id], "shown": 2, "left_out": 0, "tokens": 10, "budget": 800}, log=self.log)
        experience.record("memory.cited", "agent", {"items": [used.id], "cited": 1}, log=self.log)
        plan = up.plan(self.store, self.log)
        self.assertEqual([d["item"] for d in plan["retire"]], [stale.id])
        up.apply(self.store, plan, self.log)
        self.assertEqual(self.state()[stale.id].status, "retired")
        self.assertEqual(self.state()[used.id].status, "active")

    def test_a_line_marked_harmful_twice_retires(self):
        bad = self.fact("A misleading note.", "2026-10-01T00:00:00+00:00")
        self.store.write(m.counted(bad, harmful=2))
        self.assertEqual([d["item"] for d in up.plan(self.store, self.log)["retire"]], [bad.id])

    def test_two_lines_that_disagree_are_listed_never_settled(self):
        a = self.fact("Calls are capped at 200 a month.", "2026-10-01T00:00:00+00:00")
        b = self.fact("Calls are capped at 150 a month.", "2026-10-05T00:00:00+00:00")
        plan = up.plan(self.store, self.log)
        self.assertEqual([sorted(c["items"]) for c in plan["conflicts"]], [sorted([a.id, b.id])])
        self.assertEqual(plan["merge"], [])
        up.apply(self.store, plan, self.log)
        self.assertEqual((self.state()[a.id].status, self.state()[b.id].status), ("active", "active"))

    def test_nothing_is_deleted_and_every_change_is_an_event(self):
        self.fact("The target is churn within 30 days of the snapshot.", "2026-10-01T00:00:00+00:00")
        self.fact("The target is the churn within 30 days of the snapshot", "2026-10-20T00:00:00+00:00")
        before = len(self.store.log())
        up.apply(self.store, up.plan(self.store, self.log), self.log)
        self.assertEqual(len(self.store.log()), before + 1)  # one retirement version added, nothing removed
        self.assertEqual([e.payload["reason"] for e in self.log.read() if e.kind == "memory.retired"], ["merged"])

    def test_lines_in_different_scopes_never_merge(self):
        self.fact("The target is churn within 30 days of the snapshot.", "2026-10-01T00:00:00+00:00", scope_id="p1")
        self.fact("The target is the churn within 30 days of the snapshot", "2026-10-20T00:00:00+00:00", scope_id="p2")
        self.assertEqual(up.plan(self.store, self.log)["merge"], [])


    def test_lines_that_disagree_are_conflicts_never_merged(self):
        pairs = [("Use the time split for this project and group the folds by customer.", "Never use the time split for this project and group the folds by customer."),
                 ("Churn is counted per account, not per customer.", "Churn is counted per customer, not per account."),
                 ("The folds are grouped by customer_id when splitting the data.", "The folds are grouped by account_id when splitting the data."),
                 ("کاربر می‌خواهد پاسخ کوتاه باشد", "کاربر نمی‌خواهد پاسخ کوتاه باشد")]
        for n, (x, y) in enumerate(pairs):
            a, b = self.fact(x, "2026-10-01T00:00:00+00:00", scope_id=f"q{n}"), self.fact(y, "2026-10-02T00:00:00+00:00", scope_id=f"q{n}")
            plan = up.plan(self.store, self.log)
            self.assertFalse([d for d in plan["merge"] if d["item"] in (a.id, b.id)], x)
            self.assertTrue([c for c in plan["conflicts"] if set(c["items"]) == {a.id, b.id}], x)

    def test_numbers_are_compared_as_numbers(self):
        a = self.fact("Calls are capped at 1,000 a month.", "2026-10-01T00:00:00+00:00", scope_id="n1")
        b = self.fact("Calls are capped at 1000 a month", "2026-10-02T00:00:00+00:00", scope_id="n1")
        self.assertEqual([(d["item"], d["into"]) for d in up.plan(self.store, self.log)["merge"]], [(a.id, b.id)])

    def test_a_persons_lines_and_corrections_are_never_retired_or_merged_away(self):
        pref = w.record_fact(self.store, scope="person", scope_id="owner", text="Answer in simple words.", source="t:1", who="person:owner",
                             at="2026-10-01T00:00:00+00:00")
        again = w.record_fact(self.store, scope="person", scope_id="owner", text="Answer in simple words", source="t:2", who="agent:home",
                              at="2026-10-02T00:00:00+00:00")
        self.assertEqual((again.id, again.who), (pref.id, "person:owner"))  # the same words are the person's line, kept as it is
        old = self.fact("Calls are capped at 200 a month.", "2026-10-01T00:00:00+00:00")
        fixed = w.correct(self.store, old, "Calls are capped at 150 a month.", by="owner", at="2026-10-03T00:00:00+00:00")
        for _ in range(up.RETIRE_AFTER):
            experience.record("memory.shown", "agent", {"items": [pref.id, fixed.id], "shown": 2, "left_out": 0, "tokens": 9, "budget": 800}, log=self.log)
        plan = up.plan(self.store, self.log)
        touched = {d["item"] for d in plan["merge"]} | {d["item"] for d in plan["retire"]}
        self.assertNotIn(pref.id, touched)
        self.assertNotIn(fixed.id, touched)

    def test_usage_reads_every_memory_event(self):
        stale = self.fact("An agent note nobody cites.", "2026-10-01T00:00:00+00:00", who="agent:home")
        experience.record("memory.cited", "agent", {"items": [stale.id], "cited": 1}, log=self.log)
        for n in range(60):
            experience.record("chat.agent", "agent", {"text": f"turn {n}"}, log=self.log)
        for _ in range(up.RETIRE_AFTER):
            experience.record("memory.shown", "agent", {"items": [stale.id], "shown": 1, "left_out": 0, "tokens": 5, "budget": 800}, log=self.log)
        self.assertEqual(up.plan(self.store, self.log)["retire"], [])  # cited once: never retired as never cited

    def test_the_command_runs(self):
        from dclab_rnd.memory.__main__ import main

        self.assertEqual(main(["upkeep", "--workspace", str(self.home)]), 0)


if __name__ == "__main__":
    unittest.main()

"""Package 30.22: department agents (the owner's pattern 8).

A proposal past a limit is never applied, whatever the level; one above the approval amount waits for a person; a run never spends past
its budget; an applied step is undone exactly; a level is earned after its run of right calls and lost on a wrong one; a stale reading is
never applied; and every step is kept whole beside the record.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401
except ImportError as error:  # .venv-agent
    MISSING = str(error)
else:
    MISSING = ""


def policy(**over):
    from dclab_core import department_agent as da

    return da.Policy(**{"name": "pricing", "actions": ["set_price"], "field": "price", "max_change": 0.25, "budget": 100.0, "approval_above": 30.0,
                        "level": "act", "promote_after": 3, **over})


def proposal(before=10.0, after=9.0, impact=0.0, action="set_price", target="A"):
    from dclab_core import department_agent as da

    return da.Proposal(action=action, target=target, before=before, after=after, impact=impact)


def agent(pol, state, units=None):
    """An agent whose impact the host reckons: (before - after) x the target's units (5 a target unless the test says)."""
    from dclab_core import department_agent as da

    units = units or {}
    return da.Agent(pol, state, impact_of=lambda target, before, after: (before - after) * units.get(target, 5.0))


@unittest.skipIf(MISSING, MISSING)
class GuardrailTest(unittest.TestCase):
    def test_past_a_limit_or_an_unknown_action_is_never_applied_at_any_level(self):
        from dclab_core import department_agent as da

        for level in da.LEVELS:
            a = agent(policy(level=level), {"A": 10.0})
            for p in (proposal(after=6.0), proposal(action="delete_product")):
                step = a.consider(p)
                self.assertFalse(step.applied, (level, p.action))
                self.assertFalse(step.verdict.allowed)
            self.assertEqual(a.state["A"], 10.0)

    def test_above_the_approval_amount_waits_and_the_budget_is_never_passed(self):
        from dclab_core import department_agent as da

        a = agent(policy(), {"A": 10.0, "B": 10.0, "C": 10.0, "E": 10.0}, units={"E": 40.0, "A": 29.0, "B": 29.0, "C": 29.0})
        big = a.consider(proposal(target="E"))
        self.assertEqual((big.applied, big.verdict.needs_approval, big.flagged), (False, True, True))  # waiting: flagged to a person
        for target in ("A", "B", "C"):
            a.consider(proposal(target=target))
        self.assertLessEqual(a.spent, 100.0)
        self.assertEqual(sum(s.applied for s in a.steps), 3)
        fourth = agent(policy(), {"D": 10.0}, units={"D": 20.0})
        fourth.spent = 90.0
        self.assertFalse(fourth.consider(proposal(target="D")).applied)  # past the budget

    def test_a_declared_impact_is_never_what_the_limits_read(self):
        from dclab_core import department_agent as da

        with self.assertRaises(ValueError):  # a budget with no host reckoning
            da.Agent(policy(), {"A": 10.0})
        a = agent(policy(), {"A": 10.0}, units={"A": 40.0})
        step = a.consider(proposal(impact=0.0))  # declared 0; the host reckons 40, above the approval amount
        self.assertEqual((step.applied, step.verdict.needs_approval, step.proposal.impact), (False, True, 40.0))
        self.assertIn("declared impact 0", step.proposal.reason)

    def test_an_applied_step_is_undone_exactly_and_a_stale_reading_is_never_applied(self):
        from dclab_core import department_agent as da

        a = agent(policy(), {"A": 10.0, "B": 12.0})
        step = a.consider(proposal())
        self.assertEqual(a.state["A"], 9.0)
        undone = da.undo(a, step.id)
        self.assertEqual((a.state["A"], a.spent, undone.undone), (10.0, 0.0, True))
        with self.assertRaises(ValueError):
            da.undo(a, step.id)
        stale = a.consider(proposal(target="B", before=10.0, after=9.0))  # the proposal read 10, the value is 12 now
        self.assertFalse(stale.applied)
        self.assertEqual(a.state["B"], 12.0)

    def test_two_steps_on_one_value_are_undone_in_reverse_order_back_to_the_start(self):
        from dclab_core import department_agent as da

        a = agent(policy(), {"A": 10.0}, units={"A": 1.0})
        first, second = a.consider(proposal(before=10.0, after=9.0)), a.consider(proposal(before=9.0, after=8.0))
        with self.assertRaises(ValueError):  # the first step's value was changed by the second
            da.undo(a, first.id)
        da.undo(a, second.id)
        da.undo(a, first.id)
        self.assertEqual((a.state["A"], round(a.spent, 9)), (10.0, 0.0))

    def test_a_level_is_earned_by_a_run_of_right_calls_and_lost_on_a_wrong_one(self):
        from dclab_core import department_agent as da

        p = policy(level="shadow", promote_after=3)
        record = da.TrackRecord()
        for _ in range(2):
            record = da.judge(record, True)
        self.assertEqual(da.level_for(p, record, True)[0], "shadow")
        record = da.judge(record, True)
        level, kept = da.level_for(p, record, True)
        self.assertEqual((level, kept.right_in_a_row), ("flag", 0))  # the new level earns its own run
        self.assertEqual(da.level_for(p.model_copy(update={"level": "flag"}), kept, True)[0], "flag")  # never two levels on one run
        self.assertEqual(da.level_for(policy(level="act"), da.judge(record, False), False)[0], "act_with_check")
        self.assertEqual(da.level_for(policy(level="shadow"), da.judge(record, False), False)[0], "shadow")
        self.assertEqual(da.Policy.read(p.model_dump_json()), p)


@unittest.skipIf(MISSING, MISSING)
class PricingReplicaTest(unittest.TestCase):
    def test_the_careful_agent_earns_its_levels_the_aggressive_never_and_nothing_past_a_limit_is_applied(self):
        from dclab_rnd import department

        doc = department.run(0)
        careful, aggressive = doc["agents"]["careful"], doc["agents"]["aggressive"]
        self.assertEqual([m["to"] for m in careful["levels"]], ["flag", "act_with_check", "act"])
        self.assertEqual(aggressive["final_level"], "shadow")
        for a in (careful, aggressive):
            self.assertEqual((a["refused_applied"], a["approval_applied"]), (0, 0))
            self.assertLessEqual(a["spent"], a["budget"])
        with tempfile.TemporaryDirectory() as tmp:
            path = department.write(doc, campaign=Path(tmp) / "department_v1")
            record = json.loads(path.read_text(encoding="utf-8"))
            import gzip

            with gzip.open(path.parent.parent / record["detail"], "rt", encoding="utf-8") as fh:
                kept = [json.loads(line) for line in fh]
            self.assertEqual(len(kept), careful["proposals"] + aggressive["proposals"])  # every step, whole
            self.assertNotIn("steps", record)


    def test_every_step_is_in_the_ledger_with_the_persons_judgment_and_in_the_audit(self):
        from dclab_rnd import audit, department, ledger

        with tempfile.TemporaryDirectory() as tmp:
            book = ledger.open_ledger(Path(tmp), "wtest")
            ledger.install(book)
            self.addCleanup(ledger.install, None)
            log = audit.open_audit(Path(tmp))
            doc = department.run(0, promote_after=50, audit_log=log)
            records = book.records(point="department.proposal")
            self.assertEqual(len(records), doc["agents"]["careful"]["proposals"] + doc["agents"]["aggressive"]["proposals"])
            labels = book.labels(point="department.proposal")
            self.assertEqual(len(labels), len(records))  # the track record is in the ledger: one judgment per proposal
            judged = [s for s in doc["steps"] if s["agent"] == "aggressive"]
            self.assertEqual(sum(not lab.value for lab in labels), sum(s["judged"] is False for s in doc["steps"]))
            self.assertTrue(judged)
            total, rows = log.query(10_000, 0, kind="department")
            self.assertEqual(total, len(records))
            self.assertEqual(sum(r.get("status") == "applied" for r in rows), doc["agents"]["careful"]["applied"])


if __name__ == "__main__":
    unittest.main()

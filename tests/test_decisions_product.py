"""34.4, decisions in the product: a project states an ``act`` decision (dclab_core's DecisionSpec) or a cost matrix
over its classes, and the engine's final stage chooses it on the selected model's out-of-fold scores, reads it once on
the holdout, writes its sentence, and keeps the rows a what-if re-scores beside the project (never in a record).

The churn replica acts on a share of customers with a value per customer from a column known at decision time; a
column written after the outcome is refused as a value; the ticket replica's cost matrix costs less on the holdout
than the likeliest class when one miss is dear; a what-if on the saved rows equals the core's answer.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""


def finish(case):
    from dclab_rnd.expansion import runner
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    root = Path(tempfile.mkdtemp())
    store, pid = acceptance.build(case, root)
    with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        engine.run_all(store, pid)
    return store, pid, store.read_stage(pid, "final")


def churn(decision):
    case = kinds.get("binary").case(0)
    case.solution = {**case.solution, "decision": decision}
    return case


@unittest.skipIf(MISSING, MISSING)
class ActDecisionTest(unittest.TestCase):
    def test_an_act_decision_is_chosen_read_once_said_and_its_rows_kept_for_what_if(self):
        from dclab_core import decide
        from dclab_rnd.studio import engine

        store, pid, final = finish(churn({"action": "act", "what": "call the customer", "value_column": "monthly_fee",
                                          "cost_per_action": 4, "capacity_share": 0.3}))
        d = final["evidence"]["decision"]
        self.assertEqual(d["decision"], "operating_point", d.get("warning"))
        self.assertEqual(d["ranked_by"], "expected gain")  # a value per customer, probabilities: ranked by what each is worth
        self.assertLessEqual(d["share"], 0.3)
        self.assertIn("holdout", d)
        self.assertEqual({r["term"] for r in d["sensitivity"]["rows"]}, {"cost_per_action", "capacity_share", "value_scale", "base_rate"})
        claim = next(c for c in final["claims"] if c["claim_id"].endswith("-D1"))
        self.assertIn("call the customer", claim["statement"])
        rows = engine.load_decision_rows(store, pid, final["run"]["run"])
        self.assertIsNotNone(rows)
        self.assertEqual(len(rows["y"]), len(rows["score"]))
        self.assertLessEqual(len(rows["y"]), final["evidence"]["train_rows"])  # a time split leaves the earliest rows without one
        spec = decide.DecisionSpec(**d["spec"])
        again = decide.choose_operating_point(rows["y"], rows["score"], spec, value_per_case=rows["value"], bootstrap=0)
        self.assertEqual((again["share"], again["threshold"]), (d["share"], d["threshold"]))  # the saved rows are the chosen-on rows
        self.assertIsNone(engine.load_decision_rows(store, pid, "another-run"))  # rows of another run are never read
        text = (Path(store.directory(pid)) / "stages").exists() and "".join(p.read_text() for p in (Path(store.directory(pid)) / "stages").glob("*.json"))
        self.assertNotIn("monthly_fee\": [", text or "")  # no column's values in a record
        from dclab_core import report as core_report
        from dclab_rnd.studio import solution_report

        markdown = core_report.to_markdown(solution_report.build(store, pid, analyse=False))
        self.assertIn("The decision: call the customer on the top", markdown)
        self.assertIn("On the holdout it acted on", markdown)

    def test_the_chosen_share_reaches_the_ledger(self):
        """A bug found by another session's real-table run: the state's ``of`` was no state field, so every
        threshold.operating_point record was refused and lost (fixed in dclab-core 0.112.0)."""
        from dclab_rnd import ledger

        book = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "wtest")
        ledger.install(book)
        try:
            store, pid, final = finish(churn({"action": "act", "what": "call the customer", "value_column": "monthly_fee",
                                              "cost_per_action": 4, "capacity_share": 0.3}))
        finally:
            ledger.install(None)
        points = book.records(point="threshold.operating_point")
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].state.get("of"), "share")
        self.assertEqual(points[0].answers[0].answerer, "rule:decision_core")
        self.assertEqual(points[0].value_used, final["evidence"]["decision"]["share"])

    def test_a_capacity_per_day_is_read_day_by_day_on_the_holdout(self):
        """0.115.0: the churn table dates each customer (about 4 a day, from 1 to 11) and is split by time; 1 call a day is
        a share of the training period's rate (its rows hold their days whole), and the holdout, which holds its days
        whole, is called day by day, never more than a day's slots."""
        import pandas as pd

        from dclab_rnd.expansion.datasets import HOLDOUT_FRACTION

        stamps = pd.to_datetime(kinds.get("binary").case(0).table["signup_time"]).sort_values()
        training = stamps.iloc[: len(stamps) - int(round(HOLDOUT_FRACTION * len(stamps)))]
        rate = len(training) / ((training.max() - training.min()).total_seconds() / 86400)
        store, pid, final = finish(churn({"action": "act", "what": "call the customer", "value_per_case": 120, "cost_per_action": 10,
                                          "capacity_per_day": 1}))
        d = final["evidence"]["decision"]
        self.assertEqual(d["decision"], "operating_point", d.get("warning"))
        self.assertAlmostEqual(d["capacity_share"], 1 / rate, places=3)
        days = d["holdout"]["days"]
        self.assertGreater(days["days"], 5)
        self.assertLessEqual(days["most_in_a_day"], 1)
        self.assertLessEqual(d["holdout"]["actions"], days["slots"])
        self.assertIn("day by day", d["applied_as"])
        self.assertIn("assumes the world stays as it was", d["holdout"]["drift"])  # 0.118.1: the latest fold's interval, said
        self.assertNotIn("refits of the model", d["holdout"]["prediction_interval_method"])  # off by default (time_decision_v5)
        from dclab_core import report as core_report
        from dclab_rnd.studio import solution_report

        markdown = core_report.to_markdown(solution_report.build(store, pid, analyse=False))
        self.assertIn("assumes the world stays as it was", markdown)  # and in the solution report the person reads

    def test_a_holdout_holding_part_of_each_day_is_read_as_one_pool(self):
        """The review of 0.115.0: a time-and-group split's holdout drops the later rows of entities seen in training, so it
        holds only part of each day; ranked among a day's few holdout rows, a row is not the one the team would pick."""
        case = churn({"action": "act", "what": "call the customer", "value_per_case": 120, "cost_per_action": 10, "capacity_per_day": 1})
        case.solution = {**case.solution, "group_column": "subscriber_id"}
        store, pid, final = finish(case)
        d = final["evidence"]["decision"]
        self.assertEqual(d["decision"], "operating_point", d.get("warning"))
        self.assertNotIn("days", d["holdout"])
        self.assertIn("time-group split", d["days_note"])
        self.assertIn("one pool", d["applied_as"])

    def test_a_value_written_after_the_outcome_is_refused(self):
        """The proposal forbids the refund (written after the customer left), so a decision valued by it is refused when the
        solution is written, before any stage runs."""
        with self.assertRaisesRegex(ValueError, "final_refund_amount"):
            finish(churn({"action": "act", "value_column": "final_refund_amount", "cost_per_action": 1}))

    def test_the_solution_checks_an_act_and_a_classes_decision(self):
        from dclab_rnd.studio.solution import Solution

        base = {"target": "churned", "task": "binary", "positive_label": "yes", "prediction_moment": "At the monthly billing run."}
        with self.assertRaises(ValueError):
            Solution(**base, decision={"action": "act", "cost_per_action": 3})  # nothing is worth deciding
        with self.assertRaises(ValueError):
            Solution(**base, decision={"action": "act", "value_column": "monthly_fee"},
                     forbidden=[{"column": "monthly_fee", "reason": "x"}])
        with self.assertRaises(ValueError):
            Solution(**{**base, "task": "regression", "positive_label": None}, decision={"action": "classes", "classes": ["a", "b"],
                                                                                           "cost_matrix": [[0, 1], [1, 0]]})
        with self.assertRaises(ValueError):
            Solution(**base, decision={"action": "classes", "classes": ["no", "yes"], "cost_matrix": [[0, 1]]})
        ok = Solution(**base, decision={"action": "act", "value_per_case": 30, "cost_per_action": 3, "capacity_share": 0.1})
        self.assertEqual(ok.decision.spec().capacity_share, 0.1)
        multi = Solution(target="assigned_team", task="multiclass", prediction_moment="When the ticket arrives, before triage.",
                         decision={"action": "classes", "classes": ["a", "b", "c"], "cost_matrix": [[0, 1, 1], [1, 0, 1], [9, 9, 0]]})
        self.assertEqual(multi.decision.action, "classes")


@unittest.skipIf(MISSING, MISSING)
class DecisionApiTest(unittest.TestCase):
    def test_the_api_shows_the_decision_and_answers_a_what_if_as_the_core_does(self):
        from fastapi.testclient import TestClient

        from dclab_core import decide
        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.expansion import runner
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import engine

        app = create_app(Path(tempfile.mkdtemp()))
        client = TestClient(app)
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        store = app.state.projects
        _, pid = acceptance.build(churn({"action": "act", "what": "call the customer", "value_per_case": 120, "cost_per_action": 10}),
                                  None, store=store)
        self.assertEqual(client.get(f"/api/projects/{pid}/decision", headers=h).json()["status"], "pending")
        self.assertEqual(client.post(f"/api/projects/{pid}/decision/what-if", json={"changes": {"cost_per_action": 20}}, headers=h).status_code, 409)
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        seen = client.get(f"/api/projects/{pid}/decision", headers=h).json()
        self.assertEqual(seen["status"], "chosen")
        self.assertTrue(seen["what_if"]["available"])
        self.assertIn("call the customer", seen["sentence"])
        r = client.post(f"/api/projects/{pid}/decision/what-if", json={"changes": {"cost_per_action": 20}}, headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        rows = engine.load_decision_rows(store, pid, seen["run"])
        core = decide.what_if(rows["y"], rows["score"], decide.DecisionSpec(**seen["chosen"]["spec"]), {"cost_per_action": 20})
        self.assertEqual(out["after"], core["after"])  # one answer: the API's is the core's
        self.assertLessEqual(out["after"]["share"], out["before"]["share"])
        self.assertIn("not refitted", out["note"])
        bad = client.post(f"/api/projects/{pid}/decision/what-if", json={"changes": {"price_of_tea": 3}}, headers=h)
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(client.get(f"/api/projects/{pid}/decision", headers=h).json()["chosen"], seen["chosen"])  # nothing changed


@unittest.skipIf(MISSING, MISSING)
class ReviewFindingsTest(unittest.TestCase):
    """The review of 34.4: amounts added only from the answer that asks for them, and as DCLab's move; a broken rows
    file never fails the page; a warning never covers a requirement; decisions only where the engine draws them."""

    def test_only_the_amounts_answer_is_added_and_by_dclab(self):
        from dclab_rnd.draft.api import _add_decision, _decision_said

        stated, where = _decision_said({"action": "We call the riskiest; a call costs 5 and a lost customer costs 300"})
        self.assertEqual((stated["cost_per_action"], where), (5.0, "action"))  # heard, never added from this answer
        stated, where = _decision_said({"action": "We call the riskiest", "costs": "A call costs 5 euro; a lost customer costs 300."})
        self.assertEqual(where, "costs")
        self.assertEqual(_decision_said({"action": "We call the top 100 customers a week, a lost one costs 300"}), (None, None))
        store, pid = acceptance.build(churn(None), Path(tempfile.mkdtemp()))
        _add_decision(store, pid, stated)
        project = store.get(pid)
        self.assertEqual(project["solution"]["decision"]["what"], "call")
        self.assertEqual(project["solution_saved_by"], "agent")  # the person saved the solution; DCLab added the decision
        self.assertTrue(any(e.get("kind") == "decision_added" for e in store.activity(pid, limit=50)))

    def test_a_broken_rows_file_gives_no_what_if_and_no_error(self):
        from dclab_rnd.studio import decision as studio_decision
        from dclab_rnd.studio import engine

        store, pid = acceptance.build(churn({"action": "act", "value_per_case": 120, "cost_per_action": 6}), Path(tempfile.mkdtemp()))
        (Path(store.export_dir(pid)) / engine.DECISION_ROWS).write_bytes(b"PK\x03\x04 cut short")
        self.assertIsNone(engine.load_decision_rows(store, pid, "any-run"))
        self.assertEqual(studio_decision.view(store, pid)["status"], "pending")

    def test_a_decision_not_drawn_never_covers_a_requirement(self):
        from dclab_core.report import _where

        ask = "Choose the alert threshold from the cost of a missed churner"
        self.assertNotEqual(_where(ask, {"final.evidence.decision.warning"})[1], "covered")
        self.assertEqual(_where(ask, {"final.evidence.decision.share"})[1], "covered")

    def test_decisions_only_where_the_engine_draws_them(self):
        from dclab_rnd.studio.solution import Solution

        base = {"target": "y", "task": "binary", "positive_label": "1", "prediction_moment": "At the start of each month."}
        uplift = {"kind": "uplift", "kind_fields": {"treatment_column": "got_offer", "randomised": True}}
        Solution(**base, **uplift)
        with self.assertRaisesRegex(ValueError, "own machine"):
            Solution(**base, **uplift, decision={"action": "act", "value_per_case": 40, "cost_per_action": 2})
        with self.assertRaises(ValueError):  # a binary cost matrix over names that are not the outcome's
            Solution(**base, decision={"action": "classes", "classes": ["no", "maybe"], "cost_matrix": [[0, 1], [5, 0]]})
        Solution(**{**base, "positive_label": "yes"}, decision={"action": "classes", "classes": ["no", "yes"], "cost_matrix": [[0, 1], [5, 0]]})
        with self.assertRaises(ValueError):
            Solution(**base, decision={"action": "act", "what": "x" * 81, "value_per_case": 1})

    def test_a_value_column_that_is_not_a_number_is_refused_by_the_engine(self):
        _, _, final = finish(churn({"action": "act", "value_column": "plan", "cost_per_action": 1}))
        d = final["evidence"]["decision"]
        self.assertFalse(d["learnable"])
        self.assertIn("non-numeric", d["warning"])

    def test_the_intern_asks_the_same_what_if(self):
        from dclab_rnd.agents import default_registry
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import decision as studio_decision

        store, pid, _ = finish(churn({"action": "act", "what": "call the customer", "value_per_case": 120, "cost_per_action": 6}))
        box = Toolbox(store)
        seen = default_registry().call("decision_what_if", {"project_id": pid}, scope="project", context=box)
        self.assertEqual(seen["status"], "chosen")
        asked = default_registry().call("decision_what_if", {"project_id": pid, "changes": {"cost_per_action": 30}}, scope="project", context=box)
        self.assertEqual(asked["after"], studio_decision.what_if(store, pid, {"cost_per_action": 30})["after"])
        refused = default_registry().call("decision_what_if", {"project_id": pid, "changes": {"value_scale": 2}}, scope="project", context=box)
        self.assertIn("error", refused)  # each row has no own value here: value_scale is not this decision's number


    def test_amounts_home_heard_but_did_not_add_are_said_by_the_agents_tools(self):
        """A chat building from the person's words ("a call costs 5, a lost customer 300") gets a project whose solution has no
        decision: the amounts stay a suggestion until the person adds them. The tools an agent reads first say so, and why
        no what-if can be asked, so the chat can put it to the person instead of leaving the number out."""
        from dclab_rnd.agents import default_registry
        from dclab_rnd.intern.tools import Toolbox

        store, pid, _ = finish(kinds.get("binary").case(0))
        project = store.get(pid)
        project["suggestion"] = {**(project.get("suggestion") or {}), "decision": {"action": "act", "cost_false_positive": 5.0, "cost_false_negative": 300.0, "heard_in": "action"}}
        store.save(project)
        box = Toolbox(store)
        results = default_registry().call("get_results", {"project_id": pid}, scope="project", context=box)
        self.assertEqual(results["decision"]["status"], "none")
        self.assertEqual(results["decision"]["heard"]["cost_false_negative"], 300.0)
        self.assertIn("states no decision", results["decision"]["why_no_what_if"])
        self.assertIn("the person adds", results["decision"]["note"])
        seen = default_registry().call("decision_what_if", {"project_id": pid}, scope="project", context=box)
        self.assertEqual(seen["heard"]["cost_false_positive"], 5.0)
        self.assertIn("final", results, "the stage records are still there")


@unittest.skipIf(MISSING, MISSING)
class CostMatrixDecisionTest(unittest.TestCase):
    def test_a_dear_miss_makes_the_cost_matrix_cheaper_than_the_likeliest_class(self):
        case = kinds.get("multiclass").case(0)
        classes = sorted(str(c) for c in case.table[case.target].dropna().unique())
        cost = (1 - np.eye(len(classes))) * 1.0
        cost[classes.index("billing"), :] = 12.0  # missing a billing ticket is dear
        cost[classes.index("billing"), classes.index("billing")] = 0.0
        case.solution = {**case.solution, "decision": {"action": "classes", "classes": classes, "cost_matrix": cost.tolist()}}
        store, pid, final = finish(case)
        d = final["evidence"]["decision"]
        self.assertEqual(d["decision"], "cost_matrix", d.get("warning"))
        from dclab_core import report as core_report
        from dclab_rnd.studio import solution_report

        self.assertIn("lowest expected cost", core_report.to_markdown(solution_report.build(store, pid, analyse=False)))
        self.assertLessEqual(d["holdout"]["cost_per_row"], d["holdout"]["cost_per_row_likeliest_class"])
        self.assertEqual(sum(d["chosen_counts"].values()), d["holdout"]["rows"])


if __name__ == "__main__":
    unittest.main()

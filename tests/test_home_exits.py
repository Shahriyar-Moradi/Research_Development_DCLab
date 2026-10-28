"""Where each kind of help leads, the other route a request's words may mean, and an analysis made a prediction safely
(dclab_rnd/draft/exits.py, package H8 of the Home plan)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.agent_eval.home import tables  # noqa: E402
from dclab_rnd.draft import exits, pipeline  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

TRAP_1 = "Show me which customers are going to buy again next month."  # H1 d-p5: worded like a report, a prediction
TRAP_2 = "Build a model of last month's sales by region."  # H1 d-b4: says model, a report of the past


class RulesTests(unittest.TestCase):
    def test_one_exit_per_track_said_as_built_or_not(self):
        self.assertEqual({t: exits.exit_for(t)["kind"] for t in exits.EXITS}, exits.EXITS)
        self.assertEqual(exits.exit_for("analysis")["status"], "open")
        self.assertEqual(exits.exit_for("advice")["status"], "not_built")  # never promised (H10 built dashboards)
        self.assertIsNone(exits.exit_for("magic"))

    def test_the_traps_get_the_other_route_when_read_the_wrong_way(self):
        self.assertEqual(exits.suggest(TRAP_1, "reporting")["track"], "prediction")
        self.assertEqual(exits.suggest(TRAP_2, "prediction")["track"], "reporting")
        self.assertIn("which customers are going to buy again next month", exits.suggest(TRAP_1, "analysis")["quote"])
        self.assertIsNone(exits.suggest(TRAP_1, "prediction"))  # already on the route the words mean
        self.assertIsNone(exits.suggest("Build a model of churn by plan", "prediction"))  # a real prediction
        self.assertEqual(exits.suggest("Make a report listing every churned customer with their ID", "reporting")["track"], "preparation")
        self.assertEqual(exits.suggest("کدام مشتری‌ها ماه آینده دوباره خرید می‌کنند؟", "reporting")["track"], "prediction")

    def test_ordinary_requests_get_no_suggestion(self):  # the review of H8: each of these misfired
        quiet = [("Predict which customers will churn next month and list each one with their ID so the team can call them.", "prediction"),
                 ("Score every customer with their ID by churn risk for next month.", "prediction"),
                 ("Build a model of last year's bookings to predict cancellations next year.", "prediction"),
                 ("Train a model of the past month's transactions to flag fraud", "reporting"),
                 ("کدام ستون‌ها را بعد از پاکسازی نگه دارم؟", "preparation"),
                 ("کدام معیار برای بعد مناسب است؟", "advice"),
                 ("فهرست مشتری‌ها به تفکیک نام شهر", "analysis"),
                 ("Is it okay to train on all columns with their ids?", "advice"),
                 ("Which metric will be best for my model in the coming months?", "advice")]
        for text, track in quiet:
            self.assertIsNone(exits.suggest(text, track), (text, track))

    def test_the_conversion_carries_what_the_chat_showed(self):
        self.assertIsNone(exits.conversion({}, "churned"))  # nothing explored: converts as before
        bounded = {"exposure": [{"columns_related": [["plan", "churned"], ["city", "monthly_spend"]],
                                 "period": {"column": "signup_date", "end": "2025-06-30"}},
                                {"columns_related": [["churned", "support_tickets_90d"]], "period": {"column": "signup_date", "end": "2025-03-31"}}]}
        out = exits.conversion(bounded, "churned")
        self.assertEqual(out["holdout_after"], {"column": "signup_date", "date": "2025-06-30"})
        self.assertIn("does not choose this date by itself yet", out["note"])  # said as it is
        self.assertEqual(out["related_to_target"], ["plan", "support_tickets_90d"])
        self.assertIn("worth checking for leakage", out["note"])
        mixed = {"exposure": [{"columns_related": [], "period": {"column": "signup_date", "end": "2025-06-30"}},
                              {"columns_related": [["customer_id", "churned"]], "period": {"column": "order_date", "end": "2025-12-31"}}]}
        out = exits.conversion(mixed, "churned")
        self.assertTrue(out["holdout_seen"])  # two date columns: never one date for both
        self.assertEqual(out["related_to_target"], [])  # never an identifier
        whole = exits.conversion({"exposure": [{"columns_related": [["plan", "monthly_spend"]]}]}, "churned")
        self.assertTrue(whole["holdout_seen"])  # an answer over every row: the holdout's rows were seen
        self.assertEqual(whole["related_to_target"], [])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        self.ledger = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.ledger)
        self.addCleanup(ledger.install, None)

    def start(self, text):
        d = self.store.create(text)
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        return agent, d["id"]

    def test_the_exit_follows_the_track_and_the_ledger_keeps_it(self):
        agent, draft_id = self.start(TRAP_1)
        d = self.store.get(draft_id)
        self.assertEqual((d["exit"]["kind"], d["exit"]["status"]), ("ml_project", "open"))
        agent.choose_intent(draft_id, "reporting")  # the person picks the report the words seemed to ask for
        d = self.store.get(draft_id)
        self.assertEqual((d["exit"]["kind"], d["exit"]["status"]), ("dashboard", "open"))  # built (H10), not yet reached
        said = [m for m in d["messages"] if m.get("kind") == "route_suggestion"]
        self.assertEqual([m["suggestion"]["track"] for m in said], ["prediction"])
        self.assertIn("model trained on the past", said[0]["text"])
        agent.route(draft_id)  # said once
        self.assertEqual(len([m for m in self.store.get(draft_id)["messages"] if m.get("kind") == "route_suggestion"]), 1)
        exits_kept = [r for r in self.ledger.records(point="request.exit")]
        self.assertEqual([r.answers[0].value for r in exits_kept][:2], ["ml_project", "dashboard"])

    def test_a_reached_exit_survives_a_switch_away_and_back_and_labels_the_ledger(self):
        agent, draft_id = self.start("What was the average monthly spend per plan?")
        name = tables.write("customers", self.store.data_dir(draft_id))
        asset = pipeline.new_asset(self.store, draft_id, "upload", name, name)
        pipeline.run(self.store, draft_id, asset["id"], agent, None)
        agent.choose_intent(draft_id, "prediction")
        self.assertEqual(self.store.get(draft_id)["exit"]["kind"], "ml_project")
        agent.choose_intent(draft_id, "analysis")
        self.assertEqual(self.store.get(draft_id)["exit"]["status"], "reached")
        labels = [x.value for x in self.ledger.labels(point="request.exit")]
        self.assertIn("analysis", labels)

    def test_the_dates_an_answer_covered_reach_the_conversion(self):
        agent, draft_id = self.start("What was the average monthly spend per plan?")
        name = tables.write("customers", self.store.data_dir(draft_id))
        asset = pipeline.new_asset(self.store, draft_id, "upload", name, name)
        pipeline.run(self.store, draft_id, asset["id"], agent, None)
        agent.edit_answer(draft_id, 0, {"grain": "month"})
        agent.store.update(draft_id, lambda d: d.update(exposure=[e for e in d["exposure"] if e.get("period")]))
        from dclab_rnd import home_query
        from dclab_core import query as Q
        plan = Q.QueryPlan(measures=[{"name": "n", "op": "count"}], group_by=["plan"],
                           period={"column": "signup_date", "grain": "month", "start": "2024-01-01", "end": "2024-12-31"})
        home_query.run(self.store, draft_id, plan)
        d = self.store.get(draft_id)
        self.assertEqual(d["exposure"][-1]["period"]["end"], "2024-12-31")

    def test_a_reached_exit_is_never_undone(self):
        agent, draft_id = self.start("What was the average monthly spend per plan?")
        name = tables.write("customers", self.store.data_dir(draft_id))
        asset = pipeline.new_asset(self.store, draft_id, "upload", name, name)
        pipeline.run(self.store, draft_id, asset["id"], agent, None)
        self.assertEqual(self.store.get(draft_id)["exit"]["status"], "reached")
        agent.route(draft_id)
        self.assertEqual(self.store.get(draft_id)["exit"]["status"], "reached")
        self.assertTrue(self.store.get(draft_id)["exposure"][0]["columns_related"])

    def test_a_prediction_built_after_an_analysis_carries_what_it_showed(self):
        from dclab_rnd.draft.api import build_project
        from dclab_rnd.studio.store import ProjectStore
        agent, draft_id = self.start("Predict which customers will churn next month. The outcome is churned_next_month.")
        name = tables.write("customers", self.store.data_dir(draft_id))
        asset = pipeline.new_asset(self.store, draft_id, "upload", name, name)
        pipeline.run(self.store, draft_id, asset["id"], agent, None)
        answers = {"target": "churned_next_month", "prediction_moment": "At the start of each month, from what we knew the day before."}
        for _ in range(8):  # the person answers Home's questions, as in a real conversation
            pending = agent.pending(self.store.get(draft_id))
            if pending is None:
                break
            agent.reply(draft_id, answers.get(pending["field"], "I'm not sure"))
        from dclab_rnd.draft import map as problem_map
        problem_map.edit(self.store, draft_id, "confirm")
        self.store.update(draft_id, lambda d: d.update(  # an analysis asked earlier in this conversation
            exposure=[{"columns_related": [["plan", "churned_next_month"], ["city", "monthly_spend"]], "rows": "all"}]))
        project = build_project(self.store, ProjectStore(self.home / "projects"), draft_id)
        self.assertTrue(project["exposure"]["holdout_seen"])
        self.assertEqual(project["exposure"]["related_to_target"], ["plan"])
        agent.built(draft_id, {"project_id": project["id"], "started": False, "note": "The project is built."})
        said = [m.get("text") or "" for m in self.store.get(draft_id)["messages"]]
        self.assertTrue(any("Already compared with churned_next_month" in t for t in said))
        self.assertEqual(self.store.get(draft_id)["exit"]["status"], "reached")


if __name__ == "__main__":
    unittest.main()

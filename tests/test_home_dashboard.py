"""KPIs and dashboards on Home (package H10): each H1 reporting request gets its tiles, every number from the query engine;
a new data version refreshes every tile and records the run; a list of rows is never a tile."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import dashboard as D  # noqa: E402
from dclab_core import query as Q  # noqa: E402
from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.agent_eval.home import tables  # noqa: E402
from dclab_rnd.draft import dashboard, pipeline  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


class CoreTests(unittest.TestCase):
    def test_a_failing_tile_never_stops_the_others_and_each_run_is_recorded(self):
        t = tables.build("sales")
        good = D.Kpi(name="orders", plan=Q.QueryPlan(measures=[{"name": "rows", "op": "count"}], group_by=["region"]))
        bad = D.Kpi(name="nope", plan=Q.QueryPlan(measures=[{"name": "x", "op": "sum", "column": "no_such_column"}]))
        board, tiles = D.refresh(D.Dashboard(title="t", kpis=[good, bad]), t, data_version="v1", at="2026-10-28T10:00")
        self.assertEqual([("error" in x) for x in tiles], [False, True])
        board, _ = D.refresh(board, t, data_version="v2", at="2026-10-28T11:00")
        self.assertEqual([r.data_version for r in board.runs], ["v1", "v2"])
        self.assertEqual(D.Dashboard.read(board.model_dump_json()), board)


class ReadTests(unittest.TestCase):
    def test_each_h1_reporting_request_gets_its_tiles(self):
        sales, customers, bookings = (tables.build(n) for n in ("sales", "customers", "bookings"))
        spec, _ = dashboard.read("Build a weekly dashboard of revenue, orders and refunds by region for the sales managers.", sales)
        ops = [(k.plan.measures[0].op, k.plan.measures[0].column, [f.value for f in k.plan.filters]) for k in spec["kpis"]]
        self.assertEqual(ops, [("sum", "amount", []), ("count_distinct", "order_id", []), ("count", None, ["refunded"])])
        self.assertEqual((spec["refresh"], spec["audience"], spec["kpis"][0].plan.group_by), ("weekly", "sales managers", ["region"]))
        spec, _ = dashboard.read("I want to track churn rate and average monthly spend every month for the leadership team.", customers)
        self.assertEqual([k.plan.measures[0].op for k in spec["kpis"]], ["rate", "mean"])
        self.assertIsNone(spec["kpis"][0].plan.period)  # "every month" is how often, never a split by signup month
        spec, _ = dashboard.read("یک داشبورد روزانه از فروش هر کانال برای مدیر بازاریابی می‌خواهم.", sales)
        self.assertEqual((spec["refresh"], spec["kpis"][0].plan.group_by), ("daily", ["channel"]))
        spec, _ = dashboard.read("یک KPI dashboard برای booking ها و cancellation rate هر destination می‌خواهم که هر هفته refresh شود.", bookings)
        self.assertEqual([k.plan.measures[0].op for k in spec["kpis"]], ["count_distinct", "rate"])  # distinct bookings
        self.assertEqual(spec["kpis"][1].plan.measures[0].where.column, "cancelled")
        spec, unread = dashboard.read("Make a report listing every churned customer with their ID and monthly spend.", customers)
        self.assertIsNone(spec)
        self.assertIn("never single rows", unread[0])
        spec, unread = dashboard.read("Build a model of last month's sales by region.", sales)
        self.assertTrue(spec["kpis"])
        self.assertTrue(any("last month" in u for u in unread))  # said, never applied silently


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def upload(self, agent, draft_id, frame, name="sales.csv"):
        path = Path(self.store.data_dir(draft_id))
        path.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path / name, index=False)
        asset = pipeline.new_asset(self.store, draft_id, "upload", name, name)
        pipeline.run(self.store, draft_id, asset["id"], agent, None)

    def test_the_dashboard_is_built_and_a_new_data_version_refreshes_every_tile(self):
        d = self.store.create("Build a weekly dashboard of revenue, orders and refunds by region for the sales managers.")
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        self.upload(agent, d["id"], tables.build("sales"))
        for _ in range(4):
            if agent.pending(self.store.get(d["id"])) is None:
                break
            agent.reply(d["id"], "I'm not sure")
        d = self.store.get(d["id"])
        self.assertEqual((d["exit"]["kind"], d["exit"]["status"]), ("dashboard", "reached"))
        a = next(x for x in d["assets"] if x["id"] == d["active_asset"])
        clean = pd.read_parquet(Path(self.store.data_dir(d["id"])) / a["clean_file"])
        revenue = {r[0]: r[1] for r in d["dashboard"]["tiles"][0]["result"]["rows"]}
        for region, value in clean.groupby("region")["amount"].sum().items():
            self.assertAlmostEqual(revenue[region], value, places=6)
        first = d["dashboard"]["data_version"]
        self.upload(agent, d["id"], tables.build("sales").iloc[:1500], name="sales_v2.csv")  # a new data version
        d = self.store.get(d["id"])
        board = D.Dashboard.model_validate(d["dashboard"]["board"])
        self.assertEqual(len(board.runs), 2)
        self.assertNotEqual(board.runs[-1].data_version, first)
        self.assertEqual(len(board.runs[-1].tiles), 3)  # every tile, recorded
        self.assertTrue(any(m.get("kind") == "dashboard" and "refreshed" in m["text"] for m in d["messages"]))

    def test_a_list_of_rows_is_never_a_dashboard(self):
        d = self.store.create("Make a report listing every churned customer with their ID and monthly spend.")
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        if agent.track(self.store.get(d["id"])) != "reporting":
            agent.choose_intent(d["id"], "reporting")
        self.upload(agent, d["id"], tables.build("customers"), name="customers.csv")
        for _ in range(4):
            if agent.pending(self.store.get(d["id"])) is None:
                break
            agent.reply(d["id"], "I'm not sure")
        d = self.store.get(d["id"])
        self.assertFalse((d.get("dashboard") or {}).get("board"))
        said = " ".join(m.get("text") or "" for m in d["messages"])
        self.assertIn("never single rows", said)
        self.assertNotIn("C-00", said)


class ReviewOfH10aTests(HomeTests):
    """The review's findings (H10a), each with what showed it."""

    def table(self):
        t = tables.build("sales")
        return t.assign(discount_amount=t["amount"] * 0.1, refund_amount=t["amount"] * (t["status"] == "refunded"),
                        segment=["returning", "new"] * (len(t) // 2))

    def test_words_read_to_the_right_column_or_said(self):
        t = self.table()
        spec, _ = dashboard.read("Build a dashboard of revenue by region", t)
        self.assertEqual(spec["kpis"][0].plan.measures[0].column, "amount")  # never discount_amount
        spec, _ = dashboard.read("Build a dashboard of refund amount by region", t)
        self.assertEqual(spec["kpis"][0].plan.measures[0].column, "refund_amount")
        for text in ("Build a dashboard of complaints by region", "Build a dashboard of returns by region",
                     "Build a dashboard of revenue excluding refunds by region", "Build a dashboard of net revenue by region"):
            spec, unread = dashboard.read(text, t)
            self.assertIsNone(spec, text)  # never a near-miss value or a dropped condition
            self.assertTrue(unread, text)
        spec, _ = dashboard.read("Build a dashboard of orders by region and channel", t)
        self.assertEqual(spec["kpis"][0].plan.group_by, ["region", "channel"])  # every split, not the first only
        spec, _ = dashboard.read("Build a dashboard of revenue trend by region", t)
        self.assertEqual(spec["kpis"][0].plan.limit, 500)  # never cut at the default 50 in silence
        spec, _ = dashboard.read("یک داشبورد از فروش هر region می‌خواهم که هر روز ارسال شود.", t)
        self.assertIsNone(spec["kpis"][0].plan.period)  # "هر روز" is a cadence, not a split by day

    def test_the_answer_to_which_numbers_is_read_and_the_audience_is_never_a_kpi(self):
        spec, _ = dashboard.read("I need a weekly dashboard for the sales managers.", self.table(), kpis_said="revenue and orders by region")
        self.assertEqual([k.plan.measures[0].op for k in spec["kpis"]], ["sum", "count_distinct"])
        self.assertEqual(spec["kpis"][0].plan.group_by, ["region"])
        spec, unread = dashboard.read("I need a weekly dashboard for the sales managers.", self.table())
        self.assertIsNone(spec)  # "sales managers" is who it is for

    def test_tiles_join_the_exposure_log_and_the_log_holds_no_value(self):
        d = self.store.create("Build a weekly dashboard of revenue, orders and refunds by region for the sales managers.")
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        self.upload(agent, d["id"], tables.build("sales"))
        d = self.store.get(d["id"])
        self.assertTrue(any(e.get("by") == "dashboard" and ["region", "amount"] in e["columns_related"] for e in d["exposure"]))

    def test_a_board_none_of_whose_tiles_runs_is_said_and_never_reached(self):
        d = self.store.create("Build a weekly dashboard of revenue, orders and refunds by region for the sales managers.")
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        self.upload(agent, d["id"], tables.build("sales"))
        self.upload(agent, d["id"], tables.build("customers"), name="customers.csv")  # another table: no tile can run
        d = self.store.get(d["id"])
        self.assertIn("None of the dashboard's tiles could be computed", d["messages"][-1]["text"])

    def test_the_refresh_route_never_claims_a_refresh_that_did_not_run(self):
        import os
        from unittest import mock
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)
        c = TestClient(create_app(Path(tempfile.mkdtemp())))
        c.__enter__()
        self.addCleanup(c.__exit__, None, None, None)
        h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
        d = c.post("/api/drafts", json={"problem": "Build a weekly dashboard of revenue by region for the sales managers."}, headers=h).json()
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/dashboard/refresh", headers=h).status_code, 404)  # no board yet
        c.put(f"/api/drafts/{d['id']}/data?filename=sales.csv", content=tables.build("sales").to_csv(index=False).encode(),
              headers={**h, "Content-Type": "application/octet-stream"})
        import time
        for _ in range(300):
            if (c.get(f"/api/drafts/{d['id']}").json().get("dashboard") or {}).get("board"):
                break
            time.sleep(0.1)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/dashboard/refresh", headers=h).status_code, 200)
        c.post(f"/api/drafts/{d['id']}/intent", json={"track": "analysis"}, headers=h)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/dashboard/refresh", headers=h).status_code, 409)


if __name__ == "__main__":
    unittest.main()

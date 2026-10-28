"""Home prepares a table on request (package H9): each H1 dev preparation request makes the expected table, a step
that removes many rows waits for a yes, and the original file is never changed."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.agent_eval.home import tables  # noqa: E402
from dclab_rnd.audit import sha256_of  # noqa: E402
from dclab_rnd.draft import pipeline, prepare  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def ask(self, text, *names):
        d = self.store.create(text)
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        for name in names:  # the first table is the main one; the last uploaded is active, so upload it last
            pass
        for name in reversed(names):
            filename = tables.write(name, self.store.data_dir(d["id"]))
            asset = pipeline.new_asset(self.store, d["id"], "upload", filename, filename)
            pipeline.run(self.store, d["id"], asset["id"], agent, None)
        return agent, d["id"]

    def original(self, draft_id):
        d = self.store.get(draft_id)
        a = next(a for a in d["assets"] if a["id"] == d["active_asset"])
        return pd.read_parquet(Path(self.store.data_dir(draft_id)) / a["clean_file"]), a

    def prepared(self, draft_id):
        rec = self.store.get(draft_id).get("recipe") or {}
        return pd.read_parquet(Path(self.store.data_dir(draft_id)) / rec["file"])

    def made(self, draft_id):
        rec = self.store.get(draft_id).get("recipe") or {}
        return bool(rec.get("done")) and (Path(self.store.data_dir(draft_id)) / str(rec.get("file"))).exists()


class PrepareTests(Base):
    def test_remove_duplicate_orders_as_excel_and_the_original_untouched(self):  # H1 d-r1
        agent, draft_id = self.ask("Remove duplicate orders and export the clean table as an Excel file.", "sales")
        src, asset = self.original(draft_id)
        d = self.store.get(draft_id)
        self.assertEqual((d["recipe"]["format"], d["exit"]["kind"], d["exit"]["status"]), ("xlsx", "dataset", "reached"))
        out = self.prepared(draft_id)
        self.assertEqual(len(out), src["order_id"].nunique())
        self.assertEqual(sha256_of(Path(self.store.data_dir(draft_id)) / asset["clean_file"]), asset["clean_sha256"])  # never written
        self.assertTrue(any(m.get("kind") == "prepared" for m in d["messages"]))

    def test_dates_read_and_rows_without_an_amount_dropped_in_persian(self):  # H1 d-r3
        agent, draft_id = self.ask("ستون تاریخ را به فرمت درست تبدیل کن و ردیف‌هایی که مبلغ ندارند حذف کن.", "sales")
        src, _ = self.original(draft_id)
        out = self.prepared(draft_id)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["order_date"]))
        self.assertEqual(len(out), int(src["amount"].notna().sum()))

    def test_a_pivot_and_months_since_signup(self):  # H1 d-r4, d-r6
        agent, draft_id = self.ask("لطفاً یک pivot بساز: هر region یک ردیف و هر product_category یک ستون با جمع amount.", "sales")
        src, _ = self.original(draft_id)
        out = self.prepared(draft_id)
        self.assertEqual(len(out), src["region"].nunique())
        want = src.pivot_table(index="region", columns="product_category", values="amount", aggfunc="sum")
        self.assertAlmostEqual(out.set_index("region").loc[want.index[0], str(want.columns[0])], want.iloc[0, 0], places=6)
        agent, draft_id = self.ask("Turn signup_date into months since signup and save it as a new column.", "customers")
        self.assertIn("months_since_signup", self.prepared(draft_id).columns)

    def test_keep_only_paid_orders_from_2026_waits_for_a_yes(self):  # H1 d-r5, trap 3
        agent, draft_id = self.ask("Keep only the paid orders from 2026 and give me that table.", "sales")
        d = self.store.get(draft_id)
        q = agent.pending(d)
        self.assertEqual(q["field"], "confirm_step")
        self.assertIn("removes", q["text"])
        self.assertFalse(self.made(draft_id))  # nothing made before the yes
        agent.reply(draft_id, "Yes, apply it")
        out = self.prepared(draft_id)
        self.assertTrue((out["status"] == "paid").all())
        self.assertTrue((pd.to_datetime(out["order_date"]).dt.year == 2026).all())
        steps = self.store.get(draft_id)["recipe"]["steps"]
        self.assertTrue(all(s["confirmed"] for s in steps if (s.get("removed_share") or 0) > 0.05))

    def test_not_sure_skips_the_step(self):
        agent, draft_id = self.ask("Keep only the paid orders from 2026 and give me that table.", "sales")
        agent.reply(draft_id, "I'm not sure")
        src, _ = self.original(draft_id)
        self.assertEqual(len(self.prepared(draft_id)), len(src))  # rows are never dropped by default
        self.assertTrue(any(s.get("skipped") for s in self.store.get(draft_id)["recipe"]["steps"]))

    def test_a_join_brings_the_customers_city_and_segment(self):  # H1 d-r2
        agent, draft_id = self.ask("Join the orders with the customer table on customer_id so every order has the customer's city "
                                   "and segment.", "sales", "customers")
        d = self.store.get(draft_id)
        if agent.track(d) != "preparation":
            agent.choose_intent(draft_id, "preparation")
        out = self.prepared(draft_id)
        self.assertTrue({"city", "segment"} <= set(out.columns))
        src, _ = self.original(draft_id)
        self.assertEqual(len(out), len(src))

    def test_words_the_rules_cannot_read_are_said_and_a_follow_up_is_read(self):
        agent, draft_id = self.ask("Prepare the table nicely, please.", "sales")
        if agent.track(self.store.get(draft_id)) != "preparation":  # the point here is the follow-up, not the reading
            agent.choose_intent(draft_id, "preparation")
        self.assertFalse(self.made(draft_id))
        q = agent.pending(self.store.get(draft_id))
        if q is None:  # Home said it could not read the steps and asked for them one at a time
            said = " ".join(m.get("text") or "" for m in self.store.get(draft_id)["messages"])
            self.assertIn("could not read the steps", said)
        agent.reply(draft_id, "remove duplicate orders")  # the answer, or the reply Home invited (the review of H9a)
        self.assertTrue(self.made(draft_id))
        src, _ = self.original(draft_id)
        self.assertEqual(len(self.prepared(draft_id)), src["order_id"].nunique())


class ReviewOfH9aTests(Base):
    """The review's findings (H9a), each with the request or reply that showed it."""

    def guarded(self):
        agent, draft_id = self.ask("Keep only the paid orders from 2026 and give me that table.", "sales")
        self.assertEqual(agent.pending(self.store.get(draft_id))["field"], "confirm_step")
        return agent, draft_id

    def test_only_a_whole_yes_applies_a_guarded_step(self):
        src = None
        for reply in ("اعمال نکن", "ok no", "okay, don't apply it", "apply it? no wait", "yes but only 2025", "No, skip it", "I'm not sure"):
            agent, draft_id = self.guarded()
            src = src if src is not None else self.original(draft_id)[0]
            agent.reply(draft_id, reply)
            if self.made(draft_id):
                self.assertEqual(len(self.prepared(draft_id)), len(src), reply)  # skipped, never applied
        for reply in ("Yes, apply it", "yes", "بله"):
            agent, draft_id = self.guarded()
            agent.reply(draft_id, reply)
            self.assertTrue((self.prepared(draft_id)["status"] == "paid").all(), reply)

    def test_the_question_says_what_the_step_will_do(self):
        agent, draft_id = self.guarded()
        q = agent.pending(self.store.get(draft_id))["text"]
        self.assertIn("status = paid", q)
        self.assertIn("order_date from 2026-01-01 to 2026-12-31", q)

    def test_another_message_while_asked_is_read_not_taken_as_a_skip(self):
        agent, draft_id = self.guarded()
        agent.reply(draft_id, "Actually I want to predict which orders get refunded")
        self.assertFalse(self.made(draft_id))  # no table written on the strength of an unrelated message
        self.assertNotEqual((self.store.get(draft_id).get("exit") or {}).get("status"), "reached")

    def test_a_negation_or_an_unplaceable_year_is_never_read_as_a_filter(self):
        for text in ("Keep only the orders that are not cancelled and give me that table.",
                     "Keep only orders except refunded and give me that table."):
            agent, draft_id = self.ask(text, "sales")
            self.assertFalse(self.made(draft_id), text)
            self.assertFalse((self.store.get(draft_id).get("recipe") or {}).get("planned"), text)
        frame = tables.build("sales").assign(ship_date=lambda t: t["order_date"])
        steps, unread = prepare.read_steps("Keep only the paid orders from 2026 and give me that table.", frame)
        self.assertEqual(steps, [])  # two date columns: no partial filter
        self.assertTrue(unread)
        steps, _ = prepare.read_steps("Keep only the top 2000 orders.", tables.build("sales"))
        self.assertFalse(any(f.value and "2000-" in str(f.value) for s in steps for f in s.filters))

    def test_a_join_adds_to_the_table_where_the_key_repeats_in_the_natural_upload_order(self):
        d = self.store.create("Join the orders with the customer table on customer_id so every order has the customer's city and segment.")
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        for name in ("sales", "customers"):  # orders first, then customers: customers is the last upload
            filename = tables.write(name, self.store.data_dir(d["id"]))
            asset = pipeline.new_asset(self.store, d["id"], "upload", filename, filename)
            pipeline.run(self.store, d["id"], asset["id"], agent, None)
        out = self.prepared(d["id"])
        self.assertTrue({"order_id", "city", "segment"} <= set(out.columns))
        self.assertEqual(len(out), len(tables.build("sales").drop_duplicates()))

    def test_a_pivot_reads_each_column_by_its_role(self):
        steps, _ = prepare.read_steps("pivot the sum of amount per region by product_category", tables.build("sales"))
        self.assertEqual((steps[0].index, steps[0].spread, steps[0].values), ("region", "product_category", "amount"))
        steps, unread = prepare.read_steps("pivot the sum of region per amount by channel", tables.build("sales"))
        self.assertEqual(steps, [])  # a sum of text: refused

    def test_months_since_names_its_day_and_the_full_pivot_matches(self):
        agent, draft_id = self.ask("Turn signup_date into months since signup and save it as a new column.", "customers")
        src, _ = self.original(draft_id)
        out = self.prepared(draft_id)
        at = pd.to_datetime(src["signup_date"]).max()
        d = pd.to_datetime(src["signup_date"])
        want = (at.year - d.dt.year) * 12 + (at.month - d.dt.month) - (at.day < d.dt.day).astype(int)
        self.assertEqual(out["months_since_signup"].astype(int).tolist(), want.tolist())
        said = [m.get("text") or "" for m in self.store.get(draft_id)["messages"] if m.get("kind") == "prepared"][0]
        self.assertIn(at.date().isoformat(), said)
        agent, draft_id = self.ask("لطفاً یک pivot بساز: هر region یک ردیف و هر product_category یک ستون با جمع amount.", "sales")
        src, _ = self.original(draft_id)
        want = src.pivot_table(index="region", columns="product_category", values="amount", aggfunc="sum")
        got = self.prepared(draft_id).set_index("region")
        for region, row in want.iterrows():
            for cat, value in row.items():
                self.assertAlmostEqual(got.loc[region, str(cat)], value, places=6)

    def test_the_file_is_its_own_the_exit_stays_reached_and_the_guard_is_in_the_ledger(self):
        agent, draft_id = self.guarded()
        agent.reply(draft_id, "Yes, apply it")
        rec = self.store.get(draft_id)["recipe"]
        self.assertTrue(rec["file"].startswith("prepared__"))
        agent.choose_intent(draft_id, "analysis")
        agent.choose_intent(draft_id, "preparation")
        self.assertEqual(self.store.get(draft_id)["exit"]["status"], "reached")
        led = ledger.current_ledger()
        labels = [x.value for x in led.labels(point="recipe.step")]
        self.assertEqual(labels, [True])

    def test_combine_without_a_key_does_not_wait_for_a_second_table(self):
        agent, draft_id = self.ask("Remove duplicate orders and combine first and last name.", "sales")
        said = " ".join(m.get("text") or "" for m in self.store.get(draft_id)["messages"])
        self.assertNotIn("add the second one", said)


class DownloadTests(unittest.TestCase):
    def test_the_prepared_table_downloads_in_each_format(self):
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
        d = c.post("/api/drafts", json={"problem": "Remove duplicate orders and export the clean table as an Excel file."}, headers=h).json()
        self.assertEqual(c.get(f"/api/drafts/{d['id']}/prepared").status_code, 404)  # nothing prepared yet
        c.put(f"/api/drafts/{d['id']}/data?filename=sales.csv", content=tables.build("sales").to_csv(index=False).encode(),
              headers={**h, "Content-Type": "application/octet-stream"})
        import time
        for _ in range(300):
            if (c.get(f"/api/drafts/{d['id']}").json().get("recipe") or {}).get("done"):
                break
            time.sleep(0.1)
        for fmt, kind in (("csv", "text/csv"), ("excel", "spreadsheetml"), ("parquet", "octet-stream")):
            r = c.get(f"/api/drafts/{d['id']}/prepared?format={fmt}")
            self.assertEqual(r.status_code, 200, fmt)
            self.assertIn(kind, r.headers["content-type"])
        self.assertEqual(c.get(f"/api/drafts/{d['id']}/prepared?format=pdf").status_code, 422)

    def test_a_formula_like_cell_is_exported_as_text(self):
        import io
        frame = pd.DataFrame({"note": ["=HYPERLINK(1)", "fine", "-3", "@x"], "x": [1, -2, 3, 4]})
        safe = prepare.safe_cells(frame)
        self.assertEqual(safe["note"].tolist(), ["'=HYPERLINK(1)", "fine", "'-3", "'@x"])
        self.assertEqual(safe["x"].tolist(), [1, -2, 3, 4])  # numbers stay numbers
        self.assertEqual(frame["note"][0], "=HYPERLINK(1)")  # on a copy
        self.assertTrue(io)


if __name__ == "__main__":
    unittest.main()

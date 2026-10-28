"""Answers from the data in Home's chat (package H7): a table from the brief, a cited sentence or the template, the
edit controls, follow-up questions, the model's tools, and the notebook that reproduces every saved answer.

Done when: on the H1 dev requests every analysis request with one table gets a correct table (checked here against
pandas written by hand), no answer has an uncited number, and the notebook reproduces every saved answer.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from dclab_rnd.agent_eval import home  # noqa: E402
from dclab_rnd.agent_eval.home import tables  # noqa: E402
from dclab_rnd.draft import pipeline  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.query import answer as answers  # noqa: E402
from dclab_rnd.query.execute import floor  # noqa: E402

CASES = {c["id"]: c for c in home.load("dev")}


class Scripted:
    """A model client with fixed replies, which records what it was sent."""

    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def complete(self, messages, tools=None, **_):
        self.sent.append(messages)
        return {"content": self.replies.pop(0) if self.replies else "", "tool_calls": [], "usage": {}}


class Exploding:
    def complete(self, *a, **k):
        raise AssertionError("no model may be asked here")


def cleaned(f):
    """The cleaned table the draft answered from (the pipeline drops empty and duplicate rows), read as DCLab reads it."""
    from dclab_rnd.query.execute import table
    return table(f.store, f.draft)[0]


class Flow:
    """A Home draft on a request, with its H1 tables run through the real pipeline (in a temporary store)."""

    def __init__(self, text, data=("sales",), **clients):
        self._env = home._isolated()
        self._env.__enter__()
        self.tmp = tempfile.mkdtemp()
        self.store = DraftStore(Path(self.tmp) / "drafts")
        self.draft_id = self.store.create(text)["id"]
        self.agent = HomeAgent(self.store, None, on_request=lambda *a: None, **clients)
        self.agent.start(self.draft_id)
        for name in data:
            self.add(name)

    def add(self, name):
        filename = tables.write(name, self.store.data_dir(self.draft_id))
        asset = pipeline.new_asset(self.store, self.draft_id, "upload", filename, filename)
        pipeline.run(self.store, self.draft_id, asset["id"], self.agent, None)

    @property
    def draft(self):
        return self.store.get(self.draft_id)

    def answers(self):
        return self.draft.get("answers") or []

    def close(self):
        self._env.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)


def flow(test, case_id=None, text=None, data=None, **clients):
    case = CASES[case_id] if case_id else None
    f = Flow(text or case["text"], tuple(data if data is not None else case["data"]), **clients)
    test.addCleanup(f.close)
    return f


def rows_of(answer):
    return {tuple(r[:-1]) if len(r) > 2 else r[0]: r[-1] for r in answer["result"]["rows"]}


class H1Tests(unittest.TestCase):
    """The H1 dev analysis requests with one table, answered from the brief by code alone."""

    def test_revenue_per_region_last_quarter_paid_only(self):  # d-a1
        f = flow(self, "d-a1")
        [a] = f.answers()
        s = cleaned(f)
        part = s[(s.status == "paid") & (s.order_date >= "2026-01-01") & (s.order_date <= "2026-03-31")]
        expected = part.groupby("region")["amount"].sum().sort_values(ascending=False)
        self.assertEqual([r[0] for r in a["result"]["rows"]], list(expected.index))
        for (region, value), want in zip(a["result"]["rows"], expected.values):
            self.assertAlmostEqual(value, float(want), places=4)
        self.assertIn("I took amount as the revenue.", a["text"])  # every assumption said
        self.assertIn("the last full quarter in the data", a["text"])
        self.assertEqual(a["chart"]["kind"], "bar")

    def test_average_order_value_web_and_app_in_2025(self):  # d-a2
        f = flow(self, "d-a2")
        [a] = f.answers()
        s = cleaned(f)
        part = s[s.channel.isin(["web", "app"]) & (s.order_date.dt.year == 2025)]
        want = part.groupby("channel")["amount"].mean()
        self.assertEqual({k: round(v, 4) for k, v in rows_of(a).items()}, {k: round(float(v), 4) for k, v in want.items()})

    def test_orders_per_month_of_2025_in_persian(self):  # d-a3
        f = flow(self, "d-a3")
        [a] = f.answers()
        s = cleaned(f)
        want = s[s.order_date.dt.year == 2025].groupby(s.order_date.dt.to_period("M").astype(str)).size()
        self.assertEqual(rows_of(a), {k: int(v) for k, v in want.items()})
        self.assertEqual(a["chart"]["kind"], "line")

    def test_average_monthly_spend_by_plan_in_persian(self):  # d-a4: «به تفکیک نوع پلن» finds the plan column
        f = flow(self, "d-a4")
        [a] = f.answers()
        want = cleaned(f).groupby("plan")["monthly_spend"].mean()
        self.assertEqual({k: round(v, 4) for k, v in rows_of(a).items()}, {k: round(float(v), 4) for k, v in want.items()})

    def test_a_column_the_table_lacks_is_asked_not_answered(self):  # d-a6: the phantom-column trap
        f = flow(self, "d-a6")
        self.assertEqual(f.answers(), [])
        self.assertIn("lifetime_spend", f.agent.pending(f.draft)["text"])

    def test_no_answer_has_an_uncited_number_or_a_listed_row(self):
        for case_id in ("d-a1", "d-a2", "d-a3", "d-a4", "d-a6", "d-r2"):
            f = flow(self, case_id)
            columns = set().union(*(set(tables.build(n).columns) for n in CASES[case_id]["data"]))
            ids = set().union(*(tables.id_values(n) for n in CASES[case_id]["data"]))
            obs = home.read_draft(f.draft, {str(c) for c in columns}, ids)
            self.assertEqual((obs["uncited"], obs["rows_exposed"]), ([], []), case_id)

    def test_a_question_over_two_tables_is_not_answered_from_the_second(self):  # d-r2 needs a join (H9)
        f = flow(self, "d-r2")
        self.assertEqual(len(f.answers()), 1)  # the first table's answer, said with its assumption; none once two are there
        self.assertEqual(f.answers()[0]["asset"], f.draft["assets"][0]["id"])


class NotebookTests(unittest.TestCase):
    def test_the_notebook_reproduces_every_saved_answer(self):
        f = flow(self, "d-a1")
        f.agent.edit_answer(f.draft_id, "a1", {"split_by": "channel"})
        f.agent.edit_answer(f.draft_id, "a1", {"grain": "month", "split_by": ""})
        f.agent.answer(f.draft_id, {"group_by": ["region"], "measures": [{"name": "top", "op": "max", "column": "amount"},
                                                                         {"name": "buyers", "op": "count_distinct", "column": "customer_id"}]},
                       question="the largest order per region")
        for a in f.answers():
            f.agent.save_answer(f.draft_id, a["id"])
        saved = f.answers()
        self.assertEqual(len(saved), 4)
        from dclab_rnd.query.execute import table
        from dclab_rnd.query.validate import kinds
        frame, _ = table(f.store, f.draft)
        nb = answers.notebook(saved, floor(), [c for c, k in kinds(frame).items() if k == "identifier"])
        work = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, work, True)
        frame.to_parquet(work / "clean.parquet")
        code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
        here = os.getcwd()
        os.chdir(work)
        try:
            scope = {}
            exec(compile(code[0], "setup", "exec"), scope)  # noqa: S102 — the exported notebook, run as a person would
            for a, cell in zip(saved, code[1:]):
                *setup, last = cell.splitlines()
                exec("\n".join(setup), scope)  # noqa: S102
                got = eval(last, scope)  # noqa: S307
                self.assertEqual(list(got.columns), a["result"]["columns"], a["id"])
                self.assertEqual(len(got), len(a["result"]["rows"]), a["id"])
                for mine, theirs in zip(got.itertuples(index=False), a["result"]["rows"]):
                    for x, y in zip(mine, theirs):
                        if isinstance(y, float):
                            self.assertAlmostEqual(float(x), y, places=4, msg=a["id"])
                        else:
                            self.assertEqual(None if pd.isna(x) else x, y, a["id"])
        finally:
            os.chdir(here)


class ControlTests(unittest.TestCase):
    def test_controls_change_the_plan_without_a_model(self):
        f = flow(self, "d-a1", answer_client=Exploding(), query_client=Exploding())
        # the first answer: the exploding client was asked for the sentence and failed, so the template answered
        self.assertEqual(f.answers()[0]["written_by"], "template")
        b = f.agent.edit_answer(f.draft_id, "a1", {"split_by": "channel"})
        self.assertEqual((b["plan"]["group_by"], b["plan_by"], b["edited_from"], b["written_by"]), (["channel"], "person", "a1", "template"))
        self.assertEqual(b["plan"]["filters"], f.answers()[0]["plan"]["filters"])  # the rest of the question stays
        c = f.agent.edit_answer(f.draft_id, b["id"], {"op": "mean"})
        self.assertEqual(c["plan"]["measures"][0]["op"], "mean")
        d = f.agent.edit_answer(f.draft_id, "a1", {"grain": "month", "split_by": ""})
        self.assertEqual((d["plan"]["group_by"], d["plan"]["period"]["grain"], d["chart"]["kind"]), ([], "month", "line"))
        self.assertEqual(d["plan"]["sort"], [{"by": "period", "desc": False}])

    def test_a_control_that_breaks_the_plan_is_refused_in_words(self):
        f = flow(self, "d-a1")
        before = len(f.answers())
        self.assertIsNone(f.agent.edit_answer(f.draft_id, "a1", {"split_by": "customer_id"}))  # a person, never a group
        self.assertEqual(len(f.answers()), before)
        self.assertTrue(f.draft["messages"][-1]["text"].startswith("I couldn't compute that"))
        with self.assertRaises(ValueError):
            answers.edited(f.answers()[0]["plan"], {"op": "max"})
        with self.assertRaises(KeyError):
            f.agent.edit_answer(f.draft_id, "a99", {"op": "sum"})

    def test_a_follow_up_question_keeps_the_rest_of_the_brief(self):
        f = flow(self, "d-a1")
        f.agent.reply(f.draft_id, "And by channel?")
        last = f.answers()[-1]
        self.assertEqual(last["plan"]["group_by"], ["channel"])
        self.assertEqual(last["plan"]["filters"], f.answers()[0]["plan"]["filters"])
        self.assertEqual(last["question"], "And by channel?")
        f.agent.reply(f.draft_id, "Thanks, that helps.")  # no field in it: noted, not a new answer
        self.assertEqual(len(f.answers()), 2)


class ModelTests(unittest.TestCase):
    def test_a_sentence_is_kept_only_when_its_numbers_are_in_the_result(self):
        good = Scripted("Paid revenue was highest in Dubai at 29280.65 last quarter.")
        f = flow(self, "d-a1", answer_client=good)
        a = f.answers()[0]
        self.assertEqual(a["written_by"], "model")
        self.assertTrue(a["text"].startswith("Paid revenue was highest in Dubai at 29280.65"))
        self.assertIn("I took amount as the revenue.", a["text"])  # the assumptions follow the model's sentence
        sent = json.dumps(good.sent[0])
        self.assertNotIn("ORD-", sent)  # aggregates only: no record of the table
        bad = Scripted("Revenue was 31000 in Dubai, up 12% on the quarter before.")
        f = flow(self, "d-a1", answer_client=bad)
        self.assertEqual(f.answers()[0]["written_by"], "template")
        self.assertNotIn("31000", f.answers()[0]["text"])

    def test_no_model_reads_the_result_when_aggregates_are_off(self):
        with mock.patch.dict(os.environ, {"DCLAB_MODEL_READS_AGGREGATES": "0"}):
            f = flow(self, "d-a1", answer_client=Exploding())
        self.assertEqual(f.answers()[0]["written_by"], "template")

    def test_a_model_writes_the_plan_when_the_words_do_not_lead_to_one(self):
        plan = {"group_by": ["segment"], "measures": [{"name": "avg_spend", "op": "mean", "column": "monthly_spend"}]}
        q = Scripted(json.dumps(plan))
        f = flow(self, text="How much do our segments spend, compared?", data=("customers",), query_client=q)
        [a] = f.answers()
        self.assertEqual((a["plan_by"], a["plan"]["group_by"]), ("model", ["segment"]))
        self.assertIn("A model wrote this calculation", a["text"])
        self.assertNotIn("premium", json.dumps(q.sent))  # column summaries only, never a category

    def test_query_data_gives_the_model_the_shape_never_the_values(self):
        f = flow(self, text="How did sales go by region?", data=("sales",))
        out = f.agent.run_tool(f.draft_id, "query_data", {"plan": {"group_by": ["region"], "measures": [{"name": "total", "op": "sum", "column": "amount"}]}})
        a = f.answers()[-1]
        self.assertEqual(out["answer"], a["id"])
        shown = json.dumps(out)
        for row in a["display"]:
            self.assertNotIn(str(row[0]), shown)
            self.assertNotIn(str(row[1]), shown)
        bad = f.agent.run_tool(f.draft_id, "query_data", {"plan": {"group_by": ["order_id"], "measures": [{"name": "n", "op": "count"}]}})
        self.assertIn("error", bad)
        self.assertIn("error", f.agent.run_tool(f.draft_id, "make_chart", {"answer": a["id"], "kind": "line"}))  # bars, not a line
        self.assertEqual(f.agent.run_tool(f.draft_id, "make_chart", {"answer": a["id"], "kind": "table"})["chart"], "table")
        self.assertEqual(f.agent.run_tool(f.draft_id, "save_answer", {"answer": a["id"], "title": "Sales by region"})["title"], "Sales by region")
        self.assertIn("error", f.agent.run_tool(f.draft_id, "save_answer", {"answer": "a99"}))

    def test_no_table_no_query(self):
        f = flow(self, text="How did sales go by region?", data=())
        self.assertIn("error", f.agent.run_tool(f.draft_id, "query_data", {"plan": {"measures": [{"name": "n", "op": "count"}]}}))


class RouteTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def wait(self, did, fn, seconds=60):
        end = time.time() + seconds
        while time.time() < end:
            d = self.client.get(f"/api/drafts/{did}").json()
            if fn(d):
                return d
            time.sleep(0.1)
        self.fail("timed out")

    def test_edit_save_and_export(self):
        c = self.client
        d = c.post("/api/drafts", json={"problem": CASES["d-a1"]["text"]}, headers=self.h).json()
        body = (tables.build("sales")).to_csv(index=False).encode()
        self.assertEqual(c.put(f"/api/drafts/{d['id']}/data", params={"filename": "sales.csv"}, content=body,
                               headers={**self.h, "Content-Type": "application/octet-stream"}).status_code, 200)
        draft = self.wait(d["id"], lambda x: x.get("answers"))
        self.assertEqual(c.get(f"/api/drafts/{d['id']}/answers/notebook").status_code, 409)  # nothing saved yet
        r = c.post(f"/api/drafts/{d['id']}/answers/a1/edit", json={"split_by": "channel"}, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["answers"][-1]["plan"]["group_by"], ["channel"])
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/answers/a1/edit", json={"op": "max"}, headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/answers/a1/edit", json={}, headers=self.h).status_code, 422)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/answers/a9/edit", json={"op": "sum"}, headers=self.h).status_code, 404)
        self.assertEqual(c.post(f"/api/drafts/{d['id']}/answers/a1/edit", json={"split_by": "order_id"}, headers=self.h).status_code, 422)
        saved = c.post(f"/api/drafts/{d['id']}/answers/a1/save", json={"title": "Paid revenue by region"}, headers=self.h).json()
        self.assertTrue(saved["answers"][0]["saved"])
        nb = c.get(f"/api/drafts/{d['id']}/answers/notebook")
        self.assertEqual(nb.status_code, 200)
        cells = nb.json()["cells"]
        self.assertIn("Paid revenue by region", "".join(cells[2]["source"]))
        self.assertIn("FLOOR = ", "".join(cells[1]["source"]))
        self.assertIn('"customer_id"', "".join(cells[1]["source"]))  # the people floor, as DCLab applied it
        self.assertTrue(draft["exposure"])


if __name__ == "__main__":
    unittest.main()

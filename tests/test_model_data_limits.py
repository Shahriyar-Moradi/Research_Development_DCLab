"""What each purpose may show a model, checked on real callers (package A1.3).

Every cell of the test table is a unique marker (``Q7Kx…``). Each caller runs with a scripted model that keeps every
prompt it receives; the test counts how many distinct markers reached each purpose and compares the count with the
purpose's limit in dclab_rnd/models/settings.py PURPOSES (cell_values). A purpose with limit 0 must never see one.
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dclab_rnd import models  # noqa: E402
from dclab_rnd.models import FileUsage, Gateway, settings  # noqa: E402

MARK = re.compile(r"Q7Kx[0-9a-z]+")
NUM = re.compile(r"7311\.\d{4}\b|2031-\d\d-\d\d")  # numeric and date cells (matched against the table's own values)
CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "DCLAB_LOG_PROMPTS", "DCLAB_INTERN_MODEL", "DCLAB_NO_LIVE_MODELS")} | {
    f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.TIERS for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")} | {"OPENAI_API_KEY": "k"}


def marked_table(rows=300, seed=0):
    """Categorical columns of unique markers, a numeric column, and a binary outcome."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "segment": [f"Q7Kxseg{i % 7}" for i in range(rows)],          # 7 distinct values
        "city": [f"Q7Kxcity{i % 40}" for i in range(rows)],            # 40 distinct values
        "note": [f"Q7Kxnote{i}" for i in range(rows)],                 # unique per row
        "spend": [round(7311.0001 + (i * 37 % rows) / 1e4, 4) for i in range(rows)],  # unique, findable numbers
        "signup": pd.to_datetime("2031-01-01") + pd.to_timedelta([i % 200 for i in range(rows)], unit="D"),  # findable dates
    })
    frame["churned"] = (rng.random(rows) < 0.3).astype(int)
    return frame


class Capture:
    """A scripted transport that keeps every prompt, per purpose."""
    prompts: dict[str, list[str]] = {}

    def __init__(self, purpose, replies):
        self.purpose, self.replies = purpose, list(replies)

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        Capture.prompts.setdefault(self.purpose, []).append(json.dumps(messages, default=str))
        return self.replies.pop(0) if self.replies else {"content": "Noted.", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0}, "assistant_message": {"role": "assistant", "content": "Noted."}}

    def embed(self, texts):  # 35.5: an embedding request, kept like a prompt
        Capture.prompts.setdefault(self.purpose, []).append(json.dumps(list(texts), default=str))
        return {"vectors": [[float(len(t) % 7) + 1.0, 1.0] for t in texts], "usage": {"input_tokens": 0, "output_tokens": 0}}


def gateway(replies_by_purpose=None):
    replies_by_purpose = replies_by_purpose or {}

    def transport(tier, purpose):
        name = next(k for k, v in settings.PURPOSES.items() if v is purpose)
        return Capture(name, replies_by_purpose.get(name, []))
    return Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=transport)


def tool_call(name, arguments, n=1):
    return {"content": "", "tool_calls": [{"id": f"c{n}", "name": name, "arguments": arguments}], "usage": {"input_tokens": 0, "output_tokens": 0},
            "assistant_message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}


class DataLimitTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, CLEAN)
        env.start()
        self.addCleanup(env.stop)
        Capture.prompts = {}

    CELLS = None

    def seen(self, purpose):
        """Distinct cell values of the table found in the purpose's prompts: text markers, and numbers or dates equal to a cell."""
        if DataLimitTests.CELLS is None:
            t = marked_table()
            DataLimitTests.CELLS = {f"{v:.4f}" for v in t["spend"]} | {str(d.date()) for d in t["signup"]}
        prompts = Capture.prompts.get(purpose, [])
        return {m for p in prompts for m in MARK.findall(p)} | {m for p in prompts for m in NUM.findall(p) if m in DataLimitTests.CELLS}

    def assertWithinLimit(self, purpose):
        seen = self.seen(purpose)
        self.assertTrue(Capture.prompts.get(purpose), f"{purpose} was never asked: the test did not exercise it")
        self.assertLessEqual(len(seen), settings.PURPOSES[purpose].cell_values, f"{purpose} saw {len(seen)} cell values: {sorted(seen)[:8]}")
        return seen

    def test_home_agent_sees_summaries_never_values(self):
        from dclab_rnd.draft import analyze
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Predict which customers churn next month")
        report = analyze.analyze(marked_table())
        store.update(d["id"], lambda x: x.update(analysis=report, assets=[{"id": "a1", "status": "ready"}], active_asset="a1"))
        gw = gateway({"home_agent": [tool_call("get_profile", {}, 1), tool_call("get_analysis", {}, 2), tool_call("get_profile", {"columns": ["city", "note"]}, 3)]})
        HomeAgent(store, gw.client("home_agent", draft_id=d["id"])).reply(d["id"], "Customers who churn within 30 days")
        self.assertEqual(self.assertWithinLimit("home_agent"), set())

    def test_home_agent_sees_no_value_of_an_answer_it_asked_for(self):  # the review of H7a..H12
        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.chat import HomeAgent, Turn, query_data
        from dclab_rnd.draft.store import DraftStore
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("What was the total spend per city?")
        agent = HomeAgent(store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        marked_table().to_csv(store.data_dir(d["id"]) / "marked.csv", index=False)
        asset = pipeline.new_asset(store, d["id"], "upload", "marked.csv", "marked.csv")
        pipeline.run(store, d["id"], asset["id"], agent, None)
        query_data(Turn(agent, d["id"]), plan={"group_by": ["segment"], "measures": [{"name": "spend", "op": "sum", "column": "spend"}]})
        shown = [m for m in store.get(d["id"])["messages"] if m.get("kind") == "answer"]
        self.assertTrue(shown and any("Q7Kxseg" in m["text"] for m in shown))  # the person sees the segments
        gw = gateway({"home_agent": [{"content": "Here it is.", "tool_calls": [], "usage": {}}]})
        HomeAgent(store, gw.client("home_agent", draft_id=d["id"]), on_request=lambda *a: None).reply(d["id"], "Thanks, and what next?")
        self.assertEqual(self.assertWithinLimit("home_agent"), set())  # the model's history: what was computed, never a value

    def test_home_intent_sees_the_request_never_values(self):  # H2
        from dclab_rnd.draft import analyze
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Help with my customers")  # the rules find no kind of help, so the model reads it
        report = analyze.analyze(marked_table())
        store.update(d["id"], lambda x: x.update(analysis=report, assets=[{"id": "a1", "status": "ready"}], active_asset="a1"))
        gw = gateway({"home_intent": [{"content": '{"tracks": ["analysis"], "quote": "my customers", "confidence": 0.6}', "tool_calls": [], "usage": {}}]})
        HomeAgent(store, None, intent_client=gw.client("home_intent", draft_id=d["id"])).start(d["id"])
        self.assertEqual(self.assertWithinLimit("home_intent"), set())
        self.assertNotIn("segment", Capture.prompts["home_intent"][0])  # no column names either

    def test_home_query_sees_column_summaries_never_values(self):  # H6
        from dclab_rnd import home_query
        from dclab_rnd.draft import analyze
        from dclab_rnd.draft.store import DraftStore
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("What is the average spend per segment?")
        report = analyze.analyze(marked_table())
        store.update(d["id"], lambda x: x.update(analysis=report))
        plan = '{"group_by": ["segment"], "measures": [{"name": "avg_spend", "op": "mean", "column": "spend"}]}'
        gw = gateway({"home_query": [{"content": plan, "tool_calls": [], "usage": {}}]})
        got, why = home_query.write_plan(gw.client("home_query", draft_id=d["id"]), "What is the average spend per segment?", store.get(d["id"]))
        self.assertEqual((got.group_by, why), (["segment"], ""))
        self.assertEqual(self.assertWithinLimit("home_query"), set())
        self.assertIn("segment", Capture.prompts["home_query"][0])  # the column names, never their values

    def test_home_business_sees_the_template_brief_never_a_row(self):  # H11b
        from dclab_core import change as C
        from dclab_rnd.draft import business
        t = marked_table()
        r = C.compare(t, date_column="signup", a=C.Window(start="2031-01-01", end="2031-01-31", label="January 2031"),
                      b=C.Window(start="2031-02-01", end="2031-02-28", label="February 2031"), by="segment", measure="spend")
        template = business.brief(r)
        gw = gateway({"home_business": [{"content": "Spend changed between the two months [CHANGE].", "tool_calls": [], "usage": {}}]})
        text, how = business.reword(gw.client("home_business"), r, template)
        self.assertIn("association, not cause", text)  # the code's caveat, always
        self.assertEqual(self.assertWithinLimit("home_business") - {s.segment for s in r.segments}, set())  # group names only
        self.assertNotIn("Q7Kxnote", Capture.prompts["home_business"][0])  # never a row's own text

    def test_parse_pattern_sees_at_most_twenty_lines(self):
        from dclab_rnd.draft import structure
        path = Path(tempfile.mkdtemp()) / "odd.log"
        path.write_text("\n".join(f"<<Q7Kxa{i}>>~~[Q7Kxb{i}]~~{{Q7Kxc{i}}}" for i in range(200)))
        gw = gateway({"parse_pattern": [{"content": "{}", "tool_calls": [], "usage": {}}]})
        structure.to_table(path, client=gw.client("parse_pattern"))
        seen = self.assertWithinLimit("parse_pattern")
        self.assertLessEqual(len({m[:6] + re.sub(r"\D", "", m) for m in seen}), structure.MODEL_LINES * 3)  # at most 20 lines' worth

    def test_synthetic_schema_sees_no_data(self):
        from dclab_rnd.draft import synthetic
        gw = gateway({"synthetic_schema": [{"content": "not json", "usage": {}}, {"content": "still not json", "usage": {}}]})
        synthetic.spec_from_model(gw.client("synthetic_schema"), "subscribers who cancel", {"problem": "Predict churn", "understanding": {"target": "churned"}}, 500)
        self.assertEqual(self.assertWithinLimit("synthetic_schema"), set())

    def test_evidence_answers_see_no_project_data(self):
        from dclab_rnd.agentic.pages import evidence
        gw = gateway()
        evidence.ask("Should I oversample before splitting?", gw.client("evidence_answer"))
        self.assertEqual(self.assertWithinLimit("evidence_answer"), set())

    def _project(self):
        from dclab_rnd.studio import ProjectStore, data
        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("Marked", "general", "Predict churn")["id"]
        marked_table().to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        p = store.get(pid)
        p["solution"] = {"target": "churned", "task": "binary", "positive_label": "1", "prediction_moment": "At the monthly snapshot of each customer.",
                         "forbidden": [], "identifiers": ["note"], "time_column": None, "group_column": None, "text_columns": [], "metric": "roc_auc", "notes": ""}
        p["settings"]["quick"] = True
        store.save(p)
        return store, pid

    def test_stage_notes_and_project_answers_see_aggregates_not_values(self):
        from dclab_rnd.studio import agent, engine
        store, pid = self._project()
        models.install(gateway())
        self.addCleanup(models.install, None)
        for stage in ("data", "leakage", "features", "models", "final"):  # every stage's notes
            engine.execute(store, pid, stage, "human")
        agent.answer("Which model is best, and is there a leak?", store.get(pid), store.records(pid), "binary")
        from dclab_rnd import lessons
        lessons.propose(store.get(pid), store.read_stage(pid, "final"), models.installed().client("lesson_proposal", project_id=pid))  # A5.3
        for purpose in ("stage_notes", "project_answer", "lesson_proposal"):
            self.assertEqual(self.assertWithinLimit(purpose), set(), purpose)

    def test_the_home_orchestrator_sees_the_records_never_a_value(self):
        from dclab_rnd.draft import orchestrator
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.studio import engine
        store, pid = self._project()
        models.install(gateway())
        self.addCleanup(models.install, None)
        for stage in ("data", "leakage", "features", "models", "final"):
            engine.execute(store, pid, stage, "human")
        store.update(pid, memory=[orchestrator._new_note("preference", "we prefer fewer false alarms", "the person asked", "person", "remember")])  # its card is checked too
        names, script = list(orchestrator.READS), []
        for start in range(0, len(names), 4):  # a turn makes at most a few calls (five replies): every read tool over a few turns
            script += [tool_call(n, {}, i) for i, n in enumerate(names[start:start + 4], 1)] + [{"content": "Here they are.", "tool_calls": [], "usage": {}}]
        shared = Capture("home_orchestrator", script)  # one script across the turns, so every read tool is called once
        gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"),
                     transport=lambda tier, purpose: shared if purpose is settings.PURPOSES["home_orchestrator"] else Capture("other", []))
        drafts = DraftStore(Path(tempfile.mkdtemp()))
        did = drafts.create("Predict churn")["id"]
        drafts.update(did, lambda d: d.update(status="built", project_id=pid))
        home = HomeAgent(drafts, None)
        services = type("S", (), {"projects": store, "gateway": gw, "store": type("H", (), {"home": Path(tempfile.mkdtemp())})()})()
        with mock.patch.dict(os.environ, {"DCLAB_BUILT_CHAT_SERVE": "model"}):  # the model's prompts are what this test reads
            for _ in range(0, len(names), 4):
                home.reply_built(did, "show me everything about the models, the data and the leakage", orchestrator.Orchestrator(services, home), pid)
        shown = {m["card"]["title"] for m in drafts.get(did)["messages"] if m["kind"] == "project_card"}
        self.assertGreaterEqual(len(shown), len(names) - 2, shown)  # every read tool's card (the notebook's and the improvement run's lists are empty here)
        self.assertEqual(self.assertWithinLimit("home_orchestrator"), set())

    def test_the_reporter_in_the_chat_sends_the_model_no_value_on_a_regression_project(self):  # the review: segment names and the outcome's minimum
        from dclab_rnd.draft import orchestrator
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.studio import ProjectStore, data, engine
        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("Marked", "general", "Predict the amount")["id"]
        table = marked_table().rename(columns={"spend": "amount"}).drop(columns=["churned"])
        table.to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        p = store.get(pid)
        p["solution"] = {"target": "amount", "task": "regression", "positive_label": None, "prediction_moment": "At the monthly snapshot.", "forbidden": [],
                         "identifiers": ["note"], "time_column": None, "group_column": None, "text_columns": [], "metric": "mae", "notes": ""}
        p["settings"]["quick"] = True
        store.save(p)
        models.install(gateway())
        self.addCleanup(models.install, None)
        for stage in ("data", "leakage", "features", "models", "final"):
            engine.execute(store, pid, stage, "human")
        shared = Capture("home_orchestrator", [tool_call("write_report", {"audience": "team"}, 1), {"content": "Here it is.", "tool_calls": [], "usage": {}},
                                               tool_call("write_report", {"audience": "manager"}, 2), {"content": "Here it is.", "tool_calls": [], "usage": {}}])
        gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"),
                     transport=lambda tier, purpose: shared if purpose is settings.PURPOSES["home_orchestrator"] else Capture("other", []))
        drafts = DraftStore(Path(tempfile.mkdtemp()))
        did = drafts.create("Predict the amount")["id"]
        drafts.update(did, lambda d: d.update(status="built", project_id=pid))
        home = HomeAgent(drafts, None)
        services = type("S", (), {"projects": store, "gateway": gw, "store": type("H", (), {"home": Path(tempfile.mkdtemp())})()})()
        with mock.patch.dict(os.environ, {"DCLAB_BUILT_CHAT_SERVE": "model"}):
            home.reply_built(did, "write the report for the team", orchestrator.Orchestrator(services, home), pid)
            home.reply_built(did, "now the report for my manager", orchestrator.Orchestrator(services, home), pid)
        titles = {m["card"]["title"] for m in drafts.get(did)["messages"] if m["kind"] == "project_card"}
        self.assertTrue({"Report for the team", "Report for your manager"} <= titles, titles)
        self.assertEqual(self.assertWithinLimit("home_orchestrator"), set())

    def test_the_notebook_cell_writer_sees_the_code_and_column_names_never_values(self):
        from dclab_rnd.studio import notebook_agent
        store, pid = self._project()
        reply = {"content": json.dumps({"code": "import matplotlib.pyplot as plt\nplt.plot(fold_scores)\nplt.show()", "explanation": "folds"}),
                 "tool_calls": [], "usage": {}}
        gw = gateway({"notebook_cell": [reply]})
        cell = notebook_agent.write(store.get(pid), store.records(pid), "plot the score of each fold", gw.client("notebook_cell", project_id=pid))
        self.assertEqual(cell["status"], "proposed")
        self.assertEqual(self.assertWithinLimit("notebook_cell"), set())

    def test_the_intern_sees_at_most_three_examples_per_column(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        store, pid = self._project()
        gw = gateway({"intern": [tool_call("describe_data", {"project_id": pid}, 1), tool_call("finish", {"report": "done"}, 2)]})
        intern = Intern(SessionStore(Path(tempfile.mkdtemp())), Toolbox(store), gw.client("intern"))
        s = intern.start("Describe the data", project_id=pid)
        intern.run(s["id"])
        seen = self.assertWithinLimit("intern")
        per_column = {}
        for m in seen:
            column = "spend" if m.startswith("7311.") else "signup" if m.startswith("2031-") else re.sub(r"\d+$", "", m)
            per_column.setdefault(column, set()).add(m)
        self.assertTrue(per_column)
        self.assertTrue(all(len(v) <= 3 for v in per_column.values()), per_column)

    def test_the_leakage_reviewer_sees_names_and_summaries_never_values(self):
        from dclab_rnd.studio import data, leakage_review, solution
        table = marked_table()
        profile = data.profile_table(table)  # its previews hold markers: they must not reach the model
        proposal = solution.propose(table, profile, "churned")
        gw = gateway({"leakage_review": [{"content": "{\"columns\": []}", "tool_calls": [], "usage": {}}]})
        leakage_review.review(profile, "At the monthly snapshot of each customer.", proposal["forbidden"], "churned", proposal["identifiers"],
                              gw.client("leakage_review"))
        self.assertEqual(self.assertWithinLimit("leakage_review"), set())

    def test_the_notebook_reviewer_sees_findings_and_records_never_code_or_data(self):
        import json as _json
        from dclab_rnd.copilot import fixes, review_notebook
        nb = {"cells": [
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": ["import pandas as pd\n", "from sklearn.preprocessing import StandardScaler\n",
                                                                                                   "from sklearn.model_selection import train_test_split\n"]},
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
             "source": ["secret = 'Q7Kxcode1'  # a value in the user's code\n", "df = pd.read_csv('/Users/alice/clients/Q7KxAcme/customers_Q7Kxfile.csv')\n",
                        "feat_cols = ['age', 'final Q7Kxcolumn', 'final_Q7Kxlist']\n", "y = df['label Q7Kxtarget']\n", "closed = df[df['status'] == 'closed_Q7Kxvalue']\n",
                        "z = pd.read_csv('/Users/bob/`Q7Kxtick`/z.csv')\n", "w = df['final_Q7Kxcol']\n", "X = StandardScaler().fit_transform(df[feat_cols])\n",
                        "X_train, X_test = train_test_split(X)\n"]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        path = Path(tempfile.mkdtemp()) / "nb.ipynb"
        path.write_text(_json.dumps(nb), encoding="utf-8")
        report = review_notebook(path)
        self.assertIn("absolute_data_path", [f["detector"] for f in report["findings"]])  # the detectors that quote a path and a selector fire
        self.assertTrue(any("Q7Kx" in f["message"] + f["title"] + f["suggestion"] for f in report["findings"]))  # and do carry the literal
        gw = gateway({"notebook_review": [{"content": "{\"fixes\": []}", "tool_calls": [], "usage": {}}]})
        fixes.write_fixes(report["findings"], gw.client("notebook_review"))
        self.assertEqual(self.assertWithinLimit("notebook_review"), set())  # no cell value, and none of the code's literals
        self.assertFalse([p for p in Capture.prompts["notebook_review"] if "Q7Kx" in p or "read_csv" in p])

    def test_the_problem_analyst_sees_the_description_and_column_names_never_values(self):
        from dclab_core.understand import understand
        from dclab_rnd.core_advisor import GatewayAdvisor
        table = marked_table()
        gw = gateway({"problem_analyst": [{"content": "{}", "tool_calls": [], "usage": {}}]})
        understand("Predict whether a customer churns. The table has one row per customer.", {"customers.csv": list(table.columns)},
                   advisor=GatewayAdvisor(gw))
        self.assertEqual(self.assertWithinLimit("problem_analyst"), set())

    def test_the_conversation_reader_sees_one_answer_and_column_names_never_values(self):
        from dclab_core.agents import run_role
        from dclab_core.conversation import READER
        from dclab_rnd.core_advisor import GatewayAdvisor
        table = marked_table()
        gw = gateway({"conversation_reader": [{"content": "{}", "tool_calls": [], "usage": {}}]})
        run_role(READER, GatewayAdvisor(gw), {"field": "labels", "answer": "nothing has been confirmed yet", "columns": list(table.columns)})
        self.assertEqual(self.assertWithinLimit("conversation_reader"), set())

    def test_the_map_roles_see_the_words_the_map_and_column_names_never_values(self):  # 24.2
        from dclab_core import mapping, problem_map as pm
        from dclab_core.datacard import card_of
        from dclab_rnd.core_advisor import GatewayAdvisor

        table = marked_table()
        data = card_of({"customers": table}, name="marked", keep_categories=True)  # even a card that kept its categories: the roles are sent names and purposes only
        words = "Find the fake rides our bikers make. The team can review 200 a day."
        empty = {"content": "{}", "tool_calls": [], "usage": {}}
        gw = gateway({"decomposer": [empty], "map_reviewer": [empty], "data_need": [empty]})
        advisor = GatewayAdvisor(gw)
        made, _, _ = mapping.draw(words, advisor=advisor, data=data)
        mapping.review_with(pm.decompose(words), advisor=advisor, words=words, data=data)
        mapping.needs_for(pm.decompose(words), data, advisor)
        from dclab_core import solutions

        m = pm.decompose(words)
        solutions.route_with(m.sub_problems[1], data, m, advisor=GatewayAdvisor(gateway({"solution_router": [empty]})))  # 24.3: lines about the candidates, never values
        for purpose in ("decomposer", "map_reviewer", "data_need", "solution_router"):
            self.assertEqual(self.assertWithinLimit(purpose), set(), purpose)

    def test_the_understanding_critic_sees_the_card_and_column_names_never_values(self):  # 28.4
        from dclab_core.datacard import card_of
        from dclab_core.understanding_critic import critique
        from dclab_rnd.core_advisor import GatewayAdvisor

        table = marked_table()
        data = card_of({"customers": table}, name="marked", keep_categories=True)  # even a card that kept its categories: lines of names and bands
        columns = [c for t in json.loads(data.model_dump_json())["tables"] for c in t["columns"]]
        gw = gateway({"understanding_critic": [{"content": '{"concerns": []}', "tool_calls": [], "usage": {}}]})
        critique(GatewayAdvisor(gw), {"family": "classification", "outcome": "churned", "moment": "at signup"}, columns,
                 [{"code": "time_absent", "question": "When is the prediction made?"}])
        self.assertEqual(self.assertWithinLimit("understanding_critic"), set())

    def test_the_reflector_sees_codes_and_step_verdicts_never_a_name_or_a_value(self):  # the harness that learns, H2
        from dclab_core.challenge_pack import ChallengeKey
        from dclab_rnd.learning.reflect import reflect

        table = marked_table()
        names = list(table.columns)
        key = ChallengeKey(pack="pk-000000000001", family="groups", seed=1, partition="learning", leaks=[names[0]], group_column=names[1],
                           row_key=names[-1], known_before=names[2:4])
        row = {"pack": "pk-000000000001", "arms": {"rules": {"verdict": "fail", "holdout": 0.7, "codes": ["SPLIT_WRONG:repeated_entity"],
                                                            "checks": [{"trap": "repeated_entity", "ok": False, "code": "SPLIT_WRONG:repeated_entity"}]}}}
        gw = gateway({"reflector": [{"content": '{"lessons": []}', "tool_calls": [], "usage": {}}]})
        reflect(row, key, names, client=gw.client("reflector"), cap_eur=0.05)
        self.assertEqual(self.assertWithinLimit("reflector"), set())
        sent = " ".join(Capture.prompts.get("reflector", []))
        self.assertFalse([n for n in names if n in sent], "the reflector was sent a column's name")

    def test_the_decision_answerer_sees_bands_and_names_never_values(self):  # 22.5
        from dclab_core.answerers import ModelAnswerer
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.ledger import column_state
        from dclab_rnd.studio.data import profile_table

        table = marked_table()
        gw = gateway({"decision_answer": [{"content": '{"answer": "no", "reason": "a band"}', "tool_calls": [], "usage": {}}] * len(table.columns)})
        answerer = ModelAnswerer(GatewayAdvisor(gw), {"column.is_identifier"})
        for column in profile_table(table)["columns"]:
            answerer.answer("column.is_identifier", column_state(column, len(table)), ["yes", "no"])
        self.assertEqual(self.assertWithinLimit("decision_answer"), set())

    def test_the_decision_answerer_sees_a_tables_counts_never_its_values(self):  # 29.9: points about a table, not a column
        from dclab_core.answerers import ModelAnswerer
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.ledger import states
        from dclab_rnd.studio.data import profile_table

        table = marked_table()
        profile = profile_table(table)
        gw = gateway({"decision_answer": [{"content": '{"answer": "random", "reason": "a count"}', "tool_calls": [], "usage": {}}] * 2})
        answerer = ModelAnswerer(GatewayAdvisor(gw), {"split.strategy", "data.cases_table"})
        answerer.answer("split.strategy", states.split_state(profile, None, None), ["random", "time", "group", "time_and_group"])
        answerer.answer("data.cases_table", states.cases_state(profile, "customers.csv"), ["yes", "no"])
        self.assertEqual(self.assertWithinLimit("decision_answer"), set())

    def test_the_solver_sees_bands_and_names_never_values(self):  # 27.6: the benchmark's questions, built as self-play builds them
        from dclab_rnd.lake.selfplay import _ask, _columns

        table = marked_table()
        gw = gateway({"solver_answer": [{"content": '{"task": "binary"}', "tool_calls": [], "usage": {}}] * 2})
        client = gw.client("solver_answer")
        for masked in (False, True):
            columns, _ = _columns(table, "outcome", masked)
            messages = _ask("What kind of machine-learning problem is this?", {"columns": columns, "outcome": "outcome", "prediction_moment": "at month end"}, {})
            client.complete(messages[:2], None, 200)
        self.assertEqual(self.assertWithinLimit("solver_answer"), set())

    def test_the_labeller_sees_the_blind_view_bands_and_names_never_values(self):  # 19.8b: Claude helps label
        from dclab_rnd import ledger
        from dclab_rnd.ledger import column_state, model_labels
        from dclab_rnd.studio.data import profile_table

        table = marked_table()
        book = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "w1")
        for column in profile_table(table)["columns"] * 20:  # enough decisions that some are drawn
            ledger.decide("column.is_identifier", ("column", column["name"]), answers=[ledger.answer("rule", "r", False)], value_used=False,
                          state=column_state(column, len(table)), ledger=book)
        gw = gateway({"decision_labeler": [{"content": '{"answer": "no", "confidence": "low", "reason": "a band"}', "tool_calls": [], "usage": {}}] * 200})
        ask, _ = model_labels.asker(gw, cap_eur=1.0)
        self.assertGreater(model_labels.to_label(book, ask, "scripted")["asked"], 0)
        self.assertEqual(self.assertWithinLimit("decision_labeler"), set())

    def test_the_data_investigator_sees_column_names_and_shares_never_values(self):
        from dclab_core.analyse import analyse
        from dclab_rnd.core_advisor import GatewayAdvisor
        table = marked_table()
        gw = gateway({"data_investigator": [{"content": '{"after": [{"column": "note", "why": "a note written later"}]}', "tool_calls": [], "usage": {}}]})
        analyse({"customers": table}, target="churned", task="binary", moment="At the monthly snapshot.", advisor=GatewayAdvisor(gw))
        self.assertEqual(self.assertWithinLimit("data_investigator"), set())

    def test_the_brief_checker_sees_the_description_and_the_rules_reading_never_values(self):
        from dclab_core.understand import understand
        from dclab_rnd.core_advisor import GatewayAdvisor
        table = marked_table()
        gw = gateway({"brief_checker": [{"content": '{"disagree": []}', "tool_calls": [], "usage": {}}]})
        understand("Predict whether a customer churns. The table has one row per customer. The churned column is the label.",
                   {"customers.csv": list(table.columns)}, checker=GatewayAdvisor(gw))
        self.assertEqual(self.assertWithinLimit("brief_checker"), set())

    def test_the_feature_engineer_sees_column_names_and_shares_never_values(self):
        from dclab_core.features import Language, engineer
        from dclab_rnd.core_advisor import GatewayAdvisor
        table = marked_table()
        gw = gateway({"feature_engineer": [{"content": '{"programs": []}', "tool_calls": [], "usage": {}}]})
        engineer(GatewayAdvisor(gw), table, Language([c for c in table.columns if c != "churned"], target="churned"), moment="At signup.")
        self.assertEqual(self.assertWithinLimit("feature_engineer"), set())

    def test_the_planner_and_the_critic_see_summaries_never_values(self):
        from dclab_core.analyse import analyse
        from dclab_core.plan import plan_with
        from dclab_core.review import critique, review
        from dclab_rnd.core_advisor import GatewayAdvisor
        table = marked_table()
        findings = analyse({"customers": table}, target="churned", task="binary")
        gw = gateway({"solution_planner": [{"content": '{"decisions": []}', "tool_calls": [], "usage": {}}],
                      "critic": [{"content": '{"hypotheses": []}', "tool_calls": [], "usage": {}}]})
        plan_with(GatewayAdvisor(gw), None, findings)
        y = table["churned"].to_numpy()
        critique(GatewayAdvisor(gw), review("binary", y, np.full(len(y), 0.3), frame=table.drop(columns="churned")), "binary")
        self.assertEqual(self.assertWithinLimit("solution_planner"), set())
        self.assertEqual(self.assertWithinLimit("critic"), set())

    def test_the_debates_challenger_and_judge_see_summaries_never_values(self):  # 28.6
        import json as _json

        from dclab_core.analyse import analyse
        from dclab_core.debate import debate
        from dclab_rnd.core_advisor import GatewayAdvisor
        table = marked_table()
        findings = analyse({"customers": table}, target="churned", task="binary")
        cite = findings.findings[0].id
        # threshold.policy, not metric.primary: since 2026-10-30 an advisor may not replace the primary metric, so a primary
        # change would be refused before any dispute and the judge would never be asked
        change = {"key": "threshold.policy", "choice": {"kind": "review_budget"}, "reason": "the team can review a fixed number of flags each day", "cites": [cite]}
        gw = gateway({"solution_planner": [{"content": _json.dumps({"decisions": [change]}), "tool_calls": [], "usage": {}}],
                      "solution_challenger": [{"content": _json.dumps({"decisions": [{**change, "choice": {"kind": "expected_cost"}, "reason": "the costs of a miss and of a false alarm decide it here"}]}),
                                               "tool_calls": [], "usage": {}}],  # a different choice: a dispute, so the judge is asked
                      "debate_judge": [{"content": _json.dumps({"rulings": [{"key": "threshold.policy", "keep": "a", "reason": "the first design's policy fits the use",
                                                                             "cites": [cite]}]}), "tool_calls": [], "usage": {}}]})
        out = debate(GatewayAdvisor(gw), GatewayAdvisor(gw), GatewayAdvisor(gw), None, findings)
        self.assertEqual(out["debate"].sources["judge"], "advisor")  # the judge was asked: its request is checked below
        self.assertEqual(self.assertWithinLimit("solution_challenger"), set())
        self.assertEqual(self.assertWithinLimit("debate_judge"), set())

    def test_the_reporter_sees_the_reports_lines_never_values(self):
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine, solution_report
        from dclab_rnd.studio.solution import Solution
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid = store.create("Marked", "general", "Predict which customers churn")["id"]
        marked_table().to_csv(store.data_dir(pid) / "t.csv", index=False)
        sd.attach_data(store, pid, "t.csv")
        project = store.get(pid)
        project["solution"] = Solution(target="churned", task="binary", identifiers=["note"],
                                       prediction_moment="At signup, before the first month ends.").model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        with runner.fast_profile(), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None), \
                mock.patch("dclab_rnd.studio.agent.llm_available", return_value=False):
            engine.run_all(store, pid)
        gw = gateway({"reporter": [{"content": '{"summaries": []}', "tool_calls": [], "usage": {}}]})
        solution_report.build(store, pid, analyse=False, advisor=GatewayAdvisor(gw))
        self.assertEqual(self.assertWithinLimit("reporter"), set())

    def test_the_reporter_may_see_a_multiclass_targets_classes_never_an_inputs_values(self):
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine, solution_report
        from dclab_rnd.studio.solution import Solution
        table = marked_table(rows=400).drop(columns="churned")
        table["plan_kind"] = [f"Q7Kxcls{i % 4}" if i % 50 else "Q7Kxrare" for i in range(len(table))]
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid = store.create("Marked classes", "general", "Predict the plan kind")["id"]
        table.to_csv(store.data_dir(pid) / "t.csv", index=False)
        sd.attach_data(store, pid, "t.csv")
        project = store.get(pid)
        project["solution"] = Solution(target="plan_kind", task="multiclass", identifiers=["note"],
                                       prediction_moment="At signup, before the plan is chosen.").model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        with runner.fast_profile(), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None), \
                mock.patch("dclab_rnd.studio.agent.llm_available", return_value=False):
            engine.run_all(store, pid)
        gw = gateway({"reporter": [{"content": '{"summaries": []}', "tool_calls": [], "usage": {}}]})
        solution_report.build(store, pid, analyse=True, advisor=GatewayAdvisor(gw))
        seen = self.assertWithinLimit("reporter")
        self.assertFalse({m for m in seen if not m.startswith(("Q7Kxcls", "Q7Kxrare"))}, seen)  # the target's classes only

    def test_the_ai_first_expert_sees_the_profile_never_values(self):  # 25.3: the twin run's expert record of the problem
        from dclab_rnd.twin.claude import Transcript, TwinGateway
        from dclab_rnd.twin.design import Designer
        from dclab_rnd.twin.lane import Lane

        home = Path(tempfile.mkdtemp())
        sent = []

        class Recording:
            model, base_url = "claude-test", "https://api.anthropic.com"

            def complete(self, messages, tools=None, max_tokens=None, **options):
                sent.extend(str(m.get("content")) for m in messages)
                return {"content": "{}", "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}}

        gw = TwinGateway(Recording(), Transcript(home / "t.jsonl"))
        lane = Lane("ai_first", home / "ai_first", gw)
        pid = lane.projects.create("Marked", "general", "Predict churn")["id"]
        marked_table().to_csv(lane.projects.data_dir(pid) / "t.csv", index=False)
        from dclab_rnd.studio import data as sd

        sd.attach_data(lane.projects, pid, "t.csv")
        lane.project_id = pid
        Designer(lane, gw).initial()
        self.assertTrue(sent, "the expert was never asked")
        seen = {m for p in sent for m in MARK.findall(p)} | {m for p in sent for m in NUM.findall(p) if m in DataLimitTests.CELLS}
        self.assertEqual(seen, set())  # the profile's names and counts only: no example value reaches the expert's record
        self.assertGreaterEqual(settings.PURPOSES["ai_first"].cell_values, 0)

    def test_the_metric_chooser_sees_the_dimensions_names_never_a_number_or_another_value(self):  # 30.20
        from dclab_core import metric_change as mc
        from dclab_rnd.core_advisor import GatewayAdvisor

        table = marked_table()
        layer = mc.SemanticLayer(tables=[mc.Table(name="t", date="signup")], metrics=[mc.Metric(name="spend", label="Spend", kind="sum", table="t", column="spend")],
                                 dimensions=[mc.Dimension(name="segment", label="Segment", columns={"t": "segment"}),
                                             mc.Dimension(name="city", label="City", columns={"t": "city"})])
        gw = gateway({"metric_question": [{"content": '{"refused": true, "missing": ["x"], "reason": "a test"}', "tool_calls": [], "usage": {}}]})
        mc.ask("How come spend in Q7Kxcity3 moved this spring?", layer, {"t": table}, GatewayAdvisor(gw))
        seen = self.assertWithinLimit("metric_question")
        self.assertTrue(seen)  # the dimensions' values, so the model can name one
        self.assertFalse({m for m in seen if m.startswith("Q7Kxnote")})  # a column no dimension names is never sent
        self.assertFalse({m for m in seen if m in {f"{v:.4f}" for v in table["spend"]}})  # never a number of the table
        days = {m for m in seen if m in {str(d.date()) for d in table["signup"]}}
        self.assertLessEqual(days, {str(table["signup"].min().date()), str(table["signup"].max().date())})  # the data's first and last day only

    def test_the_workspace_chat_sees_aggregates_and_names_never_a_row(self):  # 30.21
        from fastapi.testclient import TestClient

        from dclab_rnd import workspace_chat
        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.studio import data as sd

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        s = client.app.state.services
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "t", "goal": "g"}, headers=h).json()["id"]
        marked_table().to_csv(s.projects.data_dir(pid) / "t.csv", index=False)
        sd.attach_data(s.projects, pid, "t.csv")
        args = {"table": f"{pid}:t", "op": "sum", "column": "spend", "group_by": ["segment"]}  # a sum, never equal to one row's value (a mean of evenly spaced values is one)
        gw = gateway({"workspace_chat": [tool_call("list_tables", {}, 1), tool_call("describe_table", {"table": f"{pid}:t"}, 2), tool_call("compute", args, 3),
                                         {"content": "Means by segment as computed.", "tool_calls": [], "usage": {}}]})
        workspace_chat.ask(s, "Mean spend by segment?", gateway=gw)
        seen = self.assertWithinLimit("workspace_chat")
        self.assertFalse({m for m in seen if m.startswith(("Q7Kxnote", "Q7Kxcity"))})  # a column neither grouped nor shown: never a value of it
        self.assertFalse({m for m in seen if m in {f"{v:.4f}" for v in marked_table()["spend"]}})  # never a row's number, only the means

    def test_the_store_curator_sees_a_records_shape_never_its_values(self):  # 32.9
        from dclab_core.store import Provenance, StoreRecord
        from dclab_rnd.store.curators import model

        t = marked_table()
        row = {c: str(t[c].iloc[0]) for c in t.columns}
        prov = Provenance(entry="e", sha256="0" * 64, source="s", workspace="w", at="2026-10-20T10:00:00+00:00")
        record = StoreRecord(id="r1", kind="decisions", type="decision", at="2026-10-20T10:00:00+00:00", provenance=prov,
                             body={"point": "column.is_identifier", "used": "rule", "row": row, "reason": f"see {row}", "group": "project:p1"})
        gw = gateway({"store_curator": [{"content": "[]", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0},
                                         "assistant_message": {"role": "assistant", "content": "[]"}}]})
        model.ask([record], gateway=gw)
        self.assertEqual(self.assertWithinLimit("store_curator"), set())
        self.assertFalse(any(PERSONAL.search(p) for p in Capture.prompts.get("store_curator", [])))

    def test_the_reasoning_teacher_sees_a_scrubbed_record_never_a_row(self):  # 32.10
        from dclab_core.store import Provenance, StoreRecord
        from dclab_rnd.store import reasoning, shapers

        t = marked_table()
        row = {c: str(t[c].iloc[0]) for c in t.columns}
        body = shapers._clean({"point": "column.is_identifier", "preview": list(row.values()), "top": list(row.values()),
                               "reason": f"mail {row.get('email', '')}", "used": "rule"})
        prov = Provenance(entry="e", sha256="0" * 64, source="s", workspace="w", at="2026-10-20T10:00:00+00:00", consent="train")
        record = StoreRecord(id="r1", kind="decisions", type="decision", at="2026-10-20T10:00:00+00:00", provenance=prov, body=body)
        gw = gateway({"reasoning_teacher": [{"content": "It is unique.", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0},
                                             "assistant_message": {"role": "assistant", "content": "It is unique."}}]})
        reasoning.teacher(record, "yes", gateway=gw)
        self.assertEqual(self.assertWithinLimit("reasoning_teacher"), set())
        self.assertFalse(any(PERSONAL.search(p) for p in Capture.prompts.get("reasoning_teacher", [])))

    def test_memory_embedding_sees_remembered_lines_never_values(self):
        """35.5: memory's embedding requests carry a project's notes and the question, never a cell of its table."""
        import os
        from unittest import mock

        from dclab_rnd.memory import readers, reading, vectors
        from dclab_rnd.studio import memory as notes

        table = marked_table()
        project = {"id": "p1", "memory": []}
        notes.add(project, "decision", f"Target is {table.columns[-1]}", "the person named it", "human", "save_solution")
        local = {"DCLAB_MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:11434/v1", "DCLAB_MEMORY_EMBEDDING_MODEL": "embeddinggemma:300m-qat-q8_0",
                 "DCLAB_MEMORY": ""}
        with mock.patch.dict(os.environ, local):
            emb = vectors.Embedder(Path(tempfile.mkdtemp()), gateway().client("memory_embedding"))
            chain = [("global", "dclab"), ("workspace", "local"), ("person", "owner"), ("project", "p1")]
            reading.recall(None, "what is the target", chain, readers.from_project(project), embedder=emb)
        self.assertEqual(self.assertWithinLimit("memory_embedding"), set())

    def test_route_check_sees_the_persons_words_and_column_names_never_values(self):
        """2026-11-01: the lead agent's second reading of the route is shown the person's words and the column names only."""
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent as A

        t = marked_table()
        view = [{"name": c, "kind": "category", "noticed": []} for c in t.columns]
        s = A.CaseState(case=C.hear(C.new("dl-route"), "rank our customers by risk every week"), table=A.table_of("t.csv", view, len(t)), frame=t)
        A.route_reading(gateway({"route_check": [{"content": "prediction", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0},
                                                 "assistant_message": {"role": "assistant", "content": "prediction"}}]}).client("route_check"), s)
        self.assertEqual(self.assertWithinLimit("route_check"), set())

    def test_every_purpose_has_a_limit_and_a_test(self):
        tested = {"home_agent", "home_intent", "home_query", "home_business", "parse_pattern", "synthetic_schema", "evidence_answer", "stage_notes", "project_answer", "lesson_proposal", "intern",
                  "leakage_review", "notebook_review", "problem_analyst", "conversation_reader", "data_investigator", "brief_checker", "feature_engineer", "solution_planner", "critic", "reporter",
                  "decision_answer", "decision_labeler", "routing_labeler", "home_understand", "decomposer", "map_reviewer", "data_need", "solution_router", "ai_first", "solver_answer",
                  "understanding_critic", "solution_challenger", "debate_judge", "metric_question", "workspace_chat", "store_curator", "reasoning_teacher", "notebook_cell",
                  "home_orchestrator", "reflector", "memory_embedding", "case_agent", "case_specialist", "route_check"}
        no_user_data = {"campaign", "campaign_review", "embedding",  # research tools and evidence search: the repository's records and a typed question, never a workspace's data
                        "journey_writer", "journey_checker",  # S8 B1 (below)
                        "exam_plain", "exam_person", "exam_judge"}  # 37.0: written, simulated people and invented exam tables' column views (tests/test_conversation_exam.py); S8 B1: invented replica tables' column names and written people, never a workspace's data (tests/test_people_writer.py)
        self.assertEqual(set(settings.PURPOSES), tested | no_user_data, "a new purpose needs a data-limit test here")
        self.assertTrue(all(settings.PURPOSES[p].cell_values == 0 for p in no_user_data))



PERSONAL = re.compile(r"Q7Kxmail\w*@example\.org|\+1 415 555 \d{4}|4111 1111 1111 1111|Q7Kxdx\d")


def personal_table(rows=200):
    """A table with an e-mail (named), a contact (found by its values), a phone, a card number and a diagnosis."""
    t = marked_table(rows)
    t["email"] = [f"Q7Kxmail{i}@example.org" for i in range(rows)]
    t["contact"] = [f"Q7Kxmail{i + rows}@example.org" for i in range(rows)]
    t["phone"] = [f"+1 415 555 {1000 + i}" for i in range(rows)]
    t["wallet"] = ["4111 1111 1111 1111"] * rows
    t["diagnosis"] = [f"Q7Kxdx{i % 4}" for i in range(rows)]
    return t


class PersonalDataTests(unittest.TestCase):
    """23.7: a column that holds a person's data reaches a model by name only, whatever the purpose may see of other cells."""

    def setUp(self):
        env = mock.patch.dict(os.environ, CLEAN)
        env.start()
        self.addCleanup(env.stop)
        Capture.prompts = {}

    def personal_seen(self):
        return {m for prompts in Capture.prompts.values() for p in prompts for m in PERSONAL.findall(p)}

    def test_the_home_agent_never_sees_a_personal_value_and_still_sees_the_names(self):
        from dclab_rnd.draft import analyze
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("Predict which customers churn next month")
        store.update(d["id"], lambda x: x.update(analysis=analyze.analyze(personal_table()), assets=[{"id": "a1", "status": "ready"}], active_asset="a1"))
        gw = gateway({"home_agent": [tool_call("get_profile", {}, 1), tool_call("get_analysis", {}, 2),
                                     tool_call("get_profile", {"columns": ["email", "phone", "wallet", "diagnosis"]}, 3)]})
        HomeAgent(store, gw.client("home_agent", draft_id=d["id"])).reply(d["id"], "Customers who churn within 30 days")
        prompts = " ".join(Capture.prompts.get("home_agent", []))
        self.assertTrue(Capture.prompts.get("home_agent"), "the Home agent was never asked")
        self.assertEqual(self.personal_seen(), set())
        for name in ("email", "phone", "wallet", "diagnosis"):
            self.assertIn(name, prompts)  # the names reach it: the model can say "these look personal"

    def test_the_intern_never_sees_a_personal_value(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore, data

        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("People", "general", "Predict churn")["id"]
        personal_table().to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        gw = gateway({"intern": [tool_call("describe_data", {"project_id": pid}, 1), tool_call("finish", {"report": "done"}, 2)]})
        intern = Intern(SessionStore(Path(tempfile.mkdtemp())), Toolbox(store), gw.client("intern"))
        intern.run(intern.start("Describe the data", project_id=pid)["id"])
        self.assertTrue(Capture.prompts.get("intern"), "the intern was never asked")
        self.assertEqual(self.personal_seen(), set())
        self.assertIn("sensitivity", " ".join(Capture.prompts["intern"]))

    def test_the_decision_states_and_the_analyst_see_no_personal_value(self):
        from dclab_core.analyse import analyse
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.ledger import column_state
        from dclab_rnd.studio.data import profile_table

        table = personal_table()
        for column in profile_table(table)["columns"]:
            self.assertNotRegex(json.dumps(column_state(column, len(table)), default=str), PERSONAL)  # a state is bands
        gw = gateway({"data_investigator": [{"content": '{"after": []}', "tool_calls": [], "usage": {}}]})
        analyse({"customers": table}, target="churned", task="binary", moment="At the monthly snapshot.", advisor=GatewayAdvisor(gw))
        self.assertEqual(self.personal_seen(), set())

    def test_the_leakage_reviewer_and_the_decision_answerer_see_no_personal_value(self):
        from dclab_core.answerers import ModelAnswerer
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.ledger import column_state
        from dclab_rnd.studio import data, leakage_review, solution

        table = personal_table()
        profile = data.profile_table(table)
        proposal = solution.propose(table, profile, "churned")
        gw = gateway({"leakage_review": [{"content": "{\"columns\": []}", "tool_calls": [], "usage": {}}],
                      "decision_answer": [{"content": '{"answer": "no", "reason": "a band"}', "tool_calls": [], "usage": {}}] * len(table.columns)})
        leakage_review.review(profile, "At the monthly snapshot of each customer.", proposal["forbidden"], "churned", proposal["identifiers"], gw.client("leakage_review"))
        answerer = ModelAnswerer(GatewayAdvisor(gw), {"column.is_identifier"})
        for column in profile["columns"]:
            answerer.answer("column.is_identifier", column_state(column, len(table)), ["yes", "no"])
        self.assertTrue(Capture.prompts.get("leakage_review") and Capture.prompts.get("decision_answer"))
        self.assertEqual(self.personal_seen(), set())


if __name__ == "__main__":
    unittest.main()

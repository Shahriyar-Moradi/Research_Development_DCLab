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
CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LOG_PROMPTS", "DCLAB_INTERN_MODEL", "DCLAB_NO_LIVE_MODELS")} | {
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
        for purpose in ("stage_notes", "project_answer"):
            self.assertEqual(self.assertWithinLimit(purpose), set(), purpose)

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

    def test_every_purpose_has_a_limit_and_a_test(self):
        tested = {"home_agent", "parse_pattern", "synthetic_schema", "evidence_answer", "stage_notes", "project_answer", "intern", "leakage_review", "notebook_review"}
        no_user_data = {"campaign", "campaign_review"}  # research tools: evidence/campaigns only, never a workspace's data
        self.assertEqual(set(settings.PURPOSES), tested | no_user_data, "a new purpose needs a data-limit test here")
        self.assertTrue(all(settings.PURPOSES[p].cell_values == 0 for p in no_user_data))


if __name__ == "__main__":
    unittest.main()

"""The studio's side of the query engine (dclab_rnd/home_query.py, package H6): the checked table, the floor's setting,
the exposure log, and a model's answer that is not a plan."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import query as Q  # noqa: E402
from dclab_rnd import home_query  # noqa: E402
from dclab_rnd.audit import sha256_of  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


def setup_draft(rows=60):
    store = DraftStore(Path(tempfile.mkdtemp()))
    d = store.create("Average amount per region")
    frame = pd.DataFrame({"region": (["Dubai", "Sharjah", "Ajman"] * rows)[:rows], "amount": [float(i) for i in range(rows)]})
    path = Path(store.data_dir(d["id"]))
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path / "clean.parquet")
    store.update(d["id"], lambda x: x.update(assets=[{"id": "a1", "status": "ready", "clean_file": "clean.parquet",
                                                      "clean_sha256": sha256_of(path / "clean.parquet")}], active_asset="a1"))
    return store, d["id"], path


PLAN = Q.QueryPlan(group_by=["region"], measures=[{"name": "avg", "op": "mean", "column": "amount"}])


class HomeQueryTests(unittest.TestCase):
    def test_a_run_answers_from_the_checked_table_and_logs_what_it_related(self):
        store, draft_id, _ = setup_draft()
        result = home_query.run(store, draft_id, PLAN)
        self.assertEqual(sorted(r[0] for r in result.rows), ["Ajman", "Dubai", "Sharjah"])
        exposure = store.get(draft_id)["exposure"]
        self.assertEqual(exposure[0]["columns_related"], [["region", "amount"]])

    def test_a_changed_table_is_refused(self):
        store, draft_id, path = setup_draft()
        pd.DataFrame({"region": ["x"], "amount": [1.0]}).to_parquet(path / "clean.parquet")
        with self.assertRaises(ValueError):
            home_query.run(store, draft_id, PLAN)

    def test_a_table_without_a_hash_gets_one_on_first_read_and_is_checked_after(self):
        store, draft_id, path = setup_draft()
        store.update(draft_id, lambda d: d["assets"][0].pop("clean_sha256"))
        home_query.run(store, draft_id, PLAN)
        self.assertTrue(store.get(draft_id)["assets"][0]["clean_sha256"])
        pd.DataFrame({"region": ["x"], "amount": [1.0]}).to_parquet(path / "clean.parquet")
        with self.assertRaises(ValueError):
            home_query.run(store, draft_id, PLAN)

    def test_the_floor_comes_from_the_setting(self):
        store, draft_id, _ = setup_draft(rows=12)  # four rows per region
        self.assertEqual(home_query.run(store, draft_id, PLAN).suppressed_groups, 3)
        with mock.patch.dict(os.environ, {"DCLAB_ROW_FLOOR": "2"}):
            self.assertEqual(home_query.run(store, draft_id, PLAN).suppressed_groups, 0)

    def test_an_answer_that_is_not_a_plan_or_a_refusal(self):
        store, draft_id, _ = setup_draft()
        draft = store.get(draft_id)
        say = lambda text: type("C", (), {"complete": lambda self, m, t=None: {"content": text}})()  # noqa: E731
        self.assertEqual(home_query.write_plan(say('{"refuse": "There is no date column."}'), "trend", draft), (None, "There is no date column."))
        plan, why = home_query.write_plan(say("I cannot help"), "trend", draft)
        self.assertIsNone(plan)
        self.assertIn("not a plan", why)
        self.assertEqual(home_query.write_plan(None, "trend", draft)[0], None)


class ConditionalCountText(unittest.TestCase):
    """2026-11-02: a chart of customers LOST was titled "Number of rows by Contract": the title ignored the condition."""

    def test_a_count_with_a_condition_says_what_it_counts_in_the_title_and_the_header_but_never_in_the_models_text(self):
        from dclab_core import query as Q
        from dclab_rnd import home_chart, home_query

        plan = Q.QueryPlan(group_by=["Contract"], measures=[Q.Measure(name="lost", op="count", where=Q.Filter(column="Churn", op="eq", value=True))])
        self.assertEqual(home_query.describe(plan), "Number of rows where Churn = True by Contract")
        result = Q.QueryResult(columns=["Contract", "lost"], rows=[["Month-to-month", 1655], ["Two year", 48]], rows_scanned=100, plan_hash="x")
        self.assertEqual(home_chart.spec(plan, result)["table"]["columns"], ["Contract", "rows where Churn = True"])
        self.assertNotIn("True", home_query.model_text(plan))  # the model never reads a value


if __name__ == "__main__":
    unittest.main()

"""Package 16.2: the capability matrix is built from acceptance results, never written by hand; a failing acceptance
turns its row from supported to partial; the Domain packs read their label from it."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import kinds  # noqa: E402
from dclab_rnd.kinds import acceptance  # noqa: E402


def passing(key):
    return {"kind": key, "status": "supported", "checks": [{"decision": "a decision", "ok": True, "detail": ""}], "missing": []}


class MatrixTest(unittest.TestCase):
    def test_a_failing_acceptance_turns_its_row_from_supported_to_partial(self):
        results = [passing(k.key) for k in kinds.all_kinds()]
        rows = {r["kind"]: r for r in acceptance.matrix(results)}
        self.assertEqual({r["status"] for k, r in rows.items() if kinds.get(k).runs_in_engine}, {"supported"})
        for key in [k for k in rows if not kinds.get(k).runs_in_engine]:  # passing as a library is not running in the product
            self.assertEqual(rows[key]["status"], "partial")
            self.assertIn(acceptance.ENGINE_MISSING, rows[key]["missing"])
        failed = [r if r["kind"] != "multiclass" else
                  {"kind": "multiclass", "status": "partial", "missing": ["`resolved_by` is kept out of the model"],
                   "checks": [{"decision": "`resolved_by` is kept out of the model", "ok": False, "detail": "its role: feature"},
                              {"decision": "macro-F1 above the majority", "ok": True, "detail": ""}]} for r in results]
        row = {r["kind"]: r for r in acceptance.matrix(failed)}["multiclass"]
        self.assertEqual((row["status"], row["passed"], row["decisions"]), ("partial", 1, 2))
        self.assertIn("resolved_by", row["missing"][0])

    def test_a_result_that_claims_support_without_passing_every_check_is_partial(self):
        forged = {"kind": "binary", "status": "supported", "missing": [],
                  "checks": [{"decision": "x", "ok": True}, {"decision": "y", "ok": False}]}
        self.assertEqual({r["kind"]: r for r in acceptance.matrix([forged])}["binary"]["status"], "partial")
        empty = {"kind": "binary", "status": "supported", "missing": [], "checks": []}
        self.assertEqual({r["kind"]: r for r in acceptance.matrix([empty])}["binary"]["status"], "partial")

    def test_a_kind_never_run_is_partial_and_says_so(self):
        row = {r["kind"]: r for r in acceptance.matrix([])}["regression"]
        self.assertEqual(row["status"], "partial")
        self.assertIn("kinds-check", row["missing"][0])

    def test_the_stored_results_come_from_the_harness(self):
        path = Path(tempfile.mkdtemp()) / "capability.json"
        acceptance.write([passing("binary")], path)
        doc = json.loads(path.read_text())
        self.assertEqual(doc["generated_by"], "make kinds-check")
        self.assertEqual(acceptance.load(path)[0]["kind"], "binary")

    def test_the_committed_file_lists_every_built_in_kind_as_supported(self):
        rows = {r["kind"]: r for r in acceptance.matrix()}
        for key in ("binary", "multiclass", "regression", "text_tabular", "rare_event", "anomaly", "entity_tables", "streams",
                    "ranking", "segmentation", "uplift"):  # 16.15 B: the library kinds run in the engine
            self.assertEqual(rows[key]["status"], "supported", rows[key]["missing"])


class PackTest(unittest.TestCase):
    def test_a_pack_reads_its_label_from_the_matrix(self):
        from dclab_rnd.draft import pack

        supported = [passing(k.key) for k in kinds.all_kinds()]
        cards = {p["key"]: p for p in pack.cards(supported)}
        self.assertEqual(cards["imbalanced"]["maturity"], "Supported")
        self.assertEqual(cards["tabular"]["maturity"], "Supported")  # binary, multiclass and regression all pass
        self.assertEqual(cards["timeseries"]["maturity"], "Supported")  # 16.15: the engine runs forecasting's machine
        library_only = {"kind": "ranking", "status": "supported", "missing": [], "checks": [{"decision": "d", "ok": True}]}
        from unittest import mock

        with mock.patch.object(kinds.get("ranking"), "runs_in_engine", False):  # as if the engine did not run it
            row = {r["kind"]: r for r in acceptance.matrix([library_only])}["ranking"]
        self.assertIn("engine does not run it yet", " ".join(row["missing"]))  # a kind the engine does not run yet stays partial
        self.assertEqual(cards["scenegraph"]["maturity"], "Not built")  # no kind yet
        self.assertTrue(cards["scenegraph"]["capability"]["missing"])
        one_fails = [r if r["kind"] != "regression" else {**r, "status": "partial", "missing": ["MAE below the median"],
                                                          "checks": [{"decision": "MAE below the median", "ok": False}]} for r in supported]
        cards = {p["key"]: p for p in pack.cards(one_fails)}
        self.assertEqual(cards["tabular"]["maturity"], "Partial")
        self.assertIn("regression", " ".join(cards["tabular"]["capability"]["missing"]))
        self.assertNotIn("maturity", {k for p in pack.PACKS for k in p})  # never hand-written again


if __name__ == "__main__":
    unittest.main()

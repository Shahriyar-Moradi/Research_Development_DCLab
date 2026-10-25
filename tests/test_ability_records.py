"""The measured decide, data and learn decisions become ledger records with measured labels (S8 A3, 2026-10-25), in the
measurement's own workspace where the store reads them: each names its table as its dataset (the lake's split), each
state passes the ledger's contract, a line without its truth has no label, and a second run writes nothing again."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ability_records as A  # noqa: E402


def statement(k, cap=None, oracle_share=0.12):
    s = {"spec": k, "spec_r": 0.3, "spec_fp": 0.3, "spec_action": 0.0, "spec_cap": cap, "oracle": 0.05, "perfect": 0.1,
         "share_acted": {"argmax": 0.1, "bagged": 0.11, "argmax_share": 0.1, "bagged_share": 0.11, "bayes": 0.2, "bayes_iso": 0.15}}
    if oracle_share is not None:
        s["oracle_share"] = oracle_share
    return s


LINES = {
    "threshold": [{"did": 31, "rows_used": 1000, "fit_positives": 120, "positive_rate": 0.2, "columns": 9, "oof_auc": 0.81, "oof_ece": 0.04,
                   "statements": [statement(0), statement(1, cap=0.05), statement(2, oracle_share=None)]},
                  {"did": 37, "failed": "ValueError"}],
    "more_rows": [{"did": 44, "task": "binary", "rows_used": 5000, "have_rows": 1750, "columns": 12, "positive_rate": 0.3, "rule": "likely_help",
                   "slope": {"gain": 0.01, "low": 0.002, "high": 0.02, "per_doubling": 0.02}, "curve": [{"fraction": 1.0, "score": 0.85}],
                   "gain": 0.012, "truth": "helped"},
                  {"did": 45, "task": "regression", "rows_used": 4000, "have_rows": 1400, "columns": 6, "positive_rate": None, "rule": "cannot_tell",
                   "slope": None, "curve": [], "gain": 0.0, "truth": "unclear"}],
    "retrain": [{"did": 151, "task": "binary", "rows_used": 40000, "columns": 7, "replays": [
        {"origin": 0.4, "window_rows": 2000, "promoted": True, "window_difference": 0.02, "window_interval": [0.01, 0.03],
         "future": {"difference": 0.015, "truth": "better"}},
        {"origin": 0.55, "skipped": "the window cannot be read"},
        {"origin": 0.7, "window_rows": 2000, "promoted": False, "window_difference": 0.0, "window_interval": [-0.01, 0.01],
         "future": {"difference": 0.0, "truth": "same"}}]}],
}


class RecordsTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def _run(self, name):
        folder = Path(tempfile.mkdtemp())
        (folder / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in LINES[name]))
        with mock.patch.object(A, "ROOT", folder.parent):
            first = A.write(name, folder)
            again = A.write(name, folder)
        docs = [json.loads(line) for p in (folder / "workspace" / "ledger").rglob("*.jsonl") for line in p.read_text().splitlines() if line.strip()]
        return first, again, [d for d in docs if "value_used" in d], [d for d in docs if d.get("source") == "measured"]

    def test_each_measurement_becomes_labelled_records_once(self):
        from dclab_core.ledger import check_state

        expect = {"threshold": (3, 2), "more_rows": (2, 1), "retrain": (2, 1)}  # (records, measured labels)
        for name, (n_records, n_labels) in expect.items():
            first, again, records, labels = self._run(name)
            self.assertEqual((len(records), len(labels)), (n_records, n_labels), name)
            self.assertEqual(again["records"], 0, name)  # resumable: written once
            for r in records:
                self.assertEqual(check_state(r["state"]), [], name)
                self.assertTrue(str(r.get("dataset", "")).startswith("openml:"), name)  # the table: the lake's split in the store
            self.assertTrue(all(lab["evidence"]["kind"] == A.EVIDENCE for lab in labels), name)
        _, _, records, labels = self._run("threshold")
        self.assertEqual(sorted(lab["value"] for lab in labels), [0.12, 0.12])  # the best share in hindsight; the line without it has none
        self.assertIn("capacity_share", next(r for r in records if r["subject"]["key"].endswith(":s1"))["state"])
        _, _, records, labels = self._run("retrain")
        self.assertEqual([lab["value"] for lab in labels], [True])  # better later: promoting was right; "same" stays unlabelled


if __name__ == "__main__":
    unittest.main()

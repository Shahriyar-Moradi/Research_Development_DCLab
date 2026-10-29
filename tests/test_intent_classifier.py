"""The intent classifier (P6 of the 2026-10-29 plan): the text model of understand_students v2, Platt-calibrated, measured
across writers beside the rules; flagged requests and sealed rows are never trained on; the gate is written once from
the dev figures, and the sealed set is read once, only when it is the pinned one."""

import json
import os
import random
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import intent_classifier as C  # noqa: E402

WORDS = {
    "prediction": ["predict which customers will leave", "forecast next month demand", "who will cancel soon", "will they churn next quarter"],
    "analysis": ["average sales per region last year", "how many orders per month", "compare branches last quarter", "groups of my customers"],
    "preparation": ["merge these two files", "remove duplicate rows", "clean the messy dates", "fill the missing cells"],
    "reporting": ["weekly dashboard for my team", "monthly report for the board", "track our kpis every week", "a refreshed report"],
    "business": ["why did revenue drop", "should we open a new branch", "is the discount worth it", "which option should we take"],
    "advice": ["explain what precision means", "how much data do I need", "which method should I use", "is it ok to use this"],
}


def made_up(n_per=24, seed=0):
    rng = random.Random(seed)
    rows = []
    for t, phrases in WORDS.items():
        for i in range(n_per):
            writer = "ab"[i % 2]
            text = f"{rng.choice(['hi', 'hello', 'please', 'we need'])} {rng.choice(phrases)} {rng.choice(['thanks', 'asap', '', 'for our shop'])}"
            rows.append({"id": f"IR1-{t[:4]}-{i:03d}", "text": text, "track": t, "tracks": [t], "lang": "en" if i % 3 else "fa",
                         "style": "plain", "need": "segments" if t == "analysis" and i % 4 == 0 else None, "writer": writer,
                         "batch": f"{t}-{writer}-{i // 6}", "label_source": "written"})
    return rows


class Classifier(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.dir = Path(tempfile.mkdtemp())
        self.requests = self.dir / "intent_requests_v1.json"
        self.requests.write_text(json.dumps(made_up()))
        home = mock.patch.object(C, "home_rows", lambda: [{"id": "dev:x", "words": "predict who will churn next month", "label": "prediction",
                                                           "tracks": ["prediction"], "lang": "en", "style": None, "need": None, "group": "dev",
                                                           "source": "home_dev"}])
        home.start()
        self.addCleanup(home.stop)

    def test_it_is_understand_students_text_model_v2(self):
        from dclab_rnd.understand_students import text_model_v2

        rows = [{"words": w, "label": t} for t, ws in WORDS.items() for w in ws]
        self.assertEqual(C._pipeline().get_params(deep=True).keys(), text_model_v2(rows).get_params(deep=True).keys())
        params = C._pipeline().get_params()
        self.assertEqual((params["tfidfvectorizer__analyzer"], params["tfidfvectorizer__ngram_range"], params["logisticregression__class_weight"]),
                         ("char_wb", (2, 5), "balanced"))

    def test_train_measures_across_writers_and_writes_the_card(self):
        doc = C.train(self.requests, self.dir / "model")
        self.assertEqual(set(doc["cross_writer"]), {"train_a_test_b", "train_b_test_a"})
        for v in doc["cross_writer"].values():
            self.assertEqual(set(v["per_track"]), set(C.TRACKS))
            self.assertIn("rules", v)
            self.assertIn("calibration_error", v)
            self.assertNotIn("_answers", v)
            self.assertGreater(v["accuracy"], 0.8)  # these made-up rows are easy: it learns them
        card = json.loads((self.dir / "model" / "card.json").read_text())
        self.assertEqual((card["point"], card["family"], card["label_sources"]), ("request.intent", "logistic", {"written": 144}))
        self.assertIn("Platt", card["calibration"])
        self.assertTrue((self.dir / "model" / f"{C.NAME}.joblib").is_file())

    def test_flagged_and_sealed_rows_are_never_trained_on(self):
        flagged = self.dir / "intent_requests_v1_flagged.json"
        flagged.write_text("[]")
        with self.assertRaises(ValueError):
            C.load_rows(flagged)
        rows = C.load_rows(self.requests)
        with self.assertRaises(ValueError):
            C.fit([{**rows[0], "split": "sealed"}] + rows[1:])

    def test_the_threshold_is_the_smallest_with_the_precision(self):
        answers = [{"confidence": 0.9, "right": True}] * 30 + [{"confidence": 0.5, "right": False}] * 30
        got = C.choose_threshold(answers)
        self.assertEqual((got["threshold"], got["coverage"], got["precision"]), (0.51, 0.5, 1.0))
        self.assertIsNone(C.choose_threshold([{"confidence": 0.9, "right": False}] * 30)["threshold"])

    def test_the_gate_is_written_once_and_the_sealed_set_read_once_when_pinned(self):
        C.train(self.requests, self.dir / "model")
        sealed = self.dir / "sealed.json"
        sealed.write_text(json.dumps([{"id": f"s-{i}", "text": r["text"], "tracks": r["tracks"], "lang": r["lang"], "style": "plain", "need": None}
                                      for i, r in enumerate(made_up(6, seed=3))]))
        sha = self.dir / "sealed.sha256"
        sha.write_text(C.fingerprint(sealed) + "\n")
        gate_file, campaign = self.dir / "gate.json", self.dir / "campaign"
        with self.assertRaises(SystemExit):  # no gate yet
            C.gate(campaign, self.dir / "model", sealed, sha, gate_file)
        self.assertFalse(C.spent(campaign))
        with mock.patch.object(C, "SEALED_SHA", sha):
            g = C.write_gate(self.dir / "model" / "MEASURED.json", gate_file)
            with self.assertRaises(SystemExit):
                C.write_gate(self.dir / "model" / "MEASURED.json", gate_file)  # fixed once
        self.assertEqual(g["sealed_sha256"], C.fingerprint(sealed))
        self.assertIn("seen_before", g)
        other = self.dir / "other.json"
        other.write_text("[]")
        with self.assertRaises(SystemExit):  # not the pinned set
            C.gate(campaign, self.dir / "model", other, sha, gate_file)
        result = C.gate(campaign, self.dir / "model", sealed, sha, gate_file)
        self.assertEqual(set(result["checks"]), {"accuracy", "accuracy_ci_low", "per_track_recall", "margin_over_rules",
                                                 "calibration", "alone_precision", "alone_precision_ci_low", "alone_coverage"})
        self.assertEqual(result["measured"]["n"], sum(r["lang"] == "en" for r in made_up(6, seed=3)))  # English only gates
        self.assertEqual(result["every_language"]["n"], len(made_up(6, seed=3)))  # every language reported
        self.assertTrue(C.spent(campaign))
        with self.assertRaises(SystemExit):  # read once
            C.gate(campaign, self.dir / "model", sealed, sha, gate_file)


if __name__ == "__main__":
    unittest.main()

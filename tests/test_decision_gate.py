"""Package 22.3: the decision models' gate, fixed before the runs. A model passes only when it beats the rule by table
group and misses no class; an error is counted, never skipped; the sealed set is read only when it matches its
committed fingerprint, and once under any gate version; a gate version keeps its thresholds; a model that is not its
card's is refused; every lower bound is drawn by table group."""

import json
import random
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import joblib

    from dclab_core import decision_gate as gate
    from dclab_core import decision_models as dm
    from dclab_rnd import decision_gate_run as run
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def decisions(n_groups=40, per=10, seed=0, rule_wrong=0.2):
    """Yes/no decisions; the rule is wrong on a share of them, the truth follows one band."""
    rng = random.Random(seed)
    truth, rule, groups, states = [], [], [], []
    for g in range(n_groups):
        for _ in range(per):
            y = rng.random() < 0.5
            truth.append(y)
            rule.append((not y) if rng.random() < rule_wrong else y)
            groups.append(f"g{g}")
            states.append({"uniqueness": "high" if y else "low"})
    return truth, rule, groups, states


@unittest.skipIf(MISSING, MISSING)
class VerdictTest(unittest.TestCase):
    def test_the_gate_is_pinned(self):
        # fixed before the run: a change is a new version (and a new sealed set)
        self.assertEqual(gate.VERSION, "decision_gate_v1")
        self.assertEqual(gate.THRESHOLDS, {"scored_share_min": 1.0, "accuracy_lower_min": 0.90, "precision_lower_min": 0.80,
                                           "recall_lower_min": 0.80, "margin_lower_min": 0.0, "class_min_rows": 30,
                                           "class_recall_min": 0.70, "turned_max": 0.02, "draws": 2000})
        self.assertEqual(gate.DEMOTION, {"accuracy_lower_below": 0.85, "margin_lower_below": -0.02, "turned_above": 0.05})

    def test_lower_bounds_are_drawn_by_table_group_so_a_table_does_not_count_as_many(self):
        # the model's errors come in whole tables (as a table and its twin do): by table, the bound is lower
        truth, rule, groups, _ = decisions(n_groups=40, per=20)
        model = [(not t) if g in ("g1", "g2", "g3", "g4") else t for t, g in zip(truth, groups)]
        by_table = gate.bounds(truth, model, groups)
        by_row = gate.bounds(truth, model, [f"r{i}" for i in range(len(truth))])
        self.assertEqual(by_table["accuracy"][0], by_row["accuracy"][0])
        self.assertLess(by_table["accuracy"][1], by_row["accuracy"][1])
        self.assertLess(by_table["recall"][1], by_row["recall"][1])

    def test_the_gate_reads_the_bound_by_table_not_by_row(self):
        # 93% right, the errors in seven whole tables: Wilson by row clears 0.90, the bound by table does not
        truth = [True, False] * 500
        groups = [f"g{i // 10}" for i in range(1000)]
        model = [(not t) if int(g[1:]) < 7 else t for t, g in zip(truth, groups)]
        rule = [(not t) if i % 3 == 0 else t for i, t in enumerate(truth)]
        v = gate.verdict(truth, model, rule, groups)
        self.assertGreaterEqual(v["model"]["accuracy_ci"][0], 0.90)
        accuracy = next(c for c in v["criteria"] if c["criterion"].startswith("accuracy"))
        self.assertLess(accuracy["value"], 0.90)
        self.assertFalse(accuracy["pass"])

    def test_a_class_answered_once_and_never_true_does_not_sink_the_recall(self):
        truth = ["id"] * 500 + ["text"] * 300 + ["date"] * 200
        model = list(truth)
        model[0] = "boolean"  # never true in this set
        b = gate.bounds(truth, model, [f"g{i // 10}" for i in range(1000)])
        self.assertGreater(b["recall"][0], 0.99)  # recall over the classes in the truth
        self.assertLess(b["precision"][0], 0.8)   # precision over the classes it answered: the stray answer counts

    def test_a_yes_no_model_that_never_says_no_fails_on_the_class(self):
        truth, rule, groups, _ = decisions()
        v = gate.verdict(truth, [True] * len(truth), rule, groups)
        weak = next(c for c in v["criteria"] if c["criterion"] == "no class it misses")
        self.assertFalse(weak["pass"])
        self.assertTrue(weak["value"][0].startswith("False"))

    def test_a_model_better_than_the_rule_passes(self):
        truth, rule, groups, _ = decisions()
        v = gate.verdict(truth, list(truth), rule, groups)
        self.assertTrue(v["pass"], v["criteria"])
        self.assertGreater(v["margin"]["lower"], 0)
        self.assertEqual(v["gate"], "decision_gate_v1")

    def test_matching_the_rule_is_not_enough(self):
        truth, rule, groups, _ = decisions(rule_wrong=0.0)
        v = gate.verdict(truth, list(truth), rule, groups)
        self.assertFalse(v["pass"])
        self.assertEqual([c["criterion"] for c in v["criteria"] if not c["pass"]], ["margin over the rule, 95% lower bound (by table group)"])

    def test_a_missed_class_fails_even_when_the_aggregate_is_high(self):
        truth = ["id"] * 900 + ["text"] * 60 + ["date"] * 40
        model = ["id"] * 900 + ["id"] * 60 + ["date"] * 40  # never says text
        rule = ["id"] * 800 + ["date"] * 100 + ["text"] * 60 + ["id"] * 40
        groups = [f"g{i // 10}" for i in range(1000)]
        v = gate.verdict(truth, model, rule, groups)
        weak = next(c for c in v["criteria"] if c["criterion"] == "no class it misses")
        self.assertFalse(weak["pass"])
        self.assertIn("text (0/60)", weak["value"])

    def test_a_model_that_breaks_what_the_rule_gets_right_fails(self):
        # better overall (it fixes all the rule's errors) but wrong on 5% of what the rule got right
        truth, rule, groups, _ = decisions(rule_wrong=0.3)
        model = list(truth)
        right = [i for i, (t, r) in enumerate(zip(truth, rule)) if t == r]
        for i in right[: len(right) // 20 + 1]:
            model[i] = not truth[i]
        v = gate.verdict(truth, model, rule, groups)
        kept = next(c for c in v["criteria"] if c["criterion"] == "the rule's right decisions kept")
        self.assertFalse(kept["pass"])
        self.assertFalse(v["pass"])

    def test_an_error_is_counted_not_skipped(self):
        truth, rule, groups, _ = decisions()
        v = gate.verdict(truth, list(truth), rule, groups, errors=1)
        self.assertFalse(v["criteria"][0]["pass"])

    def test_the_margin_is_drawn_by_table_group(self):
        # the model wins on whole tables: one table's decisions move together, so its interval is wider by table
        truth = [True] * 400
        model = [True] * 400
        rule = [True] * 360 + [False] * 40
        by_table = gate.margin(truth, model, rule, [f"g{i // 40}" for i in range(400)])
        by_row = gate.margin(truth, model, rule, [f"r{i}" for i in range(400)])
        self.assertLess(by_table["lower"], by_row["lower"])


@unittest.skipIf(MISSING, MISSING)
class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.generated, self.models, self.results = root / "gen", root / "models", root / "results"
        truth, rule, groups, states = decisions()
        rows = [{"record": f"r{i}", "group": g, "split": "sealed", "state": s, "label": {"value": t, "source": "benchmark"},
                 "value_used": r, "options": [True, False]} for i, (t, r, g, s) in enumerate(zip(truth, rule, groups, states))]
        (self.generated / "decisions" / "column.is_identifier").mkdir(parents=True)
        (self.generated / "decisions" / "column.is_identifier" / "sealed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.generated / "MANIFEST.json").write_text(json.dumps({"sets": {"column.is_identifier": {"sealed_fingerprint": run.sealed_fingerprint(rows)}}}))
        train_truth, _, train_groups, train_states = decisions(seed=1)
        train_rows = [{"record": f"t{i}", "group": f"t{g}", "split": "train", "state": s, "label": {"value": t, "source": "benchmark"}}
                      for i, (t, g, s) in enumerate(zip(train_truth, train_groups, train_states))]
        model, card = dm.train("column.is_identifier", train_rows, folds=3, families=("logistic",), strengths=(1.0,))
        self.models.mkdir()
        joblib.dump(model, self.models / "column.is_identifier.joblib")
        (self.models / "column.is_identifier.json").write_text(card.model_dump_json())

    def tearDown(self):
        self.tmp.cleanup()

    def judge(self):
        return run.judge("column.is_identifier", self.generated, self.models, self.results)

    def test_one_verdict_is_stored_and_the_sealed_set_is_then_spent(self):
        doc = json.loads(self.judge().read_text())
        self.assertEqual((doc["campaign_id"], doc["status"], doc["result"]), ("decision_models_v1", "completed", "pass"))
        self.assertEqual(doc["sealed"]["decisions"], 400)
        for who in ("model", "rule"):  # sealed precision and recall with their intervals, beside the rule's
            for name in ("precision", "recall"):
                self.assertEqual(len(doc["verdict"]["bounds"][who][name]), 3)
                self.assertEqual(len(doc["verdict"][who][f"{name}_ci"]), 2)
        with self.assertRaises(ValueError):
            self.judge()

    def test_a_new_gate_version_cannot_reread_a_spent_sealed_set(self):
        self.judge()
        old = gate.VERSION
        try:
            gate.VERSION = "decision_gate_v2"
            with self.assertRaises(ValueError):
                self.judge()
        finally:
            gate.VERSION = old

    def test_a_gate_version_keeps_the_thresholds_it_judged_by(self):
        path = self.judge()
        doc = json.loads(path.read_text())
        doc["point"], doc["sealed"]["fingerprint"] = "column.time", "other"
        doc["gate"]["thresholds"] = {**doc["gate"]["thresholds"], "recall_lower_min": 0.77}
        path.write_text(json.dumps(doc))  # as if another point had been judged by moved thresholds
        with self.assertRaises(ValueError):
            self.judge()

    def test_an_answer_that_raises_is_counted_and_fails_the_scoring(self):
        model = joblib.load(self.models / "column.is_identifier.joblib")
        calls = {"n": 0}
        real = model.answer

        def flaky(point, state, options):
            calls["n"] += 1
            if calls["n"] % 50 == 0:
                raise RuntimeError("broken")
            return real(point, state, options)

        model.answer = flaky
        with mock.patch("joblib.load", return_value=model):
            doc = json.loads(self.judge().read_text())
        self.assertEqual(doc["result"], "fail")
        self.assertEqual(doc["verdict"]["criteria"][0]["value"], "392/400")

    def test_a_sealed_set_that_moved_is_refused(self):
        path = self.generated / "decisions" / "column.is_identifier" / "sealed.jsonl"
        lines = path.read_text().splitlines()
        first = json.loads(lines[0])
        first["label"]["value"] = not first["label"]["value"]
        path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
        with self.assertRaises(ValueError):
            self.judge()
        self.assertFalse(self.results.exists())

    def test_a_model_that_is_not_its_cards_is_refused(self):
        card = json.loads((self.models / "column.is_identifier.json").read_text())
        card["version"] = "000000000000"
        (self.models / "column.is_identifier.json").write_text(json.dumps(card))
        with self.assertRaises(ValueError):
            self.judge()


if __name__ == "__main__":
    unittest.main()

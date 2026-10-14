"""Package 22.0: one interface for every decision model. A scripted answerer, the rule and a classical classifier
answer the same recorded decisions, and their coverage, agreement with the rule and accuracy against the labels come
out of one function; an answerer's answer is recorded beside the rule's, never used."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    from dclab_core import answerers as ans
    from dclab_rnd import ledger
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def decisions(n=120, seed=0):
    """Identifier decisions on invented columns, planted so the classifier must win: the state's uniqueness band is set
    from the label itself, and the rule reads only the name, which misses about 40% of identifiers by construction.
    It shows the plumbing (three kinds of answerer compared in one call), not a finding about any real rule."""
    rng = np.random.default_rng(seed)
    book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
    records = []
    for i in range(n):
        is_id = bool(rng.random() < 0.5)
        named = bool(rng.random() < 0.6) if is_id else bool(rng.random() < 0.1)
        name = f"{'ref' if named else 'col'}_{i}_id" if named else f"col_{i}"
        state = {"column": name, "name_tokens": [t for t in name.split("_") if not t.isdigit()], "dtype": "integer",
                 "uniqueness": "all" if is_id else rng.choice(["from_10_to_50pct", "from_1_to_10pct"]),
                 "cardinality": "very_high" if is_id else "medium"}
        rule = named  # today's rule: the name
        rec = ledger.decide("column.is_identifier", ("column", name), answers=[ledger.answer("rule", "id_like", rule)], value_used=rule,
                            state=state, ledger=book, project=f"p{i % 6}")
        ledger.label(rec, is_id, "benchmark", "replica:generated", ledger=book)
        records.append(rec)
    return book


@unittest.skipIf(MISSING, MISSING)
class InterfaceTest(unittest.TestCase):
    def tearDown(self):
        for key in [k for k in ans.REGISTRY if k.startswith(("noop", "taken"))]:
            ans.REGISTRY.pop(key)

    def test_three_kinds_of_answerer_on_the_same_decisions(self):
        book = decisions()
        records, labels = book.records(), book.labels()
        train = [r for r in records if r.project not in ("p4", "p5")]
        test = [r for r in records if r.project in ("p4", "p5")]
        truth = {l.record: l.value for l in labels}
        clf = ans.ClassifierAnswerer("column.is_identifier", name="id_classifier").fit((r.state, truth[r.id]) for r in train)
        scripted = ans.ScriptedAnswerer("always_no", False, ["column.is_identifier"])
        out = ans.compare(test, labels, [ans.RecordedRule(), scripted, clf])
        self.assertEqual(set(out), {"recorded_rule@1", "always_no@1", "id_classifier@1"})
        for key, scopes in out.items():
            row = scopes["all"]
            self.assertEqual((row["decisions"], row["labelled"]), (len(test), len(test)), key)
            self.assertIsNotNone(row["accuracy"])
        self.assertEqual(out["recorded_rule@1"]["all"]["agreement_with_rule"], 1.0)
        self.assertGreater(out["id_classifier@1"]["all"]["accuracy"], out["recorded_rule@1"]["all"]["accuracy"])  # it learned uniqueness
        self.assertIn("column.is_identifier", out["id_classifier@1"])

    def test_an_answer_is_recorded_beside_the_rules_never_used(self):
        book = decisions(n=20)
        record = book.records()[0]
        shadow = ans.with_answers(record, [ans.ScriptedAnswerer("guess", True, ["column.is_identifier"])])
        self.assertEqual([a.by for a in shadow.answers], ["rule", "model_role"])
        self.assertEqual((shadow.value_used, shadow.used), (record.value_used, record.used))

    def test_an_answerer_abstains_and_the_registry_knows_its_points(self):
        clf = ans.ClassifierAnswerer("column.is_identifier", abstain_below=0.99)
        self.assertIsNone(clf.answer("column.is_identifier", {"uniqueness": "all"}, []))  # not trained
        ans.register(ans.ScriptedAnswerer("noop", None, ["column.is_identifier"]))
        self.assertIn("noop@1", ans.REGISTRY)
        with self.assertRaises(ValueError):
            ans.register(ans.ScriptedAnswerer("bad", True, ["not.a.point"]))
        with self.assertRaises(ValueError):
            ans.ClassifierAnswerer("column.is_identifier").fit([({"a": "x"}, True)])  # one value: nothing to learn
        self.assertNotIn("column=x_id", ans.features({"column": "x_id", "name_tokens": ["x", "id"]}))  # a name only by its tokens


@unittest.skipIf(MISSING, MISSING)
class FindingsTest(unittest.TestCase):
    def leak_rows(self, n=200, spread=0.05, seed=1):
        rng = np.random.default_rng(seed)
        lift = rng.normal(0, spread, n)
        return [({"apparent_lift": float(x), "task": "binary_classification"}, bool(x > 0)) for x in lift]

    def test_a_small_measure_is_learned_once_scaled(self):
        rows = self.leak_rows()
        clf = ans.ClassifierAnswerer("feature.leakage_suspect").fit(rows[:100])
        right = sum(clf.answer("feature.leakage_suspect", s, ["yes", "no"]).value == y for s, y in rows[100:])
        self.assertGreater(right / 100, 0.85)  # a lift of a few hundredths, scaled like any other feature

    def test_the_measure_a_label_comes_from_can_be_left_out(self):
        rows = self.leak_rows()
        clf = ans.ClassifierAnswerer("feature.leakage_suspect", exclude=["apparent_lift"]).fit(rows[:100] + [({"task": "regression"}, False)])
        self.assertNotIn("apparent_lift", ans.features(rows[0][0], ["apparent_lift"]))
        self.assertEqual(clf.exclude, ("apparent_lift",))

    def test_a_choice_answer_stays_within_the_decisions_options(self):
        rows = [({"has_time": True}, "time")] * 20 + [({"has_time": False}, "random")] * 20
        clf = ans.ClassifierAnswerer("split.strategy").fit(rows)
        self.assertEqual(clf.answer("split.strategy", {"has_time": True}, []).value, "time")
        self.assertEqual(clf.answer("split.strategy", {"has_time": True}, ["random", "group"]).value, "random")  # never an option not offered
        self.assertIsNone(clf.answer("split.strategy", {"has_time": True}, ["group"]))  # no option it knows: abstains
        sure = ans.ClassifierAnswerer("split.strategy", abstain_below=0.999).fit(rows)
        self.assertIsNone(sure.answer("split.strategy", {"has_time": True}, []))  # below its confidence: abstains

    def test_shadows_are_validated_and_never_move_the_truth(self):
        book = decisions(n=10)
        rec = book.records()[0]
        bad = ans.with_answers(rec, [ans.ScriptedAnswerer("x", "maybe", ["column.is_identifier"])])
        self.assertIs(bad, rec)  # "maybe" is no answer to a yes/no point: an abstention
        exposed = ledger.label(rec, rec.answers[0].value, "user_decision_exposed", "person", ledger=book)
        labels = [exposed]
        before = ans.compare([rec], labels, [ans.ScriptedAnswerer("s", not rec.answers[0].value, ["column.is_identifier"])])
        shadowed = ans.with_answers(rec, [ans.ScriptedAnswerer("s", not rec.answers[0].value, ["column.is_identifier"])])
        self.assertEqual(shadowed.answers[-1].source, "shadow")
        after = ans.compare([shadowed], labels, [ans.ScriptedAnswerer("s", not rec.answers[0].value, ["column.is_identifier"])])
        self.assertEqual(before["s@1"]["all"]["labelled"], after["s@1"]["all"]["labelled"])  # an agreeing person stays no label
        many = rec
        for i in range(30):
            many = ans.with_answers(many, [ans.ScriptedAnswerer(f"a{i}", True, ["column.is_identifier"])])
        self.assertEqual(len(many.answers), 20)
        self.assertIs(ans.with_answers(rec, [ans.ScriptedAnswerer("other", True, ["split.strategy"])]), rec)  # not its point

    def test_coverage_abstention_and_accuracy_over_all(self):
        book = decisions(n=40)
        records, labels = book.records(), book.labels()
        half = ans.ScriptedAnswerer("half", lambda p, st, o: True if st["uniqueness"] == "all" else None, ["column.is_identifier"])
        row = ans.compare(records, labels, [half])["half@1"]["all"]
        self.assertLess(row["coverage"], 1.0)
        self.assertEqual(row["accuracy"], 1.0)  # right wherever it answered
        self.assertLess(row["accuracy_all"], 1.0)  # an abstention counted as wrong: not flattered

    def test_the_registry_refuses_a_different_answerer_under_a_taken_name(self):
        ans.register(ans.ScriptedAnswerer("taken", True, ["column.is_identifier"]))
        with self.assertRaises(ValueError):
            ans.register(ans.ScriptedAnswerer("taken", False, ["column.is_identifier"]))
        with self.assertRaises(ValueError):
            ans.register(ans.ScriptedAnswerer("taken2", True, ["column.is_identifier"], by="banana"))
        for key in [k for k in ans.REGISTRY if k.startswith("taken")]:
            ans.REGISTRY.pop(key)


if __name__ == "__main__":
    unittest.main()

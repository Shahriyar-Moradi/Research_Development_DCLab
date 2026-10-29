"""Claude's routing labels (2026-10-29): Claude sees the person's words, the decision's question and options and the state's
bands, never any answerer's reading or the value Home used; exams are never labelled; scripted examples only up to a
sample; the answers go to the store's own labels file, apart from every ledger label."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def example(i, point="request.intent", origin="real", label=None, exam=None, text="Which customers will churn next month?"):
    return {"id": f"m7_routing:ws:{point}:{i}", "point": point, "origin": origin, "split": "train", "text": text, "exam": exam,
            "state": {"has_data": False, "words": 7}, "options": ["prediction", "analysis", "preparation", "reporting", "business", "advice"],
            "readings": [{"by": "rule", "answerer": "rule:intent_rules_v3", "value": "prediction"}], "value_used": "prediction",
            "used_by": "rule", "label": label}


class RoutingLabelsTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store.layout import Layout

        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.layout = Layout(self.root)
        folder = self.layout.sets("m7_routing") / "request.intent" / "v1"
        folder.mkdir(parents=True)
        rows = [example(1), example(2, label="analysis"), example(3, exam="sealed_v5"), example(4, text=None)] + \
               [example(10 + i, origin="scripted") for i in range(5)]
        (folder / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (folder / "sealed.jsonl").write_text(json.dumps(example(99)) + "\n")  # never read

    def test_only_unlabelled_words_outside_exams_and_a_scripted_sample_are_asked(self):
        from dclab_rnd.store import routing_labels as rl

        todo = rl.pending(self.layout, scripted_per_point=2, points=("request.intent",))
        ids = [x["id"].rsplit(":", 1)[1] for x in todo]
        self.assertEqual(ids[0], "1")
        self.assertEqual(len(ids), 3)  # the real one and two of five scripted; never the labelled, the exam, the wordless or the sealed split
        self.assertNotIn("99", ids)

    def test_claude_never_sees_a_reading_and_its_answer_is_kept_apart(self):
        from dclab_rnd.store import routing_labels as rl

        seen = []

        def ask(shown):
            seen.append(json.dumps(shown))
            return {"answer": "prediction", "confidence": "high", "reason": "asks what will happen"}, {"request": "mr-1", "thought": True}

        todo = rl.pending(self.layout, scripted_per_point=0, points=("request.intent",))
        counts = rl.run(self.layout, ask, todo, "claude-sonnet-5-5")
        self.assertEqual(counts["labelled"], 1)
        self.assertNotIn("rule:intent_rules_v3", seen[0])
        self.assertNotIn("value_used", seen[0])
        self.assertIn("Which customers will churn next month?", seen[0])
        line = json.loads((self.root / "labels" / rl.FILE).read_text().splitlines()[0])
        self.assertEqual((line["value"], line["purpose"], line["request"], line["thought"]), ("prediction", "routing_labeler", "mr-1", True))
        self.assertEqual(rl.pending(self.layout, scripted_per_point=0, points=("request.intent",)), [])  # never asked twice

    def test_an_answer_outside_the_options_is_no_label(self):
        from dclab_rnd.store import routing_labels as rl

        todo = rl.pending(self.layout, scripted_per_point=0, points=("request.intent",))
        counts = rl.run(self.layout, lambda shown: ({"answer": "a dashboard please"}, {}), todo, "m")
        self.assertEqual((counts["labelled"], counts["refused"]), (0, 1))
        self.assertIsNone(json.loads((self.root / "labels" / rl.FILE).read_text().splitlines()[0])["value"])

    def test_the_purpose_sees_no_cell_values_and_is_registered(self):
        from dclab_rnd.agents import harness
        from dclab_rnd.models import settings
        from dclab_rnd.prompts import names

        self.assertEqual(settings.PURPOSES["routing_labeler"].cell_values, 0)
        self.assertIn("routing_labeler", harness.MODEL_USES)
        self.assertIn("routing_labeler", names())
        self.assertEqual(settings.thinking_for("routing_labeler"), "adaptive")  # the owner's rule: it thinks, and its summary is kept


if __name__ == "__main__":
    unittest.main()

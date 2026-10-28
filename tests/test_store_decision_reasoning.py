"""Every decision row and every step of a process carries its reasoning, written by code from what was known when it was made
(the owner, 2026-10-28: every example carries its reasoning, labelled by its source); never the label or a person's answer."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.ledger import Answer  # noqa: E402
from dclab_core.store import DecisionExample, Episode, EpisodeStep  # noqa: E402
from dclab_rnd.store.sets import reasoning_counts  # noqa: E402
from dclab_rnd.store.sets.decisions import decision_reasoning  # noqa: E402
from dclab_rnd.store.sets.others import _move_why  # noqa: E402


class DecisionReasoningTests(unittest.TestCase):
    def test_code_never_writes_a_decision_rows_reasoning(self):  # the review of C2: it would carry the label or argue against it
        answers = [Answer(by="rule", answerer="rule:id_v1", value=True, reason="unique on every row"),
                   Answer(by="person", answerer="owner", value=False)]
        self.assertIsNone(decision_reasoning("column.is_identifier", {"uniqueness": "all"}, ["yes", "no"], answers))
        row = DecisionExample.model_validate({"id": "m1:b", "model_type": "m1_students", "split": "train", "point": "p", "group": "g",
                                              "provenance": {"workspace": "w", "source": "ledger/x.jsonl", "entry": "e", "sha256": "0" * 64,
                                                             "at": "2026-10-28T00:00:00+00:00"}})
        self.assertEqual(reasoning_counts([row]), {"missing": 1})  # counted as missing, never as a stated reason

    def test_old_rows_read_and_a_reasoning_comes_with_its_source(self):
        old = {"schema_version": "dclab_core.decision_example.v2", "id": "m1:a", "model_type": "m1_students", "split": "train",
               "provenance": {"workspace": "w", "source": "ledger/x.jsonl", "entry": "e", "sha256": "0" * 64, "at": "2026-10-28T00:00:00+00:00"},
               "point": "column.is_identifier", "group": "general"}
        ex = DecisionExample.model_validate(old)
        self.assertEqual(ex.schema_version, "dclab_core.decision_example.v3")
        self.assertIsNone(ex.reasoning)
        with self.assertRaises(ValueError):
            DecisionExample.model_validate({**old, "reasoning": "because"})  # a reasoning never comes without its source
        ex = DecisionExample.model_validate({**old, "reasoning": "Known then: uniqueness all.", "reasoning_source": "from_record"})
        self.assertEqual(reasoning_counts([ex]), {"from_record": 1})

    def test_a_step_says_what_was_known_before_it_never_its_own_outcome(self):
        why = _move_why({"move": "run_stage", "from": "WF-03", "to": "WF-04", "status": "blocked", "failed_checks": ["leakage_audit"],
                         "rules": ["DCLAB-R01"]})  # the PREVIOUS move: why the agent tried again
        self.assertTrue(why.startswith("Before this move:"))
        self.assertIn("blocked by the workflow validator, failing leakage_audit (rules DCLAB-R01)", why)
        step = EpisodeStep(n=0, action="run_stage:WF-03>WF-04", reasoning=why, reasoning_source="from_record")
        ep = Episode(id="m6:x", model_type="m6_stepwise_decisions", split="train",
                     provenance={"workspace": "w", "source": "projects/p/transitions.jsonl", "entry": "e", "sha256": "0" * 64, "at": "2026-10-28T00:00:00+00:00"},
                     process="graph", steps=[step])
        self.assertEqual(reasoning_counts([ep]), {"from_record": 1})
        old = {**ep.model_dump(mode="json"), "schema_version": "dclab_core.episode.v1"}
        self.assertEqual(Episode.model_validate(old).schema_version, "dclab_core.episode.v2")


if __name__ == "__main__":
    unittest.main()

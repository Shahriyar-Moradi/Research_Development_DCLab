"""Package 24.6: the written person of a problem with several parts, from the first sentence to the one read of the holdout."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd.agent_eval.journeys import programme as journeys
except ImportError as error:  # the agent environment lacks pandas
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class ProgrammeJourneyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.journeys = journeys.load("dev")
        cls.row = journeys.play(cls.journeys[0], seed=0)

    def test_the_dev_set_is_written_and_has_what_each_journey_expects(self):
        self.assertEqual([j["id"] for j in self.journeys], ["JP-01"])
        for j in self.journeys:
            self.assertTrue(j["words"] and j["answers"] and j["expect"]["families"] and j["expect"]["approaches"])

    def test_the_person_is_carried_from_the_first_sentence_to_the_holdout_number(self):
        row = self.row
        self.assertTrue(row["right"], row["why"])
        self.assertEqual(set(row["parts"]), set(journeys.PARTS))
        for step in ("draw_map", "confirm_map"):
            self.assertIn(step, row["steps"], "the map of the problem was drawn and the person confirmed it before the build")
        self.assertEqual(row["refused_moves"], 0)

    def test_the_composition_beat_the_best_leaf_and_found_more_of_the_planted_fraud(self):
        h = self.row["holdout"]
        self.assertGreater(h["low"], 0)
        self.assertGreater(h["planted_found"], h["planted_found_best_leaf"])

    def test_the_holdout_was_read_once(self):
        self.assertNotIn("read", " ".join(self.row["why"]))
        self.assertTrue(self.row["parts"]["holdout"])


if __name__ == "__main__":
    unittest.main()

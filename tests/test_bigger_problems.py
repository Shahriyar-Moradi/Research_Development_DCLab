"""Package 30.6: bigger and more problems, after the sealed programme journeys (2 of 10 for the rules) said what was missing.

The ways people said "nothing was confirmed" that no rule read are read; fraud said by its deeds sets the fraud programme off; the
programme draws the parts the person's words support (linked accounts alone are one problem; a review list only where the team's
checks are limited); a card with nothing labelled and no kind of answer is the unusual rows; the journeys run on a bigger replica; and a
fresh sealed set, pinned, with its gate fixed before it ran, is run once per arm.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class ReadingTest(unittest.TestCase):
    def test_the_ways_people_say_nothing_was_confirmed_are_read(self):
        from dclab_core.conversation import read_answer

        for text, said in (("Nobody ever confirmed anything. We never really checked, so I have no list of bad ones.", "none"),
                           ("Nothing confirmed. We have never run an investigation, so there is no list.", "none"),
                           ("We have not marked any of them yet.", "none"),
                           ("Investigators have marked some deliveries as fraud.", "file"),
                           ("The rides export and the list the investigators made.", "file"),
                           ("No file, but the is_fraud column marks them", "column")):
            with self.subTest(text):
                self.assertEqual(read_answer("labels", text).get("labels"), said)

    def test_a_card_with_nothing_labelled_and_no_kind_said_is_the_unusual_rows(self):
        from dclab_core.conversation import card_for

        card = card_for("Some riders seem to finish deliveries very quickly. Can you tell us which ones?", {"labels": "none"})
        self.assertEqual((card.family.value, card.family.status), ("anomaly", "inferred"))
        self.assertNotIn("What should DCLab give for each row", " ".join(card.blocking))
        grouped = card_for("Group our customers into segments.", {"labels": "none"})
        self.assertIn(grouped.family.value, ("segmentation",))

    def test_fraud_said_by_its_deeds_sets_the_programme_off_and_only_its_parts_are_drawn(self):
        from dclab_core import problem_map as pm

        m = pm.decompose("We have confirmed cases of couriers closing jobs with no movement, and colluding customer accounts too. "
                         "We can review a limited number of rides per day.")
        self.assertEqual([s.family for s in m.sub_problems], ["anomaly", "rare_event", "entity_tables", "ranking"])  # a denial near nothing: the cases exist
        one = pm.decompose("Groups of customer accounts opened together order only from one biker to collect fares; find those groups.")
        self.assertEqual((one.size, [s.family for s in one.sub_problems]), ("single", ["entity_tables"]))
        no_list = pm.decompose("Find the fake rides our bikers make and the customer accounts that work with them.")
        self.assertNotIn("ranking", [s.family for s in no_list.sub_problems])  # no limit said: no review list
        limited = pm.decompose("Find the fake rides our bikers make; the team checks about 80 a day.")
        self.assertIn("ranking", [s.family for s in limited.sub_problems])


@unittest.skipIf(MISSING, MISSING)
class FreshSealedTest(unittest.TestCase):
    def test_the_fresh_set_is_pinned_and_its_gate_was_fixed_before_it_ran(self):
        from dclab_rnd.agent_eval.journeys import programme as P

        self.assertEqual(P.fingerprint(P.SPLITS["sealed_v2"]), P.SEALED_HASHES["sealed_v2"].read_text(encoding="utf-8").strip())
        self.assertEqual(len(P.load("sealed_v2")), 10)
        gate = P.GATES["sealed_v2"]
        self.assertEqual((gate["version"], gate["split"], gate["seed"]), ("journeys_programme_gate_v2", "sealed_v2", 7))
        self.assertNotIn(gate["seed"], (0, 5, *range(10, 16)))
        self.assertIn("examples", gate)
        self.assertEqual(P.LATEST_SEALED, "sealed_v2")
        self.assertGreaterEqual(P.SIZE["n_orders"], 200_000)  # the bigger tables every journey runs on

    def test_a_sealed_set_is_run_once_per_arm(self):
        from dclab_rnd.agent_eval.journeys import programme as P

        def never(*a, **k):
            raise AssertionError("a refused sealed run played its people")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(P, "RESULTS", Path(tmp)), mock.patch.object(P, "measure", never), \
                mock.patch.object(P, "load", never):
            self.assertEqual(P.main(["--split", "sealed_v2", "--examples", "--yes", "--cap-eur", "0.5"]), 2)  # the examples arm only after the roles arm
            doc = {"arm": "roles", "split": "sealed_v2", "journeys": P.fingerprint(P.SPLITS["sealed_v2"]), "rows": []}
            (Path(tmp) / "JPR-900_sealed_v2.json").write_text(json.dumps(doc), encoding="utf-8")
            self.assertEqual(P.main(["--split", "sealed_v2", "--yes", "--cap-eur", "0.5"]), 2)  # the roles arm ran: never again
            self.assertEqual(P.main(["--split", "sealed_v2", "--seed", "3", "--examples"]), 2)  # the gate's seed only
            self.assertFalse(P.roles_passed())


if __name__ == "__main__":
    unittest.main()

"""A written person who does not know says so (S8 B3, 2026-10-25): with ``dont_know`` a question they cannot answer gets
"I don't know." instead of silence; leave to build is still given first; the older journeys, without it, keep their
silence so their results stay comparable."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.journeys import _Person  # noqa: E402


class DontKnowTest(unittest.TestCase):
    def test_a_person_who_does_not_know_says_so_and_still_lets_dclab_build(self):
        answers = {"target": "churned", "entity": "Each subscriber"}
        unsure, silent = _Person(answers, dont_know=True), _Person(answers)
        self.assertEqual(unsure._answer("budget", "How many cases can your team review a day?"), "I don't know.")
        self.assertEqual(unsure.asked[-1], "budget→unsure")
        self.assertIsNone(silent._answer("budget", "How many cases can your team review a day?"))
        self.assertEqual(unsure._answer("entity", "Who is a row about?"), "Each subscriber")  # what they know, they say
        self.assertEqual(unsure.reply("Waiting", "Shall I build it now?"), _Person.GO)  # leave to build before "I don't know"
        self.assertEqual(unsure.reply("Something open?", "Anything else?"), "I don't know.")
        self.assertIsNone(silent.reply("Something open?", "Anything else?"))


if __name__ == "__main__":
    unittest.main()

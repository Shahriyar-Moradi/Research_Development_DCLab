"""34.4, a decision in a person's words: Home asks what is done with each prediction and what a wrong one costs; the
amounts said plainly become a DecisionSpec, and anything said unclearly is left out, never guessed."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.decide import read_spec  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class WordsTests(unittest.TestCase):
    def numbers(self, text):
        spec = read_spec(text)
        return None if spec is None else (spec.action, spec.cost_per_action, spec.value_per_case, spec.cost_false_positive, spec.cost_false_negative)

    def test_plain_amounts_are_read(self):
        self.assertEqual(self.numbers("A call costs 5 euro; a lost customer costs 300."), ("call", 5.0, 0.0, 0.0, 300.0))
        self.assertEqual(self.numbers("Each review takes about €12 of an analyst's time, and a missed fraud costs us €400 on average."),
                         ("review", 12.0, 0.0, 0.0, 400.0))
        self.assertEqual(self.numbers("We call them; it costs about 8 per call. If they leave we lose 1,200 a year."), ("call", 8.0, 0.0, 0.0, 1200.0))
        self.assertEqual(self.numbers("a review costs 2,5 and a missed case costs 1.5k"), ("review", 2.5, 0.0, 0.0, 1500.0))
        self.assertIsNone(read_spec("a review costs 2,5 and a missed case 1.5k"))  # no money word in the second clause: not read
        self.assertEqual(self.numbers("A false alarm costs 20 and a miss costs 500"), ("act", 0.0, 0.0, 20.0, 500.0))
        self.assertEqual(self.numbers("An offer costs $3 and a kept customer is worth $90"), ("offer", 3.0, 90.0, 0.0, 0.0))
        self.assertIn("a lost customer costs 300", read_spec("A call costs 5 euro; a lost customer costs 300.").notes)  # the words kept

    def test_what_is_not_said_plainly_is_not_read(self):
        self.assertIsNone(read_spec("No idea."))
        self.assertIsNone(read_spec(""))
        self.assertIsNone(read_spec("We send a voucher worth $10 to keep a customer worth $250 a year"))  # two amounts in one clause
        self.assertIsNone(read_spec("a call costs 5 and 10% of customers leave"))  # a share is not an amount
        self.assertIsNone(read_spec("A call costs 5."))  # nothing says what a case is worth
        self.assertIsNone(read_spec("A call costs 5; a lost customer costs 300; another lost customer costs 900."))  # said twice, differently

    def test_counts_times_shares_and_totals_are_never_read_as_money(self):
        """The review of 34.4: each of these was read as a cost, then added to the person's solution."""
        for text in ("We call the top 100 customers each week, a lost customer costs us about 300",
                     "We can review 200 a day; a missed fraud costs 400",
                     "a call takes 5 minutes; a lost customer costs 300",
                     "a call costs 5; we lose 3 customers a week",
                     "a call costs 5; churn costs us 2 million a year",
                     "a call costs 5; a lost customer costs 1.2m",
                     "a call costs .5; a lost customer costs 300",
                     "a call costs 5; a lost customer costs 10.000",
                     "a call costs 5; a lost customer costs -300",
                     "a call costs 5; we give a 15 percent discount",
                     "a call costs 5; Customers who churn leave after 12 months"):
            spec = read_spec(text)
            self.assertTrue(spec is None or (spec.cost_per_action, spec.cost_false_negative) == (5.0, 0.0), (text, spec))
            self.assertIsNone(spec, text)  # a cost of acting alone is never enough


if __name__ == "__main__":
    unittest.main()

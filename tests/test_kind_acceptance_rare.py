"""Package 16.2: the rare-event kind's acceptance on the delivery-fraud replica (the decisions of 15.1, through the
kind harness). Its own file so the parallel runner runs it beside the others."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_kind_acceptance import accept  # noqa: E402


class RareEventAcceptanceTest(unittest.TestCase):
    def test_the_rare_event_kind_reaches_its_decisions(self):
        accept(self, "rare_event")


if __name__ == "__main__":
    unittest.main()

"""Package 16.2: each problem kind's acceptance on its replica: the decisions a careful data scientist reaches. A kind is
"supported" only when every one holds, and the committed capability file (``make kinds-check``) must say what a fresh
run says. The rare-event kind runs in ``test_kind_acceptance_rare.py`` (the full engine, about half a minute)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import kinds  # noqa: E402
from dclab_rnd.kinds import acceptance  # noqa: E402

KINDS = ("binary", "multiclass", "regression", "text_tabular")


def accept(test: unittest.TestCase, key: str) -> None:
    result = acceptance.run(kinds.get(key))
    test.assertEqual(result["status"], "supported", [f"{c['decision']}: {c['detail']}" for c in result["checks"] if not c["ok"]])
    stored = {r["kind"]: r for r in acceptance.load()}
    test.assertIn(key, stored, "run make kinds-check: the capability file does not list this kind")
    test.assertEqual(stored[key]["status"], result["status"], "the capability file is stale: run make kinds-check")
    test.assertEqual([c["decision"] for c in stored[key]["checks"]], [c["decision"] for c in result["checks"]],
                     "the capability file is stale: run make kinds-check")


class AcceptanceTest(unittest.TestCase):
    def test_each_kind_reaches_its_decisions(self):
        for key in KINDS:
            with self.subTest(kind=key):
                accept(self, key)

    def test_every_registered_kind_has_a_case(self):
        for kind in kinds.all_kinds():
            self.assertIsNotNone(kind.case(0), kind.key)


if __name__ == "__main__":
    unittest.main()

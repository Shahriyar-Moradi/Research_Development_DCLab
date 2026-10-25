"""34.14 in the studio: a churn project whose table lacks the support contacts the outcome rests on is advised to collect
them; the groups it holds are weighed on the training folds with the model the models stage chose; the holdout is never
read; the advice is kept on the project, recorded, and answered by the API."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
except ImportError as error:  # pragma: no cover
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class CollectStudioTest(unittest.TestCase):
    def test_the_missing_support_contacts_are_named_and_the_holdout_is_never_read(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import collect, engine

        case = kinds.get("binary").case(0)
        case.table = case.table.drop(columns=["support_calls"])  # the support system's export was never joined
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        store.update(pid, goal="Predict which subscribers cancel next month")
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            for stage in ("data", "leakage", "features", "models"):
                engine.execute(store, pid, stage)
            uses = store.get(pid).get("holdout_uses")
            record = collect.measure(store, pid)
        advice = record["document"]
        self.assertEqual(advice["domain"], "churn")
        self.assertEqual(advice["missing"][0]["kind"], "support")
        tenure = next(g for g in advice["groups"] if g["group"] == "tenure")
        self.assertIsNotNone(tenure["low"])  # read with its interval on the training folds
        self.assertIn("not measured", advice["plain"])
        self.assertEqual(store.get(pid).get("holdout_uses"), uses)  # the holdout never read
        self.assertEqual(collect.view(store.get(pid))["document"]["missing"][0]["kind"], "support")
        self.assertTrue(any(e.get("kind") == "collect_next" for e in store.activity(pid, limit=50)))
        with self.assertRaises(collect.CollectError):
            fresh, other = acceptance.build(kinds.get("binary").case(1), Path(tempfile.mkdtemp()))
            collect.measure(fresh, other)  # no models stage yet


if __name__ == "__main__":
    unittest.main()

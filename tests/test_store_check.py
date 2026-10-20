"""Package 32.18: the storing phase's check passes on its scripted day (every kind kept and shaped, nothing twice, the deleted
project kept with its deletion, no planted key or contact, sealed groups apart, every step recorded)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401

    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class StoreCheckTest(unittest.TestCase):
    def test_the_scripted_day_passes_every_check(self):
        from dclab_rnd.store import check

        result = check.run(log=lambda *a: None)
        failed = [c for c in result["checks"] if not c["ok"]]
        self.assertEqual(failed, [], failed)


if __name__ == "__main__":
    unittest.main()

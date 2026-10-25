"""The live first-conversation test on the new people (S8 B3, 2026-10-25) plays dev people only, a fixed few of each
kind, and refuses to send anything unless a priced Claude model serves the Home agent."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import live_people as L  # noqa: E402
from dclab_rnd import understand_students as U  # noqa: E402


def person(kind, n):
    return ("homework_v1", {"id": f"JH-{kind}-0{n:02d}", "kind": kind, "family": "classification", "words": "w", "answers": {}})


class LivePeopleTest(unittest.TestCase):
    def test_dev_people_only_a_fixed_few_of_each_kind(self):
        pool = [person(k, n) for k in ("binary", "regression") for n in range(60)]
        with mock.patch.object(U, "homework", return_value=pool):
            chosen = L.people(per_kind=3)
            again = L.people(per_kind=3)
            named = L.people(only=[chosen[0]["id"]])
        self.assertEqual([p["id"] for p in chosen], [p["id"] for p in again])
        self.assertTrue(all(U.person_split(f"homework_v1:{p['id']}") == "dev" for p in chosen))
        self.assertEqual(sorted({p["kind"] for p in chosen}), ["binary", "regression"])
        self.assertTrue(all(sum(p["kind"] == k for p in chosen) == 3 for k in ("binary", "regression")))
        self.assertEqual([p["id"] for p in named], [chosen[0]["id"]])

    def test_nothing_is_sent_without_a_priced_claude_model(self):
        fake = mock.Mock(model="gpt-5.6-terra", spent_eur=0.0)
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}), \
                mock.patch("dclab_rnd.models.gateway.Gateway.client", return_value=fake), self.assertRaises(SystemExit):
            L.run(1.0, home=Path(os.environ.get("TMPDIR", "/tmp")) / "live-people-test")


if __name__ == "__main__":
    unittest.main()

"""36.7: every trained version is kept with the replay round it was trained on (which must rebuild byte for byte), the
name it is served under and its exam; its history only grows; a version that forgot is retired in one change: the roles
the tuned tier serves under its name switch back first, then its exam record and its own history say retired. A
retirement that stopped half way finishes on the next try and names every role; a retired version is never served again."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_core.model_version import TrainedVersion  # noqa: E402
from dclab_rnd.store import replay, versions  # noqa: E402
from dclab_rnd.store.layout import Layout  # noqa: E402

MT = "m4_workflow_llm"
ACCEPTED = [{"model_version": "dclab-m4-v1", "status": "accepted", "at": "2099-01-01T00:00:00+00:00", "by": "owner", "reason": "gate"}]


class VersionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.layout = Layout(self.tmp / "store")
        base = self.layout.sets(MT, "home_analysis", "v1")
        base.mkdir(parents=True)
        rows = [{"id": f"{MT}:r{i}", "model_type": MT, "messages": [{"role": "user", "content": "Sales by month?"}]} for i in range(5)]
        (base / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (base / "MANIFEST.json").write_text(json.dumps({"name": "home_analysis", "version": "v1"}))
        self.round = replay.build(self.layout, MT)

    def register(self, name="dclab-m4-v1", served_as="dclab-m4:latest"):
        return versions.register(self.layout, MT, name, "Qwen3.5-4B", 1, by="owner", reason="first fine-tune", served_as=served_as)

    def test_a_version_keeps_its_round_its_served_name_and_its_exam(self):
        v = self.register()
        self.assertEqual((v.round, v.round_train_sha256, v.replay_share, v.served_as, v.status),
                         (1, self.round.train_sha256, 0.3, "dclab-m4:latest", "trained"))
        self.assertEqual(v.sets, self.round.sets)
        self.assertEqual(TrainedVersion.read(versions.path(self.layout, MT, "dclab-m4-v1").read_text()), v)
        with self.assertRaises(ValueError):
            self.register()  # one record per version, never rewritten

    def test_any_name_has_a_file(self):
        self.register(name="DCLab-M4-v1")
        self.register(name="_tmp")
        self.assertEqual(versions.standing(self.layout, MT, "DCLab-M4-v1").version, "DCLab-M4-v1")

    def test_a_round_that_does_not_rebuild_or_does_not_exist_is_refused(self):
        train = self.layout.rounds(MT, "round_001") / "train.jsonl"
        train.write_text(train.read_text() + train.read_text().splitlines()[0] + "\n")
        with self.assertRaises(ValueError):
            self.register()
        with self.assertRaises(ValueError):
            versions.register(self.layout, MT, "dclab-m4-v2", "Qwen3.5-4B", 7, by="owner", reason="no such round")

    def test_every_exam_decision_joins_the_history_once_with_what_it_forgot(self):
        self.register()
        rows = [*ACCEPTED, {"model_version": "dclab-m4-v1", "status": "refused", "at": "2099-02-01T00:00:00+00:00", "by": "gate",
                            "reason": "forgot", "forgotten": ["persian"]}]
        with mock.patch("dclab_rnd.store.exams.versions", lambda *a, **k: rows):
            v = versions.standing(self.layout, MT, "dclab-m4-v1")
            self.assertEqual(([e.status for e in v.history], v.history[-1].forgotten), (["trained", "accepted", "refused"], ["persian"]))
            self.assertEqual(len(versions.standing(self.layout, MT, "dclab-m4-v1").history), 3)

    def test_the_roles_served_under_its_served_name_are_found_in_the_routing(self):
        from dclab_rnd.models.routing import open_routing

        home = self.tmp / "home"
        home.mkdir()
        open_routing(home).save({"purposes": {"home_agent": {"serve": "tuned", "serve_model": "dclab-m4:latest"},
                                              "intern": {"serve": "tuned", "serve_model": "another:latest"}}, "history": []})
        serving, _ = versions.served_roles(home, "dclab-m4:latest")
        self.assertEqual(serving, ["home_agent"])  # the served name, not the version's own name

    def test_retiring_switches_the_roles_back_first_then_records_it_once(self):
        self.register()
        calls = []
        with mock.patch.object(versions, "served_roles", lambda home, served: (["home_agent", "intern"], [])), \
             mock.patch("dclab_rnd.models.tuned.retire", lambda purpose, home, reason, **k: calls.append(("role", purpose))), \
             mock.patch("dclab_rnd.store.exams.versions", lambda *a, **k: ACCEPTED), \
             mock.patch("dclab_rnd.store.exams.retire", lambda *a, **k: calls.append(("exam", a[2]))):
            v = versions.retire(self.layout, MT, "dclab-m4-v1", "forgot the Persian section", self.tmp / "home", by="owner")
        self.assertEqual(calls, [("role", "home_agent"), ("role", "intern"), ("exam", "dclab-m4-v1")])
        self.assertEqual((v.status, v.history[-1].roles), ("retired", ["home_agent", "intern"]))

    def test_a_retirement_stopped_half_way_finishes_and_names_every_role(self):
        self.register()
        state = {"left": ["home_agent", "intern"], "fail": True}

        def switch(purpose, home, reason, **k):
            if purpose == "intern" and state["fail"]:
                raise RuntimeError("routing could not be written")
            state["left"].remove(purpose)

        with mock.patch.object(versions, "served_roles", lambda home, served: (list(state["left"]), [])), \
             mock.patch("dclab_rnd.models.tuned.retire", switch):
            with self.assertRaises(RuntimeError):
                versions.retire(self.layout, MT, "dclab-m4-v1", "forgot", self.tmp / "home", by="owner")
            self.assertEqual(versions.standing(self.layout, MT, "dclab-m4-v1").status, "trained")  # nothing in its record yet
            state["fail"] = False
            v = versions.retire(self.layout, MT, "dclab-m4-v1", "forgot", self.tmp / "home", by="owner")
        self.assertEqual(v.history[-1].roles, ["home_agent", "intern"])

    def test_a_retired_version_is_never_served_again(self):
        self.register()
        with mock.patch.object(versions, "served_roles", lambda home, served: ([], [])):
            versions.retire(self.layout, MT, "dclab-m4-v1", "forgot", self.tmp / "home", by="owner")
        self.assertTrue(versions.is_retired(self.layout, "dclab-m4:latest"))
        self.assertFalse(versions.is_retired(self.layout, "dclab-m4-v2:latest"))
        from dclab_rnd.models import tuned

        with mock.patch.object(versions, "is_retired", lambda layout, served: served == "dclab-m4:latest"):
            self.assertTrue(tuned._retired("dclab-m4:latest"))

    def test_an_unknown_version_cannot_be_retired(self):
        with self.assertRaises(ValueError):
            versions.retire(self.layout, MT, "never-trained", "x", self.tmp / "home", by="owner")


if __name__ == "__main__":
    unittest.main()

"""H3 of the harness that learns: the probe gate. A lesson is measured in pairs, never asserted.

The same learning variant is given without and with the one lesson (a scripted solver here: one that does the work
right in one arm and nothing in the other), graded by code, and paired. A lesson that helps over enough pairs becomes
probed; one that hurts loses its passes, is not probed and, hurting three times more than it helped, is retired; with
too few pairs the counters move and the status does not; a probe run again counts nothing twice; a check variant is
refused; the pilot's attempt of the same model and budget serves as the arm without the lesson; a pause stops between
homeworks; a homework that broke makes the probe invalid and nothing is counted.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
    import pandas  # noqa: F401
except ImportError as error:
    MISSING = str(error)

TEXT = ("When one column's values repeat across many rows and each value stands for one thing with lasting traits, split and fold "
        "by it so no thing sits on both sides, even when its name does not look like a key.")
QUIET = lambda *_: None  # noqa: E731


def lesson(home):
    from dclab_core.playbook import Bullet, Evidence, bullet_id
    from dclab_rnd.learning.playbooks import FilePlaybooks

    b = Bullet(id=bullet_id(TEXT, "domain:groups"), text=TEXT, scope="domain:groups", codes=["SPLIT_WRONG:repeated_entity"],
               evidence=[Evidence(pack="pk-000000000000", step="solution", code="SPLIT_WRONG:repeated_entity", grader="grader_v1")], by="rule:test")
    FilePlaybooks(home).add("groups", b, reason="a test lesson")
    return b.id


def solver(right_with: bool):
    """The intern_attempt the probe calls, with the client chosen by the arm: right in one arm, idle in the other."""
    from test_learning_h3 import KeyedModel

    from dclab_rnd.learning import solver as S
    from dclab_rnd.learning.packs import read_public

    real = S.intern_attempt

    def attempt(folder, client, key, playbook=None, **kw):
        right = (playbook is not None) == right_with
        model = KeyedModel(read_public(folder)[0], key) if right else KeyedModel(None)
        return real(folder, model, key, playbook=playbook, **kw)

    return attempt


def book(home, bullet):
    from dclab_rnd.learning.playbooks import FilePlaybooks

    b, _ = FilePlaybooks(home).book("groups")
    return next(x for x in b.bullets if x.id == bullet)


class Client:
    model = "scripted"


@unittest.skipIf(MISSING, MISSING)
class ProbeTests(unittest.TestCase):
    def run_probe(self, home, bullet, right_with=True, **kw):
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.learning import probe

        with fast_profile(), mock.patch.object(probe, "intern_attempt", solver(right_with)):
            return probe.run_probe(home, Client(), "scripted", bullet, log=QUIET, **{"reuse_pilot": False, **kw})

    def test_a_lesson_that_helps_over_enough_pairs_is_probed_and_counted(self):
        from dclab_rnd.learning import probe

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(probe, "MIN_PAIRS", 2), mock.patch.object(probe, "MIN_REPEATS", 1):
            home = Path(tmp)
            bullet = lesson(home)
            result = self.run_probe(home, bullet, n=2)
            self.assertTrue(result["verdict"].startswith("probed"), result["verdict"])
            self.assertGreater(result["summary"]["mean_gain"], 0)
            b = book(home, bullet)
            self.assertEqual((b.status, b.shown, b.helpful, b.harmful), ("probed", 2, 2, 0))
            with_rows = [r for r in result["rows"] if r["arm"] == "with"]
            self.assertTrue(all(r["bullet"] == bullet for r in with_rows))
            session = json.loads(next((home / with_rows[0]["kept"] / "intern").glob("*.json")).read_text())
            self.assertIn(TEXT[:40], session["messages"][1]["content"])  # the lesson in the first user message
            self.assertNotIn(TEXT[:40], session["messages"][0]["content"])  # never in the system prompt
            idle = json.loads(next((home / [r for r in result["rows"] if r["arm"] == "without"][0]["kept"] / "intern").glob("*.json")).read_text())
            self.assertNotIn(TEXT[:40], json.dumps(idle["messages"]))

            again = self.run_probe(home, bullet, n=2)  # run again: every homework found, nothing counted twice
            self.assertEqual(again["summary"]["counted_now"], {"shown": 0, "helpful": 0, "harmful": 0})
            self.assertEqual(len(list((home / "learning_attempts").glob("*/*-probe-*"))), 4)
            self.assertEqual(book(home, bullet).shown, 2)

    def test_a_lesson_that_hurts_loses_passes_is_not_probed_and_is_retired(self):
        from dclab_rnd.learning import probe

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(probe, "MIN_PAIRS", 2), mock.patch.object(probe, "MIN_REPEATS", 1):
            home = Path(tmp)
            bullet = lesson(home)
            result = self.run_probe(home, bullet, right_with=False, n=3)
            self.assertTrue(result["verdict"].startswith("not probed"), result["verdict"])
            self.assertEqual(result["summary"]["passes_lost"], 3)
            b = book(home, bullet)
            self.assertEqual((b.harmful, b.status), (3, "retired"))  # harmful minus helpful reached three: the playbook's rule

    def test_enough_pairs_without_three_repeats_of_each_variant_keep_the_status(self):
        from dclab_rnd.learning import probe

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(probe, "MIN_PAIRS", 2):
            home = Path(tmp)
            bullet = lesson(home)
            result = self.run_probe(home, bullet, n=2)  # two pairs that help, one repeat each: the plan asks three
            self.assertTrue(result["verdict"].startswith("too few pairs"), result["verdict"])
            self.assertEqual(book(home, bullet).status, "draft")

    def test_a_lost_pass_alone_blocks_a_lesson_whose_mean_gain_is_positive(self):
        from test_learning_h3 import KeyedModel

        from dclab_core import variants as V
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.learning import probe
        from dclab_rnd.learning import solver as S
        from dclab_rnd.learning.packs import read_public

        seeds = V.seeds("groups", "learning", 3)
        real = S.intern_attempt

        def attempt(folder, client, key, playbook=None, **kw):  # the lesson helps on two variants and breaks a solved third
            right = (playbook is not None) != (key.seed == seeds[2])
            return real(folder, KeyedModel(read_public(folder)[0], key) if right else KeyedModel(None), key, playbook=playbook, **kw)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(probe, "MIN_PAIRS", 3), mock.patch.object(probe, "MIN_REPEATS", 1), \
                fast_profile(), mock.patch.object(probe, "intern_attempt", attempt):
            home = Path(tmp)
            bullet = lesson(home)
            result = probe.run_probe(home, Client(), "scripted", bullet, n=3, reuse_pilot=False, log=QUIET)
            self.assertGreater(result["summary"]["mean_gain"], 0)
            self.assertEqual(result["summary"]["passes_lost"], 1)
            self.assertTrue(result["verdict"].startswith("not probed"), result["verdict"])
            self.assertEqual(book(home, bullet).status, "draft")

    def test_counts_that_retire_a_lesson_win_over_a_promotion_in_the_same_run(self):
        from dclab_rnd.learning import probe
        from dclab_rnd.learning.playbooks import FilePlaybooks

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(probe, "MIN_PAIRS", 1), mock.patch.object(probe, "MIN_REPEATS", 1), \
                mock.patch.object(probe, "_count_once", lambda home, books, family, lineage, bullet, pairs, gains:
                                  books.count(family, bullet, {"harmful": 3}, by="probe", reason="earlier probes hurt", lineage=lineage) and {}):
            home = Path(tmp)
            bullet = lesson(home)
            result = self.run_probe(home, bullet, n=1)  # this pair helps, but the counts retire it first
            self.assertEqual(book(home, bullet).status, "retired")
            self.assertTrue(result["verdict"].startswith("not probed (retired)"), result["verdict"])
            self.assertEqual(len(FilePlaybooks(home).book("groups")[1]), 0)  # no status change was refused: none was tried

    def test_a_pair_never_mixes_two_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bullet = lesson(home)
            first = self.run_probe(home, bullet, n=1, budget={"max_minutes": 45})
            again = self.run_probe(home, bullet, n=1, budget={"max_minutes": 15})
            self.assertEqual(first["budget"]["max_minutes"], 45)
            self.assertTrue(all(r["budget"]["max_minutes"] == 15 for r in again["rows"]))  # done again under its own budget
            self.assertEqual(len(list((home / "learning_attempts").glob("*/*-probe-*"))), 4)

    def test_a_homework_the_server_broke_is_marked_invalid_and_done_again(self):
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.learning import probe

        class Down:
            model = "scripted"

            def complete(self, *a, **kw):
                raise ConnectionError("connection refused")

        from dclab_rnd.learning import solver as S

        real = S.intern_attempt
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bullet = lesson(home)
            with fast_profile(), mock.patch.object(probe, "intern_attempt", lambda folder, client, key, **kw: real(folder, Down(), key, **kw)):
                broken = probe.run_probe(home, Client(), "scripted", bullet, n=1, reuse_pilot=False, log=QUIET)
            self.assertTrue(broken["verdict"].startswith("invalid"), broken["verdict"])
            folders = list((home / "learning_attempts").glob("*/*-probe-*"))
            self.assertTrue(folders and all((f / "invalid.json").exists() for f in folders))  # never the model's fail in training
            fixed = self.run_probe(home, bullet, n=1)  # the server back: both arms done again, the pair counted
            self.assertTrue(fixed["verdict"].startswith("too few pairs"), fixed["verdict"])
            self.assertEqual(book(home, bullet).shown, 1)

    def test_a_hosted_probe_measures_only_and_moves_no_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bullet = lesson(home)
            result = self.run_probe(home, bullet, n=1, count=False)
            self.assertTrue(result["verdict"].startswith("measured only"), result["verdict"])
            b = book(home, bullet)
            self.assertEqual((b.shown, b.helpful, b.harmful, b.status), (0, 0, 0, "draft"))

    def test_too_few_pairs_move_the_counters_and_keep_the_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bullet = lesson(home)
            result = self.run_probe(home, bullet, n=1)
            self.assertTrue(result["verdict"].startswith("too few pairs"), result["verdict"])
            b = book(home, bullet)
            self.assertEqual((b.status, b.shown, b.helpful), ("draft", 1, 1))

    def test_a_check_variant_is_never_read(self):
        from dclab_core import variants as V

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bullet = lesson(home)
            with self.assertRaises(ValueError):
                self.run_probe(home, bullet, seeds=V.seeds("groups", "check", 1))
            self.assertFalse((home / "learning_attempts").exists())

    def test_the_pilots_attempt_serves_as_the_arm_without_the_lesson(self):
        from test_learning_h3 import KeyedModel

        from dclab_rnd.learning.pilot import run_pilot

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            run_pilot(home, KeyedModel(None), "scripted", n=1, log=QUIET)
            bullet = lesson(home)
            result = self.run_probe(home, bullet, n=1, reuse_pilot=True)
            self.assertTrue(result["pairs"][0]["without_from_pilot"])
            self.assertEqual(len(list((home / "learning_attempts").glob("*/*-probe-*"))), 1)  # only the arm with the lesson ran
            other = self.run_probe(home, bullet, n=1, reuse_pilot=True, budget={"max_minutes": 99})
            self.assertFalse(other["pairs"][0]["without_from_pilot"])  # another budget: the pilot's attempt is not this arm

    def test_a_pause_stops_between_homeworks_and_the_probe_goes_on(self):
        from dclab_rnd.learning.probe import pause_file

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bullet = lesson(home)
            pause_file(home).parent.mkdir(parents=True)
            pause_file(home).write_text("pause")
            paused = self.run_probe(home, bullet, n=1)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(book(home, bullet).shown, 0)
            done = self.run_probe(home, bullet, n=1)
            self.assertEqual(done["status"], "completed")

    def test_a_broken_homework_makes_the_probe_invalid_and_counts_nothing(self):
        from dclab_rnd.learning import probe

        def broken(*a, **kw):
            raise ConnectionError("connection refused")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(probe, "intern_attempt", broken):
            home = Path(tmp)
            bullet = lesson(home)
            result = probe.run_probe(home, Client(), "scripted", bullet, n=1, reuse_pilot=False, log=QUIET)
            self.assertTrue(result["verdict"].startswith("invalid"), result["verdict"])
            self.assertEqual(book(home, bullet).shown, 0)

    def test_a_probe_without_a_model_or_lesson_is_refused(self):
        from dclab_rnd.learning.probe import run_probe

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_probe(Path(tmp), None, "x", "B-0000000000", log=QUIET)
            with self.assertRaises(ValueError):
                run_probe(Path(tmp), Client(), "x", "B-0000000000", log=QUIET)


if __name__ == "__main__":
    unittest.main()

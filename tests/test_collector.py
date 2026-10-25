"""The collector (S2 of the 100-students plan, 2026-10-25): guarded, one run at a time, capped by the month with its
cap counted before Claude is asked, never runs a sealed table, keeps every decision in its own workspaces even inside a
job, sets copies of known tables aside, survives a night offline, keeps a record of every run, and builds a fresh exam
set from its own tables only, once, then frozen."""

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import collector, real_shadow  # noqa: E402

STATE = {"column": "c", "name_tokens": ["c"], "dtype": "float", "uniqueness": "under_1pct", "nulls": "none", "cardinality": "low",
         "value_pattern": "digits"}


class Halt(BaseException):
    """A job's Stop (``jobs.worker.Stopped`` is a BaseException too)."""


def night() -> float:
    return datetime.now().replace(hour=2, minute=0, second=0, microsecond=0).timestamp()


class CollectorRunTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_COLLECTOR": "1", "DCLAB_COLLECTOR_HOURS": "0-24"})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.train, self.eval = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        self.lock = Path(tempfile.mkdtemp()) / "run.lock"
        self.ran = []
        splits = {1: "train", 2: "dev", 3: "sealed"}
        tables = [{"did": d, "target": "y", "columns": {"c": {}}} for d in (1, 2, 3)]

        def run_one(d, home, retry_failed=False):
            self.ran.append((d["did"], home))
            (home / "done").mkdir(parents=True, exist_ok=True)
            (home / "done" / f"{d['did']}.json").write_text("{}")
            return {"did": d["did"], "decisions": 5}
        self.silver = mock.MagicMock(return_value={"asked": {"p": 4}, "spent_eur": 0.4})
        patches = [mock.patch.object(collector, "free_gb", return_value=50.0), mock.patch.object(collector, "learning_busy", return_value=False),
                   mock.patch.object(collector, "busy", return_value=None),
                   mock.patch.object(collector, "gather", return_value={"new_train": 1}), mock.patch.object(collector, "homework", return_value={}),
                   mock.patch.object(collector, "exam_status", return_value={"ready": False}),
                   mock.patch.object(real_shadow, "TRAIN_HOME", self.train), mock.patch.object(real_shadow, "EVAL_HOME", self.eval),
                   mock.patch.object(real_shadow, "LOCK", self.lock),
                   mock.patch.object(real_shadow, "split_of", side_effect=lambda did: splits[int(did)]),
                   mock.patch.object(real_shadow, "_tables", side_effect=lambda limit=None: [t for t in tables if splits[t["did"]] != "sealed"]),
                   mock.patch.object(real_shadow, "run_one", side_effect=run_one),
                   mock.patch.object(real_shadow, "measure", return_value={}),
                   mock.patch.object(real_shadow, "silver", self.silver)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def run_it(self, **kw):
        return collector.run(self.home, **{"tables": 5, "run_eur": 1.0, "monthly_eur": 5.0, "log": lambda m: None, **kw})

    def test_a_run_shadows_new_train_and_dev_tables_never_a_sealed_one_and_keeps_its_record(self):
        out = self.run_it()
        self.assertEqual(out["status"], "done")
        self.assertEqual(sorted(d for d, _ in self.ran), [1, 2])  # the sealed table (3) is never run on
        self.assertEqual(dict(self.ran)[1], self.train)
        self.assertEqual(dict(self.ran)[2], self.eval)
        self.assertEqual((out["spent_eur"], out["month_spent_eur"]), (0.4, 0.4))  # the cap reserved, then settled to what was spent
        self.assertIn("checkpoint", self.silver.call_args.kwargs)  # a job's Stop reaches Claude's step
        self.assertEqual(len(list((self.home / "runs").glob("*.json"))), 1)
        self.ran.clear()
        time.sleep(1.1)  # a second record of its own
        self.run_it()
        self.assertEqual(self.ran, [])  # tables done are not run again
        self.assertEqual(collector.spent_this_month(self.home), 0.8)

    def test_the_month_cap_stops_claude_and_the_guards_stop_the_run(self):
        collector._save(self.home, {"months": {collector._month(): 5.0}})
        out = self.run_it()
        self.assertIn("cap is reached", out["silver"]["skipped"])
        self.assertEqual(out["spent_eur"], 0.0)
        with mock.patch.object(collector, "free_gb", return_value=1.0):
            self.assertEqual(collector.run(self.home, log=lambda m: None)["status"], "skipped")
        with mock.patch.object(collector, "learning_busy", return_value=True):
            out = collector.run(self.home, log=lambda m: None)
        self.assertEqual(out["status"], "skipped")
        self.assertIn("learning", out["why"][0])
        with mock.patch.object(collector, "busy", return_value="the Mac is busy (load 130 on 8 cores)"):
            self.assertIn("busy", collector.run(self.home, log=lambda m: None)["why"][0])

    def test_one_run_at_a_time(self):
        with real_shadow.exclusive(self.lock):  # a hand run of the real-table shadow run, or another worker's
            out = self.run_it()
        self.assertEqual(out["status"], "skipped")
        self.assertIn("another", out["why"][0])
        self.assertEqual(self.ran, [])
        self.silver.assert_not_called()
        self.assertEqual(self.run_it()["status"], "done")  # the lock is free again

    def test_a_run_killed_or_stopped_while_claude_answers_keeps_its_cap_counted(self):
        self.silver.side_effect = Halt()
        with self.assertRaises(Halt):  # a person's Stop passes through, never swallowed
            self.run_it(run_eur=0.5)
        self.assertEqual(collector.spent_this_month(self.home), 0.5)  # what it cost is not known: the whole cap stays counted
        self.silver.side_effect = RuntimeError("a bug")
        out = self.run_it(run_eur=0.5)
        self.assertIn("failed", out["silver"])
        self.assertEqual(collector.spent_this_month(self.home), 1.0)
        self.silver.side_effect = SystemExit("no model is configured")
        self.run_it(run_eur=0.5)
        self.assertEqual(collector.spent_this_month(self.home), 1.0)  # nothing asked: nothing counted

    def test_a_skipped_run_tries_again_within_the_hour_a_stopped_one_waits_a_night(self):
        with mock.patch.object(collector, "free_gb", return_value=1.0):
            collector.run(self.home, log=lambda m: None)
        now = time.time()
        self.assertFalse(collector.due(self.home, now + 60))
        self.assertTrue(collector.due(self.home, now + collector.RETRY_SECONDS + 60))  # not a whole night lost to a guard
        collector.failed(self.home, now, reason="stopped")
        self.assertFalse(collector.due(self.home, now + collector.RETRY_SECONDS + 60))  # a person stopped it: the next night
        self.assertTrue(collector.due(self.home, now + collector.NIGHTLY_SECONDS))


class CollectorJobTest(unittest.TestCase):
    """Inside a job the services are bound and their ledger wins over an installed one (the review of S2, finding 1)."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.home, self.train, self.eval = (Path(tempfile.mkdtemp()) for _ in range(3))
        from dclab_rnd.ledger import shadow

        def run_one(d, home, retry_failed=False):
            from dclab_rnd.ledger import capture

            self.shadow_seen = os.environ.get("DCLAB_DECISION_SHADOW")
            capture._put("column.semantic_role", ("column", "c"), [capture.answer("rule", "datacard:purposes", "numeric")], "numeric", "rule",
                         state=STATE, options=real_shadow.ROLES, draft=f"openml-{d['did']}")
            (home / "done" / f"{d['did']}.json").write_text("{}")
            return {"did": d["did"], "decisions": 1}
        splits = {1: "train", 2: "dev"}
        patches = [mock.patch.object(collector, "guards", return_value=[]), mock.patch.object(collector, "free_gb", return_value=50.0),
                   mock.patch.object(collector, "gather", return_value={}), mock.patch.object(collector, "exam_status", return_value={}),
                   mock.patch.object(real_shadow, "TRAIN_HOME", self.train), mock.patch.object(real_shadow, "EVAL_HOME", self.eval),
                   mock.patch.object(real_shadow, "LOCK", self.home / "run.lock"),
                   mock.patch.object(real_shadow, "split_of", side_effect=lambda did: splits[int(did)]),
                   mock.patch.object(real_shadow, "_tables", return_value=[{"did": 1}, {"did": 2}]),
                   mock.patch.object(real_shadow, "run_one", side_effect=run_one), mock.patch.object(real_shadow, "measure", return_value={}),
                   mock.patch.object(shadow, "answerers", return_value=())]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_inside_a_job_every_decision_stays_in_the_collectors_workspaces(self):
        from dclab_rnd import context, ledger
        from dclab_rnd.ledger import open_ledger

        server = open_ledger(Path(tempfile.mkdtemp()))
        fallback = object()
        ledger.install(fallback)
        self.addCleanup(ledger.install, None)
        with context.using(SimpleNamespace(ledger=server)):  # the worker's _acting binds the server's services
            out = collector.run(self.home, tables=1, run_eur=0.0, monthly_eur=0.0, log=lambda m: None)
        self.assertEqual(out["status"], "done")
        self.assertEqual(server.records(), [])  # never the server's workspace: a dev table never becomes its training data
        train, dev = open_ledger(self.train).records(), open_ledger(self.eval).records()
        self.assertEqual(({r.draft for r in train}, {r.draft for r in dev}), ({"openml-1"}, {"openml-2"}))
        self.assertEqual({r.consent for r in train}, {"train"})
        self.assertNotIn("train", {r.consent for r in dev})
        self.assertEqual(self.shadow_seen, "1")  # the students whispered inside
        self.assertIs(ledger._INSTALLED.get("ledger"), fallback)  # the process as it was: its fallback ledger
        self.assertEqual(os.environ.get("DCLAB_DECISION_SHADOW"), "0")  # and its shadow setting
        hw = out["homework"]["column.semantic_role"]
        self.assertEqual((hw["decisions"], hw["tables"], hw["labelled"], hw["bar"]), (1, 1, 0, False))

    def test_the_bar_counts_labelled_decisions_never_decisions_alone(self):
        from dclab_rnd.ledger import capture, install, label, open_ledger

        book = open_ledger(self.train)
        install(book)
        self.addCleanup(install, None)
        for i in range(320):  # 320 decisions on 32 tables, ten of them labelled by a person, one by a model
            rec = capture._put("column.semantic_role", ("column", f"c{i}"), [capture.answer("rule", "datacard:purposes", "numeric")], "numeric",
                               "rule", state=STATE, options=real_shadow.ROLES, draft=f"openml-{i % 32}")
            if i < 10:
                label(rec, "numeric", "blind", "owner", weight=2.0)
            elif i == 10:
                label(rec, "numeric", "model", "model:claude")
        with mock.patch("dclab_rnd.ledger.labels.queue", return_value={"points": {}}):
            hw = collector.homework()["column.semantic_role"]
        self.assertEqual((hw["decisions"], hw["tables"], hw["labelled"], hw["labelled_tables"]), (320, 32, 10, 10))
        self.assertFalse(hw["bar"])  # 320 decisions are not homework: 10 truths are


class GatherTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        from dclab_rnd.lake import openml

        self.cache = Path(tempfile.mkdtemp())
        self.collected = self.cache / "collector"
        (self.cache / "datasets").mkdir(parents=True)
        (self.cache / "datasets" / "43419.json").write_text(json.dumps(
            {"did": 43419, "name": "depression", "columns": [{"name": n} for n in ("age", "sex", "sleep", "mood", "score")]}))
        (self.cache / "datasets" / "7.json").write_text(json.dumps({"did": 7, "skipped": "licence"}))
        self.candidates = [{"did": 7}, {"did": 46757}, {"did": 50}, {"did": 51}]
        for target, name, value in ((openml, "CACHE", self.cache), (collector, "COLLECTED", self.collected),
                                    (collector.time, "sleep", lambda s: None)):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(openml, "candidates", side_effect=lambda limit: self.candidates)
        p.start()
        self.addCleanup(p.stop)

    def profile(self, did, cache=None):
        names = ("age", "sex", "sleep", "mood", "score") if did == 46757 else ("x1", "x2", "x3", f"own{did}")
        rec = {"did": did, "name": "depression_2020" if did == 46757 else f"t{did}", "columns": [{"name": n} for n in names], "skipped": None}
        (Path(cache) / "datasets").mkdir(parents=True, exist_ok=True)
        (Path(cache) / "datasets" / f"{did}.json").write_text(json.dumps(rec))
        return rec

    def test_new_candidates_leave_out_every_table_known_to_the_lake_or_the_collector(self):
        self.assertEqual([d["did"] for d in collector.new_candidates(10)], [46757, 50, 51])
        (self.collected / "datasets").mkdir(parents=True)
        (self.collected / "datasets" / "50.json").write_text("{}")
        self.assertEqual([d["did"] for d in collector.new_candidates(10)], [46757, 51])

    def test_a_copy_of_a_known_table_is_set_aside_never_homework_nor_exam(self):
        from dclab_rnd.lake import openml

        with mock.patch.object(openml, "profile_dataset", side_effect=self.profile), \
                mock.patch("dclab_rnd.real_homework.fetch_one", side_effect=lambda rec, derived: {"did": rec["did"]}), \
                mock.patch.object(real_shadow, "split_of", return_value="sealed"):
            counts = collector.gather(10, log=lambda m: None)
        self.assertEqual((counts["twin"], counts["new_sealed"]), (1, 2))
        twin = json.loads((self.collected / "datasets" / "46757.json").read_text())
        self.assertEqual(twin["skipped"], "near-duplicate of openml:43419")
        self.assertNotIn(46757, [r["did"] for r in collector.collected_records()])
        self.assertIsNone(collector.twin_of({"did": 9, "name": "x", "columns": [{"name": "a"}, {"name": "b"}]}, collector.known_tables()))

    def test_a_night_offline_leaves_no_mark_and_ends_the_gathering(self):
        from dclab_rnd.lake import openml

        asked = []

        def offline(did, cache=None):
            asked.append(did)
            raise TimeoutError("timed out")
        with mock.patch.object(openml, "profile_dataset", side_effect=offline):
            counts = collector.gather(10, log=lambda m: None)
        self.assertEqual((counts["network"], asked), (3, [46757, 50, 51]))
        self.assertFalse((self.collected / "datasets").is_dir() and any((self.collected / "datasets").iterdir()))  # tried again next run
        self.assertEqual([d["did"] for d in collector.new_candidates(10)], [46757, 50, 51])

    def test_a_derivation_lost_to_the_network_is_tried_again(self):
        from dclab_rnd.lake import openml

        replies = iter([{"failed": "URLError: timed out"}, {"did": 50}, {"did": 51}, {"did": 50}])
        with mock.patch.object(openml, "profile_dataset", side_effect=self.profile), \
                mock.patch("dclab_rnd.real_homework.fetch_one", side_effect=lambda rec, derived: next(replies)), \
                mock.patch.object(real_shadow, "split_of", return_value="train"):
            self.candidates = [{"did": 50}, {"did": 51}]
            first = collector.gather(10, log=lambda m: None)
            self.assertEqual((first["network"], first["new_train"]), (1, 1))
            (self.collected / "derived").mkdir(parents=True, exist_ok=True)
            (self.collected / "derived" / "51.json").write_text("{}")
            second = collector.gather(10, log=lambda m: None)  # 50 is profiled, not derived: derived again first
        self.assertEqual(second["new_train"], 1)

    def test_the_lake_never_profiles_a_table_the_collector_holds(self):
        from dclab_rnd.lake import openml

        (self.collected / "datasets").mkdir(parents=True)
        (self.collected / "datasets" / "50.json").write_text("{}")
        with mock.patch.object(openml, "profile_dataset", return_value={"skipped": None}) as profile, \
                mock.patch.object(openml.time, "sleep"):
            counts = openml.fetch(10, log=lambda m: None)
        self.assertEqual(counts["collector"], 1)
        self.assertNotIn(50, [c.args[0] for c in profile.call_args_list])


class ExamTest(unittest.TestCase):
    def setUp(self):
        self.collected = Path(tempfile.mkdtemp())
        p = mock.patch.object(collector, "COLLECTED", self.collected)
        p.start()
        self.addCleanup(p.stop)

    def test_the_exam_counts_only_derived_sealed_tables_per_point(self):
        (self.collected / "datasets").mkdir()
        (self.collected / "derived").mkdir()
        for did, ident, derived in ((1, True, {"columns": {"c": {}}, "task": {"rule": "binary"}}), (2, False, {"columns": {"c": {}}}), (3, True, {"failed": "x"}),
                                    (4, True, {"columns": {"c": {}}, "task": {"rule": "binary"}})):
            (self.collected / "datasets" / f"{did}.json").write_text(json.dumps({"did": did, "columns": [{"name": "c", "is_row_identifier": ident}]}))
            (self.collected / "derived" / f"{did}.json").write_text(json.dumps(derived))
        with mock.patch.object(real_shadow, "split_of", side_effect=lambda did: "dev" if int(did) == 4 else "sealed"):
            status = collector.exam_status()
        self.assertEqual(status["per_point"], {"column.semantic_role": 2, "column.is_identifier": 1, "task.framing": 1})
        self.assertEqual(status["sealed_tables"], 2)  # the failed derivation (3) and the dev table (4) count for none
        self.assertIn("column.is_identifier", status["thin"])
        self.assertFalse(status["ready"])

    def test_the_fresh_exam_set_waits_for_enough_sealed_tables_then_stays_frozen(self):
        exam = Path(tempfile.mkdtemp()) / "real_v2"
        with mock.patch.object(collector, "EXAM", exam), mock.patch.object(collector, "exam_status", wraps=None) as status, \
                mock.patch("dclab_rnd.real_homework.build", return_value={"sets": {}}) as build:
            status.return_value = {"sealed_tables": 3, "needed": 50, "built": False, "ready": False}
            self.assertIn("waiting", collector.build_exam()["note"])
            build.assert_not_called()
            status.return_value = {"sealed_tables": 60, "needed": 50, "built": False, "ready": True}
            collector.build_exam()
            self.assertEqual(build.call_args.kwargs["version"], "real_v2")
            status.return_value = {"sealed_tables": 80, "needed": 50, "built": True, "ready": True}
            self.assertIn("frozen", collector.build_exam()["note"])
            self.assertEqual(build.call_count, 1)

    def test_due_only_when_switched_on_and_at_night(self):
        home = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_COLLECTOR": "1", "DCLAB_COLLECTOR_HOURS": ""}):
            self.assertTrue(collector.due(home, night()))
            self.assertFalse(collector.due(home, night() + 12 * 3600))  # 14:00: never in the day
            collector._save(home, {"clock": night() - 3600})
            self.assertFalse(collector.due(home, night()))  # a run an hour ago
        with mock.patch.dict(os.environ, {"DCLAB_COLLECTOR": "0"}):
            self.assertFalse(collector.due(home, night() + 3 * 86400))


if __name__ == "__main__":
    unittest.main()

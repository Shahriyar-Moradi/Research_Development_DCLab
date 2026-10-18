"""Package 29.7: the loop made continuous. A scheduled job keeps a workspace's students up with its labels: it rebuilds the
sets, retrains the students whose homework grew by a margin (and only those), compares each new card with the rule on dev,
demotes a routed model whose new labels fall below its bar (an audited change by DCLab), and writes the register, whose
history shows the new card. Promotion is never the loop's: it stays a person's audited switch."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    HAVE_ML = True
except ImportError:  # .venv-agent
    HAVE_ML = False

POINT = "column.is_identifier"
NAMES = [("order_id", ["order", "id"]), ("product_code", ["product", "code"]), ("sku_code", ["sku", "code"]), ("customer_id", ["customer", "id"]),
         ("amount", ["amount"]), ("city", ["city"]), ("ticket_no", ["ticket", "no"]), ("zone", ["zone"])]


def _state(name, tokens, unique):
    return {"column": name, "name_tokens": tokens, "dtype": "string" if "code" in tokens else "integer", "uniqueness": unique,
            "nulls": "none", "cardinality": "very_high" if unique == "all" else "low"}


def label_projects(book, first, count, seed=0):
    """``count`` projects' identifier decisions, each labelled by the company's convention (a unique code is not an identifier)."""
    from dclab_rnd import ledger

    rng = random.Random(seed + first)
    for p in range(first, first + count):
        for name, tokens in NAMES:
            unique = rng.choice(["all", "all", "under_1pct"])
            truth = unique == "all" and "code" not in tokens
            record = ledger.decide(POINT, ("column", name), answers=[ledger.answer("rule", "rule:id", unique == "all")], value_used=unique == "all",
                                   used="rule", state=_state(name, tokens, unique), options=["yes", "no"], project=f"p{p:03d}", ledger=book)
            if truth != (unique == "all"):  # a person corrected the rule, having seen it
                ledger.label(record, truth, "user_decision_exposed", "analyst", ledger=book)
            else:
                ledger.label(record, truth, "blind", "reviewer", weight=1.0, ledger=book)


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class LoopTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import audit, ledger
        from dclab_rnd.models.routing import FileRouting

        self.home = Path(tempfile.mkdtemp())
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        self.routing = FileRouting(self.home / "model_routing.json")
        self.audit = audit.FileAudit(self.home / "audit.jsonl")
        for patcher in (mock.patch("dclab_rnd.own_models.base_model", return_value=None),  # the workspace's own data alone: no base read from disk
                        mock.patch.dict("os.environ", {"DCLAB_STUDENTS_LOOP": "1"})):  # the test runner turns the loop off; these tests turn it on
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_loop(self):
        from dclab_rnd import students_loop

        return students_loop.run(self.home, self.book, self.routing, self.audit, measure=False)

    def test_a_first_run_fits_the_point_with_homework_and_the_register_shows_its_card(self):
        from dclab_rnd import students

        label_projects(self.book, 0, 40)
        got = self.run_loop()
        self.assertEqual(got["retrained"], [POINT])
        card = got["cards"][POINT]
        self.assertGreater(card["dev_accuracy"], card["rule_dev_accuracy"])  # the dev comparison: the company's convention, learned
        last = json.loads((self.home / "students" / "register.jsonl").read_text().splitlines()[-1])
        self.assertEqual(last["points"][POINT]["own"], card["answerer"])
        self.assertEqual(students.history(self.home)[POINT][-1]["own"], card["answerer"])
        state = json.loads((self.home / "students" / "loop.json").read_text())
        self.assertEqual(state["fitted"][POINT], card["homework"])

    def test_homework_grown_by_the_margin_retrains_and_a_little_does_not(self):
        from dclab_rnd import students, students_loop

        label_projects(self.book, 0, 40)
        first = self.run_loop()["cards"][POINT]["answerer"]
        label_projects(self.book, 40, 1)  # one more project: below the margin
        quiet = self.run_loop()
        self.assertEqual(quiet["retrained"], [])
        self.assertIn(POINT, quiet["waiting"])  # said, with how much more it needs
        label_projects(self.book, 41, int(40 * students_loop.MARGIN) + 2)
        again = self.run_loop()
        self.assertEqual(again["retrained"], [POINT])
        self.assertNotEqual(again["cards"][POINT]["answerer"], first)  # a new card: trained on more
        owns = [h["own"] for h in students.history(self.home)[POINT]]
        self.assertEqual((owns[0], owns[-1]), (first, again["cards"][POINT]["answerer"]))

    def test_a_planted_slip_below_the_bar_demotes_the_routed_model_and_the_audit_says_so(self):
        from dclab_rnd import ledger
        from dclab_rnd.ledger import routing

        name = "own@slipping"
        with mock.patch.object(routing, "passed", return_value={name: {"point": POINT}}):
            routing.switch(self.routing, self.audit, POINT, name, 1, "reviewer", "flag beside the rule", home=self.home)
        for i in range(routing.MIN_LABELS + 5):  # the slip: it says "identifier" where the labels say not
            record = ledger.decide(POINT, ("column", f"code_{i}"), answers=[ledger.answer("rule", "rule:id", True), ledger.answer("classifier", name, True)],
                                   value_used=True, used="rule", state=_state(f"code_{i}", ["code"], "all"), options=["yes", "no"],
                                   project=f"q{i:03d}", ledger=self.book)
            ledger.label(record, i % 5 == 0, "blind", "reviewer", weight=1.0, ledger=self.book)
        got = self.run_loop()
        self.assertEqual([d["decision_point"] for d in got["demoted"]], [POINT])
        self.assertEqual(routing.entry(self.routing.get(), POINT)["level"], 0)
        _, rows = self.audit.query(10, 0, kind="routing")
        self.assertEqual((rows[0]["actor"], rows[0]["who"]), ("agent", "DCLab (automatic demotion)"))
        self.assertFalse(any(e.get("to", {}).get("level", 0) > (e.get("from") or {}).get("level", 0) and e.get("by") == routing.SYSTEM
                             for e in self.routing.get().get("history") or []))  # the loop never promotes

    def test_a_model_that_serves_is_never_replaced_and_a_pass_belongs_to_the_version_that_sat_it(self):
        from dclab_rnd import own_models
        from dclab_rnd.ledger import routing

        label_projects(self.book, 0, 40)
        first = self.run_loop()["cards"][POINT]["answerer"]
        results = own_models.root(self.home) / "results"
        results.mkdir(parents=True, exist_ok=True)
        version = first.split("@")[1]
        (results / f"{POINT}.json").write_text(json.dumps({"point": POINT, "result": "pass", "gate": {"version": "decision_gate_v1"},
                                                           "model": {"name": "own", "version": version}, "verdict": {}}))
        self.assertIn(first, own_models.passed(self.home))
        routing.switch(self.routing, self.audit, POINT, first, 1, "reviewer", "flag beside the rule", home=self.home)
        label_projects(self.book, 40, 10)
        frozen = self.run_loop()
        self.assertEqual(frozen["retrained"], [])
        self.assertIn("serves at level 1", frozen["waiting"][POINT]["frozen"])
        self.assertEqual([f"{m.name}@{m.version}" for m in own_models.load(self.home)], [first])  # the routed file is the one loaded
        routing.switch(self.routing, self.audit, POINT, first, 0, "reviewer", "back to shadow to learn more", home=self.home)
        again = self.run_loop()
        self.assertEqual(again["retrained"], [POINT])
        self.assertNotIn(first, own_models.passed(self.home))  # the new version never sat the exam: the old pass is not its

    def test_a_fit_that_fails_costs_its_point_only_and_the_demotion_still_runs(self):
        from dclab_rnd import ledger, own_models
        from dclab_rnd.ledger import routing

        label_projects(self.book, 0, 40)
        name = "own@slipping"
        with mock.patch.object(routing, "passed", return_value={name: {"point": "feature.leakage_suspect"}}):
            routing.switch(self.routing, self.audit, "feature.leakage_suspect", name, 1, "reviewer", "flag", home=self.home)
        for i in range(routing.MIN_LABELS + 5):
            record = ledger.decide("feature.leakage_suspect", ("column", f"c{i}"), answers=[ledger.answer("rule", "rule:audit", False),
                                                                                          ledger.answer("classifier", name, True)],
                                   value_used=False, used="rule", state={"name_tokens": ["c"]}, options=["yes", "no"], project=f"r{i:03d}", ledger=self.book)
            ledger.label(record, False, "blind", "reviewer", weight=1.0, ledger=self.book)
        real = own_models.fit

        def fit(home, points=None):
            if points == ["feature.leakage_suspect"]:
                raise RuntimeError("a broken fit")
            return real(home, points)

        with mock.patch.object(own_models, "fit", side_effect=fit):
            got = self.run_loop()
        self.assertEqual([d["decision_point"] for d in got["demoted"]], ["feature.leakage_suspect"])  # the safety step ran first
        self.assertIn("RuntimeError", got["skipped"]["feature.leakage_suspect"])
        self.assertEqual(got["retrained"], [POINT])  # the other point was fitted, and loads
        self.assertEqual([m.point for m in own_models.load(self.home)], [POINT])

    def test_deleted_labels_leave_no_set_and_no_model_and_a_shrink_refits(self):
        from dclab_rnd import ledger, own_models

        label_projects(self.book, 0, 40)
        self.run_loop()
        self.assertTrue((own_models.root(self.home) / "sets" / POINT / "rows.jsonl").is_file())
        empty = ledger.FileLedger(self.home / "ledger_after_deletion", "wtest")  # every labelled decision deleted (19.10)
        got = __import__("dclab_rnd.students_loop", fromlist=["run"]).run(self.home, empty, self.routing, self.audit, measure=False)
        self.assertEqual(got["retired"], [POINT])
        self.assertFalse((own_models.root(self.home) / "sets" / POINT).exists())
        self.assertEqual(own_models.load(self.home), [])

    def test_another_ledger_on_the_same_folder_changes_nothing_until_a_person_adopts_it(self):
        from dclab_rnd import ledger, own_models, students_loop

        label_projects(self.book, 0, 40)
        first = self.run_loop()["cards"][POINT]["answerer"]
        other = ledger.FileLedger(self.home / "elsewhere", "another-workspace")  # the folder read by another backend: no label of these
        self.assertFalse(students_loop.due(self.home, other, now=10**10))  # the sweep never runs it
        with self.assertRaises(students_loop.OtherLedger):
            students_loop.run(self.home, other, self.routing, self.audit, measure=False)
        self.assertEqual([f"{m.name}@{m.version}" for m in own_models.load(self.home)], [first])  # nothing forgotten
        self.assertTrue((own_models.root(self.home) / "sets" / POINT / "rows.jsonl").is_file())
        adopted = students_loop.run(self.home, other, self.routing, self.audit, measure=False, adopt=True)
        self.assertEqual(adopted["retired"], [POINT])  # adopted: its labels are the source now, and it has none

    def test_a_route_to_a_model_that_is_gone_or_learned_from_deleted_decisions_goes_back_to_shadow(self):
        from dclab_rnd import ledger, own_models, students_loop
        from dclab_rnd.ledger import routing

        label_projects(self.book, 0, 40)
        first = self.run_loop()["cards"][POINT]["answerer"]
        with mock.patch.object(routing, "passed", return_value={first: {"point": POINT}}):
            routing.switch(self.routing, self.audit, POINT, first, 1, "reviewer", "flag", home=self.home)
        smaller = ledger.FileLedger(self.home / "after_deletion", "wtest")  # 39 of the 40 projects deleted
        label_projects(smaller, 0, 1)
        got = students_loop.run(self.home, smaller, self.routing, self.audit, measure=False)
        self.assertIn(POINT, [d["decision_point"] for d in got["demoted"]])  # said in the run's record (the review of 29.7)
        self.assertEqual(routing.entry(self.routing.get(), POINT)["level"], 0)
        self.assertIn(POINT, got["skipped"])  # one project cannot be fitted on: and the model learned from the deleted ones goes
        self.assertEqual((got["retired"], own_models.load(self.home)), ([POINT], []))
        with mock.patch.object(routing, "passed", return_value={"own@vanished": {"point": POINT}}):
            routing.switch(self.routing, self.audit, POINT, "own@vanished", 1, "reviewer", "flag", home=self.home)
        gone = students_loop.run(self.home, smaller, self.routing, self.audit, measure=False)
        self.assertIn({"decision_point": POINT, "answerer": "own@vanished"}, [{k: d[k] for k in ("decision_point", "answerer")} for d in gone["demoted"]])

    def test_the_history_shows_runs_that_changed_the_card_and_slipping_stays_shown(self):
        from dclab_rnd import students

        path = self.home / "students" / "register.jsonl"
        path.parent.mkdir(parents=True)
        lines = [("own@a", 0.95), ("own@b", 0.80), ("own@b", 0.80), ("own@b", 0.80)]  # a slip, then two nightly lines that changed nothing
        path.write_text("".join(json.dumps({"at": f"2026-10-18T0{i}:00:00+00:00", "points": {POINT: {"own": o, "own_dev_accuracy": d}}}) + "\n"
                                for i, (o, d) in enumerate(lines)))
        got = students.history(self.home)[POINT]
        self.assertEqual([h["own"] for h in got], ["own@a", "own@b"])
        self.assertTrue(got[-1]["slipped"])

    def test_due_after_new_labels_or_a_night_and_never_when_switched_off(self):
        from dclab_rnd import students_loop as S

        self.assertFalse(S.due(self.home, self.book, now=1000.0))  # no labels: nothing to learn
        label_projects(self.book, 0, 3)
        self.assertTrue(S.due(self.home, self.book, now=1000.0))
        S.run(self.home, self.book, self.routing, self.audit, measure=False, now=1000.0)
        self.assertFalse(S.due(self.home, self.book, now=1000.0 + S.NIGHTLY_SECONDS - 1))  # nothing new since
        self.assertTrue(S.due(self.home, self.book, now=1000.0 + S.NIGHTLY_SECONDS))  # a night: the register is written again
        label_projects(self.book, 3, 1)
        self.assertFalse(S.due(self.home, self.book, now=1000.0 + S.COOLDOWN_SECONDS - 1))  # new labels wait out the cooldown
        self.assertTrue(S.due(self.home, self.book, now=1000.0 + S.COOLDOWN_SECONDS))
        with mock.patch.dict("os.environ", {"DCLAB_STUDENTS_LOOP": "0"}):
            self.assertFalse(S.due(self.home, self.book, now=1000.0 + S.COOLDOWN_SECONDS))
        with mock.patch.object(S, "signature", side_effect=AssertionError("read inside the cooldown")):
            self.assertFalse(S.due(self.home, self.book, now=1000.0 + 1))  # inside the cooldown the ledger is not read at all
        with mock.patch.object(S, "signature", return_value=None):
            self.assertFalse(S.due(self.home, self.book, now=1000.0 + S.COOLDOWN_SECONDS))  # an unreadable ledger is no deletion

    def test_the_sweep_queues_one_job_and_a_failed_run_waits_before_the_next(self):
        from dclab_rnd import students_loop as S
        from dclab_rnd.jobs import ActiveJob

        label_projects(self.book, 0, 3)
        queued = []

        def queue_job(kind, key, payload, by="human", here=False):
            if queued:
                raise ActiveJob(None)
            queued.append((kind, key, by))
            return {"id": "j1", "kind": kind}

        services = SimpleNamespace(store=SimpleNamespace(home=self.home), ledger=self.book, queue_job=queue_job)
        self.assertEqual(S.sweep_due(services, now=1000.0)["kind"], "students")
        self.assertIsNone(S.sweep_due(services, now=1001.0))  # the job is queued: never twice
        self.assertEqual(queued, [("students", "loop", "agent")])
        S.failed(self.home, now=1000.0)
        self.assertFalse(S.due(self.home, self.book, now=1000.0 + S.RETRY_SECONDS - 1))
        self.assertTrue(S.due(self.home, self.book, now=1000.0 + S.RETRY_SECONDS))

    def test_the_job_runs_through_a_worker_and_a_failed_run_waits(self):
        import time

        from dclab_rnd import students_loop
        from dclab_rnd.jobs import FileJobs, Worker
        from dclab_rnd.jobs import handlers  # noqa: F401 - registers the kinds

        label_projects(self.book, 0, 40)
        env = SimpleNamespace(store=SimpleNamespace(home=self.home), ledger=self.book, gateway=SimpleNamespace(routing=self.routing), audit=self.audit)
        jobs = FileJobs(self.home / "jobs.json")
        worker = Worker(jobs, env, threads=1, poll=0.05, beat=0.05, stale=5).start()
        self.addCleanup(worker.stop, 2.0)

        def ended(job_id):
            for _ in range(600):
                j = jobs.get(job_id)
                if j["status"] not in ("queued", "running"):
                    return j
                time.sleep(0.05)
            self.fail("the job did not end")

        job = jobs.enqueue("students", "loop", {}, by="agent")
        worker.notify()
        self.assertEqual(ended(job["id"])["status"], "done")
        self.assertEqual(students_loop.state(self.home)["fitted"][POINT] > 0, True)
        with mock.patch.object(students_loop, "run", side_effect=RuntimeError("broken")):
            bad = jobs.enqueue("students", "loop", {}, by="agent")
            worker.notify()
            self.assertEqual(ended(bad["id"])["status"], "failed")
        self.assertIn("failed_clock", students_loop.state(self.home))  # settled: the next run waits an hour

    def test_the_job_kind_is_registered_and_its_run_is_recorded(self):
        from dclab_rnd import experience, students_loop
        from dclab_rnd.agentic.services import BUSY
        from dclab_rnd.jobs import handlers, worker  # noqa: F401 - registers the kinds

        self.assertIn("students", worker.HANDLERS)
        self.assertIn("students", BUSY)
        label_projects(self.book, 0, 40)
        seen = []
        with mock.patch.object(experience, "record", side_effect=lambda kind, actor, payload, **ids: seen.append((kind, payload))):
            students_loop.run(self.home, self.book, self.routing, self.audit, measure=False)
        runs = [p for k, p in seen if k == "run.recorded"]
        self.assertEqual((runs[0]["run_kind"], runs[0]["name"], runs[0]["status"]), ("job", "students_loop", "completed"))


if __name__ == "__main__":
    unittest.main()

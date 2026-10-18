"""Package 29.8: the people's labels, made easy. Every student whispers (DCLab's four folders and the workspace's own);
a queue of the disagreements worth a person's look (a student against the rule, the student's surest first) is shown
blind and drawn as the blind draw draws, so each label keeps its weight; one click labels one; the register counts them
toward the point's "needed to act"; and a tester's labels, collected, count too."""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
    from fastapi.testclient import TestClient

    from dclab_core import ledger as contract
    from dclab_rnd import ledger
except ImportError as error:  # the agent environment
    MISSING = str(error)
else:
    MISSING = ""

POINT = "feature.leakage_suspect"
NAME = "student@v1"


class Student:
    """A student that says "a leak" with a confidence that rises with the column's number (its name: c7, t3)."""

    by, name, version = "classifier", "student", "v1"
    points = frozenset({POINT})

    def answer(self, point, state, options):
        return contract.Answer(by="classifier", answerer=NAME, value=True, confidence=0.5 + int(str(state.get("column") or "c0")[1:]) / 1000)

    def answer_many(self, point, states, options):  # as a workspace's own model answers many decisions at once
        return [self.answer(point, s, o) for s, o in zip(states, options)]


def plant(book, n=120, agree=20, project="p1"):
    """``n`` decisions where the student (in shadow) says a leak and the rule says not, ``agree`` where they agree."""
    out = []
    for i in range(n + agree):
        disagree = i < n
        out.append(ledger.decide(POINT, ("column", f"c{i}"), answers=[ledger.answer("rule", "rule:leak", not disagree),
                                                                       ledger.answer("classifier", NAME, True, source="shadow", confidence=round(0.5 + i / 1000, 4))],
                                 value_used=not disagree, state={"column": f"c{i}", "name_tokens": ["c"], "dtype": "integer"}, options=["yes", "no"],
                                 ledger=book, project=project))
    return out


@unittest.skipIf(MISSING, MISSING)
class WhisperTests(unittest.TestCase):
    def test_every_students_folder_whispers(self):
        from dclab_rnd.ledger import shadow

        self.assertEqual({p.name for p in shadow.FOLDERS}, {"models_v1", "models_v2", "models_images_v1", "workflow_models_v1"})
        seen = []
        shadow.answerers.cache_clear()
        self.addCleanup(shadow.answerers.cache_clear)
        with mock.patch.object(shadow, "_load", side_effect=lambda folder: seen.append(folder.name) or []):
            shadow.answerers()
        self.assertEqual(sorted(seen), sorted(p.name for p in shadow.FOLDERS))

    def test_the_new_students_can_answer_the_states_production_records(self):
        from dclab_rnd.ledger import shadow, states

        profile = {"rows": 5000, "columns": [{"name": "order_id", "id_like": True, "name_id_like": True, "unique_ratio": 0.4},
                                             {"name": "created_at", "time_like": True, "dtype_band": "datetime"}]}
        made = {"split.strategy": states.split_state(profile, "created_at", None), "data.cases_table": states.cases_state(profile, "cases.csv"),
                "image.near_copy": {"similarity": 0.95, "near_threshold": 0.98, "margin": -0.03, "same_label": True, "images": 336},
                "image.label_suspect": {"copies": 3, "copies_agree": 1.0, "neighbours_agree": 1.0, "images": 336},
                "image.flip_safe": {"flip_drop": 0.95, "flip_accuracy": 0.02, "plain_accuracy": 0.97, "image_side": 32, "images": 336, "classes": 2}}
        book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        for point, state in made.items():  # recorded as capture records them: a project's (a draft's) decision, never a column's
            subject = ("draft", "d1") if point == "data.cases_table" else ("project", "p1")
            r = ledger.decide(point, subject, answers=[ledger.answer("rule", "rule:x", True)], value_used=True, state=state, options=["yes", "no"],
                              ledger=book, **({"draft": "d1"} if subject[0] == "draft" else {"project": "p1"}))
            self.assertIsNotNone(r, point)
            self.assertTrue(shadow.answerable(r), point)  # the review of 29.8: they had no column band, so no student ever whispered

    def test_the_default_measure_reads_dclabs_students_never_the_workspaces_own(self):
        from dclab_rnd.ledger import shadow

        book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        plant(book, n=3, agree=0)
        with mock.patch.object(shadow, "answerers", return_value=()), mock.patch.object(shadow, "_own", return_value=[Student()]):
            self.assertEqual(shadow.measure(book.records(), book.labels()), {})  # an own student learns from these labels: in-sample


@unittest.skipIf(MISSING, MISSING)
class QueueTests(unittest.TestCase):
    def setUp(self):
        self.book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        self.records = plant(self.book)

    def test_the_queue_is_the_drawn_disagreements_shown_blind_in_an_order_that_does_not_favour_the_student(self):
        from dclab_rnd.ledger import labels

        current = {POINT: NAME}
        got = labels.queue(self.book, point=POINT, limit=500, current=current)
        drawn = [r for r in self.records if r.answers[0].value is False and contract.blind_draw(r, "blind-v1")]
        self.assertEqual({i["record"] for i in got["items"]}, {r.id for r in drawn})  # the draw's own: each keeps its weight
        by_id = {r.id: r for r in self.records}
        sure = [by_id[i["record"]].answers[1].confidence for i in got["items"]]
        first = sure[: len(sure) // 4]
        self.assertLess(abs(sum(first) / len(first) - sum(sure) / len(sure)), 0.06)  # the first quarter is not the student's surest
        self.assertTrue(all("answers" not in i and "value_used" not in i and "confidence" not in i for i in got["items"]))  # blind
        self.assertTrue(all(i["weight"] >= 1 for i in got["items"]))
        self.assertEqual(got["points"][POINT], len(drawn))
        self.assertEqual([i["record"] for i in labels.queue(self.book, point=POINT, limit=500, current=current)["items"]],
                         [i["record"] for i in got["items"]])  # the order is the seed's: the same every time
        labels.blind_label(got["items"][0]["record"], False, "reviewer:u1", ledger=self.book)
        self.assertNotIn(got["items"][0]["record"], {i["record"] for i in labels.queue(self.book, point=POINT, limit=500, current=current)["items"]})
        self.assertEqual(len(labels.queue(self.book, point=POINT, limit=10, current=current)["items"]), 10)

    def test_only_the_points_current_student_fills_the_queue(self):
        from dclab_rnd.ledger import labels

        self.assertEqual(labels.queue(self.book, point=POINT, current={POINT: "student@v0"})["items"], [])  # an older version
        self.assertEqual(labels.queue(self.book, point=POINT, current={})["items"], [])  # no student for the point (an own student)

    def test_thirty_labels_through_the_route_count_toward_acting(self):
        from dclab_rnd import students
        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.ledger import routing, shadow

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        book = client.app.state.services.ledger
        plant(book, n=160, agree=0)
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        from dclab_rnd import students as register

        card = {POINT: {"answerer": NAME, "family": "boosting", "trained_on": 100, "dev_accuracy": 0.9, "rule_accuracy": 0.8}}
        with mock.patch.object(register, "students", return_value=card):
            items = client.get("/api/ledger/queue", params={"point": POINT, "limit": 30}).json()["items"]
        self.assertEqual(len(items), 30)
        started = time.perf_counter()
        for item in items:  # one click each
            self.assertEqual(client.post(f"/api/ledger/blind/{item['record']}", json={"value": False}, headers=h).status_code, 200)
        self.assertLess(time.perf_counter() - started, 30)  # thirty in a few minutes is the person's time, not the server's
        card = {POINT: {"answerer": NAME, "family": "boosting", "trained_on": 100, "dev_accuracy": 0.9, "rule_accuracy": 0.8}}
        with mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "1"}), mock.patch.object(students, "students", return_value=card), \
                mock.patch.object(shadow, "answerers", return_value=(Student(),)), mock.patch.object(shadow, "_own", return_value=[]):
            row = next(r for r in students.read(book, {})["items"] if r["point"] == POINT)
        self.assertEqual((row["labelled_disagreements"], row["needed_to_act"]), (30, max(0, routing.MIN_LABELS - 30)))

    def test_the_queue_is_a_reviewers_route(self):
        from dclab_rnd.accounts import guard
        from dclab_rnd.accounts.principal import Principal
        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.assertEqual(client.get("/api/ledger/queue", params={"point": "a b"}).status_code, 422)
        with mock.patch.object(guard, "current", return_value=Principal(role="viewer", user_id="u1", via="session")):
            self.assertEqual(client.get("/api/ledger/queue").status_code, 403)
        with mock.patch.object(guard, "current", return_value=Principal(role="reviewer", user_id="u1", via="session")):
            self.assertEqual(client.get("/api/ledger/queue").status_code, 200)


@unittest.skipIf(MISSING, MISSING)
class TesterTests(unittest.TestCase):
    def test_a_testers_collected_labels_count_toward_acting(self):
        from dclab_rnd import students
        from dclab_rnd.ledger import shadow

        home = Path(tempfile.mkdtemp())
        mine = ledger.FileLedger(home / "ledger", "w")
        from dclab_rnd import consent

        friend = home / "testers" / "friend1"
        friend.mkdir(parents=True)
        for use in ("improve", "train"):  # a tester shares only under both switches (25.11); their records are stamped so
            consent.set_workspace_consent(friend, use, True)
        theirs = ledger.FileLedger(friend / "ledger", "friend1")  # as make collect rebuilds it
        for i in range(25):  # the tester's decisions carry no shadow: their machine had no student
            r = ledger.decide(POINT, ("column", f"t{i}"), answers=[ledger.answer("rule", "rule:leak", False)], value_used=False,
                              state={"column": f"t{i}", "name_tokens": ["t"], "dtype": "integer"}, options=["yes", "no"], ledger=theirs)  # stamped train
            ledger.label(r, i % 5 != 0, "blind", "reviewer:friend", weight=1.0, ledger=theirs)
        card = {POINT: {"answerer": NAME, "family": "boosting", "trained_on": 100, "dev_accuracy": 0.9, "rule_accuracy": 0.8}}
        with mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "1"}), mock.patch.object(students, "students", return_value=card), \
                mock.patch.object(shadow, "answerers", return_value=(Student(),)), mock.patch.object(shadow, "_own", return_value=[]):
            alone = next(r for r in students.read(mine, {})["items"] if r["point"] == POINT)
            joined = next(r for r in students.read(mine, {}, home=home)["items"] if r["point"] == POINT)
        self.assertEqual(alone["labelled_disagreements"], 0)
        self.assertEqual(joined["labelled_disagreements"], 25)  # the student answers the tester's decisions as it would now
        self.assertEqual(joined["needed_to_act"], 0)
        import shutil

        shutil.copytree(friend, home / "testers" / "friend1-again")  # the same tester shared twice (a second slug): counted once
        with mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "1"}), mock.patch.object(students, "students", return_value=card), \
                mock.patch.object(shadow, "answerers", return_value=(Student(),)), mock.patch.object(shadow, "_own", return_value=[]):
            twice = next(r for r in students.read(mine, {}, home=home)["items"] if r["point"] == POINT)
        self.assertEqual(twice["labelled_disagreements"], 25)

    def test_a_tester_record_stamped_operate_never_counts(self):
        from dclab_rnd import students
        from dclab_rnd.ledger import shadow

        home = Path(tempfile.mkdtemp())
        theirs = ledger.FileLedger(home / "testers" / "friend2" / "ledger", "friend2")  # no consent: every record stamped operate
        for i in range(25):
            r = ledger.decide(POINT, ("column", f"o{i}"), answers=[ledger.answer("rule", "rule:leak", False)], value_used=False,
                              state={"column": f"o{i}", "name_tokens": ["o"], "dtype": "integer"}, options=["yes", "no"], ledger=theirs, project="op")
            ledger.label(r, True, "blind", "reviewer:friend", weight=1.0, ledger=theirs)
        card = {POINT: {"answerer": NAME, "family": "boosting", "trained_on": 100, "dev_accuracy": 0.9, "rule_accuracy": 0.8}}
        with mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "1"}), mock.patch.object(students, "students", return_value=card), \
                mock.patch.object(shadow, "answerers", return_value=(Student(),)), mock.patch.object(shadow, "_own", return_value=[]):
            row = next(r for r in students.read(ledger.FileLedger(home / "ledger", "w"), {}, home=home)["items"] if r["point"] == POINT)
        self.assertEqual(row["labelled_disagreements"], 0)


if __name__ == "__main__":
    unittest.main()

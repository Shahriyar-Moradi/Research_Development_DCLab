"""Package 29.10: every user is a teacher. Where a point's student whispered something other than the rule on a project's
decision, the project's page and Home show a note ("the rule said X, a student thinks Y; which is right?"); one click
writes a label the person gave having seen both (user_decision_exposed, with the answers they saw), never a change of
what was decided. Such a label may demote a student, never promote it: it counts toward demotion and in the register's
people's labels, never toward "needed to act"."""

import sys
import tempfile
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

POINT = "column.is_identifier"
NAME = "student@v1"
T1, T2 = "2026-10-19T01:00:00+00:00", "2026-10-19T02:00:00+00:00"


class Student:
    """A student that reads every column as an identifier: it disagrees with the rule on every column but the key."""

    by, name, version = "classifier", "student", "v1"
    points = frozenset({POINT})

    def answer(self, point, state, options):
        return contract.Answer(by="classifier", answerer=NAME, value=True, confidence=0.8)

    def answer_many(self, point, states, options):
        return [self.answer(point, s, o) for s, o in zip(states, options)]


def record(rule=True, student=False):
    return contract.DecisionRecord(id="dr-note-000001", point=POINT, subject=contract.Subject(kind="column", key="customer_id"), state={},
                                   options=["yes", "no"], answers=[contract.Answer(by="rule", answerer="rule:id", value=rule),
                                                                   contract.Answer(by="classifier", answerer=NAME, value=student, source="shadow")],
                                   value_used=rule, used="rule", at=T1)


def exposed(value, shown=(), n=1, at=T2):
    return contract.DecisionLabel(id=f"dl-note-{n:06d}", record="dr-note-000001", point=POINT, value=value, source="user_decision_exposed",
                                  by="person:u1", at=at, shown=list(shown))


@unittest.skipIf(MISSING, MISSING)
class LabelRuleTests(unittest.TestCase):
    def test_a_person_who_saw_both_answers_is_heard_either_way(self):
        r = record()
        with_rule = exposed(True, shown=["rule:id", NAME])
        with_student = exposed(False, shown=["rule:id", NAME], n=2)
        self.assertEqual(contract.effective_label(r, [with_rule]).id, with_rule.id)  # against the student's answer they saw
        self.assertEqual(contract.effective_label(r, [with_student]).id, with_student.id)  # against the rule's
        self.assertIsNone(contract.effective_label(r, [exposed(True)]))  # 19.8 as before: saw only the rule, agreed: no evidence

    def test_an_older_label_is_read_as_the_new_version(self):
        doc = exposed(True).model_dump(mode="json")
        doc.pop("shown")
        doc["schema_version"] = "dclab_core.decision_label.v4"
        self.assertEqual(contract.DecisionLabel.model_validate(doc).shown, [])

    def test_an_exposed_label_never_counts_toward_acting_and_is_counted_apart(self):
        from dclab_rnd.ledger import shadow

        records = [record().model_copy(update={"id": f"dr-note-{i:06d}"}) for i in range(1, 31)]
        labels = [exposed(i % 2 == 0, shown=["rule:id", NAME], n=i).model_copy(update={"record": f"dr-note-{i:06d}"}) for i in range(1, 31)]
        row = shadow.measure(records, labels, [Student()])[NAME]
        self.assertEqual(row["labelled_disagreements"], 0)  # never toward needed to act: a person who saw both may demote, never promote
        self.assertEqual((row["exposed_disagreements"], row["exposed_student_wrong"]), (30, 15))


@unittest.skipIf(MISSING, MISSING)
class DemotionAndQueueTests(unittest.TestCase):
    """The review of 29.10: demotion reads every exposed label (a model right 95% of the time stays), and a note never
    shows a decision of the blind draw or one with a blind label (those are the blind queue's, which may count toward acting)."""

    def test_a_routed_model_mostly_right_by_people_who_saw_it_stays(self):
        from dclab_rnd import audit
        from dclab_rnd.ledger import routing
        from dclab_rnd.models.routing import FileRouting

        tmp = Path(tempfile.mkdtemp())
        store, log = FileRouting(tmp / "model_routing.json"), audit.FileAudit(tmp / "audit.jsonl")
        with mock.patch.object(routing, "passed", return_value={NAME: {"point": POINT}}):
            routing.switch(store, log, POINT, NAME, 1, "reviewer", "flag beside the rule")
        since = routing.entry(store.get(), POINT)["since"]
        records, labels = [], []
        for i in range(40):
            r = record().model_copy(update={"id": f"dr-dem-{i:06d}", "at": since,
                                            "answers": [contract.Answer(by="rule", answerer="rule:id", value=True), contract.Answer(by="classifier", answerer=NAME, value=False)]})
            records.append(r)
            labels.append(exposed(i % 20 == 0, shown=["rule:id", NAME], n=i + 100).model_copy(update={"record": r.id, "at": since}))  # 38 of 40 agree with it
        self.assertEqual(routing.demote(store, log, records, labels), [])  # 0.95 right: it stays

    def test_a_note_leaves_the_blind_queues_decisions_alone(self):
        from dclab_rnd.ledger import notes

        book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        made = []
        for i in range(60):
            made.append(ledger.decide(POINT, ("column", f"c{i}"), answers=[ledger.answer("rule", "rule:id", True), ledger.answer("classifier", NAME, False, source="shadow")],
                                      value_used=True, state={"column": f"c{i}", "name_tokens": ["c"], "dtype": "string", "uniqueness": "all"}, options=["yes", "no"],
                                      ledger=book, project="p1"))
        drawn = {r.id for r in made if contract.blind_draw(r, "blind-v1")}
        shown = {n["record"] for n in notes.pending(book, "p1", "person:u1", {POINT: NAME}, limit=100)["items"]}
        self.assertTrue(drawn and shown)
        self.assertFalse(drawn & shown)  # a drawn decision is labelled blind only
        one = next(iter(drawn))
        with self.assertRaises(ValueError):
            notes.answer(book, "p1", one, "yes", "person:u1", {POINT: NAME})
        free = next(iter(shown))
        ledger.label(next(r for r in made if r.id == free), True, "blind", "reviewer", weight=1.0, ledger=book)
        self.assertNotIn(free, {n["record"] for n in notes.pending(book, "p1", "person:u1", {POINT: NAME}, limit=100)["items"]})
        self.assertEqual(notes.counts(book, "person:u1", {POINT: NAME}), {"p1": len(shown) - 1})  # Home's count, from one read

    def test_an_own_students_exam_never_reads_an_exposed_label(self):
        import json

        from dclab_rnd import own_models

        home = Path(tempfile.mkdtemp())
        folder = own_models.root(home) / "sets" / POINT
        folder.mkdir(parents=True)
        kinds = [{"source": "blind"}, {"source": "user_decision_exposed", "shown": ["rule:id", NAME]}, {"source": "user_decision_exposed"}, {"source": "blind"}]
        rows = [{"record": f"r{i}", "point": POINT, "state": {"dtype": "string"}, "options": ["yes", "no"], "split": "sealed", "group": f"project:p{i}",
                 "label": {"value": True, **kinds[i]}, "labels": {"blind": {"value": True, "weight": 1.67}} if kinds[i]["source"] == "blind" else {}}
                for i in range(4)]  # as datasets.decisions writes them: a drawn blind label with its weight
        (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

        class Model:
            point, version, name, base_version = POINT, "v1", "own", None

            def answer(self, point, state, options):
                return contract.Answer(by="classifier", answerer="own@v1", value=True)

        (own_models.root(home) / "models").mkdir(parents=True)
        (own_models.root(home) / "models" / f"{POINT}.json").write_text(json.dumps({"name": "own", "version": "v1", "family": "logistic", "strength": 1.0, "trained_set": "x"}))
        with mock.patch.object(own_models, "load", return_value=[Model()]):
            doc = own_models.gate(home, [POINT])["judged"][POINT]
        self.assertEqual(doc["sealed"]["decisions"], 3)  # a note's answer (DCLab's student shown) never grades it; a correction of the rule does


@unittest.skipIf(MISSING, MISSING)
class NotesTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import students
        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.ledger import shadow
        from dclab_rnd import kinds

        self.home = Path(tempfile.mkdtemp())
        card = {POINT: {"answerer": NAME, "family": "boosting", "trained_on": 100, "dev_accuracy": 0.9, "rule_accuracy": 0.8}}
        from dclab_core import ledger as contract

        # the blind draw picks decisions by their random ids: one that drew the fixture's only disagreement left no note, a
        # failure by chance (2026-10-19, a gate); here nothing is drawn (the draw's own rule is test_a_note_leaves_the_blind_queues_decisions_alone)
        for patcher in (mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "1", "DCLAB_AGENT_HOME": str(self.home)}),
                        mock.patch.object(shadow, "answerers", return_value=(Student(),)), mock.patch.object(shadow, "_own", return_value=[]),
                        mock.patch.object(students, "students", return_value=card), mock.patch.object(contract, "blind_draw", return_value=False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(self.home))
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        _, self.pid = acceptance.build(kinds.get("binary").case(0), None, store=self.client.app.state.services.projects)
        self.book = self.client.app.state.services.ledger

    def test_a_disagreement_shows_a_note_and_a_click_is_a_label_that_changes_nothing_decided(self):
        from dclab_rnd import students

        notes = self.client.get(f"/api/projects/{self.pid}/disagreements", headers=self.h).json()
        mine = [n for n in notes["items"] if n["point"] == POINT]
        self.assertTrue(mine)  # the rule called a column no identifier; the student thinks it is one
        note = mine[0]
        self.assertEqual((note["rule"]["value"], note["student"]["value"], note["student"]["answerer"]), (False, True, NAME))
        self.assertEqual(note["options"], ["yes", "no"])
        before = self.client.app.state.services.projects.get(self.pid)
        used = next(r.value_used for r in self.book.records(project=self.pid) if r.id == note["record"])
        done = self.client.post(f"/api/projects/{self.pid}/disagreements/{note['record']}", json={"value": "yes"}, headers=self.h)
        self.assertEqual(done.status_code, 200, done.text)
        label = done.json()["label"]
        self.assertEqual((label["source"], label["value"]), ("user_decision_exposed", True))
        self.assertEqual(sorted(label["shown"]), sorted([note["rule"]["answerer"], NAME]))  # the two answers the person saw
        after = self.client.app.state.services.projects.get(self.pid)
        self.assertEqual((after["solution"], after["stages"]), (before["solution"], before["stages"]))  # nothing decided changes
        self.assertEqual(next(r.value_used for r in self.book.records(project=self.pid) if r.id == note["record"]), used)
        left = self.client.get(f"/api/projects/{self.pid}/disagreements", headers=self.h).json()["items"]
        self.assertNotIn(note["record"], [n["record"] for n in left])  # answered: this person is not asked again
        row = next(r for r in students.read(self.book, {}, home=self.home)["items"] if r["point"] == POINT)
        self.assertEqual(row["labels"].get("user_decision_exposed"), 1)  # the register counts it
        self.assertEqual(row["people_disagreements"], 1)
        self.assertEqual(row["needed_to_act"], 20)  # and never toward acting

    def test_a_reviewer_may_answer_a_note(self):
        from dclab_rnd.accounts import guard
        from dclab_rnd.accounts.principal import Principal

        note = self.client.get(f"/api/projects/{self.pid}/disagreements", headers=self.h).json()["items"][0]
        with mock.patch.object(guard, "current", return_value=Principal(role="reviewer", user_id="u7", via="session")):
            got = self.client.post(f"/api/projects/{self.pid}/disagreements/{note['record']}", json={"value": "no"}, headers=self.h)
        self.assertEqual(got.status_code, 200, got.text)  # every user is a teacher: a label, never a change

    def test_home_says_where_students_disagree_and_a_wrong_answer_is_refused(self):
        needs = self.client.get("/api/workspace", headers=self.h).json()["needs"]
        self.assertTrue(any(n.get("kind") == "disagreements" and n.get("project") == self.pid for n in needs))
        note = self.client.get(f"/api/projects/{self.pid}/disagreements", headers=self.h).json()["items"][0]
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/disagreements/{note['record']}", json={"value": "maybe"}, headers=self.h).status_code, 422)
        self.assertEqual(self.client.post(f"/api/projects/{self.pid}/disagreements/dr-missing-0001", json={"value": "yes"}, headers=self.h).status_code, 404)


if __name__ == "__main__":
    unittest.main()

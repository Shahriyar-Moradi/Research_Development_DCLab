"""Package 19.10: consent, redaction, retention and deletion. With consent off nothing is exported; on for improvement,
redacted text and no cells; on for training, the same plus decision states; a planted e-mail and a planted cell value
never pass. A deleted project or session leaves nothing in the record or the ledger; retention removes by class. The
log tests run on PostgreSQL under ``make test-pg``."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import experience as x
    from dclab_core.redact import redact_text
    from dclab_rnd import consent, experience, ledger
    from dclab_rnd.experience import retention
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""

EMAIL = "jane.roe@example.com"
CELL = "4111 1111 1111 1111"  # a value from a person's table


def stores(home: Path):
    """The product's own openers (23.1/23.3: a log opened bare, with no folder, stamps every record operate)."""
    if os.environ.get("DCLAB_DATABASE_URL"):
        from dclab_rnd.storage import db

        db.upgrade()
        wid = db.workspace_for(home)
        return experience.open_log(home, wid), ledger.open_ledger(home, wid)
    return experience.open_log(home, "wtest"), ledger.open_ledger(home, "wtest")


def events(log, project="p1", visit="v1", at=None):
    rows = [("chat.person", "person", {"text": f"Write to {EMAIL} about the late orders", "surface": "home"}),
            ("tool.call", "agent", {"tool": "preview_rows", "arguments": {"card": CELL}, "result": {"rows": [CELL]}}),
            ("model.request", "model", {"purpose": "draft_chat", "cost": 0.001, "currency": "EUR", "prompt": f"the table: {CELL}", "reply": "ok"}),
            ("stage.finished", "system", {"stage": "data", "metric": "roc_auc", "value": 0.8})]
    out = []
    for i, (kind, actor, payload) in enumerate(rows):
        when = at or datetime.now(timezone.utc)
        out.append(experience.record(kind, actor, payload, log=log, at=(when + timedelta(seconds=i)).isoformat(), visit=visit, project=project))
    return out


@unittest.skipIf(MISSING, MISSING)
class RedactionTest(unittest.TestCase):
    def test_what_identifies_someone_is_replaced(self):
        text, found = redact_text(f"Call Jane Roe on +971 50 123 4567 or {EMAIL}; refund €1,250.00 to IBAN AE07 0331 2345 6789 0123 456 at 12 Main Street.")
        for leaked in ("Jane Roe", "4567", EMAIL, "1,250", "AE07", "Main Street"):
            self.assertNotIn(leaked, text)
        self.assertEqual(set(found) >= {"NAME", "PHONE", "EMAIL", "AMOUNT", "IBAN", "ADDRESS"}, True, found)
        plain = "Predict which orders arrive late, at the moment the order is placed, on the Home page of DCLab."
        self.assertEqual(redact_text(plain), (plain, {}))

    def test_names_and_ids_anywhere_in_a_sentence(self):
        for text, gone in (("Jane Roe asked for the late orders.", "Jane"), ("Late orders. Ahmed Khan wants them.", "Ahmed"),
                           ("Call on Monday Jane Roe", "Jane"), ("Jane Roe Friday", "Jane"), ("ask the Owner Jane Roe", "Roe"), ("Mohammad Reza Hosseini wrote", "Mohammad"),
                           ("order ORD12345", "ORD12345"), ("user u3fa9c1d2e4b5a6", "u3fa9c1d2e4b5a6")):
            with self.subTest(text=text):
                self.assertNotIn(gone, redact_text(text)[0])
        self.assertEqual(redact_text("ask the Owner Jane Roe")[0], "ask the Owner [NAME]")  # the product's words stay

    def test_what_is_not_personal_stays(self):
        for text in ("Orders from 2026-01-01 to 2026-06-30", "5-fold, WF-01 and COVID-19", "rows 1000 2000 3000 4000",
                     "Use gpt-4o and qwen2.5 with 5 folds, ROC AUC 0.81"):
            with self.subTest(text=text):
                self.assertEqual(redact_text(text), (text, {}))


@unittest.skipIf(MISSING, MISSING)
class ExportTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        consent.set_workspace_consent(self.home, "improve", True)  # 23.1: what is written now carries the level now
        consent.set_workspace_consent(self.home, "train", True)
        self.log, self.book = stores(self.home)
        from dclab_rnd.storage import open_stores

        store = open_stores(self.home)[0]  # 23.1: a record of a project the store does not have is operate; the projects are real
        made = {}
        for name, settings in (("p1", {"consent": {"improve": True, "train": True}}), ("p2", {})):
            doc = store.create(f"Project {name}")
            doc["settings"] = {**(doc.get("settings") or {}), **settings}
            made[name] = store.save(doc)["id"]
        self.p1, self.p2 = made["p1"], made["p2"]
        events(self.log, project=self.p1)
        events(self.log, project=self.p2, visit="v2")
        self.rec = ledger.decide("column.is_identifier", ("column", "order_id"), answers=[ledger.answer("rule", "r", True)], value_used=True,
                                 state={"column": "order_id", "dtype": "integer"}, ledger=self.book, project=self.p1)
        self.projects = {self.p1: {"id": self.p1, "settings": {"consent": {"improve": True, "train": True}}}, self.p2: {"id": self.p2, "settings": {}}}

    def test_nothing_leaves_without_consent(self):
        off = consent.workspace_consent(Path(tempfile.mkdtemp()))
        self.assertEqual(off, {"improve": False, "train": False})  # off by default
        for use in consent.USES:
            with self.assertRaises(consent.ConsentRefused):
                consent.export_events(self.log.read(), use, off, self.projects)
        with self.assertRaises(consent.ConsentRefused):
            consent.export_decisions(self.book.records(), self.book.labels(), off, self.projects)

    def test_for_improvement_redacted_text_and_no_cells(self):
        on = {"improve": True, "train": False}  # the switches as the export is asked, apart from the stamps written at setUp
        out = consent.export_events(self.log.read(), "improve", on, self.projects)
        text = json.dumps(out)
        self.assertNotIn(EMAIL, text)
        self.assertNotIn(CELL, text)  # a cell value never leaves: not in a tool's arguments, nor in a prompt
        self.assertEqual({d["ids"]["project"] for d in out}, {self.p1})  # p2 did not consent
        chat = next(d for d in out if d["kind"] == "chat.person")
        self.assertIn("[EMAIL]", chat["payload"]["text"])
        self.assertEqual(chat["redacted_text"], {"EMAIL": 1})
        tool = next(d for d in out if d["kind"] == "tool.call")
        self.assertNotIn("arguments", tool["payload"])
        with self.assertRaises(consent.ConsentRefused):  # improvement is not training
            consent.export_decisions(self.book.records(), self.book.labels(), on, self.projects)

    def test_for_training_the_decision_states_too(self):
        consent.set_workspace_consent(self.home, "improve", True)
        on = consent.set_workspace_consent(self.home, "train", True)
        decisions = consent.export_decisions(self.book.records(), self.book.labels(), on, self.projects)
        self.assertEqual([d["state"]["column"] for d in decisions], ["order_id"])
        self.assertNotIn(CELL, json.dumps(consent.export_events(self.log.read(), "train", on, self.projects)))
        one_project_off = {**self.projects, self.p1: {"id": self.p1, "settings": {"consent": {"improve": True}}}}
        self.assertEqual(consent.export_decisions(self.book.records(), self.book.labels(), on, one_project_off), [])

    def test_the_old_training_opt_in_still_counts(self):
        self.assertEqual(consent.project_consent({"settings": {"share_for_training": True}}), {"improve": False, "train": True})
        off = {"settings": {"share_for_training": True, "consent": {"train": False}}}
        self.assertEqual(consent.project_consent(off), {"improve": False, "train": False})  # its own switch, once set, wins

    def test_no_person_leaves_in_a_meta_field_a_list_an_id_or_a_label(self):
        on = consent.set_workspace_consent(self.home, "improve", True)
        on = consent.set_workspace_consent(self.home, "train", True)
        experience.record("data.uploaded", "person", {"name": f"{EMAIL} - Jane Roe customers.csv", "bytes": 10}, log=self.log, visit="v9",
                          project=self.p1, user="u_person1", request="r_req12345")
        experience.record("chat.agent", "agent", {"text": "Which?", "question_field": "target", "options": [f"write to {EMAIL}", "late"]},
                          log=self.log, visit="v9", project=self.p1)
        experience.record("decision.accepted", "person", {"point": "policy:x", "choice": {"to": EMAIL}, "by": "person"}, log=self.log, visit="v9",
                          project=self.p1)
        out = consent.export_events(self.log.read(), "improve", on, self.projects)
        text = json.dumps(out)
        self.assertNotIn(EMAIL, text)
        self.assertNotIn("Jane Roe", text)
        self.assertNotIn("u_person1", text)
        self.assertNotIn("r_req12345", text)
        self.assertEqual(next(d for d in out if d["kind"] == "data.uploaded")["payload"]["name"], "file.csv")
        ledger.label(self.rec, False, "user_decision_exposed", "person:u_person1", ledger=self.book)
        decisions = json.dumps(consent.export_decisions(self.book.records(), self.book.labels(), on, self.projects))
        self.assertNotIn("u_person1", decisions)
        self.assertIn('"by": "person"', decisions)

    def test_a_drafts_conversation_follows_the_project_it_became(self):
        on = consent.set_workspace_consent(self.home, "improve", True)
        on = consent.set_workspace_consent(self.home, "train", True)
        experience.record("draft.created", "person", {"problem": "Predict late orders"}, log=self.log, visit="v8", conversation="d_built")
        ledger.decide("model.family", ("project", "d_built"), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge",
                      ledger=self.book, draft="d_built")
        drafts = {"d_built": self.p2}  # p2 did not consent
        self.assertNotIn("d_built", json.dumps(consent.export_events(self.log.read(), "improve", on, self.projects, drafts)))
        self.assertNotIn("d_built", json.dumps(consent.export_decisions(self.book.records(), self.book.labels(), on, self.projects, drafts)))
        self.assertIn("d_built", json.dumps(consent.export_events(self.log.read(), "improve", on, self.projects)))  # unbuilt: the workspace's


@unittest.skipIf(MISSING, MISSING)
class DeletionTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.log, self.book = stores(self.home)

    def test_a_deleted_project_leaves_nothing(self):
        events(self.log, project="p1", visit="v1")
        events(self.log, project="p2", visit="v2")
        rec = ledger.decide("model.family", ("project", "p1"), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge", ledger=self.book, project="p1")
        ledger.label(rec, "ridge", "measured", "m", ledger=self.book)
        keep = ledger.decide("model.family", ("project", "p2"), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge", ledger=self.book, project="p2")
        self.assertEqual(retention.delete_events(self.log, project="p1"), 4)
        self.assertEqual(retention.delete_decisions(self.book, project="p1"), 1)
        self.assertEqual({e.ids.project for e in self.log.read()}, {"p2"})
        self.assertEqual([r.id for r in self.book.records()], [keep.id])
        self.assertEqual(self.book.labels(), [])

    def test_a_deleted_draft_leaves_nothing(self):
        experience.record("draft.created", "person", {"problem": "x"}, log=self.log, visit="v1", conversation="d1")
        rec = ledger.decide("model.family", ("project", "d1"), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge", ledger=self.book, draft="d1")
        ledger.label(rec, "ridge", "measured", "m", ledger=self.book)
        out = retention.forget(draft="d1", log=self.log, ledger=self.book)
        self.assertEqual((out["events"], out["decisions"], out.get("failed")), (1, 1, None))
        self.assertEqual((self.book.records(), self.book.labels(), self.log.read()), ([], [], []))

    def test_an_append_during_a_deletion_is_not_lost(self):
        if os.environ.get("DCLAB_DATABASE_URL"):
            self.skipTest("the file rewrite's lock")
        import threading

        events(self.log, project="p_gone", visit="vshared")
        done = threading.Event()

        def write():
            for i in range(400):
                experience.record("chat.person", "person", {"text": f"message {i}"}, log=self.log, visit="vshared", project="p_kept")
            done.set()

        writer = threading.Thread(target=write)
        writer.start()
        while not done.is_set():
            retention.delete_events(self.log, project="p_gone")
        writer.join()
        self.assertEqual(len([e for e in self.log.read() if e.ids.project == "p_kept"]), 400)

    def test_a_deleted_session_leaves_nothing(self):
        events(self.log, project="p1", visit="v1")
        events(self.log, project="p1", visit="v2")
        retention.delete_events(self.log, visit="v1")
        self.assertEqual({e.ids.visit for e in self.log.read()}, {"v2"})
        with self.assertRaises(ValueError):
            retention.delete_events(self.log)

    def test_the_record_still_refuses_an_ordinary_delete(self):
        if not os.environ.get("DCLAB_DATABASE_URL"):
            self.skipTest("the append-only triggers are PostgreSQL's")
        import sqlalchemy as sa

        from dclab_rnd.storage.models import decision_record, experience_event

        events(self.log)
        ledger.decide("model.family", ("project", "p1"), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge", ledger=self.book, project="p1")
        for table in (experience_event, decision_record):
            with self.subTest(table=table.name), self.assertRaises(Exception):
                with self.log.engine.begin() as c:
                    c.execute(sa.delete(table).where(table.c.workspace_id == self.log.workspace))

    def test_retention_by_class(self):
        old = datetime.now(timezone.utc) - timedelta(days=500)
        events(self.log, project="p1", visit="v1", at=old)
        events(self.log, project="p1", visit="v2")
        out = retention.sweep_events(self.log, text_days=365)
        self.assertEqual(out, {"events_deleted": 0, "texts_removed": 1})
        request = next(e for e in self.log.read(visit="v1") if e.kind == "model.request")
        self.assertNotIn("prompt", request.payload)
        self.assertIn("prompt:model_text", request.redacted)
        recent = next(e for e in self.log.read(visit="v2") if e.kind == "model.request")
        self.assertIn("prompt", recent.payload)
        out = retention.sweep_events(self.log, event_days=400)
        self.assertEqual(out["events_deleted"], 4)
        self.assertEqual({e.ids.visit for e in self.log.read()}, {"v2"})
        self.assertEqual(retention.sweep_events(self.log), {"events_deleted": 0, "texts_removed": 0})  # off unless set


@unittest.skipIf(MISSING, MISSING)
class StampTest(unittest.TestCase):
    """23.1: every event and decision carries the level it was written under; a later opt-in never widens it."""

    def test_the_ladder(self):
        from dclab_core import consent as core

        self.assertEqual(core.level_of({"improve": True, "train": True}, None), "train")
        self.assertEqual(core.level_of({"improve": True, "train": True}, {"improve": True, "train": False}), "improve")
        self.assertEqual(core.level_of({"improve": False, "train": True}, None), "train")
        self.assertEqual(core.level_of({}, None), "operate")
        self.assertTrue(core.exportable("train", "improve") and core.exportable("train", "train") and core.exportable("improve", "improve"))
        self.assertFalse(core.exportable("improve", "train") or core.exportable("operate", "improve") or core.exportable("nonsense", "improve"))
        moved = core.changes({"improve": False, "train": False}, {"improve": False, "train": True})
        self.assertEqual([(c["use"], c["on"], c["was"], c["level"], c["scope"]) for c in moved], [("train", True, False, "train", "workspace")])
        self.assertTrue(all(c["notice_version"] == core.NOTICE_VERSION for c in moved))

    def test_an_event_and_a_decision_written_at_operate_never_leave_after_a_later_opt_in(self):
        from dclab_rnd import consent, ledger
        from dclab_rnd.experience import FileLog

        home = Path(tempfile.mkdtemp())
        log, book = FileLog(home / "experience", "w1"), ledger.FileLedger(home / "ledger", "w1")
        first = experience.record("chat.person", "person", {"text": "hello", "surface": "home"}, log=log, visit="v1")
        first_decision = ledger.decide("column.is_identifier", ("column", "c1"), answers=[ledger.answer("rule", "r", True)], value_used=True, ledger=book)
        self.assertEqual((first.consent, first_decision.consent), ("operate", "operate"))
        consent.set_workspace_consent(home, "train", True)
        consent.set_workspace_consent(home, "improve", True)
        second = experience.record("chat.person", "person", {"text": "again", "surface": "home"}, log=log, visit="v1")
        second_decision = ledger.decide("column.is_identifier", ("column", "c2"), answers=[ledger.answer("rule", "r", True)], value_used=True, ledger=book)
        self.assertEqual((second.consent, second_decision.consent), ("train", "train"))
        switches = consent.workspace_consent(home)
        out = consent.export_events(log.read(), "train", switches, {})
        self.assertEqual([d["id"] for d in out], [second.id])  # the first stays home, whatever the switches say now
        self.assertEqual([d["id"] for d in consent.export_decisions(book.records(), book.labels(), switches, {})], [second_decision.id])
        self.assertEqual(consent.stamped_counts(log), {"operate": 1, "improve": 0, "train": 1})

    def test_a_projects_record_follows_its_own_switch_and_an_unknown_project_is_operate(self):
        from dclab_rnd import consent

        home = Path(tempfile.mkdtemp())
        consent.set_workspace_consent(home, "train", True)

        class Projects:
            def __init__(self):
                self.docs = {"p1": {"settings": {"consent": {"train": False}}}}

            def get(self, pid):
                return self.docs[pid]

        projects = Projects()
        self.assertEqual(consent.current_level(home=home, project="p1", projects=projects), "operate")
        projects.docs["p1"]["settings"]["consent"]["train"] = True
        self.assertEqual(consent.current_level(home=home, project="p1", projects=projects), "operate")  # cached: the API invalidates
        consent.invalidate("p1")
        self.assertEqual(consent.current_level(home=home, project="p1", projects=projects), "train")
        self.assertEqual(consent.current_level(home=home, project="missing", projects=projects), "operate")
        self.assertEqual(consent.current_level(home=home), "train")
        self.assertEqual(consent.current_level(home=home, project="p9", projects=None), "operate")  # the folder's store has no p9: never leaves


@unittest.skipIf(MISSING, MISSING)
class ProductTest(unittest.TestCase):
    """The switches, the deletions and the sweep as the product runs them."""

    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.home = Path(tempfile.mkdtemp())
        cls.client = TestClient(create_app(cls.home))
        cls.services = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}

    def audit(self, move):
        return [r for r in self.services.audit.query(500, 0)[1] if r.get("move") == move]

    def test_the_workspace_switches_are_off_until_the_owner_turns_them_on(self):
        self.assertEqual(self.client.get("/api/consent").json()["workspace"], {"improve": False, "train": False})
        self.assertEqual(self.client.put("/api/consent", json={"use": "sell", "on": True}, headers=self.h).status_code, 422)
        self.assertEqual(self.client.put("/api/consent", json={"use": "improve", "on": "yes"}, headers=self.h).status_code, 422)
        out = self.client.put("/api/consent", json={"use": "improve", "on": True}, headers=self.h).json()
        self.assertEqual(out["workspace"], {"improve": True, "train": False})
        self.assertEqual(consent.workspace_consent(self.services.store.home), {"improve": True, "train": False})
        self.client.put("/api/consent", json={"use": "improve", "on": True}, headers=self.h)  # no change: not audited again
        self.assertEqual([(r["detail"]["args"]["policy"], r["detail"]["args"]["on"]) for r in self.audit("set_consent") if not r.get("project_id")],
                         [("consent_improve", True)])
        self.client.put("/api/consent", json={"use": "improve", "on": False}, headers=self.h)

    def test_a_projects_train_switch_turns_off_the_old_opt_in_too(self):
        pid = self.client.post("/api/projects", json={"name": "Opted", "goal": "g"}, headers=self.h).json()["id"]
        self.client.patch(f"/api/projects/{pid}", json={"settings": {"share_for_training": True}}, headers=self.h)
        self.assertTrue(consent.project_consent(self.services.projects.get(pid))["train"])
        r = self.client.patch(f"/api/projects/{pid}", json={"settings": {"consent": {"train": False}}}, headers=self.h).json()
        self.assertFalse(consent.project_consent(r)["train"])
        self.assertFalse(r["settings"]["share_for_training"])
        self.assertIn(("consent_train", False), [(x["detail"]["args"]["policy"], x["detail"]["args"]["on"]) for x in self.audit("set_consent")
                                                if x.get("project_id") == pid])

    def test_prompts_kept_by_the_gateway_go_with_a_project_and_with_age(self):
        usage = self.services.gateway.usage
        old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat(timespec="seconds")
        for at, project in ((old, "p_usage_old"), (datetime.now(timezone.utc).isoformat(timespec="seconds"), "p_usage_new")):
            usage.record({"at": at, "purpose": "draft_chat", "tier": "standard", "model": "m", "endpoint": "e", "project_id": project,
                          "outcome": "ok", "prompt": f"the table: {CELL}", "input_tokens": 10, "output_tokens": 2, "seconds": 0.1, "attempts": 1})
        self.assertEqual(usage.forget_prompts(project_id="p_usage_new"), 1)
        from unittest import mock

        from dclab_rnd import retention as sweeper

        with mock.patch.dict(os.environ, {"DCLAB_RETENTION_MODEL_TEXT_DAYS": "365"}):
            sweeper.sweep_due(self.services, {}, 0.0)
        self.assertNotIn(CELL, json.dumps(usage.recent(50)))
        self.assertEqual(len([e for e in usage.recent(50) if str(e.get("project_id", "")).startswith("p_usage")]), 2)  # spend rows stay

    def test_a_projects_switches_are_its_settings(self):
        pid = self.client.post("/api/projects", json={"name": "Consent", "goal": "g"}, headers=self.h).json()["id"]
        r = self.client.patch(f"/api/projects/{pid}", json={"settings": {"consent": {"train": True, "improve": "no"}}}, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["settings"]["consent"], {"train": True})
        self.assertEqual(consent.project_consent(self.services.projects.get(pid)), {"improve": False, "train": True})
        self.assertIn(("consent_train", True), [(x["detail"]["args"]["policy"], x["detail"]["args"]["on"]) for x in self.audit("set_consent")
                                               if x.get("project_id") == pid])

    def test_deleting_a_project_a_draft_or_a_session_removes_its_record(self):
        pid = self.client.post("/api/projects", json={"name": "Gone", "goal": "g"}, headers=self.h).json()["id"]
        events(self.services.experience, project=pid, visit="vgone")
        events(self.services.experience, project="pkeep", visit="vkeep")
        ledger.decide("model.family", ("project", pid), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge",
                      ledger=self.services.ledger, project=pid)
        self.assertEqual(self.client.delete(f"/api/projects/{pid}", headers=self.h).status_code, 204)
        self.assertEqual([e for e in self.services.experience.read() if e.ids.project == pid], [])
        self.assertEqual([r for r in self.services.ledger.records() if r.project == pid], [])
        tomb = self.audit("delete_project")[0]["detail"]
        self.assertEqual((tomb["events"], tomb["decisions"], tomb.get("failed")), (4, 1, None))  # the tombstone: how much, never what

        draft = self.client.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=self.h).json()
        self.assertTrue([e for e in self.services.experience.read() if e.ids.conversation == draft["id"]])
        self.assertEqual(self.client.delete(f"/api/drafts/{draft['id']}", headers=self.h).status_code, 204)
        self.assertEqual([e for e in self.services.experience.read() if e.ids.conversation == draft["id"]], [])

        self.assertEqual(self.client.delete("/api/experience/sessions/vkeep", headers=self.h).status_code, 204)
        self.assertEqual([e for e in self.services.experience.read() if e.ids.visit == "vkeep"], [])
        self.assertEqual(self.audit("delete_session")[0]["detail"]["events"], 4)
        self.assertEqual(self.client.delete("/api/experience/sessions/vkeep", headers=self.h).status_code, 404)
        self.assertEqual(self.client.delete("/api/experience/sessions/a%20b", headers=self.h).status_code, 422)

    def test_the_hourly_sweep_applies_the_record_retention(self):
        from unittest import mock

        from dclab_rnd import retention as sweeper

        events(self.services.experience, project="pold", visit="vold", at=datetime.now(timezone.utc) - timedelta(days=900))
        with mock.patch.dict(os.environ, {"DCLAB_RETENTION_EVENT_DAYS": "730"}):
            sweeper.sweep_due(self.services, {}, 0.0)
        self.assertEqual([e for e in self.services.experience.read() if e.ids.visit == "vold"], [])
        self.assertEqual(self.audit("sweep_session_record")[0]["detail"]["events_deleted"], 4)
        events(self.services.experience, project="pold", visit="vold2", at=datetime.now(timezone.utc) - timedelta(days=900))
        with mock.patch.dict(os.environ, {"DCLAB_RETENTION_EVENT_DAYS": "0", "DCLAB_RETENTION_MODEL_TEXT_DAYS": "0", "DCLAB_RETENTION_RAW_DAYS": ""}):
            sweeper.sweep_due(self.services, {}, 0.0)  # 0 turns it off
        self.assertTrue([e for e in self.services.experience.read() if e.ids.visit == "vold2"])
        with mock.patch.dict(os.environ, {"DCLAB_RETENTION_EVENT_DAYS": "", "DCLAB_RETENTION_MODEL_TEXT_DAYS": ""}):
            sweeper.sweep_due(self.services, {}, 0.0)  # unset: the owner's 730 days (2026-10-15)
        self.assertEqual([e for e in self.services.experience.read() if e.ids.visit == "vold2"], [])


@unittest.skipIf(MISSING, MISSING)
class StampRoutesTest(unittest.TestCase):
    """23.1 on the product: a switch change is an event, and the consent page counts the stamps (its own app: its audit rows)."""

    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.home = Path(tempfile.mkdtemp())
        cls.client = TestClient(create_app(cls.home))
        cls.services = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}

    def test_the_legacy_opt_in_is_a_consent_change_too(self):
        pid = self.client.post("/api/projects", json={"name": "Legacy", "goal": "x"}, headers=self.h).json()["id"]
        self.client.patch(f"/api/projects/{pid}", json={"settings": {"share_for_training": True}}, headers=self.h)
        mine = [e for e in self.services.experience.read(project=pid) if e.kind == "consent.changed"]
        self.assertEqual([(e.payload["scope"], e.payload["use"], e.payload["on"]) for e in mine], [("project", "train", True)])

    def test_a_switch_change_is_in_the_record_and_the_page_counts_the_stamps(self):
        page = self.client.get("/api/consent").json()
        self.assertEqual((page["notice_version"], set(page["stamped"])), (consent.NOTICE_VERSION, {"operate", "improve", "train"}))
        self.client.put("/api/consent", json={"use": "improve", "on": True}, headers=self.h)
        changed = [e for e in self.services.experience.read() if e.kind == "consent.changed"]
        self.assertEqual([(e.payload["scope"], e.payload["use"], e.payload["on"], e.payload["level"]) for e in changed[-1:]],
                         [("workspace", "improve", True, "improve")])
        self.assertEqual(self.client.get("/api/consent").json()["level"], "improve")
        pid = self.client.post("/api/projects", json={"name": "Stamped", "goal": "x"}, headers=self.h).json()["id"]
        self.client.patch(f"/api/projects/{pid}", json={"settings": {"consent": {"improve": True}}}, headers=self.h)
        mine = [e for e in self.services.experience.read(project=pid) if e.kind == "consent.changed"]
        self.assertEqual([(e.payload["scope"], e.payload["use"], e.payload["on"]) for e in mine], [("project", "improve", True)])
        self.assertEqual(changed[-1].consent, "improve")  # the change is written under the level it brought in


if __name__ == "__main__":
    unittest.main()

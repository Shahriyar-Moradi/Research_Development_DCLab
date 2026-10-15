"""23.6: feedback of every kind, tied to the version that wrote what was judged. The core validates a request into a
payload (a thumb, a score, a correction, an edited reply, an override), the route stamps the message's version, the
conversation set uses an edit as the reply, a thumb up as the mark and a thumb down as the end of the example, and the
main product's accept, reject and revert of a proposal read as the same kind."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import datasets as builders, experience as x, feedback
    from dclab_rnd import experience
    from dclab_rnd.experience import hooks
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class CoreTest(unittest.TestCase):
    def test_each_kind_is_checked_into_a_payload(self):
        self.assertEqual(feedback.check({"on": "answer", "ref": "m12", "rating": "helpful"}),
                         {"kind": "thumbs", "on": "answer:m12", "rating": "helpful"})  # the old request still works
        self.assertEqual(feedback.check({"on": "message", "ref": "m12", "kind": "rating", "score": 5})["rating"], "helpful")
        self.assertEqual(feedback.check({"on": "message", "ref": "m12", "kind": "rating", "score": 2})["rating"], "unhelpful")
        self.assertEqual(feedback.check({"on": "message", "ref": "m12", "score": 3})["rating"], "neutral")
        self.assertEqual(feedback.check({"on": "decision", "ref": "dr123456", "rating": "wrong", "reason": "it was a key"})["kind"], "correction")
        edit = feedback.check({"on": "answer", "ref": "m12", "kind": "edited_output", "edited": "  Ask about the date first.  "})
        self.assertEqual((edit["edited"], edit["rating"]), ("Ask about the date first.", "edited"))
        self.assertEqual(feedback.check({"on": "proposal", "ref": "p1", "kind": "override", "reason": "no"})["rating"], "overridden")
        self.assertEqual(feedback.check({"on": "note", "ref": "models", "rating": "unhelpful"})["on"], "note:models")
        self.assertEqual(feedback.check({"on": "report", "rating": "helpful"})["on"], "report")  # no ref: the thing itself
        self.assertEqual(feedback.check({"on": "answer", "ref": "free text with spaces", "rating": "helpful"})["on"], "answer")  # an id's shape or nothing

    def test_what_is_wrong_is_said(self):
        for body in ({"on": "elsewhere", "rating": "helpful"}, {"on": "answer", "rating": "great"}, {"on": "answer"},
                     {"on": "answer", "kind": "rating", "score": 6}, {"on": "answer", "kind": "rating", "score": True},
                     {"on": "answer", "ref": "m1", "kind": "edited_output"}, {"on": "answer", "kind": "edited_output", "edited": "x"},
                     {"on": "report", "ref": "r1", "kind": "edited_output", "edited": "x"}, {"on": "answer", "ref": "m1", "kind": "override"},
                     {"on": "answer", "ref": "m1", "kind": "thumbs", "rating": "wrong"}):
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    feedback.check(body)

    def test_an_edit_beats_a_rejection_which_beats_an_approval(self):
        up = {"kind": "thumbs", "rating": "helpful"}
        down = {"kind": "thumbs", "rating": "unhelpful"}
        edit = {"kind": "edited_output", "edited": "my reply", "rating": "edited"}
        self.assertIsNone(feedback.verdict([]))
        self.assertEqual(feedback.verdict([up]), {"label": "approved"})
        self.assertEqual(feedback.verdict([up, down]), {"label": "rejected"})  # the newest thumb stands
        self.assertEqual(feedback.verdict([down, up]), {"label": "approved"})
        self.assertEqual(feedback.verdict([down, edit, {**edit, "edited": "better"}]), {"label": "edit", "text": "better"})
        self.assertEqual(feedback.verdict([{"rating": "wrong"}]), {"label": "rejected"})  # written before 23.6: no kind
        self.assertEqual(feedback.kind_of({"rating": "helpful"}), "thumbs")


def event(kind, payload, at, ident, conversation="d1", consent="train"):
    return {"id": ident, "at": at, "kind": kind, "payload": payload, "ids": {"conversation": conversation, "project": "p1"}, "consent": consent}


@unittest.skipIf(MISSING, MISSING)
class ConversationsTest(unittest.TestCase):
    def conversation(self, conversation="d1", extra=()):
        base = [event("draft.created", {"problem": "Which customers will leave?"}, "2026-10-16T10:00:00+00:00", f"e1{conversation}", conversation),
                event("chat.agent", {"text": "Do you have a column that says who left?", "message": "m1", "version": "core-0.43.0/prompts-aa", "surface": "home"},
                      "2026-10-16T10:00:01+00:00", f"e2{conversation}", conversation),
                event("chat.person", {"text": "Yes, churned", "surface": "home"}, "2026-10-16T10:00:02+00:00", f"e3{conversation}", conversation),
                event("chat.agent", {"text": "Then I will predict churned.", "message": "m2", "version": "core-0.43.0/prompts-aa", "surface": "home"},
                      "2026-10-16T10:00:03+00:00", f"e4{conversation}", conversation),
                event("draft.built", {}, "2026-10-16T10:00:04+00:00", f"e5{conversation}", conversation)]
        return base + list(extra)

    def build(self, events):
        rows, manifest = builders.conversations(events, workspace="w")
        return rows, manifest

    def test_an_edited_reply_is_the_examples_reply_and_the_original_is_kept(self):
        fb = event("feedback.given", {"on": "answer:m2", "kind": "edited_output", "edited": "Understood: the outcome is the churned column.", "rating": "edited",
                                      "version": "core-0.43.0/prompts-aa"}, "2026-10-16T11:00:00+00:00", "f1")
        rows, manifest = self.build(self.conversation(extra=[fb]))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["messages"][-1], {"role": "assistant", "content": "Understood: the outcome is the churned column."})
        self.assertEqual((row["label"], [e["original"] for e in row["edits"]], row["edits"][0]["version"]),
                         ("edit", ["Then I will predict churned."], "core-0.43.0/prompts-aa"))
        self.assertNotIn("Then I will predict churned.", str(row["messages"]))  # the original is beside it, not in the example
        self.assertEqual(manifest.counts["labels"], {"edit": 1})

    def test_a_thumb_up_marks_the_example_and_a_thumb_down_ends_it_before_that_reply(self):
        up = event("feedback.given", {"on": "answer:m2", "kind": "thumbs", "rating": "helpful"}, "2026-10-16T11:00:00+00:00", "f1")
        rows, _ = self.build(self.conversation(extra=[up]))
        self.assertEqual((rows[0]["label"], rows[0]["edits"], len(rows[0]["messages"])), ("thumbs_up", [], 5))
        down = event("feedback.given", {"on": "answer:m2", "kind": "thumbs", "rating": "unhelpful"}, "2026-10-16T11:00:00+00:00", "f2")
        rows, _ = self.build(self.conversation(extra=[down]))
        self.assertEqual([m["role"] for m in rows[0]["messages"]], ["system", "user", "assistant"])  # up to the reply before it
        self.assertEqual(rows[0]["label"], None)
        first = event("feedback.given", {"on": "answer:m1", "kind": "thumbs", "rating": "unhelpful"}, "2026-10-16T11:00:00+00:00", "f3")
        self.assertEqual(self.build(self.conversation(extra=[first]))[0], [])  # nothing left to teach

    def test_feedback_on_other_things_or_other_conversations_changes_nothing(self):
        other = [event("feedback.given", {"on": "note:models", "kind": "thumbs", "rating": "unhelpful"}, "2026-10-16T11:00:00+00:00", "f1"),
                 event("feedback.given", {"on": "answer:m2", "kind": "thumbs", "rating": "unhelpful"}, "2026-10-16T11:00:00+00:00", "f2", conversation="d2")]
        rows, _ = self.build(self.conversation(extra=other))
        self.assertEqual((rows[0]["label"], len(rows[0]["messages"])), (None, 5))
        self.assertEqual([r["label"] for r in self.build(self.conversation() + self.conversation("d2"))[0]], [None, None])


@unittest.skipIf(MISSING, MISSING)
class ExportTest(unittest.TestCase):
    """The real path: events through the consent export (which redacts anything shaped like an id), then the builder."""

    def make(self, log, conversation, texts, feedback_events, consecutive=False):
        import secrets

        ids = []
        experience.record("draft.created", "person", {"problem": "Which customers will leave?"}, log=log, conversation=conversation, project="p1")
        for text in texts:
            mid = "m" + secrets.token_hex(5)  # what draft/chat.py makes
            ids.append(mid)
            experience.record("chat.agent", "agent", {"text": text, "message": mid, "version": experience.release(), "surface": "home"}, log=log,
                              conversation=conversation, project="p1")
            if not consecutive:
                experience.record("chat.person", "person", {"text": "ok", "surface": "home"}, log=log, conversation=conversation, project="p1")
        for index, payload in feedback_events:
            experience.record("feedback.given", "person", {**payload, "on": f"answer:{ids[index]}"}, log=log, conversation=conversation, project="p1")
        experience.record("draft.built", "person", {"project": "p1"}, log=log, conversation=conversation, project="p1")
        return ids

    def rows(self, log):
        from dclab_rnd import consent

        on = {"improve": True, "train": True}
        events = [e.model_copy(update={"consent": "train"}) for e in log.read()]
        docs = consent.export_events(events, "train", on, {"p1": {"id": "p1", "settings": {"consent": on}}}, {})
        return builders.conversations(docs, workspace="w")[0]

    def test_an_edit_and_a_thumb_reach_the_message_they_were_given_on_after_the_export(self):
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w1")
        edit = {"kind": "edited_output", "edited": "Which column says who left?", "rating": "edited", "version": experience.release()}
        self.make(log, "d1", ["I read your file.", "Which column marks who left?"], [(1, edit)], consecutive=True)
        rows = self.rows(log)
        assistant = [m["content"] for m in rows[0]["messages"] if m["role"] == "assistant"]
        self.assertEqual(assistant, ["I read your file.\nWhich column says who left?"])  # only the second was replaced
        self.assertEqual([e["original"] for e in rows[0]["edits"]], ["Which column marks who left?"])
        self.assertEqual(rows[0]["edits"][0]["version"], experience.release())  # the version survives the export
        self.assertEqual(rows[0]["edits"][0]["index"], 2)  # the turn it sits in after the merge (system, user, assistant)
        log2 = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w2")
        down = {"kind": "thumbs", "rating": "unhelpful"}
        self.make(log2, "d2", ["First reply.", "Second reply."], [(1, down)])
        kept = self.rows(log2)
        self.assertEqual([m["content"] for m in kept[0]["messages"] if m["role"] == "assistant"], ["First reply."])  # ends before the rejected one
        log3 = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w3")
        self.make(log3, "d3", ["First reply.", "Second reply."], [(0, {"kind": "thumbs", "rating": "helpful"})])
        self.assertEqual(self.rows(log3)[0]["label"], "thumbs_up")

    def test_a_viewers_feedback_is_recorded_and_counts_for_nothing(self):
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w4")
        down = {"kind": "thumbs", "rating": "unhelpful", "role": "viewer"}
        self.make(log, "d4", ["First reply.", "Second reply."], [(1, down)])
        rows = self.rows(log)
        self.assertEqual((len([m for m in rows[0]["messages"] if m["role"] == "assistant"]), rows[0]["label"]), (2, None))
        self.assertEqual(feedback.verdict([down]), None)
        self.assertEqual(feedback.verdict([down, {"kind": "thumbs", "rating": "helpful", "role": "owner"}]), {"label": "approved"})


@unittest.skipIf(MISSING, MISSING)
class ProductTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        cls.services = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def given(self, draft=None):
        return [e for e in self.services.experience.read(kinds=["feedback.given"]) if draft is None or e.ids.conversation == draft]

    def test_a_home_reply_carries_its_id_and_the_release_that_wrote_it(self):
        from dclab_rnd import context

        draft = self.services.drafts.create("Which customers will leave us next month?")["id"]
        with context.using(self.services):
            hooks.from_draft_event(draft, "chat", {"role": "agent", "id": "m7", "text": "Do you have a column that says who left?", "kind": "question"})
        said = [e for e in self.services.experience.read(conversation=draft, kinds=["chat.agent"])]
        self.assertEqual((said[-1].payload["message"], said[-1].payload["version"]), ("m7", experience.release()))
        self.assertRegex(experience.release(), r"^core-\d+\.\d+\.\d+/prompts-[0-9a-f]{8}$")
        import unittest.mock as mock

        with mock.patch.object(experience, "_RELEASE", []), mock.patch("dclab_rnd.prompts.fingerprints", side_effect=RuntimeError("down")):
            self.assertTrue(experience.release().endswith("prompts-unknown"))
            self.assertEqual(experience._RELEASE, [])  # a failure is not remembered for the life of the process

    def test_a_thumb_is_tied_to_the_version_of_the_message_it_judges(self):
        draft = self.services.drafts.create("Which customers will leave us next month?")["id"]
        experience.record("chat.agent", "agent", {"text": "Written by an older release.", "message": "m9", "version": "core-0.1.0/prompts-00000000", "surface": "home"},
                          log=self.services.experience, conversation=draft)
        out = self.client.post("/api/feedback", json={"on": "answer", "ref": "m9", "kind": "thumbs", "rating": "unhelpful", "draft": draft}, headers=self.h)
        self.assertEqual(out.status_code, 200, out.text)
        e = self.given(draft)[-1]
        self.assertEqual((e.payload["kind"], e.payload["on"], e.payload["rating"], e.payload["version"]),
                         ("thumbs", "answer:m9", "unhelpful", "core-0.1.0/prompts-00000000"))  # the release that wrote it, not the one now
        edit = self.client.post("/api/feedback", json={"on": "answer", "ref": "m9", "kind": "edited_output", "edited": "Ask which column says who left.",
                                                       "draft": draft}, headers=self.h)
        self.assertEqual(edit.status_code, 200, edit.text)
        e = self.given(draft)[-1]
        self.assertEqual((e.payload["kind"], e.payload["edited"], e.classes["edited"], e.payload["version"]),
                         ("edited_output", "Ask which column says who left.", "person_text", "core-0.1.0/prompts-00000000"))
        unknown = self.client.post("/api/feedback", json={"on": "answer", "ref": "m404", "rating": "helpful", "draft": draft}, headers=self.h)
        self.assertEqual(unknown.status_code, 200)
        self.assertNotIn("version", self.given(draft)[-1].payload)  # no such message: no version, never today's release claiming to have written it
        self.assertEqual(self.given(draft)[-1].payload["role"], "owner")  # no accounts: the local owner
        note = self.client.post("/api/feedback", json={"on": "note", "ref": "models", "rating": "helpful"}, headers=self.h)
        self.assertEqual(note.status_code, 200)
        self.assertEqual(self.given()[-1].payload["on"], "note:models")

    def test_a_bad_request_is_refused_and_writes_nothing(self):
        before = len(self.given())
        for body in ({"on": "answer", "ref": "m1", "kind": "edited_output"}, {"on": "answer", "kind": "rating", "score": 9},
                     {"on": "nowhere", "rating": "helpful"}, {"on": "answer", "rating": "meh"}, {"on": "answer", "rating": ["helpful"]},
                     {"on": "answer", "rating": {"a": 1}}):
            self.assertEqual(self.client.post("/api/feedback", json=body, headers=self.h).status_code, 422, body)
        self.assertEqual(len(self.given()), before)
        self.assertEqual(self.client.post("/api/feedback", json={"on": "decision", "ref": "dr-nope-0001", "kind": "override"}, headers=self.h).status_code, 404)

    def test_an_edit_leaves_only_as_redacted_person_text_and_with_consent(self):
        from dclab_rnd import consent

        e = experience.record("feedback.given", "person", {"kind": "edited_output", "on": "answer:m1", "rating": "edited", "edited": "Mail me at jane@example.com"},
                              log=self.services.experience, conversation="dX")
        self.assertEqual(e.consent, "operate")
        on = {"improve": True, "train": True}
        self.assertEqual(consent.export_events([e], "train", on, {}, {}), [])  # written at operate: it never leaves, whatever the switches say
        shown = consent.export_events([e.model_copy(update={"consent": "train"})], "train", on, {}, {})  # as if written under train
        self.assertEqual((shown[0]["payload"]["edited"], shown[0]["classes"]["edited"]), ("Mail me at [EMAIL]", "person_text"))  # redacted on the way out


@unittest.skipIf(MISSING, MISSING)
class MainProductTest(unittest.TestCase):
    def row(self, kind, **over):
        base = {"id": "r1", "workspace_id": "w", "project_id": "p", "decision_type": kind, "state": "recorded", "subject_kind": "proposal",
                "subject_digest": "abc123", "actor_kind": "human", "actor_user_id": "u1", "rationale": "the split is wrong for us",
                "rationale_untrusted": False, "event_at": "2026-10-16T10:00:00+00:00"}
        from dclab_core.adapters import main_product_experience as mp

        return {**{c: None for c in mp.COLUMNS["project_decision_records"]}, **base, **over} if hasattr(mp, "COLUMNS") else {**base, **over}

    def test_a_persons_accept_reject_and_revert_of_a_proposal_read_as_feedback(self):
        from dclab_core.adapters import main_product_experience as mp

        out = mp.feedback_events([self.row("proposal_accepted"), self.row("proposal_rejected", id="r2"), self.row("proposal_reverted", id="r3"),
                                  self.row("proposal_accepted", id="r4", actor_kind="agent"), self.row("winner_locked", id="r5")], consent="improve")
        self.assertEqual([(e.payload["kind"], e.payload["rating"], e.payload["on"]) for e in out],
                         [("thumbs", "helpful", "proposal:abc123"), ("thumbs", "unhelpful", "proposal:abc123"), ("override", "overridden", "proposal:abc123")])
        self.assertEqual({e.consent for e in out}, {"improve"})
        self.assertEqual(out[1].payload["reason"], "the split is wrong for us")
        untrusted = mp.feedback_events([self.row("proposal_rejected", rationale_untrusted=True)])
        self.assertNotIn("reason", untrusted[0].payload)  # an untrusted rationale stays out


if __name__ == "__main__":
    unittest.main()

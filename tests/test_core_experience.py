"""Package 19.1: the experience contract. One event shape with every shared id and a declared content class for each
field; the vocabulary closed and published in the JSON Schema; credentials never recorded; the timeline, redaction
and the session card as library functions; the main product's own rows (agent events, model invocations, decision
records, run events) read as the same events, every one of its recorded types either mapped or left out with a reason."""

import json
import sys
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pydantic import ValidationError

    from dclab_core import experience as x
    from dclab_core.adapters import main_product_experience as mp
    from dclab_core.adapters.main_product import recorded
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""

T0 = datetime(2026, 10, 13, 5, 56, 42, tzinfo=timezone.utc)
IDS = {"workspace": "w1", "user": "u1", "visit": "v1", "conversation": "d1"}


def at(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def morning() -> list:
    """The owner's morning, as events: the problem, the frauds' file, a question, an answer, a refused question, the build."""
    e = []

    def add(kind, actor, payload, s, **ids):
        e.append(x.make(kind, actor, {**IDS, **ids}, payload, id=f"ev{len(e):06d}", at=at(s), seq=len(e)))

    add("visit.started", "system", {"surface": "home"}, 0)
    add("draft.created", "person", {"problem": "I want build anomaly detection", "pack": "imbalanced"}, 1)
    add("data.uploaded", "person", {"name": "part2-fraud-sample.csv", "bytes": 25693, "rows": 100, "columns": 17}, 50)
    add("data.cases_found", "system", {"keys": ["order_request_id"], "case_kinds": ["reason"], "tables_matched": 1}, 53)
    add("chat.agent", "agent", {"text": "Which column is the outcome the model should predict?", "kind": "question",
                                "question_field": "target"}, 53)
    add("chat.person", "person", {"text": "it is anomaly detection", "answers_question": "q2"}, 78)
    add("decision.refused", "system", {"point": "ask.target", "rule": "plan order", "reason": "not the plan's next field"}, 79)
    add("model.request", "model", {"purpose": "home_agent", "model": "gpt-6-luna", "input_tokens": 900, "output_tokens": 80, "cost": 0.0021, "currency": "EUR",
                                   "prompt": "system: ...", "reply": "ask_user(target)"}, 79, model_request="mr1")
    add("solution.refused", "system", {"target": "customer_id", "reason": "named as a key", "rule": "Target can be an outcome"}, 120)
    return e


@unittest.skipIf(MISSING, MISSING)
class ContractTest(unittest.TestCase):
    def test_every_kind_round_trips_and_is_read_only_with_its_version(self):
        for kind, fields in x.KINDS.items():
            with self.subTest(kind=kind):
                payload = {f: (1 if c == "count" else "a") for f, c in fields.items()}
                event = x.make(kind, "system", {"workspace": "w"}, payload, id="ev-round")
                self.assertEqual(x.ExperienceEvent.read(event.model_dump_json()), event)
                self.assertEqual(set(event.classes.values()), set(fields.values()))

    def test_a_v2_event_reads_as_v6_at_operate_and_a_stamp_is_kept(self):
        import json

        e = x.make("chat.person", "person", IDS, {"text": "hi", "surface": "home"}, id="ev000001", at=at(0), consent="train")
        self.assertEqual(e.consent, "train")
        doc = json.loads(e.model_dump_json())
        old = {k: v for k, v in doc.items() if k != "consent"} | {"schema_version": "dclab_core.experience_event.v2"}
        self.assertEqual((x.ExperienceEvent.read(old).schema_version, x.ExperienceEvent.read(old).consent), ("dclab_core.experience_event.v6", "operate"))
        self.assertEqual(x.ExperienceEvent.read(doc).consent, "train")
        with self.assertRaises(Exception):
            x.make("chat.person", "person", IDS, {"text": "hi"}, id="ev000002", at=at(0), consent="whatever")
        changed = x.make("consent.changed", "person", IDS, {"scope": "workspace", "use": "train", "on": True, "was": False, "level": "train",
                                                           "notice_version": "2026-10-15"}, id="ev000003", at=at(1))
        self.assertEqual(set(changed.classes.values()), {"meta"})

    def test_an_ipv6_prefix_never_keeps_a_host_group_and_a_count_is_finite(self):
        self.assertEqual(x.truncate_ip("2001:db8::1"), "2001:db8:0::")
        self.assertEqual(x.truncate_ip("2001:db8:85a3:0:0:8a2e:370:7334"), "2001:db8:85a3::")
        self.assertEqual(x.truncate_ip("::1"), "0:0:0::")
        self.assertIsNone(x.truncate_ip("2001:db8"))
        with self.assertRaises(ValueError):
            x.ui_payload("ui.upload_started", {"view": "home", "bytes": float("inf")})

    def test_a_user_agent_is_a_hash_and_an_address_loses_its_last_part(self):
        self.assertEqual(len(x.hash_user_agent("Mozilla/5.0 (Macintosh)")), 16)
        self.assertNotEqual(x.hash_user_agent("Mozilla/5.0 (Macintosh)"), x.hash_user_agent("Mozilla/5.0 (Windows)"))
        self.assertIsNone(x.hash_user_agent(""))
        self.assertEqual(x.truncate_ip("203.0.113.9"), "203.0.113.x")
        self.assertEqual(x.truncate_ip("2001:db8:85a3::8a2e:370:7334"), "2001:db8:85a3::")
        self.assertIsNone(x.truncate_ip("not an address"))
        e = x.make("auth.sign_in", "person", IDS, {"via": "password", "session": "s1", "agent": "abcd", "net": "203.0.113.x"}, id="ev000010", at=at(0))
        self.assertEqual(set(e.classes.values()), {"meta"})
        named = x.make("auth.token_created", "person", IDS, {"token": "t1", "name": "my laptop"}, id="ev000011", at=at(0))
        self.assertEqual(named.classes["name"], "person_text")  # a token's name is the person's own words

    def test_what_may_not_be_recorded_is_refused(self):
        good = x.make("chat.person", "person", IDS, {"text": "hello"}, id="ev-good")
        cases = {
            "unknown kind": lambda: x.make("chat.whisper", "person", IDS, {}, id="ev-bad1"),
            "field not in the vocabulary": lambda: x.make("chat.person", "person", IDS, {"text": "a", "mood": "x"}, id="ev-bad2"),
            "a field recorded as a lower class": lambda: x.ExperienceEvent(**{**good.model_dump(), "classes": {"text": "meta"}}),
            "a field without its class": lambda: x.ExperienceEvent(**{**good.model_dump(), "classes": {}}),
            "prose in a meta field": lambda: x.make("graph.move", "system", IDS, {"move": "x" * 400}, id="ev-bad3"),
            "a credential": lambda: x.make("chat.person", "person", IDS, {"text": "my key is " + "sk-" + "abcdefghijklmnopqrstuvwxyz012"}, id="ev-bad4"),
            "a private key": lambda: x.make("model.request", "model", IDS, {"prompt": "-----BEGIN " + "RSA PRIVATE KEY-----"}, id="ev-bad5"),
            "no workspace": lambda: x.make("chat.person", "person", {"user": "u"}, {"text": "a"}, id="ev-bad6"),
            "a tool's result recorded as meta": lambda: x.ExperienceEvent(**{**x.make("tool.call", "agent", IDS, {"result": {"a": 1}}, id="ev-tool0").model_dump(),
                                                                          "classes": {"result": "meta"}}),
            "a word in a count": lambda: x.make("data.uploaded", "person", IDS, {"rows": "alice@x.com"}, id="ev-bad7"),
            "a yes in a count": lambda: x.make("model.request", "model", IDS, {"cost": True}, id="ev-bad8"),
            "values nested in meta": lambda: x.make("data.analysed", "system", IDS, {"roles": {"a": {"b": "alice"}}}, id="ev-bad9"),
            "a time without its zone": lambda: x.make("chat.person", "person", IDS, {"text": "a"}, id="ev-bad10", at="2026-10-13T06:00:00"),
            "a credential in an id": lambda: x.make("chat.person", "person", {**IDS, "request": "sk-" + "abcdefghijklmnopqrstuvwxyz012"}, {"text": "a"}, id="ev-bad11"),
        }
        for name, build in cases.items():
            with self.subTest(name):
                with self.assertRaises((ValidationError, ValueError)):
                    build()

    def test_credentials_are_refused_and_ordinary_words_are_not(self):
        j = "".join  # every example is assembled here: no key-shaped text sits in the repository (scripts/scan_secrets.py)
        secrets = [j(["sk-", "proj-", "abcdefghijklmnopqrstuvwxyz0123"]), j(["sk-", "ant-api03-", "abcdefghijklmnopqrstuvwx"]),
                   j(["AK", "IA", "ABCDEFGHIJKLMNOP"]), j(["AS", "IA", "ABCDEFGHIJKLMNOP"]), j(["gh", "p_", "a" * 36]), j(["github", "_pat_", "a" * 45]),
                   j(["h", "f_", "a" * 34]), j(["AI", "za", "b" * 35]), j(["sk", "_live_", "c" * 24]),
                   j(["ey", "JhbGciOiJIUzI1NiJ9.", "eyJzdWIiOiIxMjM0NTY3ODkwIn0.", "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"]),
                   j(["postgresql://dclab:", "S3cretPass", "@db.internal/dclab"]), j(["authorization: bearer ", "abcdefghijklmnopqrstuvwx"]),
                   j(["-----BEGIN ", "PRIVATE KEY-----"])]
        for secret in secrets:
            with self.subTest(secret[:12]):
                with self.assertRaises(ValueError):
                    x.make("chat.person", "person", IDS, {"text": f"here: {secret}"}, id="ev-secret")
        for words in ("I have a churn-risk-prediction-for-telco problem", "multi-task-learning-with-shared-encoders",
                      "desk-booking-no-show-prediction", "the task-specific key is order_id", "postgresql://db.internal/dclab"):
            with self.subTest(words):
                self.assertEqual(x.make("chat.person", "person", IDS, {"text": words}, id="ev-words").payload["text"], words)

    def test_times_from_any_zone_are_stored_in_utc_and_order_together(self):
        tehran = x.make("chat.person", "person", IDS, {"text": "a"}, id="ev-teh", at="2026-10-13T09:30:00+03:30")
        utc = x.make("chat.person", "person", IDS, {"text": "b"}, id="ev-utc", at="2026-10-13T06:00:00.000001+00:00")
        self.assertEqual(tehran.at, "2026-10-13T06:00:00.000000+00:00")
        self.assertEqual([e.id for e in x.timeline([utc, tehran])], ["ev-teh", "ev-utc"])

    def test_the_published_schema_holds_the_vocabulary(self):
        import jsonschema

        from dclab_core import schemas

        schema = schemas.generate()["experience_event"]
        validator = jsonschema.Draft202012Validator(schema)
        good = json.loads(x.make("chat.person", "person", IDS, {"text": "hello"}, id="ev-good").model_dump_json())
        self.assertEqual(list(validator.iter_errors(good)), [])
        for bad in ({**good, "classes": {"text": "meta"}}, {**good, "payload": {"mood": "x"}, "classes": {"mood": "meta"}}, {**good, "kind": "chat.whisper"},
                    {**good, "classes": {}}, {**good, "payload": {}}):  # a field without its class, a class without its field
            self.assertTrue(list(validator.iter_errors(bad)), bad)

    def test_the_timeline_orders_by_time_then_sequence_and_keeps_an_event_once(self):
        events = morning()
        shuffled = list(reversed(events)) + events[:3]
        ordered = x.timeline(shuffled)
        self.assertEqual([e.id for e in ordered], [e.id for e in events])

    def test_redaction_keeps_only_the_allowed_classes_and_says_what_it_removed(self):
        request = next(e for e in morning() if e.kind == "model.request")
        shared = x.redact(request, x.SHAREABLE)
        self.assertNotIn("prompt", shared.payload)
        self.assertNotIn("reply", shared.payload)
        self.assertEqual(shared.payload["cost"], 0.0021)
        self.assertEqual(shared.redacted, ["prompt:model_text", "reply:model_text"])
        self.assertEqual(x.ExperienceEvent.read(shared.model_dump_json()), shared)
        with self.assertRaises(ValueError):
            x.redact(request, {"secret"})
        mislabelled = next(e for e in morning() if e.kind == "chat.person").model_copy(update={"classes": {"text": "meta"}})  # skips validation
        self.assertNotIn("text", x.redact(mislabelled, x.SHAREABLE).payload)  # the vocabulary decides, not the event's labels

    def test_the_session_card_sums_one_visit(self):
        card = x.session_card(morning())
        self.assertEqual((card.visit, card.user, card.person_turns, card.agent_turns, card.model_requests), ("v1", "u1", 1, 1, 1))
        self.assertEqual(card.refused, 2)  # the question refused by the plan's order, and the key refused as the outcome
        self.assertEqual(card.outcome, "open")
        self.assertEqual(card.cost, {"EUR": 0.0021})
        self.assertEqual((card.problem, card.kind), ("I want build anomaly detection", "imbalanced"))
        self.assertEqual(x.ExperienceSession.read(card.model_dump_json()), card)
        later = [x.make("draft.built", "system", IDS, {"project": "p1", "target": "is_case", "kind": "rare_event"}, id="ev-built", at=at(200)),
                 x.make("graph.move", "agent", IDS, {"move": "run_stage", "allowed": False}, id="ev-graph", at=at(201)),
                 x.make("tool.call", "agent", IDS, {"tool": "run_stage", "verdict": "denied"}, id="ev-tool1", at=at(202)),
                 x.make("error.raised", "system", IDS, {"where": "engine", "code": "E1"}, id="ev-error", at=at(203)),
                 x.make("model.request", "model", IDS, {"cost": 0.01, "currency": "USD"}, id="ev-model", at=at(204))]
        built = x.session_card(morning() + later)
        self.assertEqual((built.outcome, built.refused, built.errors, built.kind), ("built", 4, 1, "rare_event"))
        self.assertEqual(built.cost, {"EUR": 0.0021, "USD": 0.01})
        ran = x.session_card(morning() + later + [x.make("stage.finished", "system", IDS, {"stage": "data"}, id="ev-stage1", at=at(300))])
        self.assertEqual(ran.outcome, "ran")
        for final in ("final", "terminal"):  # this studio's last stage, and the main product's
            done = x.session_card(morning() + [x.make("stage.finished", "system", IDS, {"stage": final}, id="ev-stage2", at=at(400))])
            self.assertEqual(done.outcome, "finished")
        other = x.make("chat.person", "person", {**IDS, "visit": "v2"}, {"text": "hi"}, id="ev-other")
        with self.assertRaises(ValueError):
            x.session_card(morning() + [other])


def row_id():
    return str(uuid.uuid4())


@unittest.skipIf(MISSING, MISSING)
class MainProductTest(unittest.TestCase):
    """Rows shaped exactly like the main product's tables at the recorded commit, read as events."""

    def setUp(self):
        self.columns = recorded()["experience_tables"]
        self.ws, self.user, self.project = row_id(), row_id(), row_id()
        self.run = {c: None for c in self.columns["agent_runs"]} | {"id": row_id(), "workspace_id": self.ws, "project_id": self.project,
                                                                     "kind": "assistant", "agent_key": "assistant", "purpose": "assistant.turn",
                                                                     "decision_point_key": "assistant.turn", "model": "m", "created_by_user_id": self.user}

    def agent_event(self, type_, payload, seq):
        return {c: None for c in self.columns["agent_events"]} | {"id": row_id(), "workspace_id": self.ws, "run_id": self.run["id"], "seq": seq,
                                                                  "type": type_, "payload": payload, "created_at": at(seq)}

    def test_every_recorded_type_is_mapped_or_left_out_with_a_reason(self):
        vocab = recorded()["experience_vocabularies"]
        self.assertEqual(set(mp.AGENT_EVENT_KINDS) | set(mp.AGENT_EVENTS_NOT_MAPPED), set(vocab["agent_events.type"]))
        self.assertFalse(set(mp.AGENT_EVENT_KINDS) & set(mp.AGENT_EVENTS_NOT_MAPPED))
        self.assertEqual(set(mp.DECISION_KINDS) | set(mp.DECISIONS_NOT_MAPPED), set(vocab["project_decision_records.decision_type"]))
        self.assertEqual(set(mp.ACTORS), set(vocab["project_decision_records.actor_kind"]))
        self.assertTrue(set(mp.AGENT_EVENT_KINDS.values()) | set(mp.DECISION_KINDS.values()) <= set(x.KINDS))
        produced = set(mp.AGENT_EVENT_KINDS.values()) | set(mp.DECISION_KINDS.values()) | {"stage.started", "stage.finished", "stage.failed", "run.recorded"}
        self.assertEqual(produced | set(mp.KINDS_MISSING), set(x.KINDS))  # every contract kind is produced or listed as missing
        self.assertFalse(produced & set(mp.KINDS_MISSING))
        self.assertEqual(set(mp.RUN_STATUS), set(vocab["ml_run_events.status"]))

    def test_a_thread_reads_as_one_conversation_with_its_person_and_project(self):
        turn = self.run | {"id": row_id(), "kind": "lead", "parent_run_id": self.run["id"]}  # each model turn is a lead run under the thread
        runs = {self.run["id"]: self.run, turn["id"]: turn}
        rows = [self.agent_event("run_started", {}, 0), self.agent_event("user_message", {"message": "Why was the holdout lower?"}, 1),
                self.agent_event("assistant_message", {"kind": "answer", "message": "The holdout covers a later week."}, 2) | {"run_id": turn["id"]},
                self.agent_event("tool_call_denied", {"call": 1, "tool": "run_stage", "code": "not_allowed"}, 3) | {"run_id": turn["id"]},
                self.agent_event("tool_call_finished", {"call": 2, "tool": "read_record", "ok": False, "code": "missing"}, 4) | {"run_id": turn["id"]},
                self.agent_event("step_rejected", {"call": 3, "reasons": ["cites an unknown record"]}, 5) | {"run_id": turn["id"]},
                self.agent_event("llm_call_finished", {}, 6) | {"run_id": turn["id"], "llm_invocation_id": row_id()}]
        events = x.timeline(mp.events(agent_events=rows, runs=runs))
        self.assertEqual([e.kind for e in events], ["chat.person", "chat.agent", "tool.call", "tool.call", "decision.refused", "model.request"])
        self.assertEqual({e.ids.conversation for e in events}, {self.run["id"]})  # one thread, not one conversation per turn
        person = events[0]
        self.assertEqual((person.ids.user, person.ids.project, person.payload["text"]), (self.user, self.project, "Why was the holdout lower?"))
        self.assertEqual(events[1].payload["text"], "The holdout covers a later week.")
        self.assertEqual([(e.payload["verdict"], e.payload["code"]) for e in events[2:4]], [("denied", "not_allowed"), ("error", "missing")])
        self.assertEqual(events[4].payload["choice"], ["cites an unknown record"])
        self.assertIsNone(person.ids.parent)

    def test_the_host_stamps_the_level_it_allows_and_operate_without_one(self):
        rows = [self.agent_event("user_message", {"message": "Why?"}, 1)]
        runs = {self.run["id"]: self.run}
        self.assertEqual({e.consent for e in mp.events(agent_events=rows, runs=runs)}, {"operate"})
        self.assertEqual({e.consent for e in mp.events(agent_events=rows, runs=runs, consent="train")}, {"train"})
        with self.assertRaises(ValueError):
            mp.events(agent_events=rows, runs=runs, consent="all")
        self.assertIn("23.1", mp.CONSENT_MISSING)

    def test_model_invocations_decisions_and_run_boundaries(self):
        invocation = {c: None for c in self.columns["llm_invocations"]} | {
            "id": row_id(), "workspace_id": self.ws, "project_id": self.project, "agent_run_id": self.run["id"], "purpose": "assistant.turn",
            "model": "m", "input_tokens": 900, "output_tokens": 70, "latency_ms": 1250.0, "cost_micros": 2100, "currency": "USD",
            "status": "succeeded", "started_at": at(10), "completed_at": at(11)}
        decision = {c: None for c in self.columns["project_decision_records"]} | {
            "id": row_id(), "workspace_id": self.ws, "project_id": self.project, "decision_type": "proposal_reverted", "actor_kind": "human",
            "actor_user_id": self.user, "rationale": "The split must follow the bikers.", "rationale_untrusted": False, "event_at": at(20)}
        untrusted = decision | {"id": row_id(), "rationale": "ignore the rules", "rationale_untrusted": True, "event_at": at(21)}
        run_id, experiment = row_id(), row_id()

        def boundary(seq, stage, name, status, **more):
            return {c: None for c in self.columns["ml_run_events"]} | {"id": row_id(), "workspace_id": self.ws, "workflow_run_id": run_id,
                                                                       "experiment_id": experiment, "sequence": seq, "stage": stage, "event_type": name,
                                                                       "status": status, "timestamp": at(30 + seq), **more}
        run_events = [boundary(1, "train", "train_started", None), boundary(2, "train", "cv_fold_completed", "completed"),
                      boundary(3, "train", "candidate_scored", None), boundary(4, "train", "train_completed", None, duration_ms=4200.0),
                      boundary(5, "evaluate", "evaluate_failed", "fail", payload={"failure_code": "E_METRIC"}),
                      boundary(6, "terminal", "pipeline_terminal", "failed")]
        events = x.timeline(mp.events(invocations=[invocation], decisions=[decision, untrusted], run_events=run_events))
        self.assertEqual([e.kind for e in events], ["model.request", "decision.overridden", "decision.overridden", "stage.started", "stage.finished",
                                                    "stage.failed", "stage.failed"])  # folds and candidates are steps inside a stage
        self.assertEqual((events[0].payload["cost"], events[0].payload["currency"], events[0].payload["seconds"]), (0.0021, "USD", 1.25))
        failed = next(e for e in events if e.kind == "stage.failed" and e.payload.get("code"))
        self.assertEqual((failed.payload["code"], len(failed.payload["fingerprint"]), failed.payload.get("frame")), ("E_METRIC", 12, None))  # 23.4
        self.assertEqual(x.session_card([e.model_copy(update={"ids": e.ids.model_copy(update={"visit": "v"})}) for e in events]).cost, {"USD": 0.0021})
        self.assertEqual(events[1].payload["reason"], "The split must follow the bikers.")
        self.assertNotIn("reason", events[2].payload)  # an untrusted rationale stays out
        self.assertEqual(events[1].actor, "person")
        self.assertEqual(events[4].payload, {"stage": "train", "seconds": 4.2})
        self.assertEqual({k: v for k, v in events[5].payload.items() if k != "fingerprint"}, {"stage": "evaluate", "code": "E_METRIC"})
        self.assertEqual(events[6].payload["stage"], "terminal")

    def test_a_column_the_recorded_commit_did_not_have_fails_loudly(self):
        row = self.agent_event("user_message", {"text": "hi"}, 1) | {"session_id": "s1"}
        with self.assertRaisesRegex(ValueError, "read the main product again"):
            mp.from_agent_event(row, self.run)
        self.assertEqual(set(mp.IDS_MISSING), {"visit", "request"})


if __name__ == "__main__":
    unittest.main()

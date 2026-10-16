"""Package 22.8, decision models for data scientists: a technical decision asked about a column gives the same answer,
with its confidence and its evidence, through the API, the MCP tool and the notebook companion; a disagreement between
the rule and a classifier is shown; a person's own answer is recorded as a decision, and as a label when it differs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import ledger as contract  # noqa: E402

try:
    import pandas  # noqa: F401

    from dclab_rnd import decide, kinds, ledger
    from dclab_rnd.ledger import shadow
    from dclab_rnd.studio.data import profile_table
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""


def states(key="binary"):
    table = kinds.get(key).case(0).table
    return [ledger.column_state(c, len(table)) for c in profile_table(table)["columns"]]


@unittest.skipIf(MISSING, MISSING)
class DecideTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        env = mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)

    def test_the_api_the_mcp_tool_and_the_companion_give_the_same_answer(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.agents import default_registry

        client = TestClient(create_app(Path(self.home)))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        asked = 0
        from dclab_rnd.ledger import routing

        name = next((f"{m.name}@{m.version}" for m in shadow.answerers() if "column.is_identifier" in m.points), None)
        if name:  # a point routed to act, read alike by the three: the API's gateway, the tool's, the companion's folder
            store = client.app.state.services.gateway.routing
            store.save({**store.get(), "decisions": {"column.is_identifier": {"answerer": name, "level": 3, "bar": 0.9, "since": "2026-01-01T00:00:00+00:00"}}})
        for state in states()[:4]:
            for point in decide.TECHNICAL:
                api = client.post("/api/decide", json={"point": point, "state": state}, headers=h)
                self.assertEqual(api.status_code, 200, api.text)
                tool = default_registry().call("decide", {"point": point, "state": state}, scope="project")
                companion = subprocess.run([sys.executable, "-m", "dclab_rnd.notebook_assist", "--decide", json.dumps({"point": point, "state": state}),
                                            "--home", self.home],
                                           capture_output=True, text=True, cwd=ROOT, env={**os.environ, "DCLAB_NO_LIVE_MODELS": "1"}, timeout=300)
                self.assertEqual(companion.returncode, 0, companion.stderr[-800:])
                one = api.json()
                self.assertEqual(one, json.loads(tool if isinstance(tool, str) else json.dumps(tool)))
                self.assertEqual(one, json.loads(companion.stdout))
                self.assertTrue(one["evidence"] and one["rule"] is not None)
                self.assertEqual(one["routing"], "workspace")
                if name and point == "column.is_identifier":
                    self.assertEqual(one["level"], 3)  # the routed model answers, on every surface
                for m in one["models"]:
                    self.assertIsNotNone(m["confidence"])  # each answer with its confidence
                asked += 1
        self.assertEqual(asked, 4 * len(decide.TECHNICAL))

    def test_a_disagreement_is_shown_and_the_rule_stands(self):
        class Says:
            by, name, version, points = "classifier", "leak.test", "t1", frozenset({"feature.leakage_suspect"})

            def answer(self, point, state, options):
                return contract.Answer(by="classifier", answerer="leak.test@t1", value=False, confidence=0.87)

        state = {"column": "final_refund_amount", "name_tokens": ["final", "refund", "amount"], "dtype": "float"}
        with mock.patch.object(shadow, "answerers", return_value=(Says(),)):
            got = decide.ask("feature.leakage_suspect", state, routing_doc={})
        self.assertEqual((got["answer"], got["decided_by"], got["rule"]["value"]), (True, "rule", True))
        self.assertEqual(got["disagree"], ["leak.test@t1"])
        self.assertIn("never hidden", got["note"])
        self.assertEqual(got["models"][0]["confidence"], 0.87)

    def test_a_model_routed_to_act_gives_the_answer(self):
        class Says:
            by, name, version, points = "classifier", "leak.test", "t1", frozenset({"feature.leakage_suspect"})
            value = True

            def answer(self, point, state, options):
                return contract.Answer(by="classifier", answerer="leak.test@t1", value=self.value, confidence=0.9)

        routed = {"decisions": {"feature.leakage_suspect": {"answerer": "leak.test@t1", "level": 3, "bar": 0.9}}}
        quiet = {"column": "zone", "name_tokens": ["zone"], "dtype": "string"}  # the rules do not flag it: a model may add the caution
        with mock.patch.object(shadow, "answerers", return_value=(Says(),)):
            got = decide.ask("feature.leakage_suspect", quiet, routing_doc=routed)
        self.assertEqual((got["rule"]["value"], got["answer"], got["decided_by"], got["level"]), (False, True, "classifier", 3))
        Says.value = False  # 26.5: freeing a column the rules flag is the validator's to refuse, whatever the level
        state = {"column": "final_refund_amount", "name_tokens": ["final", "refund", "amount"], "dtype": "float"}
        with mock.patch.object(shadow, "answerers", return_value=(Says(),)):
            got = decide.ask("feature.leakage_suspect", state, routing_doc=routed)
        self.assertEqual((got["answer"], got["decided_by"], got["level"]), (True, "rule", 3))
        self.assertTrue(got["refused_by_validator"])

    def test_a_persons_override_is_recorded_as_a_label(self):
        store = ledger.FileLedger(Path(self.home) / "ledger", "w")
        state = {"column": "customer_id", "name_tokens": ["customer", "id"], "dtype": "string", "uniqueness": "all"}
        with mock.patch.object(shadow, "answerers", return_value=()):
            same = decide.choose("column.is_identifier", state, "yes", "person:a", ledger=store)
            other = decide.choose("column.is_identifier", state, "no", "person:a", ledger=store)
        self.assertIsNone(same["label"])  # the person agreed with what DCLab said: no label
        self.assertIsNotNone(other["label"])
        labels = store.labels()
        self.assertEqual([(x.source, x.value) for x in labels], [("user_decision_exposed", False)])
        record = next(r for r in store.records() if r.id == other["recorded"])
        self.assertEqual((record.used, record.value_used), ("person", False))
        self.assertEqual(contract.effective_label(record, labels).value, False)  # it counts: it contradicts the rule's yes
        with self.assertRaises(decide.DecideError):
            decide.choose("column.semantic_role", state, "a role nobody knows", "person:a", ledger=store)

    def test_only_technical_points_and_states_of_bands(self):
        with self.assertRaises(decide.DecideError):
            decide.ask("task.framing", {})
        with self.assertRaises(decide.DecideError):  # a value is never a band
            decide.ask("column.is_identifier", {"column": "x", "rows": [1, 2, 3]})

    def test_the_choose_route_records_it(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(self.home)))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        state = {"column": "customer_id", "name_tokens": ["customer", "id"], "dtype": "string", "uniqueness": "all"}
        r = client.post("/api/decide/choose", json={"point": "column.is_identifier", "state": state, "value": "no"}, headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["recorded"] and r.json()["label"])
        self.assertEqual(client.post("/api/decide", json={"point": "nowhere", "state": state}, headers=h).status_code, 422)

    def test_the_project_page_asks_each_column(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.kinds import acceptance

        client = TestClient(create_app(Path(self.home)))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        _, pid = acceptance.build(kinds.get("binary").case(0), None, store=client.app.state.services.projects)
        got = client.get(f"/api/projects/{pid}/decisions", headers=h).json()["columns"]
        names = [c["column"] for c in got]
        self.assertNotIn("churned", names)  # the outcome is not asked about
        first = got[0]["decisions"]
        self.assertEqual(set(first), {"column.is_identifier", "column.time", "column.semantic_role", "feature.leakage_suspect"})
        self.assertTrue(all(d["evidence"] and d["rule"] for d in first.values()))
        recorded = [d for c in got for d in c["decisions"].values() if d["rule_source"] == "the project's recorded decision"]
        self.assertTrue(recorded)  # on a project the rule beside the models is the project's own, not the bands' approximation
        self.assertTrue(all(d["rule"]["answerer"] != decide.BANDS_RULE for d in recorded))

    def test_a_date_is_never_an_identifier_by_the_bands(self):
        state = {"column": "order_time", "name_tokens": ["order", "time"], "dtype": "string", "uniqueness": "over_99pct", "value_pattern": "date_like"}
        self.assertFalse(decide.rule("column.is_identifier", state).value)
        self.assertEqual(decide.rule("column.semantic_role", state).value, "datetime")

    def test_a_choice_against_any_answer_shown_is_a_label_and_a_later_one_supersedes_it(self):
        class Says:
            by, name, version, points = "classifier", "id.test", "t1", frozenset({"column.is_identifier"})

            def answer(self, point, state, options):
                return contract.Answer(by="classifier", answerer="id.test@t1", value=False, confidence=0.99)

        store = ledger.FileLedger(Path(self.home) / "ledger", "w")
        state = {"column": "customer_id", "name_tokens": ["customer", "id"], "dtype": "string", "uniqueness": "all"}
        with mock.patch.object(shadow, "answerers", return_value=(Says(),)):
            first = decide.choose("column.is_identifier", state, "yes", "person:u_alice", ledger=store)  # the rule's yes, the model's no
            again = decide.choose("column.is_identifier", state, "yes", "person:u_alice", ledger=store)
            measured = shadow.measure(store.records(), store.labels(), [Says()])
        self.assertIsNotNone(first["label"])  # it differs from an answer the person was shown
        current = contract.current_labels(store.labels())
        self.assertEqual(len(current), 1)  # one person, one column, one point: the later answer supersedes the earlier
        self.assertEqual(measured["id.test@t1"]["labelled"], 1)  # what the person was shown is measured like a shadow
        record = next(r for r in store.records() if r.id == again["recorded"])
        self.assertEqual(record.answer("person").answerer, "person")  # who: in the label only
        from dclab_rnd import consent

        with mock.patch.object(consent, "_allowed_for", return_value=True), mock.patch.object(consent, "_project_of", return_value=None):
            docs = consent.export_decisions(store.records(), store.labels(), workspace={"train": True}, projects={})
        self.assertNotIn("u_alice", json.dumps(docs))

    def test_a_choice_without_a_value_or_a_ledger_is_refused(self):
        state = {"column": "customer_id", "name_tokens": ["customer", "id"], "dtype": "string", "uniqueness": "all"}
        with mock.patch.object(shadow, "answerers", return_value=()):
            with self.assertRaises(decide.DecideError):
                decide.choose("column.is_identifier", state, None, "person:a", ledger=ledger.FileLedger(Path(self.home) / "l2", "w"))
            with mock.patch.object(ledger, "current_ledger", return_value=None):
                with self.assertRaises(decide.DecideError):
                    decide.choose("column.is_identifier", state, "yes", "person:a")

    def test_the_options_the_verdicts_and_a_broken_question(self):
        state = {"column": "customer_id", "name_tokens": ["customer", "id"], "dtype": "string", "uniqueness": "all"}
        with self.assertRaises(decide.DecideError):
            decide.ask("column.is_identifier", state, options=["maybe"])
        failed = next(n for n, v in shadow.gate_results().items() if v == "fail" and n.startswith("column.is_identifier"))

        class Judged:
            by, points = "classifier", frozenset({"column.is_identifier"})
            name, version = failed.split("@")

            def answer(self, point, state, options):
                return contract.Answer(by="classifier", answerer=failed, value=True, confidence=0.9)

        with mock.patch.object(shadow, "answerers", return_value=(Judged(),)):
            got = decide.ask("column.is_identifier", state, routing_doc={})
        self.assertEqual(got["models"][0]["sealed_gate"], "fail")  # a failed verdict is said as one, not as "not judged"
        broken = subprocess.run([sys.executable, "-m", "dclab_rnd.notebook_assist", "--decide", "{not json"], capture_output=True, text=True, cwd=ROOT,
                                env={**os.environ, "DCLAB_NO_LIVE_MODELS": "1"}, timeout=300)
        self.assertEqual(broken.returncode, 2)
        self.assertIn("error", json.loads(broken.stdout))


if __name__ == "__main__":
    unittest.main()

"""Package 30.21: ask the workspace (the owner's pattern 7).

A question about the workspace's tables is answered only with numbers a computation returned (an invented one refuses the answer); the
model sees no personal value and no row (the gateway's recorded prompts are read); a private project's table is never offered; with no
model nothing guesses; every exchange is kept whole under the workspace and written to the audit trail.
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

try:
    import pandas as pd
    from fastapi.testclient import TestClient  # noqa: F401
except ImportError as error:  # .venv-agent
    MISSING = str(error)
else:
    MISSING = ""


def call(name, arguments, n):
    return {"content": "", "tool_calls": [{"id": f"c{n}", "name": name, "arguments": arguments}], "usage": {"input_tokens": 0, "output_tokens": 0},
            "assistant_message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}", "type": "function",
                                                                                      "function": {"name": name, "arguments": json.dumps(arguments)}}]}}


def answer(text):
    return {"content": text, "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0}, "assistant_message": {"role": "assistant", "content": text}}


@unittest.skipIf(MISSING, MISSING)
class WorkspaceChatTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.models import settings
        from dclab_rnd.studio import data as sd

        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                                 "DCLAB_NO_LIVE_MODELS")} | {f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.TIERS for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")} \
            | {"OPENAI_API_KEY": "k"}
        env = mock.patch.dict(os.environ, clean)
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.s = self.client.app.state.services
        h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.pid = self.client.post("/api/projects", json={"name": "Orders", "goal": "late orders"}, headers=h).json()["id"]
        self.private = self.client.post("/api/projects", json={"name": "Secret", "goal": "g"}, headers=h).json()["id"]
        frame = pd.DataFrame({"city": (["Lyon"] * 6 + ["Lima"] * 4) * 4, "minutes": [30, 40, 35, 45, 30, 40, 50, 60, 55, 65] * 4,
                              "email": [f"Q7Kxmail{i}@example.org" for i in range(40)]})
        for pid in (self.pid, self.private):
            path = self.s.projects.data_dir(pid) / "orders.csv"
            frame.to_csv(path, index=False)
            sd.attach_data(self.s.projects, pid, "orders.csv")
        p = self.s.projects.get(self.private)
        p.setdefault("settings", {})["private"] = True
        self.s.projects.save(p)
        self.table = f"{self.pid}:orders"

    def gateway(self, replies):
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.models.usage import FileUsage

        self.sent = []
        queue = list(replies)

        class Scripted:
            def __init__(self, outer):
                self.outer = outer

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                self.outer.sent.append(json.dumps(messages, default=str))
                return queue.pop(0) if queue else answer("done")

        return Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Scripted(self))

    def test_an_answer_from_a_computation_is_given_and_kept_whole(self):
        from dclab_rnd import workspace_chat

        gw = self.gateway([call("list_tables", {}, 1), call("compute", {"table": self.table, "op": "mean", "column": "minutes", "group_by": ["city"]}, 2),
                           answer("Lyon orders take 36.7 minutes on average and Lima 57.5, over 40 rows.")])
        out = workspace_chat.ask(self.s, "How long do orders take by city?", person="owner", gateway=gw)
        self.assertEqual(out["status"], "answered", out.get("why"))
        self.assertEqual([s["tool"] for s in out["steps"]], ["list_tables", "compute"])
        self.assertNotIn(f"{self.private}:orders", out["tables"])  # a private project's table is never offered
        kept = list((Path(self.s.store.home) / "workspace_chat").glob("*.jsonl"))
        record = json.loads(kept[0].read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["question"], "How long do orders take by city?")
        self.assertEqual(record["computations"][0]["result"][0]["value"], 36.666667)
        prompts = " ".join(self.sent)
        self.assertNotIn("Q7Kxmail", prompts)  # no personal value reaches the model
        self.assertNotIn("30, 40, 35", prompts)  # nor a row

    def test_a_number_no_computation_returned_refuses_the_answer(self):
        from dclab_rnd import workspace_chat

        gw = self.gateway([call("compute", {"table": self.table, "op": "count"}, 1), answer("There are 40 orders, and about 42 of them were late.")])
        out = workspace_chat.ask(self.s, "How many orders were late?", gateway=gw)
        self.assertEqual(out["status"], "refused")
        self.assertIn(42.0, out["ungrounded"])
        self.assertIsNone(out["answer"])
        gw = self.gateway([answer("About 40 minutes.")])  # no computation at all
        self.assertEqual(workspace_chat.ask(self.s, "How long?", gateway=gw)["status"], "refused")

    def test_the_question_is_redacted_and_a_personal_column_is_never_computed_or_filtered(self):
        from dclab_rnd import workspace_chat

        gw = self.gateway([call("compute", {"table": self.table, "op": "max", "column": "minutes", "where": {"email": "Q7Kxmail3@example.org"}}, 1),
                           call("compute", {"table": self.table, "op": "distinct", "column": "email"}, 2), answer("done")])
        out = workspace_chat.ask(self.s, "How long did Q7Kxmail3@example.org wait? Call +44 7700 900123.", gateway=gw)
        first = self.sent[0]  # what the model was sent before it wrote anything (the scripted calls below are its own words)
        self.assertNotIn("Q7Kxmail3", first)  # the question's e-mail and phone never reach the model
        self.assertNotIn("7700 900123", first)
        self.assertEqual(out["question"], "How long did Q7Kxmail3@example.org wait? Call +44 7700 900123.")  # the record keeps the words
        for step in out["steps"]:
            self.assertIn("never computed on", json.dumps(step["result"]))

    def test_no_value_over_fewer_rows_than_the_minimum_and_a_numeric_filter_matches(self):
        from dclab_core import workspace_chat as wc

        frame = pd.DataFrame({"store": [7] * 6 + [8] * 2 + [9] * 12, "churn": [1.0, 0.0, None, 1.0, 1.0, 0.0] + [1.0] * 14, "minutes": range(20)})
        out = wc.compute(frame, wc.Computation(table="t", op="sum", column="minutes", group_by=["store"]))
        self.assertEqual(sorted(g["store"] for g in out["result"]), [7, 9])  # store 8 has two rows: left out
        self.assertEqual(out["groups_too_small"], 1)
        with self.assertRaises(ValueError):  # a filter leaving two rows
            wc.compute(frame, wc.Computation(table="t", op="max", column="minutes", where={"store": 8}))
        self.assertEqual(wc.compute(frame, wc.Computation(table="t", op="count", where={"churn": 1}))["result"], 17)  # 1 matches 1.0

    def test_the_grounding_reads_shares_as_percents_only_and_knows_the_filters_numbers(self):
        from dclab_core import workspace_chat as wc

        count = [{"computation": {"where": {"year": 2026, "store": 7}}, "result": 40, "rows_read": 40}]
        self.assertEqual(wc.ungrounded("In 2026, store 7 had 40 orders.", count), [])
        self.assertEqual(wc.ungrounded("There are 4100 orders.", count), [4100.0])  # 40 is a count, never a share
        self.assertEqual(wc.ungrounded("42.1% were late.", [{"result": 0.4213, "rows_read": 40}]), [])

    def test_every_status_is_in_the_audit_trail(self):
        from dclab_rnd import audit, workspace_chat

        workspace_chat.ask(self.s, "How many?", gateway=self.gateway([call("compute", {"table": self.table, "op": "count"}, 1), answer("40 orders.")]))
        workspace_chat.ask(self.s, "How long?", gateway=self.gateway([answer("About 40 minutes.")]))

        class Broken:
            def client(self, purpose):
                class C:
                    def complete(self, *a, **k):
                        raise RuntimeError("down")
                return C()

        workspace_chat.ask(self.s, "Anything?", gateway=Broken())
        _, rows = audit.for_store(self.s.projects).query(50, 0)
        self.assertEqual(sorted({r.get("status") for r in rows if r.get("kind") == "workspace_chat"}), ["answered", "failed", "refused"])

    def test_the_chat_is_a_run_of_the_workspace_chat_agent(self):
        from dclab_rnd import experience, workspace_chat

        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        gw = self.gateway([call("compute", {"table": self.table, "op": "count"}, 1), answer("There are 40 orders.")])
        self.assertEqual(workspace_chat.ask(self.s, "How many orders?", gateway=gw)["status"], "answered")
        events = log.read()
        steps = [e for e in events if e.kind == "agent.step"]
        self.assertTrue(steps, "no step recorded")
        self.assertIn("compute", {e.payload.get("tool") for e in steps})
        for c in (e for e in events if e.kind == "tool.call"):  # every tool call has its step, in the same run
            self.assertIn((c.ids.run, c.payload["tool"]), {(e.ids.run, e.payload.get("tool")) for e in steps})

    def test_a_personal_column_is_never_a_group_and_no_model_means_no_guess(self):
        from dclab_rnd import workspace_chat

        gw = self.gateway([call("compute", {"table": self.table, "op": "count", "group_by": ["email"]}, 1), answer("done")])
        out = workspace_chat.ask(self.s, "Count by email", gateway=gw)
        self.assertIn("a group", json.dumps(out["steps"][0]["result"]))
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            from dclab_rnd.models.gateway import Gateway
            from dclab_rnd.models.usage import FileUsage

            none = workspace_chat.ask(self.s, "Anything?", gateway=Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")))
        self.assertEqual(none["status"], "no_model")


if __name__ == "__main__":
    unittest.main()

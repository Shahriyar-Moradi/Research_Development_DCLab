"""The twin on Home, live (package 25.4): the owner's switch, Claude's turn beside the system's on the same draft, the
organised session folder, Claude's own conversation, the caps, the redacted sample, and what the system never sees."""

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

    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.models import settings as model_settings
    from dclab_rnd.models.gateway import Gateway
    from dclab_rnd.models.usage import FileUsage
    from dclab_rnd.twin import home as T
    MISSING = None
except ImportError as error:  # pragma: no cover
    MISSING = f"the studio needs pandas: {error}"

CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                         "DCLAB_INTERN_MODEL", "DCLAB_NO_LIVE_MODELS")} | {"OPENAI_API_KEY": "k"}

TAKE = {"problem": {"restatement": "Find customers who will cancel next month", "unit": "a customer", "moment": "at the end of each month",
                    "outcome": "the customer cancels within the next month", "target": "churned"},
        "problem_type": {"family": "classification", "task": "binary", "labels": "column", "why": "a yes or no outcome in a column"},
        "data": [{"name": "customers", "grain": "one customer", "identifiers": ["customer_id"], "times": [], "groups": [], "leak_suspects": []}],
        "eda": [], "preprocessing": [], "problem_workflow": {"size": "single", "parts": [{"title": "Churn", "family": "classification"}]},
        "algorithms": {"ranked": [{"family": "hist_gradient_boosting", "rank": 1}]}, "evaluation": {"split": "stratified", "metric": "roc_auc"},
        "solution_workflow": [{"id": "S1", "wf": "WF-01", "tool": "describe_data"}], "routes": [], "risks": [], "questions": [], "confidence": "medium",
        "changes": [], "reply": "I read this as a monthly churn problem. When do you decide whom to call?",
        "next_question": {"field": "prediction_moment", "text": "When is the prediction made?", "options": ["at month end", "daily"], "why": "it fixes the split"},
        "action": "call the customer", "pack": "churn", "ready": False, "blockers": ["the moment is not stated"]}


class Scripted:
    model, base_url = "claude-test", "https://api.anthropic.com"

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def complete(self, messages, tools=None, max_tokens=None, **options):
        self.sent.append({"messages": messages, "options": options})
        r = self.replies.pop(0) if self.replies else TAKE
        if isinstance(r, Exception):
            raise r
        return {"content": "```json\n" + json.dumps(r) + "\n```", "tool_calls": [], "usage": {"input_tokens": 1000, "output_tokens": 300},
                "thinking": [{"text": "The outcome is in the churned column.", "signature": "s"}]}


@unittest.skipIf(MISSING, MISSING)
class HomeTwinTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, CLEAN)
        self.env.start()
        self.home = Path(tempfile.mkdtemp())
        self.drafts = DraftStore(self.home / "drafts")
        self.transport = Scripted([])
        self.gateway = Gateway(FileUsage(self.home / "u.jsonl"), transport=lambda tier, purpose: self.transport)
        draft = self.drafts.create("Predict which customers cancel next month")
        self.draft_id = draft["id"]
        HomeAgent(self.drafts, None).start(self.draft_id)

    def tearDown(self):
        self.env.stop()

    def twin(self):
        return T.HomeTwin(self.drafts, self.gateway, self.home)

    def folder(self):
        return T.Sessions(self.home).find(f"draft-{self.draft_id}")

    def test_the_switch_is_off_by_default_and_then_nothing_is_written_or_asked(self):
        self.assertEqual(T.settings(self.home), {"on": False, "sample": False, "session_eur": 1.5, "monthly_eur": 30.0})
        self.assertIsNone(T.after_turn(self.drafts, self.gateway, self.home, self.draft_id, "reply", "hello"))
        self.assertIsNone(self.twin().turn(self.draft_id, "reply"))
        self.assertFalse((self.home / "twin").exists())
        self.assertEqual(self.transport.sent, [])
        with self.assertRaises(ValueError):
            T.set_settings(self.home, {"on": "yes"})
        with self.assertRaises(ValueError):
            T.set_settings(self.home, {"session_eur": 50})
        with self.assertRaises(ValueError):
            T.set_settings(self.home, {"colour": True})

    def test_a_turn_writes_the_session_folder_and_leaves_the_system_as_it_was(self):
        T.set_settings(self.home, {"on": True})
        before = self.drafts.get(self.draft_id)
        out = self.twin().turn(self.draft_id, "start")
        self.assertEqual(out["status"], "answered")
        after = self.drafts.get(self.draft_id)
        self.assertEqual({k: v for k, v in before.items() if k != "updated"}, {k: v for k, v in after.items() if k != "updated"})
        folder = self.folder()
        for name in ("session.json", "ai_first.jsonl", "system.jsonl", "steps.jsonl", "conversation.jsonl", "thinking.jsonl"):
            self.assertTrue((folder / name).is_file(), name)
        self.assertTrue((self.home / "twin" / "README.md").is_file())
        index = [json.loads(line) for line in (self.home / "twin" / "index.jsonl").read_text().splitlines()]
        self.assertEqual([r["draft_id"] for r in index], [self.draft_id])
        rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        steps = {r["step"] for r in rows}
        pack = next(r for r in rows if r["step"] == "home.pack")
        self.assertIsInstance(pack["system"]["value"].get("pack"), str)  # the pack's key, not the record the draft keeps
        self.assertTrue({"home.understanding", "home.next_question", "home.pack", "home.ready"} <= steps, steps)
        view = self.twin().view(self.draft_id)
        self.assertEqual(view["record"]["problem_type"]["family"], "classification")
        self.assertEqual(view["conversation"][-1]["role"], "claude")
        events = [json.loads(line) for line in (self.drafts.directory(self.draft_id) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["kind"], "twin")
        sent = self.transport.sent[0]["messages"]
        self.assertIn("twenty years", sent[0]["content"])
        self.assertIn("HOME TURN (start)", sent[1]["content"])

    def test_every_turn_and_pick_says_whose_it_is(self):  # testers: each person's sessions are theirs in the folder
        from dclab_rnd.accounts.principal import Principal, acting

        T.set_settings(self.home, {"on": True})
        with acting(Principal(role="data_scientist", workspace_id="w1", user_id="u_friend1", name="Friend", via="session")):
            self.twin().turn(self.draft_id, "start")
            self.twin().pick(self.draft_id, "home.pack", "ai_first")
        self.assertEqual(T.Sessions.rows(self.folder(), "ai_first.jsonl")[-1]["user"], "u_friend1")
        self.assertEqual(T.Sessions.rows(self.folder(), "picks.jsonl")[-1]["user"], "u_friend1")
        self.assertEqual(T.Sessions(self.home).manifest(self.folder())["people"], ["u_friend1"])

    def test_a_message_to_claude_alone_never_reaches_the_system(self):
        T.set_settings(self.home, {"on": True})
        n = len(self.drafts.get(self.draft_id)["messages"])
        self.twin().turn(self.draft_id, "message", "Only for Claude: what would you check first?", to="claude")
        self.assertEqual(len(self.drafts.get(self.draft_id)["messages"]), n)
        rows = T.Sessions.rows(self.folder(), "conversation.jsonl")
        self.assertEqual([r["role"] for r in rows], ["user", "claude"])
        self.assertIn("Only for Claude", self.transport.sent[0]["messages"][1]["content"])

    def test_a_failing_claude_is_kept_and_never_raises(self):
        T.set_settings(self.home, {"on": True})
        self.transport.replies = [RuntimeError("ServerError: overloaded")]
        out = self.twin().turn(self.draft_id, "reply", "monthly")
        self.assertEqual(out["status"], "failed")
        self.assertEqual(T.Sessions.rows(self.folder(), "ai_first.jsonl")[-1]["status"], "failed")

    def test_the_session_cap_stops_claude(self):
        T.set_settings(self.home, {"on": True, "session_eur": 0.5})
        self.twin().turn(self.draft_id, "start")
        T.Sessions(self.home).update(self.folder(), spent_eur=0.6)
        out = self.twin().turn(self.draft_id, "reply", "monthly")
        self.assertEqual(out["status"], "capped")
        self.assertEqual(len(self.transport.sent), 1)

    def test_the_sample_is_redacted_its_personal_columns_left_out_and_never_stored(self):
        frame = pd.DataFrame({"customer_email": [f"person{i}@example.com" for i in range(30)], "plan": ["Q7Kxgold", "Q7Kxbasic"] * 15,
                              "note": [f"call me at +44 20 7946 09{i:02d}" for i in range(30)], "churned": [0, 1] * 15})
        draft = self.drafts.get(self.draft_id)
        frame.to_parquet(self.drafts.data_dir(self.draft_id) / "clean.parquet")
        self.drafts.update(self.draft_id, lambda d: d.update(assets=[{"id": "a1", "clean_file": "clean.parquet", "status": "ready"}], active_asset="a1"))
        rows, left_out = T.redacted_sample(self.drafts, self.drafts.get(self.draft_id))
        self.assertEqual(left_out, ["customer_email"])
        self.assertEqual(len(rows), 20)
        self.assertFalse(any("7946" in str(r.get("note")) for r in rows), rows[:2])  # the phone number redacted
        T.set_settings(self.home, {"on": True, "sample": True})
        self.twin().turn(self.draft_id, "reply", "monthly")
        self.assertIn("Q7Kx", self.transport.sent[0]["messages"][1]["content"])  # Claude was shown the rows
        stored = "".join(p.read_text() for p in self.folder().iterdir())
        self.assertNotIn("Q7Kx", stored)  # the folder keeps their count, never the rows
        self.assertEqual(T.Sessions.rows(self.folder(), "ai_first.jsonl")[-1]["given"]["sample_rows"], 20)
        self.assertIsNotNone(draft)

    def test_a_person_raises_the_drafts_budget_within_their_month_and_it_is_kept(self):  # 25.10
        T.set_settings(self.home, {"on": True, "session_eur": 1.5})
        capped = self.twin().raise_budget(self.draft_id, 10, who="u1", room=2.0)  # two euros left of the person's month
        self.assertEqual((capped["before"], capped["session_eur"]), (1.5, 2.0))  # never past what was spent plus the room left
        out = self.twin().raise_budget(self.draft_id, 1.5, who="u1")
        self.assertEqual((out["before"], out["session_eur"]), (2.0, 3.5))
        self.assertEqual(self.twin().cap(self.folder(), T.settings(self.home)), 3.5)
        with self.assertRaises(ValueError):
            self.twin().raise_budget(self.draft_id, 1.5, room=0.0)  # the month is used up: no raise
        with self.assertRaises(ValueError):
            self.twin().raise_budget(self.draft_id, 100)
        rows = T.Sessions.rows(self.folder(), "budget.jsonl")
        self.assertEqual([(r["by"], r["after"]) for r in rows], [("u1", 2.0), ("u1", 3.5)])

    def test_a_raised_draft_follows_a_higher_default_and_two_raises_at_once_are_two(self):  # 25.10 review
        import threading

        T.set_settings(self.home, {"on": True, "session_eur": 1.5})
        self.twin().raise_budget(self.draft_id, 1.5)
        T.set_settings(self.home, {"session_eur": 5.0})
        self.assertEqual(self.twin().cap(self.folder(), T.settings(self.home)), 5.0)
        threads = [threading.Thread(target=lambda: self.twin().raise_budget(self.draft_id, 1.0)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.twin().cap(self.folder(), T.settings(self.home)), 7.0)
        self.assertEqual([r["after"] for r in T.Sessions.rows(self.folder(), "budget.jsonl")][-2:], [6.0, 7.0])

    def test_a_deleted_draft_takes_its_session_with_it(self):
        T.set_settings(self.home, {"on": True})
        self.twin().turn(self.draft_id, "start")
        self.assertEqual(T.Sessions(self.home).forget(self.draft_id), 1)
        self.assertIsNone(self.folder())


@unittest.skipIf(MISSING, MISSING)
class UnderstandingEditTest(unittest.TestCase):
    """25.5: the person corrects a field on the page; their words are a message and the field is recorded through the guard."""

    def test_an_edit_is_the_persons_words_recorded_through_the_guard(self):
        drafts = DraftStore(Path(tempfile.mkdtemp()))
        d = drafts.create("Predict which customers cancel next month")
        agent = HomeAgent(drafts, None)
        agent.start(d["id"])
        out = agent.edit(d["id"], "prediction_moment", "On the first day of each month, before the retention calls")
        self.assertNotIn("error", out)
        draft = drafts.get(d["id"])
        self.assertEqual(draft["understanding"]["prediction_moment"], "On the first day of each month, before the retention calls")
        self.assertEqual(draft["plan"]["prediction_moment"]["status"], "stated")
        edits = [m for m in draft["messages"] if m.get("kind") == "edit"]
        self.assertEqual([(m["role"], m["field"]) for m in edits], [("user", "prediction_moment")])
        with self.assertRaises(ValueError):
            agent.edit(d["id"], "favourite_colour", "blue")
        with self.assertRaises(ValueError):
            agent.edit(d["id"], "action", "   ")


@unittest.skipIf(MISSING, MISSING)
class RoutesTest(unittest.TestCase):
    def test_the_switch_is_the_owners_and_a_message_to_claude_needs_it_on(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        home = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            client = TestClient(create_app(home))
            h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
            self.assertEqual(client.get("/api/twin/settings").json()["settings"]["on"], False)
            d = client.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=h).json()
            self.assertEqual(client.post(f"/api/drafts/{d['id']}/twin/messages", json={"text": "hi"}, headers=h).status_code, 409)
            self.assertEqual(client.put("/api/twin/settings", json={"on": "yes"}, headers=h).status_code, 422)
            self.assertTrue(client.put("/api/twin/settings", json={"on": True}, headers=h).json()["settings"]["on"])
            services = client.app.state.services
            moves = [r for r in services.audit.query(500, 0)[1] if r.get("move") == "set_twin"]
            self.assertEqual(len(moves), 1)
            raised = client.post(f"/api/drafts/{d['id']}/twin/budget", json={"add_eur": 1.5}, headers=h).json()
            self.assertEqual(raised["session_eur"], 3.0)
            self.assertEqual(len([r for r in services.audit.query(500, 0)[1] if r.get("move") == "raise_twin_budget"]), 1)
            self.assertEqual(client.get("/api/budget").json()["monthly_eur"], 15)
            self.assertEqual(client.put("/api/workspace/budgets/u1", json={"monthly_eur": 40}, headers=h).json()["users"], {"u1": 40})
            self.assertEqual(client.put("/api/workspace/budgets/u1", json={"monthly": 40}, headers=h).status_code, 422)  # a typo clears nothing
            from dclab_rnd import limits as lim
            self.assertEqual(lim.user_monthly("u1", services.store.home), 40)
            self.assertEqual(services.gateway.home, services.store.home)  # the gateway reads the people's own limits of its workspace
            view = client.get(f"/api/drafts/{d['id']}/twin").json()
            self.assertEqual(view["settings"]["on"], True)
            self.assertEqual(client.get("/api/twin/sessions").status_code, 200)
        from dclab_rnd.accounts import guard

        self.assertEqual(guard.permission("PUT", "/api/twin/settings"), "admin")
        self.assertEqual(guard.permission("PUT", "/api/workspace/budgets/{user_id}"), "admin")


if __name__ == "__main__":
    unittest.main()

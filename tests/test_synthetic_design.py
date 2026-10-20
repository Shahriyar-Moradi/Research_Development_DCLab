"""Synthetic data, the second way (the owner, 2026-10-20): the person describes the table in the chat, sees its design, and
approves it before it is created. Nothing is made unasked; the table is made from the approved design exactly."""

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd.draft import synthetic_design
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    raise unittest.SkipTest(str(error))


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))
        self.draft = self.store.create("Predict which subscribers cancel next month")["id"]
        self.agent = HomeAgent(self.store, None)

    def test_a_short_yes_approves_and_a_change_does_not(self):
        for yes in ("approve", "Yes!", "looks good", "go ahead", "create it"):
            self.assertTrue(synthetic_design.approves(yes), yes)
        for change in ("make the churn rate 15%", "add a column for region", "", "yes but with 50,000 rows and a column for the plan and the region"):
            self.assertFalse(synthetic_design.approves(change), change)

    def test_the_design_is_proposed_in_the_conversation_and_nothing_is_created(self):
        proposal = synthetic_design.propose(self.store, self.agent, None, self.draft, "Subscribers with tenure and plan; about 20% cancel", 2000)
        draft = self.store.get(self.draft)
        self.assertEqual(draft["synthetic_proposal"]["id"], proposal["id"])
        self.assertEqual(draft["assets"], [])  # nothing was made
        said = [m for m in draft["messages"] if m["role"] == "user"][-1]
        self.assertIn("20% cancel", said["text"])  # the person's description is in the conversation
        card = draft["messages"][-1]
        self.assertEqual((card["kind"], card["actions"]), ("synthetic_spec", ["approve_synthetic", "change_synthetic"]))
        self.assertEqual(card["proposal"]["id"], proposal["id"])
        self.assertTrue(card["proposal"]["columns"] and card["proposal"]["rows"] == 2000)
        self.assertNotIn("weights", str(card["proposal"]["columns"]))  # names, kinds and categories: a design, not data

    def test_a_change_adds_to_what_was_said_and_only_the_current_design_can_be_approved(self):
        first = synthetic_design.propose(self.store, self.agent, None, self.draft, "Subscribers with tenure and plan", 2000)
        second = synthetic_design.propose(self.store, self.agent, None, self.draft, "make it 3000 rows", 3000)
        kept = self.store.get(self.draft)["synthetic_proposal"]
        self.assertEqual(kept["said"], ["Subscribers with tenure and plan", "make it 3000 rows"])
        spec, info = synthetic_design.stored(self.store, self.draft, second["id"])
        self.assertEqual((spec.rows, info["approved"]), (3000, True))
        with self.assertRaises(KeyError):
            synthetic_design.stored(self.store, self.draft, first["id"])  # an older design is not the one on the draft
        synthetic_design.close(self.store, self.draft)
        self.assertNotIn("synthetic_proposal", self.store.get(self.draft))

    def test_an_empty_description_asks_for_one(self):
        with self.assertRaises(ValueError):
            synthetic_design.propose(self.store, self.agent, None, self.draft, "   ", 2000)

    def test_asking_for_simulated_data_asks_how_and_starts_nothing(self):
        asked = []
        agent = HomeAgent(self.store, None, on_request=lambda d, what, args: asked.append(what))
        agent.offer_synthetic(self.draft, "Subscribers who cancel", 500_000)
        agent.offer_synthetic(self.draft, "again")  # not twice in a row
        offers = [m for m in self.store.get(self.draft)["messages"] if m.get("kind") == "synthetic_way"]
        self.assertEqual((len(offers), asked), (1, []))
        self.assertEqual(offers[0]["offer"]["rows"], 10_000)  # the AI way's cap
        self.assertEqual(offers[0]["actions"], ["simulate_ai", "simulate_describe"])


class ApprovedTableTests(unittest.TestCase):
    """Through the API: the approved design makes the table, as shown; a design that is not the current one is refused."""

    def test_the_approved_design_makes_the_table_it_showed(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        home = Path(tempfile.mkdtemp())
        client = TestClient(create_app(home))
        headers = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        draft = client.post("/api/drafts", json={"problem": "Predict which subscribers cancel next month"}, headers=headers).json()["id"]
        got = client.post(f"/api/drafts/{draft}/data/synthetic/design", json={"description": "Subscribers with tenure and plan; 20% cancel", "rows": 1000}, headers=headers)
        self.assertEqual(got.status_code, 200, got.text)
        proposal, shown = got.json()["proposal"], got.json()["summary"]
        self.assertEqual(client.get(f"/api/drafts/{draft}").json()["assets"], [])  # designed, not created
        refused = client.post(f"/api/drafts/{draft}/data/synthetic", json={"proposal": "spnotthere"}, headers=headers)
        self.assertEqual(refused.status_code, 409)
        accepted = client.post(f"/api/drafts/{draft}/data/synthetic", json={"proposal": proposal}, headers=headers)
        self.assertEqual(accepted.status_code, 202, accepted.text)
        for _ in range(240):  # the job makes the table in the background
            assets = client.get(f"/api/drafts/{draft}").json()["assets"]
            if assets and assets[0].get("status") in ("ready", "failed"):
                break
            time.sleep(0.5)
        self.assertEqual(assets[0]["status"], "ready", assets)
        self.assertTrue(assets[0].get("synthetic"))
        self.assertEqual(assets[0]["name"], f"Synthetic · {shown['name']}")  # the table the design named
        self.assertNotIn("synthetic_proposal", client.get(f"/api/drafts/{draft}").json())  # used: it leaves the draft
        cancelled = client.post(f"/api/drafts/{draft}/data/synthetic/design", json={"cancel": True}, headers=headers)
        self.assertEqual(cancelled.json(), {"cancelled": True})


if __name__ == "__main__":
    unittest.main()

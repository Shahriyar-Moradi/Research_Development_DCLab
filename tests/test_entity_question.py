"""A repeating key is asked with its evidence (the owner, 2026-10-26: "ask, with evidence", not "group by default"):
biker_id repeats, no entity was said, so Home says how often it repeats and why grouping keeps the score honest; yes
groups by it, no says there is none, "I don't know" groups nothing and the summary says the risk. A row's own key is
never asked."""

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EntityQuestionTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def draft(self):
        import numpy as np
        import pandas as pd

        from dclab_rnd.draft import chat
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.studio.data import profile_table

        rng = np.random.default_rng(0)
        table = pd.DataFrame({"order_id": [f"O{i}" for i in range(600)], "biker_id": np.repeat(np.arange(40), 15),
                              "fare": rng.random(600), "is_fraud": rng.integers(0, 2, 600)})
        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = chat.HomeAgent(store, None)
        d = store.create("Flag the orders that are fraud")
        table.to_parquet(Path(store.data_dir(d["id"])) / "clean.parquet")

        def ready(x):
            x["assets"], x["active_asset"] = [{"id": "a1", "status": "ready", "clean_file": "clean.parquet"}], "a1"
            x["analysis"] = {"profile": profile_table(table), "columns": [{"name": c} for c in table.columns]}
            x.setdefault("understanding", {}).update(target="is_fraud", prediction_moment="When the order is placed, before dispatch",
                                                     action="We review the flagged orders", family="classification")
        store.update(d["id"], ready)
        return store, agent, d["id"]

    def test_a_repeating_key_is_asked_with_its_evidence_and_the_answer_groups_or_not(self):
        for said, entity in (("Yes, keep each one on one side", "biker_id"), ("No, its rows are independent", "No single entity"),
                             ("I don't know", None)):
            store, agent, did = self.draft()
            agent.next_turn(did)
            q = agent.pending(store.get(did))
            self.assertEqual((q["field"], q["column"]), ("entity_check", "biker_id"), said)  # never order_id: one row each
            self.assertIn("600 rows for 40 different values", q["text"])
            agent.take_answer(did, q, said)
            d = store.get(did)
            self.assertEqual(d["understanding"].get("entity"), entity, said)
            self.assertEqual(sum(x["field"] == "entity_check" for x in d["questions"]), 1)  # asked once
            if entity is None:
                summary = [m["text"] for m in d["messages"] if m.get("kind") == "summary"][-1]
                self.assertIn("Not grouped by `biker_id`", summary)  # the risk said, never grouped silently

    def test_no_answer_declines_it_and_a_walk_goes_on(self):  # harness HRN-048: four walks stopped waiting on it
        store, agent, did = self.draft()
        agent.next_turn(did)
        q = agent.pending(store.get(did))
        self.assertEqual(q.get("source"), "advisory")
        agent.decline(did, q)
        d = store.get(did)
        self.assertIsNone(agent.pending(d))
        self.assertTrue(d["questions"][-1]["declined"])
        self.assertIn("Not grouped by `biker_id`", [m["text"] for m in d["messages"] if m.get("kind") == "summary"][-1])

    def test_a_kind_with_its_own_structure_is_never_asked(self):
        store, agent, did = self.draft()
        store.update(did, lambda x: x["understanding"].update(family="ranking"))
        self.assertIsNone(agent.entity_question(store.get(did)))  # a ranking's items are its field, not an entity

    def test_an_entity_already_said_is_never_asked_again(self):
        store, agent, did = self.draft()
        store.update(did, lambda x: x["understanding"].update(entity="the biker"))
        self.assertIsNone(agent.entity_question(store.get(did)))


if __name__ == "__main__":
    unittest.main()

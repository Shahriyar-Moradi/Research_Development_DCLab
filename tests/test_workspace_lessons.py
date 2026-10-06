"""Workspace lessons (package A5.3): proposed after the final stage, reviewed by a person, searched as evidence."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import lessons  # noqa: E402
from dclab_rnd.expansion.runner import fast_profile  # noqa: E402
from dclab_rnd.intern.tools import Toolbox  # noqa: E402
from dclab_rnd.storage import open_stores  # noqa: E402
from dclab_rnd.studio import data as studio_data, engine, graph  # noqa: E402

INDEX = ROOT / "evidence" / "knowledge" / "rag" / "records.jsonl"


def table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"age": rng.integers(18, 80, n), "plan": rng.choice(["Q7Kxbasic", "Q7Kxpro"], n), "spend": np.round(rng.gamma(2, 30, n), 2)})
    frame["churned"] = np.where(rng.random(n) < 0.2 + 0.004 * (frame["age"] - 18), "Q7Kxyes", "Q7Kxno")
    return frame


def reply(text):
    return {"content": text, "tool_calls": [], "usage": {}, "assistant_message": {"role": "assistant", "content": text}}


class Transport:
    """A scripted model per purpose: the lesson proposal and the evidence answer."""

    def __init__(self, purpose, lesson_text=None, answer=None):
        from dclab_rnd.models.settings import PURPOSES

        name = next((k for k, v in PURPOSES.items() if v is purpose), purpose)  # the gateway passes the Purpose itself
        self.purpose, self.lesson_text, self.answer = name, lesson_text, answer

    def complete(self, messages, tools=None, max_tokens=None, **_):
        if self.purpose == "lesson_proposal" and self.lesson_text is not None:
            return reply(self.lesson_text(messages[-1]["content"]))
        if self.purpose == "evidence_answer" and self.answer is not None:
            return reply(self.answer(messages[-1]["content"]))
        raise RuntimeError("not scripted")


class LessonTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        self.home = Path(tempfile.mkdtemp())
        self.store = open_stores(self.home)[0]  # the server's store: files, or PostgreSQL under test-pg
        self.app = create_app(self.home)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.addCleanup(lessons.install, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.reviewer = {**self.h, "X-DCLab-Role": "reviewer"}

    def finished(self, synthetic=False):
        pid = self.store.create("Churn", "general", "Predict churn")["id"]
        table().to_parquet(self.store.data_dir(pid) / "t.parquet", index=False)
        studio_data.attach_data(self.store, pid, "t.parquet")
        if synthetic:
            self.store.update(pid, data={**self.store.get(pid)["data"], "synthetic": True})
        Toolbox(self.store).call("set_solution", {"project_id": pid, "target": "churned", "task": "binary", "positive_label": "Q7Kxyes",
                                                   "prediction_moment": "At the monthly snapshot, before the outcome is known."})
        graph.approve_gate(self.store, pid, "solution", "owner", "The moment is right")
        with fast_profile():
            for stage in engine.STAGE_KEYS:
                if stage == "final":
                    graph.approve_gate(self.store, pid, "holdout", "owner", "Score it once")
                engine.execute(self.store, pid, stage, "human")
        return pid

    def approve_final(self, pid):
        r = self.client.post(f"/api/projects/{pid}/stages/final/approve", json={}, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        return self.client.get(f"/api/lessons?project_id={pid}").json()["lessons"]

    def test_the_final_approval_proposes_lessons_that_cite_the_stage_and_hold_for_their_scope(self):
        pid = self.finished()
        self.assertEqual(self.client.post(f"/api/projects/{pid}/lessons", headers=self.h).status_code, 409)  # not before the final approval
        proposed = self.approve_final(pid)
        self.assertTrue(1 <= len(proposed) <= 3)
        claims = {c["claim_id"] for c in self.store.read_stage(pid, "final")["claims"]}
        for lesson in proposed:
            self.assertEqual(lesson["status"], "proposed")
            self.assertTrue(lesson["cites"] and set(lesson["cites"]) <= claims)
            self.assertTrue(lesson["against"] and lesson["next_test"])
            self.assertEqual((lesson["scope"]["project"], lesson["scope"]["dataset"], lesson["scope"]["task"]), ("Churn", "t.parquet", "binary_imbalanced"))
            self.assertTrue(lesson["scope"]["rows"])
            self.assertEqual(lesson["written_by"], "code")  # no model configured
        self.assertNotIn("Q7Kx", json.dumps(proposed))  # no cell value
        again = self.client.post(f"/api/projects/{pid}/lessons", headers=self.h).json()["lessons"]
        self.assertEqual([l["id"] for l in again], [l["id"] for l in proposed])  # proposed once per final record

    def test_only_a_reviewer_reviews_and_an_edit_may_not_overclaim(self):
        pid = self.finished()
        first, second, *_ = self.approve_final(pid)
        url = f"/api/lessons/{first['id']}/review"
        self.assertEqual(self.client.post(url, json={"action": "accept"}, headers=self.h).status_code, 403)
        self.assertEqual(self.client.post(url, json={"action": "accept"}, headers={**self.h, "X-DCLab-Role": "viewer"}).status_code, 403)
        self.assertEqual(self.client.post("/api/lessons/nope/review", json={"action": "accept"}, headers=self.reviewer).status_code, 404)
        self.assertEqual(self.client.post(url, json={"action": "maybe"}, headers=self.reviewer).status_code, 422)
        bad = self.client.post(url, json={"action": "edit", "claim": "This model is production-ready for churn."}, headers=self.reviewer)
        self.assertEqual(bad.status_code, 422)
        edited = self.client.post(url, json={"action": "edit", "against": "One holdout of 80 rows is a small draw.", "reason": "sharper"}, headers=self.reviewer).json()
        self.assertEqual((edited["status"], edited["review"]["by"], edited["review"]["changed"]["against"]["to"]),
                         ("accepted", "reviewer", "One holdout of 80 rows is a small draw."))
        self.assertEqual(edited["cites"], first["cites"])  # the citations and the scope stay what code wrote
        rejected = self.client.post(f"/api/lessons/{second['id']}/review", json={"action": "reject", "reason": "obvious"}, headers=self.reviewer).json()
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("lesson_reviewed", json.dumps(self.store.activity(pid)))

    def test_an_accepted_lesson_joins_the_evidence_and_an_agent_cites_it(self):
        before = hashlib.sha256(INDEX.read_bytes()).hexdigest()
        pid = self.finished()
        first, second, *_ = self.approve_final(pid)
        self.client.post(f"/api/lessons/{first['id']}/review", json={"action": "accept"}, headers=self.reviewer)
        self.client.post(f"/api/lessons/{second['id']}/review", json={"action": "reject"}, headers=self.reviewer)
        rid = f"LESSON-{first['id']}"
        library = self.client.get("/api/evidence").json()
        found = {r["id"]: r for r in library["records"]}
        self.assertIn(rid, found)
        self.assertEqual(found[rid]["type"], "workspace_lesson")
        self.assertIn("Scope: project Churn; dataset t.parquet", found[rid]["text"])
        self.assertNotIn(f"LESSON-{second['id']}", found)  # a rejected lesson is not evidence
        self.assertEqual(hashlib.sha256(INDEX.read_bytes()).hexdigest(), before)  # evidence/knowledge is never written

        hits = Toolbox(self.store).call("search_evidence", {"query": first["claim"], "k": 3})["results"]
        self.assertEqual(hits[0]["record_id"], rid)  # the intern's search finds it, labelled by its type
        self.assertEqual(hits[0]["type"], "workspace_lesson")

        keyed = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}})
        keyed.start()
        self.addCleanup(keyed.stop)
        self.app.state.models.transport = lambda tier, purpose: Transport(purpose, answer=lambda prompt: f"This workspace measured it in one project [{rid}].")
        answer = self.client.post("/api/evidence/ask", json={"question": first["claim"]}, headers=self.h).json()
        self.assertIn(rid, json.dumps(answer))  # an agent cites the lesson, and the check keeps the sentence
        self.assertIn("This workspace measured it in one project", json.dumps(answer))

    def test_a_synthetic_lesson_is_labelled_and_never_offered_as_evidence(self):
        pid = self.finished(synthetic=True)
        first, *_ = self.approve_final(pid)
        self.assertTrue(first["synthetic"])
        accepted = self.client.post(f"/api/lessons/{first['id']}/review", json={"action": "accept"}, headers=self.reviewer).json()
        self.assertEqual(accepted["status"], "accepted")
        self.assertNotIn(f"LESSON-{first['id']}", {r["id"] for r in self.client.get("/api/evidence").json()["records"]})
        hits = Toolbox(self.store).call("search_evidence", {"query": first["claim"], "k": 5})["results"]
        self.assertNotIn(f"LESSON-{first['id']}", {h["record_id"] for h in hits})

    def test_a_models_lessons_are_kept_only_when_the_check_passes(self):
        pid = self.finished()
        record = self.store.read_stage(pid, "final")
        c1 = next(c["claim_id"] for c in record["claims"] if c["claim_id"].endswith("-C1"))
        c2 = next(c["claim_id"] for c in record["claims"] if c["claim_id"].endswith("-C2"))
        items = [{"claim": f"On this table the tuned model kept its training-CV level on the holdout [{c1}].", "against": f"A single holdout is one draw [{c1}].",
                  "next_test": "Score it on next month's snapshot."},
                 {"claim": f"Tuning raised the score by 0.4321 [{c2}].", "against": "x", "next_test": "y"},  # a number not in the claim
                 {"claim": f"This model is production-ready [{c1}].", "against": "x", "next_test": "y"}]
        keyed = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}})
        keyed.start()
        self.addCleanup(keyed.stop)
        self.app.state.models.transport = lambda tier, purpose: Transport(purpose, lesson_text=lambda prompt: json.dumps(items))
        proposed = self.approve_final(pid)
        self.assertEqual(len(proposed), 1)
        self.assertEqual((proposed[0]["written_by"], proposed[0]["cites"]), ("model", [c1]))
        self.assertEqual(proposed[0]["next_test"], "Score it on next month's snapshot.")

    def test_a_withdrawn_lesson_leaves_the_search_and_a_rerun_supersedes_unreviewed_ones(self):
        pid = self.finished()
        first, second, *_ = self.approve_final(pid)
        rid = f"LESSON-{first['id']}"
        self.client.post(f"/api/lessons/{first['id']}/review", json={"action": "accept"}, headers=self.reviewer)
        search = lambda: {h["record_id"] for h in Toolbox(self.store).call("search_evidence", {"query": first["claim"], "k": 5})["results"]}
        self.assertIn(rid, search())
        record = lessons.as_record(lessons.installed().get(first["id"]))
        self.assertIn("(run 1)", record["metadata"]["final_run"])  # which run it came from, in the metadata
        from dclab_rnd import cited
        allowed = set().union(*(cited.numbers(str(first[k]), source=True) for k in ("claim", "against", "next_test")),
                              cited.numbers(lessons.scope_line(first["scope"]), source=True))
        self.assertEqual(cited.numbers(record["text"], source=True) - allowed, set())  # no date or run number a model could borrow
        self.client.post(f"/api/lessons/{first['id']}/review", json={"action": "reject", "reason": "withdrawn"}, headers=self.reviewer)
        self.assertNotIn(rid, search())  # withdrawn: no longer evidence

        graph.approve_gate(self.store, pid, "holdout", "owner", "Score it again")
        with fast_profile():
            engine.execute(self.store, pid, "final", "human", reuse_reason="a second look")
        newer = self.approve_final(pid)
        statuses = {l["id"]: l["status"] for l in newer}
        self.assertEqual(statuses[second["id"]], "superseded")  # the earlier run's unreviewed lesson
        self.assertEqual(statuses[first["id"]], "rejected")  # a reviewed one keeps its review
        self.assertEqual(self.client.post(f"/api/lessons/{second['id']}/review", json={"action": "accept"}, headers=self.reviewer).status_code, 422)
        self.assertTrue([l for l in newer if l["status"] == "proposed" and l["basis"] != first["basis"]])

    def test_a_lesson_of_a_deleted_project_can_still_be_reviewed(self):
        pid = self.finished()
        first, *_ = self.approve_final(pid)
        self.assertIn(self.client.delete(f"/api/projects/{pid}", headers=self.h).status_code, (200, 204))
        r = self.client.post(f"/api/lessons/{first['id']}/review", json={"action": "accept"}, headers=self.reviewer)
        self.assertEqual(r.status_code, 200, r.text)

    def test_an_edit_in_the_same_second_is_searched_at_once(self):
        from dclab_rnd.evidence_index import EvidenceIndex
        store = lessons.FileLessons(self.home / "unit" / "lessons.json")
        base = EvidenceIndex([{"record_id": "DCLAB-R01", "type": "rule", "title": "Rule", "text": "A rule.", "metadata": {}, "citations": []}])
        lesson = {"id": "abc", "project_id": "p", "project_name": "P", "basis": "PRJ-p-final:run1:2026-10-06T10:00:00+00:00", "status": "proposed",
                  "claim": "Old claim text", "against": "One split.", "next_test": "Again.", "cites": ["PRJ-p-final-C1"], "scope": {}, "synthetic": False,
                  "written_by": "code", "review": None, "history": []}
        store.save(lesson)
        lessons.review(store, "abc", "accept", "reviewer")
        self.assertIn("Old claim text", lessons.merged(base, store).get("LESSON-abc")["text"])
        lessons.review(store, "abc", "edit", "reviewer", edits={"claim": "New claim text"})  # within the same second
        self.assertIn("New claim text", lessons.merged(base, store).get("LESSON-abc")["text"])

    def test_lessons_need_a_finished_project(self):
        self.assertEqual(lessons.propose({"id": "p"}, {"stage": "models", "claims": [{"claim_id": "x"}]}), [])
        self.assertEqual(self.client.post("/api/projects/nope/lessons", headers=self.h).status_code, 404)


if __name__ == "__main__":
    unittest.main()

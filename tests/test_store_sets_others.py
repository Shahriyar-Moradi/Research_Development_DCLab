"""Package 32.13: the step-by-step, judge and retrieval sets.

A project's graph moves are one M6 episode, its steps in order with their outcomes and its outcome from the record; a job's
states another; a person's correction and a fix are M7 examples with who made the candidate and who judged it; a search
through a tool is an M8 example with its query and the passages it offered.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class OtherSetsTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store, shapers

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        s = Store(self.layout)
        moves = [{"at": f"2026-10-20T10:00:0{i}+00:00", "move": m, "from": a, "to": b, "status": st, "project": "p1"}
                 for i, (m, a, b, st) in enumerate([("set_solution", "WF-01", "WF-02", "allowed"), ("read_holdout", "WF-02", "WF-02", "blocked"),
                                                    ("approve_gate", "WF-02", "WF-03", "allowed")])]
        s.keep_log("workflows", "dev", "projects/p1/transitions.jsonl", "".join(json.dumps(m) + "\n" for m in moves).encode(), consent="train")
        jobs = [{"at": f"2026-10-20T10:01:0{i}+00:00", "job": "j1", "kind": "stage", "state": {"status": st}} for i, st in enumerate(["queued", "running", "done"])]
        s.keep_log("tasks", "dev", "jobs_history.jsonl", "".join(json.dumps(j) + "\n" for j in jobs).encode(), consent="train")
        correction = {"at": "2026-10-20T10:02:00+00:00", "what": "card", "before": {"unit": "order"}, "after": {"unit": "customer"}, "source": "person", "project": "p1"}
        s.keep_log("decisions", "dev", "ledger/corrections/2026/10/20.jsonl", (json.dumps(correction) + "\n").encode(), consent="train")
        fix = {"id": "fix:r1:test_a.py", "test": "test_a.py", "status": "proposed", "failing_run": "r1", "passing_run": "r2", "failing_output": "FAIL", "commits": []}
        s.keep("fixes", "rnd", "fixes/fix_r1_test_a.py.json", json.dumps(fix).encode(), consent="train")
        step = {"at": "2026-10-20T10:03:00+00:00", "agent": "intern", "run": "run1", "tool": "search_evidence", "arguments": {"query": "duration leakage"},
                "result": {"results": [{"record_id": "LEAK-duration"}, {"record_id": "EXP-1"}]}, "ok": True}
        s.keep_log("actions", "dev", "agent_steps.jsonl", (json.dumps(step) + "\n").encode(), consent="train")
        shapers.shape(self.layout)

    def test_episodes_judgments_and_searches_are_built_with_their_evidence(self):
        from dclab_rnd.store.sets import others

        report = others.build(self.layout)
        self.assertIn("m6_stepwise_decisions/graph", report["sets"])
        self.assertIn("m6_stepwise_decisions/job", report["sets"])
        eps = {e.process: e for e in others.episodes(self.layout)}
        graph = eps["graph"]
        self.assertEqual([s.outcome for s in graph.steps], ["allowed", "blocked", "allowed"])
        self.assertEqual(graph.outcome, "WF-03")  # from the record
        self.assertEqual([s.action.split(":")[0] for s in eps["job"].steps], ["queued", "running", "done"])
        judged = {j.judged: j for j in others.judged(self.layout)}
        self.assertEqual((judged["card"].candidate_by, judged["card"].judge, judged["card"].preferred), ("dclab", "person", "other"))
        self.assertEqual((judged["code"].evidence, judged["code"].weight), ("test", 0.5))  # proposed, not yet accepted by a person
        search = others.retrievals(self.layout)[0]
        self.assertEqual((search.query, search.offered, search.corpus), ("duration leakage", ["LEAK-duration", "EXP-1"], "evidence"))

    def test_a_walk_kept_twice_is_one_episode_and_stages_teams_and_postgresql_moves_are_episodes(self):
        from dclab_rnd.store import Store, shapers
        from dclab_rnd.store.sets import others

        s = Store(self.layout)
        walk = {"id": "walk:d1", "draft": "d1", "project": None, "steps": [{"at": "2026-10-20T10:00:00+00:00", "kind": "draft_event", "ref": "d1:1"}]}
        s.keep("workflows", "dev", "walks/d1.json", json.dumps(walk).encode(), consent="train")
        s.keep("workflows", "dev", "walks/d1.json", json.dumps({**walk, "project": "p1"}).encode(), consent="train")  # the walk moved on
        stage = lambda name, at, **ev: {"stage": name, "status": "completed", "started_at": at, "evidence": ev}
        s.keep("workflows", "dev", "projects/p1/project.json", json.dumps({"id": "p1", "solution": {"task": "binary"}}).encode(), consent="train")
        for name, at, ev in (("data", "2026-10-20T11:00:00+00:00", {}), ("models", "2026-10-20T11:02:00+00:00", {"selected_model": "logistic_regression"}),
                             ("final", "2026-10-20T11:04:00+00:00", {})):
            s.keep("workflows", "dev", f"projects/p1/stages/{name}.json", json.dumps(stage(name, at, **ev)).encode(), consent="train")
        boards = [{"at": f"2026-10-20T12:00:0{i}+00:00", "team": "orchestra", "node": node, "what": node, "walk": "w1",
                   "board": {"thread": "draft:d1", "state": state, "open_questions": ["outcome"] if i == 0 else [], "blockers": []}}
                  for i, (node, state) in enumerate([("answer", "problem"), ("map", "map")])]
        s.keep_log("workflows", "dev", "orchestra/boards.jsonl", "".join(json.dumps(b) + "\n" for b in boards).encode(), consent="train")
        rows = [{"id": i, "project_id": "p9", "at": f"2026-10-20T13:00:0{i}+00:00", "move": m, "status": "allowed",
                 "entry": {"at": f"2026-10-20T13:00:0{i}+00:00", "move": m, "from": "WF-01", "to": "WF-02", "status": "allowed"}} for i, m in enumerate(["set_solution", "approve_gate"])]
        s.keep_log("workflows", "pg", "db/transition.jsonl", "".join(json.dumps(r) + "\n" for r in rows).encode(), consent="train")
        shapers.shape(self.layout)
        eps = others.episodes(self.layout)
        walks = [e for e in eps if e.process == "walk"]
        self.assertEqual(len(walks), 1)
        stages = next(e for e in eps if e.process == "stages")
        self.assertEqual([st.action for st in stages.steps], ["data", "models:logistic_regression", "final"])
        self.assertEqual(stages.outcome, "completed")
        team = next(e for e in eps if e.process == "team/orchestra")
        self.assertEqual([st.action for st in team.steps], ["answer:answer", "map:map"])
        self.assertIn("open 1", team.steps[0].outcome)
        pg = next(e for e in eps if e.id == "m6:graph:pg:p9")
        self.assertEqual([st.action.split(":")[0] for st in pg.steps], ["set_solution", "approve_gate"])
        self.assertTrue(any(e.id == "m6:graph:dev:p1" for e in eps))


if __name__ == "__main__":
    unittest.main()

"""Package 32.34: DCLab Code's sets.

Each session on this repository becomes an M6 episode of tool steps (content-free states, silver: the steps are the
agent's), the person's judgements become M7 examples (gold), and the person's choices become M2 and M1 decisions; a tool's
output never enters a set; a session's split is keyed by its SDK session id.
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

SECRET_ROWS = "client_rows_4412,9.99,alice"


def line(ts, **body):
    return json.dumps({"v": "dclab_code.record.v1", "ts": ts, **body})


def sdk(ts, message):
    return line(ts, kind="sdk", message=message)


def session_lines(sid):
    t = lambda n: f"2026-10-25T10:00:{n:02d}.000Z"  # noqa: E731
    return [
        line(t(0), kind="app", type="user_message", id="u1", text="tidy the pipeline"),
        sdk(t(1), {"type": "system", "subtype": "init", "session_id": f"sdk-{sid}", "model": "claude-haiku-5-5", "permissionMode": "default", "tools": []}),
        sdk(t(2), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "rm -rf build"}}]}}),
        line(t(3), kind="app", type="permission_request", reqId="p1", tool="Bash", input={"command": "rm -rf build"}, reason="rm deletes files", guard=None, toolUseId="t1"),
        line(t(4), kind="app", type="permission_decision", reqId="p1", decision="deny"),
        sdk(t(5), {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "The person said no.", "is_error": True}]}}),
        sdk(t(6), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/r/data.csv"}}]}}),
        sdk(t(7), {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t2", "content": SECRET_ROWS, "is_error": False}]}}),
        sdk(t(8), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_use", "id": "t3", "name": "ExitPlanMode", "input": {"plan": "1. split by time"}}]}}),
        line(t(9), kind="app", type="plan_request", reqId="l1", plan="1. split by time\n2. test", toolUseId="t3"),
        line(t(10), kind="app", type="plan_decision", reqId="l1", choice="keep", feedback="use a group split"),
        sdk(t(11), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "text", "text": "I kept planning."}]}}),
        sdk(t(12), {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.01}),
        line(t(13), kind="app", type="feedback", turnTs=t(12), value="bad", note="it ignored the groups"),
        line(t(14), kind="app", type="outcome", outcome="merged", commits=1, into="main", branch="dclab/x"),
    ]


def edge_lines():
    """The review's cases in one session: decisions across an hour (the shaper rewrites timestamps), a stop's automatic deny,
    a step with no result, a subagent's permission, two graded turns, an old grade without its turn."""
    T = lambda h, m, s: f"2026-10-25T{h:02d}:{m:02d}:{s:02d}.000Z"  # noqa: E731
    tool = lambda ts, tid, name, inp, parent=None: sdk(ts, {"type": "assistant", "parent_tool_use_id": parent, "message": {"content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}})  # noqa: E731
    return [
        line(T(10, 59, 50), kind="app", type="user_message", id="u1", text="go"),
        sdk(T(10, 59, 51), {"type": "system", "subtype": "init", "session_id": "sdk-s_edge01", "model": "claude-sonnet-5-5", "permissionMode": "default", "tools": []}),
        tool(T(10, 59, 58), "e1", "Bash", {"command": "rm -rf out"}),
        line(T(11, 0, 1), kind="app", type="permission_request", reqId="pe1", tool="Bash", input={}, reason="rm", guard=None, toolUseId="e1"),
        line(T(11, 0, 5), kind="app", type="permission_decision", reqId="pe1", decision="deny", by="person"),
        tool(T(11, 0, 6), "e2", "Read", {"file_path": "/r/x.py"}),  # no tool_result follows
        tool(T(11, 0, 7), "e3", "Bash", {"command": "pytest -q"}, parent="tp1"),  # a subagent's step
        line(T(11, 0, 8), kind="app", type="permission_request", reqId="pe3", tool="Bash", input={}, reason=None, guard=None, toolUseId="e3"),
        line(T(11, 0, 9), kind="app", type="permission_decision", reqId="pe3", decision="allow"),
        sdk(T(11, 0, 10), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "text", "text": "Answer A."}]}}),
        sdk(T(11, 0, 11), {"type": "result", "subtype": "success", "is_error": False}),
        line(T(11, 0, 12), kind="app", type="feedback", turnTs=T(11, 0, 11), turn=1, value="good", note=None),
        line(T(11, 1, 0), kind="app", type="user_message", id="u2", text="more"),
        tool(T(11, 1, 1), "e4", "Read", {"file_path": "/r/y.py"}),
        line(T(11, 1, 2), kind="app", type="permission_request", reqId="pe4", tool="Read", input={}, reason="outside", guard=None, toolUseId="e4"),
        line(T(11, 1, 3), kind="app", type="stop"),
        line(T(11, 1, 3), kind="app", type="permission_decision", reqId="pe4", decision="deny", message="The person stopped the session."),
        sdk(T(11, 1, 4), {"type": "result", "subtype": "error_during_execution", "is_error": True}),
        line(T(11, 1, 5), kind="app", type="feedback", turnTs=T(11, 1, 4), value="bad", note="no answer"),  # an old grade: no turn
    ]


@unittest.skipIf(MISSING, MISSING)
class DclabCodeSetsTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout

        self.repo = Path(tempfile.mkdtemp()) / "My R&D"
        self.repo.mkdir()
        self.home = Path(tempfile.mkdtemp()) / ".dclab-code"
        (self.home / "records").mkdir(parents=True)
        sessions = []
        for i in range(12):  # enough sessions to reach every split
            sid = f"s_{i:06d}"
            (self.home / "records" / f"{sid}.jsonl").write_text("\n".join(session_lines(sid)) + "\n", encoding="utf-8")
            sessions.append({"id": sid, "projectPath": str(self.repo), "sdkSessionId": f"sdk-{sid}", "model": "claude-haiku-5-5"})
        (self.home / "records" / "s_edge01.jsonl").write_text("\n".join(edge_lines()) + "\n", encoding="utf-8")
        sessions.append({"id": "s_edge01", "projectPath": str(self.repo), "sdkSessionId": "sdk-s_edge01", "model": "claude-sonnet-5-5"})
        (self.home / "sessions.json").write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
        # The same session's Claude Code transcript (kept by the agents keeper): it must share the session's split.
        from dclab_rnd.store.keepers import agents
        self.projects = Path(tempfile.mkdtemp()) / "projects"
        (self.projects / agents.encoded(self.repo)).mkdir(parents=True)
        (self.projects / agents.encoded(self.repo) / "sdk-s_edge01.jsonl").write_text(
            json.dumps({"type": "user", "sessionId": "sdk-s_edge01", "message": {"role": "user", "content": "go"}}) + "\n", encoding="utf-8")
        base = Path(tempfile.mkdtemp()) / "agent_runs"
        base.mkdir()
        (base / "consent.json").write_text(json.dumps({"train": True}), encoding="utf-8")
        self.env = {"agent_runs": base, "claude_projects": self.projects, "dclab_code_home": self.home, "repo": self.repo, "test_runs": Path(tempfile.mkdtemp())}
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        from dclab_rnd.store import conductor
        from dclab_rnd.store.keepers import standard

        self.record = conductor.conduct(self.env, self.layout, standard.install({}))

    def rows(self, model_type, *folder):
        base = self.layout.sets(model_type, *folder, "v1")
        out = []
        for split in ("train", "dev", "sealed"):
            p = base / f"{split}.jsonl"
            if p.exists():
                out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        return out

    def test_episodes_judgements_and_decisions_are_built(self):
        build = next(s for s in self.record["steps"] if s["name"] == "builders")
        self.assertFalse([e for e in build["errors"] if e["source"] == "dclab_code"], build["errors"])
        eps = [e for e in self.rows("m6_stepwise_decisions", "dclab_code") if "edge" not in e["id"]]
        self.assertEqual(len(eps), 12)
        ep = eps[0]
        self.assertEqual([s["action"] for s in ep["steps"]], ["Bash", "Read", "ExitPlanMode"])
        self.assertEqual([s["outcome"] for s in ep["steps"]], ["denied", "auto_ok", "plan_keep"])
        self.assertEqual(ep["steps"][0]["state"], {"step": 0, "move": "bash", "kind": "shell", "method": "ask_first", "gate_required": True,
                                                    "concern_source": "sdk", "rounds": 0})
        self.assertEqual(ep["outcome"], "merged")
        self.assertEqual(ep["weight"], 0.5)  # the steps are Claude's: silver

        steps = [j for j in self.rows("m7_judges", "dclab_code_steps") if "edge" not in j["id"]]
        self.assertEqual({(j["score"], j["judge"], j["evidence"], j["candidate"]["input"]["command"]) for j in steps}, {(0.0, "person", "person", "rm -rf build")})
        self.assertEqual(steps[0]["candidate_by"], "model:claude-haiku-5-5")
        plans = self.rows("m7_judges", "dclab_code_plans")
        self.assertEqual(plans[0]["candidate"], {"plan": "1. split by time\n2. test", "feedback": "use a group split"})
        self.assertEqual(plans[0]["score"], 0.0)
        answers = [j for j in self.rows("m7_judges", "dclab_code_answers") if "edge" not in j["id"]]
        self.assertEqual(answers[0]["candidate"], {"answer": "I kept planning.", "note": "it ignored the groups"})

        perm = [d for d in self.rows("m2_workflow_decisions", "internal", "dclab_code.permission") if "edge" not in d["id"]]
        self.assertEqual({(d["label"], d["label_source"], d["point"]) for d in perm}, {("deny", "person", "dclab_code.permission")})
        self.assertEqual(len([d for d in self.rows("m1_students", "dclab_code.permission") if "edge" not in d["id"]]), 12)
        self.assertEqual({d["label"] for d in self.rows("m1_students", "dclab_code.plan_approval")}, {"keep"})

    def test_the_reviews_cases(self):
        from dclab_core.datasets import split_of
        from dclab_rnd.store import shapers

        ep = next(e for e in self.rows("m6_stepwise_decisions", "dclab_code") if e["id"].endswith("s_edge01"))
        self.assertEqual([(s["action"], s["outcome"]) for s in ep["steps"]],
                         [("Bash", "denied"), ("Read", "auto_no_result"), ("Read", "cancelled")])  # the subagent's step is not the agent's own
        self.assertEqual(ep["steps"][0]["state"]["method"], "ask_first")
        self.assertEqual(ep["outcome"], "stopped")
        perm = {d["label"] for d in self.rows("m2_workflow_decisions", "internal", "dclab_code.permission") if "s_edge01" in d["id"]}
        self.assertEqual(perm, {"deny", "allow"})  # the person's deny across the hour and the subagent's allow; the stop's deny is not a label
        self.assertEqual(len([d for d in self.rows("m1_students", "dclab_code.permission") if "s_edge01" in d["id"]]), 2)
        answers = [j for j in self.rows("m7_judges", "dclab_code_answers") if "s_edge01" in j["id"]]
        self.assertEqual([(j["candidate"]["answer"], j["score"]) for j in answers], [("Answer A.", 1.0)])  # the bad grade had no answer: not paired with A
        # The session and its Claude Code transcript share one split.
        transcript = [r for r in shapers.records(self.layout, "agents") if r.provenance.source.startswith("claude_code/")]
        self.assertTrue(transcript)
        self.assertEqual({r.split for r in transcript}, {split_of("sdk-s_edge01")})
        self.assertEqual(ep["split"], split_of("sdk-s_edge01"))

    def test_no_tool_output_in_any_set_and_one_split_per_session(self):
        text = "".join(p.read_text(encoding="utf-8") for p in self.layout.tier("sets").rglob("*.jsonl"))
        self.assertNotIn(SECRET_ROWS, text)
        splits = {}
        for model_type, folder in (("m6_stepwise_decisions", ("dclab_code",)), ("m7_judges", ("dclab_code_steps",)),
                                   ("m2_workflow_decisions", ("internal", "dclab_code.permission"))):
            base = self.layout.sets(model_type, *folder, "v1")
            for split in ("train", "dev", "sealed"):
                p = base / f"{split}.jsonl"
                for x in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
                    ex = json.loads(x)
                    sid = ex["id"].split(":")[-2] if model_type != "m6_stepwise_decisions" else ex["id"].split(":")[-1]
                    splits.setdefault(sid, set()).add(split)
        self.assertTrue(all(len(v) == 1 for v in splits.values()), splits)  # one session, one split, in every set
        self.assertGreater(len({next(iter(v)) for v in splits.values()}), 1)  # and the sessions do spread over splits


if __name__ == "__main__":
    unittest.main()

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
        sdk(T(10, 59, 57), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "thinking", "thinking": "Clean the outputs first."}]}}),
        tool(T(10, 59, 58), "e1", "Bash", {"command": "rm -rf out"}),
        line(T(11, 0, 1), kind="app", type="permission_request", reqId="pe1", tool="Bash", input={}, reason="rm", guard=None, toolUseId="e1"),
        line(T(11, 0, 5), kind="app", type="permission_decision", reqId="pe1", decision="deny", by="person"),
        tool(T(11, 0, 5), "e5", "Bash", {"command": "rm -rf out/tmp"}),  # the next try, narrower, which the person allows
        line(T(11, 0, 5), kind="app", type="permission_request", reqId="pe5", tool="Bash", input={}, reason="rm", guard=None, toolUseId="e5"),
        line(T(11, 0, 5), kind="app", type="permission_decision", reqId="pe5", decision="allow", by="person"),
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
        line(T(11, 2, 0), kind="app", type="user_message", id="u3", text="third task"),
        line(T(11, 2, 0), kind="app", type="user_message", id="u4", text="MID-TURN MESSAGE"),
        sdk(T(11, 2, 1), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [
            {"type": "tool_use", "id": "e6", "name": "Bash", "input": {"command": "rm -rf data"}},
            {"type": "tool_use", "id": "e7", "name": "Bash", "input": {"command": "git status"}}]}}),
        line(T(11, 2, 2), kind="app", type="permission_request", reqId="pe6", tool="Bash", input={}, reason="rm", guard=None, toolUseId="e6"),
        line(T(11, 2, 3), kind="app", type="permission_decision", reqId="pe6", decision="deny", by="person"),
        line(T(11, 2, 4), kind="app", type="permission_request", reqId="pe7", tool="Bash", input={}, reason=None, guard=None, toolUseId="e7"),
        line(T(11, 2, 5), kind="app", type="permission_decision", reqId="pe7", decision="allow", by="person"),
        sdk(T(11, 2, 5), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "thinking", "thinking": "THINKING OF TURN 3"}]}}),
        sdk(T(11, 2, 6), {"type": "result", "subtype": "success", "is_error": False}),
        line(T(11, 3, 0), kind="app", type="user_message", id="u5", text="fourth task"),
        tool(T(11, 3, 1), "e8", "Bash", {"command": "rm -rf cache"}),  # no thinking before it in this turn
        line(T(11, 3, 2), kind="app", type="permission_request", reqId="pe8", tool="Bash", input={}, reason="rm", guard=None, toolUseId="e8"),
        line(T(11, 3, 3), kind="app", type="permission_decision", reqId="pe8", decision="deny", by="person"),
        tool(T(11, 3, 4), "e9", "Bash", {"command": "rm -rf cache/old"}),
        line(T(11, 3, 5), kind="app", type="permission_request", reqId="pe9", tool="Bash", input={}, reason="rm", guard=None, toolUseId="e9"),
        line(T(11, 3, 6), kind="app", type="permission_decision", reqId="pe9", decision="allow", by="person"),
        sdk(T(11, 3, 7), {"type": "result", "subtype": "success", "is_error": False}),
    ]


def undo_lines(repo):
    """An undo, a failed edit, an edit outside the repository, a committed second try, a change after the commit, an undo
    after the commit, and a data file."""
    T = lambda m, s: f"2026-10-25T12:{m:02d}:{s:02d}.000Z"  # noqa: E731
    def edit(ts, tid, name, inp, ok=True):
        return [sdk(ts, {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}}),
                sdk(ts, {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok" if ok else "not found", "is_error": not ok}]}})]
    done = lambda ts: sdk(ts, {"type": "result", "subtype": "success", "is_error": False})  # noqa: E731
    return [
        line(T(0, 0), kind="app", type="user_message", id="u1", text="fix the model"),
        sdk(T(0, 1), {"type": "system", "subtype": "init", "session_id": "sdk-s_undo01", "model": "claude-sonnet-5-5", "permissionMode": "acceptEdits", "tools": []}),
        *edit(T(0, 2), "w1", "Edit", {"file_path": f"{repo}/m.py", "old_string": "a = 1", "new_string": "a = 2"}),
        *edit(T(0, 3), "w2", "Write", {"file_path": f"{repo}/notes.md", "content": "hello"}),
        *edit(T(0, 4), "w3", "Write", {"file_path": f"{repo}/data.csv", "content": "client,9.99"}),
        *edit(T(0, 4), "w6", "Write", {"file_path": "/Users/p/.claude/memory/notes.md", "content": "outside"}),
        *edit(T(0, 4), "w7", "Edit", {"file_path": f"{repo}/m.py", "old_string": "zzz", "new_string": "BAD"}, ok=False),
        done(T(0, 5)),
        line(T(1, 0), kind="app", type="undo", file="m.py", how="restored", backup="/b/m.py", by="person"),
        line(T(1, 1), kind="app", type="user_message", id="u2", text="try again"),
        *edit(T(1, 2), "w4", "Edit", {"file_path": f"{repo}/m.py", "old_string": "a = 1", "new_string": "a = 3"}),
        done(T(1, 3)),
        line(T(2, 0), kind="app", type="outcome", outcome="committed", sha="abc1234", files=["m.py", "notes.md"], count=2, branch="dclab/x", by="person"),
        *edit(T(2, 1), "w5", "Edit", {"file_path": f"{repo}/notes.md", "old_string": "hello", "new_string": "hello again"}),
        *edit(T(2, 1), "w8", "Edit", {"file_path": f"{repo}/m.py", "old_string": "a = 3", "new_string": "LATER"}),
        done(T(2, 2)),
        line(T(3, 0), kind="app", type="undo", file="m.py", how="restored", backup="/b/m2.py", by="person"),
        line(T(3, 1), kind="app", type="person_command", id="c1", command="make test", exit=1, timedOut=False, output="1 failed", by="person", guard=None),
        line(T(4, 0), kind="app", type="person_command_start", id="c2", by="person"),  # started before the agent's next edit...
        *edit(T(4, 1), "w9", "Edit", {"file_path": f"{repo}/m.py", "old_string": "a = 1", "new_string": "a = 4"}),
        line(T(4, 2), kind="app", type="person_command", id="c2", command="make long", exit=None, timedOut=True, output="", by="person", guard=None),  # ...ended after it
        line(T(4, 3), kind="app", type="person_command", id="c3", command="cat sealed.json", exit=0, timedOut=False, output=None, by="person", guard="sealed"),
        line(T(4, 4), kind="app", type="person_command", id="c4", command="x", exit=0, timedOut=False, output="", by="app", guard=None),  # never a person step
        done(T(4, 5)),
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
        (self.home / "records" / "s_undo01.jsonl").write_text("\n".join(undo_lines(self.repo)) + "\n", encoding="utf-8")
        sessions.append({"id": "s_undo01", "projectPath": str(self.repo), "sdkSessionId": "sdk-s_undo01", "model": "claude-sonnet-5-5"})
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
        eps = [e for e in self.rows("m6_stepwise_decisions", "dclab_code") if "edge" not in e["id"] and "undo" not in e["id"]]
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
                         [("Bash", "denied"), ("Bash", "allowed_no_result"), ("Read", "auto_no_result"), ("Read", "cancelled"),
                          ("Bash", "denied"), ("Bash", "allowed_no_result"), ("Bash", "denied"), ("Bash", "allowed_no_result")])  # the subagent's step is not the agent's own
        pairs = [p for p in self.rows("m7_judges", "dclab_code_preferences") if "s_edge01" in p["id"]]
        self.assertEqual(len(pairs), 2)  # turns 1 and 4; the parallel deny and allow of the third turn are not two tries
        self.assertEqual(pairs[0]["split"], split_of("sdk-s_edge01"))
        self.assertEqual(pairs[0]["chosen"]["tool_calls"][0]["input"]["command"], "rm -rf out/tmp")
        self.assertEqual(pairs[0]["rejected"]["tool_calls"][0]["input"]["command"], "rm -rf out")
        self.assertEqual((pairs[0]["rejected"]["reasoning"], pairs[0]["rejected"]["reasoning_source"]), ("Clean the outputs first.", "recorded"))
        self.assertEqual(pairs[0]["prompt"][0]["content"], "go")
        self.assertEqual(pairs[0]["decided_by"], "person")
        silver = [p for p in self.rows("m5_solver_llm", "dclab_code_preferences") if "s_edge01" in p["id"]]
        self.assertEqual([p["weight"] for p in silver], [0.5, 0.5])  # both steps are Claude's: silver for the solver
        manifest = json.loads((self.layout.sets("m5_solver_llm", "dclab_code_preferences", "v1") / "MANIFEST.json").read_text(encoding="utf-8"))
        from dclab_rnd.store import provider_outputs
        self.assertEqual(manifest["consent"], provider_outputs.consent())  # the owner's decision on Claude's outputs (2026-10-25: train)
        # Thinking never crosses a turn: turn 3 ended with thinking, turn 4's denied step had none before it.
        fourth = next(p for p in pairs if p["rejected"]["tool_calls"][0]["input"]["command"] == "rm -rf cache")
        self.assertIsNone(fourth["rejected"].get("reasoning"))
        self.assertEqual(fourth["prompt"][0]["content"], "fourth task")
        third_prompt = [x for x in self.rows("m6_stepwise_decisions", "dclab_code") if x["id"].endswith("s_edge01")]
        self.assertTrue(third_prompt)
        self.assertEqual(ep["steps"][0]["state"]["method"], "ask_first")
        self.assertEqual(ep["outcome"], "success")  # the last turn ran to its end
        perm = {d["label"] for d in self.rows("m2_workflow_decisions", "internal", "dclab_code.permission") if "s_edge01" in d["id"]}
        self.assertEqual(perm, {"deny", "allow"})  # the person's deny across the hour and the allows; the stop's deny is not a label
        self.assertEqual(len([d for d in self.rows("m1_students", "dclab_code.permission") if "s_edge01" in d["id"]]), 7)
        answers = [j for j in self.rows("m7_judges", "dclab_code_answers") if "s_edge01" in j["id"]]
        self.assertEqual([(j["candidate"]["answer"], j["score"]) for j in answers], [("Answer A.", 1.0)])  # the bad grade had no answer: not paired with A
        # The session and its Claude Code transcript share one split.
        transcript = [r for r in shapers.records(self.layout, "agents") if r.provenance.source.startswith("claude_code/")]
        self.assertTrue(transcript)
        self.assertEqual({r.split for r in transcript}, {split_of("sdk-s_edge01")})
        self.assertEqual(ep["split"], split_of("sdk-s_edge01"))

    def test_a_scripts_decision_is_never_a_persons_label(self):
        from dclab_rnd.store import Layout, conductor
        from dclab_rnd.store.keepers import standard

        T = lambda s: f"2026-10-25T13:00:{s:02d}.000Z"  # noqa: E731
        tool = lambda s, tid, cmd, name="Bash": sdk(T(s), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_use", "id": tid, "name": name, "input": cmd if isinstance(cmd, dict) else {"command": cmd}}]}})  # noqa: E731
        result = lambda s, tid, err=False: sdk(T(s), {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": "x", "is_error": err}]}})  # noqa: E731
        def edit(s, tid, inp):
            return [sdk(T(s), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_use", "id": tid, "name": "Edit", "input": inp}]}}),
                    sdk(T(s), {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok", "is_error": False}]}})]
        lines = [
            line(T(0), kind="app", type="user_message", id="u1", text="go"),
            sdk(T(1), {"type": "system", "subtype": "init", "session_id": "sdk-s_script01", "model": "claude-sonnet-5-5", "permissionMode": "plan", "tools": []}),
            tool(1, "kp", {"plan": "1. clean"}, "ExitPlanMode"),
            line(T(1), kind="app", type="plan_request", reqId="pl1", plan="1. clean", toolUseId="kp"),
            line(T(1), kind="app", type="plan_decision", reqId="pl1", choice="auto", feedback=None, by="script"),  # the mode did switch
            tool(2, "k1", "rm -rf out"),
            line(T(3), kind="app", type="permission_request", reqId="ps1", tool="Bash", input={}, reason="rm", guard=None, toolUseId="k1"),
            line(T(4), kind="app", type="permission_decision", reqId="ps1", decision="deny", by="script"),  # a test script clicked
            result(4, "k1", err=True),
            tool(5, "k2", "rm -rf out/tmp"),  # the next same-tool step after a deny: a preference pair, were the deny a person's
            line(T(6), kind="app", type="permission_request", reqId="ps2", tool="Bash", input={}, reason="rm", guard=None, toolUseId="k2"),
            line(T(7), kind="app", type="permission_decision", reqId="ps2", decision="allow"),  # an old record: no "by", a person's
            result(7, "k2"),
            tool(7, "kq", {"questions": [{"question": "Which?"}]}, "AskUserQuestion"),
            line(T(7), kind="app", type="question_request", reqId="q1", questions=[{"question": "Which?"}], toolUseId="kq"),
            line(T(7), kind="app", type="question_answer", reqId="q1", answers={"Which?": "A"}, by="script"),
            sdk(T(8), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "text", "text": "Removed out/tmp."}]}}),
            sdk(T(9), {"type": "result", "subtype": "success", "is_error": False}),
            line(T(10), kind="app", type="feedback", turnTs=T(9), turn=1, value="good", note=None, by="script"),  # a script's grade
            *edit(11, "k3", {"file_path": f"{self.repo}/s.py", "old_string": "a", "new_string": "b"}),
            line(T(12), kind="app", type="outcome", outcome="committed", sha="s1", files=["s.py"], count=1, branch="dclab/s", by="script"),
            line(T(13), kind="app", type="outcome", outcome="committed", sha="s2", files=["s.py"], count=1, branch="dclab/s", by="person"),
            *edit(14, "k4", {"file_path": f"{self.repo}/s.py", "old_string": "b", "new_string": "c"}),
            line(T(15), kind="app", type="undo", file="s.py", how="restored", backup="/b/s.py", by="script"),
            line(T(16), kind="app", type="outcome", outcome="merged", files=["s.py"], commits=1, into="main", branch="dclab/s", by="script"),
        ]
        (self.home / "records" / "s_script01.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        listed = json.loads((self.home / "sessions.json").read_text(encoding="utf-8"))
        listed["sessions"].append({"id": "s_script01", "projectPath": str(self.repo), "sdkSessionId": "sdk-s_script01", "model": "claude-sonnet-5-5"})
        (self.home / "sessions.json").write_text(json.dumps(listed), encoding="utf-8")
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        conductor.conduct(self.env, self.layout, standard.install({}))
        perm = [d for d in self.rows("m2_workflow_decisions", "internal", "dclab_code.permission") if "s_script01" in d["id"]]
        self.assertEqual([(d["label"], d["label_source"]) for d in perm], [("allow", "person")])  # the script's deny is no person's label
        steps = [j for j in self.rows("m7_judges", "dclab_code_steps") if "s_script01" in j["id"]]
        self.assertEqual([(j["candidate"]["input"]["command"], j["score"]) for j in steps], [("rm -rf out/tmp", 1.0)])  # nor a person's judgement of the script's step
        self.assertFalse([j for j in self.rows("m7_judges", "dclab_code_answers") if "s_script01" in j["id"]])  # nor its grade
        # A script's commit and undo cut the file's history with no verdict: the person's later commit judges nothing.
        self.assertFalse([j for j in self.rows("m7_judges", "dclab_code_changes") if "s_script01" in j["id"]])
        # Nor its plan choice, nor a kept-against-refused pair built on its deny.
        for mt, folder in (("m7_judges", ("dclab_code_plans",)), ("m2_workflow_decisions", ("internal", "dclab_code.plan_approval")),
                           ("m1_students", ("dclab_code.plan_approval",)), ("m7_judges", ("dclab_code_preferences",)), ("m5_solver_llm", ("dclab_code_preferences",))):
            self.assertFalse([j for j in self.rows(mt, *folder) if "s_script01" in j["id"]], (mt, folder))
        # The episode keeps the facts: the mode the script's approval switched to, what each step did, marked as the script's.
        ep = next(e for e in self.rows("m6_stepwise_decisions", "dclab_code") if e["id"].endswith("s_script01"))
        self.assertEqual([(x["action"], x["state"]["method"], x["outcome"]) for x in ep["steps"]],
                         [("ExitPlanMode", "plan_only", "plan_auto_by_script"), ("Bash", "auto_edits", "denied_by_script"),
                          ("Bash", "auto_edits", "allowed_ok"), ("AskUserQuestion", "auto_edits", "answered_by_script"),
                          ("Edit", "auto_edits", "auto_ok"), ("Edit", "auto_edits", "auto_ok")])  # and no person:undo for the script's undo
        self.assertEqual(ep["outcome"], "merged_by_script")  # a script's merge is no sign the person kept the work

    def test_a_rewind_rejects_the_changes_since_its_message(self):
        """32.41: DCLab Code's Rewind puts files back as they were before one of the person's messages."""
        from dclab_rnd.store import Layout, conductor
        from dclab_rnd.store.keepers import standard

        T = lambda s: f"2026-10-25T15:00:{s:02d}.000Z"  # noqa: E731
        def edit(s, tid, f, new):
            return [sdk(T(s), {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_use", "id": tid, "name": "Edit", "input": {"file_path": f"{self.repo}/{f}", "old_string": "a", "new_string": new}}]}}),
                    sdk(T(s), {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok", "is_error": False}]}})]
        done = lambda s: sdk(T(s), {"type": "result", "subtype": "success", "is_error": False})  # noqa: E731
        lines = [
            line(T(0), kind="app", type="user_message", id="u1", uuid="m-1", text="add the split", by="person"),
            sdk(T(1), {"type": "system", "subtype": "init", "session_id": "sdk-s_rewind01", "model": "claude-opus-5-5", "permissionMode": "acceptEdits", "tools": []}),
            *edit(2, "r1", "a.py", "FIRST"), done(3),
            line(T(4), kind="app", type="user_message", id="u2", uuid="m-2", text="now tune it", by="person"),
            *edit(5, "r2", "a.py", "SECOND"), *edit(6, "r3", "b.py", "THIRD"), done(7),
            # the SDK names files by their real path; /private + root counts as the root
            line(T(8), kind="app", type="rewind", messageId="m-2", turn=2, filesChanged=[f"/private{self.repo}/a.py", f"{self.repo}/b.py", "/elsewhere/c.py"],
                 insertions=2, deletions=2, skippedLinks=0, by="person"),
            line(T(9), kind="app", type="outcome", outcome="committed", sha="c1", files=["a.py"], count=1, branch="dclab/r", by="person"),
            line(T(10), kind="app", type="user_message", id="u3", uuid="m-3", text="try once more", by="person"),
            *edit(11, "r4", "b.py", "FOURTH"), done(12),
            line(T(13), kind="app", type="rewind", messageId="m-3", turn=3, filesChanged=[f"{self.repo}/b.py"], insertions=1, deletions=1, skippedLinks=0, by="script"),
            # 32.42: a background job the person stopped is their step; a script's stop is not
            line(T(14), kind="app", type="job_stop", taskId="j1", jobType="local_bash", description="python train.py", by="person"),
            line(T(15), kind="app", type="job_stop", taskId="j2", jobType="local_bash", description="make test", by="script"),
            # dclab-code A3: a file the person opened is their step (its path never in the episode); a sealed one refused
            line(T(16), kind="app", type="file_opened", path="src/secret_name.py", line=3, sealed=False, by="person"),
            line(T(17), kind="app", type="file_opened", path="data/holdout.csv", line=None, sealed=True, by="person"),
            line(T(18), kind="app", type="file_opened", path="src/x.py", line=None, sealed=False, by="script"),
        ]
        (self.home / "records" / "s_rewind01.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        listed = json.loads((self.home / "sessions.json").read_text(encoding="utf-8"))
        listed["sessions"].append({"id": "s_rewind01", "projectPath": str(self.repo), "sdkSessionId": "sdk-s_rewind01", "model": "claude-opus-5-5"})
        (self.home / "sessions.json").write_text(json.dumps(listed), encoding="utf-8")
        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        conductor.conduct(self.env, self.layout, standard.install({}))
        rows = [j for j in self.rows("m7_judges", "dclab_code_changes") if "s_rewind01" in j["id"]]
        got = sorted((j["candidate"]["file"], j["candidate"]["kept_as"], j["score"], [c["new_string"] for c in j["candidate"]["changes"]]) for j in rows)
        self.assertEqual(got, [("a.py", "committed", 1.0, ["FIRST"]),    # before the rewound message: kept, then committed
                               ("a.py", "rewound", 0.0, ["SECOND"]),     # since it: rejected
                               ("b.py", "rewound", 0.0, ["THIRD"])])     # a file outside the repository is left out; a script's rewind judges nothing
        ep = next(e for e in self.rows("m6_stepwise_decisions", "dclab_code") if e["id"].endswith("s_rewind01"))
        self.assertEqual([x["action"] for x in ep["steps"] if x["action"].startswith("person:")],
                         ["person:rewind", "person:stop_job", "person:open_file", "person:open_file"])
        self.assertEqual([x["outcome"] for x in ep["steps"] if x["action"] == "person:open_file"], ["opened", "sealed_refused"])
        self.assertNotIn("python train.py", json.dumps(ep))  # a job's description never enters the content-free episode
        self.assertNotIn("secret_name", json.dumps(ep))  # nor a file's path

    def test_the_persons_steps_sit_in_the_episode(self):
        ep = next(e for e in self.rows("m6_stepwise_decisions", "dclab_code") if e["id"].endswith("s_undo01"))
        actions = [(s["action"], s["outcome"]) for s in ep["steps"]]
        self.assertEqual(actions, [("Edit", "auto_ok"), ("Write", "auto_ok"), ("Write", "auto_ok"), ("Write", "auto_ok"), ("Edit", "auto_error"),
                                   ("person:undo", "restored"), ("Edit", "auto_ok"), ("Edit", "auto_ok"), ("Edit", "auto_ok"),
                                   ("person:undo", "restored"), ("person:Bash", "exit_1"),
                                   ("person:Bash", "timed_out"), ("Edit", "auto_ok"),  # placed where it started, before the edit made meanwhile
                                   ("person:Bash", "exit_0")])                         # the app's own command is not a person step
        sealed = ep["steps"][-1]
        self.assertEqual(sealed["state"]["concern_source"], "sealed")
        self.assertEqual(ep["weight"], 0.5)  # Claude made steps in it
        person = [s for s in ep["steps"] if s["action"].startswith("person:")]
        self.assertTrue(all(s["state"]["kind"] in ("person_undo", "person_shell") and s["state"]["gate_required"] is False for s in person))
        self.assertNotIn("1 failed", json.dumps(ep))  # a command's output never enters a set

    def test_file_changes_judged_by_undo_and_commit(self):
        from dclab_core.datasets import split_of

        rows = [j for j in self.rows("m7_judges", "dclab_code_changes") if "s_undo01" in j["id"]]
        got = sorted((j["candidate"]["file"], j["candidate"]["kept_as"], j["score"], [c.get("new_string") or c.get("content") for c in j["candidate"]["changes"]]) for j in rows)
        self.assertEqual(got, [("m.py", "committed", 1.0, ["a = 3"]),     # the second try, committed
                               ("m.py", "undone", 0.0, ["LATER"]),        # an undo after the commit judges only what came after it
                               ("m.py", "undone", 0.0, ["a = 2"]),        # the person undid the first; the failed edit never counts
                               ("notes.md", "committed", 1.0, ["hello"])])  # the change after the commit has no verdict
        self.assertFalse([j for j in rows if j["candidate"]["file"].startswith("/")])  # a file outside the repository never
        self.assertFalse([j for j in rows if j["candidate"]["file"].endswith(".csv")])  # a data file never enters a set
        self.assertEqual({(j["judge"], j["evidence"], j["candidate_by"], j["split"]) for j in rows},
                         {("person", "person", "model:claude-sonnet-5-5", split_of("sdk-s_undo01"))})

    def test_a_fork_keeps_its_copied_history_once_and_its_sources_split(self):
        # dclab-code A4: a fork copies its source's records (each marked copiedFrom); they are the source's, never twice
        copied = [json.dumps({**json.loads(x), "session": "s_fork01", "copiedFrom": "s_000000"}) for x in session_lines("s_000000")]
        own = [line("2026-10-25T11:00:00.000Z", kind="app", type="user_message", id="u9", uuid="m-9", text="try another way", by="person")]
        (self.home / "records" / "s_fork01.jsonl").write_text("\n".join(copied + own) + "\n", encoding="utf-8")
        listed = json.loads((self.home / "sessions.json").read_text(encoding="utf-8"))
        listed["sessions"].append({"id": "s_fork01", "projectPath": str(self.repo), "sdkSessionId": "sdk-forked-x", "model": "claude-haiku-5-5",
                                   "forkOf": {"session": "s_000000", "messageId": "m-2"}})
        (self.home / "sessions.json").write_text(json.dumps(listed), encoding="utf-8")
        from dclab_rnd.store import Layout, conductor
        from dclab_rnd.store.keepers import standard
        from dclab_rnd.store.sets import dclab_code as sets

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        conductor.conduct(self.env, self.layout, standard.install({}))
        labels = [r["id"] for r in self.rows("m2_workflow_decisions", "internal", "dclab_code.permission")]
        self.assertTrue([i for i in labels if "s_000000" in i])
        self.assertFalse([i for i in labels if "s_fork01" in i], "the fork's copies make no second label")
        fork = next(x for x in sets.sessions(self.layout) if x.sid == "s_fork01")
        self.assertEqual((fork.split_key, len(fork.steps)), ("sdk-s_000000", 0))  # the source's split; no copied step

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

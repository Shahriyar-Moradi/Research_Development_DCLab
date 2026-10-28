"""35.8: does memory help? The plan, the earlier session written through the product's paths, the two arms, the score
against what memory holds, resuming under a cap, the gate, and every session kept whole with its thinking."""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import memory_value as mv  # noqa: E402
from dclab_rnd.models.gateway import Gateway  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402

KEYED = {"OPENAI_API_KEY": "k", "DCLAB_NO_LIVE_MODELS": "1", **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}}


def reply(text="", calls=(), thinking="I read the brief before I choose."):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": text, "tool_calls": calls, "usage": {"input_tokens": 100, "output_tokens": 20}, "thinking": [{"type": "thinking", "thinking": thinking}],
            "assistant_message": {"role": "assistant", "content": text, "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class Reader:
    """A scripted model: proposes, then saves the proposal's forbidden columns except those its brief says to keep."""
    first: list[str] = []

    def complete(self, messages, tools=None, max_tokens=None, **_):
        task = messages[1]["content"]
        pid = re.search(r"existing project ([0-9a-f]{12})", task).group(1)
        last = messages[-1]
        if last["role"] != "tool":
            Reader.first.append(task)
            return reply(calls=[("propose_solution", {"project_id": pid, "target": "target"})])
        result = json.loads(last["content"])
        if "forbidden" in result and "task" in result:
            keep = set(re.findall(r"(\w+) is answered in the first week", task))
            forbid = [{"column": f["column"], "reason": f["reason"]} for f in result["forbidden"] if f["column"] not in keep]
            if not keep and "post_signup_survey" not in {f["column"] for f in forbid}:
                forbid.append({"column": "post_signup_survey", "reason": "its name says it comes after the outcome"})
            return reply(calls=[("set_solution", {"project_id": pid, "target": "target", "task": result["task"], "forbidden": forbid,
                                                  "identifiers": result.get("identifiers") or [], "prediction_moment": "At the monthly snapshot."})])
        return reply(calls=[("finish", {"report": "Solution saved."})])


def gateway():
    return Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Reader())


SMALL = {**mv.manifest(), "repeats": 1, "scenarios": [s for s in mv.manifest()["scenarios"] if s["case"] == "AE-16"]}


class PlanTest(unittest.TestCase):
    def test_every_case_twice_a_repeat_with_the_order_alternating(self):
        plan = mv.schedule()
        m = mv.manifest()
        self.assertEqual(len(plan), len(m["scenarios"]) * m["repeats"] * 2)
        self.assertEqual(len({i["id"] for i in plan}), len(plan))
        self.assertEqual([plan[0]["arm"], plan[2]["arm"]], ["on", "off"])  # the next case starts with the other arm
        self.assertEqual({i["case"] for i in plan}, {c.id for c in mv.cases()})

    def test_the_gate_is_fixed_in_the_manifest(self):
        gate = mv.manifest()["gate"]
        for key in ("against_mean_at_most", "sign_flip_p_below", "control_loss_at_most", "input_tokens_ratio_at_most", "max_attempts", "min_pair_share", "draws", "seed"):
            self.assertIn(key, gate)


class FirstSessionTest(unittest.TestCase):
    def test_the_correction_reaches_the_brief_only_with_memory_on(self):
        from dclab_rnd.intern import Intern, SessionStore
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.memory.tools import workspace_home
        from dclab_rnd.studio import ProjectStore

        home = Path(tempfile.mkdtemp())
        store = ProjectStore(home / "projects")
        pid = store.create("t", "general", "Predict the target column")["id"]
        scenario = next(s for s in mv.manifest()["scenarios"] if s["case"] == "AE-16")
        active = mv.seed_memory(workspace_home(store), "local", pid, scenario)
        self.assertEqual(len(active), 4)  # the workspace's columns, the preference, accepted know-how, the correction
        intern = Intern(SessionStore(home / "sessions"), Toolbox(store), None)
        with mv.arm("on"):
            text, shown = intern._memory(pid)
        self.assertIn("(correction) post_signup_survey is answered in the first week", text)
        self.assertIn("monthly_spend", text)
        self.assertNotIn("its name says it comes after", text)  # the agent's wrong line is retired
        with mv.arm("off"):
            text, shown = intern._memory(pid)
        self.assertNotIn("post_signup_survey", text)
        self.assertEqual(shown, [])

    def test_against_reads_lost_and_used_columns(self):
        said = {"post_signup_survey": "keep", "x_adj7": "out", "tenure_months": "keep"}
        project = {"solution": {"target": "t", "task": "binary", "forbidden": [{"column": "post_signup_survey"}], "time_column": "tenure_months"}}
        self.assertEqual(mv.against(said, project), ["post_signup_survey", "tenure_months", "x_adj7"])
        project = {"solution": {"target": "t", "task": "binary", "forbidden": [{"column": "x_adj7"}]}}
        self.assertEqual(mv.against(said, project), [])
        self.assertEqual(mv.against(said, {}), sorted(said))  # no solution saved: doing nothing never counts as obeying memory


class RunTest(unittest.TestCase):
    def test_both_arms_run_resume_and_keep_each_session_whole(self):
        Reader.first = []
        folder = Path(tempfile.mkdtemp()) / "MV-test"
        with mock.patch.object(mv, "manifest", lambda: SMALL), mock.patch.dict(os.environ, KEYED):
            out = mv.run_chunk(gateway(), folder, None, 200, 8, log=lambda *a: None)
            self.assertEqual((len(out["ran"]), out["left"]), (2, 0))
            again = mv.run_chunk(gateway(), folder, None, 200, 8, log=lambda *a: None)
            self.assertEqual(again["ran"], [])  # resumed: nothing done twice
            rows = mv.done(folder)
            on, off = rows["AE-16-r0-on"], rows["AE-16-r0-off"]
            self.assertTrue(on["ran"] and off["ran"], (on.get("why"), off.get("why")))
            self.assertEqual((on["against"], off["against"]), ([], ["post_signup_survey"]))  # the brief changed the decision
            self.assertTrue(any("(correction) post_signup_survey" in t for t in Reader.first))
            steps = on["session"]["steps"]
            self.assertTrue(steps and all(s.get("thought") == "I read the brief before I choose." for s in steps), steps[:1])
            self.assertTrue(all(s.get("thought_source") == "recorded" for s in steps))
            self.assertEqual(json.loads((folder / "run.json").read_text())["manifest_sha256"], mv.manifest_sha())
            s = mv.summarize(folder)
        self.assertEqual(s["arms"]["all"]["on"]["against_memory"]["share"], 0.0)
        self.assertEqual(s["arms"]["all"]["off"]["against_memory"]["share"], 1.0)
        self.assertTrue(on["training_use"].startswith("eval_only"))

    def test_a_failed_session_runs_again_and_its_first_attempt_is_kept(self):
        folder = Path(tempfile.mkdtemp()) / "MV-retry"
        with mock.patch.object(mv, "manifest", lambda: SMALL), mock.patch.dict(os.environ, KEYED):
            real = mv.one_session
            calls = {"n": 0}

            def flaky(gw, item, *a, **k):
                calls["n"] += 1
                if calls["n"] == 1:
                    return {**item, "ran": False, "status": "failed", "why": "the run failed: overloaded", "eur": 0.0, "memory_shown": 0, "session": {"steps": []}}
                return real(gw, item, *a, **k)

            with mock.patch.object(mv, "one_session", flaky):
                first = mv.run_chunk(gateway(), folder, None, 200, 8, log=lambda *a: None)
                self.assertEqual(first["left"], 1)  # the failed one is still to run
                mv.run_chunk(gateway(), folder, None, 200, 8, log=lambda *a: None)
            self.assertEqual(len(list(folder.glob("*.attempt1.json"))), 1)  # the failed attempt kept whole, not overwritten
            self.assertTrue(all(r["ran"] for r in mv.done(folder).values()))
            self.assertEqual(mv.pending(folder), [])

    def test_a_changed_setup_or_thinking_off_is_refused(self):
        folder = Path(tempfile.mkdtemp()) / "MV-setup"
        with mock.patch.object(mv, "manifest", lambda: SMALL), mock.patch.dict(os.environ, KEYED):
            with mock.patch.dict(os.environ, {"DCLAB_THINKING": "off"}), self.assertRaises(SystemExit):
                mv.run_chunk(gateway(), folder, None, 200, 8, log=lambda *a: None)
            folder.mkdir(parents=True)
            (folder / "run.json").write_text(json.dumps({**mv.setup(), "model": "another-model"}))
            with self.assertRaises(SystemExit):
                mv.run_chunk(gateway(), folder, None, 200, 8, log=lambda *a: None)

    def test_a_cap_stops_before_the_next_session(self):
        folder = Path(tempfile.mkdtemp()) / "MV-cap"
        with mock.patch.object(mv, "manifest", lambda: {**SMALL, "estimate_eur_per_session": 0.6}), mock.patch.dict(os.environ, KEYED):
            out = mv.run_chunk(gateway(), folder, 0.5, 200, 8, log=lambda *a: None)
        self.assertEqual(out["ran"], [])  # the estimate does not fit the cap: nothing is sent


def row(case, r, arm, against, leak=True, alarm=False, tokens=1000, saved=True):
    return {"id": f"{case}-r{r}-{arm}", "case": case, "repeat": r, "arm": arm, "ran": True, "against_memory": against, "solution_saved": saved,
            "leak_caught": leak, "false_alarm": alarm, "tokens": [tokens, 10], "eur": 0.01, "memory_cited": 0}


TREATED = {s["case"] for s in mv.manifest()["scenarios"] if s.get("role", "treated") == "treated"}


class GateTest(unittest.TestCase):
    def write(self, rows):
        folder = Path(tempfile.mkdtemp()) / "MV-gate"
        folder.mkdir()
        for r in rows:
            (folder / f"{r['id']}.json").write_text(json.dumps(r))
        return folder

    def test_helps_when_memory_removes_the_repeated_decisions(self):
        rows = [row(i["case"], i["repeat"], i["arm"], i["arm"] == "off") for i in mv.schedule()]
        s = mv.summarize(self.write(rows))
        self.assertEqual(s["verdict"], "memory_helps", s.get("checks"))
        self.assertLess(s["against_on_minus_off"]["sign_flip_p"], 0.05)
        self.assertIsNotNone(s["arms"]["all"]["on"]["leak_caught"]["ci95"])  # every share with its interval

    def test_does_not_help_when_nothing_changes_or_tokens_grow(self):
        same = [row(i["case"], i["repeat"], i["arm"], True) for i in mv.schedule()]
        self.assertEqual(mv.summarize(self.write(same))["verdict"], "memory_does_not_help")
        costly = [row(i["case"], i["repeat"], i["arm"], i["arm"] == "off", tokens=2000 if i["arm"] == "on" else 1000) for i in mv.schedule()]
        s = mv.summarize(self.write(costly))
        self.assertEqual((s["verdict"], s["checks"]["tokens_bounded"]), ("memory_does_not_help", False))

    def test_four_cases_better_is_not_enough(self):
        better = sorted(TREATED)[:4]
        rows = [row(i["case"], i["repeat"], i["arm"], i["arm"] == "off" and i["case"] in better) for i in mv.schedule()]
        s = mv.summarize(self.write(rows))
        self.assertEqual(s["verdict"], "memory_does_not_help")
        self.assertAlmostEqual(s["against_on_minus_off"]["sign_flip_p"], 1 / 16)
        self.assertEqual(mv.sign_flip_p([-1, -1, -1, -1, -1, 0, 0, 0]), 1 / 32)

    def test_memory_that_breaks_a_control_is_refused(self):
        rows = [row(i["case"], i["repeat"], i["arm"], i["arm"] == "off" and i["case"] in TREATED,
                    leak=not (i["arm"] == "on" and i["case"] not in TREATED)) for i in mv.schedule()]
        s = mv.summarize(self.write(rows))
        self.assertEqual((s["verdict"], s["checks"]["controls_keep_their_traps"]), ("memory_does_not_help", False))

    def test_a_run_is_stored_once(self):
        rows = [row(i["case"], i["repeat"], i["arm"], i["arm"] == "off") for i in mv.schedule()]
        folder = self.write(rows)
        (folder / "run.json").write_text(json.dumps({**mv.setup(), "model": "m"}))
        results = Path(tempfile.mkdtemp())
        with mock.patch.object(mv, "RESULTS", results):
            first = mv.store_result(folder, mv.summarize(folder))
            self.assertIn("thinking", json.loads(first.read_text())["setup"])
            with self.assertRaises(SystemExit):
                mv.store_result(folder, mv.summarize(folder))

    def test_no_verdict_until_every_session_ran(self):
        rows = [row(i["case"], i["repeat"], i["arm"], False) for i in mv.schedule()][:-1]
        self.assertEqual(mv.summarize(self.write(rows))["verdict"], "incomplete")


class CommandTest(unittest.TestCase):
    def test_nothing_is_sent_without_yes(self):
        out, err = io.StringIO(), io.StringIO()
        folder = Path(tempfile.mkdtemp()) / "MV-cli"
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch.dict(os.environ, KEYED):
            code = mv.main(["--run", str(folder)], gateway=gateway())
        self.assertEqual(code, 2)
        self.assertIn("72 sessions", out.getvalue())
        self.assertIn("--yes", err.getvalue())


if __name__ == "__main__":
    unittest.main()

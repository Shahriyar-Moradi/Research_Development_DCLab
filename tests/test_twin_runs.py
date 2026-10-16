"""Twin runs on main (package 25.2, from A6.6): the current system and a scripted "Claude" on the same sample, at the
same time, in two separate workspace folders; everything each did is on disk (its ledger and session record among it);
the comparison and the twin steps read only those files; nothing is sent without a key or under DCLAB_NO_LIVE_MODELS;
the request cap stops the AI-first lane; a run never touches the main workspace."""

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

try:
    import pandas  # noqa: F401

    from dclab_rnd import twin  # noqa: E402
    from dclab_rnd.twin import claude  # noqa: E402
    MISSING = None
except ImportError as error:  # pragma: no cover - .venv-agent has no pandas
    MISSING = f"the studio needs pandas: {error}"

MOMENT = "At the end of the examination, after every listed test, before the diagnosis is confirmed."


def reply(content=None, calls=(), tokens=(120, 20)):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": content, "finish_reason": "tool_calls" if calls else "stop", "tool_calls": calls,
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
            "assistant_message": {"role": "assistant", "content": content,
                                  "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class ScriptedClaude:
    """A transport in the ChatClient's shape that plays an AI-first agent: it reads the project id from the task and
    drives the real tools in its own order (not the standard plan's), overriding the models stage's choice."""

    model, base_url = "claude-scripted", "https://api.anthropic.com"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, max_tokens=None, **options):
        self.calls += 1
        if not tools:  # a role, a stage note or the leakage reviewer asked a plain question
            return reply("No change: the rules' reading stands.", tokens=(80, 12))
        text = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        pid = re.search(r"existing project (\w{6,})", text).group(1)
        n = sum(1 for m in messages if m.get("role") == "assistant")
        script = [
            [("describe_data", {"project_id": pid})],
            [("understand_problem", {"project_id": pid})],
            [("propose_solution", {"project_id": pid, "target": "target"})],
            [("set_solution", {"project_id": pid, "target": "target", "task": "binary", "prediction_moment": MOMENT, "forbidden": [], "identifiers": [],
                               "metric": "roc_auc"})],
            [("run_stage", {"project_id": pid, "stage": "data"})],
            [("approve_stage", {"project_id": pid, "stage": "data"})],
            [("run_stage", {"project_id": pid, "stage": "leakage"})],
            [("approve_stage", {"project_id": pid, "stage": "leakage"})],
            [("run_stage", {"project_id": pid, "stage": "features"})],
            [("approve_stage", {"project_id": pid, "stage": "features"})],
            [("run_stage", {"project_id": pid, "stage": "models"})],
            [("get_results", {"project_id": pid, "stage": "models"})],
        ]
        said = f"Step {n + 1} of my plan."
        if n < len(script):
            return reply(said, script[n])
        if n == len(script):  # the AI-first agent prefers the runner-up family: the difference the comparison must show
            results = next((m for m in reversed(messages) if m.get("role") == "tool"), {})
            content = str(results.get("content") or "")
            options = re.findall(r'"model": "([a-z_]+)"', content)
            selected = re.search(r'"selected": "([a-z_]+)"', content)
            other = next((o for o in options if o != (selected.group(1) if selected else None)), None)
            return reply(said, [("approve_stage", {"project_id": pid, "stage": "models", **({"choice": other} if other else {})})])
        if n == len(script) + 1:
            return reply(said, [("run_stage", {"project_id": pid, "stage": "final"})])
        if n == len(script) + 2:
            return reply(said, [("approve_stage", {"project_id": pid, "stage": "final"})])
        return reply(said, [("finish", {"report": "AI-first run: solution set from the brief, five stages run, the runner-up family chosen."})])


@unittest.skipIf(MISSING, MISSING)
class TwinRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd import experience, ledger
        from dclab_rnd.models import gateway, open_usage

        cls.tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls.tmp.name) / "workspace"
        cls.home.mkdir(parents=True)
        cls.transport = ScriptedClaude()
        # the main workspace's own log, ledger and gateway are installed, as in a server: a lane must never reach them
        experience.install(experience.open_log(cls.home))
        ledger.install(ledger.open_ledger(cls.home))
        gateway.install(gateway.Gateway(open_usage(cls.home)))
        try:
            with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
                cls.manifest = twin.run("sample:heart_disease", home=cls.home, transport=cls.transport, max_requests=80, max_steps=24, max_minutes=10,
                                        run_id="test-run", blind=True, log=lambda *a: None)
        finally:
            experience.install(None)
            ledger.install(None)
            gateway.install(None)
        cls.root = cls.home / "twin" / "runs" / "test-run"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_lanes_ran_in_their_own_folders_and_the_main_workspace_is_untouched(self):
        m = self.manifest
        self.assertEqual((m["lanes"]["workflow"]["status"], m["lanes"]["ai_first"]["status"]), ("completed", "completed"), m)
        for lane in ("workflow", "ai_first"):
            for path in ("projects", "intern", "ledger/decisions", "experience"):
                self.assertTrue((self.root / lane / path).is_dir(), f"{lane}/{path}")
            for name in ("audit.jsonl", "agent_steps.jsonl", "summary.json"):
                self.assertTrue((self.root / lane / name).is_file(), f"{lane}/{name}")
        self.assertNotEqual(m["lanes"]["workflow"]["project_id"], m["lanes"]["ai_first"]["project_id"])
        for name in ("projects", "audit.jsonl", "ledger", "model_requests.jsonl"):
            self.assertFalse((self.home / name).exists(), name)  # nothing in the main workspace
        main = [json.loads(line) for p in sorted((self.home / "experience").rglob("*.jsonl")) for line in p.read_text().splitlines()]
        self.assertEqual({e["kind"] for e in main}, {"run.recorded"}, main)  # the main record holds the run itself, never a lane's events
        lane_events = [json.loads(line) for p in sorted((self.root / "ai_first" / "experience").rglob("*.jsonl")) for line in p.read_text().splitlines()]
        kinds = {e["kind"] for e in lane_events}
        self.assertIn("data.uploaded", kinds)  # the seed's own events are the lane's
        self.assertIn("model.request", kinds)  # every request of the AI-first lane is in its session record
        for name in ("run.json", "README.md", "compare.json", "COMPARE.md", "steps.jsonl"):
            self.assertTrue((self.root / name).is_file(), name)

    def test_the_ai_first_lane_captured_every_exchange_and_the_workflow_lane_sent_nothing(self):
        calls = [json.loads(line) for line in (self.root / "ai_first" / "transcripts" / "model_calls.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(calls), 12)
        first = next(c for c in calls if c.get("purpose") == "intern")
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertIn("tool_calls", first["answer"])
        usage = [json.loads(line) for line in (self.root / "ai_first" / "model_requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(usage), sum(1 for c in calls if "check" not in c and c.get("purpose") != "monitor"))
        self.assertFalse((self.root / "workflow" / "transcripts").exists())
        self.assertFalse((self.root / "workflow" / "model_requests.jsonl").exists())
        tool_messages = [m for c in calls if "messages" in c for m in c["messages"] if m.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertTrue(all("\"examples\"" not in str(m.get("content")) for m in tool_messages))  # cell examples scrubbed from the transcript
        self.assertEqual(self.manifest["model_usage"]["requests"], len(usage))

    def test_the_comparison_shows_the_difference_the_agent_made(self):
        c = json.loads((self.root / "compare.json").read_text())
        self.assertEqual(c["status"], {"workflow": "completed", "ai_first": "completed"})
        self.assertIn("models", c["stage_differences"], c["stages"])
        models = next(r for r in c["stages"] if r["stage"] == "models")
        self.assertTrue(models["ai_first"]["overridden"])
        self.assertFalse(models["workflow"]["overridden"])
        self.assertIsNotNone(c["holdout"])
        self.assertEqual(c["tool_calls"]["ai_first"][0], "describe_data")
        self.assertNotIn("understand_problem", c["tool_calls"]["workflow"])  # the standard plan never asks the brief
        self.assertEqual(c["moves"]["workflow"]["refused"], 0)
        md = (self.root / "COMPARE.md").read_text()
        self.assertIn("| models |", md)
        self.assertIn("## Against the planted truth", md)

    def test_the_run_is_a_session_of_the_twin_agent(self):
        main = [json.loads(line) for p in sorted((self.home / "experience").rglob("*.jsonl")) for line in p.read_text().splitlines()]
        recorded = [e for e in main if e["kind"] == "run.recorded"]
        self.assertEqual(len(recorded), 1)
        self.assertIn("twin", json.dumps(recorded[0]))

    def test_the_steps_are_twin_steps_judged_against_the_samples_solution(self):
        from dclab_core.twin import TwinStep

        steps = {s.step: s for s in (TwinStep.read(json.loads(line)) for line in (self.root / "steps.jsonl").read_text().splitlines())}
        self.assertEqual(set(steps), {"solution.columns", "stage.data", "stage.features", "stage.models"})
        columns = steps["solution.columns"]
        self.assertEqual((columns.truth.source, columns.truth.value["target"]), ("planted", "target"))
        self.assertEqual(columns.verdict.winner, "both")
        self.assertEqual(next(d for d in steps["stage.models"].differences if d.field == "family").kind, "differs")
        self.assertIsNone(steps["stage.models"].truth)
        self.assertEqual(columns.system.input_fingerprint, columns.ai_first.input_fingerprint)

    def test_each_lane_has_its_decisions_in_its_own_ledger(self):
        for lane in ("workflow", "ai_first"):
            rows = [json.loads(line) for p in sorted((self.root / lane / "ledger" / "decisions").rglob("*.jsonl")) for line in p.read_text().splitlines()]
            points = {r["point"] for r in rows}
            self.assertIn("model.family", points, lane)
            self.assertIn("split.strategy", points, lane)
        summary = json.loads((self.root / "ai_first" / "summary.json").read_text())
        self.assertTrue(summary["rule_answers"])
        self.assertGreater(summary["counts"]["ledger_decisions"], 0)
        self.assertIsNotNone(summary["split_used"])

    def test_the_workflow_lane_makes_the_choices_a_plain_run_of_the_current_system_makes(self):
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.twin.lane import Lane, TASK

        with tempfile.TemporaryDirectory() as tmp, fast_profile():
            plain = Lane("plain", Path(tmp) / "plain", claude.NoModels())
            plain.seed_sample("heart_disease", TASK[:300], blind=True)  # as the run above
            again = plain.run_project(TASK, None, 24, 10)
        lane = json.loads((self.root / "workflow" / "summary.json").read_text())
        self.assertEqual({k: v["chosen"] for k, v in again["stages"].items()}, {k: v["chosen"] for k, v in lane["stages"].items()})
        self.assertEqual(again["solution"], lane["solution"])
        for key in ("holdout", "holdout_ci", "cv_mean"):
            self.assertEqual(again["stages"]["final"][key], lane["stages"]["final"][key], key)
        self.assertEqual(again["split_used"], lane["split_used"])

    def test_compare_can_be_rebuilt_from_the_files_alone(self):
        before = json.loads((self.root / "compare.json").read_text())
        steps_before = (self.root / "steps.jsonl").read_text().count("\n")
        again = twin.compare_run(self.root)
        self.assertEqual(again["stage_differences"], before["stage_differences"])
        self.assertEqual(again["solution"], before["solution"])
        self.assertEqual((self.root / "steps.jsonl").read_text().count("\n"), steps_before)  # rebuilt, not appended
        runs = twin.list_runs(self.home)
        self.assertEqual([r["run_id"] for r in runs], ["test-run"])


@unittest.skipIf(MISSING, MISSING)
class GuardTests(unittest.TestCase):
    def test_nothing_is_sent_under_no_live_models_or_without_a_key(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            with self.assertRaises(RuntimeError):
                twin.run("sample:heart_disease", home=Path(tmp), log=lambda *a: None)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                claude.transport()

    def test_the_request_cap_stops_the_ai_first_lane_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = twin.run("sample:heart_disease", home=Path(tmp), transport=ScriptedClaude(), max_requests=3, max_steps=24, max_minutes=10,
                                run_id="capped", design=False, log=lambda *a: None)
            self.assertEqual(manifest["lanes"]["workflow"]["status"], "completed")
            self.assertEqual(manifest["lanes"]["ai_first"]["status"], "stopped")
            self.assertIn("request cap", manifest["lanes"]["ai_first"]["error"])
            self.assertLessEqual(manifest["model_usage"]["requests"], 3)
            self.assertNotIn("truth", manifest)  # not blind: the known solution was given to both lanes, nothing is judged by it
            self.assertTrue(any("not --blind" in n for n in manifest.get("notes", [])))

    def test_the_euro_cap_is_reserved_before_a_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            prices = Path(tmp) / "prices.json"
            # an output price so high that one request's upper bound (its whole output limit) is over the cap: nothing may be sent
            prices.write_text(json.dumps({"claude-scripted": {"input": 30.0, "output": 1500.0, "as_of": "2026-10-16", "source": "test"}}))
            with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(prices)}):
                manifest = twin.run("sample:heart_disease", home=Path(tmp) / "ws", transport=ScriptedClaude(), max_requests=80, cap_eur=0.2,
                                    max_steps=24, max_minutes=10, run_id="euro", design=False, log=lambda *a: None)
            self.assertTrue(manifest["priced"])
            self.assertEqual(manifest["lanes"]["ai_first"]["status"], "stopped")
            self.assertEqual(manifest["model_usage"]["requests"], 0)  # reserved before the request: spent 0, yet refused
            self.assertLessEqual(manifest["model_usage"]["eur"], 0.2)
            calls = (Path(tmp) / "ws" / "twin" / "runs" / "euro" / "ai_first" / "transcripts" / "model_calls.jsonl").read_text()
            self.assertIn("euro cap would be passed", calls)
        self.assertEqual(twin.DEFAULT_CAP_EUR, 1.5)

    def test_a_planted_truth_leaves_out_a_time_column_the_solution_forbids(self):
        from dclab_rnd.twin import compare

        suggestion = {"target": "Class", "forbidden": [{"column": "Time"}, {"column": "settled_amount"}], "identifiers": ["id"], "time_column": "Time"}
        with mock.patch("dclab_rnd.studio.data.load_sample", return_value=(None, suggestion)):
            truth = compare.planted("sample", "credit_card_fraud")
        self.assertEqual(truth, {"solution.columns": {"target": "Class", "leakage": ["settled_amount"]}})
        self.assertIsNone(compare.planted("project", "p1"))

    def test_two_lanes_given_different_tables_are_refused(self):
        from dclab_rnd.twin import compare

        a = {"status": "completed", "data_sha256": "a" * 64, "given_solution": False, "solution": {"target": "y"}}
        b = {**a, "data_sha256": "b" * 64}
        with self.assertRaises(ValueError):
            compare.steps("r", a, b)
        self.assertEqual(len(compare.steps("r", a, dict(a))), 1)

    def test_a_private_scope_asks_no_model_through_a_lane(self):
        from dclab_rnd.models.gateway import private_scope

        gw = claude.TwinGateway(ScriptedClaude(), claude.Transcript(Path(tempfile.mkdtemp()) / "t.jsonl"))
        with private_scope(True):
            self.assertIsNone(gw.client("intern"))
        self.assertIsNotNone(gw.client("intern"))
        self.assertIsNone(gw.refusal(purpose="campaign"))

    def test_a_seed_is_one_of_three_kinds(self):
        with self.assertRaises(ValueError):
            twin.run("benchmark:x", transport=ScriptedClaude(), log=lambda *a: None)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(MISSING, MISSING)
class ThinkingAndCapsTests(unittest.TestCase):
    """25.3 review: the fallback when the API refuses thinking, the reservation on the limit sent, the month's twin runs."""

    def gateway(self, transport, cap=None):
        tmp = Path(tempfile.mkdtemp())
        return claude.TwinGateway(transport, claude.Transcript(tmp / "t.jsonl"), cap_eur=cap), tmp

    def test_thinking_refused_once_is_asked_again_without_it_and_never_again(self):
        from dclab_rnd.models.client import THINKING_REFUSED

        class Refuses:
            model, base_url, thinking = "claude-scripted", "https://api.anthropic.com", "adaptive"

            def __init__(self):
                self.sent = []

            def effective_max_tokens(self, max_tokens, options=None):
                return (max_tokens or 0) + (0 if (options or {}).get("thinking") == "off" else 8000)

            def complete(self, messages, tools=None, max_tokens=None, **options):
                self.sent.append(options.get("thinking"))
                if options.get("thinking") != "off":
                    raise RuntimeError(f"BadRequestError: {THINKING_REFUSED}")
                return reply("ok")

        t = Refuses()
        gw, tmp = self.gateway(t)
        c = gw.client("intern")
        self.assertEqual(c.complete([{"role": "user", "content": "go"}])["content"], "ok")
        c.complete([{"role": "user", "content": "go"}])
        self.assertEqual(t.sent, [None, "off", "off"])  # asked with thinking once, then without it for the rest of the run
        self.assertEqual(gw.shared["thinking_refused"], THINKING_REFUSED)

        class KeyRefused(Refuses):
            def complete(self, messages, tools=None, max_tokens=None, **options):
                raise RuntimeError("AuthenticationError: the provider refused the key")

        gw2, _ = self.gateway(KeyRefused())
        with self.assertRaises(RuntimeError):
            gw2.client("intern").complete([{"role": "user", "content": "go"}])
        self.assertNotIn("thinking_refused", gw2.shared)  # any other failure leaves thinking on

    def test_the_reservation_counts_the_thinking_room(self):
        class Thinks:
            model, base_url, thinking = "claude-scripted", "https://api.anthropic.com", "adaptive"

            def effective_max_tokens(self, max_tokens, options=None):
                return (max_tokens or 0) + 8000

            def complete(self, messages, tools=None, max_tokens=None, **options):
                return reply("ok")

        with tempfile.TemporaryDirectory() as tmp:
            prices = Path(tmp) / "p.json"
            prices.write_text(json.dumps({"claude-scripted": {"input": 1.0, "output": 10.0, "as_of": "2026-10-16", "source": "test"}}))
            with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": str(prices)}):
                gw, _ = self.gateway(Thinks(), cap=0.05)  # 1800 tokens fit (0.018), 9800 do not (0.098)
                with self.assertRaises(RuntimeError):
                    gw.client("intern").complete([{"role": "user", "content": "go"}], None, 1800)

    def test_the_months_earlier_twin_runs_count_against_the_monthly_cap(self):
        from dclab_rnd.models.usage import FileUsage

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            usage = FileUsage(home / "twin" / "runs" / "earlier" / "ai_first" / "model_requests.jsonl")
            from dclab_rnd.models.gateway import Gateway

            with mock.patch.dict(os.environ, {"DCLAB_PRICES_FILE": ""}):
                Gateway(usage).record("intern", "claude-sonnet-5-5", {"input_tokens": 100000, "output_tokens": 10000}, 1.0, 1, "ok", None, None, None,
                                      base_url="https://api.anthropic.com", tier="strong")
            self.assertGreater(twin._month_spent(home), 0.2)



@unittest.skipIf(MISSING, MISSING)
class ProjectTwinTest(unittest.TestCase):
    """25.6: a twin run started from a project, as a job: the project's records unchanged, the run listed on its page."""

    def test_a_job_runs_both_lanes_on_the_projects_table_and_leaves_the_project_as_it_was(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.jobs import handlers
        from dclab_rnd.studio import data as sd
        from dclab_rnd.twin import home as twin_home

        home = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            client = TestClient(create_app(home))
        s = client.app.state.services
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "Heart", "goal": "Predict heart disease"}, headers=h).json()["id"]
        sd.use_sample(s.projects, pid, "heart_disease")
        self.assertEqual(client.post(f"/api/projects/{pid}/twin", json={}, headers=h).status_code, 409)  # the owner's switch is off
        twin_home.set_settings(s.store.home, {"on": True})
        before = json.dumps(s.projects.get(pid), sort_keys=True, default=str)
        env = {k: v for k, v in os.environ.items() if k != "DCLAB_NO_LIVE_MODELS"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("dclab_rnd.twin.claude.transport", return_value=ScriptedClaude()):
            handlers._twin_run(s, {"project_id": pid, "run_id": "proj-run", "blind": True})
        after = s.projects.get(pid)
        self.assertEqual({k: v for k, v in json.loads(before).items() if k not in ("updated", "activity")},
                         {k: v for k, v in json.loads(json.dumps(after, sort_keys=True, default=str)).items() if k not in ("updated", "activity")})
        runs = client.get(f"/api/projects/{pid}/twin").json()["runs"]
        self.assertEqual([r["run_id"] for r in runs], ["proj-run"])
        self.assertEqual(runs[0]["status"], {"workflow": "completed", "ai_first": "completed"})
        self.assertIn("# Twin run proj-run", runs[0]["compare_md"])

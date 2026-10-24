"""H3 of the harness that learns, first part: the intern as the solver of a challenge pack, and the pilot.

A scripted intern that states what the key would state solves a learning variant: the final stage runs, the grade
passes, and the attempt is kept whole (its project, its session, its grade) under the workspace's learning attempts. An
exam variant (check or sealed) leaves nothing behind. The person's request never names a trap. The local gateway refuses
a model that is not on this machine. The pilot keeps one row per homework and fails its fixed gate when the model does
not finish the work.
"""

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

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
    import pandas  # noqa: F401
except ImportError as error:
    MISSING = str(error)

PID = re.compile(r"existing project (\S+?)[ .(]")


class KeyedModel:
    """Says what the key says, as a chat model calling tools would: the solution, every stage, the report."""

    model = "scripted"

    def __init__(self, pack, key=None):
        self.pack, self.key, self.turn = pack, key, 0

    def complete(self, messages, tools, **_):
        self.turn += 1
        if self.key is None or self.turn > 3:
            return {"content": "Done.", "tool_calls": [], "finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1},
                    "assistant_message": {"role": "assistant", "content": "Done."}}
        from dclab_core import variants as V

        pid = PID.search(next(m["content"] for m in messages if m["role"] == "user")).group(1)
        b = self.pack.brief
        script = [("set_solution", {"project_id": pid, "target": b.target, "task": b.task, "prediction_moment": b.moment, **V.reference(self.key)}),
                  ("run_all", {"project_id": pid}), ("finish", {"report": "Built; the holdout score is in the project."})]
        name, args = script[self.turn - 1]
        call = {"id": f"c{self.turn}", "name": name, "arguments": args}
        return {"content": "", "tool_calls": [call], "finish_reason": "tool_calls", "usage": {"input_tokens": 10, "output_tokens": 5},
                "reasoning": f"step {self.turn}: {name}",
                "assistant_message": {"role": "assistant", "content": "", "tool_calls": [{"id": call["id"], "type": "function",
                                                                                          "function": {"name": name, "arguments": json.dumps(args)}}]}}


def variant(root, partition):
    from dclab_core import variants as V
    from dclab_rnd.learning.packs import materialize, public_dir, read_key

    seed = V.seeds("groups", partition, 1)[0]
    v = materialize(Path(root), "groups", seed)
    return v.pack, read_key(Path(root), "groups", v.pack.id), public_dir(Path(root), "groups", v.pack.id)


@unittest.skipIf(MISSING, MISSING)
class SolverTests(unittest.TestCase):
    def test_a_keyed_intern_solves_a_learning_variant_and_the_attempt_is_kept_whole(self):
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.learning.grading import grade_run
        from dclab_rnd.learning.solver import intern_attempt, keep_grade, summary_of

        with tempfile.TemporaryDirectory() as tmp:
            pack, key, folder = variant(Path(tmp) / "packs", "learning")
            home = Path(tmp) / "home"
            with fast_profile():
                out = intern_attempt(folder, KeyedModel(pack, key), key, home=home, label="t")
            self.assertTrue(out["finished"], summary_of(out["session"]))
            grade, observed = grade_run(key, out["run"])
            self.assertEqual(grade.verdict, "pass", grade.codes)
            kept = Path(out["kept"])
            self.assertEqual(kept.parent.parent, home / "learning_attempts")
            self.assertTrue(kept.name.endswith(f"{pack.id}-t"))
            self.assertTrue(any((kept / "projects").iterdir()) and any((kept / "intern").rglob("*.json")))
            offered = json.loads((kept / "tools.json").read_text())
            names = [t.get("name") or t.get("function", {}).get("name") for t in offered["tools"]]
            self.assertTrue({"set_solution", "write_plan", "finish"} <= set(names), names)  # exactly what the loop offers, its own tools too
            keep_grade(out["kept"], grade, observed)
            written = json.loads((kept / "grade.json").read_text())
            self.assertEqual(written["grade"]["verdict"], "pass")
            self.assertNotIn("seed", json.dumps(written["grade"]))  # the key never sits beside the attempt
            s = summary_of(out["session"])
            self.assertEqual(s["tools"][:2], ["set_solution", "run_all"])
            self.assertGreaterEqual(s["thoughts"]["recorded"], 1)

    def test_an_exam_variant_leaves_nothing_behind(self):
        from dclab_rnd.learning.solver import intern_attempt

        with tempfile.TemporaryDirectory() as tmp:
            pack, key, folder = variant(Path(tmp) / "packs", "check")
            home = Path(tmp) / "home"
            out = intern_attempt(folder, KeyedModel(pack), key, home=home)
            self.assertIsNone(out["kept"])
            self.assertFalse(out["finished"])
            self.assertFalse((home / "learning_attempts").exists())

    def test_an_exam_attempt_writes_no_event_and_no_decision_into_the_workspace(self):
        from dclab_rnd import experience, ledger
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.learning.solver import intern_attempt

        with tempfile.TemporaryDirectory() as tmp:
            pack, key, folder = variant(Path(tmp) / "packs", "check")
            home = Path(tmp) / "home"
            home.mkdir()
            log, book = experience.open_log(home), ledger.open_ledger(home)
            experience.install(log)
            ledger.install(book)
            try:
                with fast_profile(), mock.patch.dict(os.environ, {"DCLAB_LOG_PROMPTS": "1"}):
                    seen = {}
                    model = KeyedModel(pack, key)
                    original = model.complete

                    def complete(messages, tools, **kw):
                        seen.setdefault("prompts", os.environ.get("DCLAB_LOG_PROMPTS"))
                        return original(messages, tools, **kw)

                    model.complete = complete
                    out = intern_attempt(folder, model, key, home=home)
                    self.assertEqual(os.environ.get("DCLAB_LOG_PROMPTS"), "1")  # the switch comes back
                self.assertTrue(out["finished"])
                self.assertEqual(seen["prompts"], "0")  # inside an exam a model exchange keeps its shape only
                self.assertIs(experience._INSTALLED.get("log"), log)
                self.assertIs(ledger._INSTALLED.get("ledger"), book)
                written = [p for p in home.rglob("*") if p.is_file() and p.stat().st_size]
                text = " ".join(p.read_text(errors="ignore") for p in written)
                for name in [*key.leaks, key.row_key]:
                    self.assertNotIn(name, text)
                self.assertFalse([p for p in written if "experience" in p.parts or "ledger" in p.parts], written)
            finally:
                experience.install(None)
                ledger.install(None)

    def test_a_kept_attempt_records_its_events_and_decisions_beside_its_projects(self):
        from dclab_rnd import experience, ledger
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.learning.solver import intern_attempt

        with tempfile.TemporaryDirectory() as tmp:
            pack, key, folder = variant(Path(tmp) / "packs", "learning")
            home = Path(tmp) / "home"
            with fast_profile():
                out = intern_attempt(folder, KeyedModel(pack, key), key, home=home)
            kept = Path(out["kept"])
            self.assertTrue(any((kept / "experience").rglob("*.jsonl")))
            self.assertTrue(any((kept / "ledger").rglob("*.jsonl")))
            self.assertIsNone(experience._INSTALLED.get("log"))
            self.assertIsNone(ledger._INSTALLED.get("ledger"))

    def test_the_request_names_no_trap(self):
        from dclab_rnd.learning.solver import task_of

        with tempfile.TemporaryDirectory() as tmp:
            pack, key, _ = variant(tmp, "learning")
            text = task_of(pack)
            self.assertIn(pack.brief.target, text)
            for name in [*key.leaks, key.group_column, key.row_key, *key.decoys]:
                if name:
                    self.assertNotRegex(text, rf"\b{re.escape(name)}\b")

    def test_the_local_gateway_refuses_a_model_off_this_machine(self):
        from dclab_rnd.learning.solver import NotLocal, local_gateway

        with mock.patch.dict(os.environ, {}, clear=False):
            with self.assertRaises(NotLocal):
                local_gateway("qwen", "https://api.example.com/v1")

    def test_the_local_gateway_serves_the_intern_locally_and_reads_the_reasoning(self):
        from dclab_rnd.learning.solver import local_gateway
        from dclab_rnd.models import settings

        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}, clear=False):
            gateway = local_gateway("qwen3.5:4b-q4_K_M")
            tier = settings.tier(gateway.route("intern")[0])
            self.assertTrue(settings.is_local(tier["base_url"]))
            self.assertEqual(tier["model"], "qwen3.5:4b-q4_K_M")
            self.assertEqual(tier["reasoning_field"], "reasoning")


@unittest.skipIf(MISSING, MISSING)
class DraftSpendTests(unittest.TestCase):
    def test_the_reported_spend_is_every_challenges_not_the_last_ones(self):
        """Found live 2026-10-24: four reflector answers cost about EUR 0.017 and the command reported EUR 0.0042 (the
        client's counter holds the last challenge only); a passed challenge asks nothing and adds nothing."""
        from dclab_rnd.learning.drafting import draft_lessons

        class Priced:
            model = "scripted-reflector"

            def __init__(self):
                self.spent_eur, self.asked = 0.0, 0

            def complete(self, messages, tools=None, **_):
                self.asked += 1
                self.spent_eur += 0.01
                return {"content": '{"lessons": []}', "usage": {"input_tokens": 1, "output_tokens": 1}}

        client = Priced()
        with tempfile.TemporaryDirectory() as tmp:
            out = draft_lessons(Path(tmp), "groups", n=4, client=client, log=lambda *_: None)
        self.assertGreater(client.asked, 1)
        self.assertAlmostEqual(out["spent_eur"], 0.01 * client.asked, places=6)


@unittest.skipIf(MISSING, MISSING)
class ContextTests(unittest.TestCase):
    """Measured 2026-10-23: Ollama served the model with 4,096 tokens and cut every prompt to 2,050; the intern's first
    message is about 5,000, so the model never saw its task. A window under the intern's need is refused before a run."""

    def served(self, window):
        body = json.dumps({"models": [{"name": "qwen3.5:4b-32k", "context_length": window}]}).encode()

        class Reply(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return mock.patch("urllib.request.urlopen", return_value=Reply(body))

    def test_a_small_window_is_refused(self):
        from dclab_rnd.learning.solver import ContextTooSmall, check_context

        with self.served(4096), self.assertRaises(ContextTooSmall):
            check_context("qwen3.5:4b-32k")

    def test_a_large_window_passes_and_an_unknown_one_is_not_refused(self):
        from dclab_rnd.learning.solver import check_context

        with self.served(32768):
            self.assertEqual(check_context("qwen3.5:4b-32k"), 32768)
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no such route")):
            self.assertIsNone(check_context("qwen3.5:4b-32k"))

    def test_the_intern_first_message_needs_more_than_the_default_window(self):
        from dclab_rnd.intern.loop import POLICY
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.learning.solver import MIN_CONTEXT
        from dclab_rnd.studio import ProjectStore

        with tempfile.TemporaryDirectory() as tmp:
            chars = len(POLICY) + len(json.dumps(Toolbox(ProjectStore(Path(tmp))).schemas()))
        self.assertGreater(chars / 4, 4096 / 2)  # what the default window keeps of a prompt: less than the first message
        self.assertGreater(MIN_CONTEXT, chars / 3)


@unittest.skipIf(MISSING, MISSING)
class PilotTests(unittest.TestCase):
    def test_a_server_that_never_answers_makes_the_run_invalid_not_a_fail(self):
        from dclab_rnd.learning.pilot import run_pilot

        class Down:
            model = "down"

            def complete(self, messages, tools, **_):
                raise ConnectionError("connection refused")

        with tempfile.TemporaryDirectory() as tmp:
            result = run_pilot(Path(tmp), Down(), "down", n=2, log=lambda *_: None)
        self.assertTrue(result["verdict"].startswith("invalid"), result["verdict"])
        self.assertEqual(result["summary"]["broke"], 2)

    def test_a_paused_pilot_stops_between_homeworks_and_goes_on_where_it_stopped(self):
        from dclab_rnd.learning.pilot import pause_file, run_pilot

        class Lazy(KeyedModel):
            def __init__(self):
                super().__init__(None)

        quiet = lambda *_: None  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            first = run_pilot(home, Lazy(), "scripted", n=1, log=quiet)
            self.assertEqual(first["status"], "completed")
            pause_file(home).write_text("pause")
            paused = run_pilot(home, Lazy(), "scripted", n=3, log=quiet)
            self.assertEqual(paused["status"], "paused")  # the done homework came back, the next was not started
            self.assertEqual([r.get("resumed") for r in paused["rows"]], [True])
            self.assertFalse(pause_file(home).exists())
            went_on = run_pilot(home, Lazy(), "scripted", n=3, log=quiet)
            self.assertEqual(went_on["status"], "completed")
            self.assertEqual(len(went_on["rows"]), 3)
            self.assertEqual(went_on["rows"][0]["pack"], first["rows"][0]["pack"])  # the same attempt, found by its seed
            self.assertEqual(len(list((home / "learning_attempts").glob("*/*-pilot"))), 3)  # no homework done twice
            other = run_pilot(home, Lazy(), "another", n=1, log=quiet)
            self.assertEqual(len(list((home / "learning_attempts").glob("*/*-pilot"))), 4)  # another model does its own
            self.assertNotIn("resumed", other["rows"][0])

    def test_a_hosted_run_stops_before_a_homework_once_its_cap_is_spent(self):
        from dclab_rnd.learning.pilot import run_pilot

        class Paid(KeyedModel):  # answers in words, nothing done, and each answer costs EUR 0.02
            def __init__(self):
                super().__init__(None)
                self.spent_eur, self.run_limit_eur = 0.0, None

            def complete(self, messages, tools, **kw):
                self.spent_eur += 0.02
                return super().complete(messages, tools, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_pilot(Path(tmp), Paid(), "paid", n=3, cap_eur=0.01, log=lambda *_: None)
            self.assertEqual((result["status"], len(result["rows"])), ("paused", 1))  # the first spent the cap: no second
            self.assertGreater(result["rows"][0]["session"]["eur"], 0.01)

    def test_hosted_needs_yes_and_a_cap(self):
        from dclab_rnd.learning.__main__ import main

        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(main(["pilot", "--hosted"]), 2)
            self.assertEqual(main(["pilot", "--hosted", "--yes"]), 2)
            self.assertEqual(main(["probe", "--hosted", "--bullet", "B-0000000000", "--cap-eur", "1"]), 2)
        self.assertIn("--yes and --cap-eur", err.getvalue())

    def test_a_tagged_run_does_its_own_homeworks_and_keeps_them_apart(self):
        from dclab_rnd.learning.pilot import done_row, run_pilot

        class Lazy(KeyedModel):
            def __init__(self):
                super().__init__(None)

        quiet = lambda *_: None  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            first = run_pilot(home, Lazy(), "scripted", n=1, log=quiet)
            again = run_pilot(home, Lazy(), "scripted", n=1, log=quiet, tag="after-fix")
            self.assertNotIn("resumed", again["rows"][0])  # done anew, not taken from the untagged run
            self.assertEqual(again["tag"], "after-fix")
            seed = first["rows"][0]["seed"]
            self.assertIsNone(done_row(home, "groups", seed, "scripted", "other")["tag"] if done_row(home, "groups", seed, "scripted", "other") else None)
            self.assertEqual(done_row(home, "groups", seed, "scripted")["tag"], None)
            self.assertEqual(done_row(home, "groups", seed, "scripted", "after-fix")["tag"], "after-fix")

    def test_a_limit_does_that_many_new_homeworks_a_process_and_the_next_goes_on(self):
        from dclab_rnd.learning.pilot import run_pilot

        class Lazy(KeyedModel):
            def __init__(self):
                super().__init__(None)

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            first = run_pilot(home, Lazy(), "scripted", n=3, limit=1, log=lambda *_: None)
            self.assertEqual((first["status"], len(first["rows"])), ("paused", 1))
            second = run_pilot(home, Lazy(), "scripted", n=3, limit=1, log=lambda *_: None)
            self.assertEqual((second["status"], len(second["rows"])), ("paused", 2))
            last = run_pilot(home, Lazy(), "scripted", n=3, limit=1, log=lambda *_: None)
            self.assertEqual((last["status"], len(last["rows"])), ("completed", 3))
            self.assertEqual(len(list((home / "learning_attempts").glob("*/*-pilot"))), 3)

    def test_an_invalid_or_ungraded_attempt_is_done_again(self):
        from dclab_rnd.learning.pilot import done_row, run_pilot

        class Lazy(KeyedModel):
            def __init__(self):
                super().__init__(None)

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            row = run_pilot(home, Lazy(), "scripted", n=1, log=lambda *_: None)["rows"][0]
            folder, seed = home / row["kept"], row["seed"]
            self.assertIsNotNone(done_row(home, "groups", seed, "scripted"))
            self.assertIsNone(done_row(home, "groups", seed, "another"))
            (folder / "invalid.json").write_text("{}")
            self.assertIsNone(done_row(home, "groups", seed, "scripted"))
            (folder / "invalid.json").unlink()
            (folder / "grade.json").unlink()
            self.assertIsNone(done_row(home, "groups", seed, "scripted"))

    def test_an_attempt_kept_before_rows_were_saved_is_rebuilt_and_then_found(self):
        from dclab_rnd.learning.pilot import done_row, rebuild_row, run_pilot

        class Lazy(KeyedModel):
            def __init__(self):
                super().__init__(None)

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            row = run_pilot(home, Lazy(), "scripted", n=1, log=lambda *_: None)["rows"][0]
            folder = home / row["kept"]
            (folder / "pilot_row.json").unlink()
            self.assertIsNone(done_row(home, "groups", row["seed"], "scripted"))
            rebuilt = rebuild_row(folder, home, "groups", row["seed"])
            self.assertEqual((rebuilt["pack"], rebuilt["verdict"], rebuilt["codes"], rebuilt["finished"]),
                             (row["pack"], row["verdict"], row["codes"], row["finished"]))
            self.assertIsNotNone(done_row(home, "groups", row["seed"], "scripted"))

    def test_a_pilot_without_a_model_is_refused(self):
        from dclab_rnd.learning.pilot import run_pilot

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_pilot(Path(tmp), None, "none", n=1)

    def test_a_model_that_does_not_finish_fails_the_fixed_gate_with_a_row_per_homework(self):
        from dclab_rnd.learning.pilot import GATE, run_pilot

        class Lazy(KeyedModel):
            def __init__(self):
                super().__init__(None)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_pilot(Path(tmp), Lazy(), "scripted", n=2, log=lambda *_: None)
        self.assertEqual(result["kind"], "learning_pilot")
        self.assertEqual(result["gate"], GATE)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["summary"]["finished"], 0)
        self.assertTrue(result["verdict"].startswith("fail"))
        for row in result["rows"]:
            self.assertFalse(row["finished"])
            self.assertIsNotNone(row["kept"])  # a failed learning attempt is kept too: it is training data
            self.assertFalse(Path(row["kept"]).is_absolute())  # never a local path in committed evidence
            self.assertNotIn("seed_key", row)
        self.assertIn("generator", json.dumps(result["fingerprints"]))


if __name__ == "__main__":
    unittest.main()

"""Home's model benchmark from stored runs (2026-10-29): no model is called; a call belongs to a run by its model and
the run's time window, so two runs made side by side with different models stay apart; thinking is counted from the
kept exchanges, and a local model's is labelled as its whole reasoning, a Claude model's as the API's summary."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def result(run_id, model, local, at="2026-10-29T15:18:08+00:00", kept="agent_runs/x/dev-145744"):
    arm = {"cases": 2, "track": {"right": 2}, "track_accuracy_by_lang": {"en": 1.0}, "fields": {"right": 3, "wrong": 0, "missed": 1},
           "exit_right": 2, "traps": {"a": True, "b": False}, "uncited": 0}
    return {"id": run_id, "split": "dev", "at": at, "kept": kept, "spent_eur": 0.0 if local else 0.5,
            "plan": {"model": model, "local": local}, "summary": {"rules": arm, "model": arm, "plain": arm}}


class CompareTests(unittest.TestCase):
    def test_side_by_side_runs_stay_apart_and_thinking_is_labelled(self):
        from dclab_rnd.agent_eval.home import compare

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            results, home = tmp / "results", tmp / "home"
            results.mkdir()
            (results / "HOME-901_dev.json").write_text(json.dumps(result("HOME-901", "claude-sonnet-5-5", False)))
            (results / "HOME-902_dev.json").write_text(json.dumps(result("HOME-902", "qwen3.5:4b", True)))
            day = home / "model_exchanges" / "2026" / "10"
            day.mkdir(parents=True)
            usage, exchanges = [], []
            for model, seconds, thought in (("claude-sonnet-5-5", 4.0, "a summary"), ("qwen3.5:4b", 40.0, "the whole reasoning"),
                                            ("qwen3.5:4b", 60.0, "")):
                at = "2026-10-29T15:00:00+00:00"
                usage.append({"at": at, "purpose": "home_intent", "model": model, "seconds": seconds, "outcome": "ok", "output_tokens": 10, "cost_eur": 0.01})
                exchanges.append({"at": at, "purpose": "home_intent", "model": model, "thinking": [{"text": thought}] if thought else []})
            usage.append({"at": "2026-10-29T13:00:00+00:00", "purpose": "home_intent", "model": "qwen3.5:4b", "seconds": 1.0, "outcome": "ok"})  # before the run
            (home / "model_requests.jsonl").write_text("".join(json.dumps(u) + "\n" for u in usage))
            (day / "29.jsonl").write_text("".join(json.dumps(e) + "\n" for e in exchanges))
            with mock.patch.object(compare, "RESULTS", results), mock.patch.object(compare, "ROOT", tmp):
                claude, qwen = compare.row("HOME-901", home), compare.row("HOME-902", home, "Qwen")
                text = compare.table([claude, qwen])
        self.assertEqual(claude["calls"]["home_intent"]["calls"], 1)
        self.assertEqual((qwen["calls"]["home_intent"]["calls"], qwen["calls"]["home_intent"]["thought"]), (2, 1))
        self.assertEqual(qwen["calls"]["home_intent"]["median_s"], 50.0)
        self.assertEqual((claude["thinking_is"], qwen["thinking_is"]), ("the API's thinking summary", "the model's whole reasoning"))
        self.assertIn("| Home: traps caught | 1/2 | 1/2 |", text)
        self.assertIn("Qwen", text)

    def test_a_run_made_in_chunks_is_one_column_scored_together(self):
        from dclab_rnd.agent_eval.home import compare

        def case_row(case, track, trap=None):
            arm = {"track": track, "fields": {"measure": "right"}, "tracks_complete": None, "exit": "right", "questions": 1,
                   "quote_invalid": 0, "uncited": 0, "rows_exposed": 0, "phantom": 0, "destructive_unconfirmed": 0, "lost": 0,
                   "trap": True if trap else None}
            return {"case": case, "lang": "en", "trap": trap, "observed": {"rules": {"fields": {"measure": "x"}}},
                    "scores": {"rules": arm, "model": arm, "plain": arm}}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            results, home = tmp / "results", tmp / "home"
            results.mkdir()
            home.mkdir()
            for run_id, rows, at, kept in (("HOME-911", [case_row("d-a1", "right"), case_row("d-a2", "wrong")], "2026-10-29T16:00:00+00:00", "x/dev-155000"),
                                           ("HOME-912", [case_row("d-a3", "right", "rows_request")], "2026-10-29T17:00:00+00:00", "x/dev-165000")):
                r = result(run_id, "qwen3.5:4b", True, at=at, kept=kept)
                r["rows"] = rows
                (results / f"{run_id}_dev.json").write_text(json.dumps(r))
            (home / "model_requests.jsonl").write_text("".join(json.dumps(
                {"at": at, "purpose": "home_agent", "model": "qwen3.5:4b", "seconds": sec, "outcome": "ok"}) + "\n"
                for at, sec in (("2026-10-29T15:55:00+00:00", 10.0), ("2026-10-29T16:55:00+00:00", 30.0), ("2026-10-29T16:20:00+00:00", 99.0))))
            (results / "HOME-913_dev.json").write_text(json.dumps({**result("HOME-913", "claude-sonnet-5-5", False), "rows": [case_row("d-a1", "right")]}))
            with mock.patch.object(compare, "RESULTS", results), mock.patch.object(compare, "ROOT", tmp):
                joined = compare.row("HOME-911+HOME-912", home, "Qwen")
                with self.assertRaises(SystemExit):  # two models in one column
                    compare.row("HOME-911+HOME-913", home)
        self.assertEqual((joined["arms"]["model"]["kind_right"], joined["arms"]["model"]["cases"], joined["arms"]["model"]["traps"]), (2, 3, "1/1"))
        self.assertEqual((joined["calls"]["home_agent"]["calls"], joined["calls"]["home_agent"]["median_s"]), (2, 20.0))  # the call between chunks is no one's

    def test_an_unknown_run_is_refused(self):
        from dclab_rnd.agent_eval.home import compare

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(compare, "RESULTS", Path(tmp)), mock.patch.object(compare, "ROOT", Path(tmp)):
            with self.assertRaises(SystemExit):
                compare.row("HOME-999", Path(tmp))


if __name__ == "__main__":
    unittest.main()

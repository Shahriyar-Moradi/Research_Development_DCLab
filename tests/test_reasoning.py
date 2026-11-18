"""Package 27.9: verified reasoning traces, the `<think>` the solver model learns, written from DCLab's own records."""

import json
import tempfile
import unittest
from pathlib import Path

from dclab_core import lake as L
from dclab_core import reasoning as R

GIVEN = {"columns": [{"column": "subscriber_id", "dtype": "integer", "uniqueness": "all", "name_tokens": ["subscriber", "id"]},
                     {"column": "months_billed", "dtype": "integer", "uniqueness": "under_1pct"},
                     {"column": "plan", "dtype": "string", "uniqueness": "under_1pct"}],
         "outcome": "lifetime_months", "prediction_moment": "At signup, before the first bill",
         "train_only_signals": [{"column": "months_billed", "score": 0.998}, {"column": "plan", "score": 0.21}]}
ANSWER = {"forbidden": [{"column": "months_billed", "reason": "single column tracks the target (|Spearman| 0.998)"}], "identifiers": ["subscriber_id"]}


def example(family, given, answer, group="selfplay:time_to_event:12", split="train"):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": json.dumps(given)}, {"role": "assistant", "content": json.dumps(answer)}]
    return L.TrainingExample(example_id=f"e{L.fingerprint(messages)[:8]}", family=family, messages=messages,
                             verification=L.Verification(kind="construction", passed=True),
                             provenance=L.Provenance(source="test", origin="verified_synthetic", consent="ours", group=group, split=split))


class TraceTests(unittest.TestCase):
    def test_a_leakage_trace_reasons_from_the_moment_and_the_measured_signals_to_the_answer(self):
        steps = R.trace("leakage_judgment", GIVEN, ANSWER)
        text = "\n".join(steps)
        self.assertIn("At signup, before the first bill.", text)
        self.assertIn("`months_billed` 0.998", text)
        self.assertIn("`months_billed` stays out", text)
        self.assertIn("`subscriber_id` names a row", text)
        self.assertEqual(R.grounded(steps, GIVEN, ANSWER), [])
        self.assertTrue(R.think(steps).startswith("<think>\n") and R.think(steps).endswith("\n</think>"))

    def test_a_trace_with_a_number_or_a_column_its_records_do_not_hold_is_not_grounded(self):
        steps = R.trace("leakage_judgment", GIVEN, ANSWER)
        self.assertTrue(R.grounded([*steps, "Its lift is 0.74 over the next column."], GIVEN, ANSWER))
        self.assertTrue(R.grounded([*steps, "`refund_at` is recorded later."], GIVEN, ANSWER))

    def test_results_reason_about_the_interval_and_a_single_read(self):
        given = {"holdout_metrics": {"average_precision": 0.569, "roc_auc": 0.874}, "holdout_uses": 1,
                 "primary_interval": {"low": 0.349, "high": 0.808, "point": 0.569, "method": "cluster bootstrap over 49 groups"}}
        answer = {"primary_metric": "average_precision", "value": 0.569, "interval": given["primary_interval"], "beats_baseline": True,
                  "holdout_read_once": True, "claim": "benchmark evidence on synthetic data"}
        steps = R.trace("read_results", given, answer)
        self.assertEqual(R.grounded(steps, given, answer), [])
        self.assertIn("from 0.349 to 0.808", "\n".join(steps))

    def test_a_solution_workflow_reasons_step_by_step_from_the_moment_to_the_single_holdout_read(self):
        given = {**GIVEN, "rows": 3000}
        answer = {"task": "regression", "kind": "time_to_event", "workflow": [
            {"step": "split", "decision": {"holdout": "last 600 of 3000 rows by `signup_time`", "time_column": "signup_time", "group_column": None}},
            {"step": "leakage", "decision": {"forbidden": ["months_billed"], "identifiers": ["subscriber_id"]}},
            {"step": "features", "decision": {"recipe": "plain"}},
            {"step": "models", "decision": {"model": "hist_gradient_boosting", "selection_rule": "best mean on the folds"}},
            {"step": "final", "decision": {"metric": "integrated_brier", "holdout_read_once": True}}]}
        steps = R.trace("solution_design", given, answer)
        text = "\n".join(steps)
        self.assertEqual(R.grounded(steps, given, answer), [])
        for part in ("First the split", "`months_billed`", "plain recipe", "hist_gradient_boosting", "read once, scored by integrated_brier"):
            self.assertIn(part, text)

    def test_a_family_without_rules_has_no_trace(self):
        self.assertEqual(R.of_messages("evidence_answer", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]),
                         (None, ["the example is not a question and an answer in JSON"]))
        self.assertIsNone(R.trace("report", {}, {}))


class SelfplayPrimaryTests(unittest.TestCase):
    def test_the_primary_metric_is_the_one_its_interval_was_drawn_for(self):
        from dclab_rnd.lake.selfplay import primary_of

        metrics = {"roc_auc": 0.874, "average_precision": 0.569}
        self.assertEqual(primary_of(metrics, {"point": 0.569}), "average_precision")
        self.assertEqual(primary_of(metrics, None, "roc_auc"), "roc_auc")  # the engine's own report first
        self.assertEqual(primary_of({"mae": 3.1, "r2": 0.4}, {"point": 3.1}), "mae")


class LakeReasoningTests(unittest.TestCase):
    def test_a_grounded_trace_becomes_the_rationale_and_an_ungrounded_one_is_not_kept(self):
        from dclab_rnd.lake import reason

        from unittest import mock

        good = example("leakage_judgment", GIVEN, ANSWER)
        bad = example("split_design", GIVEN, {"time_column": None, "holdout": "a fifth"}, group="selfplay:time_to_event:13")
        real = R.trace

        def invented(family, given, answer):  # a rule that states what the records do not hold
            steps = real(family, given, answer)
            return [*steps, "Its lift is 0.74."] if family == "split_design" else steps

        with mock.patch.object(R, "trace", side_effect=invented):
            out, why = reason([good, bad, example("evidence_answer", {"q": 1}, {"a": 2}, group="research:x")])
        self.assertIsNone(out[1].rationale)
        self.assertEqual(why["reasoning not grounded"], 1)
        self.assertTrue(out[0].rationale and "months_billed" in out[0].rationale)
        self.assertEqual(why["reasoning kept"], 1)
        self.assertEqual(sum(why.values()), 3)

    def test_the_export_never_writes_a_rules_trace_as_the_models_thinking(self):
        """The owner's rule: reasoning written by code is labelled where it came from and never stored as the model's
        own thinking. Before, the solver export put it inside <think> tags (461 rows, 2026-10-17)."""
        from collections import Counter

        from dclab_rnd.lake import reason, write_set
        from dclab_rnd.lake.train import export

        lake = Path(tempfile.mkdtemp())
        rows, _ = reason([example("leakage_judgment", GIVEN, ANSWER)])
        self.assertTrue(rows[0].rationale)  # the rules wrote a trace
        write_set(lake / "gold" / "solver" / "leakage_judgment" / "v1", "training_example", "solver/leakage_judgment", rows, [], Counter())
        m = export("solver", lake, lake / "out")
        line = json.loads((lake / "out" / "train.chat.jsonl").read_text().splitlines()[0])
        self.assertTrue(line["messages"][-1]["content"].startswith("{"))
        self.assertNotIn("<think>", json.dumps(line["messages"]))
        self.assertEqual(json.loads(line["messages"][1]["content"]), GIVEN)  # the question is never rewritten
        self.assertEqual((line["metadata"]["reasoning_source"], line["metadata"]["reasoning"]), ("from_record", rows[0].rationale))
        mlx = json.loads((lake / "out" / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(set(mlx), {"messages"})
        self.assertNotIn("<think>", json.dumps(mlx))
        self.assertEqual(m["reasoning_in_metadata"], {"train": 1})

    def test_a_selfplay_example_whose_answer_misses_the_planted_leak_is_dropped(self):
        from unittest import mock

        from dclab_rnd import lake

        planted = example("leakage_judgment", GIVEN, ANSWER).model_copy(update={"provenance": L.Provenance(
            source="selfplay", origin="verified_synthetic", consent="ours", group="selfplay:time_to_event:12", split="train")})
        with mock.patch("dclab_rnd.lake.benchmark.truth", return_value={"leak": ["tickets_so_far"], "identifier": ["subscriber_id"], "time_column": None}):
            out, why = lake.reason([planted])
        self.assertEqual(out, [])
        self.assertEqual(why["answer misses the replica's planted truth (example dropped)"], 1)
        with mock.patch("dclab_rnd.lake.benchmark.truth", return_value={"leak": ["months_billed"], "identifier": ["subscriber_id"], "time_column": None}):
            out, _ = lake.reason([planted])
        self.assertTrue(out[0].rationale)

    def test_no_export_asks_a_model_to_think_inside_tags(self):
        from dclab_rnd.lake import train

        self.assertFalse(hasattr(train, "THINK_FIRST"))
        self.assertFalse(hasattr(train, "_with_think"))

    def test_the_benchmark_reads_the_answer_after_the_thinking(self):
        from dclab_rnd.lake.benchmark import _json

        self.assertEqual(_json('<think>\nThe split {by time}.\n</think>\n{"task": "binary"}'), {"task": "binary"})


if __name__ == "__main__":
    unittest.main()

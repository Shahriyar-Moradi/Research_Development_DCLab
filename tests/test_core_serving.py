"""Package 34.9: the prediction and outcome log's contracts (dclab_core/serving.py)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from dclab_core import serving  # noqa: E402
from dclab_core.contracts import CONTRACTS  # noqa: E402

SECRET = "s" * 32


def _prediction(key, **extra):
    return serving.PredictionRecord(**{"model_id": "p1", "model_version": "a" * 16, "batch": "b" * 32, "position": 0,
                                       "key": serving.key_hash(key, SECRET), "task": "binary", "score": 0.7,
                                       "decision": "alert", "threshold": 0.5, "at": "2026-10-24T10:00:00.000000+00:00", **extra})


def _outcome(key, outcome, known_at="2026-10-25T10:00:00+00:00", recorded_at="2026-10-25T11:00:00.000000+00:00"):
    return serving.OutcomeRecord(key=serving.key_hash(key, SECRET), outcome=outcome, known_at=known_at, recorded_at=recorded_at)


class ServingContractsTest(unittest.TestCase):
    def test_the_two_records_are_registered_contracts_with_their_versions(self):
        self.assertIs(CONTRACTS["prediction_record"], serving.PredictionRecord)
        self.assertIs(CONTRACTS["outcome_record"], serving.OutcomeRecord)
        record = _prediction("order-1")
        self.assertEqual(serving.PredictionRecord.read(record.model_dump_json()), record)
        self.assertEqual(record.schema_version, "dclab_core.prediction_record.v1")
        self.assertEqual(_outcome("order-1", 1).schema_version, "dclab_core.outcome_record.v1")

    def test_a_key_is_a_salted_hash_never_the_callers_value(self):
        hashed = serving.key_hash("ORD-12345", SECRET)
        self.assertRegex(hashed, r"^[0-9a-f]{32}$")
        self.assertEqual(hashed, serving.key_hash("ORD-12345", SECRET))  # the same key joins
        self.assertNotEqual(hashed, serving.key_hash("ORD-12345", "t" * 32))  # another secret, another hash
        self.assertEqual(serving.key_hash(123, SECRET), serving.key_hash(123.0, SECRET))  # a CSV's 123.0 is the JSON's 123
        self.assertEqual(serving.key_hash("123", SECRET), serving.key_hash(123, SECRET))
        self.assertEqual(serving.key_hash("470000166.0", SECRET), serving.key_hash(470000166, SECRET))  # a float column written as text
        self.assertNotEqual(serving.key_hash("007", SECRET), serving.key_hash(7, SECRET))  # a text key keeps its zeros
        for junk in ({"phone": "0912"}, ["a"], True, None, float("nan"), "  "):  # a key that cannot be a row's key is refused
            with self.assertRaises(ValueError):
                serving.key_hash(junk, SECRET)
        with self.assertRaises(ValidationError):  # a raw key cannot be stored as the key
            serving.PredictionRecord.model_validate({**_prediction("x").model_dump(), "key": "ORD-12345"})
        with self.assertRaises(ValueError):
            serving.key_hash("", SECRET)
        with self.assertRaises(ValueError):
            serving.key_hash("k", "short")  # a secret too short to hide a guessable key
        self.assertRegex(serving.new_key(), r"^[0-9a-f]{32}$")

    def test_a_record_holds_no_free_text_and_no_field_beyond_its_contract(self):
        with self.assertRaises(ValidationError):
            serving.PredictionRecord.read({**_prediction("k").model_dump(), "row": {"sender_phone": "0912"}})
        with self.assertRaises(ValidationError):  # a decision is a short code, not a sentence that could quote a value
            _prediction("k", decision="flag ride of 0912 555")
        with self.assertRaises(ValidationError):
            _prediction("k", at="yesterday")
        with self.assertRaises(ValidationError):  # a multiclass prediction names its class by index
            _prediction("k", task="multiclass", decision=None, threshold=None, score=0.6)

    def test_an_outcome_is_coded_as_the_model_predicts_it(self):
        code = serving.code_outcome
        self.assertEqual(code("fraud", "binary", positive_label="fraud", negative_label="ok"), 1.0)
        self.assertEqual(code("ok", "binary", positive_label="fraud", negative_label="ok"), 0.0)
        self.assertEqual(code(1, "binary", positive_label="1"), 1.0)
        self.assertEqual(code("0", "binary", positive_label="1"), 0.0)
        self.assertEqual(code(True, "binary"), 1.0)
        self.assertEqual(code("true", "binary", positive_label="1", negative_label="0"), 1.0)  # the labels are 1/0 themselves
        with self.assertRaises(ValueError):  # labels 1 and 2: a 0 is a typo, never read as the negative
            code(0, "binary", positive_label="2", negative_label="1")
        self.assertEqual(code(1.0, "binary", positive_label="2", negative_label="1"), 0.0)  # 1.0 is the label 1
        with self.assertRaises(ValueError):  # neither label: refused, never read as a negative
            code("Fraud?", "binary", positive_label="fraud", negative_label="ok")
        self.assertEqual(code("b", "multiclass", class_labels=["a", "b", "c"]), 1.0)
        with self.assertRaises(ValueError):
            code("d", "multiclass", class_labels=["a", "b", "c"])
        self.assertEqual(code("12.5", "regression"), 12.5)
        for bad in (float("nan"), "inf", None, "twelve"):
            with self.assertRaises(ValueError):
                code(bad, "regression")

    def test_the_logs_join_by_key_the_latest_outcome_wins_and_the_delay_is_measured(self):
        predictions = [_prediction("o1"), _prediction("o2").model_copy(update={"position": 1}), _prediction("o3").model_copy(update={"position": 2})]
        outcomes = [_outcome("o1", 0.0), _outcome("o2", 1.0),
                    _outcome("o1", 1.0, recorded_at="2026-10-26T09:00:00.000000+00:00"),  # a correction, recorded later
                    _outcome("unknown", 1.0)]
        joined = serving.join(predictions, outcomes)
        self.assertEqual(len(joined), 2)  # o3 has no outcome yet; the unknown key has no prediction
        by_key = joined.set_index("key")
        self.assertEqual(by_key.loc[serving.key_hash("o1", SECRET), "outcome"], 1.0)
        self.assertEqual(by_key.loc[serving.key_hash("o2", SECRET), "delay_seconds"], 24 * 3600)
        self.assertEqual(serving.join(predictions, []).shape[0], 0)

    def test_a_log_reads_back_line_by_line_and_a_foreign_line_is_refused(self):
        folder = Path(tempfile.mkdtemp())
        path = folder / "predictions.jsonl"
        path.write_text("".join(r.model_dump_json() + "\n" for r in (_prediction("a"), _prediction("b"))), encoding="utf-8")
        self.assertEqual([r.key for r in serving.read_log(path, serving.PredictionRecord)],
                         [serving.key_hash("a", SECRET), serving.key_hash("b", SECRET)])
        path.write_text(path.read_text() + json.dumps({"schema_version": "dclab_core.prediction_record.v1", "row": [1]}) + "\n")
        with self.assertRaises(ValueError):
            serving.read_log(path, serving.PredictionRecord)
        self.assertEqual(serving.read_log(folder / "absent.jsonl", serving.OutcomeRecord), [])
        torn = folder / "torn.jsonl"  # a writer killed mid-line, then a later batch; and a batch still being written
        whole = [_prediction(k).model_dump_json() for k in "abc"]
        torn.write_text(whole[0] + "\n" + whole[1][:40] + "\n" + whole[2] + "\n" + whole[0][:25], encoding="utf-8")
        self.assertEqual(len(serving.read_log(torn, serving.PredictionRecord)), 2)  # the torn lines are no records


class _Scorer:
    """A stand-in for the exported scorer: a score per row from its position, never from a value."""

    def __init__(self, task="binary"):
        self.key = "k" * 32
        self.meta = {"task": task, "model_id": "p1", "model_version": "v1", "threshold": 0.5, "positive_label": "fraud",
                     "negative_label": "ok", "class_labels": ["a", "b"]}

    def predict(self, frame):
        if self.meta["task"] == "multiclass":
            return [{"class": "b", "probabilities": {"a": 0.3, "b": 0.7}} for _ in range(len(frame))]
        return [{"score": 0.1 + 0.8 * (i % 2), "alert": bool(i % 2)} for i in range(len(frame))]


def _serve_log():
    import importlib.util

    path = ROOT / "dclab_rnd" / "studio" / "service_template" / "serve_log.py"
    spec = importlib.util.spec_from_file_location("serve_log_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ServeLogTest(unittest.TestCase):
    """The service's log (serve_log.py), with a stand-in scorer: the routes' logic without a fitted model."""

    def setUp(self):
        import os
        from unittest import mock

        self.folder = Path(tempfile.mkdtemp())
        patch = mock.patch.dict(os.environ, {"DCLAB_SERVICE_LOG_DIR": str(self.folder)})
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop("DCLAB_LOG_KEY", None)
        self.sl = _serve_log()

    def _logs(self):
        return (serving.read_log(self.folder / "predictions.jsonl", serving.PredictionRecord),
                serving.read_log(self.folder / "outcomes.jsonl", serving.OutcomeRecord))

    def test_a_csv_float_key_joins_its_json_outcome_and_decisions_are_codes(self):
        import pandas as pd

        log = self.sl.Log(_Scorer())
        answer = log.predict(pd.DataFrame({"order_id": ["470000166.0", "470000167.0"], "phone": ["0912-555", "0912-556"]}), "order_id")
        self.assertEqual([p["key"] for p in answer["predictions"]], ["470000166", "470000167"])
        self.assertEqual(log.outcomes({"outcomes": [{"key": 470000166, "outcome": "fraud"}, {"key": 470000167, "outcome": "ok"}]}),
                         {"recorded": 2})
        predictions, outcomes = self._logs()
        self.assertEqual([r.decision for r in predictions], ["no_alert", "alert"])
        self.assertEqual(serving.join(predictions, outcomes)["outcome"].tolist(), [1.0, 0.0])
        text = "".join(f.read_text() for f in self.folder.glob("*.jsonl"))
        for value in ("470000166", "0912-555", "fraud"):
            self.assertNotIn(value, text)

    def test_keys_join_across_exports_when_the_log_key_is_kept(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"DCLAB_LOG_KEY": "L" * 32}):
            before = self.sl.Log(_Scorer())
            before.predict([{"order_id": 123}], "order_id")
            retrained = _Scorer()
            retrained.key = "z" * 32  # a new export: a new service key
            self.sl.Log(retrained).outcomes({"key": 123, "outcome": "fraud"})
        self.assertEqual(len(serving.join(*self._logs())), 1)

    def test_a_bad_request_is_refused_and_leaves_nothing(self):
        log = self.sl.Log(_Scorer())
        for rows, key in (("abc", None), ({"rows": "abc"}["rows"], None), ([{"order_id": 1}], ["order_id"]), ([{"x": 1}], "order_id"),
                          ([{"order_id": None}], "order_id"), ([], None)):
            with self.assertRaises(self.sl.Refused):
                log.predict(rows, key)
        for body in ({"key": {"phone": "0912"}, "outcome": 1}, {"key": True, "outcome": "fraud"}, {"outcomes": []}, "abc",
                     {"outcomes": [{"key": 1, "outcome": "fraud"}, {"key": 2, "outcome": "Fraud?"}]},
                     {"key": 1, "outcome": "fraud", "known_at": "yesterday"}):
            with self.assertRaises(self.sl.Refused) as refused:
                log.outcomes(body)
            self.assertNotIn("0912", str(refused.exception))
            self.assertNotIn("Fraud?", str(refused.exception))
        self.assertEqual(self._logs(), ([], []))

    def test_a_torn_line_is_closed_before_the_next_batch(self):
        log = self.sl.Log(_Scorer("multiclass"))
        log.predict([{"x": 1}])
        with open(self.folder / "predictions.jsonl", "a", encoding="utf-8") as handle:
            handle.write('{"schema_version": "dclab_core.predic')  # a worker killed mid-line
        log.predict([{"x": 2}, {"x": 3}])
        predictions, _ = self._logs()
        self.assertEqual([(r.predicted_class, r.score) for r in predictions], [(1, 0.7)] * 3)


if __name__ == "__main__":
    unittest.main()

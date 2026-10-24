"""The students on real tables in shadow (dclab_rnd.real_shadow, 2026-10-23): the product path records each table's
decisions with every student's whisper; a train table in a workspace under train consent, a dev table in one with no
consent (only measured), a sealed table never; Claude's opinions stay apart, capped, never empty."""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import real_shadow  # noqa: E402

STATE = {"column": "plan", "name_tokens": ["plan"], "dtype": "categorical", "uniqueness": "under_1pct", "nulls": "none",
         "cardinality": "low", "value_pattern": "short_text"}


def table():
    from dclab_core.replicas import subscription_churn

    return subscription_churn.generate(seed=0).subscribers.head(400)


def parquet_bytes(frame):
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def derived(did):
    return {"did": did, "rows": 400, "target": "churned",
            "columns": {"plan": {"state": STATE, "identifier_rule": False, "role_rule": "categorical_code"}}}


def a_student():
    """A tiny fitted student for column.semantic_role (the repository's fitted ones are not in git)."""
    from dclab_core.answerers import ClassifierAnswerer

    rows = []
    for i in range(60):
        role = ["numeric", "categorical_code"][i % 2]
        state = {**STATE, "column": f"c{i}", "name_tokens": [f"c{i}"], "dtype": "float" if role == "numeric" else "categorical"}
        rows.append((state, role))
    return ClassifierAnswerer("column.semantic_role", name="column.semantic_role.test", version="t1").fit(rows)


STUDENT = []


def student():
    if not STUDENT:
        STUDENT.append(a_student())
    return STUDENT[0]


class RealShadowTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        from dclab_rnd.lake import openml
        from dclab_rnd.ledger import install, shadow

        self.addCleanup(install, None)
        self.train, self.eval = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        frame = table()
        self.fail_did = None

        def get(url, timeout=0):
            if self.fail_did is not None:
                raise TimeoutError("timed out")
            return parquet_bytes(frame)
        for target, name, value in ((openml, "_json", lambda url: {"data_set_description": {"parquet_url": "x"}}), (openml, "_get", get),
                                    (shadow, "answerers", lambda *a, **k: (student(),))):
            patch = mock.patch.object(target, name, side_effect=value)
            patch.start()
            self.addCleanup(patch.stop)
        splits = {1: "train", 2: "dev", 3: "sealed"}
        self.derived = [derived(1), derived(2), derived(3)]
        for name, value in (("split_of", lambda did: splits[int(did)]),):
            patch = mock.patch.object(real_shadow, name, side_effect=value)
            patch.start()
            self.addCleanup(patch.stop)
        tables = mock.patch.object(real_shadow, "_tables", side_effect=lambda limit=None: [d for d in self.derived if splits[d["did"]] != "sealed"])
        tables.start()
        self.addCleanup(tables.stop)

    def run_all(self, **kw):
        return real_shadow.run(self.train, self.eval, log=lambda m: None, **kw)

    def test_train_dev_and_sealed_tables_each_go_where_they_belong(self):
        from dclab_rnd.ledger import open_ledger

        counts = self.run_all()
        self.assertEqual((counts.get("done_train"), counts.get("done_dev")), (1, 1))
        train, dev = open_ledger(self.train).records(), open_ledger(self.eval).records()
        self.assertEqual({r.consent for r in train}, {"train"})  # the train table: the students' homework
        self.assertNotIn("train", {r.consent for r in dev})  # the dev table: measured only, never trained on
        self.assertEqual({r.draft for r in train}, {"openml-1"})
        self.assertEqual({r.draft for r in dev}, {"openml-2"})  # the sealed table (3) is nowhere
        role = next(r for r in train if r.point == "column.semantic_role")
        self.assertEqual([a.answerer for a in role.answers if a.by == "rule"], ["datacard:purposes"])  # the rule that answered
        self.assertTrue(any(a.source == "shadow" for a in role.answers))  # the student whispered
        self.assertFalse(any(a.by == "model_role" for r in train for a in r.answers))  # nobody "chose": the proposal was kept
        self.assertEqual(self.run_all().get("done_train"), 1)  # resumable: read once

    def test_the_dev_truths_measure_the_rule_and_the_student(self):
        self.run_all()
        with mock.patch.object(real_shadow, "_truths", return_value={("openml-2", "plan"): {"column.semantic_role": "categorical_code"}}):
            measured = real_shadow.measure(self.eval)
        point = measured["column.semantic_role"]
        self.assertEqual(point["rule"], {"truth": 1, "right": 1, "accuracy": 1.0})
        student = next(v for k, v in point.items() if k.startswith("column.semantic_role.test"))
        self.assertEqual((student["truth"], student["coverage"]), (1, 1.0))
        self.assertIn("accuracy", student)

    def test_compare_puts_generations_of_students_on_the_same_rows(self):
        from dclab_rnd.ledger import shadow

        self.run_all()
        truth = {("openml-2", "plan"): {"column.semantic_role": "categorical_code"}}
        with mock.patch.object(real_shadow, "_truths", return_value=truth), \
                mock.patch.object(shadow, "answerers", side_effect=lambda folder=None: (student(),)):
            out = real_shadow.compare({"first": Path("a"), "second": Path("b")}, self.eval)
        point = out["column.semantic_role"]
        self.assertEqual(point["rule"]["accuracy"], 1.0)
        self.assertEqual(point["first"], point["second"])  # the same student on the same rows: the same scores
        self.assertEqual(point["first"]["coverage"], 1.0)
        self.assertTrue((self.eval / "compare.json").is_file())

    def test_a_failed_download_is_kept_failed_and_read_again_only_when_asked(self):
        self.fail_did = 1
        counts = self.run_all()
        self.assertEqual(counts.get("failed_train"), 1)
        self.assertIn("timed out", json.loads((self.train / "done" / "1.json").read_text())["failed"])
        self.fail_did = None
        self.assertEqual(self.run_all().get("failed_train"), 1)  # a failure stays until asked
        self.assertEqual(self.run_all(retry_failed=True).get("done_train"), 1)

    def silver_with(self, transport, cap, **kw):
        from dclab_rnd import models
        from dclab_rnd.models import FileUsage, Gateway

        self.run_all()
        gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: transport)
        rows = real_shadow._rows(self.train)
        forced = [{**rows[0], "record": f"r{i}", "subject": f"c{i}", "students": {"s": "zz-different"}} for i in range(5)]
        with mock.patch.dict(os.environ, {"DCLAB_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k", "ANTHROPIC_MODEL": "claude-sonnet-5-5",
                                          "DCLAB_NO_LIVE_MODELS": ""}), \
                mock.patch.object(models, "for_tools", return_value=gw), mock.patch.object(real_shadow, "_rows", return_value=forced):
            return real_shadow.silver(self.train, per_point=5, cap_eur=cap, log=lambda m: None, **kw)

    def test_claudes_opinion_is_silver_apart_never_empty_and_the_cap_binds(self):
        from dclab_rnd.ledger import open_ledger

        reply = {"content": json.dumps({"answer": "categorical_code", "reason": "a code"}), "tool_calls": [],
                 "usage": {"input_tokens": 4000, "output_tokens": 50}}

        class Script:
            def complete(self, messages, tools=None, max_tokens=0, **options):
                return reply
        labels_before = len(open_ledger(self.train).labels())
        out = self.silver_with(Script(), cap=0.03)  # a priced model: a few requests reach the cap
        lines = [json.loads(l) for l in (self.train / "silver.jsonl").read_text().splitlines()]
        self.assertTrue(lines)
        self.assertTrue(all(l["claude"] == "categorical_code" and l["label"] == "silver" and l["use"] == "evaluation" for l in lines))
        self.assertLessEqual(out["spent_eur"], 0.03 + 0.02)  # the request that crossed it may finish; nothing after
        self.assertTrue(out.get("stopped") or len(lines) == 5)
        self.assertEqual(len(open_ledger(self.train).labels()), labels_before)  # never a label in the ledger

    def test_a_jobs_stop_reaches_claudes_step(self):
        class Halt(BaseException):
            pass

        class Script:
            def complete(self, messages, tools=None, max_tokens=0, **options):
                return {"content": json.dumps({"answer": "categorical_code", "reason": "a code"}), "tool_calls": [], "usage": {}}
        calls = []

        def checkpoint():
            calls.append(1)
            if len(calls) > 2:
                raise Halt()
        with self.assertRaises(Halt):
            self.silver_with(Script(), cap=1.0, checkpoint=checkpoint)
        lines = (self.train / "silver.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 2)  # asked twice, stopped before the third request

    def test_an_outage_stops_and_a_single_failure_is_an_abstention(self):
        class Broken:
            def complete(self, *a, **k):
                raise ConnectionError("down")
        with mock.patch.object(real_shadow.time, "sleep"):
            out = self.silver_with(Broken(), cap=1.0)
        self.assertEqual((out["stopped"], out["why"]), (True, "failures"))
        lines = [json.loads(l) for l in (self.train / "silver.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 2)  # the two before the third in a row, kept as failed abstentions; the third not written
        self.assertTrue(all(l["failed"] and l["abstained"] and l["claude"] is None for l in lines))


if __name__ == "__main__":
    unittest.main()

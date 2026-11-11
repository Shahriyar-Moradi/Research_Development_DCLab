"""Package 26.9: every decision DCLab's rules, Claude, classifiers and people made, gathered whole from every workspace."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def record(i, by_model=True, shadow=False):
    answers = [{"by": "rule", "answerer": "rule:x", "value": "a"}]
    if by_model:
        answers.append({"by": "model_role", "answerer": "home", "value": "b", "reason": "the person said so"})
    if shadow:
        answers.append({"by": "classifier", "answerer": "clf@1", "value": "a", "confidence": 0.9, "source": "shadow"})
    return {"id": f"dr{i:020d}", "point": "problem.pack", "subject": {"kind": "draft", "key": "d1"}, "state": {"pack": "x"},
            "options": ["a", "b"], "answers": answers, "value_used": "a", "used": "rule", "at": "2026-10-17T00:00:00+00:00"}


def workspace(root, name, records, model="claude-sonnet-5-5"):
    ws = root / name
    (ws / "ledger" / "decisions" / "2026" / "10").mkdir(parents=True)
    (ws / "ledger" / "decisions" / "2026" / "10" / "17.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (ws / "model_requests.jsonl").write_text(json.dumps({"at": "t", "purpose": "home_agent", "model": model, "outcome": "ok"}) + "\n")
    return ws


class HarvestTests(unittest.TestCase):
    def setUp(self):
        self.root, self.lake = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())

    def harvest(self):
        from dclab_rnd.lake.harvest import harvest

        return harvest(self.root, self.lake, database=False)

    def test_every_decision_is_kept_whole_once_with_where_it_was_found(self):
        workspace(self.root, "dev", [record(1), record(2, shadow=True)])
        workspace(self.root, "copy/twin", [record(1)])  # the same record kept in two places
        m = self.harvest()
        self.assertEqual((m["ledgers"], m["decisions"], m["model_requests"]), (2, 2, 1))
        rows = [json.loads(x) for x in (self.lake / "bronze" / "sessions" / "ledgers" / "decisions.jsonl").read_text().splitlines()]
        one = next(r for r in rows if r["id"] == record(1)["id"])
        self.assertEqual(sorted(one["found_in"]), ["copy/twin", "dev"])
        self.assertEqual(len(one["answers"]), 2)
        self.assertEqual(one["served_models"], {"home_agent": "claude-sonnet-5-5"})

    def test_each_answer_goes_under_its_answerer_with_what_it_may_be_used_for(self):
        workspace(self.root, "dev", [record(1, shadow=True)])
        workspace(self.root, "local", [record(2)], model="qwen2.5-coder:1.5b")  # a local model's role: not Claude's, its own answerer
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"], {"dclab-rules": 2, "claude": 1, "dclab-classifiers": 1, "qwen2.5-coder-1.5b": 1})
        base = self.lake / "bronze" / "answers"
        claude = [json.loads(x) for x in (base / "claude" / "sessions" / "problem.pack" / "all.jsonl").read_text().splitlines()]
        self.assertEqual((claude[0]["use"], claude[0]["value"], claude[0]["reason"]), ("eval_only", "b", "the person said so"))
        self.assertEqual(claude[0]["model"], "claude-sonnet-5-5")
        local = json.loads((base / "qwen2.5-coder-1.5b" / "sessions" / "problem.pack" / "all.jsonl").read_text())
        self.assertEqual(local["use"], "eval_only")  # not the owner's Qwen decision: kept, not trained, until its terms are checked
        rules = [json.loads(x) for x in (base / "dclab-rules" / "sessions" / "problem.pack" / "all.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["use"] == "train" for r in rules))

    def test_qwen_on_openrouter_answers_under_its_own_name_for_training_and_claude_stays_evaluation_only(self):
        """The owner, 2026-11-11 (store/owner_decisions.json): Qwen served through OpenRouter is trained on, kept apart by
        answerer. Each model role's answer goes under the model of its own request (its kept exchange), so a workspace
        Claude and Qwen both served files each answer under the right one."""
        both = workspace(self.root, "switched", [], model="claude-sonnet-5-5")
        claude_answer = dict(record(1), answers=[{"by": "rule", "answerer": "rule:x", "value": "a"},
                                                 {"by": "model_role", "answerer": "home", "value": "b", "source": "mrc1"}])
        qwen_answer = dict(record(2), answers=[{"by": "rule", "answerer": "rule:x", "value": "a"},
                                               {"by": "model_role", "answerer": "home", "value": "a", "source": "mrq1"}])
        day = both / "ledger" / "decisions" / "2026" / "10" / "17.jsonl"
        day.write_text("".join(json.dumps(r) + "\n" for r in (claude_answer, qwen_answer)))
        (both / "model_exchanges" / "2026" / "11").mkdir(parents=True)
        (both / "model_exchanges" / "2026" / "11" / "11.jsonl").write_text(
            json.dumps({"at": "t1", "model_request": "mrc1", "model": "claude-sonnet-5-5", "provider": "anthropic", "messages": []}) + "\n"
            + json.dumps({"at": "t2", "model_request": "mrq1", "model": "qwen/qwen3.8-flash", "provider": "openrouter",
                          "source_class": "open_weights_model", "messages": [{"role": "user", "content": "mrc1"}]}) + "\n")
        workspace(self.root, "qwen_only", [record(3)], model="qwen/qwen3.8-flash")  # no exchange kept: the one model it was served by
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"], {"dclab-rules": 3, "claude": 1, "qwen3.8-flash": 2})
        base = self.lake / "bronze" / "answers"
        qwen = {r["sample_id"]: r for r in (json.loads(x) for x in (base / "qwen3.8-flash" / "sessions" / "problem.pack" / "all.jsonl").read_text().splitlines())}
        exact = qwen[record(2)["id"]]
        self.assertEqual((exact["use"], exact["model"], exact["provider"], exact["source_class"], exact["model_known_by"]),
                         ("train", "qwen/qwen3.8-flash", "openrouter", "open_weights_model", "its request's kept exchange"))
        self.assertIn("before any fine-tune", exact["licence"])  # the licence to check before a fine-tune travels with the answer
        self.assertEqual((qwen[record(3)["id"]]["use"], qwen[record(3)["id"]]["model_known_by"]), ("train", "the one model its workspace was served by"))
        claude = json.loads((base / "claude" / "sessions" / "problem.pack" / "all.jsonl").read_text())
        self.assertEqual((claude["sample_id"], claude["use"]), (record(1)["id"], "eval_only"))  # Claude's, though Qwen served the same workspace
        decisions = {json.loads(x)["id"]: json.loads(x) for x in (self.lake / "bronze" / "sessions" / "ledgers" / "decisions.jsonl").read_text().splitlines()}
        self.assertEqual(decisions[record(2)["id"]]["role_models"]["mrq1"]["model"], "qwen/qwen3.8-flash")
        # a stale answerer's harvested folder is rebuilt whole; a captured source (lake) is never touched
        (base / "qwen3.8-flash" / "lake" / "column.time").mkdir(parents=True)
        (base / "qwen3.8-flash" / "lake" / "column.time" / "dev.jsonl").write_text("{}\n")
        import shutil

        shutil.rmtree(self.root / "qwen_only")
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"]["qwen3.8-flash"], 1)
        self.assertTrue((base / "qwen3.8-flash" / "lake" / "column.time" / "dev.jsonl").is_file())

    def test_a_qwen_answer_is_evaluation_only_when_the_owners_record_says_no(self):
        from dclab_rnd.store import provider_outputs

        workspace(self.root, "qwen_only", [record(1)], model="qwen/qwen3.8-flash")
        no = Path(tempfile.mkdtemp()) / "owner_decisions.json"
        no.write_text(json.dumps({"openrouter_qwen": {"for_training": False}}))
        with mock.patch.object(provider_outputs, "DECISIONS", no):
            self.harvest()
        row = json.loads((self.lake / "bronze" / "answers" / "qwen3.8-flash" / "sessions" / "problem.pack" / "all.jsonl").read_text())
        self.assertEqual(row["use"], "eval_only")

    def test_a_workspace_several_other_models_served_files_its_answer_as_unknown(self):
        ws = workspace(self.root, "many", [record(1)], model="qwen/qwen3.8-flash")
        with (ws / "model_requests.jsonl").open("a") as f:
            f.write(json.dumps({"at": "t", "purpose": "intern", "model": "deepseek-chat", "outcome": "ok"}) + "\n")
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"].get("unknown-model"), 1)  # never guessed as Qwen's
        self.assertNotIn("qwen3.8-flash", m["answers_by_answerer"])

    def test_after_the_switch_claudes_earlier_answers_stay_claudes(self):
        """Review of step 3 (finding 1): most roles pass no request id, and the latest model per purpose is Qwen once the
        switch has served each purpose once; Claude's earlier answers must not move into Qwen's training folder."""
        before = dict(record(1), at="2026-10-17T00:05:00+00:00")  # two minutes after Claude served it
        after = dict(record(2), at="2026-11-11T09:03:00+00:00")  # three minutes after Qwen served it
        between = dict(record(3), at="2026-11-05T12:00:00+00:00")  # no request in the ten minutes before
        ws = workspace(self.root, "dev", [before, after, between])
        (ws / "model_requests.jsonl").write_text(
            json.dumps({"at": "2026-10-17T00:03:00+00:00", "purpose": "home_agent", "model": "claude-sonnet-5-5", "outcome": "ok"}) + "\n"
            + json.dumps({"at": "2026-11-11T09:00:00+00:00", "purpose": "home_agent", "model": "qwen/qwen3.8-flash", "outcome": "ok"}) + "\n")
        m = self.harvest()
        self.assertEqual((m["answers_by_answerer"].get("claude"), m["answers_by_answerer"].get("qwen3.8-flash")), (2, 1))
        base = self.lake / "bronze" / "answers"
        qwen = json.loads((base / "qwen3.8-flash" / "sessions" / "problem.pack" / "all.jsonl").read_text())
        self.assertEqual((qwen["sample_id"], qwen["use"], qwen["model_known_by"]),
                         (after["id"], "train", "the one model served in the ten minutes before the decision"))
        claude = {json.loads(x)["sample_id"]: json.loads(x) for x in (base / "claude" / "sessions" / "problem.pack" / "all.jsonl").read_text().splitlines()}
        self.assertEqual(sorted(claude), sorted([before["id"], between["id"]]))
        self.assertTrue(all(r["use"] == "eval_only" for r in claude.values()))
        self.assertIn("ever served", claude[between["id"]]["model_known_by"])  # Claude among every model it was served by

    def test_a_shadow_models_answer_is_filed_under_its_own_model_never_the_served_one(self):
        """Review of step 3 (finding 2): a shadow's model-role answer is logged, never used: under its own model, eval_only."""
        shadowed = dict(record(1), answers=[{"by": "rule", "answerer": "rule:x", "value": "a"},
                                            {"by": "model_role", "answerer": "shadow:ollama:qwen3.5:4b", "value": "b", "source": "shadow"}])
        workspace(self.root, "dev", [shadowed], model="qwen/qwen3.8-flash")
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"], {"dclab-rules": 1, "ollama-qwen3.5-4b": 1})
        row = json.loads((self.lake / "bronze" / "answers" / "ollama-qwen3.5-4b" / "sessions" / "problem.pack" / "all.jsonl").read_text())
        self.assertEqual((row["use"], row["model"]), ("eval_only", "ollama:qwen3.5:4b"))

    def test_a_sealed_runs_qwen_answers_are_never_training_rows(self):
        """Review of step 3 (finding 4): a workspace under the exam vault (or a sealed folder) is a sealed run's."""
        workspace(self.root, "exam_vault/CEX-050_sealed/ws", [record(1)], model="qwen/qwen3.8-flash")
        self.harvest()
        row = json.loads((self.lake / "bronze" / "answers" / "qwen3.8-flash" / "sessions" / "problem.pack" / "all.jsonl").read_text())
        self.assertEqual(row["use"], "eval_only")

    def test_a_test_scripts_clicks_are_never_filed_as_peoples_answers(self):
        """The review of 2026-10-25: the end-to-end tests' kept workspaces hold scripted "person" answers; filed as people's
        they would have outnumbered every real person's answer in the lake."""
        person = dict(record(1, by_model=False), answers=[{"by": "rule", "answerer": "rule:x", "value": "a"},
                                                         {"by": "person", "answerer": "owner", "value": "b"}])
        real = dict(person, id="dr" + "9" * 20)
        ws = workspace(self.root, "e2e/2026-10-25/164537_product_e2e", [person])
        (ws.parent.parent / "scripted.json").write_text('{"by": "script"}')  # the marker sits above the kept workspace
        workspace(self.root, "dev", [real])
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"].get("people"), 1)  # the real person's only
        self.assertEqual(m["answers_by_answerer"].get("dclab-rules"), 2)  # the rules' answers in both stay

    def test_claudes_verdicts_on_real_decisions_are_silver_weighted_by_its_measured_accuracy(self):  # 29.18
        from dclab_rnd import ledger as L
        from dclab_rnd.lake import harvest as H

        book = L.FileLedger(self.root / "w1" / "ledger", "w1")
        rec = L.decide("column.is_identifier", ("column", "order_id"), answers=[L.answer("rule", "id_like", True)], value_used=True,
                       state={"column": "order_id"}, ledger=book, project="p1")
        L.label(rec, True, "model", "model:claude-sonnet-5-5", weight=1.0, ledger=book, evidence={"kind": "review", "ref": "high: the bands say so"})
        L.label(rec, False, "model", "model:qwen-local", weight=1.0, ledger=book, evidence={"kind": "review", "ref": "low: x"})  # not Claude: not here
        with mock.patch.object(H, "judge_accuracy", return_value={rec.point: 0.86}):
            H.harvest(self.root, self.lake, database=False)
        kept = [json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()]
        self.assertEqual(len(kept), 1)
        k = kept[0]
        self.assertEqual((k["use"], k["role"], k["silver_weight"], k["usable"], k["confidence"], k["reason"]),
                         ("silver", "judge", 0.86, True, "high", "the bands say so"))
        claude_led = L.decide("column.is_identifier", ("column", "rider_id"), answers=[L.answer("rule", "id_like", False), L.answer("model_role", "home", True)],
                              value_used=True, used="model_role", level=2, state={"column": "rider_id"}, ledger=book, project="p1")  # Claude decided it
        (self.root / "w1" / "model_requests.jsonl").write_text(json.dumps({"at": "t", "purpose": "home_agent", "model": "claude-sonnet-5-5"}) + "\n")
        L.label(claude_led, True, "model", "model:claude-sonnet-5-5", weight=1.0, ledger=book, evidence={"kind": "review", "ref": "high: same"})
        with mock.patch.object(H, "judge_accuracy", return_value={rec.point: 0.86}):
            H.harvest(self.root, self.lake, database=False)
        rows = {json.loads(x)["sample_id"]: json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()}
        self.assertEqual((rows[claude_led.id]["self_judged"], rows[claude_led.id]["usable"]), (True, False))  # never grading its own answer
        with mock.patch.object(H, "judge_accuracy", return_value={rec.point: 0.70, f"{rec.point}|high": 1.0}):  # its high verdicts measured apart
            H.harvest(self.root, self.lake, database=False)
        k = {json.loads(x)["sample_id"]: json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()}[rec.id]
        self.assertEqual((k["silver_weight"], k["usable"]), (1.0, True))  # a weak point overall, but its high-confidence verdicts proved right
        with mock.patch.object(H, "judge_accuracy", return_value={}):  # never measured on this point: kept, not usable
            H.harvest(self.root, self.lake, database=False)
        k = {json.loads(x)["sample_id"]: json.loads(x) for x in (self.lake / "bronze" / "answers" / "claude" / "sessions" / rec.point / "judge.jsonl").read_text().splitlines()}[rec.id]
        self.assertEqual((k["silver_weight"], k["usable"]), (None, False))

    def test_a_twin_runs_steps_are_kept_and_claudes_side_is_evaluation_only(self):
        run = self.root / "twin" / "runs" / "20261017-000000-abc"
        run.mkdir(parents=True)
        step = {"id": "st1", "step": "solution.columns", "system": {"answerer": "rules", "status": "abstained"},
                "ai_first": {"answerer": "claude-sonnet-5-5", "status": "answered", "answer": {"target": "Churn"}}, "truth": {"value": "x"}}
        (run / "steps.jsonl").write_text(json.dumps(step) + "\n")
        (run / "ai_first" / "design").mkdir(parents=True)
        (run / "ai_first" / "design" / "DESIGN.md").write_text("# design")
        m = self.harvest()
        self.assertEqual(m["twin_runs"], 1)
        self.assertTrue((self.lake / "bronze" / "sessions" / "twin" / run.name / "ai_first" / "design" / "DESIGN.md").is_file())
        claude = json.loads((self.lake / "bronze" / "answers" / "claude" / "twin" / "solution.columns" / "all.jsonl").read_text())
        self.assertEqual((claude["use"], claude["value"]), ("eval_only", {"target": "Churn"}))

    def test_a_twin_lane_qwen_served_is_qwens_for_training(self):
        run = self.root / "twin" / "runs" / "20261111-000000-qwn"
        run.mkdir(parents=True)
        step = {"id": "st1", "step": "solution.columns", "system": {"answerer": "rules", "status": "abstained"},
                "ai_first": {"answerer": "qwen/qwen3.8-flash", "provider": "openrouter", "status": "answered", "answer": {"target": "Churn"}}}
        (run / "steps.jsonl").write_text(json.dumps(step) + "\n")
        m = self.harvest()
        self.assertEqual(m["answers_by_answerer"], {"qwen3.8-flash (twin)": 1, "dclab-rules (twin)": 1})
        row = json.loads((self.lake / "bronze" / "answers" / "qwen3.8-flash" / "twin" / "solution.columns" / "all.jsonl").read_text())
        self.assertEqual((row["use"], row["value"]), ("train", {"target": "Churn"}))
        # finding 3 of the review: Home's lane keeps the model apart, the expert's lane in its answerer's prefix
        home = dict(step, id="st2", ai_first={"answerer": "ai_first@home", "model": "qwen/qwen3.8-flash", "status": "answered", "answer": {"a": 1}})
        expert = dict(step, id="st3", ai_first={"answerer": "ai_first@expert:qwen/qwen3.8-flash", "status": "answered", "answer": {"a": 2}})
        claude = dict(step, id="st4", ai_first={"answerer": "ai_first@expert:claude-sonnet-5-5", "status": "answered", "answer": {"a": 3}})
        (run / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (step, home, expert, claude)))
        m = self.harvest()
        self.assertEqual((m["answers_by_answerer"]["qwen3.8-flash (twin)"], m["answers_by_answerer"]["claude (twin)"]), (3, 1))
        claude_row = json.loads((self.lake / "bronze" / "answers" / "claude" / "twin" / "solution.columns" / "all.jsonl").read_text())
        self.assertEqual((claude_row["use"], claude_row["value"]), ("eval_only", {"a": 3}))

    def test_the_rules_answer_every_test_in_the_same_layout(self):
        from dclab_core import lake as L
        from dclab_rnd.lake import answers

        sample = L.DecisionSample(sample_id="s1", decision_point="column.time", subject_kind="column", subject_ref="c",
                                  features={"column": "at", "dtype": "datetime"}, state_text="{}", options=["yes", "no"], label=True,
                                  rule_answer=True, provenance=L.Provenance(source="t", origin="construction", consent="ours", group="g", split="dev"))
        folder = self.lake / "gold" / "decisions" / "column.time" / "v1"
        folder.mkdir(parents=True)
        (folder / "dev.jsonl").write_text(sample.model_dump_json() + "\n")
        got = answers.record_rules(("lake",), self.lake)
        self.assertEqual(got, {"lake/column.time/dev.jsonl": 1})
        rec = answers.stored("dclab-rules", "lake", "column.time", "dev", self.lake)[0]
        self.assertEqual((rec["value"], rec["use"], rec["answerer"]), (True, "train", "dclab-rules"))


if __name__ == "__main__":
    unittest.main()

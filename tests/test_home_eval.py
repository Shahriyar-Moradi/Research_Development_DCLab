"""The Home evaluation suite (package H1): written requests, a reader of what Home understood, three arms and a gate."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import home  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_the_splits_have_the_planned_mix(self):
        seen_ids, seen_texts = set(), set()
        for split in ("dev", "sealed"):
            cases = home.load(split)
            self.assertEqual(len(cases), 36, split)
            mains = [c["gold"]["tracks"][0] for c in cases]
            self.assertEqual({t: mains.count(t) for t in home.TRACKS}, {t: 6 for t in home.TRACKS}, split)
            langs = [c["lang"] for c in cases]
            self.assertEqual((langs.count("fa"), langs.count("mixed")), (8, 4), split)
            self.assertEqual(sum(bool(c["gold"].get("both")) for c in cases), 5, split)  # five need two tracks done
            either = [c for c in cases if len(c["gold"]["tracks"]) >= 2 and not c["gold"].get("both")]
            self.assertGreaterEqual(len(either), 1, split)  # and some accept either of two tracks
            self.assertTrue(all(len(c["gold"]["tracks"]) >= 2 for c in cases if c["gold"].get("both")), split)
            traps = sorted(c["trap"] for c in cases if c.get("trap"))
            self.assertEqual(traps, sorted(home.TRAPS), split)
            for c in cases:
                self.assertNotIn(c["id"], seen_ids)
                self.assertNotIn(c["text"], seen_texts)
                seen_ids.add(c["id"]), seen_texts.add(c["text"])
                self.assertTrue(set(c["gold"]["tracks"]) <= set(home.TRACKS), c["id"])
                exits = c["gold"]["exit"] if isinstance(c["gold"]["exit"], list) else [c["gold"]["exit"]]
                self.assertTrue(set(exits) <= set(home.EXITS), c["id"])
                for data in c.get("data") or []:
                    self.assertIn(data, home.tables.TABLES, c["id"])
                if c.get("trap") == "missing_column":  # the column the request names is really absent
                    columns = {col for d in c["data"] for col in home.tables.build(d).columns}
                    self.assertNotIn(c["missing_column"], columns)
        self.assertEqual(home.fingerprint(home.SPLITS["sealed"]), home.recorded_fingerprint())

    def test_a_changed_sealed_file_is_refused(self):
        with mock.patch.object(home, "recorded_fingerprint", return_value="0" * 64):
            with self.assertRaises(ValueError):
                home.load("sealed")


class ScoringTests(unittest.TestCase):
    def test_scoring_reads_persian_and_english(self):
        self.assertEqual(home.score("لغو اشتراک ظرف ۳۰ روز", {"any": ["لغو"]}), "right")
        self.assertEqual(home.score("فروش هر منطقه", {"any": ["لغو"]}), "wrong")  # a Persian value is not matched by nothing
        self.assertEqual(home.score("Revenue per region", {"any": ["revenue"]}), "right")
        self.assertEqual(home.score(None, None), "abstained")
        self.assertEqual(home.score(None, {"any": ["x"]}), "missed")
        self.assertEqual(home.score("lifetime_spend", None), "wrong")
        self.assertEqual(home.score("پلن", {"any": ["پلن"]}), "right")  # Arabic yeh or kaf typed by the person still matches
        self.assertEqual(home.score("هزينه ماهانه", {"any": ["هزینه"]}), "right")


class ReaderTests(unittest.TestCase):
    def draft(self):
        return {
            "problem": "How much revenue per region last quarter?",
            "messages": [
                {"role": "user", "kind": "text", "text": "How much revenue per region last quarter?"},
                {"role": "agent", "kind": "analysis", "text": "Your data is ready: 2,000 rows and 9 columns, after 2 cleaning steps."},
                {"role": "agent", "kind": "text", "text": "Revenue in Dubai was 41,250 and order ORD-10007 was the largest."},
                {"role": "agent", "kind": "question", "text": "How many alerts can the team review?", "options": ["About 50 a day"],
                 "question": {"id": "q1"}},
                {"role": "agent", "kind": "question", "text": "Should I use a 45-day window?", "question": {"id": "q3"}},
            ],
            "questions": [{"id": "q1", "field": "target", "answered": None}, {"id": "q2", "field": "x", "superseded": True}],
            "plan": {"target": {"status": "stated", "source": "model", "quote": "revenue per region"},
                     "action": {"status": "stated", "source": "model", "quote": "we will call them"}},
            "understanding": {"target": "revenue per region", "action": "call"},
            "track_brief": {"analysis": {"measure": {"value": "lifetime_spend", "source": "model", "quote": "much revenue"},
                                         "period": {"value": "all rows", "source": "default"}}},
            "analysis": {"summary": {"rows": 2000, "columns": 9}},
            "cleaning_log": [{"step": "a"}, {"step": "b"}],
            "intent": {"track": "analysis", "tracks": ["analysis"], "status": "stated", "quote": "How much revenue"},
            "exit": "analysis",
        }

    def test_the_reader_finds_bad_quotes_uncited_numbers_exposed_rows_and_phantoms(self):
        obs = home.read_draft(self.draft(), columns={"order_id", "region", "amount"}, id_values={"ORD-10007", "ORD-10008"})
        self.assertEqual(obs["track"], "analysis")
        self.assertEqual(obs["questions"], 1)
        self.assertEqual(obs["exit"], "analysis")
        self.assertEqual(obs["fields"]["target"], "revenue per region")
        self.assertEqual(obs["fields"]["measure"], "lifetime_spend")
        self.assertNotIn("period", obs["fields"])  # a default the person did not state is not an understanding
        self.assertEqual(obs["quote_invalid"], ["plan.action"])
        # 2,000, 9 and 2 are the data's own numbers, ORD-10007 is a name, a button's "50" is the product's words; the
        # model's own question invents 45
        self.assertEqual(obs["uncited"], ["41250", "45"])
        self.assertEqual(obs["rows_exposed"], ["ORD-10007"])
        self.assertEqual(obs["phantom"], ["lifetime_spend"])

    def test_numbers_in_words_and_with_units_are_read_as_numbers(self):
        draft = {"problem": "Show refunds", "messages": [{"role": "agent", "kind": "text", "text": "About sixty days, or 4.5k orders, for the top-200 customers."}]}
        self.assertEqual(sorted(home.read_draft(draft, set(), set())["uncited"]), ["-200", "4.5k", "60"])  # cited.py reads top-200 as -200: caught either way
        draft["problem"] = "Show refunds for sixty days and 4.5k orders, top-200 customers"
        self.assertEqual(home.read_draft(draft, set(), set())["uncited"], [])

    def test_automatic_cleaning_that_drops_many_rows_counts_as_destructive(self):
        draft = {"problem": "x", "analysis": {"summary": {"rows": 1800}},
                 "cleaning_log": [{"step": "duplicates", "columns": [], "changed": 200}, {"step": "trim", "changed": 900}]}
        self.assertEqual(home.read_draft(draft, set(), set())["destructive_unconfirmed"], 1)  # 200 of 2,000 rows: 10%
        draft["cleaning_log"][0]["changed"] = 20
        self.assertEqual(home.read_draft(draft, set(), set())["destructive_unconfirmed"], 0)

    def test_the_missing_column_trap_needs_the_column_refused_and_the_rest_understood(self):
        case = next(c for c in home.load("dev") if c.get("trap") == "missing_column")
        obs = home.empty_observation()
        self.assertFalse(home._trap(case, obs))  # understood nothing: not a pass
        obs["fields"] = {"breakdown": "by segment"}
        self.assertTrue(home._trap(case, obs))
        obs["fields"] = {"breakdown": "by segment", "measure": "average lifetime spend"}  # the absent column, spelled out
        self.assertFalse(home._trap(case, obs))
        obs["fields"] = {"breakdown": "by segment", "measure": "average spend"}  # a measure where the request supports none
        self.assertFalse(home._trap(case, obs))

    def test_todays_home_reads_every_request_as_a_prediction(self):
        draft = self.draft()
        for key in ("intent", "exit", "track_brief"):
            draft.pop(key)
        draft["messages"].append({"role": "agent", "kind": "summary", "text": "Here is the solution I would build.", "actions": ["build"]})
        obs = home.read_draft(draft, columns=set(), id_values=set())
        self.assertEqual((obs["track"], obs["tracks"], obs["exit"]), ("prediction", ["prediction"], "ml_project"))


class RunTests(unittest.TestCase):
    def test_todays_home_is_scored_without_a_model(self):
        cases = [c for c in home.load("dev") if c["id"] in ("d-p1", "d-a1", "d-v1")]
        self.assertEqual(len(cases), 3)
        result = home.measure(cases)
        rows = {r["case"]: r for r in result["rows"]}
        self.assertEqual(set(rows["d-p1"]["scores"]), {"rules"})
        self.assertEqual(rows["d-p1"]["scores"]["rules"]["track"], "right")
        self.assertEqual(rows["d-a1"]["scores"]["rules"]["track"], "right")  # H2: Home reads the kind of help (HOME-001 kept the baseline)
        self.assertEqual(rows["d-v1"]["scores"]["rules"]["track"], "right")
        # H1 found that an "I'm not sure" to the column question replaced the outcome Home understood; H4 keeps it
        self.assertEqual(rows["d-p1"]["observed"]["rules"]["lost"], [])
        self.assertEqual(result["requests"], 0)
        s = result["summary"]["rules"]
        self.assertEqual(s["cases"], 3)
        self.assertAlmostEqual(s["track_accuracy"], 1.0, places=3)
        verdict = home.verdict(result, "dev")
        self.assertTrue(verdict["status"].startswith("not scored"))
        self.assertTrue(verdict["checks"]["rules_track_accuracy"])

    def test_a_run_touches_neither_the_workspace_files_nor_the_roles(self):
        import os

        seen = {}
        real = home.tables.write

        def write(name, folder):
            seen.update(files=os.environ.get("DCLAB_FILES_URL"), roles=os.environ.get("DCLAB_ROLES"))
            return real(name, folder)

        case = next(c for c in home.load("dev") if c["id"] == "d-a1")
        with mock.patch.dict(os.environ, {"DCLAB_FILES_URL": "s3://somebody-elses-bucket/x", "DCLAB_ROLES": "problem_analyst"}), \
                mock.patch.object(home.tables, "write", side_effect=write):
            home.observe(case)
            self.assertEqual(os.environ["DCLAB_FILES_URL"], "s3://somebody-elses-bucket/x")  # restored afterwards
        self.assertEqual(seen, {"files": None, "roles": "none"})

    def test_an_arm_asked_for_that_cannot_run_makes_the_run_unscored(self):
        cases = [c for c in home.load("dev") if c["id"] == "d-v1"]
        result = home.measure(cases, client_for=lambda draft_id: None, plain_for=lambda case_id: None)
        self.assertEqual(result["arms"], ["rules", "model", "plain"])
        self.assertEqual(len(result["failures"]), 2)
        self.assertEqual(home.verdict(result, "sealed")["status"], "not scored")

    def test_the_verdict_with_the_model(self):
        arm = {"track_accuracy": 0.9, "quote_invalid": 0, "uncited": 0, "rows_exposed": 0, "destructive_unconfirmed": 0,
               "median_questions": 2, "traps": {t: True for t in home.TRAPS}, "new_wrong": []}
        result = {"arms": ["rules", "model", "plain"], "requests": 40, "failures": [],
                  "summary": {"rules": {**arm, "track_accuracy": 0.75}, "model": dict(arm), "plain": dict(arm)}}
        self.assertEqual(home.verdict(result, "sealed")["status"], "pass")
        self.assertTrue(home.verdict(result, "dev")["status"].startswith("pass (dev"))
        result["summary"]["model"]["uncited"] = 1
        self.assertEqual(home.verdict(result, "sealed")["status"], "fail")
        result["summary"]["model"]["uncited"] = 0
        del result["summary"]["plain"]
        self.assertEqual(home.verdict(result, "sealed")["status"], "not scored")  # an arm asked for did not run
        result["summary"]["plain"], result["requests"] = dict(arm), 0
        self.assertEqual(home.verdict(result, "sealed")["status"], "not scored")  # a model run that sent nothing

    def test_the_plain_arm_takes_the_models_answer_as_given(self):
        case = next(c for c in home.load("dev") if c["id"] == "d-a1")
        answer = {"tracks": ["analysis"], "exit": "analysis", "fields": {"measure": "revenue", "breakdown": "region", "period": "last quarter"}}

        class Scripted:
            def __init__(self):
                self.sent = []

            def complete(self, messages, tools=None, **kw):
                self.sent.append(messages)
                return {"content": "Here it is:\n```json\n" + json.dumps(answer) + "\n```"}

        client = Scripted()
        obs = home.read_plain(case, client)
        self.assertEqual((obs["track"], obs["exit"], obs["fields"]["breakdown"]), ("analysis", "analysis", "region"))
        text = json.dumps(client.sent)
        self.assertIn("region", text)  # the request names the table's columns
        self.assertNotIn("Dubai", text)  # never a value from the table
        broken = type("Broken", (), {"complete": lambda self, m, t=None, **k: {"content": "no json here"}})()
        self.assertEqual(home.read_plain(case, broken)["track"], None)

    def test_a_live_plan_is_printed_before_anything_is_sent(self):
        gateway = mock.Mock()
        gateway.route.return_value = ("standard", None)
        gateway.available.return_value = False
        plan = home.plan(gateway, home.load("dev"), 1500)
        self.assertEqual(plan["cases"], 36)
        self.assertFalse(plan["available"])
        self.assertGreater(plan["max_requests"], 36)



class CommandLineTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        from dclab_rnd.agent_eval.home import __main__ as cli

        self.cli = cli
        self.results = Path(tempfile.mkdtemp()) / "results"
        self.patches = [mock.patch.object(home, "RESULTS", self.results), mock.patch.object(cli, "RESULTS", self.results)]
        for p in self.patches:
            p.start()
        self.gateway = mock.Mock()
        self.patches.append(mock.patch("dclab_rnd.models.gateway.for_workspace", return_value=self.gateway))
        self.patches[-1].start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def plan(self, **kw):
        return mock.patch.object(self.cli, "plan", return_value={"available": True, "local": False, "max_eur": 1.0, **kw})

    def test_nothing_is_sent_or_stored_without_yes(self):
        with self.plan():
            self.assertEqual(self.cli.main(["--split", "sealed"]), 0)
        self.gateway.client.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_refusals(self):
        with self.plan(available=False):
            self.assertEqual(self.cli.main(["--split", "dev", "--yes", "--cap-eur", "2"]), 2)
        with self.plan(max_eur=None):  # a remote model with no price: no cap can stop it
            self.assertEqual(self.cli.main(["--split", "dev", "--yes", "--cap-eur", "2"]), 2)
        with self.plan():
            self.assertEqual(self.cli.main(["--split", "dev", "--yes"]), 2)  # no cap
        with self.plan(), mock.patch.object(self.cli, "sealed_runs", return_value=[Path("HOME-009_sealed.json")]):
            self.assertEqual(self.cli.main(["--split", "sealed", "--yes", "--cap-eur", "2"]), 2)
        self.gateway.client.assert_not_called()

    def test_a_failed_request_stores_an_unscored_record(self):
        class Failing:
            spent_eur, run_limit_eur = 0.0, None

            def complete(self, messages, tools=None, **kw):
                raise RuntimeError("refused by the cap")

        self.gateway.client.side_effect = lambda purpose, draft_id=None: Failing()
        case = [c for c in home.load("dev") if c["id"] == "d-v1"]
        with self.plan(local=True, max_eur=0.0), mock.patch.object(self.cli, "load", return_value=case):
            self.assertEqual(self.cli.main(["--split", "dev", "--yes", "--cap-eur", "1"]), 0)
        stored = list(self.results.glob("HOME-*_dev_unscored.json"))
        self.assertEqual(len(stored), 1)
        record = json.loads(stored[0].read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertNotIn("rows", record)  # no values or scores from a run that cannot be scored

    def test_one_cap_bounds_every_client_of_the_run(self):
        made = []

        def client(purpose, draft_id=None):
            c = mock.Mock(spent_eur=0.0, run_limit_eur=None)
            made.append(c)
            return c

        self.gateway.client.side_effect = client
        budget = self.cli.Budget(self.gateway, 2.0)
        budget.client("a")
        made[0].spent_eur = 1.5
        budget.client("home-eval-plain")
        made[1].spent_eur = 0.4
        budget.client("b")
        for got, want in zip([c.run_limit_eur for c in made], [2.0, 0.5, 0.1]):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(budget.spent_eur(), 1.9)


if __name__ == "__main__":
    unittest.main()

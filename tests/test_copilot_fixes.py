"""Model-written fixes for the notebook copilot's findings (package A3.4): findings never change; a fix cites its rule and pitfall."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.copilot import review_notebook  # noqa: E402
from dclab_rnd.copilot import fixes  # noqa: E402
from dclab_rnd.copilot.render import annotate_notebook, to_html, to_markdown  # noqa: E402

LEAKY = ROOT / "dclab_rnd" / "copilot" / "examples" / "leaky_bank_marketing.ipynb"
CLEAN = ROOT / "dclab_rnd" / "copilot" / "examples" / "clean_bank_marketing.ipynb"


class Scripted:
    def __init__(self, content):
        self.content, self.seen, self.verdicts = content, [], []

    def complete(self, messages, tools=None, max_tokens=None, **_):
        self.seen.append(messages[-1]["content"])
        if isinstance(self.content, Exception):
            raise self.content
        return {"content": self.content, "tool_calls": [], "usage": {}, "assistant_message": {"role": "assistant", "content": self.content}}

    def output(self, ok, reason=""):
        self.verdicts.append((ok, reason))


def precedent(finding):
    return next((p["record_id"] for p in finding["proof"] if p["type"] in fixes.PRECEDENT_TYPES), None)


def good_reply(report, skip=()):
    """What a careful model would answer: two sentences per finding, the first citing the rule, the second the pitfall."""
    items = []
    for n, f in enumerate(report["findings"]):
        if n in skip:
            continue
        second = f" The measured case shows why [{precedent(f)}]." if precedent(f) else ""
        items.append({"finding": n, "fix": f"Change this before it reaches the model [{f['rules'][0]}].{second}"})
    return json.dumps({"fixes": items})


class FixTests(unittest.TestCase):
    def setUp(self):
        self.report = review_notebook(LEAKY)

    def test_the_leaky_notebook_keeps_every_finding_and_gets_a_fix_that_cites_its_rule(self):
        before = copy.deepcopy(self.report)
        out = fixes.with_fixes(self.report, Scripted(good_reply(self.report)))
        self.assertEqual(self.report, before)  # the review passed in is not touched
        stripped = copy.deepcopy(out)
        for f in stripped["findings"]:
            f.pop("fix", None)
        self.assertEqual(stripped, before)  # the same findings, in the same order, with the same proof
        self.assertEqual(len(out["findings"]), 10)
        for f in out["findings"]:
            self.assertIn(f["rules"][0], f["fix"]["cites"], f["detector"])
            if precedent(f):
                self.assertIn(precedent(f), f["fix"]["cites"], f["detector"])  # the matching pitfall or precedent record
            self.assertEqual(f["fix"]["written_by"], "model")
        self.assertEqual(next(f for f in out["findings"] if f["detector"] == "resample_before_split")["fix"]["cites"], ["DCLAB-R02", "PIT-003"])

    def test_a_fix_without_its_rule_or_pitfall_or_with_a_made_up_number_is_dropped(self):
        by = {}
        for n, f in enumerate(self.report["findings"]):
            by.setdefault(f["detector"], (n, f))
        (n0, f0), (n1, f1), (n2, f2), (n3, f3) = (by[d] for d in ("known_leakage_column", "aggregate_before_split", "resample_before_split", "preprocess_before_split"))
        n4, f4 = by["holdout_reuse"]
        n5, f5 = by["missing_random_state"]  # a rule and no pitfall record
        replies = {"fixes": [
            {"finding": n0, "fix": f"Drop it [{f0['rules'][0]}]. The gap was 0.99 in the campaigns [{precedent(f0)}]."},  # the number is not in the record; what is left cites no precedent
            {"finding": n1, "fix": f"Do it inside each fold [{f1['rules'][0]}]."},  # a pitfall exists for this finding but is not cited
            {"finding": n2, "fix": "Split first, then fit."},  # cites nothing
            {"finding": n3, "fix": f"First [{f3['rules'][0]}]. Next [{precedent(f3)}]. Last [{f3['rules'][0]}]."},  # only two sentences are kept
            {"finding": n4, "fix": f"This is production-ready [{f4['rules'][0]}]. Fit inside the fold [{f4['rules'][0]}] [{precedent(f4)}]."},
            {"finding": n5, "fix": f"Pass a seed [{f5['rules'][0]}]."},  # nothing but a rule to cite: that is enough
        ]}
        model = Scripted(json.dumps(replies))
        written = fixes.write_fixes(self.report["findings"], model)
        self.assertEqual(sorted(written), sorted([n3, n4, n5]))
        self.assertEqual(written[n3]["text"], "First. Next.")
        self.assertEqual(written[n4]["text"], "Fit inside the fold.")  # production-ready is removed
        self.assertEqual(written[n5]["cites"], [f5["rules"][0]])
        self.assertFalse(model.verdicts[0][0])

    def test_without_a_model_or_with_an_unreadable_reply_the_review_is_exactly_what_it_was(self):
        for client in (None, Scripted("Sure! Here are some fixes."), Scripted(json.dumps({"fixes": "no"})), Scripted(RuntimeError("BudgetExceeded: the cap is reached"))):
            self.assertEqual(fixes.with_fixes(self.report, client), self.report)
        self.assertEqual(fixes.with_fixes(review_notebook(CLEAN), Scripted(good_reply(self.report)))["findings"], [])  # nothing to fix: no request needed

    def test_the_model_sees_the_findings_and_records_never_the_notebooks_code_or_data(self):
        model = Scripted(good_reply(self.report))
        fixes.write_fixes(self.report["findings"], model)
        prompt = "\n".join(model.seen)
        self.assertEqual(len(model.seen), 2)  # ten findings, eight to a request
        for line in ('pd.read_parquet(ROOT / "data/public/bank_marketing/X.parquet")', "df_bal = pd.concat(", "best_acc, best_depth = 0, None"):
            self.assertNotIn(line, prompt)  # no line of the notebook
        self.assertIn("Finding 9 (medium)", prompt)
        self.assertIn("[PIT-003]", prompt)

    def test_what_a_finding_quotes_from_the_code_is_replaced_before_the_request(self):
        self.assertEqual(fixes.plain("Loads `/Users/alice/Q7KxAcme/customers.csv`, uses `final Q7Kxcolumn` and `duration` and `imblearn.pipeline.Pipeline`, e.g. `df = df.drop(columns=['x'])`."),
                         "Loads `<code>`, uses `<code>` and `duration` and `imblearn.pipeline.Pipeline`, e.g. `<code>`.")
        path_finding = {"detector": "absolute_data_path", "severity": "low", "cell": 2, "line": 3, "title": "Hard-coded absolute data path",
                        "message": "`/Users/alice/Q7KxAcme/customers_Q7Kxfile.csv` only exists on one machine.", "suggestion": "Use a relative path.",
                        "rules": ["DCLAB-R21"], "proof": [{"record_id": "DCLAB-R21", "type": "rule", "title": "Store claims", "text": "Record it.", "citations": []}]}
        self.assertNotIn("Q7Kx", fixes.prompt_for([path_finding]))
        self.assertIn("hard-coded absolute path that exists on one machine only", fixes.prompt_for([path_finding]))  # a fixed text, not the finding's

    def test_what_two_detectors_quote_from_the_code_never_leaves_even_in_a_plain_looking_name(self):
        quoted = {"detector": "suspicious_column_name", "severity": "info", "cell": 1, "line": 1, "title": "Check when `closed_Q7Kxvalue` is created",
                  "message": "The name `closed_Q7Kxvalue` suggests a value that may only exist after the outcome.", "suggestion": "Confirm that `closed_Q7Kxvalue` exists at prediction time.",
                  "rules": ["DCLAB-R05"], "proof": [{"record_id": "DCLAB-R05", "type": "rule", "title": "Name rule", "text": "A name is not proof.", "citations": []}]}
        path = {**quoted, "detector": "absolute_data_path", "title": "Hard-coded absolute data path",
                "message": "`/Users/a/`Q7Kxsecret`/x.csv` only exists on one machine.", "suggestion": "Load from a relative path.", "rules": ["DCLAB-R21"]}
        prompt = fixes.prompt_for([quoted, path])
        self.assertNotIn("Q7Kx", prompt)
        self.assertIn("Check when `<column>` is created", prompt)
        self.assertIn("hard-coded absolute path", prompt)
        known = {**quoted, "detector": "known_leakage_column", "title": "`duration` is a known leakage column", "message": "`duration` was measured as post-outcome leakage.", "suggestion": "Drop `duration`."}
        self.assertIn("`duration` is a known leakage column", fixes.prompt_for([known]))  # a name from the index stays

    def test_a_long_review_is_asked_in_pieces_and_a_boolean_is_not_a_finding_number(self):
        findings = [{**self.report["findings"][0], "line": n} for n in range(20)]  # the same finding twenty times: 3 requests
        model = Scripted(json.dumps({"fixes": []}))
        fixes.write_fixes(findings, model)
        self.assertEqual(len(model.seen), 3)
        self.assertIn("Finding 8 (", model.seen[1])  # numbered by place in the whole review
        self.assertEqual(fixes._parse(json.dumps({"fixes": [{"finding": True, "fix": "x"}, {"finding": 2, "fix": "y"}]})), {2: "y"})

    def test_a_fix_keeps_its_finding_when_the_review_is_computed_again(self):
        keys = [fixes.key_of(f) for f in self.report["findings"]]
        self.assertEqual(len(set(keys)), len(keys))  # unique: the review never reports a detector twice on a line
        self.assertEqual(keys[0], "known_leakage_column:3:2")

    def test_the_fix_is_shown_beside_the_finding_and_labelled(self):
        out = fixes.with_fixes(self.report, Scripted(good_reply(self.report)))
        text = to_markdown(out)
        self.assertEqual(text.count("Written by a model from the rule and the pitfall record"), 10)
        self.assertIn("**Fix.** Drop `duration`", text)  # the deterministic fix is still there
        self.assertIn("Written by a model", to_html(out))
        self.assertNotIn("Written by a model", to_markdown(self.report) + to_html(self.report))
        notebook = annotate_notebook(LEAKY, out)
        notes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown" and c.get("metadata", {}).get("dclab_copilot")]
        self.assertEqual((len(notes), sum(n.count("Written by a model") for n in notes)), (4, 10))  # a review cell per flagged cell, a fix per finding


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        import contextlib
        import io

        from dclab_rnd.copilot.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["review", str(LEAKY), "--json", *args])
        return code, json.loads(out.getvalue()), err.getvalue()

    def test_fixes_flag_without_a_model_prints_the_same_review_and_says_so(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            _, plain, _ = self.run_cli()
            code, with_flag, err = self.run_cli("--fixes")
        self.assertEqual((code, with_flag), (0, plain))
        self.assertIn("No model is configured", err)

    def test_fixes_flag_with_a_model_adds_fixes_and_keeps_every_finding(self):
        import os
        import tempfile
        from unittest import mock

        from dclab_rnd import models
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.models.usage import FileUsage

        report = review_notebook(LEAKY)
        reply = good_reply(report)

        class Writer(Scripted):
            def __init__(self):
                super().__init__(reply)
        keyed = {"OPENAI_API_KEY": "k", "DCLAB_NO_LIVE_MODELS": "", **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}}
        with mock.patch.dict(os.environ, keyed):
            models.install(Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Writer()))
            self.addCleanup(models.install, None)
            _, plain, _ = self.run_cli()
            code, written, err = self.run_cli("--fixes")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual([{k: v for k, v in f.items() if k != "fix"} for f in written["findings"]], plain["findings"])
        self.assertTrue(all("fix" in f for f in written["findings"]))
        with mock.patch.dict(os.environ, keyed):  # a model that is asked and writes nothing usable: the CLI says so
            models.install(Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Scripted("Sure! Here are some fixes.")))
            code, unchanged, err = self.run_cli("--fixes")
        self.assertEqual((code, unchanged), (0, plain))
        self.assertIn("wrote no fix", err)


if __name__ == "__main__":
    unittest.main()

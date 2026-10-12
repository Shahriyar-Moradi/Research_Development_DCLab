"""Package 18.7: the reporter. A model's summaries are added above a report section's lines, kept sentence by sentence
only when the host's check passes them (dclab_rnd.cited.check); the lines written by code always stay."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.agents import ScriptedAdvisor
    from dclab_core.contracts import ReportSection, SolutionReport
    from dclab_core.report import PROSE_LABEL, write_prose
    from dclab_rnd import cited
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""

RESULTS = ("The final model, `lightgbm` on the `raw` recipe, scored ROC AUC 0.746 on the holdout, used once.\n"
           "Its bootstrap interval runs from 0.701 to 0.789 (iid row bootstrap).\n"
           "- Baseline `majority_class` on the same holdout: accuracy 0.816, ROC AUC 0.500.")


def report():
    return SolutionReport(title="Solution report: churn", sections=[
        ReportSection(title="The problem as understood", body="- **Task**: binary", cites=[]),
        ReportSection(title="Results, with their intervals", body=RESULTS, cites=["final.evidence.holdout_metrics.roc_auc"])])


@unittest.skipIf(MISSING, MISSING)
class ReporterTest(unittest.TestCase):
    def test_sentences_are_kept_only_when_their_numbers_are_in_the_lines_they_cite(self):
        advisor = ScriptedAdvisor({"reporter": {"summaries": [{"title": "Results, with their intervals", "text":
            "The model scored ROC AUC 0.746 on the holdout, against ROC AUC 0.500 for the baseline [L1][L3]. "
            "That is a 49% lift over the baseline [L1][L3]. "
            "It is ready for production [L1]. "
            "Its bootstrap interval runs from 0.701 to 0.789 [L2]."}]}})
        out, info = write_prose(advisor, report(), cited.check)
        body = out.sections[1].body
        self.assertTrue(body.startswith(PROSE_LABEL + "The model scored ROC AUC 0.746 on the holdout, against ROC AUC 0.500"))
        self.assertIn("runs from 0.701 to 0.789.", body.split("\n")[0])
        self.assertNotIn("49%", body)  # a computed number: dropped
        self.assertNotIn("ready for production", body)  # an overclaim: dropped
        self.assertNotIn("[L", body)  # the line labels are the model's citations, not the reader's
        self.assertTrue(body.endswith(RESULTS))  # every line code wrote stays, after the summary
        self.assertEqual((info["source"], info["kept"], info["dropped"]), ("advisor", 2, 2))
        self.assertEqual(out.sections[0].body, "- **Task**: binary")  # not a prose section
        [(purpose, evidence)] = advisor.requests
        self.assertEqual(purpose, "reporter")
        self.assertEqual(evidence["sections"][0]["lines"]["L2"], RESULTS.split("\n")[1])

    def test_true_numbers_beside_the_wrong_subject_or_claim_are_dropped(self):
        misleading = [
            "The final model reached an accuracy of 0.816 [L1][L3].",           # the baseline's accuracy, given to the model
            "The holdout was used once, so its score is unbiased [L1].",        # unbiased: no line says it
            "The model is not yet production-ready [L1].",                      # a denial: the report's rule is strict
            "Its interval is 0.701 to 0.789 percent wide [L2].",                # a number not written as the line writes it
            "It scored <img src=x onerror=alert(1)> 0.746 ROC AUC [L1].",       # no subject beside 0.746 in its own words
        ]
        for text in misleading:
            with self.subTest(text=text):
                advisor = ScriptedAdvisor({"reporter": {"summaries": [{"title": "Results, with their intervals", "text": text}]}})
                self.assertEqual(write_prose(advisor, report(), cited.check)[0], report())

    def test_kept_prose_is_escaped_and_renders(self):
        from dclab_core import report as rp

        advisor = ScriptedAdvisor({"reporter": {"summaries": [{"title": "Results, with their intervals",
                                                               "text": "The `lightgbm **strong` recipe scored ROC AUC 0.746 [L1]."}]}})
        out, _ = write_prose(advisor, report(), cited.check)
        self.assertIn("\\*\\*", out.sections[1].body)  # Markdown in a model's text is escaped like a record's
        self.assertIn("written by a model", rp.to_markdown(out))
        self.assertIn("written by a model", rp.to_html(out))
        try:
            import reportlab  # noqa: F401
        except ImportError:
            return
        import tempfile
        rp.to_pdf(out, Path(tempfile.mkdtemp()) / "r.pdf")

    def test_nothing_kept_leaves_the_deterministic_report(self):
        bad = ScriptedAdvisor({"reporter": {"summaries": [{"title": "Results, with their intervals", "text": "It scored 0.75 [L1]."}]}})
        out, info = write_prose(bad, report(), cited.check)
        self.assertEqual(out, report())
        self.assertEqual((info["source"], info["kept"]), ("fallback", 0))

    def test_no_check_or_no_advisor_keeps_no_model_prose(self):
        advisor = ScriptedAdvisor({"reporter": {"summaries": [{"title": "Results, with their intervals", "text": "It scored 0.746 [L1]."}]}})
        self.assertEqual(write_prose(advisor, report(), None)[0], report())  # this library never keeps unchecked prose
        self.assertEqual(advisor.requests, [])
        self.assertEqual(write_prose(None, report(), cited.check)[1]["reason"], "no advisor")

    def test_an_unknown_section_or_a_malformed_answer_is_ignored(self):
        odd = ScriptedAdvisor({"reporter": {"summaries": [{"title": "Invented section", "text": "Nothing [L1]."}, "not an item"]}})
        out, info = write_prose(odd, report(), cited.check)
        self.assertEqual(out, report())
        broken = ScriptedAdvisor({"reporter": {"summaries": "all good"}})
        self.assertEqual(write_prose(broken, report(), cited.check)[0], report())


@unittest.skipIf(MISSING, MISSING)
class HarnessTest(unittest.TestCase):
    def test_the_measurement_rechecks_every_kept_sentence_and_the_lines(self):
        from dclab_rnd.agent_eval.roles import measure_reporter, reporter_verdict

        echo = ScriptedAdvisor({"reporter": lambda ev: {"summaries": [{"title": s["title"], "text": list(s["lines"].values())[0] + " [L1]"}
                                                                      for s in ev["sections"]]}})
        result = measure_reporter(echo)
        self.assertEqual(len(result["rows"]), 3)
        self.assertTrue(all(r["lines_intact"] and not r["failed_recheck"] for r in result["rows"]))
        self.assertEqual(reporter_verdict(result)["status"], "pass")
        self.assertEqual(reporter_verdict(measure_reporter(None))["status"], "not scored")

    def test_a_kept_sentence_the_report_cannot_trace_fails_the_gate(self):
        from unittest import mock

        from dclab_core import report as core_report
        from dclab_rnd.agent_eval.roles import measure_reporter, reporter_verdict

        def careless(advisor, report, check):  # a write_prose that let an untraced number through
            sections = [s.model_copy(update={"body": "In short: It scored 0.999.\n\n" + s.body}) if s.title in core_report.PROSE_SECTIONS
                        else s for s in report.sections]
            return report.model_copy(update={"sections": sections}), {"source": "advisor", "kept": 1, "dropped": 0, "reason": ""}
        with mock.patch.object(core_report, "write_prose", careless):
            self.assertEqual(reporter_verdict(measure_reporter(object()))["status"], "fail")

    def test_a_private_project_sends_no_report_to_a_model(self):
        from dclab_rnd.agent_eval.roles import _report_project
        from dclab_rnd.studio import solution_report

        store, pid = _report_project("binary")
        project = store.get(pid)
        project["settings"]["private"] = True
        store.save(project)
        advisor = ScriptedAdvisor({"reporter": {"summaries": []}})
        solution_report.build(store, pid, analyse=False, advisor=advisor)
        self.assertEqual(advisor.requests, [])


if __name__ == "__main__":
    unittest.main()

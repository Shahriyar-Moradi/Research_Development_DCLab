"""Package 17.7: the solution report, from the records (dclab_core.report and the studio's adapter).

The three replica projects (fraud with its label file, ticket routing, delivery time) run end to end on the fast
profile; their reports render to Markdown, HTML and PDF, every number written in the Markdown is traced to a record
value, and no sentence calls a model production-ready. A report built from missing or broken records says what is
missing instead of failing.
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:  # requirements/ci.txt
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""

try:
    import reportlab  # noqa: F401
except ImportError as error:  # an optional extra of dclab-core
    REPORTLAB_MISSING = str(error)
else:
    REPORTLAB_MISSING = ""

# the test's own reading of "a number in the text": digits not glued to a name (x1, top_2, v1.2), a sign when it stands
# alone, one decimal part, a percent sign
NUMBER = re.compile(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?%?")
PRODUCTION_READY = re.compile(r"production[\s\-‐-―]*ready|ready\s+for\s+production", re.I)
SECTIONS = ("The problem as understood", "What the data shows", "Decisions and reasons", "Results, with their intervals",
            "Limits, and what the data cannot show", "Next steps")


def written_value(token):
    percent = token.endswith("%")
    body = token.rstrip("%")
    decimals = len(body.split(".")[1]) if "." in body else 0
    value = float(body)
    return (value / 100, decimals + 2) if percent else (value, decimals)


def same_number(token, value):
    """The record's value, rounded to the decimals the text writes, is the number written."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    written, decimals = written_value(token)
    return abs(written - value) <= 0.5 * 10 ** -decimals + 1e-9 * max(1.0, abs(value))


class TraceMixin:
    def assert_traced(self, report, ctx):
        from dclab_core import report as rp

        markdown = rp.to_markdown(report)
        sources = rp.sources(report)
        tokens = NUMBER.findall(markdown)
        self.assertTrue(tokens)
        for token in tokens:
            with self.subTest(number=token):
                self.assertIn(token, sources, f"{token} is written but not traced")
                path = sources[token]
                value = rp.lookup(ctx, path)  # the record itself, read again
                self.assertTrue(same_number(token, value), f"{token} is not the record's {path} = {value}")
                self.assertEqual(report.numbers[path], value)
        # every digit the text writes is a traced number or part of a name
        spans = [m.span() for m in NUMBER.finditer(markdown)]
        for run in re.finditer(r"\d+", markdown):
            inside = any(a <= run.start() < b for a, b in spans)
            before = markdown[run.start() - 1] if run.start() else " "
            glued = before.isalpha() or before == "_" or (before == "." and run.start() > 1 and markdown[run.start() - 2].isalnum())
            self.assertTrue(inside or glued, f"digits {run.group()!r} at {run.start()} escape the trace")
        self.assertEqual(rp.check(report, ctx), [])
        return markdown

    def assert_no_production_claim(self, report):
        from dclab_core import report as rp

        for text in [rp.to_markdown(report), rp.to_html(report), *report.limits]:
            self.assertIsNone(PRODUCTION_READY.search(text))


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class UnitTest(TraceMixin, unittest.TestCase):
    def test_numbers_are_read_as_written_and_names_are_not_numbers(self):
        from dclab_core import report as rp

        text = "MAE 8.552 from -11.3 to +16.5; 6.0% at the cap; columns x1, top_2 and v1.2; WF-08; 2025-05-05"
        self.assertEqual([t for t, _ in rp.numbers_in(text)], ["8.552", "-11.3", "16.5", "6.0%", "08", "2025", "05", "05"])
        self.assertTrue(rp.matches("6.0%", 0.0595))
        self.assertTrue(rp.matches("0.503", 0.5031055))
        self.assertFalse(rp.matches("0.504", 0.5031055))
        self.assertFalse(rp.matches("60", 61.0))
        self.assertFalse(rp.matches("1", True))
        self.assertEqual(rp.fmt(0.0, "num"), "0.000")
        self.assertEqual(rp.fmt(60.0, "auto"), "60")
        self.assertEqual(rp.fmt(1500, "num"), "1500")
        self.assertEqual(rp.fmt(0.00012), "0.00012")
        self.assertEqual(rp.fmt(0.005, "pct"), "0.5%")

    def test_lookup_finds_dotted_keys_and_the_numbers_of_texts_and_keys(self):
        from dclab_core import report as rp

        ctx = {"plan": {"metric.primary": {"choice": "mae", "reason": "the cap at 60 and 5.9% of rows"}},
               "final": {"evidence": {"intervals": {"80%": {"coverage": 0.81}}}}}
        self.assertEqual(rp.lookup(ctx, "plan.metric.primary.choice"), "mae")
        self.assertAlmostEqual(rp.lookup(ctx, "plan.metric.primary.reason@1"), 0.059)
        self.assertAlmostEqual(rp.lookup(ctx, "final.evidence.intervals.80%#0"), 0.8)
        self.assertEqual(rp.lookup(ctx, "final.evidence.intervals.80%.coverage"), 0.81)
        with self.assertRaises(KeyError):
            rp.lookup(ctx, "plan.metric.primary.reason@2")

    def test_an_empty_build_says_what_is_missing(self):
        from dclab_core import report as rp

        report = rp.build()
        markdown = rp.to_markdown(report)
        self.assertEqual([s.title for s in report.sections][:6], list(SECTIONS))
        self.assertIn("No problem brief was given", markdown)
        self.assertIn("The holdout has not been scored", markdown)
        self.assertTrue(any("No final stage record" in limit for limit in report.limits))
        self.assertTrue(any("No record for these stages" in limit for limit in report.limits))
        self.assertIn("Run the stages that have no record yet", markdown)
        self.assertEqual(rp.check(report, rp.context()), [])
        self.assertEqual(NUMBER.findall(markdown), [])
        rp.to_html(report)

    def test_broken_inputs_are_stated_limits_not_crashes(self):
        from dclab_core import report as rp

        records = {"data": {"task": "binary", "primary_metric": "average_precision",
                            "evidence": {"source_rows": 100, "train_rows": 80, "holdout_rows": 20,
                                         "holdout_split": "last 20 of 100 rows by `created_at`",
                                         "target_summary": {"positive_rate_train": 0.05, "positives_train": 4,
                                                            "majority_baseline_accuracy_train": 0.95}}},
                   "final": "not a record", "models": None}
        brief = {"goal": {"status": "stated", "value": "find fraud"}}  # a stated fact without its quote: invalid
        report = rp.build(brief=brief, findings={"findings": "nope"}, plan=[{"decision": "metric.primary", "choice": "mae",
                                                                             "why": "MAE reads in minutes"}], records=records)
        ctx = rp.context(brief=brief, findings={"findings": "nope"}, plan=[{"decision": "metric.primary", "choice": "mae",
                                                                            "why": "MAE reads in minutes"}], records=records)
        markdown = self.assert_traced(report, ctx)
        self.assertIn("could not be read", markdown)
        self.assertIn("The table had 100 rows; the run trained on 80 and locked 20", markdown)
        self.assertIn("**metric.primary**: mae", markdown)
        self.assertIn("audit leakage", " ".join(report.limits))
        self.assertIn("The holdout has not been scored", markdown)
        self.assertEqual(rp.build(records=["not", "a", "mapping"]).sections[0].title, SECTIONS[0])

    def test_a_record_that_claims_production_readiness_is_left_out_and_html_is_escaped(self):
        from dclab_core import report as rp

        final = {"task": "binary", "primary_metric": "roc_auc",
                 "evidence": {"holdout_metrics": {"roc_auc": 0.81}, "holdout_primary_metric_ci": {"low": 0.74, "high": 0.87, "method": "iid row bootstrap"},
                              "precision_note": "This model is production-ready.",
                              "limits": ["<script>alert(1)</script> 7 label rows were lost", "Ready for production use now."]}}
        report = rp.build(records={"final": final}, title="Q3 <fraud> review")
        ctx = rp.context(records={"final": final}, title="Q3 <fraud> review")
        markdown = self.assert_traced(report, ctx)
        self.assert_no_production_claim(report)
        self.assertIn("ROC AUC 0.810", markdown)
        self.assertIn("from 0.740 to 0.870", markdown)
        self.assertEqual(rp.sources(report)["7"], "final.evidence.limits.0@1")
        self.assertIn("About this report", [s.title for s in report.sections])
        page = rp.to_html(report)
        self.assertNotIn("<script", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertIn("<title>Q3 &lt;fraud&gt; review</title>", page)

    def test_a_rubric_becomes_the_deliverables_map(self):
        from dclab_core import report as rp

        rubric = [{"requirement": "Report precision at the alert budget", "points": 20},
                  {"requirement": "Serve the model behind an API in Docker", "points": 15},
                  {"name": "A notebook with the analysis", "points": 5},
                  {"requirement": "Explain the leakage checks", "met_by": "the leakage section, 2 pages"}]
        report = rp.build(rubric=rubric)
        markdown = self.assert_traced(report, rp.context(rubric=rubric))
        body = next(s.body for s in report.sections if s.title == "Deliverables map")
        rows = [line for line in body.splitlines() if line.startswith("| ") and "Requirement" not in line]
        self.assertEqual(len(rows), 4)
        self.assertIn("| 20 | Results, with their intervals | open: the tuning and the holdout stage has no record |", rows[0])
        self.assertIn("outside this report", rows[1])
        self.assertIn("a notebook", rows[2])
        self.assertIn("named by the rubric", rows[3])
        self.assertIn("2 pages", markdown)

    def test_the_pdf_renderer_says_what_it_needs_without_reportlab(self):
        from dclab_core import report as rp

        blocked = {name: None for name in list(sys.modules) if name.startswith("reportlab")} | {"reportlab": None}
        with mock.patch.dict(sys.modules, blocked):
            with self.assertRaises(rp.PdfUnavailable) as raised:
                rp.to_pdf(rp.build(), Path(tempfile.mkdtemp()) / "r.pdf")
        self.assertIn("reportlab", str(raised.exception))

    def test_a_line_with_more_numbers_than_a_section_may_cite_is_left_out(self):
        from dclab_core import report as rp

        records = {"final": {"evidence": {"precision_note": " ".join(str(i) for i in range(60))}}}
        report = rp.build(records=records)
        self.assertIn("About this report", [s.title for s in report.sections])
        self.assert_traced(rp.build(records={"final": {"evidence": {"precision_note": "3 of 4"}}}),
                           rp.context(records={"final": {"evidence": {"precision_note": "3 of 4"}}}))
        self.assertEqual(rp.check(report, rp.context(records=records)), [])

    def test_a_long_section_is_split_so_every_part_cites_its_numbers(self):
        from dclab_core import report as rp

        ranking = [{"model": f"family_{chr(97 + i % 26)}{chr(97 + i // 26)}", "mean": 0.5 + i / 1000, "std": 0.01 + i / 10000} for i in range(10)]
        rounds = [{"round": i + 1, "experiment": "loss.median", "family": "ridge", "base_mean": 8 + i / 100, "variant_mean": 7.9 + i / 100,
                   "adopted": False, "why": f"paired gain {i / 10} does not beat {i / 5}"} for i in range(30)]
        records = {"models": {"task": "regression", "primary_metric": "mae",
                              "evidence": {"cv_protocol": "GroupKFold(3)", "ranking": ranking, "review": {"rounds": rounds}}}}
        report = rp.build(records=records)
        self.assertTrue(all(len(s.cites) <= 50 for s in report.sections))
        self.assertTrue(any(s.title.endswith("(continued)") for s in report.sections))
        self.assert_traced(report, rp.context(records=records))


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class ReplicaReportTest(TraceMixin, unittest.TestCase):
    """The report of each replica project, run end to end through the studio on the fast profile."""

    projects: dict = {}

    @classmethod
    def setUpClass(cls):
        from dclab_core.replicas import delivery_fraud as dfr, delivery_time, ticket_routing
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution, core_plan

        cls.store = store = ProjectStore(Path(tempfile.mkdtemp()))

        def table_project(name, goal, frame, solution):
            project = store.create(name, "general", goal)
            path = store.data_dir(project["id"]) / "data.csv"
            frame.to_csv(path, index=False)
            loaded = sd.load_table(path)
            project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path),
                               "profile": sd.profile_table(loaded)}
            project["solution"] = Solution(**solution).model_dump()
            project["settings"]["quick"] = True
            target = solution["target"]
            project["plan"] = core_plan(loaded.drop(columns=[target]), loaded[target], target, solution["task"])[1].model_dump(mode="json")
            store.save(project)
            return project["id"]

        with runner.fast_profile():
            tickets = ticket_routing.generate(seed=0, n=1500).tickets.drop(columns=["resolved_by", "first_response_minutes"])
            cls.projects["tickets"] = table_project(
                "Ticket routing", "Route each support ticket to the right team when it is created.", tickets,
                {"target": "assigned_team", "task": "multiclass", "identifiers": ["ticket_id"], "time_column": "created_at",
                 "prediction_moment": "When the ticket is created, from its subject, body and plan."})
            orders = delivery_time.generate(seed=0, n=1500).orders.drop(columns=["delivered_at", "courier_rating"])
            cls.projects["delivery_time"] = table_project(
                "Delivery time", "Predict how many minutes each delivery takes, when the customer orders.", orders,
                {"target": "actual_minutes", "task": "regression", "identifiers": ["order_id"], "group_column": "restaurant_id",
                 "prediction_moment": "When the customer places the order."})
            # the fraud replica through the label-file flow, as in test_fraud_acceptance's end-to-end test
            replica = dfr.generate(seed=0)
            project = store.create("Rides", "delivery", "Find fraudulent activity of bikers and customers")
            paths = dfr.save(replica, Path(tempfile.mkdtemp()))
            directory = store.data_dir(project["id"])
            (directory / dfr.RIDE_FILE).write_bytes(paths["rides"].read_bytes())
            sd.attach_data(store, project["id"], dfr.RIDE_FILE)
            (directory / dfr.SAMPLE_FILE).write_bytes(paths["sample"].read_bytes())
            project = sd.attach_labels(store, project["id"], dfr.SAMPLE_FILE, target="is_fraud")
            project["solution"] = Solution(**{**project["suggestion"], "group_column": "biker_id", "alert_budget": 0.005,
                                              "prediction_moment": "After delivery, in a daily audit of the day's rides; no label is known when a ride is scored."}).model_dump()
            project["settings"]["quick"] = True
            store.save(project)
            cls.projects["delivery_fraud"] = project["id"]
            for project_id in cls.projects.values():
                engine.run_all(store, project_id)

    RUBRIC = [{"requirement": "Describe the data and its quality", "points": 10},
              {"requirement": "Build a model that detects fraudulent rides", "points": 30},
              {"requirement": "Report precision and recall at the review budget", "points": 20},
              {"requirement": "Serve the model behind an API in Docker", "points": 20},
              {"requirement": "Explain the limits of the analysis", "points": 5}]

    def report_of(self, name, rubric=None):
        from dclab_core import report as rp
        from dclab_rnd.studio import solution_report

        inputs = solution_report.inputs(self.store, self.projects[name], rubric)
        return rp.build(**inputs), rp.context(**inputs), inputs

    def check_replica(self, name, rubric=None):
        from dclab_core import report as rp

        report, ctx, inputs = self.report_of(name, rubric)
        markdown = self.assert_traced(report, ctx)
        self.assert_no_production_claim(report)
        titles = [s.title for s in report.sections]
        for title in SECTIONS:
            self.assertIn(title, titles)
        self.assertNotIn("About this report", titles)  # nothing had to be left out
        final = inputs["records"]["final"]
        metric = final["primary_metric"]
        path = f"final.evidence.holdout_metrics.{metric}"
        self.assertEqual(report.numbers[path], final["evidence"]["holdout_metrics"][metric])  # the headline number, from its record
        self.assertIn(path, rp.sources(report).values())
        for bound in ("low", "high"):
            self.assertIn(f"final.evidence.holdout_primary_metric_ci.{bound}", report.numbers)
        self.assertGreater(sum("@" not in p and "#" not in p for p in report.numbers), 20)  # most numbers are record fields
        self.assertTrue(inputs["findings"] is not None and inputs["findings"].findings)
        self.assertIn("Implication:", markdown)
        page = rp.to_html(report)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertNotIn("<script", page)
        self.assertEqual(rp.SolutionReport.read(report.model_dump_json()), report)  # the contract round-trips
        if not REPORTLAB_MISSING:
            pdf = rp.to_pdf(report, Path(tempfile.mkdtemp()) / f"{name}.pdf")
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        return report, markdown

    def test_the_fraud_report(self):
        report, markdown = self.check_replica("delivery_fraud", self.RUBRIC)
        limits = " ".join(report.limits)
        self.assertIn("lower bound", limits)            # partial labels: precision is a floor
        self.assertIn("sender_phone", limits)           # a column that means something else in the label file
        self.assertIn("known_frauds.csv", markdown)
        self.assertIn("At the review budget of 0.5% of rows", markdown)
        self.assertIn("cluster bootstrap", markdown)
        self.assertIn("entity_behaviour", markdown)
        body = next(s.body for s in report.sections if s.title == "Deliverables map")
        self.assertIn("| Serve the model behind an API in Docker | 20 | outside this report |", body)
        self.assertIn("| Report precision and recall at the review budget | 20 | Results, with their intervals | covered |", body)

    def test_the_ticket_routing_report(self):
        report, markdown = self.check_replica("tickets")
        self.assertIn("Per class on the holdout", markdown)
        self.assertIn("Confused:", markdown)
        self.assertIn("**split.strategy**", markdown)    # the adopted plan's decisions, with their reasons
        self.assertNotIn("Deliverables map", [s.title for s in report.sections])

    def test_the_delivery_time_report(self):
        report, markdown = self.check_replica("delivery_time")
        self.assertIn("The target is capped at 60", markdown)
        self.assertRegex(markdown, r"The 80% interval adds from -?\d")
        self.assertIn("final.evidence.task_assessment.intervals.80%#0", report.numbers)
        self.assertIn("restaurant_id", markdown)

    def test_the_adapter_writes_the_three_formats(self):
        from dclab_rnd.studio import solution_report

        formats = ("md", "html") + (() if REPORTLAB_MISSING else ("pdf",))
        written = solution_report.write(self.store, self.projects["tickets"], Path(tempfile.mkdtemp()), formats=formats, analyse=False)
        self.assertEqual(set(written), set(formats))
        self.assertIn("# Solution report: Ticket routing", written["md"].read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            solution_report.write(self.store, self.projects["tickets"], Path(tempfile.mkdtemp()), formats=("docx",))

    def test_a_project_without_records_gets_a_report_that_says_so(self):
        from dclab_core import report as rp
        from dclab_rnd.studio import solution_report

        project = self.store.create("Empty", "general", "Predict churn next month")
        report = solution_report.build(self.store, project["id"])
        markdown = rp.to_markdown(report)
        self.assertIn("The holdout has not been scored", markdown)
        self.assertIn("**Goal**: Predict churn next month", markdown)
        self.assertEqual(rp.check(report, rp.context(**solution_report.inputs(self.store, project["id"]))), [])


if __name__ == "__main__":
    unittest.main()

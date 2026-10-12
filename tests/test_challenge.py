"""Package 16.13: challenge mode on a written two-part mock challenge (a classification with a pricing question, and the
fraud replica), solved end to end: every rubric line covered or explained, and the private data in no deliverable and
no model request."""

import subprocess
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.replicas import delivery_fraud as dfr, delivery_time
    from dclab_rnd.challenge import solve
    from dclab_rnd.expansion import runner
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""

BRIEF = """Data Science Challenge. The attached data is confidential and must not be disclosed or distributed.

Part 1: Late orders
We want to know whether an order will arrive more than 40 minutes after it is placed, when the customer places it,
and what delivery fee to charge so that late orders are fairly priced. Each row of orders.csv is one order and the
late column is the label.
- Build and compare classification models (4 pts.)
- Answer the pricing question with numbers (2 pts.)

Part 2: Fraud detection
Find fraudulent rides of bikers from the ride table and the sample of known frauds. Rides are scored after delivery,
in a daily audit of the day's rides.
- Solve with classification and anomaly detection approaches (4 pts.)
- Innovative feature extraction (3 pts.)

Deliverables
- An API server using Flask and gUnicorn (2 pts.)
- Dockerize your code (1 pt.)
- Use source control (1 pt.)
- A report explaining your approach and results (2 pts.)
"""


@unittest.skipIf(MISSING, MISSING)
class ChallengeTest(unittest.TestCase):
    def test_a_two_part_mock_challenge_end_to_end(self):
        work = Path(tempfile.mkdtemp())
        orders = delivery_time.generate(seed=0, n=2500).orders
        orders["late"] = (orders["actual_minutes"] > 40).astype(int)  # actual_minutes and delivered_at stay: leaks the plan must keep out
        orders.to_csv(work / "orders.csv", index=False)
        replica = dfr.generate(seed=0)
        paths = dfr.save(replica, work / "fraud")
        files = {"part1": {"table": work / "orders.csv", "target": "late", "group_column": "restaurant_id"},
                 "part2": {"table": paths["rides"], "labels": paths["sample"], "label_target": "is_fraud", "group_column": "biker_id"}}
        out = work / "deliverables"

        from dclab_rnd.models.gateway import Gateway

        handed, real = [], Gateway.client

        def recording(self, purpose, **kwargs):  # every client the gateway hands out, whoever asks (a caller may swallow errors)
            client = real(self, purpose, **kwargs)
            if client is not None:
                handed.append(purpose)
            return client

        # a model is available everywhere: only the private scope can keep every caller from getting one
        with runner.fast_profile(), mock.patch.object(Gateway, "available", return_value=True), \
                mock.patch.object(Gateway, "client", recording):
            result = solve(BRIEF, files, out, store_root=work / "store", approve=True)
        self.assertEqual(handed, [])  # no purpose got a model: nothing about the private data was sent
        self.assertTrue(result["reading"]["private"])
        self.assertEqual(set(result["parts"]), {"part1", "part2"})
        self.assertTrue(all("error" not in p for p in result["parts"].values()), result["parts"])
        self.assertTrue(all(p["approved"] and p["service"] for p in result["parts"].values()))
        self.assertNotIn("service_key", (out / "COVERAGE.md").read_text())
        # every requirement is covered, or says why not
        self.assertEqual(len(result["coverage"]), len(result["reading"]["requirements"]))
        for c in result["coverage"]:
            self.assertTrue(c["met"] or c.get("why"), c)
        pricing = next(c for c in result["coverage"] if c["kind"] == "decision")
        self.assertFalse(pricing["met"])
        self.assertIn("16.9", pricing["why"])
        coverage = (out / "COVERAGE.md").read_text()
        self.assertIn("Points covered", coverage)
        for part in ("part1", "part2"):
            self.assertTrue((out / part / "REPORT.md").exists())
            self.assertTrue((out / part / "REPORT.html").exists())  # the traced report of 17.7, not the fallback
            self.assertTrue((out / part / "service" / "Dockerfile").exists())
            self.assertTrue((out / part / "service" / "tests" / "test_service.py").exists())
        self.assertNotIn("production-ready", (out / "part1" / "REPORT.md").read_text().lower())
        # the plan kept part 1's leaks out (the minutes themselves, the delivery time, the rating given after it)
        from dclab_rnd.studio import ProjectStore

        solution = ProjectStore(work / "store").get(result["parts"]["part1"]["project"])["solution"]
        self.assertTrue({"actual_minutes", "delivered_at", "courier_rating"} <= {f["column"] for f in solution["forbidden"]})
        self.assertEqual(ProjectStore(work / "store").get(result["parts"]["part2"]["project"])["solution"]["group_column"], "biker_id")
        self.assertNotIn("notebook.ipynb", next(c["where"] for c in result["coverage"] if c["kind"] == "model" and c["part"] == "part2"))
        log = subprocess.run(["git", "log", "--oneline"], cwd=out, capture_output=True, text=True, check=True).stdout
        moves = [t["move"] for t in ProjectStore(work / "store").transitions(result["parts"]["part1"]["project"])]
        self.assertIn("set_solution", moves)  # the solution went through the validator, as an agent's move
        self.assertEqual(len(log.strip().splitlines()), 1)
        # the private data is in no deliverable: no data file, and no value of a row
        self.assertFalse([f for f in out.rglob("*") if f.suffix in (".csv", ".parquet", ".xlsx")])
        blob = b"".join(f.read_bytes() for f in out.rglob("*") if f.is_file() and ".git" not in f.parts and "vendor" not in f.parts)
        # the service's salted hashes ("h:" and 16 hex digits, a random key) can hold any six digits by chance: they are
        # one-way, so they are left out of the search rather than letting a coincidence fail the test
        blob = re.sub(rb"h:[0-9a-f]{16}", b"h:", blob)
        for value in list(replica.rides["order_id"].astype(str).head(20)) + list(replica.rides["sender_phone"].dropna().astype(str).head(20)) \
                + list(orders["order_id"].astype(str).head(20)):
            self.assertNotIn(value.encode(), blob)


    def test_a_stated_price_decision_is_delivered_with_its_interval(self):
        from dclab_core.replicas import ride_pricing as rp

        work = Path(tempfile.mkdtemp())
        rp.generate(0, n=3000).offers.to_csv(work / "offers.csv", index=False)
        brief = ("Part 1: Courier fees\nEach row of offers.csv is a fee offered to a courier, accepted or not.\n"
                 "- Predict whether a courier accepts an offer (3 pts.)\n- Choose the fee to offer, with numbers (2 pts.)\n")
        files = {"part1": {"table": work / "offers.csv", "target": "accepted", "moment": "When the fee is offered, before the courier answers.",
                           "decision": {"action": "price", "price_column": "fee", "reference_column": "usual_fee", "side": "pay",
                                        "unit_value_factor": 1.6}}}
        with runner.fast_profile():
            result = solve(brief, files, work / "out", store_root=work / "store")
        fee = next(c for c in result["coverage"] if c["kind"] == "decision")
        self.assertTrue(fee["met"], fee)
        self.assertTrue(result["parts"]["part1"]["decision"]["made"])
        report = (work / "out" / "part1" / "REPORT.md").read_text()
        self.assertIn("times the reference", report)  # the decision's claim, with its interval, is in the report
        self.assertIn("| Choose the fee to offer, with numbers | 2.000 | Results, with their intervals | covered |", report)
        self.assertIn("| Predict whether a courier accepts an offer | 3.000 | Results, with their intervals | covered |", report)

    def test_without_approval_no_service_and_the_coverage_says_why(self):
        work = Path(tempfile.mkdtemp())
        orders = delivery_time.generate(seed=1, n=1500).orders
        orders["late"] = (orders["actual_minutes"] > 40).astype(int)
        orders.to_csv(work / "orders.csv", index=False)
        brief = BRIEF.split("Part 2:")[0] + "Deliverables\n- An API server using Flask and gUnicorn (2 pts.)\n"
        with runner.fast_profile():
            result = solve(brief, {"part1": {"table": work / "orders.csv", "target": "late"}}, work / "out")
        api = next(c for c in result["coverage"] if c["kind"] == "api")
        self.assertFalse(api["met"])
        self.assertIn("approve=True", api["why"])
        self.assertFalse((work / "out" / "part1" / "service").exists())


class ReadBriefTest(unittest.TestCase):
    def test_parts_points_shared_sections_and_confidentiality(self):
        from dclab_core.challenge import read_brief

        out = read_brief(BRIEF)
        self.assertEqual([p["title"] for p in out["parts"]], ["Part 1: Late orders", "Part 2: Fraud detection"])
        self.assertEqual(out["parts"][1]["task"], "binary")
        self.assertTrue(out["parts"][1]["rare_event"])
        kinds = {r["text"]: (r["kind"], r["part"], r["points"]) for r in out["requirements"]}
        self.assertEqual(kinds["Answer the pricing question with numbers"], ("decision", "part1", 2.0))
        self.assertEqual(kinds["An API server using Flask and gUnicorn"], ("api", None, 2.0))  # a deliverable for the whole challenge
        self.assertEqual(kinds["Dockerize your code"][0], "docker")
        self.assertEqual(out["total_points"], 19.0)
        self.assertTrue(out["private"])
        open_brief = read_brief("Predict which customers churn next month.\n- A short report")
        self.assertFalse(open_brief["private"])
        self.assertEqual([p["id"] for p in open_brief["parts"]], ["part1"])
        self.assertEqual(open_brief["requirements"][0]["kind"], "report")

    def test_inside_the_private_scope_no_purpose_gets_a_model(self):
        from dclab_rnd.models.gateway import Gateway, private_scope

        with mock.patch.object(Gateway, "available", return_value=True):
            gateway = Gateway()
            self.assertIsNotNone(gateway.client("stage_notes"))
            with private_scope():
                self.assertIsNone(gateway.client("stage_notes"))
                with private_scope(False):  # an inner scope cannot lift it
                    self.assertIsNone(gateway.client("intern"))
            self.assertIsNotNone(gateway.client("stage_notes"))


if __name__ == "__main__":
    unittest.main()

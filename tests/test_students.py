"""Package 29.1: the students' register: one row per decision point, the numbers equal to the files they are read from,
the passed students shown as whisper and ready after N labels, the reasons of the points nothing decides; the CLI and
the owner's route."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core.ledger import POINTS
    from dclab_rnd import ledger as L
    from dclab_rnd import students
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class RegisterTests(unittest.TestCase):
    @staticmethod
    def exams():
        """Every sealed result file, as the register reads them: the expectations come from the files, not from today."""
        out = {}
        for campaign in ("decision_models_v1", "workflow_models_v1"):
            for f in (students.CAMPAIGNS / campaign / "results").glob("*.json"):
                d = json.loads(f.read_text())
                out[d["point"]] = d
        return out

    def test_every_point_has_a_row_and_the_numbers_are_the_files(self):
        reg = students.register()
        self.assertEqual([r["point"] for r in reg["items"]], list(POINTS))
        self.assertEqual(reg["total"], len(POINTS))
        by = {r["point"]: r for r in reg["items"]}
        manifest = json.loads((students.DECISIONS / "generated_v1" / "MANIFEST.json").read_text())
        for point, s in manifest["sets"].items():
            self.assertEqual(by[point]["homework"]["decisions"], s["decisions"])
            self.assertEqual(by[point]["homework"]["sealed"], s["splits"]["sealed"])
            self.assertEqual(by[point]["homework"]["rule_accuracy"], round(s["rule_right"] / s["decisions"], 4))
        images = json.loads((students.DECISIONS / "generated_images_v1" / "MANIFEST.json").read_text())
        for point, s in images["sets"].items():
            self.assertEqual(by[point]["homework"]["decisions"], s["decisions"])
        work = json.loads((students.DECISIONS / "workflow_v1" / "MANIFEST.json").read_text())
        exams = self.exams()
        for point, s in work["sets"].items():  # the review's F1 and F2: labelled rows, people, and the exam's own sealed set
            hw = by[point]["homework"]
            self.assertEqual((hw["decisions"], hw["played"], hw["tables"]), (s["labelled"], s["rows"], s["people"]))
            self.assertEqual(hw["rule_accuracy"], round(s["rule_right"] / s["labelled"], 4))
            if point in exams:
                self.assertEqual(hw["sealed"], exams[point]["sealed"]["decisions"])
        for folder in ("models_v1", "models_images_v1", "workflow_models_v1"):
            for f in (students.DECISIONS / folder).glob("*.json"):
                if f.name in ("fitted.json", "SKIPPED.json"):
                    continue
                card = json.loads(f.read_text())
                st = by[card["point"]]["student"]
                self.assertEqual((st["answerer"], st["dev_accuracy"], st["rule_accuracy"]),
                                 (f"{card['name']}@{card['version']}", card["dev"]["accuracy"], card["rule_dev"]["accuracy"]))
        for point, d in exams.items():
            v = by[point]["verdict"]
            self.assertEqual((v["pass"], v["sealed_accuracy"]), (d["result"] == "pass", d["verdict"]["model"]["accuracy"]))
        summary = reg["summary"]
        self.assertEqual((summary["points"], summary["technical"] + summary["workflow"]), (len(POINTS), len(POINTS)))
        self.assertEqual(summary["with_homework"], sum(bool(r["homework"]) for r in reg["items"]))
        self.assertEqual(summary["passed"], sum(d["result"] == "pass" for d in exams.values()))

    def test_every_row_says_where_its_student_stands(self):
        reg = students.register()
        by = {r["point"]: r for r in reg["items"]}
        exams = self.exams()
        passed = [p for p, d in exams.items() if d["result"] == "pass"]
        failed = [p for p, d in exams.items() if d["result"] == "fail"]
        for point in passed:
            self.assertEqual(by[point]["status"], f"whisper, ready to act after {reg['summary']['min_labels_to_act']} labels")  # no shadow labels here
        for point in failed:
            self.assertEqual(by[point]["status"], "whisper, failed its exam")
            self.assertTrue(by[point]["verdict"]["failed"])
        unexamined = [r for r in reg["items"] if r["student"].get("answerer") and r["point"] not in exams]
        self.assertTrue(all(r["status"] == "whisper, not examined" for r in unexamined))
        bare = [r for r in reg["items"] if not r["homework"] and not r["student"] and not r["not_decided"]]
        self.assertTrue(bare and all(r["status"] == "no homework" for r in bare))
        for r in reg["items"]:
            if r["not_decided"]:
                self.assertEqual(r["status"], "not decided")
        self.assertTrue(all(r["captured"] or r["not_decided"] for r in reg["items"]))

    def test_a_retrained_student_has_not_sat_its_predecessors_exam(self):  # the review's F4
        point = next(p for p, d in self.exams().items() if d["result"] == "pass")
        real = students.students

        def retrained():
            out = real()
            out[point] = {**out[point], "answerer": out[point]["answerer"].split("@")[0] + "@newversion"}
            return out

        with mock.patch.object(students, "students", retrained):
            row = {r["point"]: r for r in students.register()["items"]}[point]
        self.assertIsNone(row["verdict"])
        self.assertEqual(row["status"], "whisper, not examined (an earlier version passed)")

    def test_a_workspace_ledger_counts_its_decisions_and_labels_and_a_routed_student_shows_its_level(self):
        home = Path(tempfile.mkdtemp())
        book = L.open_ledger(home)
        rec = L.decide("column.is_identifier", ("column", "order_id"), answers=[L.answer("rule", "id_like", False)], value_used=False,
                       state={"column": "order_id"}, ledger=book, project="p1")
        L.label(rec, True, "benchmark", "replica:delivery_fraud", ledger=book)
        by = {r["point"]: r for r in students.register(book)["items"]}
        self.assertEqual((by["column.is_identifier"]["recorded"], by["column.is_identifier"]["labels"]), (1, {"benchmark": 1}))
        name = by["feature.leakage_suspect"]["student"]["answerer"]
        routed = {"decisions": {"feature.leakage_suspect": {"answerer": name, "level": 2, "bar": 0.9}}}
        measured = {name: {"labelled_disagreements": 25, "agreement": 0.9}}
        row = {r["point"]: r for r in students.register(book, routed, measured)["items"]}["feature.leakage_suspect"]
        self.assertEqual((row["level"], row["status"], row["needed_to_act"]), (2, "acts", 0))
        other = {"decisions": {"feature.leakage_suspect": {"answerer": "someone@9", "level": 3}}}
        row = {r["point"]: r for r in students.register(book, other)["items"]}["feature.leakage_suspect"]
        self.assertIsNone(row["level"])  # another answerer serves: this student's level is none

    def test_the_command_prints_the_register_and_keeps_it(self):
        out, home = io.StringIO(), tempfile.mkdtemp()
        with mock.patch.dict("os.environ", {"DCLAB_AGENT_HOME": home}), redirect_stdout(out):
            self.assertEqual(students.main([]), 0)
        text = out.getvalue()
        self.assertIn("feature.leakage_suspect", text)
        self.assertIn("passed their exam", text)
        kept = [json.loads(x) for x in (Path(home) / "students" / "register.jsonl").read_text().splitlines()]  # every run's result, kept
        self.assertEqual((len(kept), kept[0]["summary"]["passed"]), (1, students.register()["summary"]["passed"]))
        with redirect_stdout(io.StringIO()) as js, mock.patch.dict("os.environ", {"DCLAB_AGENT_HOME": tempfile.mkdtemp()}):
            students.main(["--json", "--no-shadow"])
        self.assertEqual(json.loads(js.getvalue())["total"], len(POINTS))

    def test_the_command_and_the_page_read_the_same_way(self):  # the review's F3
        home = Path(tempfile.mkdtemp())
        book = L.open_ledger(home)
        with mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "0"}):
            reg = students.read(book)
        self.assertIn("shadows are off", reg["warnings"][0])  # said, never shown as a number
        from dclab_rnd.ledger import shadow

        with mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "1"}), mock.patch.object(shadow, "measure", side_effect=RuntimeError("broken")):
            reg = students.read(book)
        self.assertIn("failed", reg["warnings"][0])
        name = {r["point"]: r for r in reg["items"]}["feature.leakage_suspect"]["student"]["answerer"]
        with mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "1"}), mock.patch.object(shadow, "measure", return_value={name: {"labelled_disagreements": 25}}):
            row = {r["point"]: r for r in students.read(book)["items"]}["feature.leakage_suspect"]
        self.assertEqual((row["needed_to_act"], row["status"]), (0, "whisper, may act"))


@unittest.skipIf(MISSING, MISSING)
class RouteTests(unittest.TestCase):
    def test_the_register_is_the_owners_route(self):
        from dclab_rnd.accounts import guard
        from dclab_rnd.accounts.principal import Principal
        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        reg = client.get("/api/students").json()
        self.assertEqual(reg["total"], len(POINTS))
        self.assertEqual(reg["summary"]["passed"], students.register()["summary"]["passed"])
        self.assertEqual(guard.permission("GET", "/api/students"), "admin")
        with mock.patch.object(guard, "current", return_value=Principal(role="reviewer", user_id="u1", via="session")):
            self.assertEqual(client.get("/api/students").status_code, 403)


if __name__ == "__main__":
    unittest.main()

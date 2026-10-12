"""Package 17.1: dclab_core, the portable solving library, and its contracts.

The library is what DCLab's main product adopts (Part 6 of the playbook), so these tests pin its boundary as well as
its behaviour: it imports neither dclab_rnd nor anything that reaches a database, the network or a model provider; it
depends only on numpy, pandas, scikit-learn and pydantic; its code parses as Python 3.10; every contract is frozen and
versioned, refuses NaN, round-trips through JSON and cannot change shape under the same version; evidence never
carries rows; and the published JSON Schemas enforce what Python enforces.
"""

import ast
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "dclab_core"
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:  # requirements/ci.txt
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""

ALLOWED_THIRD_PARTY = {"numpy", "pandas", "sklearn", "pydantic", "pydantic_core", "typing_extensions"}
# the standard library is allowed, except what reaches a database, the network, other processes or dynamic imports
DENIED_STDLIB = {"sqlite3", "dbm", "shelve", "socket", "ssl", "http", "urllib", "ftplib", "smtplib", "imaplib", "poplib",
                 "subprocess", "multiprocessing", "importlib", "asyncio", "xmlrpc", "webbrowser", "ctypes"}
# what importing the four dependencies loads on its own (checked in a subprocess)
DEPENDENCY_TREE = ALLOWED_THIRD_PARTY | {"annotated_types", "typing_inspection", "dateutil", "pytz", "tzdata", "six",
                                         "scipy", "joblib", "threadpoolctl", "dclab_core", "_distutils_hack", "sitecustomize",
                                         "_virtualenv", "pyarrow", "numexpr", "bottleneck", "packaging"}


def _sources() -> list[Path]:
    files = sorted(p for p in CORE.rglob("*.py") if not ({"build", "dist"} & set(p.parts)) and ".egg-info" not in str(p))
    assert files, "dclab_core has no modules"
    return files


def _imports(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            names.append("__import__")
    return names


def _violations(names: list[str]) -> list[str]:
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for name in names:
        top = name.split(".")[0]
        if name == "__import__" or top in DENIED_STDLIB or not (top in stdlib or top in ALLOWED_THIRD_PARTY or top == "dclab_core"):
            bad.append(name)
    return bad


class BoundaryTest(unittest.TestCase):
    def test_the_core_imports_only_the_safe_standard_library_its_four_dependencies_and_itself(self):
        for path in _sources():
            with self.subTest(file=path.name):
                self.assertEqual(_violations(_imports(ast.parse(path.read_text(encoding="utf-8")))), [],
                                 f"{path.relative_to(ROOT)} imports outside the boundary")

    def test_the_checker_catches_what_it_must(self):
        for source in ("import dclab_rnd", "import sqlalchemy", "import sqlite3", "from urllib import request",
                       "import importlib", "x = __import__('dclab_rnd')", "import openai", "import socket"):
            with self.subTest(source=source):
                self.assertTrue(_violations(_imports(ast.parse(source))))

    def test_the_code_parses_as_python_3_10(self):
        for path in _sources():
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))

    @unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
    def test_importing_the_core_loads_nothing_outside_its_dependencies(self):
        # only what importing the core adds: an environment's .pth files load their own modules at start-up
        modules = ", ".join("dclab_core" if p.stem == "__init__" else f"dclab_core.{p.stem}" for p in _sources())
        code = ("import sys, json; before = set(sys.modules); "
                f"import {modules}; std = set(sys.stdlib_module_names); "
                "print(json.dumps(sorted({m.split('.')[0] for m in set(sys.modules) - before} - std)))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        loaded = {m for m in json.loads(out.stdout.strip()) if not m.startswith("_sysconfigdata")}  # stdlib, named per platform
        self.assertNotIn("dclab_rnd", loaded)
        self.assertEqual(sorted(loaded - DEPENDENCY_TREE), [], "the core loads a module outside its dependencies")

    def test_the_package_declares_only_its_four_dependencies(self):
        text = (CORE / "pyproject.toml").read_text(encoding="utf-8")
        deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
        names = sorted(line.strip().strip('",').split(">")[0].split("=")[0].split("<")[0] for line in deps.splitlines() if line.strip())
        self.assertEqual(names, ["numpy", "pandas", "pydantic", "scikit-learn"])
        self.assertIn('"pydantic>=2.10"', text)  # the published schemas are what pydantic 2.10 and later generate
        self.assertIn('requires-python = ">=3.10"', text)

    def test_build_output_is_ignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("dclab_core/build/", ignore)
        self.assertIn("dclab_core/*.egg-info/", ignore)


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class ContractTest(unittest.TestCase):
    def setUp(self):
        from dclab_core import contracts as c

        self.c = c

    def samples(self):
        c = self.c
        finding = c.Finding(id="F-entity", topic="entities", statement="Labels sit on 38 of 240 bikers.",
                            evidence={"fraud_bikers": 38, "bikers": 240}, implication="Group the folds by biker.",
                            affects=["split.strategy"])
        decision = c.Decision(key="split.strategy", choice={"kind": "group", "column": "biker_id"},
                              reason="Labels concentrate on a few bikers (F-entity).", cites=["F-entity"],
                              alternatives=[{"choice": "stratified", "why_not": "A biker would sit on both sides."}])
        return {
            "problem_brief": c.ProblemBrief(goal=c.TextFact(value="Find fraud", status="stated", quote="find fraud", source="text"),
                                            task=c.TaskFact(value="binary", status="inferred", reason="asks yes or no"),
                                            constraints=[c.Constraint(kind="confidential", text="not disclosed", quote="not disclosed")],
                                            deliverables=[c.Deliverable(name="api", points=2, quote="an API (2 pts.)")]),
            "data_findings": c.DataFindings(task="binary", tables=[c.TableSummary(name="rides.csv", rows=10, columns=3)], findings=[finding]),
            "solution_plan": c.SolutionPlan(task="binary", decisions=[decision], risks=[c.Risk(statement="s", check="c")]),
            "decision": decision,
            "review": c.Review(weaknesses=[finding], hypotheses=[c.Hypothesis(statement="h", cites=["F-entity"], test="t")],
                               next_experiments=[c.Experiment(name="e", hypothesis=0, expected_gain="+0.1 AP")]),
            "solution_report": c.SolutionReport(title="t", sections=[c.ReportSection(title="s", body="b")], numbers={"ap": 0.52}),
        }

    def test_every_contract_is_frozen_strict_and_versioned(self):
        from pydantic import ValidationError

        for name, model in self.c.CONTRACTS.items():
            with self.subTest(contract=name):
                self.assertRegex(model.model_fields["schema_version"].default, rf"^dclab_core\.{name}\.v\d+$")
                self.assertTrue(model.model_config.get("frozen"))
                self.assertEqual(model.model_config.get("extra"), "forbid")
                with self.assertRaises(ValidationError):
                    model.model_validate({"not_a_field": 1})

    def test_every_contract_round_trips_through_json_and_is_read_only_with_its_version(self):
        for name, sample in self.samples().items():
            with self.subTest(contract=name):
                model = type(sample)
                text = sample.model_dump_json()
                self.assertEqual(model.read(text), sample)
                data = json.loads(text)
                data.pop("schema_version")
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    model.read(data)  # an unversioned document is never stamped as the current version
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    model.read({**json.loads(text), "schema_version": f"dclab_core.{name}.v99"})

    def test_numbers_that_json_cannot_hold_are_refused(self):
        from pydantic import ValidationError

        c = self.c
        with self.assertRaises(ValidationError):
            c.SolutionReport(title="t", numbers={"auc": math.nan})
        with self.assertRaises(ValidationError):
            c.Decision(key="threshold.policy", choice={"threshold": math.inf}, reason="a reason long enough")
        with self.assertRaises(ValidationError):
            c.Finding(id="F-x", topic="x", statement="x", evidence={"rate": math.nan}, implication="x")
        with self.assertRaises(ValidationError):
            c.Deliverable(name="api", points=math.nan, quote="q")

    def test_a_fact_says_how_it_is_known(self):
        from pydantic import ValidationError

        c = self.c
        self.assertEqual(c.TextFact.unknown().status, "unknown")
        self.assertEqual(c.TextFact(value="x", status="stated", quote="x", source="text").quote, "x")
        for bad in (dict(value="binary", status="stated", source="text"),        # stated without its quote
                    dict(value="binary", status="stated", quote="q", source="rule"),  # a rule does not state
                    dict(value=None, status="stated", quote="q", source="text"),    # stated, but of nothing
                    dict(value="binary", status="inferred", source="data"),         # inferred without a reason
                    dict(value=0, status="unknown"), dict(value=False, status="unknown"), dict(value={}, status="unknown")):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                c.Fact(**bad)
        with self.assertRaises(ValidationError):
            c.TaskFact(value="clustering", status="inferred", reason="r")       # not a task type
        with self.assertRaises(ValidationError):
            c.Constraint(kind="latency", text="fast")                           # stated without a quote

    def test_the_brief_covers_the_main_products_problem_spec(self):
        fields = self.c.ProblemBrief.model_fields
        # ProblemSpec: business_objective, task_type, target_column, prediction_unit, prediction_time_column,
        # prediction_horizon, primary_metric, constraints, success_criteria
        for field in ("goal", "task", "target_column", "unit", "time_column", "horizon", "primary_metric", "constraints",
                      "success_criteria"):
            self.assertIn(field, fields)
        self.assertIn("target_definition", fields)  # a label defined by another file, with no column

    def test_evidence_is_a_summary_never_rows(self):
        import numpy as np
        import pandas as pd
        from pydantic import ValidationError

        ok = self.c.Finding(id="F-grain", topic="grain", statement="Several rows per biker.",
                            evidence={"rows_per_biker_median": np.int64(61), "share": np.float64(0.25)},
                            implication="Group the folds by biker.", affects=["split.strategy"])
        self.assertEqual(ok.evidence, {"rows_per_biker_median": 61, "share": 0.25})  # numpy scalars become plain numbers
        self.assertIsInstance(ok.evidence["rows_per_biker_median"], int)
        frame = pd.DataFrame({"a": range(50), "b": range(50)})
        for bad in ({"values": list(range(500))}, {"v": {str(i): i for i in range(5000)}}, {"rows": frame.head(10).to_dict("records")},
                    {"t": [[[1] * 15] * 15] * 15}, {"s": frozenset(range(10))}, {"b": b"a,b\n1,2"}, {"r": range(5)},
                    {"a": np.array([1, 2])}, {"a0": np.array(5)}, {"f": frame}, {"k": {"x" * 500: 1}}):
            with self.subTest(bad=type(next(iter(bad.values()))).__name__), self.assertRaises(ValidationError):
                self.c.Finding(id="F-x", topic="x", statement="x", evidence=bad, implication="x")

    def test_a_plan_holds_one_decision_per_key_each_with_a_reason(self):
        from pydantic import ValidationError

        decision = self.samples()["decision"]
        plan = self.c.SolutionPlan(task="binary", decisions=[decision])
        self.assertEqual(plan.decision("split.strategy").choice["column"], "biker_id")
        self.assertIsNone(plan.decision("metric.primary"))
        with self.assertRaises(ValidationError):
            self.c.SolutionPlan(task="binary", decisions=[decision, decision])
        with self.assertRaises(ValidationError):
            self.c.Decision(key="metric.primary", choice="average_precision", reason="")

    def test_an_experiment_tests_an_existing_hypothesis(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.c.Review(next_experiments=[self.c.Experiment(name="e", hypothesis=7, expected_gain="x")])


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class SchemaTest(unittest.TestCase):
    def test_the_published_schemas_are_the_generated_ones(self):
        from dclab_core import schemas

        generated = schemas.generate()
        published = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in schemas.SCHEMA_DIR.glob("*.json")}
        self.assertEqual(sorted(published), sorted(generated), "run: make core-schemas")
        for name, schema in generated.items():
            self.assertEqual(schemas.normalize(published[name]), schemas.normalize(schema),
                             f"{name}.json is stale: run make core-schemas")

    def test_no_contract_changes_shape_under_the_same_version(self):
        from dclab_core import schemas

        for name, schema in schemas.generate().items():
            version = schemas.version_of(name)
            baseline = schemas.BASELINE_DIR / f"{version}.json"
            with self.subTest(version=version):
                self.assertTrue(baseline.exists(), f"no baseline for {version}: run make core-schemas")
                self.assertEqual(schemas.normalize(schema), json.loads(baseline.read_text(encoding="utf-8")),
                                 f"{name} changed shape but is still {version}: move its version (and run make core-schemas)")

    def test_the_baseline_comparison_sees_a_rename(self):
        from dclab_core import schemas

        schema = schemas.generate()["problem_brief"]
        renamed = json.loads(json.dumps(schema).replace('"primary_metric"', '"main_metric"'))
        self.assertNotEqual(schemas.normalize(renamed), schemas.normalize(schema))

    def test_the_schema_enforces_what_python_enforces(self):
        import jsonschema

        from dclab_core import schemas

        validator = jsonschema.Draft202012Validator(schemas.generate()["problem_brief"])
        good = {"schema_version": "dclab_core.problem_brief.v1",
                "task": {"value": "binary", "status": "inferred", "reason": "asks yes or no"}}
        self.assertEqual(list(validator.iter_errors(good)), [])
        for bad in ({"task": {"value": "foo", "status": "inferred", "reason": "r"}},           # not a task type
                    {"task": {"value": "binary", "status": "stated", "source": "text"}},        # stated without a quote
                    {"task": {"value": "binary", "status": "inferred"}},                        # inferred without a reason
                    {"task": {"value": "binary", "status": "unknown"}},                         # unknown with a value
                    {"goal": {"value": "x", "status": "stated", "quote": "x", "source": "rule"}}):
            with self.subTest(bad=bad):
                self.assertTrue(list(validator.iter_errors({"schema_version": "dclab_core.problem_brief.v1", **bad})))
        self.assertTrue(list(validator.iter_errors({"task": good["task"]})))  # the version is required


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class AdvisorTest(unittest.TestCase):
    def test_no_advisor_means_the_deterministic_path(self):
        from dclab_core.advisor import NoAdvisor, consult

        self.assertIsNone(consult(NoAdvisor(), "rank_targets", {"columns": 12}, parse=dict))
        self.assertIsNone(consult(None, "rank_targets", {"columns": 12}, parse=dict))

    def test_a_failing_or_invalid_advisor_is_ignored_and_reported(self):
        from dclab_core.advisor import consult

        class Broken:
            def ask(self, purpose, evidence):
                raise TimeoutError("provider down")

        class Wrong:
            def ask(self, purpose, evidence):
                return {"target": 7}

        def parse(answer):
            if not isinstance(answer.get("target"), str):
                raise ValueError("target must be a column name")
            return answer["target"]

        errors = []
        self.assertIsNone(consult(Broken(), "rank_targets", {"columns": 12}, parse=parse, on_error=lambda k, e: errors.append(k)))
        self.assertIsNone(consult(Wrong(), "rank_targets", {"columns": 12}, parse=parse, on_error=lambda k, e: errors.append(k)))
        self.assertEqual(errors, ["failed", "invalid"])

    def test_rows_never_reach_an_advisor(self):
        import numpy as np
        import pandas as pd

        from dclab_core.advisor import EvidenceError, consult

        seen = []

        class Recorder:
            def ask(self, purpose, evidence):
                seen.append(evidence)
                return {"ok": True}

        frame = pd.DataFrame({"a": range(100), "b": range(100)})
        for bad in ({"frame": frame}, {"values": list(range(200))}, {"v": dict(zip(map(str, range(500)), range(500)))},
                    {"rows": frame.head(30).to_dict("records")}, {"csv": frame.to_csv().encode()}, {"col": frame["a"]},
                    {"view": {"a": 1}.values()}, {"gen": (i for i in range(3))}, {"arr": np.zeros(3)}):
            with self.subTest(bad=list(bad)[0]), self.assertRaises(EvidenceError):
                consult(Recorder(), "rank_targets", bad, parse=dict)
        self.assertEqual(seen, [])
        consult(Recorder(), "rank_targets", {"rows": np.int64(5000), "positive_rate": np.float64(0.006)}, parse=dict)
        self.assertEqual(seen, [{"rows": 5000, "positive_rate": 0.006}])  # numpy scalars cross as plain numbers


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class StudioAdapterTest(unittest.TestCase):
    PROJECT = {"goal": "Flag fraudulent rides for the quality team", "solution": {
        "target": "is_fraud", "task": "binary", "positive_label": "yes", "group_column": "biker_id", "time_column": "created_date",
        "prediction_moment": "After delivery, in a daily audit of the day's rides.",
        "forbidden": [{"column": "refund_issued", "reason": "Written after the ride is reviewed."}],
        "identifiers": ["order_id", "terminal_ref"], "text_columns": [], "metric": "average_precision", "notes": "From the probe."}}

    def test_a_project_round_trips_through_the_brief_and_the_plan_losing_nothing(self):
        from dclab_rnd.core_adapter import brief_from_project, plan_from_project, project_fields
        from dclab_rnd.studio.solution import Solution

        brief, plan = brief_from_project(self.PROJECT), plan_from_project(self.PROJECT)
        self.assertEqual(plan.decision("features.forbidden").choice, self.PROJECT["solution"]["forbidden"])
        back = project_fields(brief, plan)
        self.assertEqual(back["goal"], self.PROJECT["goal"])
        self.assertEqual(Solution(**back["solution"]).model_dump(), Solution(**self.PROJECT["solution"]).model_dump())

    def test_an_unsigned_solution_is_an_inference_not_the_persons_words(self):
        from dclab_rnd.core_adapter import brief_from_project, plan_from_project

        brief = brief_from_project(self.PROJECT)
        self.assertEqual(brief.goal.status, "stated")          # the goal is what the person typed
        self.assertEqual(brief.task.status, "inferred")         # the solution may come from the intern's tool
        self.assertIn("not signed", brief.task.reason)
        self.assertEqual(plan_from_project(self.PROJECT).decision("metric.primary").status, "proposed")

    def test_a_signed_solution_is_the_persons(self):
        from dclab_rnd.core_adapter import brief_from_project, plan_from_project
        from dclab_rnd.studio.graph import solution_hash

        signed = {**self.PROJECT, "signoff": {"gate": "solution", "by": "owner", "at": "2026-10-12T10:00:00",
                                              "solution_hash": solution_hash(self.PROJECT["solution"])}}
        self.assertEqual(brief_from_project(signed).task.status, "stated")
        self.assertEqual(plan_from_project(signed).decision("metric.primary").status, "accepted")

    def test_a_project_without_a_solution_gives_an_honest_brief(self):
        from dclab_rnd.core_adapter import brief_from_project

        brief = brief_from_project({"goal": "", "solution": None})
        self.assertFalse(brief.goal.known)
        self.assertFalse(brief.target_column.known)


if __name__ == "__main__":
    unittest.main()

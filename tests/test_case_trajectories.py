"""Package 30.19: the case trajectories, a whole case per built project for the solver model.

The builder keeps the request, the conversation, the cards, the brief, the recipe, the Workflow Run and the report together, holds whole
projects out, and refuses below its minimum (saying why in its manifest); the host assembles a case only for a project the consent
allows, with the person's words redacted; and the lake teaches a case as a solution design.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def item(i, project=None):
    return {"draft": f"d{i}", "project": project or f"p{i}", "request": "Find the late orders", "conversation": [{"role": "user", "text": "When placed"}],
            "problem_card": {"family": "classification"}, "data_card": {"tables": []}, "fit_verdict": {"verdict": "fits"}, "brief": {"approval": None},
            "recipe": {"kind": "binary"}, "workflow_run": {"steps": [{"id": 1}]}, "gates": {"g4": {"passed": True}}, "report": {"title": "r"},
            "bar_verdict": {"met": True}}


@unittest.skipIf(MISSING, MISSING)
class BuilderTest(unittest.TestCase):
    def test_below_the_minimum_nothing_is_written_and_the_manifest_says_why(self):
        from dclab_core import datasets as builders

        rows, manifest = builders.case_trajectories([item(i) for i in range(3)])
        self.assertEqual(rows, [])
        self.assertEqual(manifest.counts["refused"], {"yes": 1})
        self.assertEqual(manifest.counts["projects"], {"found": 3, "minimum": builders.CASE_MIN_PROJECTS})
        self.assertIn("Refused", " ".join(manifest.limits))
        self.assertEqual(manifest.schema_version, "dclab_core.dataset_manifest.v8")

    def test_a_case_is_whole_and_a_project_is_held_out_whole(self):
        from dclab_core import datasets as builders

        items = [item(i) for i in range(12)] + [item(99, project="p0")]  # a second draft of p0: the same group, the same split
        rows, manifest = builders.case_trajectories(items)
        self.assertEqual(len(rows), 13)
        self.assertEqual(set(builders.CASE_PARTS) - set(rows[0]["complete"]), set())
        by_project = {}
        for r in rows:
            by_project.setdefault(r["project"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in by_project.values()))
        self.assertEqual(manifest.counts["refused"], {"no": 1})
        self.assertIsNone(builders.case_trajectories([{**item(1), "workflow_run": None}], minimum=0)[0] or None)  # no run: no case

    def test_the_lake_teaches_a_case_as_a_solution_design(self):
        from dclab_core import datasets as builders
        from dclab_rnd import lake

        rows, _ = builders.case_trajectories([item(i) for i in range(10)])
        out = lake.session_examples({"case_trajectories/problem_to_report": [{"row": r, "source": "w1:case_trajectories"} for r in rows]}, lake.Run())
        self.assertTrue(out)
        self.assertEqual({e.family for e in out}, {"solution_design"})


@unittest.skipIf(MISSING, MISSING)
class HostTest(unittest.TestCase):
    def test_a_case_is_assembled_only_where_the_consent_allows_and_without_the_persons_words(self):
        from fastapi.testclient import TestClient

        from dclab_rnd import consent, datasets
        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        s = client.app.state.services
        home = Path(s.store.home)
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        for use in ("improve", "train"):
            consent.set_workspace_consent(home, use, True)
        yes = client.post("/api/projects", json={"name": "Consented", "goal": "g"}, headers=h).json()["id"]
        no = client.post("/api/projects", json={"name": "Not consented", "goal": "g"}, headers=h).json()["id"]
        client.patch(f"/api/projects/{yes}", json={"settings": {"consent": {"train": True, "improve": True}}}, headers=h)
        for project in (yes, no):
            d = s.drafts.create("Predict which orders arrive late, call John Smith on +44 7700 900123")["id"]
            s.drafts.update(d, lambda x, project=project: x.update(project_id=project, messages=[{"role": "user", "text": "Ask jane.roe@example.org"}]))
        items = datasets._case_items(s, {yes: s.projects.get(yes)})
        self.assertEqual([i["project"] for i in items], [yes])
        text = json.dumps(items, default=str)
        for gone in ("John Smith", "7700", "jane.roe"):
            self.assertNotIn(gone, text)
        self.assertIsNotNone(items[0]["workflow_run"])  # a project with no stage run still has its twelve steps, each not run yet
        out = {(b["builder"], b["name"]): b for b in datasets.build(s)["built"]}
        self.assertEqual(out["case_trajectories", "problem_to_report"]["rows"], 0)  # one case, below the default minimum of ten: refused


if __name__ == "__main__":
    unittest.main()

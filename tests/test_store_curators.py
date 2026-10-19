"""Package 32.9: the curators.

A record reaches every model type it fits, in the right folder, and no other; a duplicate and a broken record are kept out
with their reason and listed in quarantine, still whole in raw; a teacher's label is silver, apart; a record whose consent
does not allow training is refused for a set; a placement past the layout is refused and kept; a model's answers are kept
in shadow, never used; the model sees each record's shape only.
"""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)

AT = "2026-10-20T10:00:00+00:00"


def decision(i, point="column.is_identifier", project="p1", **extra):
    return {"schema_version": "dclab_core.decision_record.v2", "id": f"dg{i:020d}", "point": point, "at": AT, "project": project,
            "state": {"rows": "1k-10k"}, "answers": [{"by": "rule", "value": True}], "used": "rule", "value_used": True, **extra}


@unittest.skipIf(MISSING, MISSING)
class CuratorsTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(self.layout)
        lines = [decision(1), decision(2, point="workflow.next_question"), decision(3, point="department.proposal"),
                 decision(1),  # the same record again: a duplicate
                 {**decision(4), "answers": None}]  # broken: no answers
        store.keep_log("decisions", "dev", "ledger/decisions/2026/10/20.jsonl", "".join(json.dumps(x) + "\n" for x in lines).encode(), consent="train")
        labels = [{"schema_version": "dclab_core.decision_label.v2", "id": "dl1", "record": "dg00000000000000000001", "point": "column.is_identifier", "value": True,
                   "source": "user_decision_exposed", "at": AT},
                  {"schema_version": "dclab_core.decision_label.v2", "id": "dl2", "record": "dg00000000000000000002", "point": "workflow.next_question", "value": "data",
                   "source": "model", "at": AT}]
        store.keep_log("decisions", "dev", "ledger/labels/2026/10/20.jsonl", "".join(json.dumps(x) + "\n" for x in labels).encode(), consent="train")
        store.keep_log("decisions", "tester", "ledger/decisions/2026/10/20.jsonl", (json.dumps(decision(9, project="q1")) + "\n").encode(),
                       consent="improve")  # a tester who allows improving, not training
        store.keep_log("workflows", "dev", "projects/p1/transitions.jsonl",
                       (json.dumps({"at": AT, "move": "set_solution", "from": "WF-01", "to": "WF-02", "status": "allowed", "project": "p1"}) + "\n").encode(),
                       consent="train")
        from dclab_rnd.store import shapers

        shapers.shape(self.layout)

    def decisions(self):
        from dclab_rnd.store import Store

        store = Store(self.layout)
        return [json.loads(line) for e in store.entries("curation") for line in store.read(e).decode().splitlines()]

    def test_each_record_goes_where_it_fits_and_the_rest_is_kept_out_with_why(self):
        from dclab_rnd.store import curators

        report = curators.curate(self.layout, run="r1")
        made = self.decisions()
        by = {}
        for d in made:
            by.setdefault(d["model_type"], []).append(d)
        folders = {d["folder"] for d in by["m3_ml_decisions"] if d["quality"] == "keep"}
        self.assertIn("general/column.is_identifier", folders)  # a technical point: M3, by family
        self.assertIn("rnd/workflow.next_question", {d["folder"] for d in by["m2_workflow_decisions"]})
        self.assertIn("internal/department.proposal", {d["folder"] for d in by["m2_workflow_decisions"]})
        self.assertFalse(any(d["folder"].endswith("column.is_identifier") for d in by["m2_workflow_decisions"]))  # never a workflow model's
        qualities = {d["reason"].split(":")[0] for d in made if d["quality"] == "out"}
        self.assertIn("a duplicate of a record already placed", qualities)
        self.assertIn("broken", qualities)
        silver = [d for d in made if d["quality"] == "silver"]
        self.assertTrue(silver and all(d["weight"] < 1 for d in silver))  # the model's label: silver, apart
        refused = [d for d in made if d.get("refused")]
        self.assertTrue(any("consent" in d["refused"] for d in refused))  # the tester's 'improve' consent: no set
        self.assertIn("m6_stepwise_decisions", by)  # the graph move, a step of a process
        self.assertTrue(report["quarantined"])
        self.assertTrue(list((self.layout.root / "quarantine").rglob("*.jsonl")))
        self.assertEqual(curators.curate(self.layout, run="r2")["decisions"], 0)  # nothing new: nothing curated again

    def test_a_models_answers_stay_in_shadow_and_a_bad_path_is_refused_and_kept(self):
        from dclab_rnd.store import curators
        from dclab_rnd.store.curators import model

        seen = []

        def scripted(records):
            seen.extend(model.shape_of(r) for r in records)
            first = records[0].id
            return [{"record": first, "model_type": "m1_students", "folder": "../../etc", "quality": "keep", "weight": 1.0, "reason": "x"},
                    {"record": first, "model_type": "m1_students", "folder": "column.is_identifier", "quality": "keep", "weight": 1.0, "reason": "y"}]

        report = curators.curate(self.layout, run="r1", model=scripted)
        self.assertEqual(report["shadow"], 2)
        shadow = [d for d in self.decisions() if d["curator"] == "model:store_curator"]
        self.assertTrue(all(d["used"] is False for d in shadow))  # never used while in shadow
        self.assertTrue(any("folder" in (d.get("refused") or "") for d in shadow))  # the path past the layout: refused and kept
        self.assertFalse(any("@" in json.dumps(s) for s in seen))
        self.assertTrue(all(set(s) == {"record", "kind", "type", "flags", "point", "fields"} for s in seen))


if __name__ == "__main__":
    unittest.main()

"""A scripted person is no human (S8, 2026-10-25; the other session's e2e runs and the written people's plays): in a
workspace whose folder, or a folder above it within agent_runs, holds scripted.json, a person's label never confirms a
decision in the decision sets, and a person's correction is never a human's judgement or preference."""

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

AT = "2026-10-25T10:00:00+00:00"


@unittest.skipIf(MISSING, MISSING)
class ScriptedTest(unittest.TestCase):
    def test_the_marker_covers_a_workspace_and_every_folder_below_it(self):
        from dclab_rnd.store.keepers import workspaces

        base = Path(tempfile.mkdtemp())
        play = base / "journey_plays" / "homework_v1" / "JH-1" / "s0" / "workspace"
        real = base / "real_problems" / "x" / "workspace"
        for w in (play, real):
            (w / "ledger").mkdir(parents=True)
        (base / "journey_plays" / "scripted.json").write_text(json.dumps({"by": "script"}))
        self.assertTrue(workspaces.scripted(play, base))
        self.assertFalse(workspaces.scripted(real, base))
        names = workspaces.scripted_names(base)
        self.assertEqual(len(names), 1)
        self.assertTrue(next(iter(names)).startswith("journey_plays"))

    def test_a_scripted_persons_label_and_correction_count_for_nothing(self):
        from dclab_rnd.store import Layout, Store, curators, shapers
        from dclab_rnd.store.sets import decisions, others

        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(layout)
        for ws in ("scripted_ws", "human_ws"):
            recs = [{"schema_version": "dclab_core.decision_record.v2", "id": f"dg{ws[:3]}{i:017d}", "point": "column.is_identifier", "at": AT,
                     "project": f"{ws}-p{i}", "state": {"rows": "medium"}, "answers": [{"by": "rule", "value": True}], "used": "rule", "value_used": True}
                    for i in range(12)]
            labs = [{"schema_version": "dclab_core.decision_label.v2", "id": f"dl{ws[:3]}{i}", "record": r["id"], "point": "column.is_identifier",
                     "value": False, "source": "user_decision_exposed", "at": AT} for i, r in enumerate(recs)]
            fix = [{"id": f"c{ws}", "what": "role", "before": {"role": "feature"}, "after": {"role": "identifier"}, "project": f"{ws}-p0", "at": AT}]
            store.keep_log("decisions", ws, "ledger/decisions/2026/10/25.jsonl", "".join(json.dumps(d) + "\n" for d in recs + fix).encode(), consent="train")
            store.keep_log("decisions", ws, "ledger/labels/2026/10/25.jsonl", "".join(json.dumps(x) + "\n" for x in labs).encode(), consent="train")
        shapers.shape(layout)
        curators.curate(layout, run="r1")
        report = decisions.build(layout, scripted=frozenset({"scripted_ws"}))
        rows = []
        for split in ("train", "dev", "sealed"):
            path = layout.sets("m1_students", "column.is_identifier", "v1") / f"{split}.jsonl"
            rows += [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        by_ws = {(r["provenance"]["workspace"], r["label_source"]) for r in rows}
        self.assertIn(("human_ws", "person"), by_ws)
        self.assertNotIn(("scripted_ws", "person"), by_ws)  # the script's label confirms nothing
        self.assertEqual(report["out_of_sets"].get("decisions whose scripted person's label was set aside"), 12)
        judged = others.judged(layout, scripted=frozenset({"scripted_ws"}))
        prefs = others.preferences(layout, scripted=frozenset({"scripted_ws"}))
        self.assertEqual({j.provenance.workspace for j in judged if j.judge == "person"}, {"human_ws"})
        self.assertEqual({p.provenance.workspace for p in prefs}, {"human_ws"})


if __name__ == "__main__":
    unittest.main()

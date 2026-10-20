"""Package 32.22: cost, time and context.

Every whole run records its context (the code's commit, whether the tree was clean, Python and the libraries, the platform,
a hash of the machine), every record shaped from it carries that context and its commit, a model exchange carries its
system prompt's hash, and the catalog totals the model spend by purpose and by model type.
"""

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


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@unittest.skipIf(MISSING, MISSING)
class ContextTest(unittest.TestCase):
    def test_every_record_carries_its_context_and_the_catalog_totals_the_spend(self):
        from dclab_rnd.store import Layout, catalog, conductor, shapers
        from dclab_rnd.store.keepers import standard

        base = Path(tempfile.mkdtemp()) / "agent_runs"
        write(base / "consent.json", json.dumps({"train": True}))
        write(base / "model_requests.jsonl", json.dumps({"at": "2026-10-20T10:00:00+00:00", "purpose": "home_agent", "cost_eur": 0.012}) + "\n")
        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        record = conductor.conduct({"agent_runs": base, "test_runs": Path(tempfile.mkdtemp()), "claude_projects": None}, layout, standard.install({}))
        ctx = record["context"]
        self.assertTrue(ctx["commit"] and ctx["python"] and "numpy" in ctx["libraries"])
        self.assertEqual(len(ctx["machine"]), 12)  # a hash, never the machine's name
        shapers.shape(layout, force=True)
        r = next(iter(shapers.records(layout, "model_requests")))
        self.assertEqual(r.provenance.commit, ctx["commit"][:64])
        self.assertEqual(r.body["_context"]["python"], ctx["python"])
        doc = catalog.build(layout)
        self.assertEqual(doc["costs"]["by_purpose"], {"home_agent": 0.012})


if __name__ == "__main__":
    unittest.main()

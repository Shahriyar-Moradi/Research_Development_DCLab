"""Package 32.1: the store's layout, contracts, capture rules and append-only keeping.

A kept source reads back byte for byte by its hash; the same bytes kept again are nothing new; a changed source is a new
version beside the old; every store path is built by ``layout.py``; every store contract is registered with its schema;
every raw kind has a capture rule; a record missing a field, or whose state was written after its decision, is flagged and
never removed; the catalog, the README and the schemas are written at the store's root.
"""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class StoreTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        self.store = Store(self.layout)

    def test_a_kept_source_reads_back_and_a_change_is_a_new_version_beside_the_old(self):
        first = self.store.keep("sessions", "wtest", "experience/2026/10/20/v1.jsonl", b'{"kind": "visit.started"}\n', consent="train")
        self.assertEqual(self.store.read(first), b'{"kind": "visit.started"}\n')
        self.assertIsNone(self.store.keep("sessions", "wtest", "experience/2026/10/20/v1.jsonl", b'{"kind": "visit.started"}\n'))
        second = self.store.keep("sessions", "wtest", "experience/2026/10/20/v1.jsonl", b'{"kind": "visit.started"}\n{"kind": "chat.person"}\n')
        self.assertEqual((first.version, second.version), (1, 2))
        self.assertTrue((self.layout.root / first.path).exists() and (self.layout.root / second.path).exists())  # the old one stays
        self.assertEqual(len(self.store.entries("sessions")), 2)
        import gzip

        with gzip.open(self.layout.root / first.path, "wb") as fh:  # a valid file of other bytes: the hash, not gzip, refuses it
            fh.write(b"evil")
        with self.assertRaises(ValueError):
            self.store.read(first)

    def test_two_sources_never_share_an_id_a_long_source_is_kept_and_a_torn_index_line_is_set_aside(self):
        a = self.store.keep("sessions", "w1", "experience/a.jsonl", b"A")
        b = self.store.keep("sessions", "w1", "experience-a.jsonl", b"B")
        self.assertNotEqual(a.id, b.id)
        self.assertNotEqual(a.path, b.path)
        long = self.store.keep("tests", "w1", "s" * 1000, b"E")  # never refused at capture
        self.assertEqual(self.store.read(long), b"E")
        with open(self.layout.index("sessions"), "a", encoding="utf-8") as fh:
            fh.write('{"schema_version": "dclab_core.store_ent')  # a write cut short
        self.assertEqual(len(self.store.entries("sessions")), 2)  # still readable
        c = self.store.keep("sessions", "w1", "experience/b.jsonl", b"C")
        self.assertEqual(len(self.store.entries("sessions")), 3)
        self.assertIn("store_ent", self.layout.torn("sessions").read_text(encoding="utf-8"))  # set aside, never lost
        self.assertEqual(self.store.read(c), b"C")

    def test_a_log_keeps_whole_lines_only_and_a_newer_entry_version_stays_in_the_index(self):
        log = Path(tempfile.mkdtemp()) / "audit.jsonl"
        log.write_bytes(b'{"a": 1}\n{"b": 2')  # a line half written
        first = self.store.keep_log("audit", "w1", "audit.jsonl", log)
        self.assertEqual(self.store.read(first), b'{"a": 1}\n')
        log.write_bytes(b'{"a": 1}\n{"b": 2}\n')
        second = self.store.keep_log("audit", "w1", "audit.jsonl", log)
        self.assertEqual((second.offset, self.store.read(second)), (len(b'{"a": 1}\n'), b'{"b": 2}\n'))
        with open(self.layout.index("audit"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"schema_version": "dclab_core.store_entry.v2", "id": "future"}) + "\n")
        self.store.keep("audit", "w1", "other.json", b"{}")
        self.assertIn("store_entry.v2", self.layout.index("audit").read_text(encoding="utf-8"))  # kept for the code that reads it
        self.assertFalse(self.layout.torn("audit").exists())

    def test_the_layout_refuses_a_name_that_climbs_out_and_every_store_path_is_built_by_it(self):
        with self.assertRaises(ValueError):
            self.layout.raw_file("sessions", "../other", "2026-10-20", "x.gz")
        with self.assertRaises(ValueError):
            self.layout.sets("m9_unknown")
        with self.assertRaises(ValueError):
            self.layout.raw("secrets")
        def from_store(node):  # an expression built on a layout or a store's root
            return any((isinstance(n, ast.Name) and n.id in ("layout", "self_layout")) or (isinstance(n, ast.Attribute) and n.attr in ("layout", "root"))
                       for n in ast.walk(node))

        for path in sorted((ROOT / "dclab_rnd" / "store").rglob("*.py")):
            if path.name == "layout.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    self.assertFalse(from_store(node.left), f"{path.name}:{node.lineno} joins a store path itself: ask layout.py")
        with self.assertRaises(ValueError):
            self.layout.entry("../../etc/passwd")  # an index line from a merged branch never reads outside the store
        self.assertEqual(self.layout.workspace("wtest"), "wtest")
        self.assertTrue(self.layout.workspace("/Users/x/agent_runs").startswith("w") and "/" not in self.layout.workspace("/Users/x/agent_runs"))

    def test_every_store_contract_is_registered_with_its_schema_and_every_kind_has_a_rule(self):
        from dclab_core import store, store_rules
        from dclab_core.contracts import CONTRACTS
        from dclab_rnd.store import catalog

        for name in catalog.SCHEMAS:
            self.assertIn(name, CONTRACTS)
            self.assertIn("schema_version", CONTRACTS[name].model_json_schema()["properties"])
        self.assertEqual(len(store.MODEL_TYPES), 9)  # the eight of ModelType and m7_routing (P8, 2026-10-29)
        self.assertEqual(set(store.EXAMPLE_SHAPES), set(store.MODEL_TYPES))
        ruled = {r.kind for r in store_rules.RULES.values()}
        self.assertEqual(ruled, set(store.KINDS), "every raw kind has a capture rule")
        from dclab_core.ledger import POINTS
        from dclab_rnd.agents import harness
        from dclab_rnd.store.tasks import AGENT_TASKS

        self.assertEqual(set(AGENT_TASKS), set(harness.AGENTS), "every agent of the harness says which capture rules check it")
        for agent, tasks in AGENT_TASKS.items():
            self.assertTrue(set(tasks) <= set(store_rules.RULES), (agent, set(tasks) - set(store_rules.RULES)))
        self.assertIn("decisions", store_rules.RULES["decision"].kind)  # every ledger point's decision is checked by the decision rule
        self.assertTrue(POINTS)

    def test_every_check_flags_and_nothing_is_removed(self):
        from dclab_core import store_rules

        at = "2026-10-20T10:00:00+00:00"
        planted = {
            "missing": {"kind": "decisions", "type": "decision", "at": at, "body": {"point": "p", "state": {"rows": "1k-10k"}, "answers": [1],
                                                                                  "used": "rule", "group": "project:p", "state_at": at}},
            "late": {"kind": "decisions", "type": "decision", "at": at, "body": {"point": "p", "state": {}, "answers": [1], "used": "rule",
                                                                               "value_used": True, "group": "g", "state_at": "2026-10-20T11:00:00+00:00"}},
            "untimed": {"kind": "decisions", "type": "decision", "at": at, "body": {"point": "p", "state": {}, "answers": [1], "used": "rule",
                                                                                  "value_used": True, "group": "g"}},
            "unconfirmed": {"kind": "decisions", "type": "label", "at": at, "body": {"record": "dr-1", "value": True, "source": "teacher", "group": "g"}},
            "unknown_rule": {"kind": "decisions", "type": "mystery", "body": {}},
        }
        for i, (what, record) in enumerate(planted.items()):
            entry = self.store.keep("decisions", "wtest", f"planted/{what}.json", json.dumps(record).encode())
            self.assertIn(what, [f.what for f in store_rules.check(record)], what)
            self.assertEqual(self.store.read(entry), json.dumps(record).encode())  # kept whole whatever the check said
        self.assertEqual(len(self.store.entries("decisions")), len(planted))
        shadow = {"kind": "decisions", "type": "decision", "at": at, "body": {"point": "p", "state": {}, "answers": [1], "used": "none",
                                                                            "group": "g", "state_at": at}}
        self.assertEqual(store_rules.check(shadow), [])  # no value used, in shadow: not missing
        person = {"kind": "decisions", "type": "label", "at": at, "body": {"record": "dr-1", "value": True, "source": "person", "group": "g"}}
        self.assertEqual(store_rules.check(person), [])
        self.assertTrue(all(f.keeper for f in store_rules.check(planted["missing"])))  # the keeper that should have caught it is named
        unsealed = {"kind": "teachers", "type": "answer", "body": {"teacher": "claude", "point": "p", "answer": True, "version": "1"}}
        self.assertIn(("missing", "group"), {(f.what, f.field) for f in store_rules.check(unsealed)})  # no group: it cannot be sealed
        ledger_label = {"kind": "decisions", "type": "label", "at": at, "body": {"record": "d", "value": True, "source": "user_decision_exposed", "group": "g"}}
        self.assertEqual(store_rules.check(ledger_label), [])  # the ledger's words read as the store's

    def test_every_rule_that_feeds_a_set_is_sealed_and_names_a_real_keeper(self):
        from dclab_core import store_rules
        from dclab_core.ledger import LABEL_SOURCES
        from dclab_core.store import LEDGER_SOURCES
        from dclab_rnd.store.keepers import standard

        keepers = set(standard.install({})) | {"hooks", "agents", "curators"}  # the last two come with 32.7 and 32.9
        for task, rule in store_rules.RULES.items():
            self.assertIn(rule.keeper, keepers, task)
            if rule.feeds:
                self.assertTrue(rule.sealed_key, f"{task} feeds a set but has no key to seal it by")
        self.assertEqual(set(LEDGER_SOURCES), set(LABEL_SOURCES))

    def test_the_catalog_readme_and_schemas_are_written_at_the_root(self):
        from dclab_rnd.store import catalog

        self.store.keep("tests", "wtest", "test_runs/runs.jsonl", b'{"ok": true}\n')
        copy = Path(tempfile.mkdtemp()) / "STORE_CATALOG.json"
        before = catalog.COPY.read_bytes() if catalog.COPY.exists() else None
        catalog.write(self.layout)  # a scratch store never replaces the repository's view
        self.assertEqual(catalog.COPY.read_bytes() if catalog.COPY.exists() else None, before)
        doc = catalog.write(self.layout, copy=copy)
        self.assertNotIn("built_at", json.loads(copy.read_text(encoding="utf-8")))
        self.assertEqual(doc["kinds"]["tests"]["entries"], 1)
        self.assertEqual(set(doc["model_types"]), {"m1_students", "m2_workflow_decisions", "m3_ml_decisions", "m4_workflow_llm",
                                                    "m5_solver_llm", "m6_stepwise_decisions", "m7_judges", "m8_retrieval", "m7_routing"})
        self.assertIn("m8_retrieval", self.layout.readme.read_text(encoding="utf-8"))
        for name in catalog.SCHEMAS:
            self.assertTrue((self.layout.schemas / f"{name}.json").exists())
        self.assertEqual(json.loads(copy.read_text(encoding="utf-8"))["kinds"]["tests"]["entries"], 1)

    def test_examples_carry_provenance_and_reasoning_is_labelled(self):
        from pydantic import ValidationError

        from dclab_core import store

        prov = store.Provenance(entry="decisions:wtest:x:v1", sha256="0" * 64, source="ledger", workspace="wtest", at="2026-10-20T10:00:00+00:00")
        ex = store.ChatExample(id="c1", model_type="m4_workflow_llm", split="train", provenance=prov, task="explain", stage="report",
                               messages=[store.Message(role="user", content="Why?"),
                                         store.Message(role="assistant", content="Because.", reasoning="The record says so.", reasoning_source="from_record")])
        self.assertEqual(store.ChatExample.read(ex.model_dump_json()), ex)
        with self.assertRaises(ValidationError):
            store.Message(role="assistant", content="x", reasoning="y", reasoning_source="guessed")
        with self.assertRaises(ValidationError):  # an assistant turn with no reasoning, and no reason why
            store.Message(role="assistant", content="x")
        store.Message(role="assistant", content="x", no_reasoning="a rule's answer with no explanation in its record")
        with self.assertRaises(ValidationError):
            store.TrajectoryStep(n=0, tool="read")
        with self.assertRaises(ValidationError):  # a shape that is not this model type's
            store.DecisionExample(id="d", model_type="m4_workflow_llm", split="train", provenance=prov, point="p", group="g")
        with self.assertRaises(ValidationError):  # a table's value in a state
            store.DecisionExample(id="d", model_type="m3_ml_decisions", split="train", provenance=prov, point="p", group="binary",
                                  state={"sample": "alice@example.com"})
        with self.assertRaises(ValidationError):  # a label that does not say who knows it
            store.DecisionExample(id="d", model_type="m3_ml_decisions", split="train", provenance=prov, point="p", group="binary", label=True)
        steps = [store.EpisodeStep(n=0, action="a"), store.EpisodeStep(n=2, action="b")]
        with self.assertRaises(ValidationError):  # a gap not said
            store.Episode(id="e", model_type="m6_stepwise_decisions", split="train", provenance=prov, process="walk", steps=steps)
        store.Episode(id="e", model_type="m6_stepwise_decisions", split="train", provenance=prov, process="walk", steps=steps, gaps=[1])


if __name__ == "__main__":
    unittest.main()

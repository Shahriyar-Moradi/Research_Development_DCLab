"""Package 24.7: the capture of phase 2. A problem of several parts leaves what was decided about it in the record: the map drawn and the person's changes to it, the approach
routed to each sub-problem with what blocked the others, the composition chosen on validation and the programme built; the ledger holds the decisions with their reasons and the
holdout's measured label; and `make datasets` builds the two sets (problem_maps, routes) from it, under the consent that allows them and no other."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import datasets as builders, experience as x, ledger as core_ledger
    from dclab_core.replicas import fraud_programme as fp
    from dclab_rnd import consent, datasets, experience
    from dclab_rnd.agent_eval.journeys import programme as journeys
    from dclab_rnd.draft import map as home_map
except ImportError as error:  # the agent environment lacks numpy or fastapi
    MISSING = str(error)
else:
    MISSING = ""

FRAUD = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review about 2% of the rides each day.")
NEW = ("problem.mapped", "problem.part_changed", "solution.routed", "solution.composed", "programme.built")


@unittest.skipIf(MISSING, MISSING)
class ContractTest(unittest.TestCase):
    def test_the_five_kinds_are_declared_with_their_classes_and_the_contract_is_v9(self):
        self.assertTrue(set(NEW) <= set(x.KINDS))
        self.assertEqual(x.ExperienceEvent.model_fields["schema_version"].default, "dclab_core.experience_event.v9")
        self.assertIn("dclab_core.experience_event.v8", x.READ_AS_CURRENT)
        for kind in NEW:
            self.assertTrue(set(x.KINDS[kind].values()) <= {"meta", "count", "person_text"}, "nothing a model said and no cell value")
        self.assertEqual(x.KINDS["problem.part_changed"]["reason"], "person_text")

    def test_a_person_s_words_cannot_pass_as_meta(self):
        ids = {"workspace": "w1", "conversation": "d1"}
        ok = x.make("problem.part_changed", "person", ids, {"map": "abc", "action": "remove", "part": "part2", "by": "person", "parts_before": 4, "parts_after": 3},
                    id="ev0000001", at="2026-10-16T10:00:00+00:00", seq=1)
        self.assertEqual(ok.classes["parts_before"], "count")
        with self.assertRaises(Exception):
            x.make("problem.part_changed", "person", ids, {"map": "abc", "action": "remove this because the fraud on my route is a thing the team never reviews at all, as " * 5},
                   id="ev0000002", at="2026-10-16T10:00:01+00:00", seq=2)

    def test_a_v8_event_is_read_as_v9_and_a_v2_label_as_v3(self):
        old = {"schema_version": "dclab_core.experience_event.v8", "id": "ev0000003", "at": "2026-10-16T10:00:00+00:00", "seq": 0, "ids": {"workspace": "w1"},
               "actor": "person", "kind": "visit.started", "payload": {"surface": "web"}, "classes": {"surface": "meta"}}
        self.assertEqual(x.ExperienceEvent.read(old).schema_version, "dclab_core.experience_event.v9")
        label = core_ledger.DecisionLabel.read({"schema_version": "dclab_core.decision_label.v2", "id": "dl0000001", "record": "r1", "point": "column.is_identifier",
                                                "value": True, "source": "benchmark", "by": "replica", "at": "2026-10-16T10:00:00+00:00"})
        self.assertEqual(label.schema_version, "dclab_core.decision_label.v3")
        self.assertIn("holdout_vs_best_leaf", core_ledger.Evidence.model_fields["kind"].annotation.__args__)

    def test_the_manifest_is_v5_and_reads_v4(self):
        self.assertEqual(builders.DatasetManifest.model_fields["schema_version"].default, "dclab_core.dataset_manifest.v5")
        self.assertEqual(sorted(set(builders.Builder.__args__) & {"problem_maps", "routes"}), ["problem_maps", "routes"])

    def test_the_new_decision_point_is_declared_with_its_state(self):
        self.assertIn("solution.compose", core_ledger.POINTS)
        self.assertEqual(core_ledger.check_state({"parts": 3, "positives": 40, "holdout_used": 0}), [])
        self.assertTrue(core_ledger.check_state({"positives": "forty"}))


@unittest.skipIf(MISSING, MISSING)
class BuildersTest(unittest.TestCase):
    def maps(self):
        proposed = {"fingerprint": "aaaa", "size": "complex", "parts": [{"id": "part1"}]}
        final = {"fingerprint": "bbbb", "size": "complex", "parts": [{"id": "part1"}, {"id": "part2"}]}
        return [{"draft": "d1", "project": "p1", "problem_card": {"goal": "find fraud"}, "proposed": proposed, "final": final, "confirmed": True,
                 "edits": [{"action": "add", "by": "person", "part": "part2", "reason": "dropped by the builder", "from": "aaaa", "to": "bbbb"}],
                 "outcome": {"beats_best_leaf": True, "difference": 0.3}, "map": "bbbb",
                 "routes": [{"sub_problem": "sp1", "chosen": "anomaly", "source": "rules", "state": {"family": "anomaly", "labels": "none"},
                             "candidates": [{"approach": "anomaly", "rank": 1, "blocked_by": [], "unknown": []}, {"approach": "rare_event", "rank": 2, "blocked_by": ["labels"], "unknown": []}],
                             "outcome": {"leaf": "built", "chosen": "nearest_neighbours", "beats_best_leaf": True}}]},
                {"draft": "d2", "project": "p2", "problem_card": {"goal": "x"}, "proposed": None, "final": None, "routes": []}]

    def test_a_brief_to_map_example_per_draft_with_its_edits_as_the_correction(self):
        rows, m = builders.problem_maps(self.maps())
        self.assertEqual(len(rows), 1, "a draft with no map built from is no example")
        row = rows[0]
        self.assertTrue(row["corrected"])
        self.assertEqual(row["edits"], [{"action": "add", "by": "person", "part": "part2", "keep": None, "other": None, "from": "aaaa", "to": "bbbb"}], "no reason text in an edit")
        self.assertEqual((row["group"], row["split"] in ("train", "dev", "sealed")), ("project:p1", True))
        self.assertEqual((m.builder, m.name, m.schema_version), ("problem_maps", "maps.brief_to_map", "dclab_core.dataset_manifest.v5"))
        self.assertEqual(m.counts["corrected"], {"yes": 1})
        self.assertEqual(m.counts["outcome"], {"beat_best_leaf": 1})
        self.assertEqual({(s.kind, s.id) for s in m.sources}, {("draft", "d1"), ("project", "p1")})

    def test_a_sub_problem_to_approach_example_per_route_with_codes_only(self):
        rows, m = builders.routes(self.maps())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["chosen"], row["source"], row["outcome"]["leaf"]), ("anomaly", "rules", "built"))
        self.assertEqual(row["candidates"][1], {"approach": "rare_event", "rank": 2, "blocked_by": ["labels"], "unknown": []})
        self.assertEqual((m.builder, m.counts["chosen"], m.counts["outcome"]), ("routes", {"anomaly": 1}, {"built": 1}))
        self.assertIn("Codes and counts only", " ".join(m.limits))

    def test_no_programme_is_counted_as_such(self):
        items = self.maps()
        items[0]["outcome"] = None
        items[0]["routes"][0]["outcome"] = None
        self.assertEqual(builders.problem_maps(items)[1].counts["outcome"], {"no_programme": 1})
        self.assertEqual(builders.routes(items)[1].counts["outcome"], {"no_programme": 1})


@unittest.skipIf(MISSING, MISSING)
class RoleStepsTest(unittest.TestCase):
    """The new roles, when a model serves them, are steps of whatever asked them, recorded by the harness as the older roles are."""

    def setUp(self):
        import os
        from unittest import mock

        from dclab_rnd.draft.store import DraftStore

        self.home = Path(tempfile.mkdtemp())
        self.log = experience.FileLog(self.home / "experience", "wtest")
        self.log.home = self.home
        experience.install(self.log)
        self.addCleanup(experience.install, None)
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-never-logged", "OPENAI_MODEL": "std-model", "DCLAB_LLM_PROVIDER": ""})
        env.start()
        self.addCleanup(env.stop)
        self.store = DraftStore(self.home / "drafts")

    def test_the_decomposer_the_map_reviewer_and_the_router_each_leave_a_step(self):
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.models.usage import FileUsage
        from tests.test_models_gateway import gateway as scripted

        gw, _ = scripted(FileUsage(self.home / "usage.jsonl"), [{"content": "{}", "usage": {"input_tokens": 3, "output_tokens": 2}}] * 12)
        did = self.store.create(FRAUD)["id"]
        advisor = GatewayAdvisor(gw, draft_id=did)
        home_map.draw(self.store, did, advisor=advisor)
        home_map.edit(self.store, did, "confirm", advisor=advisor)
        steps = [e for e in self.log.read(kinds=["agent.step"], limit=1000)]
        agents = {e.payload["agent"] for e in steps}
        self.assertTrue({"role.decomposer", "role.map_reviewer", "role.solution_router"} <= agents, agents)
        self.assertTrue(all(e.ids.conversation == did for e in steps if e.payload["agent"].startswith("role.")))
        self.assertTrue(all(e.payload.get("reply") is None for e in steps), "a role's JSON may name columns: it stays in its request")


@unittest.skipIf(MISSING, MISSING)
class JourneyCaptureTest(unittest.TestCase):
    """The written person of 24.6, played in a workspace whose log and ledger can be read afterwards."""

    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        cls.s = cls.client.app.state.services
        cls.home = Path(cls.s.store.home)
        for use in ("improve", "train"):
            consent.set_workspace_consent(cls.home, use, True)
        cls.row = journeys.play(journeys.load("dev")[0], seed=0, stores=(cls.s.drafts, cls.s.projects))
        cls.events = cls.s.experience.read(limit=100_000)
        cls.records = cls.s.ledger.records()
        cls.labels = cls.s.ledger.labels()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def kind(self, name):
        return [e for e in self.events if e.kind == name]

    def test_the_journey_itself_was_right(self):
        self.assertTrue(self.row["right"], self.row["why"])

    def test_the_map_the_changes_the_routes_the_composition_and_the_programme_are_in_the_record(self):
        mapped = self.kind("problem.mapped")
        self.assertTrue(mapped)
        self.assertEqual((mapped[0].payload["size"], mapped[0].payload["families"], mapped[0].payload["parts"]), ("complex", ["anomaly", "rare_event", "entity_tables", "ranking"], 4))
        self.assertIn("confirm", [e.payload["action"] for e in self.kind("problem.part_changed")])
        routed = self.kind("solution.routed")
        self.assertEqual({e.payload["sub_problem"] for e in routed}, {"sp1", "sp2", "sp3", "sp4"})
        rings = [e for e in routed if e.payload["sub_problem"] == "sp3"]
        self.assertGreaterEqual(len(rings), 2, "the rings were routed again when the table of accounts arrived")
        self.assertIsNone(rings[0].payload.get("chosen"), "before the table the rings had no approach")
        self.assertEqual(rings[-1].payload["chosen"], "entity_tables")
        composed = self.kind("solution.composed")
        built = self.kind("programme.built")
        self.assertEqual(len(composed), 1)
        self.assertEqual((built[-1].payload["state"], built[-1].payload["holdout_read"], built[-1].payload["beats_best_leaf"]), ("composed", "yes", "yes"))
        self.assertEqual(composed[0].payload["chosen"], "rank_max")
        self.assertGreater(built[-1].payload["low"], 0)

    def test_nothing_of_a_rows_values_or_the_persons_words_is_in_these_events(self):
        text = json.dumps([e.payload for k in NEW for e in self.kind(k)])
        for word in ("Find the fake rides", "ph_", "dv_"):
            self.assertNotIn(word, text)

    def test_the_ledger_holds_each_decision_with_its_reason_and_the_holdout_labels_the_composition(self):
        points = {r.point for r in self.records}
        self.assertTrue({"map.size", "map.edit", "solution.route", "solution.compose"} <= points)
        routes = [r for r in self.records if r.point == "solution.route"]
        self.assertGreaterEqual(len(routes), 5, "four sub-problems, and the rings again when the table arrived")
        compose = next(r for r in self.records if r.point == "solution.compose")
        self.assertEqual(compose.value_used, "rank_max")
        self.assertIn("rank_max", compose.options)
        self.assertIn("blk3", compose.options)
        self.assertEqual([a.by for a in compose.answers], ["rule"])
        self.assertIn("validation", compose.answers[0].reason)
        self.assertGreater(compose.state["positives"], 5)
        self.assertEqual(core_ledger.check_state(compose.state), [])
        label = next(l for l in self.labels if l.record == compose.id)
        self.assertEqual((label.source, label.value, label.evidence.kind), ("measured", "rank_max", "holdout_vs_best_leaf"))
        self.assertGreater(label.evidence.value, 0)

    def test_the_timeline_reads_them(self):
        lines = experience.lines([e for k in NEW for e in self.kind(k)])
        self.assertTrue(any("problem.mapped" in l and "families" in l for l in lines))
        self.assertTrue(any("programme.built" in l and "state=composed" in l for l in lines))

    def test_make_datasets_builds_both_sets_from_it_and_only_under_train(self):
        project = self.s.projects.list()[0]["id"]
        proj = self.s.projects.get(project)
        proj["settings"] = {**(proj.get("settings") or {}), "consent": {"train": True, "improve": True}}
        self.s.projects.save(proj)
        out = {(b["builder"], b["name"]): b for b in datasets.build(self.s)["built"]}
        maps, routes = out["problem_maps", "brief_to_map"], out["routes", "sub_problem_to_approach"]
        self.assertEqual(maps["rows"], 1)
        self.assertGreaterEqual(routes["rows"], 4)
        rows = [json.loads(l) for f in ("rows.jsonl", "sealed.jsonl") if (Path(maps["folder"]) / f).exists() for l in (Path(maps["folder"]) / f).read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["final"]["size"], "complex")
        self.assertFalse(row["corrected"], "the person confirmed the map as drawn")
        self.assertTrue(row["confirmed"])
        self.assertTrue(row["outcome"]["beats_best_leaf"])
        self.assertEqual(row["outcome"]["state"], "composed")
        route_rows = [json.loads(l) for f in ("rows.jsonl", "sealed.jsonl") if (Path(routes["folder"]) / f).exists() for l in (Path(routes["folder"]) / f).read_text().splitlines()]
        by_sub = {}
        for r in route_rows:
            by_sub.setdefault(r["sub_problem"], r)
        self.assertEqual(by_sub["sp1"]["chosen"], "anomaly")
        self.assertEqual(by_sub["sp1"]["outcome"]["leaf"], "built")
        self.assertTrue(any(c["blocked_by"] for r in route_rows for c in r["candidates"]), "the candidates a route refuted are in the example, with the code that refuted them")
        manifest = builders.DatasetManifest.read((Path(maps["folder"]) / "manifest.json").read_text())
        self.assertEqual({s.kind for s in manifest.sources}, {"draft", "project"})
        consent.set_workspace_consent(self.home, "train", False)
        try:
            again = {(b["builder"], b["name"]) for b in datasets.build(self.s)["built"]}
            self.assertNotIn(("problem_maps", "brief_to_map"), again, "without train these sets are not built")
        finally:
            consent.set_workspace_consent(self.home, "train", True)

    def test_the_turn_carries_the_map_and_the_programme(self):
        rows, manifest = builders.episodes([e.model_dump(mode="json") for e in self.events], [], text=True, consent="train")
        parts = [p for r in rows for p in r["problem_parts"]]
        self.assertTrue(any(p["kind"] == "problem.mapped" for p in parts))
        rows_improve, _ = builders.episodes([e.model_dump(mode="json") for e in self.events], [], text=False, consent="improve")
        text = json.dumps([r["problem_parts"] for r in rows_improve])
        self.assertNotIn("reason", text, "under improve alone: codes and counts, no free text")


if __name__ == "__main__":
    unittest.main()

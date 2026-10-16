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
    from dclab_rnd import consent, datasets, experience, ledger as ledger_module
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

    def test_the_hosts_decision_that_the_person_corrected_the_map_wins_and_real_fingerprints_are_compared_otherwise(self):
        items = self.maps()
        items[0]["proposed"] = {"fingerprint": "c28fb91769d89fc3"}
        items[0]["final"] = {"fingerprint": "c28fb91769d89fc3", "size": "complex"}
        items[0]["corrected"] = True                                             # the person added a part and then a redraw gave the same map back: the host says so
        self.assertTrue(builders.problem_maps(items)[0][0]["corrected"])
        del items[0]["corrected"]
        self.assertFalse(builders.problem_maps(items)[0][0]["corrected"], "no flag: the same fingerprints are the same map")
        items[0]["final"]["fingerprint"] = "1fccb00cc350286b"
        self.assertTrue(builders.problem_maps(items)[0][0]["corrected"], "no flag: a map that is not the first one drawn is corrected")

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

    def test_the_data_need_agent_leaves_a_step_too_once_there_is_a_table(self):
        from dclab_rnd.agent_eval.journeys import _upload
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.models.usage import FileUsage
        from tests.test_models_gateway import gateway as scripted

        replica = fp.generate(seed=0, n_orders=3000, n_bikers=60, n_accounts=400, n_rings=3, n_markers=4)
        mine = Path(tempfile.mkdtemp())
        fp.save(replica, mine)
        gw, _ = scripted(FileUsage(self.home / "usage.jsonl"), [{"content": "{}", "usage": {"input_tokens": 3, "output_tokens": 2}}] * 30)
        did = self.store.create(FRAUD)["id"]
        agent = HomeAgent(self.store, None)
        agent.start(did)
        _upload(self.store, agent, did, fp.RIDE_FILE, mine / fp.RIDE_FILE)
        home_map.draw(self.store, did, advisor=GatewayAdvisor(gw, draft_id=did), redraw=True)
        agents = {e.payload["agent"] for e in self.log.read(kinds=["agent.step"], limit=1000)}
        self.assertIn("role.data_need", agents)


@unittest.skipIf(MISSING, MISSING)
class WhatAPersonTypesTest(unittest.TestCase):
    """A title a person types is their words: it is recorded as an id, never as meta, and an improve row never carries a reason."""

    TITLE = "Mr Karimi's cousin Ali at the 5 Vali St shop"

    def setUp(self):
        from dclab_rnd.draft.store import DraftStore

        self.home = Path(tempfile.mkdtemp())
        self.log = experience.FileLog(self.home / "experience", "wtest")
        self.log.home = self.home
        experience.install(self.log)
        self.addCleanup(experience.install, None)
        self.store = DraftStore(self.home / "drafts")
        self.did = self.store.create(FRAUD)["id"]
        experience.record("draft.created", "person", {"problem": FRAUD}, conversation=self.did)  # the turn every later event of the conversation belongs to
        home_map.draw(self.store, self.did)

    def events(self):
        return [e for e in self.log.read(limit=10_000) if e.kind in NEW]

    def test_an_added_part_is_recorded_by_its_id_and_its_title_is_in_no_field(self):
        home_map.edit(self.store, self.did, "add", title=self.TITLE, family="anomaly", unit="ride", labels="none")
        changed = [e for e in self.events() if e.kind == "problem.part_changed" and e.payload["action"] == "add"]
        self.assertEqual(len(changed), 1)
        self.assertRegex(changed[0].payload["part"], r"^part\d+$")
        self.assertEqual(changed[0].classes["part"], "meta")
        for e in self.events():
            self.assertNotIn("Karimi", json.dumps(e.payload), e.kind)
            self.assertNotIn("Vali", json.dumps(e.payload), e.kind)

    def test_an_improve_row_carries_codes_and_counts_and_never_a_reason(self):
        home_map.edit(self.store, self.did, "add", title=self.TITLE, family="anomaly", unit="ride", labels="none")
        experience.record("problem.part_changed", "person", {"map": "abcd", "action": "drop_leaf", "part": "blk3", "by": "person", "parts_before": 3, "parts_after": 2,
                                                             "reason": "Ali's shop at 5 Vali St has no accounts file"}, conversation=self.did)
        events = [e.model_dump(mode="json") for e in self.log.read(limit=10_000)]
        rows, _ = builders.episodes(events, [], text=False, consent="improve")
        text = json.dumps([r["problem_parts"] for r in rows])
        self.assertIn("drop_leaf", text)
        for word in ("Karimi", "Vali", "Ali", "reason"):
            self.assertNotIn(word, text)

    def test_the_correction_is_decided_from_the_persons_edits_before_any_redaction(self):
        from types import SimpleNamespace

        home_map.edit(self.store, self.did, "add", title=self.TITLE, family="anomaly", unit="ride", labels="none")
        home_map.edit(self.store, self.did, "confirm")
        self.store.update(self.did, lambda d: d.update(project_id="p1", status="built"))
        items = datasets._map_items(SimpleNamespace(drafts=self.store), {"p1": {"id": "p1"}})
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertTrue(item["corrected"])
        self.assertRegex(item["final"]["fingerprint"], r"^[0-9a-f]{16}$")
        self.assertRegex(item["proposed"]["fingerprint"], r"^[0-9a-f]{16}$")
        self.assertNotEqual(item["proposed"]["fingerprint"], item["final"]["fingerprint"])
        add = next(e for e in item["edits"] if e["action"] == "add")
        self.assertEqual(add["to"] in (item["final"]["fingerprint"], None) or add["from"] == item["proposed"]["fingerprint"], True, "an edit joins the maps by the same fingerprints")
        rows, _ = builders.problem_maps(items)
        self.assertTrue(rows[0]["corrected"])

    def test_a_redraw_is_a_new_proposal_not_a_correction_and_a_draft_from_before_has_no_first_map(self):
        from types import SimpleNamespace

        home_map.draw(self.store, self.did, redraw=True)
        home_map.edit(self.store, self.did, "confirm")
        self.store.update(self.did, lambda d: d.update(project_id="p1", status="built"))
        item = datasets._map_items(SimpleNamespace(drafts=self.store), {"p1": {"id": "p1"}})[0]
        self.assertFalse(item["corrected"])

        def forget(d):
            d["problem_map"].pop("first", None)
            d["problem_map"]["history"] = [{"action": "remove", "by": "person", "part": "part2"}]
        self.store.update(self.did, forget)
        older = datasets._map_items(SimpleNamespace(drafts=self.store), {"p1": {"id": "p1"}})[0]
        self.assertTrue(older["corrected"], "the edits say so even when the first map was not kept")
        self.assertIsNone(older["proposed"])

    def test_a_dropped_leaf_is_a_part_change_with_the_persons_reason_as_their_text(self):
        from dclab_rnd.studio import programme as sp
        from dclab_rnd.studio.store import ProjectStore
        from tests.test_studio_programme import make_project

        store = ProjectStore(self.home / "projects")
        pid, _ = make_project(store, tables=False, plan_with_accounts=False)
        sp.run(store, pid)
        sp.drop(store, pid, "blk3", "we have no accounts file at all")
        dropped = [e for e in self.log.read(limit=10_000) if e.kind == "problem.part_changed" and e.payload["action"] == "drop_leaf"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual((dropped[0].payload["part"], dropped[0].ids.project, dropped[0].classes["reason"]), ("blk3", pid, "person_text"))
        self.assertEqual((dropped[0].payload["parts_before"], dropped[0].payload["parts_after"]), (4, 3))

    def test_a_part_that_is_not_an_id_is_not_recorded_at_all(self):
        from dclab_rnd.experience import hooks

        held = home_map.current(self.store.get(self.did))
        hooks.map_changed(self.did, held, held, "remove", "person", part="my own words")
        last = [e for e in self.events() if e.kind == "problem.part_changed"][-1]
        self.assertNotIn("part", last.payload)


@unittest.skipIf(MISSING, MISSING)
class CompositionLabelTest(unittest.TestCase):
    """What the holdout supports about a composition, and when it supports nothing."""

    @classmethod
    def setUpClass(cls):
        from dclab_core import programme as core
        from dclab_rnd.studio.store import ProjectStore
        from tests.test_studio_programme import make_project
        from dclab_rnd.studio import programme as sp

        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid, _ = make_project(store)
        sp.run(store, pid)
        cls.base = sp.stored(store.get(pid))
        cls.core = core

    def variant(self, chosen="rank_max", **holdout):
        """The programme as if the composition were a fusion (the small replica chooses its best leaf alone) with the given holdout numbers."""
        data = self.base.model_dump()
        data["composition"]["chosen"] = chosen
        data["composition"]["candidates"].setdefault(chosen, 0.9)
        data["holdout"].update({"chosen": chosen, "beats_best_leaf": True, "difference": {"value": 0.3, "low": 0.1, "high": 0.5}, **holdout})
        data["fingerprint"] = ""
        return self.core.Programme.model_validate(data)

    def labels_for(self, programme):
        from dclab_rnd.ledger import capture

        book = ledger_module.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "wtest")
        ledger_module.install(book)
        try:
            capture.composition_chosen("p1", programme)
            return book.records(), book.labels()
        finally:
            ledger_module.install(None)

    def test_the_composition_that_beat_the_best_leaf_is_the_label(self):
        records, labels = self.labels_for(self.variant())
        self.assertEqual(len(records), 1)
        self.assertEqual([(l.value, l.source, l.evidence.kind) for l in labels], [("rank_max", "measured", "holdout_vs_best_leaf")])
        self.assertEqual(labels[0].evidence.value, 0.3)

    def test_a_leaf_chosen_alone_labels_nothing_since_nothing_was_compared(self):
        records, labels = self.labels_for(self.variant(chosen=self.base.composition.best_leaf, beats_best_leaf=False, difference={"value": 0.0, "low": 0.0, "high": 0.0}))
        self.assertEqual((len(records), labels), (1, []))

    def test_a_composition_clearly_below_its_best_leaf_labels_the_best_leaf(self):
        worse = self.variant(beats_best_leaf=False, difference={"value": -0.3, "low": -0.5, "high": -0.1})
        records, labels = self.labels_for(worse)
        self.assertEqual([l.value for l in labels], [self.base.composition.best_leaf])
        self.assertEqual(records[0].value_used, "rank_max", "the record is what was chosen; the label is what the holdout supports instead")

    def test_an_interval_across_zero_labels_nothing(self):
        records, labels = self.labels_for(self.variant(beats_best_leaf=False, difference={"value": 0.05, "low": -0.2, "high": 0.3}))
        self.assertEqual((len(records), labels), (1, []))

    def test_a_second_look_at_the_holdout_labels_nothing(self):
        data = self.variant().model_dump()
        data["reads_before"] = 1
        data["fingerprint"] = ""
        records, labels = self.labels_for(self.core.Programme.model_validate(data))
        self.assertEqual((len(records), labels), (1, []))
        self.assertEqual(records[0].state["holdout_used"], 1)

    def test_a_programme_that_chose_nothing_records_nothing(self):
        data = self.base.model_dump()
        data.update(composition=None, holdout=None, state="planned", fingerprint="")
        records, labels = self.labels_for(self.core.Programme.model_validate(data))
        self.assertEqual((records, labels), ([], []))

    def test_a_reviewer_is_never_asked_to_guess_it(self):
        self.assertFalse(core_ledger.blind_eligible(self.labels_for(self.variant())[0][0]))


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
        from collections import Counter

        self.assertEqual(Counter(e.payload["sub_problem"] for e in routed), {"sp1": 1, "sp2": 1, "sp3": 2, "sp4": 1}, "each approach once; the rings again when the table arrived")
        rings = [e for e in routed if e.payload["sub_problem"] == "sp3"]
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
        self.assertEqual(len(routes), 5, "four sub-problems, and the rings again when the table arrived")
        rings = [r for r in routes if r.state.get("sub_problem") == "sp3"]
        self.assertEqual([r.state["related_tables"] for r in rings], [0, 1], "the fact that changed is in the state: a related table arrived")
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
        self.assertRegex(row["final"]["fingerprint"], r"^[0-9a-f]{16}$", "a fingerprint is not an identifier to redact")
        self.assertEqual(row["proposed"]["fingerprint"], row["final"]["fingerprint"])
        tables = {n["table"] for sub in row["final"]["sub_problems"] for n in sub.get("needs") or [] if n.get("table")}
        self.assertTrue(all(t.startswith("table_") for t in tables), tables)
        self.assertTrue(row["confirmed"])
        self.assertTrue(row["outcome"]["beats_best_leaf"])
        self.assertEqual(row["outcome"]["state"], "composed")
        route_rows = [json.loads(l) for f in ("rows.jsonl", "sealed.jsonl") if (Path(routes["folder"]) / f).exists() for l in (Path(routes["folder"]) / f).read_text().splitlines()]
        by_sub = {}
        for r in route_rows:
            by_sub.setdefault(r["sub_problem"], r)
        self.assertEqual(by_sub["sp1"]["chosen"], "anomaly")
        self.assertEqual(by_sub["sp1"]["outcome"]["leaf"], "built")
        rings = [r for r in route_rows if r["sub_problem"] == "sp3"]
        self.assertEqual({r["chosen"] for r in rings}, {"entity_tables"}, "the route that ran, from the project's plan, not the draft's earlier one")
        self.assertEqual(rings[0]["outcome"]["leaf"], "built")
        self.assertIsNotNone(rings[0]["outcome"]["holdout"], "this leaf's own holdout number")
        self.assertEqual({r["outcome"]["holdout"] for r in route_rows if r["outcome"] and r["sub_problem"] in ("sp1", "sp2", "sp3")}.__len__() >= 2, True)
        self.assertTrue(any(c["blocked_by"] for r in route_rows for c in r["candidates"]), "the candidates a route refuted are in the example, with the code that refuted them")
        manifest = builders.DatasetManifest.read((Path(maps["folder"]) / "manifest.json").read_text())
        self.assertEqual({s.kind for s in manifest.sources}, {"draft", "project"})
        consent.set_workspace_consent(self.home, "train", False)
        try:
            again = {(b["builder"], b["name"]) for b in datasets.build(self.s)["built"]}
            self.assertNotIn(("problem_maps", "brief_to_map"), again, "without train these sets are not built")
        finally:
            consent.set_workspace_consent(self.home, "train", True)
        proj = self.s.projects.get(project)
        proj["settings"] = {**proj["settings"], "consent": {"train": False, "improve": True}}
        self.s.projects.save(proj)
        try:
            out = {(b["builder"], b["name"]): b for b in datasets.build(self.s)["built"]}
            self.assertEqual(out["problem_maps", "brief_to_map"]["rows"], 0, "a project that withheld train is in neither set")
            self.assertEqual(out["routes", "sub_problem_to_approach"]["rows"], 0)
        finally:
            proj["settings"]["consent"] = {"train": True, "improve": True}
            self.s.projects.save(proj)

    def test_the_turn_carries_the_map_and_the_programme(self):
        rows, manifest = builders.episodes([e.model_dump(mode="json") for e in self.events], [], text=True, consent="train")
        parts = [p for r in rows for p in r["problem_parts"]]
        self.assertTrue(any(p["kind"] == "problem.mapped" for p in parts))
        rows_improve, _ = builders.episodes([e.model_dump(mode="json") for e in self.events], [], text=False, consent="improve")
        text = json.dumps([r["problem_parts"] for r in rows_improve])
        self.assertNotIn("reason", text, "under improve alone: codes and counts, no free text")


if __name__ == "__main__":
    unittest.main()

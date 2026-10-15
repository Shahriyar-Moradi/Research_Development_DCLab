"""Package 24.2 on Home: the problem map of a draft is drawn from the person's words, shown with the reviewer's questions and the
reason for each part, edited by the person (each edit through the map validator, refused with its reasons, recorded as a graph move in
the session record and as a decision in the ledger), and confirmed. A problem of several parts is not built until the map is confirmed;
one problem is not held by it. The coordinator draws the map, asks for the confirmation and goes on."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import mapping, problem_map as pm
    from dclab_core.agents import ScriptedAdvisor
    from dclab_core.replicas import delivery_fraud as dfr
    from dclab_rnd import experience, ledger
    from dclab_rnd.draft import map as home_map
    from dclab_rnd.draft.api import build_blockers
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.coordinator import Coordinator
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.studio import ProjectStore, team
    from tests.test_morning_guard import upload
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FRAUD = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review 200 a day.")
CHURN = "I want to know which subscribers will cancel at their renewal."


@unittest.skipIf(MISSING, MISSING)
class HomeMapTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home / "drafts")
        self.log = experience.FileLog(self.home / "experience", "wtest")
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        self.log.home = self.book.home = self.home
        experience.install(self.log)
        ledger.install(self.book)
        self.addCleanup(experience.install, None)
        self.addCleanup(ledger.install, None)
        self.draft = self.store.create(FRAUD)["id"]

    def test_looking_at_the_map_writes_nothing_and_drawing_keeps_it(self):
        shown = home_map.view(self.store.get(self.draft))
        self.assertEqual((shown["drawn"], shown["map"]["size"], len(shown["map"]["parts"])), (False, "complex", 4))
        self.assertNotIn("problem_map", self.store.get(self.draft))
        drawn = home_map.draw(self.store, self.draft)
        self.assertTrue(drawn["drawn"] and not drawn["confirmed"] and drawn["needs_confirmation"])
        again = home_map.draw(self.store, self.draft)
        self.assertEqual(again["map"]["fingerprint"], drawn["map"]["fingerprint"])  # drawn once; new words do not undo what was kept
        self.assertEqual(len(self.store.get(self.draft)["problem_map"]["history"]), 1)

    def test_every_part_shows_the_reason_it_is_there(self):
        shown = home_map.view(self.store.get(self.draft))
        for part in shown["map"]["parts"]:
            self.assertTrue(part["quote"] or part["why"], part["title"])
        reasons = [p["why"] for p in shown["map"]["parts"]]
        self.assertEqual(len(set(reasons)), 4)  # each part says why it is there, in its own words
        self.assertIn("screen needs no labels", reasons[0])
        self.assertTrue(shown["review"]["questions"])

    def test_one_problem_needs_no_confirmation(self):
        draft = self.store.create(CHURN)["id"]
        shown = home_map.draw(self.store, draft)
        self.assertEqual((shown["map"]["size"], shown["confirmed"], shown["needs_confirmation"]), ("single", True, False))
        self.assertIsNone(home_map.blocker(self.store.get(draft)))
        self.assertEqual(home_map.facts(self.store.get(draft)), {"map_size": "single", "map_parts": 1, "map_blockers": [], "map_confirmed": True})

    def test_a_person_removes_a_part_and_it_is_a_graph_move_and_a_decision(self):
        home_map.draw(self.store, self.draft)
        shown = home_map.edit(self.store, self.draft, "remove", part="part2")
        self.assertEqual(len(shown["map"]["parts"]), 3)
        self.assertEqual((shown["map"]["source"], shown["source"]), ("person", "person"))  # the map is the person's now; the first draw is in the history
        self.assertEqual(shown["history"][0]["by"], "rule")
        moves = [e for e in self.log.read() if e.kind == "graph.move"]
        self.assertEqual([(e.payload["move"], e.payload["allowed"]) for e in moves], [("map:remove", True)])
        self.assertNotEqual(moves[0].payload["from"], moves[0].payload["to"])
        edits = [r for r in self.book.records() if r.point == "map.edit"]
        self.assertEqual([(r.value_used, r.used, r.state.get("parts_before"), r.state.get("parts_after")) for r in edits], [("remove", "person", 4, 3)])
        drawn = [r for r in self.book.records() if r.point == "map.size"]
        self.assertEqual([(r.value_used, r.used) for r in drawn], [("complex", "rule")])

    def test_an_edit_that_would_make_the_map_unsound_is_refused_with_its_reasons_and_changes_nothing(self):
        before = home_map.draw(self.store, self.draft)
        with self.assertRaises(mapping.MapEditRefused) as caught:
            home_map.edit(self.store, self.draft, "merge", keep="part1", other="part3")
        self.assertIn("one part, one unit", str(caught.exception))
        after = home_map.view(self.store.get(self.draft))
        self.assertEqual(after["map"]["fingerprint"], before["map"]["fingerprint"])
        refused = [e for e in self.log.read() if e.kind == "graph.move"]
        self.assertEqual([(e.payload["move"], e.payload["allowed"]) for e in refused], [("map:merge", False)])
        self.assertTrue(refused[0].payload["failed_checks"])
        self.assertEqual([r for r in self.book.records() if r.point == "map.edit"], [])  # a refused edit is not a decision of the person

    def test_adding_a_part_and_confirming(self):
        home_map.draw(self.store, self.draft)
        added = home_map.edit(self.store, self.draft, "add", title="Check the couriers' hours", family="anomaly", unit="ride",
                              moment=home_map.view(self.store.get(self.draft))["map"]["parts"][0]["moment"], labels="none")
        self.assertEqual(len(added["map"]["parts"]), 5)
        confirmed = home_map.edit(self.store, self.draft, "confirm")
        self.assertTrue(confirmed["confirmed"] and not confirmed["needs_confirmation"])
        self.assertEqual([h["action"] for h in confirmed["history"]], ["draw", "add", "confirm"])
        self.assertIsNone(home_map.blocker(self.store.get(self.draft)))

    def test_an_edit_after_confirmation_asks_for_it_again(self):
        home_map.draw(self.store, self.draft)
        home_map.edit(self.store, self.draft, "confirm")
        shown = home_map.edit(self.store, self.draft, "remove", part="part2")
        self.assertFalse(shown["confirmed"])
        self.assertIsNotNone(home_map.blocker(self.store.get(self.draft)))

    def test_a_map_that_is_not_sound_cannot_be_confirmed(self):
        home_map.draw(self.store, self.draft)

        def spoil(d):
            m = pm.ProblemMap.read(d["problem_map"]["map"])
            bad = m.model_copy(update={"sub_problems": [s.model_copy(update={"reads": [pm.Read(sub_problem="sp2", what="outcome")]}) if s.id == "sp1" else s for s in m.sub_problems]})
            bad = pm.ProblemMap.model_validate({**bad.model_dump(mode="json"), "fingerprint": ""})
            d["problem_map"]["map"] = json.loads(bad.model_dump_json())
        self.store.update(self.draft, spoil)
        with self.assertRaises(mapping.MapEditRefused) as caught:
            home_map.edit(self.store, self.draft, "confirm")
        self.assertIn("leak", str(caught.exception))
        self.assertTrue(home_map.facts(self.store.get(self.draft))["map_blockers"])

    def test_drawing_again_drops_the_edits_and_the_confirmation(self):
        home_map.draw(self.store, self.draft)
        home_map.edit(self.store, self.draft, "remove", part="part2")
        home_map.edit(self.store, self.draft, "confirm")
        shown = home_map.edit(self.store, self.draft, "redraw")
        self.assertEqual((len(shown["map"]["parts"]), shown["confirmed"]), (4, False))

    def test_words_that_change_nothing_the_rules_read_leave_the_map_as_it_is(self):
        home_map.draw(self.store, self.draft)

        def talk(d):
            d.setdefault("messages", []).append({"role": "user", "text": "The labels are in a file we will upload tomorrow."})  # an answer about the data
        self.store.update(self.draft, talk)
        self.assertFalse(home_map.view(self.store.get(self.draft))["stale"])  # a walk that goes on asking never redraws the map under itself
        self.assertEqual(home_map.facts(self.store.get(self.draft))["map_size"], "complex")

    def test_a_map_whose_words_changed_is_marked_stale_until_a_person_touched_it(self):
        draft = self.store.create("I want to build anomaly detection")["id"]
        home_map.draw(self.store, draft)

        def talk(d):
            d.setdefault("messages", []).append({"role": "user", "text": "Find the fake rides and fake orders our bikers make. A few cases were confirmed."})
        self.store.update(draft, talk)
        self.assertTrue(home_map.view(self.store.get(draft))["stale"])
        home_map.edit(self.store, draft, "add", title="Check the couriers' hours", family="anomaly", unit="ride", labels="none")
        self.assertFalse(home_map.view(self.store.get(draft))["stale"])  # the person's map is theirs

    def test_a_stale_map_that_nothing_of_the_person_is_in_never_holds_or_lets_through_a_build(self):
        single = self.store.create("I want to build anomaly detection")["id"]
        home_map.draw(self.store, single)
        self.assertEqual(home_map.facts(self.store.get(single))["map_size"], "single")
        self.assertIsNone(home_map.blocker(self.store.get(single)))

        def talk(d):
            d.setdefault("messages", []).append({"role": "user", "text": "Find the fake rides and fake orders our bikers make. We have a few confirmed cases and the team can review 200 a day."})
        self.store.update(single, talk)
        draft = self.store.get(single)
        self.assertTrue(home_map.stale(draft))
        self.assertIsNone(home_map.facts(draft)["map_size"])  # drawn again by the coordinator
        self.assertIn("4 parts", home_map.blocker(draft) or "")  # never built from the single map drawn from fewer words
        shown = home_map.draw(self.store, single)  # a stale map is drawn again without being asked
        self.assertEqual((shown["map"]["size"], shown["stale"]), ("complex", False))
        self.assertEqual([h["action"] for h in shown["history"]], ["draw", "draw"])

    def test_a_map_the_person_confirmed_or_edited_is_never_drawn_again_by_new_words(self):
        home_map.draw(self.store, self.draft)
        home_map.edit(self.store, self.draft, "confirm")

        def talk(d):
            d.setdefault("messages", []).append({"role": "user", "text": "I also care about refunds."})
        self.store.update(self.draft, talk)
        self.assertFalse(home_map.stale(self.store.get(self.draft)))
        self.assertTrue(home_map.facts(self.store.get(self.draft))["map_confirmed"])

    def test_drawing_again_is_a_graph_move_and_a_decision_of_the_person(self):
        home_map.draw(self.store, self.draft)
        home_map.edit(self.store, self.draft, "remove", part="part2")
        home_map.edit(self.store, self.draft, "redraw")
        moves = [(e.payload["move"], e.payload["allowed"]) for e in self.log.read() if e.kind == "graph.move"]
        self.assertEqual(moves, [("map:remove", True), ("map:redraw", True)])
        edits = [(r.value_used, r.state.get("parts_before"), r.state.get("parts_after")) for r in self.book.records() if r.point == "map.edit"]
        self.assertEqual(edits, [("remove", 4, 3), ("redraw", 3, 4)])
        self.assertEqual(home_map.view(self.store.get(self.draft))["history"][-1]["by"], "person")

    def test_two_edits_at_once_are_both_kept_and_a_confirm_never_brings_a_removed_part_back(self):
        import threading

        home_map.draw(self.store, self.draft)
        errors = []

        def run(part):
            try:
                home_map.edit(self.store, self.draft, "remove", part=part)
            except Exception as error:  # noqa: BLE001
                errors.append(error)
        threads = [threading.Thread(target=run, args=(p,)) for p in ("part2", "part3")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([e for e in errors if not isinstance(e, mapping.MapEditRefused)], [])
        kept = [p["id"] for p in home_map.view(self.store.get(self.draft))["map"]["parts"]]
        self.assertTrue({"part2", "part3"}.isdisjoint(kept) or errors, kept)  # both gone, or the second refused with its reason: never one lost silently
        home_map.edit(self.store, self.draft, "confirm")
        self.assertEqual([p["id"] for p in home_map.view(self.store.get(self.draft))["map"]["parts"]], kept)

    def test_a_removed_id_is_not_used_again_by_a_part_added_later(self):
        home_map.draw(self.store, self.draft)
        home_map.edit(self.store, self.draft, "remove", part="part4")
        shown = home_map.edit(self.store, self.draft, "add", title="A new check", family="anomaly", unit="ride", labels="none")
        self.assertEqual([p["id"] for p in shown["map"]["parts"]], ["part1", "part2", "part3", "part5"])
        self.assertIn("sp5", [s["id"] for s in shown["map"]["sub_problems"]])

    def test_the_reviewers_model_questions_are_added_after_an_edit_and_kept_with_the_map(self):
        advisor = ScriptedAdvisor({"decomposer": None, "map_reviewer": {"questions": [{"kind": "ambiguity", "target": "part3", "question": "Are the linked accounts the bikers?",
                                                                                      "quote": "our bikers and customers"}]}})
        home_map.draw(self.store, self.draft, advisor=advisor)
        shown = home_map.view(self.store.get(self.draft))
        self.assertIn("Are the linked accounts the bikers?", [q["question"] for q in shown["review"]["questions"]])
        again = home_map.edit(self.store, self.draft, "remove", part="part4", advisor=advisor)  # a new map: asked again about it
        self.assertEqual(again["review"]["source"], "advisor")

    def test_an_unknown_edit_is_refused(self):
        with self.assertRaises(mapping.MapEditRefused):
            home_map.edit(self.store, self.draft, "explode")

    def test_the_roles_serve_only_when_switched_on_and_measured(self):
        class Gate:
            def available(self, purpose):
                return True

        self.assertFalse(home_map.roles_passed())
        self.assertIsNone(home_map.advisor_for(Gate()))
        self.assertIsNone(home_map.advisor_for(None, force=True))
        self.assertIsNotNone(home_map.advisor_for(Gate(), force=True))  # a measurement asks for them

    def test_a_decomposer_that_serves_is_drawn_into_the_map_and_the_decision_names_it(self):
        reply = {"parts": [{"title": "Unusual rides", "quote": "Find the fake rides and fake orders"}, {"title": "The review list", "quote": "the team can review 200 a day"}],
                 "sub_problems": [{"part": 1, "family": "anomaly", "labels": "none", "predict_rank": 1, "quote": "Find the fake rides and fake orders"},
                                  {"part": 2, "family": "ranking", "predict_rank": 2, "known_rank": 3, "reads": [{"sub_problem": 1, "what": "score"}], "quote": "the team can review 200 a day"}],
                 "composition": {"method": "sequence", "budget": "the team can review 200 a day", "steps": [{"kind": "sequence", "inputs": ["sp1", "sp2"], "output": "list"}]}}
        shown = home_map.draw(self.store, self.draft, advisor=ScriptedAdvisor({"decomposer": reply, "map_reviewer": {"questions": []}}))
        self.assertEqual((shown["source"], len(shown["map"]["parts"])), ("advisor", 2))
        drawn = [r for r in self.book.records() if r.point == "map.size"]
        self.assertEqual([(a.by, a.answerer) for a in drawn[0].answers], [("rule", "rule:decompose"), ("model_role", "decomposer")])


@unittest.skipIf(MISSING, MISSING)
class BuildGateTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home / "drafts")
        self.agent = HomeAgent(self.store, None)
        self.draft = self.store.create(FRAUD)["id"]
        self.agent.start(self.draft)
        upload(self.store, self.agent, self.draft, "rides.csv", dfr.generate(seed=0, n_orders=3000, n_bikers=120).rides)

    def rules(self):
        return [e.rule for e in build_blockers(self.store, self.draft)]

    def test_a_problem_of_several_parts_is_not_built_until_the_map_is_confirmed(self):
        self.assertIn("problem map not confirmed", self.rules())
        home_map.draw(self.store, self.draft)
        self.assertIn("problem map not confirmed", self.rules())
        home_map.edit(self.store, self.draft, "confirm")
        self.assertNotIn("problem map not confirmed", self.rules())

    def test_one_problem_is_not_held_by_it(self):
        other = self.store.create(CHURN)["id"]
        self.assertNotIn("problem map not confirmed", [e.rule for e in build_blockers(self.store, other)])

    def test_the_message_says_what_to_do(self):
        text = next(str(e) for e in build_blockers(self.store, self.draft) if e.rule == "problem map not confirmed")
        self.assertIn("4 parts", text)
        self.assertIn("Confirm the map", text)


@unittest.skipIf(MISSING, MISSING)
class RoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def draft(self, problem=FRAUD):
        return self.client.post("/api/drafts", json={"problem": problem}, headers=self.h).json()["id"]

    def test_the_map_is_shown_drawn_edited_refused_and_confirmed_through_the_routes(self):
        did = self.draft()
        shown = self.client.get(f"/api/drafts/{did}/map").json()
        self.assertEqual((shown["drawn"], len(shown["map"]["parts"])), (False, 4))
        drawn = self.client.post(f"/api/drafts/{did}/map", json={}, headers=self.h).json()
        self.assertTrue(drawn["drawn"])
        removed = self.client.post(f"/api/drafts/{did}/map/edit", json={"action": "remove", "part": "part2"}, headers=self.h)
        self.assertEqual((removed.status_code, len(removed.json()["map"]["parts"])), (200, 3))
        refused = self.client.post(f"/api/drafts/{did}/map/edit", json={"action": "merge", "keep": "part1", "other": "part3"}, headers=self.h)
        self.assertEqual(refused.status_code, 409)
        self.assertIn("one part, one unit", refused.json()["detail"])
        self.assertEqual(self.client.post(f"/api/drafts/{did}/map/edit", json={"action": "explode"}, headers=self.h).status_code, 409)
        confirmed = self.client.post(f"/api/drafts/{did}/map/edit", json={"action": "confirm"}, headers=self.h).json()
        self.assertTrue(confirmed["confirmed"])
        self.assertEqual(self.client.get(f"/api/drafts/{did}/map").json()["history"][-1]["action"], "confirm")

    def test_the_body_of_an_edit_is_whitelisted_and_the_person_cannot_be_named_by_the_client(self):
        did = self.draft()
        self.client.post(f"/api/drafts/{did}/map", json={}, headers=self.h)
        spoof = {"action": "confirm", "who": "owner", "draft_id": "x", "advisor": "x", "drafts": "x"}
        r = self.client.post(f"/api/drafts/{did}/map/edit", json=spoof, headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["history"][-1]["by"], "person")
        self.assertEqual(self.client.post(f"/api/drafts/{did}/map/edit", json={"action": ["confirm"]}, headers=self.h).status_code, 409)
        too_long = self.client.post(f"/api/drafts/{did}/map/edit", json={"action": "add", "title": "x", "family": "anomaly", "unit": "u" * 300}, headers=self.h)
        self.assertEqual(too_long.status_code, 409)
        self.assertIn("at most 120", too_long.json()["detail"])
        self.assertEqual(self.client.post(f"/api/drafts/{did}/map/edit", json={"action": "add", "title": "x", "family": "anomaly", "unit": 5}, headers=self.h).status_code, 409)

    def test_only_a_json_true_draws_the_map_again(self):
        did = self.draft()
        self.client.post(f"/api/drafts/{did}/map", json={}, headers=self.h)
        self.client.post(f"/api/drafts/{did}/map/edit", json={"action": "remove", "part": "part2"}, headers=self.h)
        for flag in ("false", 1, "true", 0, None):
            kept = self.client.post(f"/api/drafts/{did}/map", json={"redraw": flag}, headers=self.h).json()
            self.assertEqual(len(kept["map"]["parts"]), 3, flag)  # the person's edit stands
        again = self.client.post(f"/api/drafts/{did}/map", json={"redraw": True}, headers=self.h).json()
        self.assertEqual(len(again["map"]["parts"]), 4)

    def test_an_unknown_draft_is_not_found(self):
        self.assertEqual(self.client.get("/api/drafts/nope/map").status_code, 404)
        self.assertEqual(self.client.post("/api/drafts/nope/map/edit", json={"action": "confirm"}, headers=self.h).status_code, 404)

    def test_the_routes_are_checked_like_every_other_route(self):
        from dclab_rnd.accounts.guard import permission

        self.assertEqual(permission("GET", "/api/drafts/x/map"), permission("GET", "/api/drafts/x/card"))
        self.assertEqual(permission("POST", "/api/drafts/x/map/edit"), permission("POST", "/api/drafts/x/pack"))


def person(approving=True):
    asked = []
    p = team.Person(answer=lambda field, question: None, approve=lambda gate, summary: asked.append((gate, summary)) or approving, who="scripted person")
    p.approvals = asked
    return p


def known(draft):
    """What the person has said, so the problem card is ready: the moment, the outcome column and the action."""
    draft.setdefault("understanding", {}).update({"prediction_moment": "After the ride, in the daily audit of the day's rides", "target": "status",
                                                   "action": "Send the top cases to the team", "labels": "A sample of the cases we found",
                                                   "family": "Find the fraudulent rides", "entity": "One biker"})


@unittest.skipIf(MISSING, MISSING)
class CoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store, self.projects = DraftStore(self.home / "drafts"), ProjectStore(self.home / "projects")
        self.agent = HomeAgent(self.store, None)
        self.log = experience.FileLog(self.home / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)
        self.draft = self.store.create(FRAUD)["id"]
        self.agent.start(self.draft)
        upload(self.store, self.agent, self.draft, "rides.csv", dfr.generate(seed=0, n_orders=3000, n_bikers=120).rides)
        self.store.update(self.draft, known)

    def test_it_draws_the_map_and_waits_for_the_person_to_confirm_it(self):
        who = person(approving=False)
        out = Coordinator(self.store, self.projects, self.agent).run(self.draft, who)
        actions = [s["action"] for s in out["steps"]]
        self.assertEqual(actions, ["draw_map", "confirm_map"])
        self.assertEqual((out["state"], out["next"]), ("map", "stop"))
        self.assertIn("Waiting for the person to confirm the map", out["says"])
        self.assertEqual(who.approvals[0][0], "map")  # asked at the map gate
        self.assertEqual(len(who.approvals[0][1]["parts"]), 4)
        self.assertIsNone(out["project_id"])
        self.assertFalse(self.store.get(self.draft)["problem_map"]["confirmed"])

    def test_confirmed_by_the_person_it_goes_on_to_the_data(self):
        out = Coordinator(self.store, self.projects, self.agent).run(self.draft, person(approving=True))
        actions = [s["action"] for s in out["steps"]]
        self.assertLess(actions.index("draw_map"), actions.index("confirm_map"))
        self.assertTrue(self.store.get(self.draft)["problem_map"]["confirmed"])
        self.assertNotEqual(out["state"], "map")  # past it: the next stop, if any, is about the data or the card
        self.assertIn("map", out["passed"])
        moves = [e for e in self.log.read() if e.kind == "graph.move"]
        self.assertIn(("map:confirm", True), [(e.payload["move"], e.payload["allowed"]) for e in moves])

    def test_a_walk_does_not_draw_again_what_the_person_confirmed(self):
        Coordinator(self.store, self.projects, self.agent).run(self.draft, person(approving=True))
        again = Coordinator(self.store, self.projects, self.agent).run(self.draft, person(approving=True))
        self.assertNotIn("draw_map", [s["action"] for s in again["steps"]])
        self.assertNotIn("confirm_map", [s["action"] for s in again["steps"]])

    def test_one_problem_draws_its_map_and_is_not_asked_to_confirm(self):
        draft = self.store.create(CHURN)["id"]
        self.agent.start(draft)
        self.store.update(draft, known)  # the same answers: none of them is about fraud, and the map reads only the words
        who = person()
        out = Coordinator(self.store, self.projects, self.agent).run(draft, who)
        self.assertEqual(who.approvals, [])
        self.assertIn("draw_map", [s["action"] for s in out["steps"]])
        self.assertEqual(self.store.get(draft)["problem_map"]["map"]["size"], "single")


if __name__ == "__main__":
    unittest.main()

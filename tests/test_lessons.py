"""31.5: a lesson with the kind of gap it shows, where it goes, and who must confirm it."""
import json
import tempfile
import unittest
from pathlib import Path

from dclab_core import lesson as Ls
from dclab_core.challenge_pack import CODES

PACK, RUN, AT = "pk-aaaaaaaaaaaa", "learning-111111111111", "2026-10-29T01:00:00+00:00"


def one(gap="rule", status="proposed", confirmed_by=None):
    return Ls.Lesson(id="L-0123456789", family="groups", gap_type=gap, proposed_by="rule:gaps_v1", observed="a column leaked",
                     expected="kept out", lesson="Keep it out.", confidence=0.5, at=AT, status=status, confirmed_by=confirmed_by,
                     evidence=[Ls.Evidence(pack=PACK, trace=RUN, grader="grader_v1", code="LEAK_KEPT:after_outcome")])


class LessonTests(unittest.TestCase):
    def test_every_grader_code_has_the_rules_proposal_and_every_gap_a_route(self):
        self.assertEqual(set(Ls.RULES), set(CODES))
        self.assertEqual(set(Ls.ROUTE), set(Ls.GAPS))
        from dclab_core.ledger import POINTS

        for code, (gap, point, *_rest) in Ls.RULES.items():
            self.assertIn(gap, Ls.GAPS)
            self.assertTrue(point is None or point in POINTS, f"{code}: {point} is not a ledger decision point")

    def test_a_rule_or_capability_gap_acts_only_once_a_person_confirmed_it(self):
        self.assertFalse(one("rule").actionable)
        with self.assertRaises(ValueError):
            one("rule", "confirmed", "model:claude")  # a model never confirms code
        self.assertTrue(one("rule", "confirmed", "person:owner").actionable)
        self.assertTrue(one("decision").actionable)  # a labelled sample needs no one's yes to be proposed
        self.assertFalse(one("data").actionable)  # the right answer is "does not fit": nothing to act on
        self.assertFalse(one("decision", "refused", "person:owner").actionable)
        with self.assertRaises(ValueError):
            one("decision", "refused")  # who decided is said
        self.assertEqual(one("capability").route, ("recipe_part", 3))

    def test_the_first_wrong_step_decides_the_lesson_and_no_column_is_named(self):
        from dclab_core import variants as V

        from dclab_rnd.learning.packs import materialize, read_key
        from dclab_rnd.learning.reflect import lessons_for

        work = Path(tempfile.mkdtemp())
        variant = materialize(work, "groups", V.seeds("groups", "learning", 1)[0])
        key = read_key(work, "groups", variant.pack.id)
        codes = ["LEAK_KEPT:after_outcome", "SPLIT_WRONG:repeated_entity", "NO_SIGNAL:not_above_baseline"]
        made = lessons_for("groups", variant.pack.id, RUN, codes, {"SPLIT_WRONG:repeated_entity": "B-0123456789"}, AT)
        self.assertEqual([(x.evidence[0].code, x.gap_type, x.decision_point, x.bullet) for x in made],
                         [("SPLIT_WRONG:repeated_entity", "rule", "split.strategy", "B-0123456789")])  # the solution step comes first
        self.assertFalse(made[0].actionable)  # DCLab's rules made the attempt: a missing check, which a person confirms
        text = json.dumps([x.model_dump(mode="json") for x in made])
        for column in filter(None, [*key.leaks, key.row_key, key.group_column, *key.decoys, *key.known_before]):
            self.assertNotIn(f"{column}", text)
        self.assertEqual(made[0].proposed_by, "rule:gaps_v1")

    def test_past_drafts_get_their_lessons_once_and_the_loop_record_names_them(self):
        from dclab_rnd.learning.drafting import backfill_lessons, lessons_path
        from dclab_rnd.learning.loop_records import build

        home = Path(tempfile.mkdtemp())
        (home / "learning").mkdir(parents=True)
        from dclab_core.playbook import Bullet, Evidence, bullet_id

        from dclab_rnd.learning.playbooks import FilePlaybooks

        text = "Split by the repeated entity, so no entity sits on both sides."
        bid = bullet_id(text, "domain:groups")
        FilePlaybooks(home).add("groups", Bullet(id=bid, text=text, scope="domain:groups", codes=["SPLIT_WRONG:repeated_entity"], by="rule:reflect_v1",
                                                 evidence=[Evidence(pack=PACK, run=RUN, step="solution", code="SPLIT_WRONG:repeated_entity", grader="grader_v1")]),
                                reason="drawn from a graded attempt")
        (home / "learning" / "drafts.jsonl").write_text(json.dumps({"family": "groups", "pack": PACK, "run": RUN, "codes": ["SPLIT_WRONG:repeated_entity"],
                                                                   "step": "solution", "filed": [bid], "refused": [], "why": "", "at": AT}) + "\n")
        self.assertEqual(backfill_lessons(home), 1)
        self.assertEqual(backfill_lessons(home), 0)  # appended once
        kept = [json.loads(line) for line in lessons_path(home).read_text().splitlines()]
        self.assertEqual((kept[0]["gap_type"], kept[0]["bullet"], kept[0]["proposed_by"]), ("rule", bid, "rule:gaps_v1"))
        records, _ = build(home, home / "no-results")
        draft = next(r for r in records if r.attempt.kind == "draft")
        evidence = kept[0]["evidence"][0]
        self.assertEqual((evidence["pack"], evidence["trace"]), (draft.pack.id, draft.attempt.run))  # the lesson joins its attempt's record

    def test_who_made_the_attempt_and_whether_it_was_solvable_decide_the_gap(self):
        from dclab_rnd.learning.reflect import lessons_for

        codes = ["SPLIT_WRONG:repeated_entity"]
        by_model = lessons_for("groups", PACK, RUN, codes, {}, AT, producer="model:qwen3.5:4b-16k")
        self.assertEqual((by_model[0].gap_type, by_model[0].actionable), ("decision", True))  # a model chose the split: its student learns
        by_rules = lessons_for("groups", PACK, RUN, codes, {}, AT, producer="rule:dclab")
        self.assertEqual((by_rules[0].gap_type, by_rules[0].actionable), ("rule", False))
        unsolvable = lessons_for("groups", PACK, RUN, ["NO_SIGNAL:not_above_baseline"], {}, AT, solvable=False)
        self.assertEqual((unsolvable[0].gap_type, unsolvable[0].actionable), ("data", False))
        from dclab_core.challenge_pack import GRADER

        self.assertEqual(by_model[0].evidence[0].grader, GRADER)

    def test_an_attempt_that_delivered_nothing_writes_no_lesson(self):
        from unittest import mock

        from dclab_rnd.learning import drafting

        home = Path(tempfile.mkdtemp())
        every = ["LEAK_KEPT:after_outcome", "SPLIT_WRONG:repeated_entity", "IDENTIFIER_USED:row_key", "NO_SIGNAL:not_above_baseline"]
        crashed = {"pack": PACK, "run": RUN, "solvable": True, "kept": None,
                   "arms": {"rules": {"verdict": "fail", "codes": every, "holdout": None, "checks": []}, "reference": {"verdict": "pass"}}}
        with mock.patch.object(drafting, "attempt", lambda *a, **k: crashed), mock.patch.object(drafting, "read_key", lambda *a: None), \
                mock.patch.object(drafting, "read_public", lambda *a: (type("P", (), {"columns": []})(), None)), \
                mock.patch.object(drafting, "public_dir", lambda *a: home), \
                mock.patch.object(drafting.V, "seeds", lambda *a: [0]):
            drafting.draft_lessons(home, "groups", 1)
        self.assertFalse(drafting.lessons_path(home).exists() and drafting.lessons_path(home).read_text().strip())
        (home / "learning" / "drafts.jsonl").write_text(json.dumps({"family": "groups", "pack": PACK, "run": RUN, "codes": every, "step": None,
                                                                   "filed": [], "at": AT}) + "\n")
        self.assertEqual(drafting.backfill_lessons(home), 0)  # nor from its note


if __name__ == "__main__":
    unittest.main()

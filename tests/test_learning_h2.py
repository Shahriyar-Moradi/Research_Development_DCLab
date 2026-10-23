"""H2 of the harness that learns: the playbook, the reflector, the name filter, case memory, what a solver is shown.

A playbook is rebuilt from append-only deltas whose rules are code: a duplicate merges, counters only rise, a bullet
harmful three more times than helpful is retired, a retired bullet stays retired, text never changes. The reflector
finds the first wrong step, drafts general lessons from codes alone (the rules by default, a model screened the same
way and capped), and teaches nothing from a passed attempt or one that delivered nothing. The screen refuses a lesson
naming a column, carrying a score, citing no step or answering no code. Case memory keeps no name. The intern shows
accepted lessons in its first user message, under budget, never in the system prompt.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)

AT = "2026-10-23T12:00:00+00:00"


def key(**over):
    from dclab_core.challenge_pack import ChallengeKey

    base = dict(pack="pk-000000000001", family="groups", seed=3, partition="learning", leaks=["refund_settled"], group_column="household",
                row_key="order_id", decoys=["prior_refund_count"], known_before=["basket_value", "loyalty_tier"], knobs={})
    return ChallengeKey(**{**base, **over})


def bullet(text="Split and fold by a column whose values repeat and stand for one thing with lasting traits.", scope="domain:groups",
           code="SPLIT_WRONG:repeated_entity"):
    from dclab_core.playbook import Bullet, Evidence, bullet_id

    return Bullet(id=bullet_id(text, scope), text=text, scope=scope, codes=[code],
                  evidence=[Evidence(pack="pk-1", step="solution", code=code, grader="grader_v1")], by="rule:test")


def delta(op, **kw):
    from dclab_core.playbook import Delta

    return Delta(family="groups", lineage="main", op=op, by=kw.pop("by", "test"), reason=kw.pop("reason", "a test"), at=AT, **kw)


class PlaybookTests(unittest.TestCase):
    def test_deltas_merge_duplicates_count_up_retire_on_harm_and_never_change_text(self):
        from dclab_core.playbook import Refused, apply, build

        b = bullet()
        same = bullet(text="Split and fold by a column whose values repeat, and stand for one thing with lasting traits!")
        book, refused = build("groups", "main", [delta("add", bullet=b), delta("add", bullet=same)])
        self.assertEqual((len(book.bullets), refused), (1, []))  # merged by its normalised words, not added twice
        self.assertEqual(len(book.bullets[0].evidence), 1)
        book = apply(book, delta("count", target=b.id, counts={"shown": 2, "helpful": 1}))
        self.assertEqual((book.bullets[0].shown, book.bullets[0].helpful), (2, 1))
        with self.assertRaises(Refused):
            apply(book, delta("count", target=b.id, counts={"helpful": -1}))  # counters only rise
        with self.assertRaises(Refused):
            apply(book, delta("count", target=b.id, counts={"text": 1}))
        book = apply(book, delta("count", target=b.id, counts={"harmful": 4}))  # harmful 4 - helpful 1 = 3: quarantined
        self.assertEqual(book.bullets[0].status, "retired")
        with self.assertRaises(Refused):
            apply(book, delta("status", target=b.id, status="accepted"))  # a retired bullet stays retired
        self.assertFalse(hasattr(apply, "edit"))
        with self.assertRaises(Refused):
            apply(book, delta("add", bullet=bullet(scope="domain:matching")))  # another family's lesson
        self.assertEqual(book.version, 4)

    def test_an_id_names_one_text_and_a_status_moves_one_step_forward(self):
        from pydantic import ValidationError

        from dclab_core.playbook import Bullet, Refused, apply, build

        b = bullet()
        with self.assertRaises(ValidationError):  # a forged id cannot replace another bullet's text and counters
            Bullet(**{**b.model_dump(), "text": "A different lesson entirely, about something else."})
        book, _ = build("groups", "main", [delta("add", bullet=b)])
        with self.assertRaises(Refused):
            apply(book, delta("status", target=b.id, status="accepted"))  # never skips the probe
        book = apply(book, delta("status", target=b.id, status="probed"))
        book = apply(book, delta("status", target=b.id, status="accepted"))
        with self.assertRaises(Refused):
            apply(book, delta("status", target=b.id, status="draft"))  # never back

    def test_render_shows_accepted_bullets_best_first_within_the_budget(self):
        from dclab_core.playbook import apply, build, render, tokens

        a = bullet()
        b = bullet(text="A column different on every row identifies the row, not the case: keep it out of the inputs.", scope="invariant",
                   code="IDENTIFIER_USED:row_key")
        c = bullet(text="Check the model beats the simplest baseline on the holdout before reporting it.", scope="invariant",
                   code="NO_SIGNAL:not_above_baseline")
        book, _ = build("groups", "main", [delta("add", bullet=x) for x in (a, b, c)])
        self.assertEqual(render(book), ("", []))  # drafts are never shown
        for x in (a, b):
            book = apply(book, delta("status", target=x.id, status="probed"))
            book = apply(book, delta("status", target=x.id, status="accepted"))
        text, ids = render(book)
        self.assertEqual(ids, [b.id, a.id])  # invariant first
        self.assertIn(f"[{a.id}]", text)
        small, small_ids = render(book, budget=tokens(f"- [{b.id}] {b.text}"))
        self.assertEqual(small_ids, [b.id])


class FilterTests(unittest.TestCase):
    def screen(self, **draft):
        from dclab_rnd.learning.filter import screen

        base = {"text": "Group the folds by the column that repeats across rows for one lasting thing.", "scope": "domain",
                "codes": ["SPLIT_WRONG:repeated_entity"], "step": "solution"}
        return screen({**base, **draft}, key(), ["household", "basket_value", "refund_settled"], ["solution", "leakage", "models", "final"],
                      "pk-1", "run-1", "model:test")

    def test_a_general_lesson_passes_and_every_leak_or_gap_is_refused(self):
        ok, why = self.screen()
        self.assertEqual((ok.scope, ok.codes, ok.evidence[0].step, why), ("domain:groups", ["SPLIT_WRONG:repeated_entity"], "solution", ""))
        cases = {
            "names the table's columns": {"text": "Group the folds by household, which repeats across rows."},
            "names it with spaces": {"text": "Keep refund settled out because it is written after the outcome."},
            "names it with hyphens": {"text": "Keep refund-settled out because it is written after the outcome."},
            "names it in CamelCase": {"text": "Keep RefundSettled out because it is written after the outcome."},
            "names it with dots": {"text": "Keep refund.settled out because it is written after the outcome."},
            "names it joined": {"text": "Keep refundsettled out because it is written after the outcome."},
            "names another skin's key": {"text": "Group the folds by patient_id, which repeats across rows."},
            "names another skin's leak": {"text": "Keep recovery amount out: it is written after the outcome."},
            "carries a number": {"text": "Validation at 0.71 beat the holdout, so look for a leak first."},
            "carries a whole number": {"text": "With more than seventy groups, fold by the entity: seed 3 showed it."},
            "carries an identifier's value": {"text": "Values like HH-00012 repeat: group the folds by that column."},
            "cites no step": {"step": "somewhere"},
            "answers no code": {"codes": []},
            "answers an unknown code": {"codes": ["MADE_UP:code"]},
            "its scope is neither": {"scope": "domain:matching"},
            "too short": {"text": "Group."},
        }
        for name, change in cases.items():
            bullet_, why = self.screen(**change)
            self.assertIsNone(bullet_, name)
            self.assertTrue(why, name)

    def test_at_most_three_lessons_from_one_attempt(self):
        from dclab_rnd.learning.filter import MOST, screen_all

        drafts = [{"text": f"A general lesson number {w} about splitting by a repeated thing.", "scope": "domain",
                   "codes": ["SPLIT_WRONG:repeated_entity"], "step": "solution"} for w in ("one", "two", "three", "four")]
        kept, refused = screen_all(drafts, key(), [], ["solution"], "pk-1", None, "rule:test")
        self.assertEqual((len(kept), len(refused), MOST), (3, 1, 3))


class ReflectorTests(unittest.TestCase):
    def row(self, codes, verdict="fail", holdout=0.7):
        from dclab_rnd.learning.reflect import CODE_STEP

        checks = [{"trap": "x", "ok": False, "code": c} for c in codes]
        return {"pack": "pk-1", "arms": {"rules": {"verdict": verdict, "codes": codes, "checks": checks, "holdout": holdout}},
                "codes_step": [CODE_STEP[c] for c in codes]}

    def test_the_first_wrong_step_and_rule_lessons_from_codes_alone(self):
        from dclab_rnd.learning.reflect import first_wrong_step, reflect, steps_of

        row = self.row(["FALSE_ALARM:known_before", "SPLIT_WRONG:repeated_entity"])
        self.assertEqual(first_wrong_step(row["arms"]["rules"]), "solution")  # the split is chosen before the leakage review
        self.assertEqual(steps_of(row["arms"]["rules"]), [{"step": "solution", "verdict": "wrong"}, {"step": "leakage", "verdict": "wrong"},
                                                          {"step": "models", "verdict": "ok"}, {"step": "final", "verdict": "ok"}])
        out = reflect(row, key(), ["household", "prior_refund_count"])
        self.assertEqual(sorted(b.codes[0] for b in out["bullets"]), ["FALSE_ALARM:known_before", "SPLIT_WRONG:repeated_entity"])
        self.assertEqual(out["refused"], [])
        self.assertEqual({b.scope for b in out["bullets"]}, {"invariant", "domain:groups"})

    def test_no_lesson_from_a_pass_from_nothing_delivered_or_a_code_already_answered(self):
        from dclab_core.playbook import build
        from dclab_rnd.learning.reflect import reflect

        self.assertEqual(reflect(self.row([], verdict="pass"), key(), [])["bullets"], [])
        self.assertEqual(reflect(self.row(["SPLIT_WRONG:repeated_entity"], holdout=None), key(), [])["bullets"], [])
        book, _ = build("groups", "main", [delta("add", bullet=bullet())])
        self.assertEqual(reflect(self.row(["SPLIT_WRONG:repeated_entity"]), key(), [], book)["bullets"], [])

    def test_no_rule_lesson_names_any_skins_column(self):
        from dclab_core.challenge_pack import names_in
        from dclab_core.variants import generate
        from dclab_rnd.learning.reflect import RULE_LESSONS

        for seed in range(12):
            v = generate("groups", seed)
            for code, (_, text) in RULE_LESSONS.items():
                self.assertEqual(names_in(text, v.key, list(v.table.columns)), [], (seed, code))

    def test_a_models_drafts_are_screened_and_its_spend_capped(self):
        from dclab_rnd.learning.reflect import reflect

        class Model:
            model = "claude-sonnet-5-5"
            run_limit_eur = None
            sent = None

            def complete(self, messages, max_tokens=None):
                Model.sent = messages
                lessons = [{"text": "Group the folds by household, it repeats.", "scope": "domain", "codes": ["SPLIT_WRONG:repeated_entity"], "step": "solution"},
                           {"text": "When one column repeats across many rows for a thing with lasting traits, fold by it.", "scope": "domain",
                            "codes": ["SPLIT_WRONG:repeated_entity"], "step": "solution"}]
                return {"content": "Here: " + json.dumps({"lessons": lessons})}

        client = Model()
        client.spent_eur = 0.04  # what an earlier challenge spent: this one starts again at zero
        out = reflect(self.row(["SPLIT_WRONG:repeated_entity"]), key(), ["household"], client=client, cap_eur=0.05)
        self.assertEqual((client.run_limit_eur, client.spent_eur), (0.05, 0.0))
        self.assertEqual(len(out["bullets"]), 1)
        self.assertIn("names the table's columns", out["refused"][0]["why"])
        self.assertEqual(out["bullets"][0].by, "model:claude-sonnet-5-5")
        sent = json.dumps(Model.sent)
        for name in ("household", "refund_settled", "order_id", "prior_refund_count", "basket_value"):  # the model never sees a name
            self.assertNotIn(name, sent)
        self.assertIn("SPLIT_WRONG:repeated_entity", sent)

    def test_a_models_drafts_obey_the_editor_and_a_reached_cap_teaches_nothing(self):
        from dclab_core.playbook import build
        from dclab_rnd.learning.reflect import reflect

        class Model:
            model = "claude-sonnet-5-5"

            def __init__(self, lessons=None, fail=False):
                self.lessons, self.fail = lessons, fail

            def complete(self, messages, max_tokens=None):
                if self.fail:
                    raise RuntimeError("BudgetExceeded: this run has used its budget")
                return {"content": json.dumps({"lessons": self.lessons})}

        general = "When one column repeats across many rows for a thing with lasting traits, fold by it."
        drafts = [{"text": general, "scope": "domain", "codes": ["GAP_OPTIMISTIC:validation_over_holdout"], "step": "models"},  # not failed here
                  {"text": general + " Always.", "scope": "domain", "codes": ["SPLIT_WRONG:repeated_entity"], "step": "final"}]  # the wrong step
        out = reflect(self.row(["SPLIT_WRONG:repeated_entity"]), key(), [], client=Model(drafts))
        self.assertEqual(out["bullets"], [])
        self.assertEqual(sorted(r["why"][:28] for r in out["refused"]), ["answers a code this attempt ", "cites no step of the graded "])
        book, _ = build("groups", "main", [delta("add", bullet=bullet())])
        again = [{"text": general, "scope": "domain", "codes": ["SPLIT_WRONG:repeated_entity"], "step": "solution"}]
        covered = reflect(self.row(["SPLIT_WRONG:repeated_entity"]), key(), [], book, client=Model(again))
        self.assertEqual((covered["bullets"], covered["refused"][0]["why"]), ([], "answers a code a lesson already answers"))
        capped = reflect(self.row(["SPLIT_WRONG:repeated_entity"]), key(), [], client=Model(fail=True))
        self.assertEqual(capped["bullets"], [])
        self.assertIn("gave no answer", capped["why"])

    def test_the_reflector_is_a_classified_capped_role(self):
        from dclab_rnd.agents import harness
        from dclab_rnd.models import settings

        self.assertIn("reflector", harness.ROLES)
        p = settings.purpose("reflector")
        self.assertEqual((p.cell_values, p.keeps_text), (0, True))


class CaseMemoryTests(unittest.TestCase):
    def test_cases_keep_codes_and_ids_never_a_name_and_come_back_by_family_and_code(self):
        from dclab_rnd.learning.cases import FileCases, lines, similar

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        store = FileCases(home)
        store.add("groups", "pk-a", ["SPLIT_WRONG:repeated_entity", "household is the key"], "fail", lessons=["B-0123456789"])
        store.add("groups", "pk-b", ["FALSE_ALARM:known_before"], "fail")
        store.add("matching", "pk-c", ["SPLIT_WRONG:repeated_entity"], "fail")
        found = similar(store.all(), "groups", ["SPLIT_WRONG:repeated_entity"])
        self.assertEqual([c["pack"] for c in found], ["pk-a"])
        self.assertEqual(found[0]["codes"], ["SPLIT_WRONG:repeated_entity"])  # an unknown code (here, a name) is never kept
        self.assertNotIn("household", json.dumps(store.all()))
        self.assertIn("B-0123456789", lines(found)[0])


@unittest.skipIf(MISSING, MISSING)
class InjectionTests(unittest.TestCase):
    def test_the_intern_shows_accepted_lessons_in_its_first_user_message_never_the_system_prompt(self):
        from dclab_core.playbook import apply, build
        from dclab_rnd.intern.loop import POLICY, Intern
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.learning.inject import cited_in, for_solver
        from tests.test_intern import make

        b = bullet()
        book, _ = build("groups", "main", [delta("add", bullet=b)])
        book = apply(book, delta("status", target=b.id, status="probed"))
        book = apply(book, delta("status", target=b.id, status="accepted"))
        payload = for_solver(book, [{"family": "groups", "codes": ["SPLIT_WRONG:repeated_entity"], "verdict": "fail", "lessons": []}])
        self.assertEqual(payload["shown"], [b.id])
        projects, sessions = make(tempfile.mkdtemp())
        session = Intern(sessions, Toolbox(projects)).start("Predict late orders", {"max_steps": 3, "max_minutes": 1}, playbook=payload)
        system, user = session["messages"][0], session["messages"][1]
        self.assertEqual(system["content"], POLICY)  # the system prompt is the hash-gated file, untouched
        self.assertIn(b.text, user["content"])
        self.assertIn("Similar past problems:", user["content"])
        self.assertEqual(session["playbook"]["shown"], [b.id])
        self.assertEqual(len(session["playbook"]["sha256"]), 16)  # what it was shown, traceable
        session["steps"] = [{"thought": f"Following [{b.id}] I group the folds."}]
        self.assertEqual(cited_in(session), [b.id])
        self.assertIsNone(for_solver(build("groups", "main", [])[0]))  # nothing accepted, no case: nothing shown

    def test_the_rendered_payload_stays_under_its_budget(self):
        from dclab_core.playbook import apply, build, tokens
        from dclab_rnd.learning.inject import for_solver

        deltas, ids = [], []
        for i in range(60):
            b = bullet(text=f"A general lesson about splitting by a repeated thing, variant {chr(97 + i % 26)}{chr(97 + i // 26)}.")
            deltas.append(delta("add", bullet=b))
            ids.append(b.id)
        book, _ = build("groups", "main", deltas)
        for i in ids:
            book = apply(book, delta("status", target=i, status="probed"))
            book = apply(book, delta("status", target=i, status="accepted"))
        payload = for_solver(book, [{"family": "groups", "codes": ["SPLIT_WRONG:repeated_entity"], "verdict": "fail"}] * 20, budget=300)
        self.assertLessEqual(tokens(payload["text"]), 300)


@unittest.skipIf(MISSING, MISSING)
class DraftingTests(unittest.TestCase):
    def test_drafting_reads_learning_variants_only_and_files_screened_drafts_as_a_log(self):
        from dclab_core import variants
        from dclab_rnd.learning import drafting
        from dclab_rnd.learning.playbooks import FilePlaybooks

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        asked = []
        real = variants.seeds

        def seeds(family, partition, n, start=0):
            asked.append(partition)
            return real(family, partition, n, start)

        with mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"}), mock.patch.object(drafting.V, "seeds", seeds):
            out = drafting.draft_lessons(home, "groups", 3, log=lambda *_: None)
        self.assertEqual(asked, ["learning"])
        book, refused = FilePlaybooks(home).book("groups")
        self.assertEqual(refused, [])
        self.assertTrue(book.bullets)
        self.assertTrue(all(b.status == "draft" for b in book.bullets))  # nothing shown to a solver before H3's probe
        log = (home / "playbook" / "groups" / "main.deltas.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log), book.version)
        self.assertEqual(len((home / "learning" / "cases.jsonl").read_text(encoding="utf-8").splitlines()), 3)
        self.assertEqual(out["playbook"]["by_status"], {"draft": len(book.bullets)})
        self.assertTrue(all(b.evidence[0].run and b.evidence[0].run.startswith("learning-") for b in book.bullets))
        kept = [json.loads(x) for x in (home / "learning" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(kept), 3)

    def test_the_stores_door_screens_a_bullet_however_it_was_made(self):
        from dclab_core.playbook import Refused
        from dclab_rnd.learning.playbooks import FilePlaybooks

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        books = FilePlaybooks(home)
        for text in ("Group the folds by household, which repeats.", "Fold by the entity when there are 3 or more rows of it."):
            with self.assertRaises(Refused):
                books.add("groups", bullet(text=text), reason="a direct add")
        self.assertEqual(books.deltas("groups"), [])  # nothing written
        books.add("groups", bullet(), reason="a general lesson")
        self.assertEqual(len(books.deltas("groups")), 1)


if __name__ == "__main__":
    unittest.main()

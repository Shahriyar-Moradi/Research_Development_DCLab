"""2026-11-01: a written person's story must fit the table they share (conversation_exam.table_fit), and the brief for the writer of
a new set is generated from the real tables (conversation_exam.brief). The route test found at least three of eight played
people of a set written without the tables telling a story the table could not hold."""
import copy
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import conversation_exam as X  # noqa: E402
from dclab_rnd.agent_eval.conversation_exam import REQUIRE_CLAIMS, brief, check_person, load, table_fit  # noqa: E402


class TableFitTests(unittest.TestCase):
    def person(self, upload="customers", claims=None):
        base = copy.deepcopy(next(p for p in load("dev_route") if p["route"] == "preparation"))
        base["upload"] = upload
        base.pop("table_claims", None)
        if claims is not None:
            base["table_claims"] = claims
        return base

    def test_the_stories_that_broke_the_route_test_would_have_been_refused_with_the_reason(self):
        r17 = {"values": {"city": ["Bombay", "Bengaluru"]}, "repeated": ["customer_id"]}  # Indian cities, repeated customers: in a UAE table
        problems = table_fit.fit(self.person("customers", r17))
        self.assertTrue(any("Bombay" in p and "Dubai" in p for p in problems), problems)
        self.assertTrue(any("customer_id" in p and "never does" in p for p in problems), problems)
        r20 = {"values": {"destination": ["Machu Picchu", "Cusco"]}}  # a Peruvian tour operator: in a table of Paris and Dubai
        problems = table_fit.fit(self.person("bookings", r20))
        self.assertTrue(any("Machu Picchu" in p and "Istanbul" in p for p in problems), problems)

    def test_a_story_that_fits_passes_whatever_the_case_of_its_words(self):
        claims = {"columns": ["customer_id", "city"], "values": {"city": ["dubai", "SHARJAH"]}, "duplicate_rows": False,
                  "unique": ["customer_id"], "missing": ["monthly_spend"]}
        self.assertEqual(table_fit.fit(self.person("customers", claims)), [])
        self.assertEqual(table_fit.fit(self.person("sales", {"duplicate_rows": True, "repeated": ["order_id", "customer_id"], "missing": ["amount"]})), [])

    def test_each_kind_of_false_claim_is_named(self):
        cases = [({"columns": ["customer_email"]}, "does not have"), ({"values": {"state": ["Texas"]}}, "does not have"),
                 ({"duplicate_rows": True}, "has none"), ({"duplicate_rows": False}, None), ({"missing": ["plan"]}, "has none"),
                 ({"values": {"customer_id": ["C-00001"]}}, "too many distinct values"), ({"repeated": ["plan"]}, "id columns"), ({"repeated": ["customer_id"]}, "never does"), ({"repeated": ["customer_ref_id"]}, "no such column"),
                 ({"rows": 50}, "takes only")]
        for claims, expect in cases:
            problems = table_fit.fit(self.person("customers", claims))
            if expect is None:
                self.assertEqual(problems, [], claims)
            else:
                self.assertTrue(any(expect in p for p in problems), (claims, problems))

    def test_a_badly_shaped_claim_is_refused_with_its_shape_never_a_traceback(self):
        """The review: {"columns": 5}, {"values": ["city"]}, a null and a bare string crashed or read character by character."""
        for claims in ({"columns": 5}, {"columns": "city"}, {"values": ["city"]}, {"values": {"city": None}}, {"values": {"city": "Dubai"}},
                       {"duplicate_rows": "no"}, {"missing": [3]}, {"repeated": {"customer_id": True}}, {"unique": None, "columns": 7}):
            problems = table_fit.fit(self.person("customers", claims))
            self.assertTrue(problems and all("malformed table_claims" in p for p in problems), (claims, problems))
        self.assertTrue(table_fit.fit(self.person("customers", ["columns"])))  # not an object at all

    def test_claims_that_say_nothing_do_not_satisfy_a_set_that_requires_them(self):
        """The review: "table_claims": {} met the rule the package exists for."""
        for claims in ({}, {"columns": []}, {"values": {}}, {"repeated": [], "missing": []}):
            problems = table_fit.fit(self.person("customers", claims), require=True)
            self.assertTrue(any("says nothing" in p for p in problems), (claims, problems))
            self.assertEqual(table_fit.fit(self.person("customers", claims)), [], claims)  # optional claims may be empty
        self.assertEqual(table_fit.fit(self.person("customers", {"duplicate_rows": False}), require=True), [])  # "no duplicates" is a claim

    def test_uniqueness_can_be_claimed_and_is_checked(self):
        self.assertEqual(table_fit.fit(self.person("customers", {"unique": ["customer_id"]})), [])
        problems = table_fit.fit(self.person("sales", {"unique": ["customer_id"]}))  # 'one row per customer' on a table where it repeats
        self.assertTrue(any("repeats" in p for p in problems), problems)
        self.assertTrue(any("id columns" in p for p in table_fit.fit(self.person("sales", {"unique": ["status"]}))))

    def test_numbers_and_yes_no_columns_can_be_claimed_when_they_hold_few_values(self):
        self.assertEqual(table_fit.fit(self.person("bookings", {"values": {"cancelled": [0, 1], "free_cancellation": [True, "false"]}})), [])
        problems = table_fit.fit(self.person("bookings", {"values": {"cancelled": [2]}}))
        self.assertTrue(any("names 2 in 'cancelled'" in p and "holds only 0, 1" in p for p in problems), problems)
        self.assertTrue(any("too many" in p for p in table_fit.fit(self.person("bookings", {"values": {"lead_time_days": [5]}}))))
        facts = table_fit.facts("bookings")
        self.assertEqual(facts["ranges"]["lead_time_days"], [1, 179])
        self.assertEqual(facts["examples"]["price"], "1257 AED")
        self.assertEqual(facts["dates"]["travel_date"][0], "2025-01-16")

    def test_the_exams_main_refuses_a_set_whose_people_lack_claims_when_it_requires_them(self):
        err = io.StringIO()
        with mock.patch.object(X, "REQUIRE_CLAIMS", frozenset({"dev_route"})), redirect_stderr(err):
            code = X.main(["--split", "dev_route", "--only", "r-01,r-03"])
        self.assertEqual(code, 2)
        self.assertIn("table_claims", err.getvalue())
        with mock.patch.object(X, "REQUIRE_CLAIMS", frozenset()), redirect_stderr(err):
            self.assertEqual(X.main(["--split", "dev_route", "--only", "r-01,r-03"]), 0)  # the plan only; nothing is sent

    def test_check_person_asks_for_claims_only_where_the_set_requires_them(self):
        no_claims = self.person("customers")
        self.assertEqual(check_person(no_claims), [])  # the sets that exist have none and are unchanged
        self.assertTrue(any("table_claims" in p for p in check_person(no_claims, require_claims=True)))
        self.assertEqual(check_person(self.person("customers", {"values": {"city": ["Dubai"]}}), require_claims=True), [])
        self.assertTrue(check_person(self.person("customers", {"values": {"city": ["Bombay"]}})))  # a false claim is refused even if optional
        no_table = self.person(None, {"columns": ["city"]})
        self.assertTrue(any("without a table" in p for p in check_person(no_table)))
        self.assertEqual(check_person(self.person(None), require_claims=True), [])  # no table, nothing to claim
        self.assertIn("sealed_v3", REQUIRE_CLAIMS)

    def test_the_sets_that_exist_are_unchanged(self):
        for split in ("dev", "dev_route"):
            self.assertEqual([p["id"] for p in load(split) if check_person(p)], [], split)


class BriefTests(unittest.TestCase):
    def test_the_brief_check_command_passes_when_current_and_fails_when_stale(self):
        self.assertEqual(brief.main(["--check"]), 0)
        with mock.patch.object(brief, "render", return_value="a stale brief"), redirect_stderr(io.StringIO()):
            self.assertEqual(brief.main(["--check"]), 1)


    def test_the_brief_is_the_one_generated_from_the_real_tables(self):
        self.assertTrue(brief.BRIEF.exists())
        self.assertEqual(brief.BRIEF.read_text(encoding="utf-8"), brief.render(), "run python -m dclab_rnd.agent_eval.conversation_exam.brief")
        text = brief.render()
        for token in ("Istanbul", "Ras Al Khaimah", "refunded", "`monthly_spend`", "table_claims", "Put the hard kinds first", "quarter euro",
                      "1257 AED", "2025-01-16", "from 1 to 179", "holds only: 0, 1", "SEALED_ARMS", "technical advice conversation costs more", "clear_request"):
            self.assertIn(token, text)
        self.assertNotIn("Machu Picchu", text.split("## The record")[0].split("## The three tables")[1])  # the tables hold no such place


if __name__ == "__main__":
    unittest.main()

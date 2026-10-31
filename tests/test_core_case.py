"""Package 37.1 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): the case file, the one living record of a case. Every fact says
where it came from (the person's own words, quoted; a fact code computed; or an assumption, shown and never counted as
known); every change is a new version that says who made it and why; readiness and the options' checks are code; the
problem card is a view of it and comes back unchanged."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_core.card import ProblemCard, problem_card  # noqa: E402
from dclab_core.contracts import CONTRACTS  # noqa: E402
from dclab_core.understand import understand  # noqa: E402

WORDS = ["our sales have been going down for a few months",
         "I want to know which customers are about to leave so my team can call them",
         "the retention team, four people, about 300 calls a month"]


def said(value, quote):
    return C.CaseFact(value=value, status="said", quote=quote)


class FactTests(unittest.TestCase):
    def test_a_said_fact_must_quote_the_persons_own_words(self):
        case = C.new("case-1", WORDS)
        ok = C.apply(case, "goal", said("know which customers will leave", "which customers are about to leave"), by="agent", why="their second message")
        self.assertEqual(ok.goal.value, "know which customers will leave")
        with self.assertRaises(C.CaseRefused) as refused:  # the model cannot claim understanding the person never gave
            C.apply(case, "decision", said("call the risky ones", "call the riskiest customers first"), by="agent", why="guess")
        self.assertIn("not the person's words", str(refused.exception))

    def test_a_computed_fact_names_what_computed_it_and_an_assumption_is_never_known(self):
        case = C.new("case-1", WORDS)
        with self.assertRaises(C.CaseRefused):
            C.apply(case, "cannot_answer", C.CaseFact(value="no cancellation date", status="computed"), by="code", why="profile")
        case = C.apply(case, "cannot_answer", C.CaseFact(value="no column says when a customer left", status="computed", computed="profile.columns"),
                       by="code", why="the table's profile")
        case = C.apply(case, "assumptions", C.CaseFact(value="a month is the horizon", status="assumed"), by="agent", why="usual for churn")
        self.assertEqual(case.cannot_answer[0].computed, "profile.columns")
        self.assertFalse(C.known(case.assumptions[0]))

    def test_an_assumption_the_person_confirms_counts_and_one_they_reject_does_not(self):
        case = C.new("case-1", WORDS)
        case = C.apply(case, "assumptions", C.CaseFact(value="next month is the horizon", status="assumed"), by="agent", why="usual")
        with self.assertRaises(C.CaseRefused):  # an earlier message cannot answer an assumption made after it (the review)
            C.confirm(case, 0, quote="the retention team", by="person")
        case = C.hear(case, "yes, next month is right")
        with self.assertRaises(C.CaseRefused):  # only the person confirms
            C.confirm(case, 0, quote="yes, next month is right", by="agent")
        case = C.confirm(case, 0, quote="yes, next month is right", by="person")
        self.assertTrue(C.known(case.assumptions[0]))
        with self.assertRaises(C.CaseRefused):
            C.confirm(case, 0, quote="something they never wrote", by="person")

    def test_a_quote_is_two_whole_words_of_a_message_sharing_a_word_with_the_value(self):  # the review: "a" and "ales te" passed
        case = C.new("case-1", ["call the sales team about leaving customers"])
        for quote in ("a", "ales te", "the sales"):
            with self.assertRaises(C.CaseRefused, msg=quote):
                C.apply(case, "goal", said("maximise revenue with discounts", quote), by="agent", why="w")
        C.apply(case, "goal", said("keep the leaving customers", "leaving customers"), by="agent", why="w")

    def test_only_code_computes_and_only_about_the_data(self):  # the review: an agent reached readiness with "computed" facts
        case = C.new("case-1", WORDS)
        fact = C.CaseFact(value="invented", status="computed", computed="my reading")
        for field in ("goal", "decision", "user"):
            with self.assertRaises(C.CaseRefused):
                C.apply(case, field, fact, by="code", why="w")
        with self.assertRaises(C.CaseRefused):
            C.apply(case, "data", fact, by="agent", why="w")
        self.assertEqual(C.apply(case, "data", fact, by="code", why="the upload").data.status, "computed")

    def test_the_persons_words_grow_only_by_hearing_them(self):  # the review: apply could replace them
        case = C.new("case-1", WORDS)
        with self.assertRaises(C.CaseRefused):
            C.apply(case, "words", ["I want discounts for all"], by="agent", why="w")
        heard = C.hear(case, "and quickly")
        self.assertEqual((heard.version, heard.changes[-1].by, heard.words[-1]), (1, "person", "and quickly"))


class VersionTests(unittest.TestCase):
    def test_every_change_is_a_version_with_who_and_why_and_the_old_value(self):
        case = C.new("case-1", WORDS)
        one = C.apply(case, "goal", said("fewer customers leaving", "which customers are about to leave"), by="agent", why="first reading")
        two = C.apply(one, "goal", said("call the customers about to leave", "so my team can call them"), by="person", why="corrected on the panel")
        self.assertEqual((case.version, one.version, two.version), (0, 1, 2))
        last = two.changes[-1]
        self.assertEqual((last.field, last.by, last.why), ("goal", "person", "corrected on the panel"))
        self.assertEqual(last.before["value"], "fewer customers leaving")
        self.assertEqual(case.goal, None)  # the earlier version is untouched (frozen)
        heard = C.hear(one, "more words")
        self.assertIsNot(heard.words, one.words)  # the review: versions shared their lists
        self.assertEqual(len(one.words), len(WORDS))

    def test_a_change_that_would_break_the_contract_is_refused_and_counting_is_versioned(self):
        case = C.new("case-1", WORDS)
        option = C.CaseOption(title="Rank", what="a list", track="prediction", tool="ml_project")
        with self.assertRaises(C.CaseRefused):  # five options: never storable (the review)
            C.apply(case, "options", [option] * 5, by="agent", why="w")
        with self.assertRaises(C.CaseRefused):
            C.apply(case, "options", [option.model_copy(update={"track": "advice"})], by="agent", why="w")
        counted = C.asked(case, declined=True)
        self.assertEqual((counted.version, counted.asked, counted.declined), (1, 1, 1))

    def test_the_contract_is_registered_and_reads_its_own_json(self):
        self.assertIs(CONTRACTS["case_file"], C.CaseFile)
        case = C.apply(C.new("case-1", WORDS), "goal", said("which customers leave", "which customers are about to leave"), by="agent", why="w")
        self.assertEqual(C.CaseFile.read(case.model_dump_json()), case)


class ReadinessTests(unittest.TestCase):
    def full(self):
        case = C.new("case-1", WORDS)
        case = C.apply(case, "goal", said("know who will leave", "which customers are about to leave"), by="agent", why="w")
        case = C.apply(case, "decision", said("the team calls them", "so my team can call them"), by="agent", why="w")
        case = C.apply(case, "user", said("the retention team", "the retention team, four people"), by="agent", why="w")
        return C.apply(case, "data", C.CaseFact(value="customers.csv, one row per customer", status="computed", computed="assets.customers"),
                       by="code", why="the upload")

    def test_ready_needs_the_goal_the_decision_the_user_and_the_data(self):
        self.assertTrue(C.readiness(self.full()).ready)
        r = C.readiness(C.new("case-1", WORDS))
        self.assertFalse(r.ready)
        self.assertEqual(set(r.missing), {"goal", "decision", "user", "data"})

    def test_an_assumed_decision_is_not_known_and_a_blocking_unknown_holds_it(self):
        case = self.full()
        case = C.apply(case, "decision", C.CaseFact(value="maybe a discount", status="assumed"), by="agent", why="w")
        self.assertIn("decision", C.readiness(case).missing)
        case = C.apply(self.full(), "unknowns", C.Unknown(field="outcome", question="How do you know a customer left?", blocks="all"), by="agent", why="w")
        self.assertFalse(C.readiness(case).ready)

    def test_no_data_said_by_the_person_counts_as_known(self):
        case = C.hear(self.full().model_copy(update={"data": None}), "we have no data, I only need advice")
        case = C.apply(case, "data", said("no data, advice only", "we have no data"), by="agent", why="w")
        self.assertNotIn("data", C.readiness(case).missing)

    def test_the_question_ceiling_lets_it_propose_with_assumptions(self):  # the owner: about six, or two declined
        case = C.new("case-1", WORDS)
        self.assertFalse(C.readiness(case).may_propose)
        for _ in range(C.CEILING["asked"]):
            case = C.asked(case)
        self.assertTrue(C.readiness(case).may_propose)
        self.assertFalse(C.readiness(case).ready)
        case = C.new("case-1", WORDS)
        case = C.asked(C.asked(case, declined=True), declined=True)
        self.assertTrue(C.readiness(case).may_propose)


class OptionTests(unittest.TestCase):
    COLUMNS = {"customer_id", "plan", "monthly_spend", "churned_next_month"}

    def option(self, title, track, tool, columns=(), what="what they get"):
        return C.CaseOption(title=title, what=what, gets="a list", tradeoff="needs labels", effort="a day", track=track, tool=tool, columns=list(columns))

    def test_two_or_three_options_that_differ_in_kind_with_real_columns_and_no_invented_number(self):
        case = C.new("case-1", WORDS)
        options = [self.option("Rank customers by risk", "prediction", "ml_project", ["churned_next_month"]),
                   self.option("Compare churn by plan", "analysis", "query", ["plan"]),
                   self.option("Same ranking again", "prediction", "ml_project"),
                   self.option("Use their lifetime value", "analysis", "query", ["lifetime_spend"]),
                   self.option("Call the top 12% first", "prediction", "ml_project", what="12% of them")]
        kept, refused = C.check_options(options, case, columns=self.COLUMNS, computed=[])
        self.assertEqual([o.title for o in kept], ["Rank customers by risk", "Compare churn by plan"])
        reasons = " ".join(refused.values())
        self.assertIn("same kind", reasons)
        self.assertIn("lifetime_spend", reasons)
        self.assertIn("12", reasons)

    def test_one_tool_is_one_kind_whatever_the_track_says(self):  # the review: three tracks on ml_project were kept
        options = [self.option("Rank by risk", "prediction", "ml_project"), self.option("Advice on risk", "advice", "ml_project"),
                   self.option("Business case", "business", "business_brief")]
        kept, refused = C.check_options(options, C.new("c", WORDS), columns=self.COLUMNS)
        self.assertEqual([o.title for o in kept], ["Rank by risk", "Business case"])

    def test_numbers_with_units_and_ratios_are_claims_and_equal_forms_match(self):  # the review
        self.assertEqual(C.numbers_unsaid("an AUC of .85, cuts churn 5x, 9 in 10 stay", []), [".85", "5x", "9 in 10"])
        self.assertEqual(C.numbers_unsaid("about 12% of them, 300.0 calls", ["12 percent leave", "300 calls a month"]), [])
        self.assertEqual(C.numbers_unsaid("two options, step 3", []), [])

    def test_an_option_says_what_they_get_its_trade_off_and_its_effort(self):  # 37.5
        bare = C.CaseOption(title="Rank by risk", what="a ranked list", track="prediction", tool="ml_project")
        kept, refused = C.check_options([bare], C.new("c", WORDS), columns=self.COLUMNS)
        self.assertEqual(kept, [])
        self.assertIn("trade-off", refused["Rank by risk"])

    def test_a_route_no_tool_runs_is_refused(self):
        kept, refused = C.check_options([self.option("Write a poem", "advice", "poem_writer")], C.new("c", WORDS), columns=set(), computed=[])
        self.assertEqual(kept, [])
        self.assertIn("poem_writer", " ".join(refused.values()))


class CardViewTests(unittest.TestCase):
    def card(self):
        words = "predict which customers will churn next month so the retention team can call the top 300"
        return problem_card(understand(words), words=words, data={"columns": ["customer_id", "plan", "churned_next_month"]})

    def test_the_problem_card_comes_back_unchanged(self):
        card = self.card()
        case = C.from_card("case-1", card, words=["predict which customers will churn next month so the retention team can call the top 300"])
        self.assertEqual(C.to_card(case), card)
        self.assertEqual(ProblemCard.read(C.to_card(case).model_dump_json()), card)

    def test_a_cards_quote_not_in_these_words_is_only_an_assumption(self):  # the review
        words = "flag fraudulent orders so the reviewers can block them before shipping"
        card = problem_card(understand(words), words=words, answers={"decision_owner": "the risk lead", "decision": "hold the order for review"})
        case = C.from_card("case-1", card, words=[words])
        self.assertEqual(case.user.status, "assumed")
        self.assertFalse(C.known(case.decision))

    def test_a_decision_the_person_gave_closes_its_open_field_and_an_assumption_keeps_the_card(self):
        words = "flag fraudulent orders so the reviewers can block them before shipping"
        card = problem_card(understand(words), words=words)
        case = C.hear(C.from_card("case-1", card, words=[words]), "the fraud lead decides, she holds the order for review")
        filled = C.apply(case, "decision", said("hold the order for review", "holds the order for review"), by="agent", why="w")
        self.assertNotIn("decision", [f.field for f in C.to_card(filled).open_fields])
        guessed = C.apply(case, "decision", C.CaseFact(value="refund them", status="assumed"), by="agent", why="w")
        self.assertEqual(C.to_card(guessed).decision, card.decision)

    def test_a_goal_the_person_corrected_reaches_the_card(self):
        words = ["predict which customers will churn next month so the retention team can call the top 300", "no, I want to keep them, not just predict"]
        case = C.from_card("case-1", self.card(), words=words)
        case = C.apply(case, "goal", said("keep the customers", "I want to keep them"), by="person", why="corrected")
        self.assertEqual(C.to_card(case).brief.goal.quote, "I want to keep them")


if __name__ == "__main__":
    unittest.main()

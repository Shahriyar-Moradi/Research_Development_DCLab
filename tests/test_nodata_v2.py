"""Solving without data, the rebuild (dclab_rnd.nodata, 2026-11-16): one free model pass writes the solution, code checks
it and owns its numbers, one second reader says what is weak. Scripted models only."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import experience, nodata, prompts  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models import settings  # noqa: E402
from dclab_rnd.nodata import architect, checks, critic, economics, render, simulate, stance  # noqa: E402

GYM = ("I own three gyms. Members quit all the time. About 2,400 members, maybe 90 quit a month. A personal call from a trainer costs "
       "me about 10 euros, and a member pays 45 a month. I would rather not hand over member details.")
SAAS = ("We want to reduce churn before renewal, but we do not have a clean dataset yet. We have 5,000 enterprise accounts, an 8% annual "
        "churn rate, a $50 CSM intervention cost, and $2,500 saved per retained account.")
GOOD = """Need: full

## The problem
Who the trainers call first. 90 / 2,400 = 3.75% of members quit a month.

## What needs care
A payment that stops is the outcome, not a warning.

## Options
Call everyone, call nobody, or call the members whose visits fell. The last costs the least trainer time for the most members kept.

## Business solution
Call the members whose visits fell.

## Data to record
| column | what it is | known when | role |
|---|---|---|---|
| member_id | the member | before | key |
| list_date | the day the list is made | at | time |
| visits_recent | visits in the last weeks | before | input |
| payment_stopped | the payment stopped | after | not used |
| quit | the member quit afterwards | after | label |

One row is a member on a list date.

## Model
A rule first, then a model compared with it.
Kind: binary classification

## Plan
1. Make the list by hand. Needs: the membership software. Finished when: a list exists.
2. Call. Needs: trainers' time. Finished when: every call is noted.
3. Review. Needs: the notes. Finished when: the rule is kept or changed.

## What only real data can settle
How well any of this works.

## Questions for you
- How many calls can the trainers make in a week?
"""


class Client:
    def __init__(self, *replies, cost=0.01):
        self.replies, self.seen, self.spent_eur, self.cost = list(replies), [], 0.0, cost

    def complete(self, messages, tools=None, **kwargs):
        self.seen.append(json.loads(json.dumps(messages)))
        self.spent_eur += self.cost
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply}


class Gateway:
    def __init__(self, **clients):
        self.clients, self.asked = clients, []

    def client(self, purpose, **kwargs):
        self.asked.append((purpose, kwargs))
        return self.clients.get(purpose)


def live():
    return mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "0"})


class NumbersTests(unittest.TestCase):
    def test_code_writes_sums_from_the_persons_numbers_and_assumes_nothing(self):
        self.assertEqual([s["sum"] for s in economics.known(SAAS)],
                         ["50 / 2,500 = 2%", "5,000 × 8% = 400", "5,000 × 50 = 250,000", "5,000 × 8% × 2,500 = 1,000,000"])
        self.assertEqual(economics.known("Should we open a second warehouse? Deliveries south take 6 days, north 2."), [])
        for s in economics.known(GYM):  # the first design filled what it could not read with defaults (10,000 cases, 8%, 180)
            self.assertFalse(economics.unsaid(s["sum"], [GYM]), s)

    def test_a_sum_is_verified_only_when_its_operands_are_sourced_and_its_arithmetic_is_right(self):
        text = ("90 / 2,400 = 3.75% quit a month. One call is 10 / 45 = 22.2% of a month's fee. Wrong: 90 / 2,400 = 5%. "
                "Not theirs: 45 × 12 = 540. A count: 3 x 45 = 135. A range of 2-4 weeks is no sum.")
        out = economics.verify(text, [GYM])
        self.assertEqual(out["verified"], ["90 / 2,400 = 3.75%", "10 / 45 = 22.2%", "3 x 45 = 135"])
        self.assertEqual([w["sum"] for w in out["wrong"]], ["90 / 2,400 = 5%"])
        self.assertEqual(out["unsourced"][0]["operands"], ["12"])
        self.assertEqual(economics.unsaid(text, [GYM]), ["5%", "12", "540"])

    def test_a_small_number_or_a_near_miss_never_makes_an_invented_figure_a_checked_one(self):
        """The review of 2026-11-16: "9 / 10 = 90%" passed as checked, and so did a result within one part in a hundred."""
        for text in ("Gyms usually lose 1 / 4 = 25% of members a year.", "A model is right 9 / 10 = 90% of the time.", "A text costs $5 × 2,400 = $12,000."):
            self.assertEqual(economics.verify(text, [GYM])["verified"], [], text)
            self.assertTrue(economics.unsaid(text, [GYM]), text)
        for text in ("2,400 × 45 = 108,900", "45 x 90 = 4,090", "2,400 - 90 = 2,330", "90 / 2,400 = 3.78%", "10 + 45 = 55.5"):
            self.assertEqual(len(economics.verify(text, [GYM])["wrong"]), 1, text)
        for text in ("90 / 2,400 ≈ 3.8%", "90 / 2,400 = 4%", "2,400 × 45 = 108,000"):  # rounded to the digits shown is right
            self.assertEqual(len(economics.verify(text, [GYM])["verified"]), 1, text)

    def test_a_right_sum_is_read_in_the_forms_people_write(self):
        for text in ("90 / 2,400 = **3.75%**", "2,400 members × €45 = €108,000", "3.75% (90 / 2,400)", "۹۰ / ۲٬۴۰۰ = ۳٫۷۵٪", "90 / 2,400 = 3.75٪"):
            self.assertEqual(len(economics.verify(text, [GYM])["verified"]), 1, text)
            self.assertEqual(economics.unsaid(text, [GYM]), [], text)

    def test_the_number_of_a_list_item_states_nothing(self):
        plan = "\n".join(f"{i}. Do the next thing." for i in range(1, 14))
        self.assertEqual(economics.unsaid(plan, [GYM]), [])
        self.assertEqual(checks.strip_unsaid(plan, [GYM]), (plan, []))
        self.assertEqual(economics.unsaid("12. Call 500 members.", [GYM]), ["500"])

    def test_a_verified_result_can_be_used_in_a_later_sum(self):
        out = economics.verify("5,000 × 8% = 400 churners, worth 400 × 2,500 = 1,000,000.", [SAAS])
        self.assertEqual(len(out["verified"]), 2)

    def test_persian_number_marks_are_numbers(self):
        from dclab_rnd import cited

        words = "۲۰٬۰۰۰ مشترک داریم و نرخ ریزش ۱۰٪ است."
        self.assertEqual(cited.unsaid("20,000 مشترک و 10% ریزش، یعنی ۲۰٬۰۰۰ و ۱۰٪.", [words]), [])
        self.assertEqual(cited.unsaid("35% ریزش", [words]), ["35%"])
        self.assertEqual(cited.unsaid("نرخ 10% است", ["نرخ ریزش ۱۰ درصد است"]), [])  # the Persian word for percent
        self.assertEqual([s["sum"] for s in economics.known("هنوز دیتا ندارم. ۲۰٬۰۰۰ مشترک داریم و نرخ ریزش ۱۰٪ است.")], ["20,000 × 10% = 2,000"])


class ChecksTests(unittest.TestCase):
    def failed(self, text, words=GYM):
        return [c["check"] for c in checks.failed(checks.run(text, [words]))]

    def test_a_sound_answer_passes_every_check(self):
        self.assertEqual(self.failed(GOOD), [])
        self.assertEqual(checks.need_of(GOOD), "full")
        self.assertEqual([(r["column"], r["known"], r["role"]) for r in checks.columns(checks.sections(GOOD)["Data to record"])][3:],
                         [("payment_stopped", "after", "not used"), ("quit", "after", "label")])
        self.assertEqual(len(checks.steps(checks.sections(GOOD)["Plan"])), 3)
        self.assertEqual(checks.kind_of(checks.sections(GOOD)["Model"]), "binary classification")

    def test_each_rule_fails_on_its_own_fault(self):
        faults = {
            "need": GOOD.replace("Need: full", "Need: everything"),
            "parts": GOOD.replace("## What needs care\nA payment that stops is the outcome, not a warning.\n", ""),
            "sums": GOOD.replace("= 3.75%", "= 5%"),
            "numbers": GOOD.replace("not a warning.", "not a warning. Gyms usually lose 30% a year."),
            "data table": GOOD.replace("| list_date | the day the list is made | at | time |\n", "").replace("| visits_recent | visits in the last weeks | before | input |\n", ""),
            "inputs known in time": GOOD.replace("| after | not used |", "| after | input |"),
            "an outcome and an input": GOOD.replace("| quit | the member quit afterwards | after | label |\n", "| note | a free note | before | not used |\n"),
            "label after the decision": GOOD.replace("| the member quit afterwards | after | label |", "| the member quit | before | label |"),
            "plan": GOOD.replace("3. Review. Needs: the notes. Finished when: the rule is kept or changed.\n", ""),
            "model": GOOD.replace("Kind: binary classification\n", ""),
            "no upload asked": GOOD.replace("- How many calls", "- Could you upload an export of the member table?\n- How many calls"),
            "no data made": GOOD.replace("How well any of this works.", "How well any of this works. I have generated a table of 3 members."),
        }
        for name, text in faults.items():
            self.assertIn(name, self.failed(text), name)

    def test_an_unclear_cell_takes_the_strict_reading_and_fails_the_table(self):
        for row in ("| x | late | before or after | input |", "| x | both | after | label / input |", "| x | short | before |"):
            text = GOOD.replace("| visits_recent | visits in the last weeks | before | input |", "| visits_recent | visits in the last weeks | before | input |\n" + row)
            self.assertIn("data table", self.failed(text), row)
        late = GOOD.replace("| visits_recent | visits in the last weeks | before | input |", "| x | late | before or after | input |")
        self.assertIn("inputs known in time", self.failed(late))

    def test_a_sound_answer_in_common_markdown_passes(self):
        loose = (GOOD.replace("Need: full", "**Need:** full").replace("## Plan", "## 6. Plan:").replace("## What needs care", "**What needs care**")
                 .replace("1. Make", "**1.** Make").replace("2. Call.", "**2.** Call.").replace("3. Review.", "3) Review."))
        self.assertEqual(self.failed(loose), [])
        self.assertEqual(checks.need_of("## Plan\n1. Call. Need: trainers' time."), None)  # the need is named at the top only
        self.assertEqual(checks.without_need_line("**Need: full**\n\n## The problem\nNeed: trainers"), "## The problem\nNeed: trainers")

    def test_a_request_for_the_persons_data_is_read_and_an_honest_sentence_is_not(self):
        for text in ("If you can, send me a CSV.", "Upload your file and I will go on.", "Can you share a sample?", "Could you upload an export of the member table?"):
            self.assertTrue(checks.asks_upload(text), text)
        for text in ("Which system could export the visits table each week?", "Please send the trainers a spreadsheet of the quiet members.",
                     "You do not need to share your data.", "When real data is ready, upload/connect it."):
            self.assertFalse(checks.asks_upload(text), text)
        for text in ("The table below was generated for you.", "We simulated a table of members."):
            self.assertTrue(checks.says_made(text), text)
        for text in ("I built this plan so that the data you record later fits it.", "Once I have created a design, you approve it before any rows exist.",
                     "No data has been created."):
            self.assertFalse(checks.says_made(text), text)

    def test_a_person_who_asked_for_no_model_gets_none(self):
        business = GOOD.replace("Need: full", "Need: business")
        self.assertIn("no model unasked", self.failed(business))
        without = re.sub(r"## Model\n.*?\n\n(?=## Plan)", "", business, flags=re.S)
        self.assertEqual(self.failed(without), [])

    def test_made_up_data_is_asked_about_before_anything(self):
        asked = GOOD.replace("Need: full", "Need: simulate")
        self.assertEqual(self.failed(asked), [])
        silent = asked[:asked.index("## Questions for you")]
        self.assertIn("questions before made-up data", self.failed(silent))

    def test_what_states_a_number_nobody_gave_is_removed_and_a_right_sum_stays(self):
        text = "It pays. Gyms lose 30% a year. 90 / 2,400 = 3.75% quit.\n- Aim for 500 calls.\n- Call the quiet ones."
        kept, removed = checks.strip_unsaid(text, [GYM])
        self.assertEqual(removed, ["Gyms lose 30% a year.", "- Aim for 500 calls."])
        self.assertEqual(kept, "It pays. 90 / 2,400 = 3.75% quit.\n- Call the quiet ones.")

    def test_a_table_row_keeps_its_place_and_loses_only_the_number(self):
        """The review: a label row "quit within 60 days" was removed whole, and every check then passed on a table with no outcome."""
        windowed = GOOD.replace("the member quit afterwards", "the member quit within 60 days")
        kept, removed = checks.strip_unsaid(windowed, [GYM])
        self.assertIn("| quit | the member quit within [a number you did not give] days | after | label |", kept)
        self.assertEqual(len(removed), 1)
        self.assertEqual(self.failed(kept), [])  # the outcome row is still there, so the table is still sound
        late = GOOD.replace("| payment_stopped | the payment stopped | after | not used |", "| payment_30d | paid within 30 days | after | input |")
        self.assertIn("inputs known in time", self.failed(checks.strip_unsaid(late, [GYM])[0]))  # a fault is never stripped out of sight

    def test_a_result_is_sourced_only_while_its_sum_is_still_there(self):
        text = "The fees at stake are 108,000 a month.\n- 2,400 × 45 = 108,000 a month, about 30% of costs."
        self.assertEqual(checks.strip_unsaid(text, [GYM])[0], "")


class StanceTests(unittest.TestCase):
    def test_no_table_is_read_from_natural_sentences_in_english_and_persian(self):
        cases = {
            "We have not started logging sensor tables yet.": "none",
            "Because of HIPAA privacy rules we cannot upload patient records.": "private",
            "I would rather not hand over member details.": "private",
            "We don't have data ready to upload yet.": "none",
            "I do not know how to get anything out of the membership software.": "not_ready",
            "Nothing is in a system I could export for you.": "none",
            "به دلیل محرمانگی اطلاعات بانکی نمی‌توانیم دیتا آپلود کنیم.": "private",
            "هنوز دیتا ندارم و می‌خواهم ریزش را پیش‌بینی کنم.": "none",
            "There isn't any data so far.": "none",
            "We haven't collected anything.": "none",
            "I have nothing to upload.": "none",
            "The data must stay on our servers.": "private",
            "دیتایی در کار نیست.": "none",
        }
        for words, why in cases.items():
            out = stance.read(words)
            self.assertEqual((out["no_table"], out["why"]), (True, why), words)
            self.assertTrue(out["quote"] and out["quote"] in words, words)

    def test_a_person_with_a_table_or_with_nothing_said_is_not_read_as_having_none(self):
        self.assertFalse(stance.read("We cannot share patient names, but here is the table.", has_table=True)["no_table"])
        for words in ("Which customers will churn next month?", "We have no problem sharing the data.", "There is no missing data in the table.",
                      "I can't wait to share this dataset with you.", "Private customers churn more than business ones.",
                      "The NDA is signed, so I can share everything.", "The model ran without any data issues."):
            self.assertFalse(stance.read(words)["no_table"], words)


class SolveTests(unittest.TestCase):
    def test_without_a_model_the_rules_answer_and_say_so(self):
        for gateway in (None, Gateway(no_data_architect=Client(GOOD))):
            with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
                out = nodata.solve(SAAS, gateway=gateway)
            self.assertEqual((out["mode"], out["calls"], out["cost_eur"]), ("rules", [], 0.0))
            self.assertIn("written by DCLab's rules alone", nodata.markdown(out))

    def test_a_sound_answer_is_one_pass_and_one_second_reading(self):
        writer, reader = Client(GOOD), Client("- It never asks how often a call works.\n- It assumes 50% of calls work.")
        with live():
            out = nodata.solve(GYM, gateway=Gateway(no_data_architect=writer, no_data_critic=reader), draft_id="d1")
        self.assertEqual((out["mode"], out["need"], out["calls"], out["repaired"]), ("architect", "full", ["architect", "critic"], False))
        self.assertEqual(out["critic"], ["It never asks how often a call works."])  # a finding with a number nobody gave is dropped
        self.assertEqual(out["sums"]["verified"], ["90 / 2,400 = 3.75%"])
        self.assertEqual(out["questions"], ["How many calls can the trainers make in a week?"])
        self.assertTrue(all(c["ok"] for c in out["checks"]))
        self.assertIn(GYM, writer.seen[0][1]["content"])  # the person's words, whole
        self.assertEqual(writer.seen[0][0]["content"], prompts.text("no_data_architect"))
        shown = nodata.markdown(out)
        self.assertNotIn("Need:", shown)
        self.assertIn("## A second reader's findings", shown)
        self.assertIn("Checks DCLab will run when real data arrives", shown)
        self.assertEqual(economics.unsaid(shown, [GYM, *out["economics"]["as_said"]]), [])  # every number shown is the person's or code's
        self.assertEqual(out["economics"]["as_said"], ["90 / 2,400 = 3.75%"])  # code's sums and the checked ones only: never the design's own notes

    def test_a_failed_check_is_sent_back_once_and_never_twice(self):
        bad = GOOD.replace("| after | not used |", "| after | input |")
        writer = Client(bad, bad.replace("Call the members", "Phone the members"), GOOD)  # the rewrite still fails: no third request
        with live():
            out = nodata.solve(GYM, gateway=Gateway(no_data_architect=writer), second_reader=False)
        self.assertEqual((len(writer.seen), out["calls"], out["repaired"]), (2, ["architect", "architect_repair"], True))
        self.assertEqual(out["first_failed"], ["inputs known in time"])
        self.assertIn("payment_stopped", writer.seen[1][-1]["content"])  # the failed check, as it was found
        self.assertEqual([c["check"] for c in checks.failed(out["checks"])], ["inputs known in time"])
        self.assertIn("Not passed (inputs known in time)", nodata.markdown(out))  # what still fails is shown, never hidden

    def test_a_number_nobody_gave_is_removed_by_code_and_shown_as_removed(self):
        invented = GOOD.replace("not a warning.", "not a warning. Gyms usually lose 30% a year.")
        with live():
            out = nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(invented, invented)), second_reader=False)
        self.assertEqual(out["removed"], ["Gyms usually lose 30% a year."])
        self.assertIn("Code took out what stated a number you did not give", nodata.markdown(out))
        self.assertNotIn("30%", nodata.markdown(out))  # said to have gone, never quoted back to the person

    def test_a_worse_rewrite_never_replaces_a_better_draft(self):
        invented = GOOD.replace("not a warning.", "not a warning. Gyms usually lose 30% a year.")
        with live():
            out = nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(invented, "Sorry, here is the corrected answer.")), second_reader=False)
        self.assertEqual((out["repaired"], out["need"], len(out["plan"])), (False, "full", 3))
        self.assertTrue(all(c["ok"] for c in out["checks"]))

    def test_a_request_that_fails_never_loses_the_person_their_answer(self):
        bad = GOOD.replace("| after | not used |", "| after | input |")
        with live():
            down = nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(RuntimeError("timeout"))))
            later = nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(bad, RuntimeError("timeout"))), second_reader=False)
            with self.assertRaises(RuntimeError):  # a test arm sees the failure: a run's cap is never scored as the design's answer
                nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(RuntimeError("cap"))), on_error="raise")
        self.assertEqual((down["mode"], down["why"], down["errors"]), ("rules", "The model could not be reached.", ["RuntimeError"]))
        self.assertEqual((later["mode"], later["repaired"], later["errors"]), ("architect", False, ["RuntimeError"]))
        self.assertEqual(len(later["plan"]), 3)  # the draft stands

    def test_the_second_reader_may_number_its_findings_and_one_line_never_drops_the_others(self):
        reader = Client("1. It never asks how often a call works.\n2. No serious weakness found in the table.\n- The plan has no owner.")
        with live():
            out = nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(GOOD), no_data_critic=reader))
        self.assertEqual(out["critic"], ["It never asks how often a call works.", "The plan has no owner."])

    def test_a_persian_speaker_reads_persian_headings_and_the_checks_still_read_the_answer(self):
        words = "هنوز دیتا ندارم و می‌خواهم بدون آپلود فایل، ریزش مشتریان را پیش‌بینی کنم. ۲٬۴۰۰ مشترک داریم و ۹۰ نفر در ماه می‌روند."
        answer = GOOD.replace("Who the trainers call first. 90 / 2,400 = 3.75% of members quit a month.", "چه کسی اول تماس بگیرد. ۹۰ / ۲٬۴۰۰ = ۳٫۷۵٪ در ماه می‌روند.")
        with live():
            out = nodata.solve(words, gateway=Gateway(no_data_architect=Client(answer)), second_reader=False)
        self.assertTrue(all(c["ok"] for c in out["checks"]), [c for c in out["checks"] if not c["ok"]])
        shown = nodata.markdown(out)
        self.assertIn("## برنامه", shown)
        self.assertIn("| ستون | چیست | کی معلوم است | نقش |", shown)
        self.assertNotIn("## Plan", shown)
        self.assertIn("## Plan", nodata.markdown({**out, "words": GYM}))  # an English speaker's answer keeps its headings

    def test_a_business_question_gets_a_business_answer_with_no_model(self):
        answer = re.sub(r"## (?:Data to record|Model)\n.*?\n\n(?=## )", "", GOOD.replace("Need: full", "Need: business"), flags=re.S)
        with live():
            out = nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(answer)), second_reader=False)
        self.assertEqual((out["need"], out["kind"], out["columns"], out["planner"]), ("business", None, [], {}))
        self.assertNotIn("Model", [p["title"] for p in nodata.card(out)["parts"]])
        self.assertTrue(all(c["ok"] for c in out["checks"]))

    def test_a_model_that_does_not_answer_or_a_reader_that_fails_never_loses_the_person_their_answer(self):
        with live():
            silent = nodata.solve(SAAS, gateway=Gateway(no_data_architect=Client("")))
            broken = nodata.solve(GYM, gateway=Gateway(no_data_architect=Client(GOOD), no_data_critic=Client(RuntimeError("down"))))
        self.assertEqual((silent["mode"], silent["why"]), ("rules", "The model did not answer."))
        self.assertEqual((broken["mode"], broken["critic"]), ("architect", []))

    def test_the_cap_stops_the_repair_and_the_second_reading(self):
        bad = GOOD.replace("| after | not used |", "| after | input |")
        writer, reader = Client(bad, GOOD, cost=0.2), Client("- A finding.")
        with live():
            out = nodata.solve(GYM, gateway=Gateway(no_data_architect=writer, no_data_critic=reader), cap_eur=0.1)
        self.assertEqual((out["calls"], len(reader.seen)), (["architect"], 0))

    def test_nothing_in_the_package_creates_data(self):
        source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "dclab_rnd" / "nodata").glob("*.py"))
        self.assertIsNone(re.search(r"import synthetic\b|synthetic\.generate|\.generate\(|to_csv|to_parquet|DataFrame\(", source))
        out = {"words": GYM, "questions": ["How many calls can the trainers make in a week?"], "columns": checks.columns(checks.sections(GOOD)["Data to record"])}
        asked = simulate.questions(out)
        self.assertEqual(asked[0], "How many calls can the trainers make in a week?")
        self.assertTrue(any("How many rows" in q for q in asked))
        self.assertIn("member_id", simulate.description(out, ["two years", "visits drive it"]))

    def test_the_first_design_is_untouched_by_the_rebuild(self):
        from dclab_rnd.draft import no_data

        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            old = no_data.solve_without_data(SAAS, gateway=None)
        self.assertEqual(old["harness_report"]["mode"], "deterministic_floor")


CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "DCLAB_NO_LIVE_MODELS",
                         "DCLAB_THINKING")} | {"OPENAI_API_KEY": "k"}  # a configured tier with a scripted transport: no request leaves the test


def reply(text, why):
    return {"content": text, "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}, "thinking": [{"text": why}],
            "assistant_message": {"role": "assistant", "content": text}}


class Scripted:
    def __init__(self, replies):
        self.replies, self.thinking = list(replies), None

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        return self.replies.pop(0)


class ThinkingKeptTests(unittest.TestCase):
    def test_the_architect_and_the_second_reader_keep_their_thinking_whole(self):
        """Rule 8: a new purpose's thinking is stored. Both are shown the person's words only (no cell), so it is kept whole."""
        home = Path(tempfile.mkdtemp())
        by_purpose = {"no_data_architect": Scripted([reply(GOOD, "why the architect chose this")]),
                      "no_data_critic": Scripted([reply("No serious weakness found.", "why the reader found nothing")])}
        names = {id(v): k for k, v in settings.PURPOSES.items()}
        with mock.patch.object(gw, "_transport", lambda tier, purpose: by_purpose[names[id(purpose)]]), \
                mock.patch.dict(os.environ, {**CLEAN, "DCLAB_AGENT_HOME": str(home)}), \
                mock.patch.object(experience, "current_log", lambda: None):
            out = nodata.solve(GYM, gateway=gw.Gateway(transport=gw._transport))
        self.assertEqual(out["calls"], ["architect", "critic"])
        docs = [json.loads(line) for p in sorted(home.rglob("model_exchanges/**/*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines()]
        kept = {d["purpose"]: d for d in docs}
        self.assertEqual(set(kept), {"no_data_architect", "no_data_critic"})
        self.assertIn("why the architect chose this", json.dumps(kept["no_data_architect"].get("thinking")))
        self.assertIn("why the reader found nothing", json.dumps(kept["no_data_critic"].get("thinking")))
        self.assertTrue(all(d["whole"] for d in docs))


if __name__ == "__main__":
    unittest.main()

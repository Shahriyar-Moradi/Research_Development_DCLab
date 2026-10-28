"""Persian and English without a model (package H3): the normalisation, the Persian tables, quotes in the person's
original script, and English unchanged.

The Persian sentences here are written for these tests; the dev requests of the H1 suite are read too (the dev split is
for building against; the sealed split is never read here).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft import intent, lang, plan  # noqa: E402
from dclab_rnd.draft.chat import BUILD_WORDS, SIMULATE_WORDS, HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

ZWNJ = "‌"


class NormaliseTests(unittest.TestCase):
    def test_letters_digits_zwnj_and_marks_fold_and_map_back(self):
        text = "هزينهٔ ماهانهٔ مشتري" + ZWNJ + "ها در ۲۰۲۵ و ٣ ماه كاري"
        norm, index = lang.normalize(text)
        self.assertEqual(norm, "هزینه ماهانه مشتری ها در 2025 و 3 ماه کاری")
        self.assertEqual(len(norm), len(index))
        start = norm.index("2025")
        a, b = lang.span(text, index, start, start + 4)
        self.assertEqual(text[a:b], "۲۰۲۵")  # the person's digits, not the folded ones

    def test_a_span_grows_to_whole_words_across_a_zwnj(self):
        text = "پیش" + ZWNJ + "بینی کنید"
        norm, index = lang.normalize(text)
        a, b = lang.span(text, index, norm.index("بینی"), norm.index("بینی") + 2)
        self.assertEqual(text[a:b], "پیش" + ZWNJ + "بینی")

    def test_the_script_of_a_text(self):
        self.assertEqual(lang.script("چرا فروش کم شد؟"), "fa")
        self.assertEqual(lang.script("Why did sales drop?"), "en")
        self.assertEqual(lang.script(""), "en")

    def test_the_quote_rule_wants_the_persons_exact_words(self):
        text = "می" + ZWNJ + "خواهم پیش" + ZWNJ + "بینی کنم کدام مشتری" + ZWNJ + "ها ظرف 30 روز آینده می" + ZWNJ + "خرند"
        self.assertTrue(plan.quoted("پیش" + ZWNJ + "بینی کنم", [text]))
        self.assertFalse(plan.quoted("پیش بینی کنم", [text]))  # a space where the person wrote a ZWNJ is not their words
        self.assertFalse(plan.quoted("پيش" + ZWNJ + "بيني کنم", [text]))  # nor Arabic yeh
        self.assertFalse(plan.quoted("۳۰ روز آینده", [text]))  # nor other digits
        self.assertFalse(plan.quoted("می" + ZWNJ + "خرند", [text]))  # one word with a ZWNJ is still one word
        self.assertFalse(plan.quoted("بینی", [text]))

    def test_one_index_entry_per_character(self):
        norm, index = lang.normalize("İ İ Which customers")
        self.assertEqual(len(norm), len(index))
        self.assertEqual(lang.normalize("a\nb")[0], "a\nb")  # a line break stays


class IntentTests(unittest.TestCase):
    PERSIAN = {
        "prediction": ["پیش" + ZWNJ + "بینی کن کدام کاربرها ماه آینده خرید می" + ZWNJ + "کنند", "فروش هفتهٔ بعد هر شعبه را تخمین بزن"],
        "analysis": ["میانگین مبلغ سفارش" + ZWNJ + "ها به تفکیک شهر چقدر است؟", "پارسال چند تا مشتری جدید داشتیم؟"],
        "preparation": ["دو فایل مشتری را ادغام کن و تکراری" + ZWNJ + "ها را حذف کن", "ستون تاریخ را به ماه تبدیل کن"],
        "reporting": ["یک داشبورد هفتگی از درآمد هر کانال می" + ZWNJ + "خواهم", "گزارش ماهانهٔ بازگشت وجه برای مدیر مالی"],
        "business": ["چرا درآمد فروردین کم شد؟", "تخفیف ارسال رایگان را برداریم؟ چه کار باید بکنیم؟"],
        "advice": ["فرق دقت و بازیابی چیست؟", "برای داده" + ZWNJ + "های نامتوازن چه معیاری بهتر است؟"],
    }

    def test_each_track_in_persian(self):
        for track, sentences in self.PERSIAN.items():
            for sentence in sentences:
                reading = intent.read(sentence)
                self.assertEqual((reading["track"], reading["status"]), (track, "stated"), sentence)
                self.assertTrue(plan.quoted(reading["quote"], [sentence]), (sentence, reading["quote"]))
                self.assertIn(reading["quote"], sentence)  # cut from the person's text, ZWNJ and digits kept

    def test_the_same_request_in_other_spellings_reads_the_same(self):
        base = "پیش" + ZWNJ + "بینی کن کدام مشتری" + ZWNJ + "ها ظرف ۳۰ روز لغو می" + ZWNJ + "کنند"
        for variant in (base, base.replace(ZWNJ, " "), base.replace("ی", "ي").replace("ک", "ك"), base.replace("۳۰", "30")):
            self.assertEqual(intent.read(variant)["track"], "prediction", variant)

    def test_ordinary_persian_predictions_stay_predictions(self):
        for sentence in ("پیش" + ZWNJ + "بینی کن کدام مشتری" + ZWNJ + "ها با پلن" + ZWNJ + "های متفاوت لغو می" + ZWNJ + "کنند",
                         "فرقی نمی" + ZWNJ + "کند چه الگوریتمی، فقط می" + ZWNJ + "خواهم بدانم کدام مشتری" + ZWNJ + "ها لغو می" + ZWNJ + "کنند",
                         "بر اساس این مجموعه داده، کدام مشتری" + ZWNJ + "ها لغو می" + ZWNJ + "کنند؟",
                         "کدام مشتری" + ZWNJ + "ها برای پیشنهاد تخفیف مناسب است؟",
                         "مشتری جدید چقدر خرید خواهد کرد؟",
                         "مدلی بساز که هر هفته پیش" + ZWNJ + "بینی کند کدام مشتری" + ZWNJ + "ها لغو می" + ZWNJ + "کنند و نتیجه را گزارش کند"):
            reading = intent.read(sentence)
            self.assertEqual(reading["track"], "prediction", sentence)
            self.assertNotEqual(reading["status"], "asked", sentence)

    def test_a_past_share_is_an_analysis(self):
        self.assertEqual(intent.read("چند درصد مشتری" + ZWNJ + "ها در ماه بعد از اولین خرید دوباره خرید کردند؟")["track"], "analysis")

    def test_a_newline_still_ends_an_english_rule(self):
        self.assertIn("how_much", intent.read("How many orders did we have last month?\nDo we need a model for that?")["signals"])

    def test_they_leave_is_not_a_trend(self):
        self.assertNotIn("fa_trend", intent.read("چرا مشتری" + ZWNJ + "ها می" + ZWNJ + "روند؟")["signals"])

    def test_persian_dev_requests_route_as_well_as_english_ones(self):
        cases = json.loads((ROOT / "dclab_rnd/agent_eval/home/requests_dev.json").read_text())

        def accuracy(lang_):
            mine = [c for c in cases if c["lang"] == lang_]
            return sum(intent.read(c["text"])["track"] == c["gold"]["tracks"][0] and intent.read(c["text"])["status"] != "asked"
                       for c in mine) / len(mine)
        self.assertGreaterEqual(accuracy("fa"), accuracy("en"))
        self.assertGreaterEqual(accuracy("mixed"), accuracy("en"))

    def test_english_readings_are_unchanged(self):
        for sentence, track in (("Predict which subscribers will cancel next quarter", "prediction"),
                                ("Why did profit drop in the second quarter?", "business"),
                                ("Remove duplicate rows", "preparation"),
                                ("Build a dashboard of daily signups for the team", "reporting")):
            reading = intent.read(sentence)
            self.assertEqual(reading["track"], track)
            self.assertFalse([s for s in reading["signals"] if s.startswith("fa_")])


class PlanTests(unittest.TestCase):
    def test_a_persian_outcome_with_its_window(self):
        text = "می" + ZWNJ + "خواهم پیش" + ZWNJ + "بینی کنم کدام مشتری" + ZWNJ + "ها ظرف ۳۰ روز آینده اشتراکشان را لغو می" + ZWNJ + "کنند تا تیم فروش تماس بگیرد."
        found = plan.extract(text)
        self.assertEqual(found["target"], "کدام مشتری" + ZWNJ + "ها ظرف ۳۰ روز آینده اشتراکشان را لغو می" + ZWNJ + "کنند")
        self.assertNotIn("prediction_moment", found)  # no moment set apart: nothing claimed
        self.assertTrue(plan.quoted(found["target"], [text]))

    def test_a_persian_moment_set_apart_by_a_comma(self):
        text = "پیش" + ZWNJ + "بینی کن کدام سفارش" + ZWNJ + "ها ظرف ۲ ساعت لغو می" + ZWNJ + "شوند، در لحظهٔ ثبت سفارش"
        found = plan.extract(text)
        self.assertEqual(found["prediction_moment"], "در لحظهٔ ثبت سفارش")
        self.assertTrue(plan.quoted(found["prediction_moment"], [text]))

    def test_estimate_and_number_word_moments(self):
        self.assertIn("هفته آینده", plan.extract("تخمین بزن چند سفارش در هفته آینده ثبت می" + ZWNJ + "شود")["target"])
        found = plan.extract("پیش" + ZWNJ + "بینی کن کدام رزروها ظرف ۷ روز لغو می" + ZWNJ + "شوند، دو روز قبل از ورود")
        self.assertEqual(found["prediction_moment"], "دو روز قبل از ورود")

    def test_persian_words_can_support_a_value(self):
        self.assertTrue(plan.supports("کدام مشتری" + ZWNJ + "ها لغو می" + ZWNJ + "کنند", "مشتری‌هایی که لغو می‌کنند"))

    def test_no_window_no_outcome(self):
        self.assertEqual(plan.extract("پیش" + ZWNJ + "بینی کن کدام مشتری" + ZWNJ + "ها لغو می" + ZWNJ + "کنند"), {})

    def test_english_extraction_is_unchanged(self):
        found = plan.extract("Predict which customers will cancel within 30 days, scored on the first day of each month")
        self.assertEqual(found, {"target": "which customers will cancel within 30 days", "prediction_moment": "on the first day of each month"})


class ChatWordTests(unittest.TestCase):
    def test_build_and_simulate_words_in_both_languages(self):
        for text in ("let's build", "بریم سراغ ساخت", "راه" + ZWNJ + "حل را بساز", "شروع کنیم"):
            self.assertTrue(BUILD_WORDS.search(text), text)
        self.assertFalse(BUILD_WORDS.search("یک pivot بساز"))  # making a table is not building the solution
        self.assertFalse(BUILD_WORDS.search("از کجا شروع کنیم؟"))
        self.assertFalse(BUILD_WORDS.search("قبل از اینکه شروع کنیم، ستون مبلغ نهایی بعد از رزرو پر می" + ZWNJ + "شود"))
        for text in ("simulate it", "داده" + ZWNJ + "ی مصنوعی بساز", "شبیه" + ZWNJ + "سازی کن"):
            self.assertTrue(SIMULATE_WORDS.search(text), text)

    def test_a_persian_request_with_no_model_gets_its_track_and_outcome(self):
        store = DraftStore(Path(tempfile.mkdtemp()))
        d = store.create("می" + ZWNJ + "خواهم پیش" + ZWNJ + "بینی کنم کدام مشتری" + ZWNJ + "ها ظرف ۳۰ روز آینده اشتراکشان را لغو می" + ZWNJ + "کنند")
        HomeAgent(store, None).start(d["id"])
        d = store.get(d["id"])
        self.assertEqual(d["intent"]["track"], "prediction")
        self.assertIn("لغو", d["understanding"]["target"])
        a = store.create("چرا فروش منطقهٔ شمال در بهار کم شد؟")
        HomeAgent(store, None).start(a["id"])
        a = store.get(a["id"])
        self.assertEqual(a["intent"]["track"], "business")
        self.assertFalse(a.get("workflow"))


if __name__ == "__main__":
    unittest.main()

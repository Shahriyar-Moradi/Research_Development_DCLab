"""Persian beside English with no model (package H3): the normalization, the intent words, the plan's outcome and
moment, the chat's word checks (build, simulate, a bare yes, "I don't know"), and a Persian request routed by Home.

The sentences here are written for these tests; none is a request of the H1 suite (dev or sealed).
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft import chat, intent, lang, plan  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

ZWNJ = "‌"
BY_TRACK = {
    "prediction": [f"می{ZWNJ}خواهیم پیش{ZWNJ}بینی کنیم کدام بیمارها ظرف یک هفته دوباره بستری می{ZWNJ}شوند",
                   "پیش بینی فروش هفتهٔ بعد هر شعبه را لازم داریم"],
    "analysis": ["میانگین زمان پاسخ تیکت‌ها در هر تیم چقدر است؟",
                 "تعداد ثبت‌نام‌های هر شهر را به تفکیک ماه نشان بده"],
    "preparation": ["دو فایل مشتری را ادغام کن و ردیف‌های تکراری را حذف کن",
                    "ستون مبلغ را به عدد تبدیل کن"],
    "reporting": ["یک داشبورد برای فروش هفتگی هر فروشگاه می‌خواهم",
                  "گزارش ماهانهٔ بازگشت کالا برای مدیر مالی لازم است"],
    "business": ["چرا سود ما در پاییز کم شد؟", "مشتری‌ها کمتر خرید می‌کنند، چه کار باید بکنیم؟"],
    "advice": ["فرق دقت و بازیابی با هم چیست؟", "برای تشخیص تقلب کدام معیار بهتر است؟"],
}


class NormalizeTests(unittest.TestCase):
    def test_arabic_letters_digits_joiners_and_marks_read_as_persian(self):
        self.assertEqual(lang.normalize("كيك"), "کیک")  # Arabic kaf and yeh
        self.assertEqual(lang.normalize("۳۰ و ٤٥"), "30 و 45")  # Persian and Arabic-Indic digits, for matching only
        self.assertEqual(lang.normalize(f"پیش{ZWNJ}بینی"), "پیش بینی")
        self.assertEqual(lang.normalize("هزینهٔ"), "هزینه")  # the hamza above is a mark
        self.assertEqual(lang.normalize("حتماً"), "حتما")
        self.assertEqual(lang.normalize("Predict churn"), "Predict churn")  # English is untouched

    def test_a_span_of_the_normalized_text_is_a_span_of_the_persons(self):
        text = "میانگینِ هزینهٔ ماهانه"
        norm, at = lang.normalize_map(text)
        start = norm.index("هزینه ماهانه")
        a, b = lang.span(at, start, start + len("هزینه ماهانه"))
        self.assertEqual(text[a:b], "هزینهٔ ماهانه")  # the quote keeps the person's spelling, mark included


class IntentTests(unittest.TestCase):
    def test_each_track_is_read_from_ordinary_persian_requests(self):
        for track, sentences in BY_TRACK.items():
            for sentence in sentences:
                reading = intent.read(sentence)
                self.assertEqual((reading["track"], reading["source"]), (track, "rules"), sentence)
                self.assertIn(reading["quote"], sentence)  # the person's own script, joiners and all
                self.assertTrue(plan.quoted(reading["quote"], [sentence]), reading["quote"])

    def test_probably_is_not_a_prediction(self):
        reading = intent.read("احتمالاً فروش کم شده؛ چرا؟")
        self.assertEqual(reading["track"], "business")
        self.assertNotIn("fa_likely", intent.rules("احتمالاً فروش کم شده")["signals"])

    def test_no_persian_signal_fires_on_english(self):
        cases = json.loads((ROOT / "dclab_rnd/agent_eval/home/requests_dev.json").read_text(encoding="utf-8"))
        persian_ids = {sid for _, sid, _, _ in lang.SIGNALS}
        for case in cases:
            if case["lang"] == "en":
                self.assertFalse(persian_ids & set(intent.rules(case["text"])["signals"]), case["id"])

    def test_english_readings_are_unchanged(self):
        self.assertEqual(intent.read("What is the difference between precision and recall?")["track"], "advice")
        self.assertEqual(intent.read("Help with my customers")["source"], "default")


class PlanTests(unittest.TestCase):
    def test_a_persian_outcome_and_moment_are_quoted_from_the_persons_words(self):
        text = f"هر دوشنبه صبح، پیش{ZWNJ}بینی کن کدام راننده{ZWNJ}ها ظرف ۷ روز آینده استعفا می{ZWNJ}دهند."
        found = plan.extract(text)
        self.assertEqual(found["target"], f"کدام راننده{ZWNJ}ها ظرف ۷ روز آینده استعفا می{ZWNJ}دهند")
        self.assertEqual(found["prediction_moment"], "هر دوشنبه صبح")
        for value in found.values():
            self.assertIn(value, text)
            self.assertTrue(plan.quoted(value, [text]))

    def test_a_moment_before_an_event(self):
        found = plan.extract("کدام سفارش‌ها در ۳ روز آینده مرجوع می‌شوند، ۲ روز قبل از ارسال")
        self.assertEqual(found["prediction_moment"], "۲ روز قبل از ارسال")
        self.assertNotIn("،", found["target"])

    def test_no_outcome_without_a_window(self):
        self.assertNotIn("target", plan.extract("کدام مشتری‌ها لغو می‌کنند"))  # as in English: a window is needed

    def test_english_extract_is_unchanged(self):
        found = plan.extract("Predict which customers will cancel within 30 days, scored on the first day of each month.")
        self.assertEqual(found, {"target": "which customers will cancel within 30 days",
                                 "prediction_moment": "on the first day of each month"})

    def test_quotes_and_support_compare_persian_spellings(self):
        self.assertTrue(plan.quoted("پیش بینی کنیم", [f"می{ZWNJ}خواهیم پیش{ZWNJ}بینی کنیم"]))
        self.assertTrue(plan.quoted("كدام مشتري", ["کدام مشتری لغو می‌کند"]))  # Arabic letters in the quote
        self.assertTrue(plan.supports("هزینهٔ ماهانه", "هزینه ماهانه هر مشتری"))
        self.assertFalse(plan.quoted("مشتری", ["مشتری‌ها"]))  # still whole words, at least eight characters or two words


class WordTests(unittest.TestCase):
    def test_not_known_in_persian(self):
        for said in ("نمی‌دانم", "نمیدانم.", "مطمئن نیستم", "نمی دونم!", "اطلاعی ندارم"):
            self.assertTrue(chat.not_known(said), said)
        self.assertFalse(chat.not_known("نمی‌دانم چرا فروش کم شد"))  # a sentence that says more is an answer

    def test_build_orders_and_not_orders(self):
        for said in ("بسازش", "برو جلو", "پروژه را بساز", "شروع کن", "باشه، بریم جلو"):
            self.assertTrue(chat.build_order(said), said)
            self.assertTrue(chat.BUILD_WORDS.search(said), said)
        for said in ("یک داشبورد بساز", "pivot بساز", "آیا باید مدل بسازیم؟", "مدل را کی بسازیم؟"):
            self.assertFalse(chat.build_order(said), said)
        self.assertFalse(chat.BUILD_WORDS.search("یک داشبورد بساز"))
        self.assertTrue(chat.build_order("yes build it"))  # English unchanged

    def test_a_persian_yes_after_an_offer_builds(self):
        offer = {"role": "agent", "kind": "summary", "text": "Shall I build it?"}
        for said in ("بله", "آره", "حتماً", "باشه."):
            self.assertTrue(chat.asks_to_build([offer, {"role": "user", "text": said}]), said)
        self.assertFalse(chat.asks_to_build([offer, {"role": "user", "text": "بله ولی اول ستون‌ها را ببینم"}]))
        self.assertTrue(chat.CONFIRMED.match("درسته"))

    def test_simulate_words(self):
        self.assertTrue(chat.SIMULATE_WORDS.search("داده ندارم، شبیه‌سازی کن"))
        self.assertTrue(chat.SIMULATE_WORDS.search("داده ساختگی بساز"))
        self.assertFalse(chat.SIMULATE_WORDS.search("داده‌ها را ادغام کن"))


SIGNAL_EXAMPLES = {  # one ordinary sentence per Persian signal, each must fire it (written for this test)
    "fa_predict": "پیش‌بینی فروش هفته بعد", "fa_which_will": "کدام کارمندها استعفا خواهند داد",
    "fa_window": "ظرف ۱۴ روز چه می‌شود", "fa_likely": "احتمال خرید هر مشتری را بگو",
    "fa_model": "یک مدلی بساز برای ریزش", "fa_how_much": "درآمد پارسال چقدر بود؟", "fa_count": "تعداد کاربران فعال",
    "fa_aggregate": "مجموع فروش", "fa_analyse": "داده‌های فروش را تحلیل کن", "fa_breakdown": "فروش بر حسب شهر", "fa_show": "سفارش‌ها را نشان بده",
    "fa_compare": "مقایسه دو فروشگاه", "fa_trend": "روند فروش امسال", "fa_convert": "به فرمت تاریخ ببر",
    "fa_remove": "ردیف‌های خالی را حذف کنید", "fa_merge": "دو جدول را ادغام کن", "fa_clean": "پاک‌سازی داده‌های مشتری",
    "fa_duplicates": "رکوردهای تکراری", "fa_dashboard": "یک داشبورد لازم دارم", "fa_periodic": "گزارش هفتگی فروش",
    "fa_report": "یک گزارش از فروش", "fa_kpi": "شاخص رضایت مشتری", "fa_why": "چرا سود کم شد",
    "fa_what_to_do": "چه کار کنیم", "fa_difference": "تفاوت میانگین و میانه چیست", "fa_which_metric": "کدام معیار بهتر است",
    "fa_meaning": "بیش‌برازش یعنی چه",
}


class ReviewOfH3Tests(unittest.TestCase):
    """The review's findings (H3), each with the sentence that showed it."""

    def test_every_persian_signal_fires_on_an_ordinary_sentence(self):
        self.assertEqual(set(SIGNAL_EXAMPLES), {sid for _, sid, _, _ in lang.SIGNALS})
        for sid, sentence in SIGNAL_EXAMPLES.items():
            self.assertIn(sid, intent.rules(sentence)["signals"], sentence)

    def test_a_word_inside_another_word_is_not_a_signal(self):
        cases = {"چند تاریخ مختلف": "fa_how_much", "چند استان": "fa_how_much", "مشتری‌ها می‌روند": "fa_trend",
                 "دیروز بعد از ظهر": "fa_window", "فروش ماه بعد از کمپین": "fa_window",
                 "کدام مشتری‌ها می‌خواهند بخرند": "fa_which_will"}
        for sentence, sid in cases.items():
            self.assertNotIn(sid, intent.rules(sentence)["signals"], sentence)
        self.assertEqual(intent.read("داده‌ها از چند تاریخ مختلف است و می‌خواهم مدلی بسازم که مرجوعی‌ها را تشخیص دهد")["track"], "prediction")
        self.assertEqual(intent.read("دیروز بعد از ظهر چند تا سفارش داشتیم؟")["track"], "analysis")

    def test_an_answer_that_mentions_starting_is_not_an_order_to_build(self):
        for said in ("وقتی مشتری اولین خرید را شروع کند", "نه، اول داده را ببینیم بعد شروع کنیم", "روبرو جلوی فروشگاه",
                     "پروژه‌ها را بعدا شروع کنیم؟"):
            self.assertFalse(chat.build_order(said), said)
            self.assertFalse(chat.BUILD_WORDS.search(said), said)

    def test_that_is_right_confirms_but_never_builds(self):
        offer = {"role": "agent", "kind": "summary", "text": "Shall I build it?"}
        self.assertTrue(chat.CONFIRMED.match("درسته"))
        self.assertFalse(chat.asks_to_build([offer, {"role": "user", "text": "درسته"}]))

    def test_a_moment_inside_another_word_or_a_refresh_is_not_a_moment(self):
        for text in ("کدام سفارش‌ها ظرف ۷ روز آینده مرجوع می‌شوند؛ داده از مهر ماه است",
                     "سازمان خیریه هستیم و کدام اهداکننده‌ها ظرف ۳۰ روز آینده کمک می‌کنند",
                     "کدام سفارش‌ها ظرف ۷ روز آینده مرجوع می‌شوند، داده‌ها هر ماه به‌روز می‌شوند",
                     "پیش‌بینی ظهر روز سفارش: کدام سفارش‌ها ظرف ۳ روز آینده مرجوع می‌شوند"):
            self.assertNotIn("prediction_moment", plan.extract(text), text)

    def test_a_persian_clause_never_erases_the_english_outcome(self):
        found = plan.extract("Predict which customers will cancel within 30 days. داده‌ها هر ماه به‌روز می‌شوند")
        self.assertEqual(found, {"target": "which customers will cancel within 30 days"})
        found = plan.extract("Predict which customers will cancel within 30 days، هر دوشنبه صبح")
        self.assertEqual(found["target"], "which customers will cancel within 30 days")
        self.assertEqual(found["prediction_moment"], "هر دوشنبه صبح")

    def test_persian_function_words_are_no_support(self):
        self.assertFalse(plan.supports("کدام مشتری‌ها ظرف ۳۰ روز آینده لغو می‌کنند", "کدام ستون"))
        self.assertFalse(plan.supports("برای این ماه", "برای آن سال"))
        self.assertTrue(plan.supports("لغو اشتراک", "اشتراک لغو شده"))

    def test_a_one_word_signal_at_the_end_now_makes_a_quote(self):
        # a deliberate change to English (H3): the quote grows back when the signal ends the text, so "Can you pivot"
        # reads as preparation from the person's words instead of falling to the default
        reading = intent.read("Can you pivot")
        self.assertEqual((reading["track"], reading["source"]), ("preparation", "rules"))
        self.assertTrue(plan.quoted(reading["quote"], ["Can you pivot"]))


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home)
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def test_a_persian_analysis_request_is_routed_with_no_model(self):
        d = self.store.create(BY_TRACK["analysis"][1])
        HomeAgent(self.store, None).start(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual((d["intent"]["track"], d["intent"]["source"]), ("analysis", "rules"))
        self.assertIn(d["intent"]["quote"], d["problem"])
        self.assertIsNone(HomeAgent(self.store, None).pending(d))  # no prediction question off a prediction

    def test_a_persian_prediction_keeps_its_outcome(self):
        text = f"پیش{ZWNJ}بینی کن کدام مشتری{ZWNJ}ها ظرف ۳۰ روز آینده خریدشان را تمدید نمی{ZWNJ}کنند"
        d = self.store.create(text)
        HomeAgent(self.store, None).start(d["id"])
        d = self.store.get(d["id"])
        self.assertEqual(d["intent"]["track"], "prediction")
        self.assertIn("ظرف ۳۰ روز", d["understanding"]["target"])
        self.assertIn(d["understanding"]["target"], text)


if __name__ == "__main__":
    unittest.main()

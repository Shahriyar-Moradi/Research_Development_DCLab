"""Persian and English read without a model (package H3).

The no-model path of Home read English only: a Persian request fell to "prediction, the usual start" (HOME-002: Persian
0.125). This module makes Persian a first language of the rules, beside English:

1. ``normalize``: Arabic yeh and alef maksura become Persian yeh, Arabic kaf Persian kaf, teh marbuta heh; Persian and
   Arabic-Indic digits become 0-9 (for matching only); the zero-width non-joiner and joiner become a space, so
   "پیش‌بینی" and "پیش بینی" read alike; diacritics and the tatweel are dropped. ``normalize_map`` keeps, for every
   character, where it came from, so a match is quoted from the person's own text, never from the normalized one.
2. ``SIGNALS``: Persian phrases per track, weighted as the English ones in ``intent.py`` (a strong phrase 3, a medium 2;
   a question about the difference between two things 4, as "how much data do I need" is in English), written from the
   H1 dev requests and ordinary Persian, never from the sealed set.
3. ``NOT_KNOWN``, ``BUILD``, ``ASSENT``, ``CONFIRMED`` and ``SIMULATE``: a person who does not know, an order to build (the
   whole message), a bare yes, a yes to what Home understood, a request for simulated data; ``TARGET`` and ``MOMENTS``: the plan's outcome and prediction moment (``plan.extract``).

Arabic is a later table (the plan's owner decision 5).
"""

from __future__ import annotations

import re

_ONE_FOR_ONE = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه",
    "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ",
    **{chr(0x06F0 + i): str(i) for i in range(10)},  # Persian digits
    **{chr(0x0660 + i): str(i) for i in range(10)},  # Arabic-Indic digits
})
# dropped: the short vowels and other marks (U+064B..U+065F), the superscript alef and the tatweel, so "هزینهٔ" reads "هزینه"
_DROPPED = frozenset(chr(c) for c in [*range(0x064B, 0x0660), 0x0670, 0x0640])


def normalize_map(text: str) -> tuple[str, list[int]]:
    """The text for matching and, for each of its characters, the position of the person's character it came from (one
    more entry at the end, the text's length), so a match's span maps back to the person's own words."""
    text = str(text or "")
    out: list[str] = []
    at: list[int] = []
    for i, ch in enumerate(text):
        if ch in _DROPPED:
            continue
        out.append(ch.translate(_ONE_FOR_ONE))
        at.append(i)
    at.append(len(text))
    return "".join(out), at


def normalize(text: str) -> str:
    """The text for matching (see the module docstring); compare normalized with normalized, quote from the original."""
    return normalize_map(text)[0]


def span(at: list[int], start: int, end: int) -> tuple[int, int]:
    """A match's span in the normalized text, as a span of the person's text."""
    return at[start], (at[end - 1] + 1 if end > start else at[start])


def has_persian(text: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", str(text or "")))


_S = r"\s*"  # "می‌کنند", "می کنند" and "میکنند" all read: the joiner became a space, or there was none
_MI = rf"می{_S}"
_L = "[\u0621-\u063A\u0641-\u064A\u0671-\u06D3\u06FA-\u06FF]"  # Persian and Arabic letters, never "؟", "،" or "؛"
_B = rf"(?<!{_L})"  # a Persian word starts here ("هر" is not the end of "مهر", "زمان" not the end of "سازمان")
_E = rf"(?!{_L})"  # and ends here ("چند تا" is not the start of "چند تاریخ")
_NOT_MI = r"(?<!می )(?<!می)"  # not the end of a present-tense verb ("روند" in "می‌روند", "خواهند" in "می‌خواهند")
SIGNALS: list[tuple[str, str, int, str]] = [
    # (track, id, weight, pattern) on normalized text: Persian, written from the H1 dev requests and ordinary Persian
    ("prediction", "fa_predict", 3, rf"{_B}پیش{_S}بینی"),
    ("prediction", "fa_which_will", 2, rf"{_B}کدام\s.{{0,60}}?({_MI}(شوند|کنند|شود|کند|روند|رود|خرند|خرد){_E}|{_NOT_MI}{_B}(خواهند|خواهد){_E})"),
    ("prediction", "fa_window", 2, rf"{_B}ظرف\s+\d+\s+(روز|هفته|ماه|سال){_E}|{_B}(روز|هفته|ماه|سال)\s+(آینده|بعدی?){_E}(?!\s+از{_E})"),
    ("prediction", "fa_likely", 2, rf"{_B}احتمال(?![اً])"),  # not "احتمالاً" (probably), an everyday hedge
    ("prediction", "fa_model", 3, rf"{_B}مدل(ی)?\s.{{0,25}}?(بساز|آموزش|{_MI}خواهم)"),
    ("analysis", "fa_how_much", 3, rf"{_B}چقدر\s*(بود|است|هست|شد){_E}|{_B}چند\s*(تا|تاست|بود|است|عدد){_E}"),
    ("analysis", "fa_count", 2, rf"{_B}تعداد{_E}"),
    ("analysis", "fa_analyse", 3, rf"{_B}(تحلیل|آنالیز|آمار){_E}"),  # H4: "analyse", "statistics", asked outright
    ("analysis", "fa_aggregate", 2, rf"{_B}(میانگین|مجموع|جمع\s+کل|بیشترین|کمترین|متوسط){_E}"),
    ("analysis", "fa_breakdown", 2, rf"{_B}(به\s*تفکیک|بر\s*حسب){_E}"),
    ("analysis", "fa_show", 2, rf"{_B}نشان\s*(بده|بدهید|دهید){_E}"),
    ("analysis", "fa_compare", 2, rf"{_B}مقایسه"),
    ("analysis", "fa_trend", 2, rf"{_NOT_MI}{_B}روند{_E}"),
    ("preparation", "fa_convert", 3, rf"{_B}تبدیل\s*(کن|کنید){_E}|{_B}به\s+فرمت{_E}"),
    ("preparation", "fa_remove", 3, rf"{_B}(حذف|پاک)\s*(کن|کنید){_E}"),
    ("preparation", "fa_merge", 3, rf"{_B}ادغام|{_B}(ترکیب|یکی)\s*(کن|کنید){_E}"),
    ("preparation", "fa_clean", 3, rf"{_B}پاک{_S}سازی|{_B}تمیز\s*(کن|کنید){_E}"),
    ("preparation", "fa_duplicates", 3, rf"{_B}تکراری"),
    ("reporting", "fa_dashboard", 3, rf"{_B}داشبورد"),
    ("reporting", "fa_periodic", 3, rf"{_B}گزارش\s+(روزانه|هفتگی|ماهانه|فصلی){_E}|{_B}(روزانه|هفتگی|ماهانه)\s.{{0,20}}?گزارش"),
    ("reporting", "fa_report", 2, rf"{_B}گزارش"),
    ("reporting", "fa_kpi", 2, rf"{_B}شاخص"),
    ("business", "fa_why", 3, rf"{_B}چرا{_E}"),
    ("business", "fa_what_to_do", 3, rf"{_B}چه\s*کار\s+(باید\s+)?(بکنیم|کنیم){_E}|{_B}چه\s+کنیم{_E}"),
    ("advice", "fa_difference", 4, rf"{_B}(فرق|تفاوت)\s.{{1,60}}?(\sبا\s|\sو\s).{{1,60}}?چیست"),
    ("advice", "fa_which_metric", 3, rf"{_B}چه\s+معیاری|{_B}کدام\s+معیار|{_B}کدام\s+مدل\s.{{0,20}}?(مناسب|بهتر)"),
    ("advice", "fa_meaning", 2, rf"{_B}یعنی\s+چه{_E}"),
]

# a person who does not know (the whole answer), in Persian beside the English of chat.NOT_KNOWN
NOT_KNOWN = re.compile(rf"\s*(ن{_MI}دانم|نمیدانم|مطمئن\s+نیستم|نمی{_S}دونم|اطلاعی\s+ندارم|خبر\s+ندارم|نه\s+نمی{_S}دانم)\s*[.!؟?]*\s*")
# an order to build the project, as the whole message (never a bare "بساز", "make": "یک داشبورد بساز" is a dashboard;
# never a sentence that only contains "شروع کنیم": "نه، اول داده را ببینیم بعد شروع کنیم" is a no). The review (H3)
_ORDER = rf"(?:بسازش|بسازیدش|(?:پروژه|راه{_S}حل)(?:\s+(?:را|رو))?\s+(?:را\s+)?بساز(?:ید)?|(?:برو|بریم)\s+جلو|شروع\s+(?:کن|کنید|کنیم))"
_YES = r"(?:بله|بلی|آره|اره|باشه|حتما|اوکی|خب|خوب)"
BUILD = re.compile(rf"\s*(?:{_YES}[\s،,.!]*)?(?:لطفا\s+)?{_ORDER}(?:\s+(?:لطفا|دیگه|الان|همین\s+الان))?\s*[.!]*\s*")  # fullmatch only
# a bare yes ("بله", "آره", "باشه", "حتماً", "اوکی") as the whole answer: a yes to an offer to build
ASSENT = re.compile(rf"\s*{_YES}(\s+(لطفا|ممنون))?\s*[.!]*\s*")
# a bare yes or "that's right" to a question about what Home understood (the critic's), never to an offer to build
CONFIRMED = re.compile(rf"\s*(?:{_YES}|موافقم|درسته|درست\s+است|همینه|دقیقا)(\s+(لطفا|ممنون|درسته))?\s*[.!]*\s*")
SIMULATE = re.compile(rf"{_B}شبیه{_S}سازی|{_B}داده(ی)?\s+(ساختگی|مصنوعی){_E}")

# The plan's outcome and moment in Persian (plan.extract), on normalized text. Persian puts the verb last, so an outcome
# runs from "which" or "whether" through its window to the verb: "کدام مشتری‌ها ظرف ۳۰ روز آینده اشتراکشان را لغو می‌کنند".
_FA_UNIT = r"(?:ساعت|روز|هفته|ماه|فصل|سال)"
_FA_WINDOW = rf"{_B}(?:ظرف|در|طی)\s+(?:\d+|یک|دو|سه|چند)\s+{_FA_UNIT}{_E}(?:\s+(?:آینده|بعدی?){_E})?|{_B}{_FA_UNIT}\s+(?:آینده|بعدی?){_E}(?!\s+از{_E})"
_W = r"[^\s،,.;؛?؟!]+"  # one word, never its punctuation
_FA_VERB = rf"(?:{_MI}{_W}|خواهند\s+{_W}|خواهد\s+{_W}|{_W}\s+(?:{_MI})?(?:شوند|شود|کنند|کند){_E})"
TARGET = re.compile(rf"({_B}(?:کدام|آیا|چه\s+کسانی|چه\s+تعداد|چند)\s[^،,.;؛?؟]*?(?:{_FA_WINDOW})[^،,.;؛?؟]*?{_FA_VERB})")
# A moment is claimed only where it is set apart (the text's or a clause's start, or right after "predict"/"score"), as in
# English, and never when it says how often the data is refreshed: "داده‌ها هر ماه به‌روز می‌شوند" is not a moment.
_APART = rf"(?:^|[،,.;؛:]\s*|{_B}(?:پیش{_S}بینی|امتیاز{_S}دهی|محاسبه)(?:\s+(?:کن|کنید|شود|شوند))?\s+)"
_NOT_REFRESH = rf"(?!\s+(?:به{_S}روز|بروز|آپدیت|refresh|update))"
_END = r"(?=\s+(?:و|تا|که)\s|[،,.;؛?؟]|$)"
MOMENTS = [re.compile(_APART + "(" + p + ")" + _NOT_REFRESH) for p in (
    rf"{_B}(?:\d+|یک|دو|سه)\s+(?:ساعت|روز|هفته|ماه){_E}\s+(?:قبل|پیش)\s+از\s+[^،,.;؛?؟]+?{_END}",
    rf"{_B}(?:در\s+)?(?:لحظه|زمان|هنگام){_E}\s*(?:ی\s+)?\s*[^،,.;؛?؟]+?{_END}",
    rf"{_B}(?:اول|ابتدای|آخر|پایان)\s+هر\s+(?:روز|هفته|ماه|فصل|سال){_E}",
    rf"{_B}هر\s+(?:روز\s+صبح|شب|صبح|روز|هفته|ماه|شنبه|یکشنبه|دوشنبه|سه\s*شنبه|چهارشنبه|پنجشنبه|جمعه){_E}(?:\s+صبح{_E})?",
    rf"{_B}(?:قبل|پیش)\s+از\s+(?:تماس|ارسال|تحویل|پرداخت|ثبت\s*نام|ثبت\s+سفارش|تمدید|رزرو|ورود){_E}",
)]

__all__ = ["normalize", "normalize_map", "span", "has_persian", "SIGNALS", "NOT_KNOWN", "BUILD", "ASSENT", "CONFIRMED", "SIMULATE", "TARGET", "MOMENTS"]

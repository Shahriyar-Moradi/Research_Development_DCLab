"""Persian and English without a model (package H3): one normalisation and the pattern tables Home's rules read.

Home's deterministic path (the intent rules, the plan's outcome and moment rules, the "let's build" and "simulate" words)
was English regular expressions, so a Persian request with no model fell through to a prediction or to "Noted". This
module gives every rule the same footing in both languages:

- **Normalise before matching** (``normalize``): Arabic yeh and kaf become the Persian letters, Persian and Arabic-Indic
  digits become 0-9, the zero-width non-joiner and other invisible marks become a space, diacritics and the tatweel go,
  case folds and runs of space become one. It returns an index back into the original text, so every quote is cut from
  what the person wrote (their ZWNJ, their digits), never from the normalised copy.
- **Pattern tables per language**, written in the normalised form (``ی`` and ``ک``, a plain space where people put a ZWNJ
  or a space, Western digits): ``INTENT`` for the six tracks, ``TARGET`` and ``MOMENTS`` for the plan, ``BUILD`` and
  ``SIMULATE`` for the chat.

English and Persian now; Arabic is a later, additive table (the owner's decision 5 in the Home plan).
"""

from __future__ import annotations

import re
from typing import Iterator, Optional

# --------------------------------------------------------------------------------------------- normalisation
_LETTERS = {"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه"}  # ي ى ك ۀ ة
_DIGITS = {**{chr(0x06F0 + i): str(i) for i in range(10)}, **{chr(0x0660 + i): str(i) for i in range(10)}}
_SPACES = {"‌", "‍", "‎", "‏", " ", " ", " ", "​"}  # ZWNJ, ZWJ, marks, odd spaces
_DROP = {chr(c) for c in range(0x064B, 0x0660)} | {"ٰ", "ـ"}  # harakat, hamza above (ٔ), superscript alef, tatweel


def normalize(text: str) -> tuple[str, list[int]]:
    """The text as the rules read it, and for each of its characters the index of the original character it came from."""
    out: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(str(text or "")):
        if ch in _DROP:
            continue
        if ch == "\n":
            pass  # a line break stays one: an English rule that stops at the end of a line still does
        elif ch in _SPACES or ch.isspace():
            ch = " "
        else:
            ch = _LETTERS.get(ch) or _DIGITS.get(ch) or ch
        if ch == " " and (not out or out[-1] in (" ", "\n")):
            continue
        for c in ch.lower():  # "İ" lowers to two characters: each keeps its own index entry
            out.append(c)
            index.append(i)
    return "".join(out), index


def fold(text: str) -> str:
    """The normalised text alone (for comparing two texts, never for quoting)."""
    return normalize(text)[0].strip()


def _word(ch: str) -> bool:
    return ch.isalnum() or ch in ("‌", "_")


def span(text: str, index: list[int], start: int, end: int) -> tuple[int, int]:
    """A match on the normalised text, as a span of the original text, grown to whole words (a ZWNJ is inside a word)."""
    if not index or start >= end:
        return 0, 0
    a, b = index[start], index[min(end, len(index)) - 1] + 1
    while a > 0 and _word(text[a - 1]):
        a -= 1
    while b < len(text) and _word(text[b]):
        b += 1
    return a, b


def finditer(pattern: re.Pattern[str], text: str, group: int = 0) -> Iterator[tuple[int, int]]:
    """Each match of a pattern written in the normalised form, as an original-text span of ``group``."""
    norm, index = normalize(text)
    for m in pattern.finditer(norm):
        if m.group(group):
            yield span(text, index, m.start(group), m.end(group))


def search(pattern: re.Pattern[str], text: str, group: int = 0) -> Optional[tuple[int, int]]:
    return next(finditer(pattern, text, group), None)


def has(pattern: re.Pattern[str], text: str) -> bool:
    return bool(pattern.search(normalize(text)[0]))


def script(text: str) -> str:
    """"fa" when the text is mostly Persian letters, "en" otherwise (for choosing the words Home answers in)."""
    letters = [c for c in str(text or "") if c.isalpha()]
    persian = sum(1 for c in letters if "؀" <= c <= "ۿ")
    return "fa" if letters and persian * 2 >= len(letters) else "en"


# ------------------------------------------------------------------------------------------------ the tables
# Persian intent signals, (track, id, weight, pattern), in the same weights as the English ones in intent.SIGNALS: a
# weight of 3 is one strong phrase, the only kind that takes a request off the prediction path. Written from the H1
# dev requests and ordinary Persian, never from the sealed set. "\s" stands where people write a ZWNJ or a space.
_W = r"(?:\S+\s+){0,3}"  # up to three words between
INTENT_FA: list[tuple[str, str, int, str]] = [
    ("prediction", "fa_predict", 3, r"پیش\s?بینی|پیشگویی"),
    ("prediction", "fa_estimate", 3, r"تخمین|برآورد"),
    ("prediction", "fa_model", 3, r"مدل(?:ی)?\s+" + _W + r"(?:بساز|بسازم|بسازیم|بسازید|آموزش)"),
    ("prediction", "fa_future", 3, r"(?:چقدر|چند)\s+(?:\S+\s+){0,4}?(?:خواهد|خواهند|خواهیم)|(?:خواهد|خواهند)\s+(?:کرد|شد|خرید|رفت|داشت)"),
    ("prediction", "fa_which_will", 3, r"کدام\s+" + _W + r"(?:ظرف|طی|در)\s+\d+\s+(?:روز|هفته|ماه|سال)"),
    ("prediction", "fa_likely", 2, r"احتمال|احتمالا"),
    ("prediction", "fa_next_period", 2, r"(?:روز|هفته|ماه|فصل|سال)\s+(?:آینده|بعد(?!\s+از))|فردا|ظرف\s+\d+\s+(?:روز|هفته|ماه|سال)"),
    ("analysis", "fa_how_much", 3, r"چقدر|چند\s?تا|چند\s+(?:نفر|بار|سفارش|مشتری|عدد)"),
    ("analysis", "fa_what_was", 3, r"چه\s+(?:مقدار|تعداد)\s+" + _W + r"(?:بود|بوده)"),
    ("analysis", "fa_share", 3, r"چند\s+درصد|چه\s+درصدی|درصد\s+(?:\S+\s+){0,2}?(?:چقدر|چند)"),
    ("analysis", "fa_compare", 2, r"مقایسه"),
    ("analysis", "fa_aggregate", 2, r"میانگین|مجموع|جمع\s+کل|میانه|تعداد"),
    ("analysis", "fa_show", 2, r"نشان\s+بده|نشان\s+دهید|نمایش\s+بده"),
    ("analysis", "fa_per_group", 1, r"به\s+تفکیک|بر\s+اساس|در\s+هر|برای\s+هر"),
    ("analysis", "fa_trend", 1, r"(?<!می )(?<!نمی )روند|توزیع"),  # never "می‌روند" (they leave)
    ("preparation", "fa_join", 3, r"ادغام|ترکیب\s+کن|یکی\s+کن|به\s+هم\s+وصل"),
    ("preparation", "fa_remove", 3, r"حذف\s+(?:کن|کنید|شود)|(?:ردیف|سطر)\s*(?:ها|های)?\s+" + _W + r"حذف|تکراری\s*ها\s+را\s+حذف"),
    ("preparation", "fa_derive", 3, r"تبدیل\s+(?:کن|کنید)|ستون\s+جدید"),
    ("preparation", "fa_clean", 3, r"پاک\s?سازی|تمیز\s+(?:کن|کنید)"),
    ("preparation", "fa_keep_only", 3, r"فقط\s+" + _W + r"(?:را\s+)?نگه\s+(?:دار|دارید)"),
    ("preparation", "fa_export", 2, r"خروجی\s+" + _W + r"(?:اکسل|csv)|فایل\s+اکسل"),
    ("reporting", "fa_dashboard", 3, r"داشبورد"),
    ("reporting", "fa_periodic", 3, r"گزارش\s+(?:روزانه|هفتگی|ماهانه|فصلی)|(?:روزانه|هفتگی|ماهانه)\s+گزارش"),
    ("reporting", "fa_report", 2, r"گزارش"),
    ("reporting", "fa_kpi", 2, r"شاخص"),
    ("reporting", "fa_every_period", 2, r"هر\s+(?:روز|هفته|ماه|فصل)"),
    ("reporting", "fa_track", 2, r"پایش|رصد|پیگیری"),
    ("reporting", "fa_refresh", 2, r"به\s?روز\s+(?:شود|بشود|کن)|بروزرسانی|به\s?روزرسانی"),
    ("business", "fa_why", 3, r"چرا"),  # a whole word: the boundaries are added to every Persian pattern below
    ("business", "fa_what_to_do", 3, r"چه\s+کار\s+(?:باید\s+)?(?:بکنیم|کنیم)|چه\s+باید\s+کرد|آیا\s+باید"),
    ("business", "fa_worth", 3, r"(?:ارزش|می\s?ارزد)\s+" + _W + r"(?:دارد|که)|می\s?ارزد"),
    ("business", "fa_decide", 1, r"تصمیم"),
    ("advice", "fa_difference", 3, r"(?:فرق|تفاوت)\s+(?:بین|میان|\S+(?:\s+\S+)?\s+(?:با|و))|چه\s+(?:فرقی|تفاوتی)\s+(?:دارد|دارند|هست)"),
    ("advice", "fa_which_metric", 3, r"(?:چه\s+(?:معیاری|روشی|مدلی|الگوریتمی)|کدام\s+(?:معیار|روش|مدل|الگوریتم))\s+(?:\S+\s+){0,2}?(?:بهتر|مناسب|استفاده)"),
    ("advice", "fa_is_it_ok", 3, r"آیا\s+(?:\S+\s+){0,6}?(?:مناسب|درست|خوب)\s+است|مناسب\s+است\s+که|اشکالی\s+دارد|ایرادی\s+دارد"),
    ("advice", "fa_how_much_data", 4, r"چند\s+(?:ردیف|نمونه|داده)\s+" + _W + r"(?:لازم|نیاز)"),
    ("advice", "fa_explain", 3, r"توضیح\s+(?:بده|بدهید)|یعنی\s+چه|(?:یعنی\s+)?چیست\s*[؟?]?\s*$"),  # "… چیست؟" asks what it is
]

_FA_LETTER = "\u0621-\u064a\u066e-\u06d3\u06d5\u06fa-\u06ff"  # the letters only: "،" and "؟" are in the block too
# Every Persian signal is a whole word or phrase: never "مجموع" inside "مجموعه", "رصد" inside "درصد", "فرق" in "متفاوت"
INTENT_FA = [(t, i, w, rf"(?<![{_FA_LETTER}])(?:{p})(?![{_FA_LETTER}])") for t, i, w, p in INTENT_FA]

# The plan's outcome with its window, in Persian word order: "پیش‌بینی کنم کدام مشتری‌ها ظرف ۳۰ روز آینده … لغو می‌کنند"
# (the window comes before the verb, and the clause ends at "تا", "که" or punctuation).
_FA_WINDOW = r"(?:ظرف|طی|در)\s+\d+\s+(?:ساعت|روز|هفته|ماه|سال)(?:\s+(?:آینده|بعد))?|(?:هفته|ماه|سال)\s+(?:آینده|بعد)"
TARGET_FA = re.compile(r"(?:پیش\s?بینی|تخمین|برآورد)\s+(?:کنم|کنید|کن|کنیم|شود|بزن|بزنید|بزنیم)?\s*"
                       r"((?:کدام|آیا|چه\s+کسی|چه\s+کسانی|چند)\s+[^,.;؟?،]*?(?:" + _FA_WINDOW + r")[^,.;؟?،]*?)"
                       r"(?=\s+تا\s|\s+که\s|[,.;؟?،!]|$)")
# A moment only when the sentence sets it apart (after a comma), as in English: "…، هر دوشنبه صبح" or "…، قبل از تماس".
MOMENTS_FA = [re.compile(r"[،,]\s*(?:و\s+)?(" + p + r")") for p in (
    r"(?:\d+|یک|دو|سه|چهار|پنج|شش|هفت)\s+(?:ساعت|روز|هفته|ماه)\s+(?:قبل|پیش)\s+از\s+[^,.;؟?،]+",
    r"(?:قبل|پیش)\s+از\s+(?:تماس|ارسال|ثبت|خرید|رزرو|تمدید|پرداخت|ورود)[^,.;؟?،]*",
    r"(?:در\s+لحظه|هنگام|زمان)\s+(?:ثبت|ایجاد|ارسال|رزرو|خرید|ورود)[^,.;؟?،]*",
    r"(?:اول|آخر|ابتدای|انتهای)\s+هر\s+(?:ماه|هفته|فصل|سال)[^,.;؟?،]*",
    r"هر\s+(?:روز|هفته|ماه|صبح|شب|شنبه|یکشنبه|دوشنبه|سه\s?شنبه|چهارشنبه|پنجشنبه|جمعه)[^,.;؟?،]*",
)]

# The chat's word checks, both languages, on normalised text. "Build" names the solution, the project or the model,
# because "بساز" alone is also "make me a pivot".
BUILD = re.compile(r"\b(?:let'?s build|build (?:it|the solution)|go ahead|start the project|ready to build)\b"
                   r"|(?:راه\s?حل|پروژه|مدل)\s+(?:را\s+)?بساز|^\s*(?:بیا\s+|خب\s+|پس\s+)?شروع\s+کنیم\s*[.!]?\s*$|بریم\s+(?:سراغ\s+)?ساخت|بسازیمش", re.I)
SIMULATE = re.compile(r"\b(?:simulat\w*|synthetic|fake data|generate (?:some )?data)\b"
                      r"|شبیه\s?سازی|داده\s+(?:ی\s+)?(?:مصنوعی|ساختگی|فرضی)|داده\s+بساز", re.I)


# "I'm not sure" in both languages (package H4): an answer that is not a value. Home keeps what it understood and,
# where the field has a safe default, records the default (shown as one, editable), never the words as the answer.
NOT_SURE_BUTTON = "Not sure"
# 3: only a whole short reply: "Not sure, but probably whether they cancel within 30 days" is an answer
NOT_SURE = re.compile(r"^\s*(?:(?:i'?m\s+|i\s+am\s+|i\s+(?:really\s+)?)?(?:not\s+sure|unsure|no\s+idea|don'?t\s+know|do\s+not\s+know|skip(?:\s+it)?|no\s+clue)"
                      r"(?:\s+(?:yet|really|sorry))?|مطمئن\s+نیستم|نمی\s?دانم|نمی\s?دونم|ایده\s+ای\s+ندارم|نظری\s+ندارم|رد\s+شو|فعلا\s+نه)\s*[.!،]*\s*$", re.I)


def not_sure(text: str) -> bool:
    return has(NOT_SURE, text or "")

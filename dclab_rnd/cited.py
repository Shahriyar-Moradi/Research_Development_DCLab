"""Sentences a model writes, kept only when they cite a source and their numbers are in it (CLAUDE.md rule 5).

One check for every place a model writes prose from records: the Evidence library's answers, the stage explanations
(package A3.3) and the notebook fixes (A3.4). Given the text and the sources it may cite (an id and the text behind
it), code keeps a sentence only when

- it cites at least one source, in square brackets, and every id it cites was given;
- every number in it appears in a source it cites: as written (so 0.30 is 0.3, ``.99`` is 0.99), with its unit (61%,
  61k and 61 are different numbers) and its sign when the sentence writes one; a number word (five, half, twice,
  million) must be a word of a cited source; an id such as WF-03 is not a number. A model that rounds, converts or
  computes a number loses the sentence;
- it does not claim production readiness, a guarantee, a proof or a cause, unless it denies that right where it says it
  ("is not yet production-ready"); and
- it fits in the word limit, when there is one.

What this does not prove: a number is checked against the sources the sentence cites, not against what it describes.
A sentence that cites two sources can put a number from one beside the subject of the other.

Everything else is dropped, with a fixed reason (never the model's own words, so a reason is safe to log).
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

CITE = re.compile(r"\[([A-Za-z][A-Za-z0-9_.-]*)\]")
IDS = re.compile(r"\b(?:DCLAB-R\d+|WF-\d+|EXP-\d+|PIT-\d+|CAT-\d+|LEAK-[a-z_]+|FINDING-[a-z-]+|DATASET-[a-z_]+|PRJ-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\b")
# a number as written: sign, digits (or .99), and the unit that changes what it means (%, k, M, thousand, million, percent)
NUMBER = re.compile(r"(?<![\d.])([-+]?)(\d+(?:[.,]\d+)*|\.\d+)(?:\s?(%|percent\b|[kKmMbB]\b|thousand\b|million\b|billion\b))?")
# number words that stand for a digit are read as digits ("ten folds" is 10 folds); the others (half, twice, million…) cannot be
# compared and must be words of a cited source
COUNTS = {w: str(n) for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
COUNTS |= {w: str(10 * n) for n, w in zip(range(3, 10), "thirty forty fifty sixty seventy eighty ninety".split())}
_UNIT_AFTER = r"(?:\s?(%|percent\b|per cent\b|thousand\b|million\b|billion\b))?"
_TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_UNITS = "one|two|three|four|five|six|seven|eight|nine"
# twenty-one, then two, ten, ninety, then "one" only where it is a quantity (one percent, a one-point gain), each with the unit that follows
COUNT_WORD = re.compile(rf"\b(?:((?:{_TENS})[-\s](?:{_UNITS}))|({'|'.join(w for w in COUNTS if w != 'one')})|(one)(?=-(?:point|percent|fold)\b|\s(?:percent|per cent|fold|times|in)\b))\b{_UNIT_AFTER}", re.I)
NUMBER_WORDS = re.compile(r"\b(?:hundreds?|thousands?|millions?|billions?|half|twice|thrice|double[sd]?|triple[sd]?|dozens?)\b", re.I)
NAME_WITH_DIGITS = re.compile(r"\b[A-Z]{1,2}\d{1,3}\b")  # C1, R22, V12: a name, not a quantity, when a cited source writes that name
ABBREVIATIONS = re.compile(r"\b(e\.g|i\.e|vs|etc|approx|cf)\.", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
AFTER_STOP = re.compile(r"([.!?])((?:\s*\[[A-Za-z][A-Za-z0-9_.-]*\])+)")  # "holdout. [ID]" is "holdout [ID]."
BULLET = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+)")
DASHES = re.compile(r"[‐-―−]")

# What a model must not say about a stage result. Matched on text that was normalised (NFKC, one kind of dash and of space).
OVERCLAIM = re.compile(
    r"\b(?:production|deployment)[\s-]*(?:ready|readiness|grade|quality|approved)\b|approved for production|fit for production|"
    r"ready (?:for|to go to|for use in) (?:production|deploy\w*)|ready to (?:ship|launch|deploy|be deployed)|safe (?:to deploy|for production)|"
    r"\bgo(?:es)? (?:live|to production)\b|\bguarantee[sd]?\b|\bproves?\b|\bproven\b|\bcauses\b|\bcaused\b|\bcausal\b", re.I)
# A denial counts only when it sits right before the phrase ("is not yet production-ready"): not "not only", and not a "no" elsewhere.
DENIAL = re.compile(r"(?:\bnot(?!\s+(?:only|just|merely|simply)\b)|\bnever|\bcannot|\bcan't|n't\b|\bno longer)"
                    r"(?:\s+(?!(?:doubt|deny|wrong|fail|unlike|hard|unreasonable)\b)\w+){0,4}\s+$", re.I)

REASONS = ("no citation", "cites a source that was not given", "a number is not in the cited source",
           "claims production readiness, a guarantee or a cause", "over the word limit")


def normalise(text: str) -> str:
    """One kind of dash and of space, compatibility forms folded: so "production‑ready" and "production  ready" match."""
    return re.sub(r"\s+", " ", DASHES.sub("-", unicodedata.normalize("NFKC", text)))


def _digits(text: str) -> str:
    text = text.replace(",", "")
    try:
        return format(Decimal(text).normalize(), "f")
    except InvalidOperation:
        return text


def numbers(text: str, source: bool = False, names: set[str] | None = None) -> set[str]:
    """The numbers a text states. Identifiers and citations are not numbers; 0.30 is 0.3; ``.99`` is 0.99; a unit stays
    with its number. A sentence's number carries its sign only when it writes one ("-0.01", "+0.01"); for a source
    (``source=True``) every number is stored with its real sign (unsigned counts as positive) and without it."""
    out: set[str] = set()
    plain = NAME_WITH_DIGITS.sub(lambda m: " " if source or m.group(0) in (names or ()) else m.group(0), IDS.sub(" ", CITE.sub(" ", normalise(text))))
    for sign, digits, unit in NUMBER.findall(plain):
        token = _digits(digits) + {"percent": "%", "thousand": "k", "million": "m", "billion": "b"}.get(unit.lower(), unit.lower())
        if source:
            out |= {token, ("-" if sign == "-" else "+") + token}
        else:
            out.add(sign + token)
    for compound, count, one, unit in COUNT_WORD.findall(plain):  # "ten folds" and "10 folds" are the same number, in a sentence and in a source
        if compound:
            tens, units = re.split(r"[-\s]", compound.lower(), maxsplit=1)
            value = str(int(COUNTS[tens]) + int(COUNTS[units]))
        else:
            value = COUNTS[(count or one).lower()]
        token = value + {"percent": "%", "per cent": "%", "thousand": "k", "million": "m", "billion": "b"}.get(unit.lower(), unit.lower())
        out |= {token, "+" + token} if source else {token}
    return out


def words_of(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", normalise(text).lower()))


def number_words(sentence: str) -> set[str]:
    return {w.lower() for w in NUMBER_WORDS.findall(CITE.sub(" ", normalise(sentence)))}


def overclaims(sentence: str) -> bool:
    text = CITE.sub(" ", normalise(sentence))
    return any(not DENIAL.search(text[:m.start()]) for m in OVERCLAIM.finditer(text))


def words(text: str) -> int:
    return len(CITE.sub(" ", text).split())


GLUED = re.compile(r"(\S+?)([.!?])(?=[A-Z])(\w+)")


MODULES = frozenset("np pd tf plt sns pandas numpy sklearn imblearn scipy torch keras xgboost lightgbm catboost matplotlib seaborn statsmodels nn "
                    "xgb lgb sm mpl pl pathlib os joblib shap optuna mlflow polars dask jnp".split())


def _unglue(match: re.Match[str]) -> str:
    token, stop, word = match.groups()
    # pd.DataFrame, sklearn.Pipeline, sklearn.pipeline.Pipeline: a code name, not two sentences; a citation before the stop always ends one
    if not token.endswith("]") and (token.lstrip("`'\"([{<").rsplit(".", 1)[-1].lower() in MODULES or re.search(r"[A-Za-z_]\.[A-Za-z_]", token)):
        return match.group(0)
    return f"{token}{stop}\n{word}"


def sentences(text: str) -> list[str]:
    """Split at a full stop, a question or exclamation mark, and at every line break (so a bullet is a sentence);
    "e.g." and "vs." do not end one, nor does the dot of a code name such as pd.DataFrame. A stop glued to the next
    capital ("fair.It scored") ends one. A fragment this leaves has no citation of its own, so it fails closed."""
    protected = ABBREVIATIONS.sub(lambda m: m.group(1) + "\u2024", AFTER_STOP.sub(lambda m: m.group(2) + m.group(1), text.strip()))
    protected = GLUED.sub(_unglue, protected)
    return [BULLET.sub("", s.strip()).replace("\u2024", ".") for s in SENTENCE.split(protected) if BULLET.sub("", s.strip())]


def _tidy(sentence: str) -> str:
    text = re.sub(r"\s+", " ", CITE.sub("", sentence))
    return re.sub(r"\s+(?=[.,;:!?](?:\s|$))", "", text).strip()


def check(text: str, sources: dict[str, str], limit_words: int | None = None) -> dict[str, Any]:
    """Keep the sentences that pass. Returns ``{"kept": [{"text", "cites"}], "dropped": [reason, ...], "words": n}``."""
    known = {source: numbers(body, source=True) for source, body in sources.items()}
    spoken = {source: words_of(body) for source, body in sources.items()}
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    used = 0
    for sentence in sentences(text):
        cited = list(dict.fromkeys(CITE.findall(sentence)))
        if not cited:
            dropped.append(REASONS[0])
        elif any(c not in sources for c in cited):
            dropped.append(REASONS[1])
        elif not (numbers(sentence, names=set().union(*(set(NAME_WITH_DIGITS.findall(sources[c])) for c in cited))) <= set().union(*(known[c] for c in cited))
                  and number_words(sentence) <= set().union(*(spoken[c] for c in cited))):
            dropped.append(REASONS[2])
        elif overclaims(sentence):
            dropped.append(REASONS[3])
        elif limit_words is not None and used + words(sentence) > limit_words:
            dropped.append(REASONS[4])
        else:
            kept.append({"text": _tidy(sentence), "cites": cited})
            used += words(kept[-1]["text"])
    return {"kept": kept, "dropped": dropped, "words": used}


def strip_overclaims(text: str) -> tuple[str, int]:
    """For advisory text with no citations to check: remove the sentences that overclaim. Returns the rest and the
    count; with nothing to remove the text comes back exactly as it was (its lines and lists included)."""
    parts = sentences(text)
    kept = [p for p in parts if not overclaims(p)]
    if len(kept) == len(parts):
        return text, 0
    return ("\n" if "\n" in text.strip() else " ").join(kept), len(parts) - len(kept)

"""Column sensitivity (package 23.7): which columns hold a person's data, read from a column's name and the shape of its
values, never keeping a value.

Four classes, strongest first:

* ``restricted`` — secrets and identifiers that must reach nothing: a password, a token, a secret key, a card number
  (found by the Luhn check), a government id's shape, an IBAN.
* ``pii`` — what names or reaches a person: an e-mail, a phone number, a name, an address, a date of birth, an IP address.
* ``sensitive`` — what a person did not choose to share: health, religion, ethnicity, orientation, politics, income.
* ``none``.

A column that is not ``none`` reaches a model by name only: its example values, category names, prefixes and shapes
are never kept (``strip``). ``classify`` returns the class and the reason in words, naming a rule, never a value.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Optional

from dclab_core.bands import name_tokens

CLASSES: tuple[str, ...] = ("restricted", "pii", "sensitive", "none")
RANK = {c: i for i, c in enumerate(CLASSES)}  # lower is stronger

# a name's tokens (see bands.name_tokens: snake, camel and spaced names all split the same way; trailing digits are dropped: phone1)
_RESTRICTED_TOKENS = {"password", "passwd", "pwd", "secret", "token", "apikey", "credential", "credentials", "cvv", "cvc", "iban", "ssn", "passport"}
_RESTRICTED_PAIRS = {("api", "key"), ("private", "key"), ("access", "key"), ("secret", "key"), ("card", "number"), ("credit", "card"), ("debit", "card"),
                     ("national", "id"), ("social", "security"), ("tax", "id"), ("bank", "account"), ("account", "number"), ("routing", "number"),
                     ("card", "no"), ("card", "num"), ("cc", "num"), ("cc", "number"), ("cc", "no"), ("account", "no"), ("acct", "no"), ("acct", "num"),
                     ("account", "num"), ("pan", "number")}
_PII_TOKENS = {"email", "mail", "phone", "mobile", "telephone", "tel", "fax", "firstname", "lastname", "fname", "lname", "surname", "fullname", "dob",
               "birthday", "birthdate", "address", "street", "postcode", "postal", "zipcode", "zip", "ip", "geolocation", "username", "msisdn"}
_PII_PAIRS = {("first", "name"), ("last", "name"), ("full", "name"), ("given", "name"), ("family", "name"), ("middle", "name"), ("date", "birth"),
              ("birth", "date"), ("home", "address"), ("ip", "address"), ("e", "mail"), ("phone", "number"), ("mobile", "number"), ("cell", "phone"),
              ("customer", "name"), ("user", "name"), ("zip", "code"), ("post", "code"), ("postal", "code"), ("street", "address"), ("mac", "address"),
              ("home", "latitude"), ("home", "longitude"), ("home", "location")}
_SENSITIVE_TOKENS = {"diagnosis", "disease", "illness", "medication", "prescription", "disability", "religion", "ethnicity", "race", "orientation",
                     "politics", "salary", "income", "wage", "convicted", "criminal", "pregnant", "pregnancy", "biometric", "fingerprint"}
_SENSITIVE_PAIRS = {("credit", "score"), ("health", "condition"), ("sexual", "orientation"), ("political", "party"), ("union", "member"),
                    ("blood", "type"), ("medical", "history"), ("criminal", "record")}
_PERSON_WORDS = {"client", "employee", "patient", "person", "member", "owner", "contact", "guest", "driver", "rider", "applicant", "student"}
# a count, a flag, a rate, a duration or a frequency about a kind of data is not that data (email_opens, has_phone, phone_type,
# handle_time, word_freq_mail, orientation_deg, race_time)
_ABOUT = {"count", "sent", "opened", "opens", "open", "clicks", "click", "is", "has", "flag", "rate", "total", "valid", "verified", "bounce",
          "bounced", "domain", "type", "status", "consent", "optin", "opt", "missing", "null", "length", "service", "freq", "frequency", "time",
          "deg", "degrees", "data", "gb", "mb", "size", "minutes", "seconds", "days", "id", "ids", "band", "bucket"}


def _tokens(name: Any) -> list[str]:
    """The tokens of a name with trailing digits dropped (phone1 reads as phone) and the small words of date_of_birth removed."""
    out = []
    for t in name_tokens(str(name)):
        t = re.sub(r"\d+$", "", t)
        if t and t not in ("of", "the", "a", "an"):
            out.append(t)
    return out


def of_name(name: Any) -> Optional[tuple[str, str]]:
    """The class a column's name gives it and the rule in words, or None."""
    tokens = _tokens(name)
    pairs = set(zip(tokens, tokens[1:]))
    squashed = "".join(tokens)
    about = bool(set(tokens) & _ABOUT)
    for cls, singles, joined in (("restricted", _RESTRICTED_TOKENS, _RESTRICTED_PAIRS), ("pii", _PII_TOKENS, _PII_PAIRS), ("sensitive", _SENSITIVE_TOKENS, _SENSITIVE_PAIRS)):
        hit = (set(tokens) & singles) or ({squashed} & singles)
        if hit and not about:
            return cls, f"the name says {sorted(hit)[0]}"
        pair = pairs & joined
        if pair and not about:
            return cls, f"the name says {' '.join(sorted(pair)[0])}"
        if cls == "pii" and not about and "name" in tokens and (tokens in (["name"], ["names"]) or set(tokens) & _PERSON_WORDS):
            return cls, "the name says a person's name"
    return None


EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[A-Za-z]{2,}$")
PHONE = re.compile(r"^(\+\d[\d\s().-]{6,18}\d|\(?\d{2,4}\)?[\s.-]\d{2,4}[\s.-]\d{3,4}([\s.-]\d{2,4})?)$")
DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T].*)?$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")
SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
IBAN = re.compile(r"^[A-Za-z]{2}\d{2}[A-Za-z0-9]{11,30}$")
IPV4 = re.compile(r"^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$")
IPV6 = re.compile(r"^(?=(?:.*:){2,7}[^:]*$)[0-9A-Fa-f:]{3,39}$")
SAMPLE = 200
PLACEHOLDERS = {"", "nan", "none", "null", "<na>", "n/a", "na", "-", "--", "?", "unknown", "missing", "undefined", "nil"}  # not a value: out of the share


def sample_of(values: Any) -> list[str]:
    """Up to ``SAMPLE`` non-empty values as text, drawn evenly across the column (a sorted file's first rows are not the column), a float
    that is a whole number written as one (4111111111111111.0 reads as the number it is)."""
    if hasattr(values, "iloc"):  # a pandas series: stride over it before listing
        values = values.dropna()
        step = max(1, len(values) // (SAMPLE * 3))
        values = values.iloc[::step]
    out = []
    for v in values:
        text = str(v).strip()
        if text.lower() in PLACEHOLDERS:
            continue
        out.append(re.sub(r"^(\d+)\.0$", r"\1", text)[:80])
    if len(out) > SAMPLE:
        stride = len(out) / SAMPLE
        out = [out[int(i * stride)] for i in range(SAMPLE)]
    return out


def luhn(digits: str) -> bool:
    """The Luhn check of a card number's digits."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def _card(v: str) -> bool:
    digits = re.sub(r"[\s-]", "", v)
    return bool(re.fullmatch(r"[\d\s-]{13,23}", v)) and luhn(digits)


# strongest first: a value that is a card number is not also read as a phone number. (class, rule, test, share of the values it must match):
# an e-mail address is named by a fifth of a column (the rest of it leaks anyway); the shapes that share their form with other
# things (a phone number, a card number's digits, an id) by half.
VALUE_RULES: tuple[tuple[str, str, Any, float], ...] = (
    ("restricted", "values are shaped like card numbers (they pass the Luhn check)", _card, 0.5),
    ("restricted", "values are shaped like national ids", lambda v: bool(SSN.match(v)), 0.5),
    ("restricted", "values are shaped like IBANs", lambda v: bool(IBAN.match(v.replace(" ", ""))) and bool(re.search(r"\d", v)), 0.5),
    ("pii", "values are shaped like e-mail addresses", lambda v: bool(EMAIL.match(v)), 0.2),
    ("pii", "values are shaped like IP addresses", lambda v: bool(IPV4.match(v)) or bool(IPV6.match(v)), 0.5),
    ("pii", "values are shaped like phone numbers", lambda v: bool(PHONE.match(v)) and not DATE.match(v) and 7 <= len(re.sub(r"\D", "", v)) <= 15, 0.5),
)
UNION = ("pii", "values are shaped like e-mail addresses, phone numbers or IP addresses", (3, 4, 5), 0.5)  # a column that mixes them


def of_values(values: Iterable[Any]) -> Optional[tuple[str, str]]:
    """The class the shape of a column's values gives it and the rule in words, or None. Reads at most ``SAMPLE`` non-empty
    values as text (``sample_of``), keeps none."""
    sample = sample_of(values)
    if len(sample) < 3:
        return None
    for cls, why, test, share in VALUE_RULES:
        if sum(1 for v in sample if test(v)) / len(sample) >= share:
            return cls, why
    cls, why, which, share = UNION
    if sum(1 for v in sample if any(VALUE_RULES[i][2](v) for i in which)) / len(sample) >= share:
        return cls, why
    return None


def classify(name: Any, values: Iterable[Any] = ()) -> tuple[str, Optional[str]]:
    """``(class, why)`` for a column: the stronger of what its name and its values say; ``("none", None)`` otherwise."""
    found = [r for r in (of_name(name), of_values(values)) if r is not None]
    if not found:
        return "none", None
    cls, why = min(found, key=lambda r: RANK[r[0]])
    return cls, why


# what carries a column's values, by the names the data card, the profile and the Home analysis give it
NAMING = frozenset({"categories", "numeric_levels", "prefix", "shape", "digits", "top", "preview"})  # category names, examples, a shared prefix
DISTRIBUTION = frozenset({"quantiles", "step", "start", "end", "date_format", "shares", "repeats", "hist", "min", "max", "mean", "std", "time",
                          "other_count", "top_share"})


def upgrade_column(doc: dict[str, Any]) -> dict[str, Any]:
    """A description of a column written before the class existed (a stored profile, a v1 or v2 data card) brought up to the rule:
    classified by its name and by the values it still holds (its examples, its category names), then stripped. A description that
    already has its class is stripped to it again (the same result)."""
    if doc.get("sensitivity") in CLASSES:
        return strip(doc, doc["sensitivity"])
    cls, why = classify(doc.get("name"), list(doc.get("preview") or []) + list(doc.get("categories") or []))
    return strip({**doc, "sensitivity": cls, "sensitivity_why": why}, cls)


def upgrade_card(card: dict[str, Any]) -> dict[str, Any]:
    """The same for a data card document: every column of every table."""
    return {**card, "tables": [{**t, "columns": [upgrade_column(c) for c in t.get("columns") or []]} for t in card.get("tables") or []]}


def strip(doc: dict[str, Any], cls: Optional[str]) -> dict[str, Any]:
    """A column's description (a data card's column, a profile's, the Home analysis's) without what carries its values: a
    ``restricted`` or ``pii`` column keeps its name, type and bands and nothing of its distribution; a ``sensitive`` one
    keeps its distribution but no example, category name or prefix."""
    if cls in (None, "none"):
        return dict(doc)
    drop = NAMING | DISTRIBUTION if cls in ("restricted", "pii") else NAMING
    out = {k: v for k, v in doc.items() if k not in drop}
    if "preview" in doc:
        out["preview"] = []  # the key stays: a reader that indexes it finds nothing, not an error
    return out


__all__ = ["CLASSES", "RANK", "of_name", "of_values", "classify", "strip", "upgrade_column", "upgrade_card", "sample_of", "luhn"]

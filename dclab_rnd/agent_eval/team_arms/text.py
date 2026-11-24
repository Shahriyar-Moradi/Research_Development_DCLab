"""The text rules the team-arms scorer reads with, copied here and pinned with the gate (B4 re-review, N1, 2026-11-24).

A copy, on purpose: the scorer must read a number, a sentence and a claimed action the same way for the whole life of
``team_arms_gate_v1``, while the product's own copies (``dclab_rnd/cited.py``: ``numbers``, ``sentences``;
``dclab_core/team_brain.py``: ``claimed_actions``, ``unrecorded_claims``) keep changing as the arms are improved (B5 edits
team_brain). Copied unchanged from those files at commit 25c6e830 and never kept equal to them: a change the gate should
read is a new gate version that copies again. A test pins this copy's own behaviour on fixed sentences.

B4.1 (2026-11-24, before any sealed run, so re-pinned as ``team_arms_gate_v1``): the first dev round found three claim
readings wrong on real sentences, and the same fix went into the product's ``team_brain`` (0.173.2): "scored" is a claim only
when finished ("each account is scored once" is a design), "approved" as an adjective ("an approved offer") is no approval,
and "recorded" claims a decision only with "decision" as its object (never "recorded before the decision point").
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

# ---------------------------------------------------------------------------------------------------------- dclab_rnd/cited.py
CITE = re.compile(r"\[([A-Za-z][A-Za-z0-9_.-]*)\]")
IDS = re.compile(r"\b(?:DCLAB-R\d+|WF-\d+|EXP-\d+|PIT-\d+|CAT-\d+|LEAK-[a-z_]+|FINDING-[a-z-]+|DATASET-[a-z_]+|PRJ-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|LESSON-[A-Za-z0-9]+)\b")
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

# Persian and Arabic number marks (2026-11-16): "۲۰٬۰۰۰" is 20,000 and "۱۰٪" is 10%. NFKC folds none of the three, so a
# Persian speaker's own number was read as two small ones and a reply that repeated it counted as a number nobody gave.
NUMBER_MARKS = str.maketrans({"\u066c": ",", "\u066b": ".", "\u066a": "%"})
PERCENT_WORD = re.compile(r"(?<=\d)\s?درصد")  # "۱۰ درصد" is 10%


def marks(text: str) -> str:
    """Persian and Arabic number marks, and the Persian word for percent after a number, as the marks numbers are read by."""
    return PERCENT_WORD.sub("%", str(text).translate(NUMBER_MARKS))


def normalise(text: str) -> str:
    """One kind of dash and of space, compatibility forms folded: so "production‑ready" and "production  ready" match."""
    return re.sub(r"\s+", " ", DASHES.sub("-", unicodedata.normalize("NFKC", marks(text))))


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



# ---------------------------------------------------------------------------------------------------------- dclab_core/team_brain.py
_WORDS = r"(?:\s+[\w'-]+){0,4}\s+"
# A sentence that says an action was done. Each claim is checked against the action log: an action with no terminal
# record ("succeeded" or "simulated") on the task or a task it handed out is a claim code refuses. The last four have no
# tool in any release of the team workspace, so a claim of them is always refused (unless the sentence denies it).
_BEEN = r"(?:\s+[\w'-]+){0,4}\s+(?:was|were|has been|have been|had been|got|is now|are now)\s+"  # the passive: "the schema was inspected"
# B4.1 (the team-arms dev run, 2026-11-24): what reads like a claim and is none.
# 1. "scored" claims the scoring only when it is finished: after a subject that did it ("we scored", "Data Science has scored")
#    or a finished passive ("the accounts were scored", "is now scored"); never "each account is scored once", "what the model
#    is scored on" or a design written as a participle ("rank accounts scored seven days before renewal").
_SUBJECTS = r"(?:we|i|they|it|he|she|science|engineering|sales|marketing|lead|team|model|code|dclab)"
_NOT_PRESENT = r"(?!(?:is|are|be|being|been|to|get|gets)\b)"
# the review of B4.1 (F2): a contraction ("we've scored"), a finished list ("All accounts scored; top 50 sent") and "are scored
# now" are claims too; "each account scored at its own T-7" (a design) is not
_SCORED = (rf"\b(?:was|were|been|got|have|has|had)\s+(?:{_NOT_PRESENT}[\w'-]+\s+){{0,2}}?scored\b|\b(?:is|are)\s+(?:now\s+scored|scored\s+now)\b|"
           rf"\b{_SUBJECTS}\s+(?:{_NOT_PRESENT}[\w'-]+\s+){{0,2}}?scored\b|\b(?:i|we|they|you|it)'(?:ve|d)\s+(?:[\w'-]+\s+){{0,2}}?scored\b|"
           r"\b(?:all|every)\s+(?:[\w'-]+\s+){0,2}?scored\b(?=\s*(?:[;,.!]|$|and\b|already\b))|^\s*scored\s+(?:all|every|the|\d)")
# 3. "recorded" claims a decision only with "decision" as its direct object ("we recorded the decision"), never a decision
#    point, moment or date after a preposition ("a renewal recorded before the decision point")
_DETERMINER = r"(?:the|a|an|our|your|their|its|his|her|this|that|each|every)"
_PREPOSITION = r"(?:before|after|at|on|in|by|for|until|of|to|from|with|near|around)"
_NOT_A_DECISION = r"(?!\s+(?:points?|moments?|dates?|times?|windows?|rules?|thresholds?|boundar(?:y|ies)|deadlines?|makers?|days?))"
# 2. "approved" as an adjective ("an approved offer", "the already approved budget") claims no approval
_ADJECTIVE_BEFORE = re.compile(r"\b(?:a|an|the|any|no|their|its|our|your|his|her|this|that|these|those|each|every|some)\s+"
                               r"(?:(?:already|newly|pre|previously|recently|formally|officially|fully)[- ]?\s*)?$", re.I)
# an adjective comes before its noun: "an approved offer", never "this approved by finance" or "that approved yesterday"
_NOUN_AFTER = re.compile(r"\s+(?!(?:by|yesterday|today|now|already|earlier|last|this|on|at|in|for|and|or|but|so|then|it|them)\b)[a-z][\w-]*", re.I)
CLAIMS: dict[str, "re.Pattern[str]"] = {name: re.compile(pattern, re.I) for name, pattern in (
    ("inspect_schema", rf"\b(?:inspected|examined|read|reviewed|looked at|went through)\b{_WORDS}(?:schema|columns?)\b|"
                       rf"\b(?:schema|columns?)\b{_BEEN}(?:inspected|examined|read|reviewed|looked at)\b"),
    ("check_leakage", rf"\b(?:checked|tested|screened|audited|ran)\b(?:{_WORDS})?(?:for\s+)?(?:\w+\s+)?leak\w*|"
                      rf"\bleak\w*\b{_BEEN}(?:checked|tested|screened|audited|ruled out)\b"),
    ("define_baseline", rf"\b(?:defined|set|named|chose|picked|fixed)\b{_WORDS}baseline\b|\bbaseline\b{_BEEN}(?:defined|set|named|chosen|picked|fixed)\b"),
    ("validate_contract", rf"\bvalidated\b{_WORDS}contract\b|\bcontract\b{_BEEN}validated\b"),
    ("review_evidence", rf"\breviewed\b{_WORDS}(?:evidence|repl(?:y|ies)|reports?|answers?)\b|"
                        rf"\b(?:evidence|repl(?:y|ies)|reports?|answers?)\b{_BEEN}reviewed\b"),
    ("delegate", rf"\b(?:delegated|handed|assigned|opened)\b{_WORDS}(?:tasks?|work|it|this|request)\b|"
                 r"\basked\s+(?:the\s+)?(?:data science|ml engineering|sales|marketing|business)\b|"
                 rf"\b(?:tasks?|work|request)\b{_BEEN}(?:delegated|handed|assigned|opened)\b"),
    ("score", _SCORED),
    ("train", r"\b(?:trained|retrained|fitted|fit)\b(?:\s+[\w'-]+){0,3}\s+models?\b|\bmodels?\b(?:\s+[\w'-]+){0,4}\s+(?:was|were|has been|have been|had been|got|is now)\s+(?:trained|retrained|fitted)\b"),
    ("deploy", r"\bdeployed\b"),
    ("approve", r"\bapproved\b|\bsigned[- ]off\b|\bapproval\b(?:\s+[\w'-]+){0,3}\s+(?:was|has been|had been)\s+(?:given|granted)\b|"
                r"\bgave\s+(?:the\s+|their\s+|his\s+|her\s+|its\s+)?(?:approval|sign-?off|go-ahead)\b"),
    # 0.173.0, the review of B3 (R2): every directory action a team performs, and what a person said
    ("draft_contact_list", rf"\b(?:drafted|prepared|built|made|compiled|produced)\b{_WORDS}(?:call |contact )?list\b|"
                           rf"\b(?:call |contact )?list\b{_BEEN}(?:drafted|prepared|built|made|compiled)\b"),
    ("check_exclusions", rf"\b(?:removed|excluded|filtered out|took out)\b{_WORDS}(?:do-not-contact|exclusions?)\b|"
                         rf"\bchecked\b{_WORDS}(?:exclusions?|do-not-contact)\b|\bexclusions?\b{_BEEN}(?:checked|applied|removed)\b"),
    ("check_eligibility", rf"\bchecked\b{_WORDS}eligib\w*|\beligib\w*\b{_BEEN}checked\b"),
    ("check_campaign_dates", rf"\bchecked\b{_WORDS}(?:campaign(?:'s)? )?dates\b|\bdates\b{_BEEN}checked\b"),
    ("check_cost", rf"\bchecked\b{_WORDS}(?:cost|budget)\b|\b(?:cost|budget)\b{_BEEN}checked\b"),
    ("check_prerequisites", rf"\bchecked\b{_WORDS}prerequisites?\b|\bprerequisites?\b{_BEEN}checked\b"),
    ("present_options", rf"\bpresented\b{_WORDS}(?:options?|report|findings)\b|\b(?:options?|report)\b{_BEEN}presented\b"),
    ("record_decision", rf"\brecorded\s+(?:{_DETERMINER}\s+)?(?:(?!{_PREPOSITION}\b)[\w'/-]+\s+){{0,3}}?decisions?\b{_NOT_A_DECISION}|"
                        rf"\bdecisions?\b{_NOT_A_DECISION}{_BEEN}recorded\b"),
    ("run_evaluation", rf"\bevaluated\b{_WORDS}models?\b|\bmodels?\b{_BEEN}evaluated\b"),
    ("register_version", rf"\bregistered\b{_WORDS}(?:models?|versions?)\b|\b(?:models?|versions?)\b{_BEEN}registered\b"),
    ("compare_baseline", rf"\bcompared\b{_WORDS}(?:with |to |against )?(?:the |a )?baseline\b"),
    ("person_confirmed", r"\b(?P<who>[A-Z][a-z]+|the (?:person|owner|decision owner|member|requester|customer|client))\b(?:\s+[\w'-]+){0,3}?\s+"
                         r"(?:has |have |had )?(?:confirmed|agreed|accepted|said)\b|\b(?:confirmed|agreed|accepted)\s+by\s+"
                         r"(?:[A-Z][a-z]+|the (?:person|owner|decision owner|member|requester|customer|client))\b"),
    ("contact_customers", rf"\b(?:called|rang|contacted|e-?mailed|phoned|messaged|texted|reached out to|sent|told|notified|informed|offered)\b"
                          r"(?:\s+[\w'-]+){0,5}?\s+(?:to\s+)?(?:the\s+|our\s+|all\s+)?(?:[\w-]+\s+)?(?:customers?|accounts?|clients?|users?|people|them)\b|"
                          rf"\b(?:customers?|accounts?|clients?|users?|people|they)\b{_BEEN}(?:called|rang|contacted|e-?mailed|phoned|messaged|"
                          r"texted|reached|sent|told|notified|informed|offered)\b"),
)}
NEVER_RECORDED = ("train", "deploy", "approve", "contact_customers")  # no team tool performs them
# a name that is a team or the agent itself, never a person who confirmed something ("Sales confirmed the capacity")
_NOT_PEOPLE = frozenset({"i", "we", "it", "this", "that", "they", "he", "she", "you", "data", "science", "sales", "marketing", "business",
                         "lead", "engineering", "ml", "code", "dclab", "the", "a", "an", "our", "its", "their", "team", "both", "each", "all"})
PERSON = "human_response"  # what meets a person's confirmation: a finished human-bound action (a person's recorded answer)
# The directory actions each claim is met by (any of them, with a terminal record)
CLAIM_ACTIONS: dict[str, tuple[str, ...]] = {
    "delegate": ("delegate", "delegate_specialist", "delegate_analysis", "request_sales_input", "request_marketing_input", "open_case"),
    "review_evidence": ("review_evidence", "reconcile_replies", "present_options", "present_report"),
    "draft_contact_list": ("draft_contact_list", "revise_on_correction"),
    "present_options": ("present_options", "present_report"),
    "person_confirmed": (PERSON,),
}
# A claim is no claim when the words just before it deny it, ask for it, or put it in the future or a condition:
# "nothing was trained", "Sales should call them", "before anyone contacted them"
_NOT_DONE = re.compile(r"\b(?:no|not|never|nothing|none|neither|nor|without|cannot|can't|won't|n't|before|until|if|unless|"
                       r"will|would|should|could|can|may|might|must|needs?|asks?|asked to|wants?|plans?|once)\b", re.I)
# A denial inside a claim's own words ("we called no customers"). "Not just", "not only" and "no fewer than" widen the
# claim instead of denying it ("we emailed not just the top accounts but all customers"), so they never count.
# what starts a new clause inside a list's item: then the denial before the list does not reach the claim ("no data was
# missing or late and we scored every account" claims the scoring; the review of the smoke fix, 2026-11-24)
_NEW_CLAUSE = re.compile(r"\b(?:and|so|then|when|while|whereas|because|since|after|before|as|yet|but|until|once|which|that|we|they|it|"
                         r"he|she|i|you|sales|marketing|business|engineering|science|lead|team)\b", re.I)
_LIST_DENIAL = re.compile(r"\b(?:not|never|no|neither|nor|without|cannot)\b|n't\b", re.I)  # a denial a list's "or" carries
# a word joined by a dash or a slash is no denial ("a go/no-go decision", "the no-contact list"): B4.1 review, F2
_INNER_DENIAL = re.compile(r"(?<![-/])\b(?:not|never|no)\b(?![-/])(?!\s+(?:just|only|merely|simply|fewer|less|more|later)\b)", re.I)


def claimed_actions(sentence: Any) -> list[str]:
    """The actions a sentence says were done (keys of :data:`CLAIMS`), in their order. A claim the words just before it
    deny, ask for or put in the future ("nothing was trained", "Sales should call them") is no claim. Words only: whether
    the action was done is the host's to check against its action log."""
    text = " ".join(str(sentence or "").split())
    out: list[str] = []
    for name, pattern in CLAIMS.items():
        for match in pattern.finditer(text):
            # a denial counts in its own clause only: "No list was drafted, but we called the customers" claims the call
            head = text[:match.start()]
            cut = max((m.end() for m in re.finditer(r"[;,:]|\bbut\b", head, re.I)), default=0)
            before = head[cut:].split()[-6:]
            # denied before it, or inside it ("the accounts on the list were never ... called")
            denied = bool(_NOT_DONE.search(" ".join(before)) or _INNER_DENIAL.search(match.group(0)))
            if not denied:  # a denial carries over a list joined by "or" or "nor" in its own clause: "it does not say a model was
                # trained or an account was scored" claims neither (live B4 smoke, 2026-11-24). Not over "and", which may start a
                # new statement ("we did not train a model, and we scored the accounts" claims the scoring).
                clause = head[cut:]
                joins = list(re.finditer(r"\b(?:or|nor)\b", clause, re.I))
                between = clause[joins[-1].end():] if joins else ""  # the list's item: one, with no new clause or subject in it
                denied = (bool(joins) and bool(_LIST_DENIAL.search(clause[:joins[-1].start()])) and len(between.split()) <= 6
                          and not _NEW_CLAUSE.search(between))
            if denied:
                continue
            if (name == "approve" and match.group(0).lower().startswith("approved") and _ADJECTIVE_BEFORE.search(head)
                    and _NOUN_AFTER.match(text[match.end():])):
                continue  # "an approved offer": an adjective, no approval claimed (B4.1); "we got this approved by finance" claims one
            who = (match.groupdict().get("who") or "") if name == "person_confirmed" else ""
            if who and who.split()[-1].lower() in _NOT_PEOPLE:
                continue  # a team or the agent confirmed it: its own action, not a person's answer
            if name not in out:
                out.append(name)
            break
    return out


def unrecorded_claims(sentence: Any, done: Sequence[str]) -> list[str]:
    """The actions ``sentence`` claims that ``done`` (the directory actions with a terminal record; :data:`PERSON` when a
    person's answer is recorded) does not hold."""
    have = set(done)
    return [c for c in claimed_actions(sentence)
            if c in NEVER_RECORDED or not (have & ({c} | set(CLAIM_ACTIONS.get(c, ()))))]



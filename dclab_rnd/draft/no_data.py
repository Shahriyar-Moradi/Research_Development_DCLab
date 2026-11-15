"""Solve a user's business, data science, and ML problem honestly before any table exists.

When a person has no dataset yet, is still planning collection, or prefers not to upload sensitive data,
DCLab should never stall on "upload a CSV first". Instead, this module builds an honest three-layer
blueprint using DCLab's problem-map decomposer, domain kind registry, Task Catalog planner (TP-08) over
virtual schemas, evidence index (PIT-* / ev-*), business examples, and deterministic Python decision-
economics math:

1. Business solution & decision economics (owner, operational action, non-ML baseline, ROI/break-even math)
2. Data science solution & Minimum Viable Data Spec (virtual_tables, prediction cutoff, target maturity, leakage traps)
3. ML solution & pre-data task plan (routing, baseline -> primary model, calibration/thresholding, TP-08 breakdown)

Synthetic data is NEVER auto-generated here: it provides tailored questions to ask the user first how
simulated data should look, keeping ``Build project`` completely unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from .. import business_examples, evidence_index, kinds, tools as rnd_tools
from ..tasks import catalog as task_catalog


# Natural detection (English & Persian) — no magic keyword required. Only a stated lack of data, or a stated refusal to share
# it, counts (the review of 22ad1bf8: "no data quality issues", "no data scientist", "without data leakage", a privacy word or
# the word "simulate" alone all matched).
_NOT_HAVING = (r"(?!\s+(?:quality|scientists?|science|engineers?|engineering|analysts?|team|leaks?|leakage|loss|breach|privacy|protection|"
               r"issues?|problems?|governance|entry|types?|models?|warehouse|pipelines?|strategy|platform))")
_DATA = r"(?:data|dataset|datasets|csv|file|files|table|tables|records?|samples?)"
_NO_DATA_PATTERNS = (
    re.compile(r"\bno\s+(?:historical\s+)?" + _DATA + r"\b" + _NOT_HAVING, re.I),
    re.compile(r"\b(?:do\s+not|don'?t|have\s+not|haven'?t)\s+(?:actually\s+)?have\s+(?:any\s+|a\s+)?(?:historical\s+)?" + _DATA + r"\b" + _NOT_HAVING, re.I),
    re.compile(r"\b(?:without|haven'?t\s+collected|have\s+not\s+collected)\s+(?:any\s+)?(?:historical\s+)?" + _DATA + r"\b" + _NOT_HAVING, re.I),
    re.compile(r"\bbefore\s+(?:we\s+|i\s+|they\s+|you\s+)?(?:collect(?:ing)?|hav(?:e|ing)|upload(?:ing)?|get(?:ting)?)\s+(?:any\s+|historical\s+)?(?:data|dataset|csv|table)\b", re.I),
    re.compile(r"\b(?:data|dataset|csv|table)\s+(?:is|are)\s+not\s+(?:ready|collected|available)(?:\s+yet)?\b", re.I),
    re.compile(r"\b(?:solve|plan|design|solution|help|advise)\b[^.?!]{0,40}\bwithout\s+(?:uploading\s+|sharing\s+|any\s+)?data\b" + _NOT_HAVING, re.I),
    re.compile(r"\black(?:s|ing)?\s+(?:any\s+|the\s+)?(?:historical\s+)?(?:data|dataset|records)\b" + _NOT_HAVING, re.I),
    re.compile(r"\b(?:data|dataset|datasets|table)\s+(?:does|do)\s+not\s+exist\b", re.I),
    re.compile(r"\bzero\s+data\b" + _NOT_HAVING, re.I),
    re.compile(r"\b(?:still|currently)\s+(?:gathering|collecting)\s+(?:the\s+|our\s+)?data\b", re.I),
    re.compile(r"(?<!کیفیت )(?<!کیفیت\u200c)(?:دیتا|داده|فایل|جدول)\s*(?:ندارم|نداریم|نیست|آماده نیست|هنوز ندارم|نمی‌خوام|نمیخوام)", re.I),
    re.compile(r"بدون\s*(?:دیتا|داده|آپلود|فایل)", re.I),
    re.compile(r"هیچ\s*(?:دیتا|داده)", re.I),
    re.compile(r"(?:اطلاعات|دیتا|داده)[\u200cیا]*\s*(?:نداریم|ندارم|در\s*دسترس\s*نیست)", re.I),
)

_SHARED_THING = (r"(?:(?:my|our|the|any|this|that|customer|company|real|raw|actual|sensitive|private|confidential|internal)\s+){0,3}"
                 r"(?:data|dataset|datasets|databases?|files?|tables?|csv|records|it|them|anything)\b")
# a refusal to share the data itself; a privacy word alone ("the file is confidential, I will upload it now") is no refusal
_PRIVACY_PATTERNS = (
    re.compile(r"\bwithout\s+(?:uploading|sharing|sending|exposing)\s+" + _SHARED_THING, re.I),
    re.compile(r"\b(?:don'?t|do\s+not|can'?t|cannot|won'?t|prefer\s+not\s+to|not\s+allowed\s+to|may\s+not)\s+(?:want\s+to\s+)?(?:upload|share|send|expose)\s+" + _SHARED_THING, re.I),
    re.compile(r"(نمی‌خوام\s+آپلود|نمیخوام\s+آپلود|نمی‌تونم\s+بفرستم|به\s+اشتراک\s+نمی)", re.I),
)
_PRIVACY_WORDS = re.compile(r"\b(confidential|nda|private|sensitive|compliance|hipaa|gdpr|internal\s+only)\b|محرمانه|خصوصی", re.I)

_SIMULATE_ASK_PATTERNS = (
    re.compile(r"\b(simulat\w*|synthetic|mock\s+data|dummy\s+data|fake\s+data)\b", re.I),
    re.compile(r"(شبیه‌سازی|مصنوعی|دیتای\s+فرضی|تولید\s+داده)", re.I),
)

# What the person wants is a prediction (or new groups), said in their words; and the shapes of a question that is none
_PREDICT_WORDS = re.compile(r"\b(predict\w*|forecast\w*|classif\w*|scor(?:e|ing)\b|likely\s+to|probability|risk\s+of|which\s+\w+\s+will|who\s+will|"
                            r"segment\w*|cluster\w*|detect\w*|estimat\w*|rank\w*)|پیش‌بینی|پیش بینی|احتمال|خوشه|سگمنت", re.I)
_NOT_A_PREDICTION = re.compile(r"\bwhy\b|\broot\s+cause|\bwhat\s+changed|\ba/?b\s+test|\bexperiment\w*|\boptimi[sz]\w*|\broutes?\b|\brouting\b|"
                               r"\bschedul\w*|\bexplain\w*|\bdifference\s+between|\bwhat\s+is\b|\bwhat\s+does\b|\breport\b|\bdashboard\b|"
                               r"\bcompare\b|\bcomparison\b|چرا|علت|گزارش|داشبورد|مقایسه", re.I)

_NUMBER_RE = re.compile(r"(?P<prefix>[$€£¥]?)\s*(?P<num>\d[\d,]*(?:\.\d+)?)(?P<unit>[kKmMbB])?\b(?!\s*%)")
_DIGIT_TRANS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٬٫٪", "01234567890123456789,.%")


def _norm_digits(text: str) -> str:
    return str(text or "").translate(_DIGIT_TRANS)


def detect_no_data_stance(text: str, draft: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Detect whether the person lacks data yet or prefers not to share/upload it."""
    raw = str(text or "").strip()
    reasons: list[str] = []
    lacks = any(p.search(raw) for p in _NO_DATA_PATTERNS)
    if lacks:
        reasons.append("no_data_yet")
    privacy = any(p.search(raw) for p in _PRIVACY_PATTERNS)
    if privacy:
        reasons.append("privacy_or_withholding")
    wants_sim = any(p.search(raw) for p in _SIMULATE_ASK_PATTERNS)
    if wants_sim:
        reasons.append("asks_about_simulation")
    # a table that is still being read is data too: only a failed or stopped upload leaves the person without one
    has_asset = bool(draft and any(str(a.get("status")) not in ("failed", "stopped", "error") for a in draft.get("assets") or []))
    return {
        "matched": (lacks or privacy) and not has_asset,  # the word "simulate" or a privacy word alone is no stance
        "reasons": reasons,
        "withholding_privacy": privacy,
        "mentions_privacy": bool(_PRIVACY_WORDS.search(raw)),
        "wants_simulation": wants_sim,
        "has_ready_data": has_asset,
    }


def _family_of(problem: str) -> tuple[str, str]:
    """The family the person's words point to, and where it came from: "kind" (a registered kind's own detector), "words" (a
    domain word in the sentence) or "default" (nothing pointed anywhere: no family was found)."""
    found = kinds.detect(problem, None) or {}
    family_key = str(found.get("family") or found.get("key") or found.get("pack") or "prediction").lower()
    source = "default" if str(found.get("why") or "").startswith("nothing points elsewhere") or not found else "kind"
    low = problem.lower()
    if family_key in ("prediction", "binary", "binary_classification", "rare_event", "tabular"):
        named = None
        if "fraud" in low or "تقلب" in low or "کلاهبرداری" in low:
            named = "fraud"
        elif any(w in low for w in ("loan", "underwriting", "default rate", "credit risk", "micro-lending", "وام", "نکول")):
            named = "credit_risk"
        elif any(w in low for w in ("churn", "retention", "cancel", "ریزش", "لغو اشتراک")):
            named = "churn"
        elif any(w in low for w in ("forecast", "demand", "sales next", "inventory", "تقاضا", "پیش‌بینی فروش", "پیش بینی فروش")):
            named = "forecasting"
        elif (
            any(w in low for w in ("price", "revenue", "ltv", "lifetime value", "claim severity", "expected loss", "duration", "eta", "قیمت", "درآمد"))
            and not any(c in low for c in ("costs $", "action cost", "intervention cost", "review costs", "call costs", "visit costs", "inspection costs"))
        ):
            named = "regression"
        elif any(w in low for w in ("segment", "cluster", "سگمنت", "خوشه‌بندی", "خوشه بندی")):
            named = "segmentation"
        if named:
            family_key, source = named, "words"
    return family_key, source


def blueprint_fit(problem_text: str) -> dict[str, Any]:
    """Whether a model blueprint answers what the person asked. It does when their words ask for a prediction, a forecast, a
    score or groups; or when they name a family and the sentence is not a why, a test's design, a route plan, a report or
    an explanation. Otherwise DCLab has no outcome to design for, and says so instead of drawing a yes/no model."""
    problem = str(problem_text or "").strip()
    family, source = _family_of(problem)
    asks = bool(_PREDICT_WORDS.search(problem))
    other = _NOT_A_PREDICTION.search(problem)
    if asks and source != "default":
        fits, why = True, "the words ask for a prediction and name its kind"
    elif asks:
        fits, why = True, "the words ask for a prediction; its kind is not named, so the blueprint is a general one"
    elif source != "default" and not other:
        fits, why = True, f"the words name a {family} problem"
    elif other:
        fits, why = False, f'the words ask for something that is not a prediction ("{other.group(0).strip()}")'
    else:
        fits, why = False, "the words name nothing to predict"
    return {"fits": fits, "family": family, "family_source": source, "why": why}


NOT_A_BLUEPRINT = ("You do not need to upload data to go on. I could not tell what you want to predict, so I did not draw a model "
                   "plan. If you want a prediction, tell me the outcome in one sentence (for example: which customers will cancel "
                   "next month). If your question is something else (why a number changed, how to design a test, how to plan "
                   "routes, a report), say so and I will work on that instead.")


# where a phrase ends: a number on the other side of it belongs to something else ("a $50 intervention cost, and $2,500 saved"
# gives the cost 50, never 2,500). A comma inside a number ("2,500") is no boundary: only a comma followed by a space is.
_CLAUSE_END = re.compile(r"[.;!?\n](?:\s|$)|,\s|\s(?:and|but)\s")


def _occurrences(low: str, kw: str) -> list[int]:
    """Where a keyword stands as a whole word ("rate" is not found inside "operate"); a Persian keyword as written."""
    if re.search(r"[a-z]", kw):
        return [m.start() for m in re.finditer(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", low)]
    return [m.start() for m in re.finditer(re.escape(kw), low)]


def _windows(text: str, idx: int, kw: str, before: int, after: int) -> tuple[str, str]:
    """The words just before and just after a keyword, cut at the end of its clause: a number in another sentence is not its."""
    left = text[max(0, idx - before): idx]
    ends = [m.end() for m in _CLAUSE_END.finditer(left)]
    left = left[ends[-1]:] if ends else left
    right = text[idx + len(kw): idx + len(kw) + after]
    end = _CLAUSE_END.search(right)
    return left, (right[: end.start()] if end else right)


def _parse_money_or_count(text: str, keywords: tuple[str, ...]) -> Optional[float]:
    """The number said with the first of these keywords the person used; when they said it again later (a correction), the
    later number."""
    text = _norm_digits(text)
    low = text.lower()
    found: list[tuple[int, float]] = []  # (where the keyword first stands, its latest number)
    for kw in keywords:
        spots = _occurrences(low, kw)
        prefer_before = kw in (
            "customers", "users", "subscribers", "accounts", "applications", "discharges",
            "machines", "shipments", "orders", "tickets", "patients", "leads", "rows",
            "transactions", "مشتری", "مشترک", "کاربر", "سفارش", "تراکنش", "بیمار",
        )
        for idx in reversed(spots):  # the latest statement first
            before, after = _windows(text, idx, kw, 22, 28)
            order = ((before, True), (after, False)) if prefer_before else ((after, False), (before, True))
            value = None
            for window, pick_last in order:
                matches = [m for m in _NUMBER_RE.finditer(window) if m.group("num")]
                if not matches:
                    continue
                m = matches[-1] if pick_last else matches[0]
                value = float(m.group("num").replace(",", "")) * {"k": 1e3, "m": 1e6, "b": 1e9}.get((m.group("unit") or "").lower(), 1.0)
                break
            if value is not None:
                found.append((spots[0], value))
                break
    return min(found)[1] if found else None


def _parse_rate(text: str, keywords: tuple[str, ...]) -> Optional[float]:
    text = _norm_digits(text)
    low = text.lower()
    found: list[tuple[int, float]] = []
    for kw in keywords:
        spots = _occurrences(low, kw)
        for idx in reversed(spots):
            before, after = _windows(text, idx, kw, 20, 25)
            value = None
            for window, pick_last in ((before, True), (after, False)):
                matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*%", window))
                if not matches:
                    continue
                pct = float((matches[-1] if pick_last else matches[0]).group(1))
                if 0.0 < pct < 100.0:
                    value = pct / 100.0
                    break
            if value is not None:
                found.append((spots[0], value))
                break
    return min(found)[1] if found else None


_VOLUME_WORDS = (
    "customers", "users", "subscribers", "accounts", "applications", "discharges",
    "machines", "shipments", "orders", "tickets", "patients", "leads", "rows",
    "transactions", "مشتری", "مشترک", "کاربر", "سفارش", "تراکنش", "بیمار",
)
_RATE_WORDS = ("rate", "churn", "fraud", "default", "convert", "conversion", "cancel", "risk", "نرخ", "ریزش")
_COST_WORDS = ("cost per", "offer cost", "discount", "intervention", "review cost", "call cost", "costs", "cost", "هزینه")
# what one caught case is worth: never a company's revenue or a loss in general ("we generate $2M revenue" is not it)
_VALUE_WORDS = ("ltv", "lifetime value", "margin", "ticket size", "order value", "worth", "value per", "value of", "saves", "saved",
                "avoids", "avoided", "penalty", "ارزش")
ECONOMICS_INPUTS = ("monthly_volume", "base_rate", "action_cost", "value_saved", "action_success_rate")


def checked_overrides(overrides: Optional[Mapping[str, Any]]) -> dict[str, float]:
    """The numbers a person typed for the economics, as numbers in their range; ValueError says which one is not."""
    out: dict[str, float] = {}
    for name, raw in (overrides or {}).items():
        if name not in ECONOMICS_INPUTS or raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} must be a number")
        if name in ("base_rate", "action_success_rate") and not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be a share between 0 and 1")
        if name == "monthly_volume" and value < 1:
            raise ValueError("monthly_volume must be at least 1")
        if name in ("action_cost", "value_saved") and value < 0:
            raise ValueError(f"{name} cannot be negative")
        out[name] = value
    return out


def compute_decision_economics(
    problem_text: str,
    *,
    understanding: Optional[Mapping[str, Any]] = None,
    case_facts: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Deterministically compute break-even precision, capacity budget, and scenario net value."""
    und = understanding or {}
    facts = case_facts or {}
    ov = checked_overrides(overrides)
    combined = " ".join(
        str(x)
        for x in (
            problem_text,
            und.get("target"),
            und.get("action"),
            und.get("notes"),
            (facts.get("goal") or {}).get("value") if isinstance(facts.get("goal"), Mapping) else facts.get("goal"),
            (facts.get("decision") or {}).get("value") if isinstance(facts.get("decision"), Mapping) else facts.get("decision"),
            (facts.get("constraints") or [{}]) if isinstance(facts.get("constraints"), list) else "",
        )
        if x
    )

    said_volume = _parse_money_or_count(combined, _VOLUME_WORDS)
    monthly_volume = ov.get("monthly_volume") or said_volume or 10_000.0
    volume_assumed = not (ov.get("monthly_volume") or said_volume)

    said_rate = _parse_rate(combined, _RATE_WORDS)
    base_rate = ov.get("base_rate") or said_rate or 0.08
    rate_assumed = not (ov.get("base_rate") or said_rate)

    said_cost = _parse_money_or_count(combined, _COST_WORDS)
    action_cost = ov["action_cost"] if "action_cost" in ov else (said_cost or 15.0)
    cost_assumed = not ("action_cost" in ov or said_cost)

    said_value = _parse_money_or_count(combined, _VALUE_WORDS)
    value_saved = ov["value_saved"] if "value_saved" in ov else (said_value or 180.0)
    value_assumed = not ("value_saved" in ov or said_value)

    action_success_rate = float(ov.get("action_success_rate") or 0.35)
    monthly_volume, base_rate, action_cost, value_saved = float(monthly_volume), float(base_rate), float(action_cost), float(value_saved)
    expected_positives = round(monthly_volume * base_rate)
    effective_value_per_tp = value_saved * action_success_rate
    break_even_precision = min(1.0, round(action_cost / max(effective_value_per_tp, 1e-6), 4))
    suggested_top_k_pct = 0.15 if base_rate <= 0.12 else 0.25
    flagged_count = max(1, round(monthly_volume * suggested_top_k_pct))

    scenarios: list[dict[str, Any]] = []
    for label, lift in (("Conservative (1.5x lift)", 1.5), ("Target ML (2.5x lift)", 2.5), ("Strong ML (3.5x lift)", 3.5)):
        precision = min(0.95, round(base_rate * lift, 4))
        true_positives = round(flagged_count * precision)
        gross_saved = round(true_positives * effective_value_per_tp, 2)
        total_cost = round(flagged_count * action_cost, 2)
        net_gain = round(gross_saved - total_cost, 2)
        scenarios.append({
            "scenario": label,
            "precision_at_top_k": precision,
            "flagged_per_month": flagged_count,
            "expected_saves": round(true_positives * action_success_rate, 1),
            "intervention_cost": total_cost,
            "gross_value": gross_saved,
            "net_monthly_value": net_gain,
            "viable": net_gain > 0,
        })

    econ = {
        "inputs": {
            "monthly_volume": int(round(monthly_volume)),
            "monthly_volume_assumed": volume_assumed,
            "base_rate": round(base_rate, 4),
            "base_rate_assumed": rate_assumed,
            "action_cost": round(action_cost, 2),
            "action_cost_assumed": cost_assumed,
            "value_per_positive": round(value_saved, 2),
            "value_per_positive_assumed": value_assumed,
            "action_success_rate": round(action_success_rate, 4),
            "action_success_rate_assumed": "action_success_rate" not in ov,
        },
        "expected_positives_per_month": int(expected_positives),
        "effective_value_per_true_positive": round(effective_value_per_tp, 2),
        "break_even_precision": break_even_precision,
        "recommended_action_rate": suggested_top_k_pct,
        "flagged_capacity_per_month": int(flagged_count),
        "scenarios": scenarios,
        "note": (
            "Planning math uses your stated numbers where given and marked defaults where missing — "
            "replace any assumed input once internal finance/ops numbers are confirmed."
        ),
    }
    labels = {"monthly_volume": "monthly volume", "base_rate": "base rate", "action_cost": "cost per action",
              "value_per_positive": "value per caught case", "action_success_rate": "share of actions that work"}
    econ["assumed"] = [label for key, label in labels.items() if econ["inputs"][f"{key}_assumed"]]
    # What a model may quote as a source (the lead agent's number check reads this): the person's own numbers only. A figure
    # DCLab assumed, and every figure computed from one, is no source: it is shown on the card, marked assumed, and nowhere else.
    stated = {key: econ["inputs"][key] for key in labels if not econ["inputs"][f"{key}_assumed"]}
    econ["as_said"] = [json.dumps({"no_data_stated_inputs": stated}, ensure_ascii=False)] if stated else []
    if not econ["assumed"]:  # every input is the person's: what code computed from them is theirs to quote
        econ["as_said"].append(json.dumps({"no_data_economics": {
            **stated, "break_even_precision": break_even_precision, "flagged_capacity_per_month": flagged_count, "scenarios": scenarios,
        }}, ensure_ascii=False))
        econ["as_said"].append(
            f"Decision math: at cost {econ['inputs']['action_cost']}, value {econ['inputs']['value_per_positive']}, and "
            f"{round(action_success_rate * 100)}% save success ({econ['effective_value_per_true_positive']} effective value per caught case), "
            f"break-even precision is {round(break_even_precision * 100, 1)}%. Across {econ['inputs']['monthly_volume']} units/month "
            f"({round(base_rate * 100, 1)}% base rate = {expected_positives} events), flagging top {round(suggested_top_k_pct * 100)}% "
            f"({flagged_count} cases) yields net monthly value {scenarios[1]['net_monthly_value']} under 2.5x lift."
        )
    return econ

def _virtual_schema_for_kind(kind_key: str, problem_text: str, target_hint: Optional[str] = None) -> dict[str, Any]:
    k = kind_key.lower()
    text_l = problem_text.lower()
    if k == "credit_risk" or any(w in text_l for w in ("loan", "underwriting", "default rate", "credit risk", "micro-lending", "وام", "نکول")):
        target = target_hint or "is_default_90d"
        return {
            "entity_id": "application_id / borrower_id",
            "prediction_time": "application_submitted_at",
            "horizon": "90+ days past due within 6-12 months post-origination",
            "tables": {
                "loan_applications_blueprint": [
                    "application_id", "borrower_id", "application_submitted_at", "requested_loan_amount",
                    "loan_term_months", "declared_monthly_income", "debt_to_income_ratio",
                    "borrower_tenure_days", "prior_delinquencies_12m", "bureau_utilization_pct",
                    "channel", target,
                ],
                "repayments_log": ["application_id", "due_date", "paid_at", "dpd_days"],
            },
            "tier1_features": ["requested_loan_amount", "debt_to_income_ratio", "borrower_tenure_days", "prior_delinquencies_12m", "bureau_utilization_pct", "declared_monthly_income"],
            "tier2_features": ["payment_to_income_ratio", "recent_credit_inquiry_velocity_30d", "historical_borrower_repayment_ratio"],
            "forbidden_traps": ["approved_interest_rate_post_decision", "collections_agency_assigned", "chargeoff_settled_flag", "recovery_amount"],
            "target": target,
            "unit": "loan application",
        }
    if k == "fraud" or "fraud" in text_l or "تقلب" in text_l:
        target = target_hint or "is_fraud_confirmed"
        return {
            "entity_id": "transaction_id / account_id",
            "prediction_time": "tx_authorized_at",
            "horizon": "30-60 days for chargeback/dispute settlement",
            "tables": {
                "transactions_blueprint": [
                    "transaction_id", "account_id", "tx_authorized_at", "amount", "currency",
                    "merchant_category", "channel", "device_age_days", "ip_country_mismatch",
                    "tx_count_1h", "tx_amount_24h", "account_tenure_days", target,
                ],
                "chargebacks_log": ["transaction_id", "reported_at", "chargeback_reason"],
            },
            "tier1_features": ["amount", "merchant_category", "channel", "ip_country_mismatch", "tx_count_1h", "account_tenure_days"],
            "tier2_features": ["velocity_ratio_24h_vs_30d", "device_account_fanout_7d", "merchant_historical_dispute_rate_asof"],
            "forbidden_traps": ["chargeback_filed_at", "manual_review_outcome", "account_frozen_after_tx", "refund_issued_flag"],
            "target": target,
            "unit": "transaction",
        }
    if k == "forecasting" or (
        k not in ("fraud", "churn", "regression", "segmentation")
        and any(w in text_l for w in ("forecast", "demand", "sales next", "inventory", "پیش‌بینی فروش", "پیش بینی فروش", "پیش‌بینی تقاضا", "پیش بینی تقاضا", "تقاضا"))
    ):
        target = target_hint or "target_demand_units"
        return {
            "entity_id": "sku_id / store_id",
            "prediction_time": "forecast_origin_date",
            "horizon": "h = 7 to 28 days ahead (strict origin cutoff)",
            "tables": {
                "daily_series_blueprint": [
                    "sku_id", "store_id", "date", "unit_price", "promo_active_planned",
                    "stock_on_hand_start", "day_of_week", "is_holiday", "lag_7d_units",
                    "rolling_28d_mean", target,
                ],
            },
            "tier1_features": ["lag_7d_units", "rolling_28d_mean", "day_of_week", "is_holiday", "unit_price", "promo_active_planned"],
            "tier2_features": ["stockout_adjusted_demand_28d", "same_store_category_trend", "lead_time_days"],
            "forbidden_traps": ["actual_stockout_during_horizon", "end_of_day_revenue", "future_markdown_applied"],
            "target": target,
            "unit": "sku-store-day",
        }
    if k == "churn" or (
        k not in ("fraud", "forecasting", "regression", "segmentation")
        and any(w in text_l for w in ("churn", "retention", "cancel", "attrition", "ریزش", "لغو اشتراک"))
    ):
        target = target_hint or "churned_within_30d"
        return {
            "entity_id": "customer_id",
            "prediction_time": "snapshot_date (e.g. monthly billing cutoff / T-30d)",
            "horizon": "next 30 days after snapshot_date (exclude immature accounts)",
            "tables": {
                "customer_snapshots_blueprint": [
                    "customer_id", "snapshot_date", "plan_tier", "monthly_fee", "tenure_months",
                    "active_days_last_30d", "usage_trend_30d_vs_90d", "support_tickets_30d",
                    "payment_failures_90d", "contract_days_remaining", target,
                ],
                "events_log": ["customer_id", "event_time", "event_type"],
            },
            "tier1_features": ["tenure_months", "monthly_fee", "plan_tier", "active_days_last_30d", "support_tickets_30d", "payment_failures_90d"],
            "tier2_features": ["usage_trend_30d_vs_90d", "nps_last_response", "seat_utilization_pct"],
            "forbidden_traps": ["cancellation_reason", "final_invoice_date", "retention_offer_accepted_after_snapshot", "account_closed_flag"],
            "target": target,
            "unit": "customer",
        }
    if k == "regression" or (
        k not in ("fraud", "forecasting", "churn", "segmentation")
        and any(w in text_l for w in ("price", "revenue", "ltv", "lifetime value", "claim severity", "expected loss", "duration", "eta", "قیمت", "درآمد"))
        and not any(c in text_l for c in ("costs $", "action cost", "intervention cost", "review costs", "call costs", "visit costs", "inspection costs"))
    ):
        target = target_hint or "target_value"
        return {
            "entity_id": "record_id",
            "prediction_time": "decision_timestamp",
            "horizon": "at decision time / fixed outcome window",
            "tables": {
                "entities_blueprint": [
                    "record_id", "decision_timestamp", "segment", "baseline_size_metric",
                    "historical_avg_value_asof", "channel", "region", "season_index", target,
                ],
            },
            "tier1_features": ["segment", "baseline_size_metric", "historical_avg_value_asof", "channel", "region"],
            "tier2_features": ["rolling_peer_benchmark_asof", "recency_days", "frequency_90d"],
            "forbidden_traps": ["settlement_fee_after_outcome", "actual_discount_granted_post_hoc", "post_event_adjustment"],
            "target": target,
            "unit": "record",
        }
    if k == "segmentation" or any(w in text_l for w in ("segment", "cluster", "cohort", "گروه", "سگمنت")):
        return {
            "entity_id": "entity_id",
            "prediction_time": "as_of_date",
            "horizon": "descriptive cohort stability across consecutive windows",
            "tables": {
                "behavior_profile_blueprint": [
                    "entity_id", "as_of_date", "recency_days", "frequency_90d",
                    "monetary_90d", "channel_mix", "tenure_days", "return_rate_90d",
                ],
            },
            "tier1_features": ["recency_days", "frequency_90d", "monetary_90d", "tenure_days", "channel_mix"],
            "tier2_features": ["category_entropy", "margin_per_order_90d", "interpurchase_cv"],
            "forbidden_traps": ["future_churn_label", "post_campaign_response"],
            "target": None,
            "unit": "entity",
        }
    target = target_hint or "outcome_label"
    return {
        "entity_id": "entity_id",
        "prediction_time": "decision_timestamp",
        "horizon": "fixed future window after decision_timestamp",
        "tables": {
            "decision_snapshots_blueprint": [
                "entity_id", "decision_timestamp", "segment", "tenure_days",
                "activity_count_30d", "historical_rate_asof", "recent_change_pct", target,
            ],
        },
        "tier1_features": ["segment", "tenure_days", "activity_count_30d", "historical_rate_asof"],
        "tier2_features": ["recent_change_pct", "peer_group_rate_asof", "RecencyFrequencyRatio"],
        "forbidden_traps": ["outcome_timestamp", "manual_intervention_taken_after_cutoff", "post_decision_status"],
        "target": target,
        "unit": "entity",
    }


def _gather_citations(problem_text: str, kind_key: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    queries = (f"{kind_key} leakage validation split", f"{problem_text[:120]} baseline evaluation")
    for q in queries:
        try:
            res = rnd_tools.search_evidence(q, k=3) or {}
            hits = res.get("results") or res.get("records") or []
        except Exception:  # noqa: BLE001
            hits = []
        for item in hits:
            cid = str(item.get("record_id") or item.get("id") or "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            citations.append({
                "id": cid,
                "title": str(item.get("title") or item.get("headline") or item.get("summary") or cid)[:140],
                "type": str(item.get("type") or "evidence"),
            })
    try:
        idx = evidence_index.EvidenceIndex.load()
        for rec in idx.search(f"{kind_key} leakage holdout", k=3):
            cid = str(rec.get("record_id") or rec.get("id") or "")
            if cid and cid not in seen:
                seen.add(cid)
                citations.append({
                    "id": cid,
                    "title": str(rec.get("title") or rec.get("finding") or cid)[:140],
                    "type": str(rec.get("type") or "evidence"),
                })
    except Exception:  # noqa: BLE001
        pass

    ex_out: list[dict[str, Any]] = []
    try:
        for ex in business_examples.find(f"{kind_key} {problem_text[:120]}", top=3):
            ex_out.append({
                "title": str(ex.get("title") or ex.get("name") or ex.get("track") or "Reference case"),
                "source": str(ex.get("source") or ex.get("id") or "business_examples"),
                "decision": str(ex.get("decision") or ex.get("action") or ex.get("summary") or "")[:180],
            })
    except Exception:  # noqa: BLE001
        pass
    return citations[:6], ex_out[:3]


def build_data_science_blueprint(
    problem_text: str,
    *,
    kind_key: str,
    schema: Mapping[str, Any],
    citations: list[dict[str, str]],
) -> dict[str, Any]:
    """Build Minimum Viable Data Spec, split protocol, metrics, and pre-data leakage checklist."""
    kind_obj = kinds.for_pack(kind_key)
    try:
        kc = rnd_tools.kind_checks(kind_key) or {}
    except Exception:  # noqa: BLE001
        kc = {}

    if kind_key == "forecasting":
        split_rule = "Rolling temporal origin backtest (train <= T_k, gap >= horizon, test at T_k + h); never random shuffle."
        metrics = ["wape", "mae", "rmse", "pinball_loss_p90"]
    elif kind_key in ("fraud", "churn", "classification"):
        split_rule = "Out-of-time snapshot split PLUS entity-group isolation (no entity appears in both train and holdout)."
        metrics = ["pr_auc", "roc_auc", "precision_at_top_k", "brier_score"]
    elif kind_key == "regression":
        split_rule = "Time-ordered split (or entity-grouped CV if no timestamp exists) with outlier-robust residual checks."
        metrics = ["mae", "rmse", "r2"]
    elif kind_key == "segmentation":
        split_rule = "Multi-window stability test: fit clusters on Window 1, measure Adjusted Rand / drift on Window 2."
        metrics = ["silhouette_score", "temporal_cluster_stability", "segment_actionability_lift"]
    else:
        split_rule = "Entity-grouped + time-ordered validation split keeping the final holdout strictly untouched until sign-off."
        metrics = list((getattr(kind_obj, "metrics", None) or kc.get("metrics") or ["roc_auc", "brier_score"]))

    pitfalls = [
        {
            "rule": "Strict as-of point-in-time join",
            "detail": f"Compute every feature strictly using events <= {schema['prediction_time']}.",
            "cite": citations[0]["id"] if citations else None,
        },
        {
            "rule": "Ban post-decision columns",
            "detail": f"Drop/audit post-cutoff fields before training: {', '.join(schema.get('forbidden_traps') or [])}.",
            "cite": citations[1]["id"] if len(citations) > 1 else None,
        },
        {
            "rule": "Lock label maturity window",
            "detail": f"Only assign negative labels after {schema['horizon']} has elapsed, or censoring bias inflates false negatives.",
            "cite": citations[2]["id"] if len(citations) > 2 else None,
        },
    ]

    return {
        "entity_id": schema["entity_id"],
        "prediction_time": schema["prediction_time"],
        "label_horizon": schema["horizon"],
        "target_definition": schema["target"] or "Unsupervised segment / KPI cluster",
        "virtual_tables": dict(schema["tables"]),
        "tier1_minimum_features": list(schema["tier1_features"]),
        "tier2_high_leverage_features": list(schema["tier2_features"]),
        "forbidden_leakage_columns": list(schema["forbidden_traps"]),
        "split_strategy": split_rule,
        "evaluation_metrics": [str(m) for m in metrics[:4]],
        "pitfall_checklist": pitfalls,
        "minimum_viable_rows": "2,000+ labeled entities (with >= 150 positive events) or >= 2 full seasonal cycles for time series.",
    }


def build_ml_blueprint(
    problem_text: str,
    *,
    kind_key: str,
    schema: Mapping[str, Any],
    economics: Mapping[str, Any],
    understanding: Optional[Mapping[str, Any]] = None,
    gateway: Any = None,
    project_id: Optional[str] = None,
    planner_cap_eur: float = 0.05,
) -> dict[str, Any]:
    """Decompose subproblems, route solutions, and invoke Task Catalog planner (TP-08) over virtual schemas."""
    from dclab_core import problem_map as PM
    from dclab_core import solutions as SL

    pmap = PM.decompose(problem_text or "predict business outcome")
    routed_subs: list[dict[str, Any]] = []
    for sp in pmap.sub_problems:
        try:
            rt = SL.route(sp, None, pmap, detected=kind_key)
            routed_subs.append({
                "id": sp.id,
                "title": sp.title,
                "needs": list(sp.needs or ()),
                "approach": rt.chosen or rt.family or "tabular_baseline",
                "reason": rt.reason,
            })
        except Exception:  # noqa: BLE001
            routed_subs.append({"id": sp.id, "title": sp.title, "needs": list(sp.needs or ()), "approach": "tabular_baseline", "reason": "default"})

    all_vcols = [str(c) for cols in (schema.get("tables") or {}).values() for c in (cols or [])]
    known_facts = {
        "target": schema.get("target"),
        "unit": schema.get("unit"),
        "entity_key": str(schema.get("entity_id") or "entity_id").split("/")[0].strip(),
        "time_column": str(schema.get("prediction_time") or "decision_timestamp").split("(")[0].strip(),
        "columns": all_vcols[:25],
        "virtual_schema": True,
        "no_data_yet": True,
        "cost_ratio": round(economics["inputs"]["action_cost"] / max(economics["effective_value_per_true_positive"], 1.0), 4),
        "daily_review_cap": max(1, round(economics["flagged_capacity_per_month"] / 30)),
    }
    if (understanding or {}).get("prediction_moment"):
        known_facts["prediction_moment"] = understanding["prediction_moment"]

    try:
        if gateway is None:  # no gateway was given: the rules' plan. Never the process's own gateway (a hidden live call)
            catalog_plan = {**task_catalog.deterministic_catalog_plan(problem_text, schema["tables"], known_facts),
                            "routed_to": "deterministic", "schema_source": "virtual_blueprint", "cost_usd": 0.0}
        else:
            catalog_plan = task_catalog.plan_with_model(
                problem_text,
                schema["tables"],
                known_facts=known_facts,
                project_id=project_id or "tp08_case",
                gateway=gateway,
                max_revisions=1,
                cap_eur=planner_cap_eur,
            )
    except Exception:  # noqa: BLE001
        catalog_plan = {"selected_tasks": [], "caught_traps": [], "clarifying_questions": [], "schema_source": "virtual_blueprint"}

    if kind_key == "forecasting":
        architecture = {
            "stage1_baseline": "Seasonal Naive / Rolling Mean + Calendar Ridge Regression (must beat before adding complexity).",
            "stage2_primary": "Global LightGBM / CatBoost over lagged, calendar & promo features with recursive/direct multi-horizon loss.",
            "stage3_decision_layer": "Quantile / conformal prediction intervals mapped directly to safety-stock & reorder cost asymmetry.",
        }
    elif kind_key in ("fraud", "churn", "classification", "credit_risk", "prediction", "binary_classification", "rare_event"):
        architecture = {
            "stage1_baseline": "L2-regularized Logistic Regression with missing-value indicators & target/frequency encoding inside CV folds.",
            "stage2_primary": "LightGBM / CatBoost with class weighting (or focal objective), early stopping on validation PR-AUC.",
            "stage3_decision_layer": (
                f"Isotonic / Platt probability calibration -> act when P(y=1) >= break-even {round(economics['break_even_precision'] * 100, 1)}% "
                f"capped at top {round(economics['recommended_action_rate'] * 100)}% capacity."
            ),
        }
    elif kind_key == "regression":
        architecture = {
            "stage1_baseline": "Median/Segment baseline -> Ridge / Huber Regression.",
            "stage2_primary": "Gradient Boosted Trees (LightGBM / CatBoost) with Huber or Tweedie loss for skewed targets.",
            "stage3_decision_layer": "Residual quantile bands + segment error audit before automated downstream pricing/SLA action.",
        }
    elif kind_key == "segmentation":
        architecture = {
            "stage1_baseline": "RFM / Rule-based business tiers checked against outcome conversion.",
            "stage2_primary": "Robust-scaled PCA/UMAP + KMeans / Gaussian Mixture / HDBSCAN selected via stability & silhouette.",
            "stage3_decision_layer": "Assign stable playbook per cluster and track migration between cohorts monthly.",
        }
    else:
        architecture = {
            "stage1_baseline": "Regularized linear model / simple heuristic baseline with strict leak-free CV.",
            "stage2_primary": "Gradient Boosted Decision Trees (LightGBM / CatBoost) tuned within nested CV.",
            "stage3_decision_layer": f"Probability calibration & cost threshold tuned to {round(economics['break_even_precision'] * 100, 1)}% break-even.",
        }

    return {
        "architecture": architecture,
        "problem_map_subproblems": routed_subs,
        "catalog_plan": {
            "schema_source": catalog_plan.get("schema_source") or "virtual_blueprint",
            "routed_to": catalog_plan.get("routed_to") or kind_key,
            "selected_tasks": catalog_plan.get("selected_tasks") or [],
            "caught_traps": catalog_plan.get("caught_traps") or [],
            "clarifying_questions": catalog_plan.get("clarifying_questions") or [],
            "cost_eur": float(catalog_plan.get("cost_usd") or catalog_plan.get("cost_eur") or 0.0),
        },
        "pre_training_deployment_rule": "Ship Stage 1 baseline + decision logging first to collect clean unbiased labels before automating Stage 2.",
    }


def simulation_questions(solution: Mapping[str, Any]) -> list[str]:
    """Questions DCLab must ask the user BEFORE designing or generating synthetic data."""
    econ = solution.get("economics") or {}
    inp = econ.get("inputs") or {}
    ds = solution.get("data_science") or {}
    tables = ds.get("virtual_tables") or {}
    first_table = next(iter(tables.values()), [])
    target = ds.get("target_definition") or "outcome"
    return [
        f"How many rows should I simulate (for example 2,000 or 5,000 {ds.get('entity_id', 'records')})?",
        f"Which columns from the blueprint ({', '.join(first_table[:6])}) do you want included, changed, or dropped?",
        f"What approximate rate or distribution should '{target}' have (currently modeled at ~{round(float(inp.get('base_rate', 0.08)) * 100, 1)}%)?",
        "Are there any specific relationships, segments, or constraints I should inject (or keep it purely realistic baseline noise)?",
    ]


def simulation_seed_description(solution: Mapping[str, Any]) -> str:
    """Short seed description if the user asks to design synthetic data."""
    ds = solution.get("data_science") or {}
    tables = ds.get("virtual_tables") or {}
    cols = next(iter(tables.values()), [])
    return (
        f"Problem: {solution.get('problem', '')}. "
        f"Target: {ds.get('target_definition') or 'outcome'}. "
        f"Suggested columns: {', '.join(cols)}."
    )


def to_markdown(solution: Mapping[str, Any]) -> str:
    """Render clean downloadable/copyable Markdown of the 3-Layer No-Data Solution Blueprint + AI Understanding."""
    und = solution.get("understanding") or {}
    ufields = und.get("fields") or {}
    hrep = solution.get("harness_report") or {}
    biz = solution.get("business") or {}
    ds = solution.get("data_science") or {}
    ml = solution.get("ml") or {}
    arch = ml.get("architecture") or {}
    debate = ml.get("debate") or {}
    econ = solution.get("economics") or {}
    inp = econ.get("inputs") or {}
    lines = [
        f"# DCLab No-Data Solution Blueprint ({solution.get('family', 'prediction')})",
        "",
        f"**Problem:** {solution.get('problem', '')}",
        f"**Data mode:** {solution.get('mode_label', 'Pre-data design (no uploaded file)')}",
        f"**Harness mode:** `{hrep.get('mode', 'deterministic_floor')}` · {gates_line(hrep)}",
        "",
        "## 0. Deep Problem & Domain Understanding (Epistemic Provenance)",
        f"- **Restatement:** {und.get('restatement') or solution.get('problem', '')}",
    ]
    for k, item in ufields.items():
        if isinstance(item, Mapping):
            badge = BADGES.get(str(item.get("status")), "DCLab default")
            lines.append(f"- **{k.replace('_', ' ').title()}** (`{badge}`): {item.get('value')} — *{item.get('basis', '')}*")
    if und.get("alternatives"):
        lines.append("- **Alternative Problem Framings Considered:**")
        for alt in und.get("alternatives") or []:
            lines.append(f"  - `{alt.get('family')}`: {alt.get('why_not')}")
    lines += [
        "",
        "## 1. Business Solution & Decision Economics",
        f"- **Decision owner & action:** {biz.get('decision_owner', '')} — {biz.get('operational_action', '')}",
        f"- **Day-0 non-ML baseline:** {biz.get('non_ml_baseline', '')}",
    ]
    said = lambda key: "assumed by DCLab" if inp.get(f"{key}_assumed") else "stated by you"  # noqa: E731
    if econ.get("assumed"):
        lines.append(f"- **Read this first:** {len(econ['assumed'])} of 5 inputs below are DCLab's placeholders ({', '.join(econ['assumed'])}), "
                     "not facts about your business. Every figure computed from them is an illustration until you give your own numbers.")
    lines += [
        f"- **Monthly volume:** {inp.get('monthly_volume')} ({said('monthly_volume')}) · **Base rate:** {round(float(inp.get('base_rate', 0)) * 100, 1)}% ({said('base_rate')})",
        f"- **Cost per action:** {inp.get('action_cost')} ({said('action_cost')}) · **Value per caught case:** {inp.get('value_per_positive')} ({said('value_per_positive')}) · "
        f"**Share of actions that work:** {round(float(inp.get('action_success_rate', 0)) * 100)}% ({said('action_success_rate')})",
        f"- **Break-even precision:** {round(float(econ.get('break_even_precision', 0)) * 100, 1)}%" + (" (an illustration: computed from assumed inputs)" if econ.get("assumed") else ""),
    ]
    for df in (biz.get("domain_expert_findings") or [])[:3]:
        lines.append(f"- **Domain Expert ({df.get('kind', 'insight')}):** {df.get('text', '')}")
    lines += [
        "",
        "### ROI Scenarios" + (" (an illustration: computed from assumed inputs; the lifts are DCLab's assumptions)" if econ.get("assumed") else " (the lifts are DCLab's assumptions)"),
        "| Scenario | Precision | Flagged/mo | Gross Value | Cost | Net/mo |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in econ.get("scenarios") or []:
        lines.append(
            f"| {s['scenario']} | {round(s['precision_at_top_k'] * 100, 1)}% | {s['flagged_per_month']} | {s['gross_value']} | {s['intervention_cost']} | {s['net_monthly_value']} |"
        )
    lines += [
        "",
        "## 2. Data Science Solution & Minimum Viable Data Spec",
        f"- **Unit / Entity:** `{ds.get('entity_id')}`",
        f"- **Prediction Cutoff Time:** `{ds.get('prediction_time')}`",
        f"- **Label Horizon:** {ds.get('label_horizon')}",
        f"- **Target:** `{ds.get('target_definition')}`",
        f"- **Tier-1 Day-1 Features:** {', '.join(ds.get('tier1_minimum_features') or [])}",
        f"- **Tier-2 High-Leverage Features:** {', '.join(ds.get('tier2_high_leverage_features') or [])}",
        f"- **Forbidden Post-Decision Columns:** {', '.join(ds.get('forbidden_leakage_columns') or [])}",
        f"- **Validation Split:** {ds.get('split_strategy')}",
        f"- **Metrics:** {', '.join(ds.get('evaluation_metrics') or [])}",
    ]
    if ds.get("verified_feature_programs"):
        lines.append(
            f"- **Verified Fold-Safe Feature Programs (`dclab_core.features`):** "
            + ", ".join(f"`{p.get('name')} ({p.get('op')})`" for p in (ds.get("verified_feature_programs") or [])[:6])
        )
    lines += [
        "",
        "### Pre-Data Leakage Traps",
    ]
    for pit in ds.get("pitfall_checklist") or []:
        lines.append(f"- **{pit['rule']}**" + (f" ({pit['cite']})" if pit.get("cite") else "") + f": {pit['detail']}")
    for lf in (ds.get("leakage_audit_findings") or [])[:4]:
        if lf.get("column"):
            lines.append(f"- **Leak Suspect `{lf['column']}` ({lf.get('action', 'exclude')}):** {lf.get('why', '')}")
        elif lf.get("text"):
            lines.append(f"- **Leakage Auditor ({lf.get('kind', 'risk')}):** {lf.get('text', '')}")
    lines += [
        "",
        "## 3. ML Solution, Multi-Model Debate & Pre-Data Task Plan",
        f"- **Stage 1 Baseline:** {arch.get('stage1_baseline', '')}",
        f"- **Stage 2 Primary:** {arch.get('stage2_primary', '')}",
        f"- **Stage 3 Decision Layer:** {arch.get('stage3_decision_layer', '')}",
        f"- **What Evidence Would Change Algorithm Choice:** {ml.get('would_change_trigger', '')}",
        debate_line(debate, ds),
        f"- **TP-08 Selected Pre-Data Tasks:** {', '.join((ml.get('catalog_plan') or {}).get('selected_tasks') or []) or 'baseline, leakage_check, threshold_for_cost'}",
        "",
        "## 4. Next Options (When Ready)",
        "- Tell me missing economics or operational constraints to tighten these numbers.",
        "- If you want a **synthetic prototype**, answer the simulation questions so I can show the schema design for your approval first.",
        "- When real data is ready, upload/connect it — `Build project` remains unchanged whenever you attach data.",
    ]
    return "\n".join(lines)


BADGES = {"user_stated": "Stated by you", "ai_inferred": "AI inferred", "rules_read": "Read by DCLab's rules", "dclab_default": "DCLab default"}
GATE_NAMES = {"gate_1_epistemic_provenance": "provenance", "gate_2_number_grounding": "numbers", "gate_3_virtual_schema_and_features": "schema and features",
              "gate_4_workflow_and_debate": "workflow", "gate_5_budget": "budget"}


def gates_line(report: Mapping[str, Any]) -> str:
    """The gates as counted: how many passed, and the names of the ones that did not."""
    gates = report.get("gates") or {}
    if not gates:
        return "Gates: not run"
    failed = [GATE_NAMES.get(name, name) for name, gate in gates.items() if not gate.get("passed")]
    return f"Gates passed: `{len(gates) - len(failed)}/{len(gates)}`" + (f" · not passed: {', '.join(failed)}" if failed else "")


def _split_words(choice: Any) -> str:
    if isinstance(choice, Mapping):
        return str(choice.get("kind") or choice.get("strategy") or "") + "".join(f", {k} {v}" for k, v in choice.items() if k not in ("kind", "strategy") and v)
    return str(choice or "")


def debate_line(debate: Mapping[str, Any], ds: Mapping[str, Any]) -> str:
    """The debate's verdict when models debated; when none did, the rules' plan under its own name, with the same split the
    card gives above (a second, different split would contradict it)."""
    sources = [str(v) for v in (debate.get("sources") or {}).values()]
    by_models = bool(sources) and not any(v.startswith("fallback") for v in sources)
    families = ", ".join(debate.get("judged_families") or [])
    if by_models:
        return (f"- **Planner-Challenger-Judge Debate Verdict:** Split `{_split_words(debate.get('judged_split')) or 'see the validation split above'}` · "
                f"Metric `{debate.get('judged_metric') or 'see the metrics above'}` · Families `{families}`")
    return (f"- **DCLab's rules' plan (no model debate ran):** Split: {ds.get('split_strategy') or 'see the validation split above'} · "
            f"Metric `{(ds.get('evaluation_metrics') or ['see the metrics above'])[0]}`" + (f" · Families `{families}`" if families else ""))


def concise_summary(solution: Mapping[str, Any]) -> str:
    """Short, number-grounded summary (<220 words) safe under CaseAgent's 300-word & cited.unsaid guards."""
    und = solution.get("understanding") or {}
    ufields = und.get("fields") or {}
    econ = solution.get("economics") or {}
    ds = solution.get("data_science") or {}
    ml = solution.get("ml") or {}
    arch = ml.get("architecture") or {}
    scenarios = econ.get("scenarios") or []
    target_sc = scenarios[1] if len(scenarios) > 1 else (scenarios[0] if scenarios else {})
    moment_val = ((ufields.get("prediction_moment") or {}).get("value") or ds.get("prediction_time") or "decision cutoff")
    return (
        f"You can start without uploading data. I put together your 3-layer blueprint below for **{solution.get('family', 'prediction')}**:\n\n"
        f"0. **Problem Understanding:** Predict `{ds.get('target_definition')}` per `{ds.get('entity_id')}` strictly at `{moment_val}`.\n"
        + (f"1. **Business & Economics:** I had no numbers from you for {', '.join(econ['assumed'])}, so the card's money figures use "
           "placeholders, each marked \"assumed by DCLab\". They are an illustration, not facts about your business.\n" if econ.get("assumed") else
           f"1. **Business & Economics:** From your numbers, break-even precision is **{round(float(econ.get('break_even_precision', 0)) * 100, 1)}%** "
           f"(net **{target_sc.get('net_monthly_value', 0)}/mo** in the middle scenario, whose lift is DCLab's assumption).\n") +
        f"2. **Data Science Blueprint:** Track `{ds.get('entity_id')}` at `{ds.get('prediction_time')}` ({ds.get('label_horizon')}), "
        f"excluding post-decision traps like {', '.join((ds.get('forbidden_leakage_columns') or [])[:2])}.\n"
        f"3. **ML & Debate Plan:** Start with `{arch.get('stage1_baseline', 'regularized baseline')}`, then move to gradient boosted trees calibrated to your decision threshold.\n\n"
        f"Tell me your real volume, base rate, cost per action and value per caught case, and I will redo the figures. "
        f"And if you'd like me to **simulate a synthetic dataset** to test this flow, tell me how many rows and what distributions you want so I can design it for your approval first."
    )


def solve_without_data(
    problem_text: str,
    *,
    understanding: Optional[Mapping[str, Any]] = None,
    case_facts: Optional[Mapping[str, Any]] = None,
    economics_overrides: Optional[Mapping[str, Any]] = None,
    withholding_privacy: bool = False,
    gateway: Any = None,
    models: Any = None,
    draft_id: Optional[str] = None,
    project_id: Optional[str] = None,
    cap_eur: float = 0.25,
    target_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Build the full honest 3-layer Business + Data Science + ML blueprint with multi-model AI & 5-Gate Harness."""
    from . import no_data_harness

    problem = str(problem_text or "Predict a business outcome without data yet").strip()
    family_key, family_source = _family_of(problem)
    fit = blueprint_fit(problem)

    schema = _virtual_schema_for_kind(family_key, problem, target_hint=target_hint or (understanding or {}).get("target"))
    econ = compute_decision_economics(
        problem,
        understanding=understanding,
        case_facts=case_facts,
        overrides=economics_overrides,
    )
    citations, examples = _gather_citations(problem, family_key)
    ds = build_data_science_blueprint(problem, kind_key=family_key, schema=schema, citations=citations)

    # Prepare structured virtual table floor for the 5-Gate AI Harness
    first_tname = next(iter(schema["tables"].keys()), "decision_snapshots_blueprint")
    first_ekey = str(schema["entity_id"]).split("/")[0].strip()
    first_tcol = str(schema["prediction_time"]).split("(")[0].strip()
    first_target = str(schema["target"] or "outcome_label")
    rules_vtable_list = [
        {
            "name": first_tname,
            "grain": f"One row per `{first_ekey}` at `{first_tcol}`",
            "entity_key": first_ekey,
            "time_col": first_tcol,
            "target": first_target,
            "must_have_columns": list(schema["tables"][first_tname]),
            "tier2_engineered_columns": list(schema["tier2_features"]),
            "forbidden_leakage_columns": list(schema["forbidden_traps"]),
        }
    ]
    harness_out = no_data_harness.run_ai_understanding_and_harness(
        problem,
        family_key=family_key,
        rules_virtual_tables=rules_vtable_list,
        rules_ml_arch={"primary_metrics": ds["evaluation_metrics"]},
        as_said=econ["as_said"],
        gateway=gateway,
        models=models,
        draft_id=draft_id,
        project_id=project_id,
        cap_eur=round(max(0.0, cap_eur) * 0.70, 4),  # the caller's cap decides: a case with little left gets little
    )

    # Update schema and Data Science blueprint from harnessed AI output
    enriched_vt_list = harness_out["virtual_tables"]
    if enriched_vt_list:
        evt0 = enriched_vt_list[0]
        tname0 = str(evt0.get("name") or first_tname)
        must0 = list(evt0.get("must_have_columns") or schema["tables"][first_tname])
        schema["tables"] = {tname0: must0, **{k: v for k, v in schema["tables"].items() if k != first_tname}}
        ds["virtual_tables"] = dict(schema["tables"])
        ds["virtual_table_specs"] = enriched_vt_list
        ds["tier1_minimum_features"] = [c for c in must0 if c not in (evt0.get("entity_key"), evt0.get("time_col"), evt0.get("target"))][:8] or ds["tier1_minimum_features"]
        ds["tier2_high_leverage_features"] = list(evt0.get("tier2_engineered_columns") or ds["tier2_high_leverage_features"])
        ds["forbidden_leakage_columns"] = list(evt0.get("forbidden_leakage_columns") or ds["forbidden_leakage_columns"])
    ds["verified_feature_programs"] = harness_out["verified_feature_programs"]
    ds["leakage_audit_findings"] = harness_out["leakage_audit_findings"]
    ds["understanding_critic_concerns"] = harness_out["understanding_critic_concerns"]
    ds["eda_plan"] = harness_out["understanding"]["eda_plan"]
    ds["preprocessing_plan"] = harness_out["understanding"]["preprocessing_plan"]

    enriched_und = dict(understanding or {})
    if not enriched_und.get("prediction_moment"):
        enriched_und["prediction_moment"] = harness_out["understanding"]["fields"]["prediction_moment"]["value"]

    ml = build_ml_blueprint(
        problem,
        kind_key=family_key,
        schema=schema,
        economics=econ,
        understanding=enriched_und,
        # the planner's own model call only when a gateway was given, live calls are allowed, and the harness's roles did
        # not already answer; its cost is added to the report below
        gateway=gateway if (gateway is not None and not no_data_harness.live_models_blocked()
                            and harness_out["harness_report"]["mode"] != "ai_harnessed") else None,
        project_id=project_id,
        planner_cap_eur=min(0.05, max(0.0, cap_eur * 0.2)),
    )
    planner_cost = float((ml.get("catalog_plan") or {}).get("cost_eur") or 0.0)
    if planner_cost:
        harness_out["harness_report"]["cost_eur"] = round(float(harness_out["harness_report"].get("cost_eur") or 0.0) + planner_cost, 6)
        harness_out["harness_report"].setdefault("roles_called", []).append("task_decomposer")
    if harness_out.get("subproblems_routed"):
        ml["problem_map_subproblems"] = [
            {
                "id": sp["id"],
                "title": sp["title"],
                "needs": sp.get("needs") or [],
                "approach": sp.get("chosen_route") or "tabular_baseline",
                "reason": sp.get("route_reason") or "",
                "source": sp.get("route_source") or "rules",
            }
            for sp in harness_out["subproblems_routed"]
        ]
    ml["debate"] = harness_out["debate"]
    ml["ranked_algorithms"] = harness_out["ranked_algorithms"]
    ml["would_change_trigger"] = harness_out["would_change_trigger"]
    ml["map_review"] = harness_out["map_review"]

    und = understanding or {}
    facts = case_facts or {}
    decision_val = (
        und.get("action")
        or ((facts.get("decision") or {}).get("value") if isinstance(facts.get("decision"), Mapping) else facts.get("decision"))
        or harness_out["understanding"]["fields"]["decision_action"]["value"]
        or f"Rank `{schema['unit']}` by predicted risk/value and trigger top-{round(econ['recommended_action_rate'] * 100)}% action."
    )
    user_val = (
        (facts.get("user") or {}).get("value") if isinstance(facts.get("user"), Mapping) else facts.get("user")
    ) or "Operations / Analytics Lead"

    business = {
        "decision_owner": str(user_val),
        "operational_action": str(decision_val),
        "non_ml_baseline": (
            "Scorecard heuristic using top 3 domain risk rules + randomized control holdout to measure baseline lift before ML."
        ),
        "domain_expert_findings": harness_out["domain_expert_findings"],
        "privacy_note": (
            "Designed zero-upload: keep raw tables local/internal and only share schema or approved synthetic parameters."
            if withholding_privacy
            else "Pre-data architecture: no dataset uploaded yet; ready for schema review or approved synthetic prototyping."
        ),
        "matched_cases": examples,
    }

    solution: dict[str, Any] = {
        "version": 2,
        "problem": problem,
        "family": family_key,
        "family_source": family_source,  # "kind" or "words": found in what the person wrote; "default": nothing pointed anywhere
        "fit": fit,
        "mode_label": "Private / Zero-Upload Design" if withholding_privacy else "No-Data Solution Design",
        "understanding": harness_out["understanding"],
        "business": business,
        "economics": econ,
        "data_science": ds,
        "ml": ml,
        "debate": harness_out["debate"],
        "harness_report": harness_out["harness_report"],
        "citations": citations,
        "as_said": econ["as_said"],
    }
    solution["simulation"] = {
        "requires_user_approval": True,
        "questions": simulation_questions(solution),
        "seed_description": simulation_seed_description(solution),
    }
    solution["markdown"] = to_markdown(solution)
    solution["summary"] = concise_summary(solution)
    if not fit["fits"]:  # a caller that built it anyway is told, in the words the person reads first
        solution["summary"] = ("Note: your words name nothing to predict, so this is a general template, not a plan for your problem. "
                               + solution["summary"])
    no_data_harness.finish_gates(solution, problem)  # gate 2 on the model-written text as it is shown, with the assumed inputs named
    solution["markdown"] = to_markdown(solution)
    return solution

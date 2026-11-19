"""Invented tables for the planning practice set (planning_practice_v1).

Each ``make_<id>(seed)`` returns ``{file name: DataFrame}``. The first entry is
the main table and holds the target; any further entries are related tables
joined to it by a key column. Only numpy and pandas are used, every draw comes
from ``np.random.default_rng(seed)``, and the same seed gives the same tables.

Every value is invented. No row comes from a real company. The traps planted in
each case are listed in ``cases.json`` beside this file; the honest columns are
honest by construction (each is computed only from what is known before the
prediction moment of its case).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ids(prefix: str, n: int, start: int = 1, width: int = 4) -> np.ndarray:
    return np.array([f"{prefix}{i:0{width}d}" for i in range(start, start + n)])


def _iso(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(ts).strftime("%Y-%m-%d"))


def _prior_count(entity: np.ndarray, order_key: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """How many earlier rows of the same entity had ``flag`` (rows ordered by ``order_key``)."""
    frame = pd.DataFrame({"e": entity, "k": order_key, "f": flag.astype(int)})
    frame = frame.sort_values(["e", "k"], kind="mergesort")
    prior = frame.groupby("e")["f"].cumsum() - frame["f"]
    return prior.reindex(range(len(entity))).to_numpy()


# --------------------------------------------------------------------------- pr01
def make_pr01(seed: int) -> dict[str, pd.DataFrame]:
    """Online fashion retailer: will an order be returned?"""
    rng = np.random.default_rng(seed)
    n_cust, n = 160, 1000
    cust = _ids("C", n_cust, start=10001, width=5)
    cust_eff = rng.normal(0.0, 1.5, n_cust)
    weight = rng.gamma(1.5, 1.0, n_cust)
    ci = rng.choice(n_cust, n, p=weight / weight.sum())

    order_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    category = rng.choice(["apparel", "shoes", "home", "electronics", "beauty"], n, p=[0.35, 0.2, 0.2, 0.15, 0.1])
    base_price = {"apparel": 45, "shoes": 80, "home": 35, "electronics": 140, "beauty": 25}
    price = np.round(np.array([base_price[c] for c in category]) * rng.lognormal(0, 0.4, n), 2)
    discount_pct = rng.choice([0, 10, 20, 30, 50], n, p=[0.45, 0.2, 0.15, 0.12, 0.08])
    n_items = 1 + rng.poisson(0.8, n)
    shipping_method = rng.choice(["standard", "express", "pickup"], n, p=[0.6, 0.25, 0.15])
    device = rng.choice(["app", "web_mobile", "web_desktop"], n, p=[0.45, 0.35, 0.2])

    logit = (
        -1.7
        + cust_eff[ci]
        + 0.7 * (category == "apparel")
        + 0.5 * (category == "shoes")
        + 0.015 * discount_pct
        + 0.25 * (n_items - 1)
        - 0.4 * (shipping_method == "pickup")
    )
    returned = rng.random(n) < _sigmoid(logit)
    refund_amount = np.where(returned, np.round(price * n_items * rng.uniform(0.5, 1.0, n), 2), 0.0)
    # A few kept orders get a small goodwill refund (price match, late parcel).
    goodwill = (~returned) & (rng.random(n) < 0.04)
    refund_amount = np.where(goodwill, np.round(rng.uniform(2, 15, n), 2), refund_amount)

    prior_returns = _prior_count(cust[ci], order_date.to_numpy(), returned)
    orders = pd.DataFrame(
        {
            "order_id": _ids("O", n, start=500001, width=6),
            "customer_id": cust[ci],
            "order_date": _iso(order_date),
            "category": category,
            "price": price,
            "discount_pct": discount_pct,
            "n_items": n_items,
            "shipping_method": shipping_method,
            "device": device,
            "prior_returns": prior_returns,
            "refund_amount": refund_amount,
            "returned": returned.astype(int),
        }
    )
    return {"orders.csv": orders.sort_values("order_date", kind="mergesort").reset_index(drop=True)}


# --------------------------------------------------------------------------- pr02
def make_pr02(seed: int) -> dict[str, pd.DataFrame]:
    """Parcel carrier: how many hours will a shipment take?"""
    rng = np.random.default_rng(seed)
    n = 1100
    week_idx = rng.integers(0, 52, n)
    t = week_idx / 51.0
    ship_week = _iso(pd.Timestamp("2025-01-06") + pd.to_timedelta(7 * week_idx, unit="D"))

    origin_hub = rng.choice(["HUB-N", "HUB-S", "HUB-E", "HUB-W", "HUB-C"], n)
    dest_zone = rng.choice(["metro", "suburban", "rural", "remote"], n, p=[0.4, 0.3, 0.2, 0.1])
    zone_km = {"metro": 60, "suburban": 160, "rural": 380, "remote": 720}
    distance_km = np.round(np.array([zone_km[z] for z in dest_zone]) * rng.lognormal(0, 0.35, n), 1)
    weight_kg = np.round(rng.lognormal(1.2, 0.8, n), 2)
    carrier = rng.choice(["NorthFreight", "SwiftLine", "ParcelCo"], n, p=[0.4, 0.35, 0.25])
    carrier_eff = {"NorthFreight": 0.0, "SwiftLine": -3.0, "ParcelCo": 4.0}
    pickup_hour = rng.integers(7, 20, n)
    promised_hours = np.ceil((12 + distance_km / 60) / 12) * 12

    # Drift: a capacity squeeze through the year slows every lane and makes distance cost more.
    delivery_hours = (
        8
        + 14 * t
        + distance_km / 45 * (1 + 0.6 * t)
        + 6 * (dest_zone == "remote")
        + np.array([carrier_eff[c] for c in carrier])
        + 0.8 * np.sqrt(weight_kg)
        + 0.6 * np.maximum(pickup_hour - 15, 0)
        + rng.gamma(2.0, 3.0, n)
    )
    delivery_hours = np.round(delivery_hours, 1)
    late_fee_charged = np.where(
        delivery_hours > promised_hours, np.round(5 + 2.5 * (delivery_hours - promised_hours), 2), 0.0
    )

    shipments = pd.DataFrame(
        {
            "shipment_id": _ids("SH", n, start=880001, width=6),
            "ship_week": ship_week,
            "origin_hub": origin_hub,
            "dest_zone": dest_zone,
            "distance_km": distance_km,
            "weight_kg": weight_kg,
            "carrier": carrier,
            "pickup_hour": pickup_hour,
            "promised_hours": promised_hours,
            "late_fee_charged": late_fee_charged,
            "delivery_hours": delivery_hours,
        }
    )
    order = rng.permutation(n)  # the export is not sorted by week
    return {"shipments.csv": shipments.iloc[order].reset_index(drop=True)}


# --------------------------------------------------------------------------- pr03
def make_pr03(seed: int) -> dict[str, pd.DataFrame]:
    """B2B SaaS: which workspaces will not renew?"""
    rng = np.random.default_rng(seed)
    n_acct = 200
    acct = _ids("A-", n_acct, start=1, width=4)
    acct_eff = rng.normal(0.0, 1.4, n_acct)
    industry = rng.choice(["retail", "logistics", "finance", "health", "media"], n_acct)
    region = rng.choice(["EMEA", "NA", "APAC"], n_acct, p=[0.45, 0.4, 0.15])
    employees = np.round(rng.lognormal(5, 1.1, n_acct)).astype(int)
    signup = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, n_acct), unit="D")
    accounts = pd.DataFrame(
        {"acct": acct, "industry": industry, "region": region, "employees": employees, "signup_date": _iso(signup)}
    )

    n_ws = np.clip(rng.poisson(5.5, n_acct), 1, 14)
    ai = np.repeat(np.arange(n_acct), n_ws)
    n = len(ai)
    month_idx = rng.integers(0, 30, n)  # renewals from 2023-01 to 2025-06
    t = month_idx / 29.0
    period = np.asarray(pd.period_range("2023-01", periods=30, freq="M").astype(str))[month_idx]

    plan = rng.choice(["starter", "team", "business"], n, p=[0.35, 0.45, 0.2])
    seats = np.round(rng.lognormal(2.3, 0.7, n)).astype(int) + 1
    logins_30d = rng.poisson(seats * rng.uniform(2, 12, n))
    tickets_90d = rng.poisson(1.2, n)
    nps_last = np.clip(np.round(rng.normal(7, 2, n)), 0, 10).astype(int)
    price_change_pct = rng.choice([0, 0, 0, 5, 8, 12], n)
    months_active = rng.integers(3, 48, n)

    usage = np.log1p(logins_30d / seats)
    # Drift: a cheaper competitor launched in 2024; churn climbs through the period.
    logit = (
        -2.4
        + acct_eff[ai]
        + 1.6 * t
        - 0.5 * (usage - usage.mean())
        + 0.25 * tickets_90d
        - 0.12 * (nps_last - 7)
        + 0.06 * price_change_pct
        - 0.3 * (plan == "business")
    )
    churned = rng.random(n) < _sigmoid(logit)
    outcome_code = np.where(
        churned,
        rng.choice(["CX-PRICE", "CX-FIT", "CX-BUDGET", "CX-MERGER"], n),
        rng.choice(["RN-AUTO", "RN-NEGOTIATED", "RN-UPSELL"], n, p=[0.7, 0.2, 0.1]),
    )

    subs = pd.DataFrame(
        {
            "workspace_id": _ids("WS", n, start=30001, width=5),
            "acct": acct[ai],
            "period": period,
            "plan": plan,
            "seats": seats,
            "logins_30d": logins_30d,
            "tickets_90d": tickets_90d,
            "nps_last": nps_last,
            "price_change_pct": price_change_pct,
            "months_active": months_active,
            "outcome_code": outcome_code,
            "churned": churned.astype(int),
        }
    )
    subs = subs.iloc[rng.permutation(n)].reset_index(drop=True)
    return {"renewals.csv": subs, "accounts.csv": accounts}


# --------------------------------------------------------------------------- pr04
def make_pr04(seed: int) -> dict[str, pd.DataFrame]:
    """Outpatient clinic: will the patient miss the appointment?"""
    rng = np.random.default_rng(seed)
    n_pat, n = 180, 1050
    patient = np.array([f"P{rng.integers(1000, 9999)}-{i:03d}" for i in range(n_pat)])
    pat_eff = rng.normal(0.0, 1.5, n_pat)
    age_band = rng.choice(["18-29", "30-44", "45-64", "65+"], n_pat, p=[0.25, 0.3, 0.3, 0.15])
    weight = rng.gamma(1.2, 1.0, n_pat)
    pi = rng.choice(n_pat, n, p=weight / weight.sum())

    start = pd.Timestamp("2024-01-01 08:00")
    booking_minutes = rng.integers(0, 545 * 24 * 60, n)
    booking_date = start + pd.to_timedelta(booking_minutes, unit="min")
    lead_days = rng.integers(0, 60, n)
    appt_date = (booking_date + pd.to_timedelta(lead_days, unit="D")).normalize() + pd.to_timedelta(
        rng.integers(8, 18, n), unit="h"
    )
    clinic = rng.choice(["North", "Riverside", "Central"], n)
    appt_type = rng.choice(["follow_up", "new_patient", "physio", "imaging"], n, p=[0.45, 0.2, 0.25, 0.1])
    after_change = booking_date >= pd.Timestamp("2025-01-01")
    # Drift: reminder phone calls were cut at the start of 2025 and no-shows rose.
    reminder_sent = rng.random(n) < np.where(after_change, 0.35, 0.9)
    t = booking_minutes / booking_minutes.max()

    logit = (
        -2.2
        + pat_eff[pi]
        + 0.025 * lead_days
        + 0.4 * (appt_type == "physio")
        - 0.6 * reminder_sent
        + 1.1 * after_change
        + 0.4 * t
        + 0.3 * (age_band[pi] == "18-29")
    )
    no_show = rng.random(n) < _sigmoid(logit)

    # prior_no_shows: misses of this patient on appointments that took place before this booking.
    prior = np.zeros(n, dtype=int)
    frame = pd.DataFrame({"p": pi, "b": booking_date, "a": appt_date, "m": no_show})
    for _, g in frame.groupby("p"):
        a = g["a"].to_numpy()
        m = g["m"].to_numpy()
        for idx, b in zip(g.index, g["b"].to_numpy()):
            prior[idx] = int(((a < b) & m).sum())

    appts = pd.DataFrame(
        {
            "appt_id": _ids("AP", n, start=70001, width=5),
            "patient_ref": patient[pi],
            "booking_date": booking_date,
            "appt_date": appt_date,
            "lead_days": lead_days,
            "clinic": clinic,
            "appt_type": appt_type,
            "age_band": age_band[pi],
            "reminder_sent": reminder_sent.astype(int),
            "prior_no_shows": prior,
            "no_show": no_show.astype(int),
        }
    )
    return {"appointments.csv": appts.sort_values("appt_date", kind="mergesort").reset_index(drop=True)}


# --------------------------------------------------------------------------- pr05
def make_pr05(seed: int) -> dict[str, pd.DataFrame]:
    """Injection-moulding plant: will a unit fail final inspection?"""
    rng = np.random.default_rng(seed)
    n_mach, n = 30, 1050
    serial = np.array([f"MX{rng.integers(10, 99)}-{rng.integers(100000, 999999)}" for _ in range(n_mach)])
    mach_eff = rng.normal(0.0, 1.3, n_mach)
    machines = pd.DataFrame(
        {
            "machine_serial": serial,
            "model": rng.choice(["Krona 220", "Krona 300", "Avant 180"], n_mach),
            "install_year": rng.integers(2009, 2024, n_mach),
            "vendor": rng.choice(["Krona", "Avant"], n_mach),
        }
    )
    mi = rng.integers(0, n_mach, n)
    produced_at = pd.Timestamp("2025-03-01") + pd.to_timedelta(rng.integers(0, 120 * 24 * 60, n), unit="min")
    shift = np.where(produced_at.hour < 6, "night", np.where(produced_at.hour < 14, "early", "late"))
    line = np.array([f"L{(i % 5) + 1}" for i in mi])
    batch_temp_c = np.round(rng.normal(232, 6, n), 1)
    pressure_bar = np.round(rng.normal(88, 5, n), 1)
    operator_tenure_months = rng.integers(1, 120, n)
    material_lot = np.array([f"LOT{v}" for v in rng.integers(4100, 4140, n)])
    cycle_time_s = np.round(rng.normal(31, 2.5, n), 2)

    logit = (
        -2.6
        + mach_eff[mi]
        + 0.06 * np.abs(batch_temp_c - 232)
        + 0.05 * np.abs(pressure_bar - 88)
        - 0.008 * operator_tenure_months
        + 0.35 * (shift == "night")
    )
    defect = rng.random(n) < _sigmoid(logit)
    rework_minutes = np.where(defect, np.round(rng.gamma(3.0, 9.0, n), 0), 0.0)
    touch_up = (~defect) & (rng.random(n) < 0.05)
    rework_minutes = np.where(touch_up, rng.integers(1, 6, n).astype(float), rework_minutes)

    insp = pd.DataFrame(
        {
            "unit_id": _ids("U", n, start=200001, width=6),
            "machine_serial": serial[mi],
            "produced_at": produced_at.strftime("%Y-%m-%d %H:%M"),
            "shift": shift,
            "line": line,
            "batch_temp_c": batch_temp_c,
            "pressure_bar": pressure_bar,
            "cycle_time_s": cycle_time_s,
            "operator_tenure_months": operator_tenure_months,
            "material_lot": material_lot,
            "rework_minutes": rework_minutes,
            "defect": defect.astype(int),
        }
    )
    return {"inspections.csv": insp, "machines.csv": machines}


# --------------------------------------------------------------------------- pr06
def make_pr06(seed: int) -> dict[str, pd.DataFrame]:
    """Accounts receivable: how many days until an invoice is paid?"""
    rng = np.random.default_rng(seed)
    n = 1000
    day_idx = np.sort(rng.integers(0, 912, n))  # 2023-01-01 .. 2025-06-30
    t = day_idx / 911.0
    issued = pd.Timestamp("2023-01-01") + pd.to_timedelta(day_idx, unit="D")
    terms_days = rng.choice([15, 30, 45, 60], n, p=[0.15, 0.5, 0.2, 0.15])
    due = issued + pd.to_timedelta(terms_days, unit="D")
    segment = rng.choice(["smb", "mid_market", "enterprise", "public_sector"], n, p=[0.45, 0.3, 0.15, 0.1])
    amount = np.round(rng.lognormal(8.0, 1.0, n), 2)
    currency = rng.choice(["EUR", "GBP", "USD"], n, p=[0.6, 0.25, 0.15])
    payment_method = rng.choice(["bank_transfer", "direct_debit", "card"], n, p=[0.6, 0.3, 0.1])
    prior_late_pct = np.round(np.clip(rng.beta(2, 5, n) * 100, 0, 100), 1)
    seg_eff = {"smb": 2.0, "mid_market": 0.0, "enterprise": 6.0, "public_sector": 14.0}

    # Drift: a slowdown from 2024 on stretches payment times by about three weeks.
    days_to_pay = (
        terms_days * 0.85
        + 22 * t**1.5
        + np.array([seg_eff[s] for s in segment])
        + 1.5 * np.log(amount / 3000)
        + 0.15 * prior_late_pct
        - 8 * (payment_method == "direct_debit")
        + rng.normal(0, 6, n)
    )
    days_to_pay = np.maximum(np.round(days_to_pay), 0).astype(int)
    overdue = days_to_pay - terms_days
    # Collections log: a statement e-mail every 10 days while the invoice is open, plus a
    # weekly call once it is overdue; a few clients get an extra courtesy call.
    n_followups = days_to_pay // 10 + np.where(overdue > 0, 1 + overdue // 7, 0) + (rng.random(n) < 0.08)
    n_followups = n_followups.astype(int)

    inv = pd.DataFrame(
        {
            "invoice_no": [f"INV-{2023 + d // 365}-{i:05d}" for i, d in enumerate(day_idx)],
            "ts": _iso(issued),
            "due_date": _iso(due),
            "terms_days": terms_days,
            "segment": segment,
            "amount": amount,
            "currency": currency,
            "payment_method": payment_method,
            "prior_late_pct": prior_late_pct,
            "n_followups": n_followups,
            "days_to_pay": days_to_pay,
        }
    )
    return {"invoices.csv": inv}


# --------------------------------------------------------------------------- pr07
def make_pr07(seed: int) -> dict[str, pd.DataFrame]:
    """Software vendor's help desk: will a new ticket be escalated to engineering?"""
    rng = np.random.default_rng(seed)
    n = 900
    created_at = pd.Timestamp("2025-02-01") + pd.to_timedelta(rng.integers(0, 200 * 24 * 60, n), unit="min")
    channel = rng.choice(["email", "chat", "phone", "portal"], n, p=[0.4, 0.3, 0.1, 0.2])
    product_area = rng.choice(["billing", "integrations", "reporting", "mobile", "sso"], n)
    priority = rng.choice(["low", "normal", "high", "urgent"], n, p=[0.25, 0.5, 0.18, 0.07])
    customer_tier = rng.choice(["free", "pro", "enterprise"], n, p=[0.4, 0.4, 0.2])
    body_words = np.round(rng.lognormal(4.3, 0.7, n)).astype(int)
    has_attachment = (rng.random(n) < 0.3).astype(int)
    n_prior_tickets_30d = rng.poisson(0.9, n)
    sentiment_score = np.round(np.clip(rng.normal(-0.1, 0.4, n), -1, 1), 2)

    logit = (
        -2.0
        + 0.9 * (product_area == "integrations")
        + 0.7 * (product_area == "sso")
        + np.select([priority == "high", priority == "urgent"], [0.7, 1.4], 0.0)
        + 0.5 * (customer_tier == "enterprise")
        + 0.004 * (body_words - 70)
        + 0.5 * has_attachment
        - 0.6 * sentiment_score
    )
    escalated = rng.random(n) < _sigmoid(logit)
    days_to_resolve = np.round(
        np.where(escalated, rng.gamma(4.0, 2.5, n) + 3, rng.gamma(1.5, 0.8, n)), 1
    )

    tickets = pd.DataFrame(
        {
            "ticket_id": _ids("T", n, start=41001, width=5),
            "created_at": created_at,
            "channel": channel,
            "product_area": product_area,
            "priority": priority,
            "customer_tier": customer_tier,
            "body_words": body_words,
            "has_attachment": has_attachment,
            "n_prior_tickets_30d": n_prior_tickets_30d,
            "sentiment_score": sentiment_score,
            "days_to_resolve": days_to_resolve,
            "escalated": escalated.astype(int),
        }
    )
    return {"tickets.csv": tickets.sort_values("created_at", kind="mergesort").reset_index(drop=True)}


# --------------------------------------------------------------------------- pr08
def make_pr08(seed: int) -> dict[str, pd.DataFrame]:
    """Facilities group: tomorrow's electricity use per building meter."""
    rng = np.random.default_rng(seed)
    n_m, n_days = 26, 48
    meter = np.array([f"MTR-{rng.integers(100, 999)}{chr(65 + i % 26)}" for i in range(n_m)])
    btype = rng.choice(["office", "warehouse", "lab", "retail"], n_m, p=[0.4, 0.25, 0.15, 0.2])
    area = np.round(rng.lognormal(7.5, 0.5, n_m)).astype(int)
    meter_mult = rng.lognormal(0.0, 0.5, n_m)  # what the building's own kit and habits add
    tariff = rng.choice(["flat", "tou"], n_m, p=[0.6, 0.4])
    btype_k = {"office": 0.09, "warehouse": 0.05, "lab": 0.16, "retail": 0.11}

    mi = np.repeat(np.arange(n_m), n_days)
    di = np.tile(np.arange(n_days), n_m)
    ts = pd.Timestamp("2025-09-15") + pd.to_timedelta(di, unit="D")
    t = di / (n_days - 1)
    # Autumn into winter: forecast temperature falls over the period.
    temp_forecast_c = np.round(17 - 13 * t + rng.normal(0, 2.0, len(mi)), 1)
    dow = ts.dayofweek.to_numpy()
    is_holiday = np.isin(ts.strftime("%m-%d"), ["10-03", "11-01"]).astype(int)
    weekend = dow >= 5
    occ = np.where(weekend | (is_holiday == 1), 0.45, 1.0)

    base = area[mi] * np.array([btype_k[b] for b in btype])[mi] * meter_mult[mi]
    heat = 1 + 0.035 * np.maximum(16 - temp_forecast_c, 0)
    # Drift: electric vehicle chargers were added across the estate during the period.
    ev = 1 + 0.25 * t
    kwh = np.round(base * occ * heat * ev * rng.lognormal(0, 0.08, len(mi)), 1)

    readings = pd.DataFrame(
        {
            "meter_id": meter[mi],
            "ts": ts,
            "building_type": btype[mi],
            "floor_area_m2": area[mi],
            "tariff": tariff[mi],
            "temp_forecast_c": temp_forecast_c,
            "day_of_week": ts.day_name().to_numpy(),
            "is_holiday": is_holiday,
            "kwh": kwh,
        }
    )
    return {"meter_readings.csv": readings.sort_values(["ts", "meter_id"], kind="mergesort").reset_index(drop=True)}


# --------------------------------------------------------------------------- pr09
def make_pr09(seed: int) -> dict[str, pd.DataFrame]:
    """College: which course enrolments end in a pass, flagged in week 4?"""
    rng = np.random.default_rng(seed)
    n_stu = 300
    student = np.array([f"S{rng.integers(21, 25)}{i:05d}" for i in range(n_stu)])
    ability = rng.normal(0.0, 1.0, n_stu)
    students = pd.DataFrame(
        {
            "student_no": student,
            "programme": rng.choice(["BSc Computing", "BA Business", "BSc Nursing", "BEng Civil"], n_stu),
            "entry_score": np.round(np.clip(120 + 18 * (0.5 * ability + rng.normal(0, 0.87, n_stu)), 60, 180)).astype(int),
            "part_time": (rng.random(n_stu) < 0.2).astype(int),
            "age_band": rng.choice(["<21", "21-25", "26+"], n_stu, p=[0.55, 0.3, 0.15]),
        }
    )
    per = np.clip(rng.poisson(4, n_stu), 1, 8)
    si = np.repeat(np.arange(n_stu), per)
    n = len(si)
    course = np.array([f"{p}{c}" for p, c in zip(rng.choice(["CS", "BU", "NU", "CE", "MA"], n), rng.integers(101, 105, n))])
    course_diff = {c: d for c, d in zip(np.unique(course), rng.normal(0, 4, len(np.unique(course))))}
    term = rng.choice(["2024-T1", "2024-T2", "2025-T1", "2025-T2"], n)
    credits = rng.choice([10, 15, 20], n, p=[0.3, 0.5, 0.2])
    attendance_rate_wk4 = np.round(np.clip(0.75 + 0.08 * ability[si] + rng.normal(0, 0.12, n), 0, 1), 2)
    quiz_avg_wk4 = np.round(np.clip(62 + 8 * ability[si] + rng.normal(0, 12, n), 0, 100), 1)
    assignments_submitted_wk4 = rng.binomial(3, np.clip(0.75 + 0.1 * ability[si], 0.05, 0.98))
    lms_logins_wk4 = rng.poisson(np.exp(2.3 + 0.2 * ability[si]))

    final_grade = (
        52
        + 11 * ability[si]
        + 18 * (attendance_rate_wk4 - 0.75)
        + 0.15 * (quiz_avg_wk4 - 62)
        + 2 * (assignments_submitted_wk4 - 2)
        - np.array([course_diff[c] for c in course])
        + rng.normal(0, 7, n)
    )
    final_grade = np.round(np.clip(final_grade, 0, 100)).astype(int)
    passed = (final_grade >= 50).astype(int)

    enr = pd.DataFrame(
        {
            "enrolment_id": _ids("E", n, start=1, width=5),
            "student_no": student[si],
            "course_code": course,
            "term": term,
            "credits": credits,
            "attendance_rate_wk4": attendance_rate_wk4,
            "quiz_avg_wk4": quiz_avg_wk4,
            "assignments_submitted_wk4": assignments_submitted_wk4,
            "lms_logins_wk4": lms_logins_wk4,
            "final_grade": final_grade,
            "passed": passed,
        }
    )
    enr = enr.iloc[rng.permutation(n)].reset_index(drop=True)
    return {"enrolments.csv": enr, "students.csv": students}


# --------------------------------------------------------------------------- pr10
def make_pr10(seed: int) -> dict[str, pd.DataFrame]:
    """City hotel group: will a booking be cancelled?"""
    rng = np.random.default_rng(seed)
    n = 1100
    day_idx = rng.integers(0, 730, n)  # bookings made 2024-01-01 .. 2025-12-30
    booking_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(day_idx, unit="D")
    lead_time_days = np.round(rng.gamma(1.4, 40, n)).astype(int)
    arrival = booking_date + pd.to_timedelta(lead_time_days, unit="D")
    channel = rng.choice(["direct", "ota", "corporate", "travel_agent"], n, p=[0.3, 0.45, 0.15, 0.1])
    room_type = rng.choice(["standard", "superior", "suite"], n, p=[0.65, 0.28, 0.07])
    deposit_type = rng.choice(["none", "partial", "non_refundable"], n, p=[0.7, 0.15, 0.15])
    adults = rng.choice([1, 2, 3], n, p=[0.35, 0.55, 0.1])
    children = rng.choice([0, 0, 0, 1, 2], n)
    adr = np.round(rng.normal(130, 30, n) * np.where(room_type == "suite", 2.1, np.where(room_type == "superior", 1.3, 1.0)), 2)
    previous_cancellations = rng.poisson(0.25, n)
    special_requests = rng.poisson(0.7, n)

    flex = booking_date >= pd.Timestamp("2025-04-01")
    # Drift: a free-cancellation policy from April 2025 raised cancellations, most of all through OTAs.
    logit = (
        -1.9
        + 0.008 * lead_time_days
        + 0.5 * (channel == "ota")
        + flex * (0.9 + 0.8 * (channel == "ota"))
        - 2.0 * (deposit_type == "non_refundable")
        - 0.35 * special_requests
        + 0.6 * previous_cancellations
    )
    is_canceled = rng.random(n) < _sigmoid(logit)

    bookings = pd.DataFrame(
        {
            "booking_ref": [f"BK{rng.integers(10, 99)}{i:05d}" for i in range(n)],
            "booking_date": _iso(booking_date),
            "arrival_date": _iso(arrival),
            "lead_time_days": lead_time_days,
            "channel": channel,
            "room_type": room_type,
            "deposit_type": deposit_type,
            "adults": adults,
            "children": children,
            "adr": adr,
            "previous_cancellations": previous_cancellations,
            "special_requests": special_requests,
            "is_canceled": is_canceled.astype(int),
        }
    )
    return {"bookings.csv": bookings.sort_values("arrival_date", kind="mergesort").reset_index(drop=True)}


# --------------------------------------------------------------------------- pr11
def make_pr11(seed: int) -> dict[str, pd.DataFrame]:
    """Card issuer: is a card payment fraudulent?"""
    rng = np.random.default_rng(seed)
    n_cards, n = 260, 1400
    card = np.array([f"{rng.integers(4000, 4999)}-XXXX-{rng.integers(1000, 9999)}-{i:03d}" for i in range(n_cards)])
    compromised = rng.random(n_cards) < 0.09
    cards = pd.DataFrame(
        {
            "card_ref": card,
            "card_type": rng.choice(["debit", "credit", "prepaid"], n_cards, p=[0.5, 0.4, 0.1]),
            "card_age_months": rng.integers(1, 96, n_cards),
            "credit_limit": rng.choice([500, 1000, 2500, 5000, 10000], n_cards),
        }
    )
    weight = rng.gamma(2.0, 1.0, n_cards) * np.where(compromised, 2.5, 1.0)
    ci = rng.choice(n_cards, n, p=weight / weight.sum())

    minutes = rng.integers(0, 300 * 24 * 60, n)
    txn_time = pd.Timestamp("2025-01-01") + pd.to_timedelta(minutes, unit="min")
    t = minutes / minutes.max()
    merchant_cat = rng.choice(["grocery", "fuel", "travel", "digital_goods", "electronics", "restaurants"], n)
    amount = np.round(rng.lognormal(3.6, 1.0, n), 2)
    hour = txn_time.hour.to_numpy()
    country_mismatch = (rng.random(n) < 0.08).astype(int)
    device_new = (rng.random(n) < 0.15).astype(int)

    # Drift: fraud grows through the year and moves into digital goods from the summer.
    logit = (
        -5.0
        + 3.4 * compromised[ci]
        + 1.9 * t
        + 1.8 * (merchant_cat == "digital_goods") * (t > 0.5)
        + 0.25 * np.log(amount / 40)
        + 1.0 * country_mismatch
        + 0.8 * device_new
        + 0.5 * ((hour < 5) | (hour > 22))
    )
    fraud = rng.random(n) < _sigmoid(logit)
    chargeback_flag = np.where(fraud, rng.random(n) < 0.85, rng.random(n) < 0.01).astype(int)

    txns = pd.DataFrame(
        {
            "txn_id": _ids("TX", n, start=9000001, width=7),
            "card_ref": card[ci],
            "txn_time": txn_time,
            "amount": amount,
            "merchant_cat": merchant_cat,
            "country_mismatch": country_mismatch,
            "device_new": device_new,
            "chargeback_flag": chargeback_flag,
            "fraud": fraud.astype(int),
        }
    )
    txns = txns.sort_values("txn_time", kind="mergesort").reset_index(drop=True)
    return {"transactions.csv": txns, "cards.csv": cards}


# --------------------------------------------------------------------------- pr12
def make_pr12(seed: int) -> dict[str, pd.DataFrame]:
    """Garden-centre chain: what each store will sell in a week."""
    rng = np.random.default_rng(seed)
    n_s, n_w = 30, 40
    store = np.array([f"{s:03d}" for s in rng.choice(np.arange(101, 400), n_s, replace=False)])
    size = np.round(rng.lognormal(7.2, 0.35, n_s)).astype(int)
    store_mult = rng.lognormal(0.0, 0.45, n_s)  # location, catchment, manager: not in any column
    stores = pd.DataFrame(
        {
            "store_no": store,
            "size_sqm": size,
            "region": rng.choice(["North", "Midlands", "South", "Wales"], n_s),
            "format": rng.choice(["superstore", "high_street", "retail_park"], n_s, p=[0.3, 0.3, 0.4]),
            "opened_year": rng.integers(1995, 2022, n_s),
        }
    )
    si = np.repeat(np.arange(n_s), n_w)
    wi = np.tile(np.arange(n_w), n_s)
    t = wi / (n_w - 1)
    week = pd.Timestamp("2025-03-03") + pd.to_timedelta(7 * wi, unit="D")
    promo_flag = (rng.random(len(si)) < 0.2).astype(int)
    price_index = np.round(rng.normal(1.0, 0.04, len(si)), 3)
    holiday_week = np.isin(week.strftime("%m-%d"), ["04-14", "05-05", "05-26", "08-25"]).astype(int)
    temp_forecast_c = np.round(10 + 9 * np.sin(np.pi * t) + rng.normal(0, 2, len(si)), 1)

    footfall = size[si] * 1.6 * store_mult[si] * (1 + 0.03 * (temp_forecast_c - 12)) * rng.lognormal(0, 0.1, len(si))
    # Drift: a discount chain opened nearby in spring; sales sink about a third by autumn.
    units_sold = (
        footfall
        * 0.32
        * (1 - 0.35 * t)
        * (1 + 0.35 * promo_flag)
        * (1 - 1.5 * (price_index - 1))
        * (1 + 0.2 * holiday_week)
        * rng.lognormal(0, 0.08, len(si))
    )
    units_sold = np.round(units_sold).astype(int)
    footfall_prev_week = (
        pd.Series(np.round(footfall)).groupby(si).shift(1).to_numpy()
    )  # last week's door count: known before the week starts

    sw = pd.DataFrame(
        {
            "store_no": store[si],
            "week_start": _iso(week),
            "promo_flag": promo_flag,
            "price_index": price_index,
            "holiday_week": holiday_week,
            "temp_forecast_c": temp_forecast_c,
            "footfall_prev_week": footfall_prev_week,
            "units_sold": units_sold,
        }
    )
    sw = sw.iloc[rng.permutation(len(sw))].reset_index(drop=True)
    return {"store_weeks.csv": sw, "stores.csv": stores}


MAKERS = {name[len("make_"):]: fn for name, fn in sorted(globals().items()) if name.startswith("make_pr")}

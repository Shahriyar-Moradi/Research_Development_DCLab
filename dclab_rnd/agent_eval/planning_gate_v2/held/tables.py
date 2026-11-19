"""The held-out tables of the planning exam hp_gate_v2: one seeded generator per case, numpy and pandas only.

Every table is invented. Each trap named in ``cases.json`` (``enforceable_traps``) is planted for real here:

* a late column is written after the prediction moment, from the outcome itself, and predicts it strongly;
* a group column repeats its values, and its rows share a hidden effect on the outcome;
* a time column orders the rows, and the outcome drifts with it, so a random split is too optimistic.

Some columns have frightening names but are known at the prediction moment (an honest column): they must stay in.

Written for the exam only; read by the exam run, once. ``make_<case_id>(seed)`` returns ``{file name: DataFrame}``,
with the file names and columns that ``cases.json`` lists for the case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASE_IDS = ("hg2_01", "hg2_02", "hg2_03", "hg2_04", "hg2_05", "hg2_06", "hg2_07", "hg2_08")


def _rng(seed: int, case: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), case])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _repeat_pick(rng: np.random.Generator, k: int, n: int, shape: float = 1.6) -> np.ndarray:
    """``n`` draws from ``k`` members, some members far more often than others (people, machines and accounts repeat)."""
    weights = rng.pareto(shape, k) + 1.0
    return rng.choice(k, n, p=weights / weights.sum())


# --- hg2_01: insurance claims, flagged at filing -------------------------------------------------------------------
def make_hg2_01(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 1)
    n, holders = 3000, 1100
    risk = rng.normal(0.0, 1.0, holders)
    holder = _repeat_pick(rng, holders, n)
    days = np.sort(rng.integers(0, 900, n))
    filed = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D") + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    kinds = np.array(["water_damage", "theft", "collision", "fire", "glass"])
    kind = rng.choice(kinds, n, p=[0.28, 0.17, 0.33, 0.05, 0.17])
    base = {"water_damage": 2600.0, "theft": 4200.0, "collision": 3100.0, "fire": 9000.0, "glass": 450.0}
    amount = np.round(np.array([base[k] for k in kind]) * rng.lognormal(0.0, 0.6, n), 2)
    policy_age = rng.integers(5, 3650, n)
    channel = rng.choice(["agent", "online", "phone"], n, p=[0.35, 0.45, 0.20])
    prior = pd.Series(holder).groupby(holder).cumcount().to_numpy()  # claims of the same holder filed earlier: known at filing
    drift = days / 900.0
    logit = (-3.0 + 1.1 * risk[holder] + 0.7 * (kind == "theft") + 0.35 * np.log(amount / np.median(amount))
             - 0.3 * np.log1p(policy_age / 365.0) + 1.0 * drift + 0.12 * prior + 0.3 * (channel == "online"))
    fraud = rng.random(n) < _sigmoid(logit)
    # logged by the special investigations unit when its review closes, weeks after filing; fraud reviews run long
    hours = np.where(fraud, rng.gamma(6.0, 4.0, n), rng.gamma(1.6, 1.8, n))
    claims = pd.DataFrame({
        "claim_id": [f"CL-{700000 + i}" for i in range(n)],
        "policyholder_id": [f"PH-{h:05d}" for h in holder],
        "filed_at": filed,
        "incident_type": kind,
        "claim_amount": amount,
        "policy_age_days": policy_age,
        "prior_claims_count": prior,
        "channel": channel,
        "investigation_hours": np.round(hours, 1),
        "is_fraud": fraud.astype(int),
    })
    return {"claims.csv": claims}


# --- hg2_02: machine failures on a production line (two tables) ----------------------------------------------------
def make_hg2_02(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 2)
    m, days, every = 60, 170, 3
    models = rng.choice(["K200", "K300", "X9"], m, p=[0.4, 0.4, 0.2])
    model_effect = {"K200": 0.0, "K300": -0.3, "X9": 0.5}
    frailty = rng.normal(0.0, 1.0, m)
    starts = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, m), unit="D")
    history = rng.poisson(np.exp(0.9 * frailty))  # failures logged before the first reading here
    machines = pd.DataFrame({
        "machine_id": [f"MC-{i:03d}" for i in range(m)],
        "model": models,
        "site": rng.choice(["line_1", "line_2", "line_3"], m),
        "install_date": pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 2500, m), unit="D"),
    })
    rows = []
    for i in range(m):
        wear, failed_days, downtime = 0.0, np.zeros(days, dtype=bool), np.zeros(days)
        wear_at = np.zeros(days)
        for d in range(days):
            wear += rng.gamma(1.0, 1.0)
            wear_at[d] = wear
            hazard = _sigmoid(-6.0 + 1.5 * frailty[i] + 0.05 * wear + model_effect[models[i]])
            if rng.random() < hazard:
                failed_days[d] = True
                downtime[d] = rng.integers(120, 720)
                wear = 0.0
        for d in range(0, 150, every):
            window = slice(d, d + 7)  # the reading is taken in the morning; the next seven days, that day included
            rows.append({
                "machine_id": machines.machine_id[i],
                "reading_time": starts[i] + pd.Timedelta(days=d, hours=int(rng.integers(6, 9))),
                "temperature_c": round(58.0 + 0.12 * wear_at[d] + 1.5 * frailty[i] + rng.normal(0, 2.0), 2),
                "vibration_mm_s": round(max(0.2, 2.0 + 0.045 * wear_at[d] + 0.4 * frailty[i] + rng.normal(0, 0.5)), 3),
                "pressure_bar": round(6.0 + rng.normal(0, 0.35), 3),
                "runtime_hours": int(20 * d + rng.integers(0, 15)),
                "previous_failure_count": int(history[i] + failed_days[:d].sum()),  # failures before this reading: known when it is taken
                # minutes the machine stood still in the seven days after the reading, logged by the crew afterwards
                "downtime_minutes": int(downtime[window].sum() + rng.integers(0, 40)),
                "failure_within_7d": int(failed_days[window].any()),
            })
    readings = pd.DataFrame(rows).sort_values("reading_time", kind="stable").reset_index(drop=True)
    readings.insert(0, "reading_id", [f"RD-{i:06d}" for i in range(len(readings))])
    return {"readings.csv": readings, "machines.csv": machines}


# --- hg2_03: personal loans, judged at application ------------------------------------------------------------------
def make_hg2_03(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 3)
    n = 3500
    extract = pd.Timestamp("2025-06-30")
    days = np.sort(rng.integers(0, 1186, n))
    applied = pd.Timestamp("2022-01-01") + pd.to_timedelta(days, unit="D")
    income = np.round(rng.lognormal(np.log(11000), 0.45, n), -1)
    term = rng.choice([12, 24, 36, 48, 60], n, p=[0.1, 0.25, 0.3, 0.2, 0.15])
    loan = np.round(income * rng.uniform(1.0, 8.0, n), -2)
    employment = np.round(rng.gamma(2.0, 3.0, n), 1)
    dti = np.round(np.clip(rng.beta(2.5, 5.0, n) + 0.05 * (loan / income) / 8.0, 0.02, 0.95), 3)
    previous_default = (rng.random(n) < 0.08).astype(int)  # from the credit bureau, read at application
    macro = ((applied >= pd.Timestamp("2023-07-01")) & (applied < pd.Timestamp("2024-07-01"))).astype(float)
    logit = (-2.7 + 1.6 * previous_default + 2.2 * (dti - 0.35) - 0.07 * employment + 0.45 * np.log(loan / income / 4.0)
             + 0.7 * np.asarray(macro) + 0.1 * (term >= 48))
    will_default = rng.random(n) < _sigmoid(logit)
    default_at = applied + pd.to_timedelta(rng.integers(60, 365, n), unit="D")
    seen = will_default & (default_at <= extract)  # applications of the last year have not had their twelve months yet
    cured = (~seen) & (rng.random(n) < 0.15)
    # the collections system's risk score, computed once an account falls 90 days behind, back-filled for the rest
    score = np.where(seen, rng.normal(780, 60, n), np.where(cured, rng.normal(560, 80, n), rng.normal(310, 70, n)))
    loans = pd.DataFrame({
        "application_id": [f"AP-{300000 + i}" for i in range(n)],
        "applied_on": applied,
        "branch": rng.choice(["north", "south", "east", "west", "online"], n, p=[0.2, 0.2, 0.15, 0.15, 0.3]),
        "income_monthly": income,
        "loan_amount": loan,
        "term_months": term,
        "employment_years": employment,
        "debt_to_income": dti,
        "previous_default_flag": previous_default,
        "score_v2": np.round(np.clip(score, 0, 1000), 0).astype(int),
        "defaulted_12m": seen.astype(int),
    })
    return {"applications.csv": loans}


# --- hg2_04: hospital readmissions, judged at discharge -------------------------------------------------------------
def make_hg2_04(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 4)
    patients, end = 1000, pd.Timestamp("2024-12-31")
    groups = np.array(["cardiac", "respiratory", "orthopaedic", "digestive", "renal", "other"])
    dx_effect = {"cardiac": 0.5, "respiratory": 0.4, "orthopaedic": -0.6, "digestive": 0.0, "renal": 0.7, "other": -0.2}
    rows = []
    for p in range(patients):
        frailty = rng.normal(0.0, 1.0)
        age = int(np.clip(rng.normal(64, 15), 18, 99))
        sex = rng.choice(["F", "M"])
        admit = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 1000)))
        prior = 0
        while admit <= end:
            dx = rng.choice(groups, p=[0.2, 0.18, 0.15, 0.17, 0.1, 0.2])
            los = int(max(1, rng.gamma(2.0, 2.2 + 0.4 * max(frailty, 0))))
            discharge = admit + pd.Timedelta(days=los, hours=int(rng.integers(8, 18)))
            if discharge > end:
                break
            to = rng.choice(["home", "home_with_support", "care_facility"], p=[0.65, 0.2, 0.15])
            meds = int(rng.poisson(4 + 0.06 * (age - 60) if age > 60 else 4))
            logit = (-3.2 + 1.8 * frailty + 0.025 * (age - 64) + dx_effect[dx] + 0.06 * los + 0.25 * min(prior, 4)
                     + 0.4 * (to == "care_facility") + 0.05 * meds)
            back = rng.random() < _sigmoid(logit)
            rows.append({"patient_id": f"PT-{p:05d}", "admitted_at": admit, "discharged_at": discharge, "age": age, "sex": sex,
                         "diagnosis_group": dx, "length_of_stay_days": los, "n_prior_admissions": prior, "discharge_to": to,
                         "meds_on_discharge": meds,
                         # outpatient and emergency visits in the 30 days after discharge, counted by the clinic later
                         "followup_visits_30d": int(1 + rng.poisson(3.0) if back else rng.poisson(0.8)),
                         "readmitted_30d": int(back)})
            prior += 1
            gap = int(rng.integers(1, 31)) if back else int(rng.integers(31, 500))
            admit = discharge + pd.Timedelta(days=gap)
            if not back and rng.random() < 0.2:
                break  # most people are not admitted again
    frame = pd.DataFrame(rows).sort_values("discharged_at", kind="stable").reset_index(drop=True)
    frame.insert(0, "admission_id", [f"AD-{500000 + i}" for i in range(len(frame))])
    return {"admissions.csv": frame}


# --- hg2_05: online shop returns, judged at checkout (two tables) ---------------------------------------------------
def make_hg2_05(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 5)
    n, customers = 1200, 700
    propensity = rng.normal(0.0, 0.8, customers)
    customer = _repeat_pick(rng, customers, n)
    days = np.sort(rng.integers(0, 455, n))
    ordered = pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D") + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    channel = rng.choice(["web", "app", "marketplace"], n, p=[0.5, 0.35, 0.15])
    cats = np.array(["fashion", "shoes", "electronics", "home", "beauty"])
    price_of = {"fashion": 45.0, "shoes": 80.0, "electronics": 160.0, "home": 35.0, "beauty": 18.0}
    lines, returned, totals, contacts = [], np.zeros(n, dtype=int), np.zeros(n), np.zeros(n, dtype=int)
    for o in range(n):
        k = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.3, 0.15, 0.1]))
        order_cats = rng.choice(cats, k, p=[0.35, 0.15, 0.15, 0.2, 0.15])
        bracket = 0
        items = []
        for c in order_cats:
            sku = f"{c[:3].upper()}-{int(rng.integers(100, 160))}"
            size = rng.choice(["S", "M", "L", "XL"]) if c == "fashion" else (str(int(rng.integers(36, 46))) if c == "shoes" else None)
            items.append((sku, c, size))
            if c in ("fashion", "shoes") and rng.random() < 0.2:  # the same item in a second size: buy two, keep one
                other = rng.choice(["S", "M", "L", "XL"]) if c == "fashion" else str(int(rng.integers(36, 46)))
                items.append((sku, c, other))
                bracket = 1
        total, fashion = 0.0, 0
        discount_max = 0
        for sku, c, size in items:
            price = round(price_of[c] * float(rng.lognormal(0, 0.35)), 2)
            qty = int(rng.choice([1, 2], p=[0.85, 0.15]))
            disc = int(rng.choice([0, 10, 20, 40], p=[0.55, 0.2, 0.15, 0.1]))
            discount_max = max(discount_max, disc)
            total += price * qty * (1 - disc / 100)
            fashion += c in ("fashion", "shoes")
            lines.append({"order_id": f"OR-{800000 + o}", "sku": sku, "category": c, "size": size, "unit_price": price,
                          "quantity": qty, "discount_pct": disc})
        policy = float(ordered[o] >= pd.Timestamp("2024-10-01"))  # free returns from October 2024: returns rise after it
        logit = (-2.4 + 1.0 * propensity[customer[o]] + 1.4 * bracket + 0.9 * fashion / len(items) + 0.6 * (discount_max >= 40)
                 - 0.4 * (channel[o] == "app") + 0.6 * policy)
        returned[o] = int(rng.random() < _sigmoid(logit))
        totals[o] = round(total, 2)
        # contacts with customer service on the order so far; most are return requests made after delivery
        contacts[o] = 1 + rng.poisson(1.8) if returned[o] else rng.poisson(0.3)
    orders = pd.DataFrame({
        "order_id": [f"OR-{800000 + o}" for o in range(n)],
        "customer_id": [f"CU-{c:05d}" for c in customer],
        "ordered_at": ordered,
        "channel": channel,
        "shipping_country": rng.choice(["DE", "FR", "NL", "AT", "BE"], n, p=[0.45, 0.2, 0.15, 0.1, 0.1]),
        "order_total": totals,
        "contact_count_total": contacts,
        "returned": returned,
    })
    order_lines = pd.DataFrame(lines)
    order_lines.insert(0, "line_id", [f"LN-{i:07d}" for i in range(len(order_lines))])
    return {"orders.csv": orders, "order_lines.csv": order_lines}


# --- hg2_06: B2B sales leads, scored after their first week ---------------------------------------------------------
def make_hg2_06(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 6)
    n = 2800
    extract = pd.Timestamp("2025-03-31")
    days = np.sort(rng.integers(0, 790, n))
    created = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")
    source = rng.choice(["inbound", "webinar", "partner", "outbound", "paid_search"], n, p=[0.3, 0.15, 0.15, 0.25, 0.15])
    src_effect = {"inbound": 0.5, "webinar": 0.2, "partner": 0.6, "outbound": -0.6, "paid_search": -0.1}
    size = rng.choice(["1-10", "11-50", "51-200", "201-1000", "1000+"], n, p=[0.25, 0.3, 0.25, 0.15, 0.05])
    pages = rng.poisson(4, n)
    demo = (rng.random(n) < _sigmoid(-1.8 + 0.2 * pages)).astype(int)
    prior_churned = (rng.random(n) < 0.07).astype(int)  # the company was a customer once and left: known when the lead arrives
    logit = (-2.6 + np.array([src_effect[s] for s in source]) + 0.12 * pages + 1.1 * demo - 0.5 * prior_churned
             + 0.3 * np.isin(size, ["51-200", "201-1000"]) - 0.5 * days / 790.0)
    wins = rng.random(n) < _sigmoid(logit)
    won_at = created + pd.to_timedelta(rng.integers(10, 90, n), unit="D")
    seen = wins & (won_at <= extract)  # leads of the last weeks have not had their ninety days yet
    stages_won = rng.choice(["contract_sent", "negotiation", "proposal"], n, p=[0.6, 0.3, 0.1])
    stages_open = rng.choice(["new", "qualified", "proposal", "negotiation"], n, p=[0.35, 0.35, 0.2, 0.1])
    leads = pd.DataFrame({
        "lead_id": [f"LD-{40000 + i}" for i in range(n)],
        "created_at": created,
        "source": source,
        "industry": rng.choice(["retail", "logistics", "finance", "health", "manufacturing", "software"], n),
        "company_size": size,
        "country": rng.choice(["UK", "DE", "AE", "US", "NL"], n, p=[0.3, 0.2, 0.15, 0.25, 0.1]),
        "pages_viewed_first_week": pages,
        "demo_requested": demo,
        "prior_churned_account": prior_churned,
        # the furthest pipeline stage the lead has reached, read from the CRM on the day of the export
        "sales_stage_max": np.where(seen, stages_won, stages_open),
        "converted_90d": seen.astype(int),
    })
    return {"leads.csv": leads}


# --- hg2_07: household electricity, forecast a day ahead ------------------------------------------------------------
def make_hg2_07(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 7)
    meters, days = 40, 90
    dates = pd.date_range("2024-11-01", periods=days, freq="D")
    season = 9.0 - 6.0 * np.sin(np.linspace(0, np.pi, days))
    temp = season + np.cumsum(rng.normal(0, 0.8, days)) * 0.5
    holiday = np.isin(dates.strftime("%m-%d"), ["12-24", "12-25", "12-26", "12-31", "01-01"]) | (dates.dayofweek >= 5)
    heating = rng.choice(["gas", "heat_pump", "electric_resistance"], meters, p=[0.5, 0.3, 0.2])
    heat_coef = {"gas": 0.15, "heat_pump": 0.7, "electric_resistance": 1.6}
    size = rng.integers(1, 7, meters)
    level = rng.lognormal(0.0, 0.25, meters)  # each home's own habits, shared by all its days
    tariff = rng.choice(["flat", "time_of_use"], meters, p=[0.7, 0.3])
    rows = []
    for m in range(meters):
        noise = 0.0
        last = np.nan
        for d in range(days):
            noise = 0.6 * noise + rng.normal(0, 1.0)
            kwh = max(0.5, level[m] * (4.0 + 2.2 * size[m]) + heat_coef[heating[m]] * max(0.0, 16.0 - temp[d])
                      + 1.5 * holiday[d] + noise)
            rows.append({"meter_id": f"MT-{m:03d}", "reading_date": dates[d], "household_size": int(size[m]), "tariff": tariff[m],
                         "heating_type": heating[m], "temp_forecast_c": round(float(temp[d] + rng.normal(0, 1.0)), 1),
                         "is_holiday": int(holiday[d]), "previous_day_kwh": last,
                         # the highest demand of the day being forecast, read from the smart meter at its end
                         "peak_kw": round(kwh / 24.0 * float(rng.uniform(2.9, 3.1)), 3),
                         "kwh": round(kwh, 2)})
            last = round(kwh, 2)
    frame = pd.DataFrame(rows).sort_values(["reading_date", "meter_id"], kind="stable").reset_index(drop=True)
    return {"daily_usage.csv": frame}


# --- hg2_08: staff attrition, quarterly snapshots -------------------------------------------------------------------
def make_hg2_08(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 8)
    teams = 70
    team_effect = rng.normal(0.0, 0.7, teams)  # a team's manager and workload, shared by everyone in it
    snapshots = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"])
    extract = pd.Timestamp("2025-06-30")
    staff: list[dict] = []

    def hire(start: pd.Timestamp, tenure: int) -> None:
        i = len(staff)
        team = int(rng.integers(0, teams))
        staff.append({"employee_id": f"EM-{i:05d}", "team": team, "department": ["sales", "ops", "tech", "finance", "support"][team % 5],
                      "hired": start - pd.Timedelta(days=30 * tenure), "band": int(rng.integers(1, 6)),
                      "rating": int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.45, 0.25, 0.1])), "left": None})

    for _ in range(820):
        hire(snapshots[0], int(rng.integers(1, 120)))
    quarters = pd.date_range("2024-01-01", "2025-07-01", freq="QS")
    for q0, q1 in zip(quarters[:-1], quarters[1:]):
        for s in staff:
            if s["left"] is None and s["hired"] <= q0:
                tenure = (q0 - s["hired"]).days / 30.0
                reorg = 0.7 * (q0 >= pd.Timestamp("2024-07-01"))  # a reorganisation in mid-2024: more people leave after it
                hazard = _sigmoid(-2.5 + 0.9 * team_effect[s["team"]] + 0.3 * (3 - s["rating"]) - 0.006 * tenure - 0.1 * s["band"] + reorg)
                if rng.random() < hazard:
                    s["left"] = q0 + pd.Timedelta(days=int(rng.integers(0, (q1 - q0).days)))
        for _ in range(60):
            hire(q1, 0)
    rows = []
    for snap in snapshots:
        for s in staff:
            if s["hired"] > snap or (s["left"] is not None and s["left"] <= snap):
                continue
            leaves = s["left"] is not None and s["left"] <= snap + pd.Timedelta(days=182)
            rows.append({"snapshot_month": snap, "employee_id": s["employee_id"], "team_id": f"TM-{s['team']:03d}",
                         "department": s["department"], "tenure_months": int((snap - s["hired"]).days // 30),
                         "salary_band": s["band"], "previous_rating": s["rating"],  # the last annual review, already given
                         "overtime_hours_q": round(float(max(0.0, rng.normal(20 + 10 * team_effect[s["team"]], 8))), 1),
                         # HR cases opened this year as of the export (offboarding opens one to three for a leaver)
                         "hr_case_count_ytd": int(rng.poisson(0.4) + (rng.integers(1, 4) if leaves else 0)),
                         "left_within_6m": int(leaves)})
    assert extract > snapshots[-1] + pd.Timedelta(days=182)  # every label has had its six months
    return {"employees.csv": pd.DataFrame(rows)}


MAKERS = {cid: globals()[f"make_{cid}"] for cid in CASE_IDS}

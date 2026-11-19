"""The held-out tables of the planning exam hp_gate_v3: one seeded generator per case, numpy and pandas only.

Every table is invented. Each trap named in ``cases.json`` (``enforceable_traps``) is planted for real here:

* a late column is written after the prediction moment, from the outcome itself, and predicts it strongly;
* a group column repeats its values over the whole period, and its rows share a hidden effect on the outcome;
* a time column orders the rows, and the outcome drifts with it, so a random split is too optimistic.

Some columns have frightening names but are known at the prediction moment (an honest column): they must stay in.
Some cases repeat an entity (a customer, a shop) that the person never asks to generalise past: no group trap there.

Written for the exam only; read by the exam run, once. ``make_<case_id>(seed)`` returns ``{file name: DataFrame}``,
with the file names and columns that ``cases.json`` lists for the case; the first table holds the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASE_IDS = ("hg3_01", "hg3_02", "hg3_03", "hg3_04", "hg3_05", "hg3_06", "hg3_07", "hg3_08", "hg3_09", "hg3_10")


def _rng(seed: int, case: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), 3000 + case])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _repeat_pick(rng: np.random.Generator, k: int, n: int, shape: float = 1.8) -> np.ndarray:
    """``n`` draws from ``k`` members, some members far more often than others."""
    weights = rng.pareto(shape, k) + 1.0
    return rng.choice(k, n, p=weights / weights.sum())


def _iso(ts: pd.Series | pd.DatetimeIndex) -> list[str]:
    return [t.strftime("%Y-%m-%d") for t in pd.DatetimeIndex(ts)]


# --- hg3_01: vineyard blocks, yield estimated at veraison --------------------------------------------------------------
def make_hg3_01(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 1)
    parcels, years = 84, np.arange(2014, 2026)
    soil = rng.normal(0.0, 1.0, parcels)  # the hidden quality of each parcel
    variety = rng.choice(["syrah", "grenache", "chardonnay", "merlot"], parcels, p=[0.3, 0.25, 0.25, 0.2])
    v_base = {"syrah": 7.5, "grenache": 8.5, "chardonnay": 9.0, "merlot": 8.0}
    slope = rng.choice(["flat", "gentle", "steep"], parcels, p=[0.4, 0.4, 0.2])
    area = np.round(rng.uniform(0.6, 4.5, parcels), 2)
    planted = rng.integers(1975, 2012, parcels)
    rows = []
    for y in years:
        season = rng.normal(0.0, 0.8)
        for p in range(parcels):
            if rng.random() < 0.06:  # a block left fallow or replanted that year: no row
                continue
            age = y - planted[p]
            rain = max(80.0, rng.normal(420 + 60 * season, 70))
            frost = float(np.clip(rng.gamma(0.6, 4.0) - 1.5, 0, 60))  # spring frost damage, scouted in April: known
            clusters = max(5.0, rng.normal(22 + 3 * soil[p] + 1.2 * season, 3.0))  # bunches per vine, sampled at veraison
            t_ha = (v_base[variety[p]] + 1.3 * soil[p] + 0.9 * season + 0.004 * (rain - 420) - 0.06 * frost
                    + 0.18 * (clusters - 22) - 0.02 * max(0, age - 30) - 0.6 * (slope[p] == "steep") + rng.normal(0, 0.7))
            t_ha = max(1.0, t_ha)
            # picking bins (about 0.4 t each) delivered to the winery at harvest, counted from the weighbridge slips
            bins = int(round(t_ha * area[p] / 0.4 * rng.normal(1.0, 0.04)))
            rows.append({"parcel_code": f"P{p:03d}-{chr(65 + p % 7)}", "vintage": int(y), "grape_variety": variety[p],
                         "slope_class": slope[p], "area_ha": area[p], "vine_age_years": int(age),
                         "winter_rain_mm": round(rain, 0), "frost_damage_pct": round(frost, 1),
                         "clusters_per_vine_sample": round(clusters, 1), "bins_delivered": bins,
                         "yield_t_per_ha": round(t_ha, 2)})
    return {"blocks.csv": pd.DataFrame(rows)}


# --- hg3_02: used vans and cars at a trade auction, price estimated at cataloguing ------------------------------------
def make_hg3_02(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 2)
    n, sellers = 1800, 140
    seller = _repeat_pick(rng, sellers, n)
    seller_eff = rng.normal(0.0, 0.08, sellers)  # dealers who prepare their cars well get more
    days = np.sort(rng.integers(0, 760, n))
    catalogued = pd.Timestamp("2024-01-08") + pd.to_timedelta(days, unit="D")
    sold = catalogued + pd.to_timedelta(rng.integers(6, 22, n), unit="D")
    make = rng.choice(["Ford", "VW", "Toyota", "Renault", "BMW", "Kia"], n, p=[0.22, 0.2, 0.18, 0.15, 0.12, 0.13])
    base = {"Ford": 14000, "VW": 16000, "Toyota": 17000, "Renault": 11000, "BMW": 24000, "Kia": 13000}
    age = rng.integers(1, 13, n)
    km = np.round(np.maximum(3000, age * rng.normal(17000, 5000, n)), -2)
    body = rng.choice(["hatch", "estate", "suv", "van"], n, p=[0.35, 0.2, 0.3, 0.15])
    owners = np.minimum(1 + rng.poisson(age / 4.0), 6)
    accident = rng.random(n) < 0.12  # recorded on the history check before cataloguing: known
    grade = rng.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.2, 0.4, 0.22, 0.1])
    market = 1.0 - 0.32 * (days / 760.0)  # used prices fall steadily over the two years
    value = (np.array([base[m] for m in make]) * np.exp(-0.11 * age - 0.0000025 * km + 0.12 * (body == "suv")
             - 0.18 * accident + 0.06 * (grade - 3) + seller_eff[seller]) * market)
    price = np.round(value * rng.lognormal(0.0, 0.07, n), -1)
    reserve = np.round(value * rng.uniform(0.75, 0.95, n), -2)  # set by the seller before the sale: known
    # bids placed during the sale, from the auction log: hot lots draw many bids and fetch more than expected
    surprise = np.log(price / value)
    bids = np.maximum(1, rng.poisson(np.exp(1.5 + 6.0 * surprise + 0.9 * np.log(price / 6000.0)))).astype(int)
    lots = pd.DataFrame({
        "lot_no": [f"L{24000 + i}" for i in range(n)],
        "catalogued_on": _iso(catalogued),
        "sale_date": _iso(sold),
        "seller_id": [f"DLR-{s:04d}" for s in seller],
        "make": make, "body_type": body, "age_years": age, "odometer_km": km.astype(int),
        "previous_owners": owners, "accident_on_record": accident.astype(int), "condition_grade": grade,
        "reserve_price": reserve.astype(int), "bid_count": bids, "hammer_price": price.astype(int),
    })
    return {"lots.csv": lots}


# --- hg3_03: blood donors, will they give again within six months --------------------------------------------------------
def make_hg3_03(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 3)
    donors = 700
    loyalty = rng.normal(0.0, 1.0, donors)
    age = rng.integers(18, 70, donors)
    sex = rng.choice(["F", "M"], donors)
    blood = rng.choice(["O+", "A+", "B+", "O-", "A-", "AB+", "B-", "AB-"], donors, p=[0.35, 0.3, 0.1, 0.08, 0.07, 0.04, 0.03, 0.03])
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2025-06-30")  # the export: six months after the last donation kept
    rows = []
    for d in range(donors):
        t = start + pd.Timedelta(days=int(rng.integers(0, 900)))
        visits = []
        while t < end - pd.Timedelta(days=183):
            visits.append(t)
            gap = rng.gamma(2.0, 120 * np.exp(-0.5 * loyalty[d]))
            t = t + pd.Timedelta(days=int(84 + gap))
        n_prev = 0
        for i, v in enumerate(visits):
            later = [w for w in visits[i + 1:] if w <= v + pd.Timedelta(days=182)]
            hb = round(float(rng.normal(14.2 if sex[d] == "M" else 13.1, 1.0)), 1)
            reaction = int(rng.random() < 0.05)  # fainting or bruising at this donation: noted before the donor leaves
            rows.append({"donation_date": v, "donor_number": f"BD{d:05d}", "age": int(age[d] + (v - start).days // 365),
                         "sex": sex[d], "blood_group": blood[d], "previous_donations": n_prev,
                         "site_type": rng.choice(["fixed_centre", "mobile_unit"], p=[0.6, 0.4]),
                         "haemoglobin_g_dl": hb, "post_donation_reaction": reaction,
                         "_returns": bool(later)})
            n_prev += 1
    frame = pd.DataFrame(rows).sort_values("donation_date").reset_index(drop=True)
    # the return rate falls after the centre stopped sending postcards (mid-2023); reactions also discourage
    days = (frame["donation_date"] - start).dt.days.to_numpy()
    shift = np.where(frame["donation_date"] >= pd.Timestamp("2023-07-01"), -0.9, 0.0) - 0.0006 * days
    keep = rng.random(len(frame)) < _sigmoid(2.2 + shift - 1.2 * frame["post_donation_reaction"].to_numpy())
    returns = frame["_returns"].to_numpy() & keep
    # loyalty points on the donor's card as of the export: 100 for every donation ever made, the later ones included
    donor_total = frame.groupby("donor_number")["donation_date"].transform("count").to_numpy()
    points = 100 * (frame["previous_donations"].to_numpy() + 1 + returns * (1 + rng.integers(0, 2, len(frame)))) \
        + 25 * rng.integers(0, 3, len(frame)) + 10 * (donor_total > 6)
    frame["points_balance"] = points
    frame["gave_again_6m"] = returns.astype(int)
    frame = frame.drop(columns="_returns")
    cols = ["donation_date", "donor_number", "age", "sex", "blood_group", "previous_donations", "site_type",
            "haemoglobin_g_dl", "post_donation_reaction", "points_balance", "gave_again_6m"]
    frame = frame[cols]
    if len(frame) > 2000:
        frame = frame.iloc[np.sort(rng.choice(len(frame), 2000, replace=False))].reset_index(drop=True)
    return {"donations.csv": frame}


# --- hg3_04: a mobile puzzle game, will a new player pay in the first two weeks ------------------------------------------
def make_hg3_04(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 4)
    n = 2000
    secs = np.sort(rng.integers(0, 300 * 86400, n))
    install = pd.Timestamp("2025-01-01") + pd.to_timedelta(secs, unit="s")
    platform = rng.choice(["ios", "android"], n, p=[0.45, 0.55])
    country = rng.choice(["US", "DE", "BR", "IN", "JP", "GB", "FR", "MX"], n)
    source = rng.choice(["organic", "paid_social", "video_ad", "referral"], n, p=[0.4, 0.3, 0.2, 0.1])
    levels = rng.poisson(6, n)
    minutes = np.round(rng.gamma(2.0, 12.0, n), 1)
    ads = rng.poisson(4, n)  # rewarded ads watched on day one: known at the end of day one
    tutorial = (rng.random(n) < 0.8).astype(int)
    age_days = (install - install.min()).days.to_numpy()
    after_change = age_days > 170  # a starter pack at half price after a store change: more payers
    logit = (-3.3 + 0.12 * levels + 0.025 * minutes + 0.5 * (platform == "ios") + 0.6 * np.isin(country, ["US", "JP"])
             - 0.4 * (country == "IN") + 0.4 * (source == "referral") - 0.08 * ads + 0.4 * tutorial + 1.0 * after_change)
    payer = rng.random(n) < _sigmoid(logit)
    # the player's club as shown on the profile at export: clubs above bronze come with the first purchase
    club = np.where(payer, rng.choice(["silver", "gold", "platinum", "bronze"], n, p=[0.5, 0.25, 0.1, 0.15]),
                    rng.choice(["bronze", "silver"], n, p=[0.93, 0.07]))
    players = pd.DataFrame({
        "player_id": [f"u{rng.integers(10**8, 10**9)}" for _ in range(n)],
        "install_ts": install, "platform": platform, "country": country, "acquisition_source": source,
        "day1_levels_cleared": levels, "day1_minutes_played": minutes, "day1_rewarded_ads": ads,
        "tutorial_completed": tutorial, "club_tier": club, "paid_within_14d": payer.astype(int),
    })
    return {"players.csv": players}


# --- hg3_05: a bakery chain, tomorrow's unsold loaves per shop ----------------------------------------------------------
def make_hg3_05(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 5)
    shops, days = 32, pd.date_range("2025-03-01", periods=56, freq="D")
    shop_eff = rng.normal(0.0, 4.0, shops)
    size = rng.choice(["kiosk", "standard", "flagship"], shops, p=[0.3, 0.5, 0.2])
    s_base = {"kiosk": 60, "standard": 110, "flagship": 180}
    rows = []
    for t, day in enumerate(days):
        dow = day.dayofweek
        weather = rng.choice(["sun", "cloud", "rain"], p=[0.35, 0.4, 0.25])
        for s in range(shops):
            demand = s_base[size[s]] * (1.25 if dow == 5 else 0.7 if dow == 6 else 1.0) * (0.88 if weather == "rain" else 1.0)
            planned = int(round(demand * 1.12 + rng.normal(0, 5)))
            # a sourdough line launched mid-period that sells poorly: waste rises
            unsold = max(0.0, 8 + shop_eff[s] + 0.12 * (planned - demand) + 3 * (weather == "rain") + 0.22 * max(0, t - 20)
                         + rng.normal(0, 3.0))
            unsold = int(round(unsold))
            rows.append({"sale_date": day.strftime("%Y-%m-%d"), "shop_code": f"S{s + 1:02d}", "shop_format": size[s],
                         "weekday": day.day_name(), "weather_forecast": weather, "planned_bake": planned,
                         # loaves on the shelf at the last-hour count, before the evening markdown
                         "closing_count": int(round(unsold * 1.6 + rng.normal(4, 2.0))),
                         "promo_flag": int(rng.random() < 0.1), "unsold_loaves": unsold})
    frame = pd.DataFrame(rows)
    # yesterday's waste at the same shop: known before tomorrow's bake, an honest lag
    frame["yesterday_unsold"] = frame.groupby("shop_code")["unsold_loaves"].shift(1)
    cols = ["sale_date", "shop_code", "shop_format", "weekday", "weather_forecast", "planned_bake", "yesterday_unsold",
            "promo_flag", "closing_count", "unsold_loaves"]
    return {"shop_days.csv": frame[cols]}


# --- hg3_06: house listings in a growing agency, sold within 60 days -------------------------------------------------------
def make_hg3_06(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 6)
    n, areas = 1600, 44
    area_eff = rng.normal(0.0, 0.9, areas)
    area = rng.integers(0, areas, n)
    days = np.sort(rng.integers(0, 640, n))
    listed = pd.Timestamp("2024-02-01") + pd.to_timedelta(days, unit="D")
    photos = listed + pd.to_timedelta(rng.integers(1, 9, n), unit="D")  # the photographer's visit, after the listing goes up
    ptype = rng.choice(["flat", "terrace", "semi", "detached"], n, p=[0.35, 0.3, 0.2, 0.15])
    beds = np.clip(rng.poisson(2.6, n), 1, 6)
    fair = 180000 * np.exp(0.25 * area_eff[area]) * (1 + 0.18 * (beds - 3)) * np.where(ptype == "detached", 1.5, 1.0)
    asking = np.round(fair * rng.lognormal(0.04, 0.1, n), -3)
    epc = rng.choice(list("ABCDEFG"), n, p=[0.03, 0.12, 0.3, 0.3, 0.15, 0.07, 0.03])
    chain_free = (rng.random(n) < 0.3).astype(int)
    rates_up = days > 330  # mortgage rates rose in the second year: fewer quick sales
    logit = (0.4 + 0.9 * area_eff[area] - 4.0 * np.log(asking / fair) + 0.4 * chain_free - 0.2 * (beds >= 5)
             + 0.3 * (ptype == "flat") - 1.1 * rates_up)
    quick = rng.random(n) < _sigmoid(logit)
    viewings = np.where(quick, rng.poisson(14, n), rng.poisson(5, n)) + rng.poisson(1, n)  # booked over the listing's life
    listings = pd.DataFrame({
        "listing_ref": [f"AG{70000 + i}" for i in range(n)],
        "listed_on": _iso(listed), "photos_taken_on": _iso(photos),
        "area_code": [f"AR{a:02d}" for a in area], "property_type": ptype, "bedrooms": beds,
        "asking_price": asking.astype(int), "energy_rating": epc, "chain_free": chain_free,
        "viewings_total": viewings, "sold_within_60d": quick.astype(int),
    })
    return {"listings.csv": listings}


# --- hg3_07: a dairy herd, weekly milk per cow -----------------------------------------------------------------------
def make_hg3_07(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 7)
    cows, weeks = 110, pd.date_range("2025-03-03", periods=16, freq="W-MON")
    genetic = rng.normal(0.0, 22.0, cows)
    breed = rng.choice(["holstein", "jersey", "cross"], cows, p=[0.6, 0.2, 0.2])
    b_base = {"holstein": 210.0, "jersey": 150.0, "cross": 180.0}
    parity = rng.integers(1, 7, cows)
    dim0 = rng.integers(10, 200, cows)
    price = 0.42 + np.cumsum(rng.normal(0, 0.004, len(weeks)))
    rows = []
    for w, wk in enumerate(weeks):
        grass = -3.5 * w  # the grazing gets poorer as summer dries out: yields fall week by week
        for c in range(cows):
            dim = int(dim0[c] + 7 * w)
            lact = -0.18 * max(0, dim - 60)
            feed = round(float(rng.normal(19 + 0.02 * genetic[c], 1.5)), 1)
            mastitis = int(rng.random() < 0.04)  # treated this week, recorded by the vet at the visit before milking
            kg = max(40.0, b_base[breed[c]] + genetic[c] + grass + lact + 4.0 * (feed - 19) - 3 * (parity[c] == 1)
                     - 25 * mastitis + rng.normal(0, 9))
            rows.append({"week_start": wk.strftime("%Y-%m-%d"), "ear_tag": f"IE{4400 + c * 7:05d}", "breed": breed[c],
                         "parity": int(parity[c]), "days_in_milk": dim, "concentrate_kg_day": feed,
                         "mastitis_treated": mastitis,
                         # the processor's payment for the week's milk, on the statement sent a fortnight later
                         "payout_eur": round(kg * price[w] * rng.normal(1.0, 0.02), 2),
                         "milk_kg_week": round(kg, 1)})
    return {"herd_weeks.csv": pd.DataFrame(rows)}


# --- hg3_08: a public library, will a loan come back late --------------------------------------------------------------
def make_hg3_08(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 8)
    n, members = 2000, 420
    member = _repeat_pick(rng, members, n)
    habit = rng.normal(0.0, 1.0, members)
    secs = np.sort(rng.integers(0, 540 * 86400, n))
    checkout = pd.Timestamp("2024-01-02 09:00") + pd.to_timedelta(secs, unit="s")
    item = rng.choice(["book", "dvd", "audiobook", "magazine", "game"], n, p=[0.6, 0.1, 0.12, 0.1, 0.08])
    period = {"book": 21, "dvd": 7, "audiobook": 21, "magazine": 7, "game": 14}
    due = checkout.normalize() + pd.to_timedelta([period[i] for i in item], unit="D")
    branch = rng.choice(["central", "north", "riverside", "east"], n, p=[0.4, 0.2, 0.2, 0.2])
    age_band = rng.choice(["child", "teen", "adult", "senior"], n, p=[0.2, 0.15, 0.5, 0.15])
    prev_late = pd.Series(np.zeros(n)).to_numpy()
    days = (checkout - checkout.min()).days.to_numpy()
    no_fines = days > 270  # fines abolished in the second autumn: more loans come back late
    logit = (-1.7 + 1.0 * habit[member] + 0.6 * (item == "dvd") + 0.5 * (item == "game") - 0.4 * (age_band == "senior")
             + 0.3 * (age_band == "teen") + 1.0 * no_fines)
    late = rng.random(n) < _sigmoid(logit)
    counts: dict[int, int] = {}
    for i in range(n):  # late returns by the same member earlier on: known at checkout
        prev_late[i] = counts.get(member[i], 0)
        if late[i]:
            counts[member[i]] = counts.get(member[i], 0) + 1
    # notices e-mailed about this loan: one "due soon" to everybody who opted in, then one a week once overdue
    opted = rng.random(n) < 0.7
    notices = opted.astype(int) + np.where(late, 1 + rng.poisson(1.2, n), 0)
    loans = pd.DataFrame({
        "loan_id": np.arange(500000, 500000 + n), "checkout_date": checkout, "due_date": due,
        "member_no": [f"M{m:06d}" for m in member], "branch": branch, "item_type": item, "age_band": age_band,
        "previous_late_returns": prev_late.astype(int), "renewal_allowed": (rng.random(n) < 0.85).astype(int),
        "notices_sent": notices, "returned_late": late.astype(int),
    })
    return {"loans.csv": loans}


# --- hg3_09: a haulier's trips, fuel per 100 km for vehicles joining the fleet -----------------------------------------
def make_hg3_09(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 9)
    n, units = 1900, 64
    unit_eff = rng.normal(0.0, 2.2, units)  # engine wear, tyres, driver habits that go with the vehicle
    axle = rng.choice(["4x2", "6x2", "6x4"], units, p=[0.5, 0.35, 0.15])
    euro = rng.choice(["euro5", "euro6"], units, p=[0.35, 0.65])
    unit = rng.integers(0, units, n)
    secs = np.sort(rng.integers(0, 420 * 86400, n))
    departed = pd.Timestamp("2024-05-01") + pd.to_timedelta(secs, unit="s")
    km = np.round(rng.uniform(80, 900, n), 0)
    payload = np.round(rng.uniform(0, 26, n), 1)
    motorway = np.round(rng.uniform(0.2, 0.95, n), 2)
    temp = np.round(12 + 9 * np.sin(2 * np.pi * (departed.dayofyear.to_numpy() - 110) / 365) + rng.normal(0, 3, n), 1)
    l100 = (27 + unit_eff[unit] + 0.42 * payload - 5.0 * (motorway - 0.6) + 1.8 * (axle[unit] == "6x4")
            - 0.8 * (euro[unit] == "euro6") - 0.06 * (temp - 12) + 0.000 * (km - 400) + rng.normal(0, 1.6, n))
    trips = pd.DataFrame({
        "trip_id": [f"T{900000 + i}" for i in range(n)], "departed_at": departed,
        "unit_no": [f"{u + 101}" for u in unit], "axle_config": axle[unit], "emission_class": euro[unit],
        "planned_km": km, "payload_t": payload, "motorway_share": motorway, "forecast_temp_c": temp,
        "litres_per_100km": np.round(l100, 2),
    })
    return {"trips.csv": trips}


# --- hg3_10: a charity's appeals, will a lapsed supporter give again ----------------------------------------------------
def make_hg3_10(seed: int) -> dict[str, pd.DataFrame]:
    rng = _rng(seed, 10)
    appeals = pd.to_datetime(["2022-03-07", "2022-10-03", "2023-03-06", "2023-10-02", "2024-03-04", "2024-10-07", "2025-03-03"])
    rows = []
    for a, mailed in enumerate(appeals):
        for _ in range(int(rng.integers(220, 280))):
            last_gift = float(np.round(rng.lognormal(3.2, 0.7), 0))
            months = int(rng.integers(13, 60))
            gifts = int(1 + rng.poisson(2.5))
            pack = rng.choice(["letter", "letter_and_pen", "email_only"], p=[0.5, 0.2, 0.3])
            region = rng.choice(["north", "midlands", "south", "wales", "scotland"])
            fatigue = -0.25 * a  # fewer lapsed supporters answer each year
            logit = (-1.0 + 0.25 * np.log(last_gift) - 0.03 * months + 0.18 * gifts + 0.5 * (pack == "letter_and_pen")
                     - 0.7 * (pack == "email_only") + fatigue)
            gave = rng.random() < _sigmoid(logit)
            rows.append({"appeal_mailed": mailed, "supporter_ref": f"SR{int(rng.integers(10**6, 10**7))}", "region": region,
                         "last_gift_gbp": last_gift, "months_since_last_gift": months, "lifetime_gifts": gifts,
                         "mail_pack": pack, "unsubscribe_requests": int(rng.poisson(0.15)),
                         # tax relief reclaimed on the supporter's gifts by the export: only gifts after this appeal raise it
                         "relief_claimed_gbp": round(float(gave * last_gift * 0.25 * rng.uniform(0.8, 1.6)
                                                           + (rng.random() < 0.06) * rng.uniform(1, 8)), 2),
                         "responded": int(gave)})
    frame = pd.DataFrame(rows)
    if len(frame) > 2000:
        frame = frame.iloc[np.sort(rng.choice(len(frame), 2000, replace=False))].reset_index(drop=True)
    return {"appeal_mailings.csv": frame}


MAKERS = {cid: globals()[f"make_{cid}"] for cid in CASE_IDS}

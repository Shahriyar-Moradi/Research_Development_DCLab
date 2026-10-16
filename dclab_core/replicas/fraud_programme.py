"""A fraud programme, invented: rides, accounts and a partial list of confirmed cases, with three kinds of fraud planted (package 24.6).

A courier platform wants to find the fake rides and fake orders its bikers and customers make. It has a ride table, the table of its customer accounts and a
list of the cases an investigator confirmed; a team can review a small share of the rides each day. The problem is several problems that depend on each other,
and each kind of fraud is visible to one of them only:

* **Fake delivery marks.** A few bikers mark a ride delivered without moving: the ride ends within metres of where it began, in a minute or two, for a normal
  fare. The behaviour of the ride gives it away; no other table is needed. A screen with no labels finds it. Cancelled rides also end where they began, for no fare:
  a screen that does not leave them out drowns in them, and short deliveries to a neighbour are honest rides that look a little like marks.
* **Fake orders for the fare, in rings.** Groups of accounts open together, share a phone or a device, and order almost only from one colluding biker. Each ride
  is normal: its distance, its duration and its fare say nothing. Only the table of accounts shows the links (accounts sharing a phone, a biker that serves those
  accounts and almost nobody else). Families and offices honestly share a phone, so one shared phone is not a ring.
* **Both.** Some ring rides are also fake marks.
* **The labels are partial and biased.** The investigators confirmed more of the visible kind (a mark) than of the hidden one (a ring), and none of the rest: an
  unmarked ride is unknown, not clean.
* **Rows cluster on entities and move in time.** The holdout is the latest period; the bikers and the accounts repeat across it.
* **The review budget.** A team can check about 2% of the rides of the period; a score is judged at that budget.

The scale is chosen for power, not for a result: with few fraudulent bikers a bootstrap over bikers is too wide to tell a real gain from chance, so the
replica has 14 rings and 28 marker bikers, each with few rides (calibrated on the seeds 10 to 15; the acceptance runs the seeds 0 to 2, which the calibration never saw).

``truth`` says which rides were planted and is for tests only. Invented data: a score on it says whether the decisions are sound, never how a model would do on real
rides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dclab_core.replicas import LABELS

CENTER = (12.480, 45.020)
M_PER_DEG = 111_195.0
MARK, RING, BOTH = "fake_mark", "ring_order", "mark_in_ring"
RIDE_FILE, ACCOUNT_FILE, CASE_FILE, INFO_FILE = "rides.csv", "accounts.csv", "confirmed_cases.csv", "replica.json"
BUDGET_SHARE = 0.02


@dataclass(frozen=True)
class Replica:
    """What a person would upload, plus what only the tests may see."""

    rides: pd.DataFrame      # one row per ride
    accounts: pd.DataFrame   # one row per customer account: its phone and device
    cases: pd.DataFrame      # the rides an investigator confirmed as fraud (partial)
    truth: pd.DataFrame      # every planted fraud ride (order_id, kind, biker_id, account_id, confirmed): tests only
    facts: dict[str, Any] = field(default_factory=dict)


def _ids(rng: np.random.Generator, n: int, prefix: str) -> np.ndarray:
    digits = np.array(list("0123456789abcdef"))[rng.integers(0, 16, size=(n, 12))]
    return np.array([prefix + "".join(row) for row in digits])


def generate(seed: int = 0, n_orders: int = 80_000, n_accounts: int = 4_800, n_bikers: int = 300, days: int = 60, n_rings: int = 14, n_markers: int = 28,
             confirm: tuple[float, float, float] = (0.45, 0.30, 0.55), ring_rides: tuple[int, int] = (22, 38),
             mark_rate: tuple[float, float] = (0.015, 0.035)) -> Replica:
    """Draw a replica. Deterministic per ``seed``. ``confirm``: the share of planted marks, ring orders and mark-in-ring rides that were confirmed;
    ``ring_rides``: the range of rides a ring orders in its last 45 days; ``mark_rate``: the range of a marker biker's share of rides that are fake marks."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-05")

    # ---- accounts: phones and devices, honest sharing, and the rings
    phone = _ids(rng, n_accounts, "ph_")
    device = _ids(rng, n_accounts, "dv_")
    honest_pairs = rng.choice(n_accounts, size=int(0.06 * n_accounts) * 2, replace=False).reshape(-1, 2)  # families and offices share a phone
    phone[honest_pairs[:, 1]] = phone[honest_pairs[:, 0]]
    ring_members: list[np.ndarray] = []
    pool = rng.permutation(np.setdiff1d(np.arange(n_accounts), honest_pairs.ravel()))
    cursor = 0
    for _ in range(n_rings):
        k = int(rng.integers(3, 7))
        members = pool[cursor:cursor + k]
        cursor += k
        ring_members.append(members)
        shared = _ids(rng, 2, "ph_")
        phone[members] = rng.choice(shared, k)
        device[members] = np.where(rng.random(k) < 0.7, device[members[0]], device[members])
    signup = np.datetime64("2025-09-01") + rng.integers(0, 120, n_accounts).astype("timedelta64[D]")
    for members in ring_members:  # a ring opens its accounts together, recently
        opened = np.datetime64(start.strftime("%Y-%m-%d")) + np.timedelta64(int(rng.integers(0, days - 25)), "D")
        signup[members] = opened - rng.integers(1, 9, len(members)).astype("timedelta64[D]")
    account_ids = np.arange(7_000_000, 7_000_000 + n_accounts)
    accounts = pd.DataFrame({"account_id": account_ids, "signup_date": pd.DatetimeIndex(signup).strftime("%Y-%m-%d"), "phone": phone, "device_id": device,
                             "city": rng.choice(["north", "centre", "south"], n_accounts, p=[0.3, 0.5, 0.2])})

    # ---- bikers
    biker_ids = rng.choice(np.arange(100_000, 999_999), n_bikers, replace=False)
    activity = rng.gamma(2.0, 1.0, n_bikers)
    order_pool = rng.permutation(n_bikers)
    markers = order_pool[:n_markers]
    colluders = order_pool[n_markers:n_markers + n_rings]  # one colluding biker for each ring
    marker_rate = np.zeros(n_bikers)
    marker_rate[markers] = rng.uniform(mark_rate[0], mark_rate[1], n_markers)

    # ---- honest rides
    account_weight = rng.gamma(1.5, 1.0, n_accounts)
    account_weight[np.concatenate(ring_members)] *= 0.15  # ring accounts have little honest use
    acc_idx = rng.choice(n_accounts, n_orders, p=account_weight / account_weight.sum())
    biker_idx = rng.choice(n_bikers, n_orders, p=activity / activity.sum())
    distance = np.exp(rng.normal(np.log(2400), 0.6, n_orders))
    speed_mps = np.exp(rng.normal(np.log(4.6), 0.25, n_orders))
    duration = distance / speed_mps / 60 + rng.gamma(4.0, 1.0, n_orders)  # minutes, with the time to find the door
    short = rng.random(n_orders) < 0.04  # an honest delivery next door
    distance = np.where(short, rng.uniform(40, 200, n_orders), distance)
    duration = np.where(short, rng.uniform(3.5, 9, n_orders), duration)
    fare = 1.2 + 0.00085 * distance + rng.normal(0, 0.25, n_orders)
    day = rng.integers(0, days, n_orders)
    minute = np.clip(rng.normal(13 * 60, 4.5 * 60, n_orders), 6 * 60, 23 * 60).astype(int)
    kind = np.full(n_orders, "", dtype=object)

    # ---- fake delivery marks: a marker biker's ride that ends where it began, in a minute or two, for a normal fare
    is_marker_ride = rng.random(n_orders) < marker_rate[biker_idx]
    distance = np.where(is_marker_ride, rng.uniform(0, 45, n_orders), distance)
    duration = np.where(is_marker_ride, rng.uniform(0.4, 2.6, n_orders), duration)
    fare = np.where(is_marker_ride, 1.2 + 0.00085 * np.exp(rng.normal(np.log(2400), 0.6, n_orders)) + rng.normal(0, 0.25, n_orders), fare)
    kind[is_marker_ride] = MARK

    # ---- ring orders: the ring's accounts order from its colluding biker through the last 45 days; each ride is normal
    ring_rows = []
    for ring, members in enumerate(ring_members):
        n_ring = int(rng.integers(ring_rides[0], ring_rides[1]))
        a = rng.choice(members, n_ring)
        d = rng.integers(days - 45, days, n_ring)
        ring_rows.append(pd.DataFrame({"acc_idx": a, "biker_idx": colluders[ring], "day": d}))
    ring = pd.concat(ring_rows, ignore_index=True)
    n_ring_rides = len(ring)
    r_distance = np.exp(rng.normal(np.log(2400), 0.6, n_ring_rides))
    r_duration = r_distance / np.exp(rng.normal(np.log(4.6), 0.25, n_ring_rides)) / 60 + rng.gamma(4.0, 1.0, n_ring_rides)
    r_fare = 1.2 + 0.00085 * r_distance + rng.normal(0, 0.25, n_ring_rides)
    r_kind = np.full(n_ring_rides, RING, dtype=object)
    also_mark = rng.random(n_ring_rides) < 0.12
    r_distance = np.where(also_mark, rng.uniform(0, 45, n_ring_rides), r_distance)
    r_duration = np.where(also_mark, rng.uniform(0.4, 2.6, n_ring_rides), r_duration)
    r_kind[also_mark] = BOTH
    r_minute = np.clip(rng.normal(13 * 60, 4.5 * 60, n_ring_rides), 6 * 60, 23 * 60).astype(int)

    rides = pd.DataFrame({"acc_idx": np.concatenate([acc_idx, ring["acc_idx"].to_numpy()]), "biker_idx": np.concatenate([biker_idx, ring["biker_idx"].to_numpy()]),
                          "day": np.concatenate([day, ring["day"].to_numpy()]), "minute": np.concatenate([minute, r_minute]),
                          "distance_m": np.concatenate([distance, r_distance]), "duration_min": np.concatenate([duration, r_duration]),
                          "fare": np.concatenate([fare, r_fare]), "kind": np.concatenate([kind, r_kind])})
    rides = rides.iloc[rng.permutation(len(rides))].reset_index(drop=True)

    # ---- cancelled rides end where they began, for no fare: a screen must leave them out
    cancelled = (rides["kind"] == "") & (rng.random(len(rides)) < 0.03)
    rides.loc[cancelled, ["distance_m", "duration_min", "fare"]] = [0.0, 0.0, 0.0]
    rides["status"] = np.where(cancelled, "CANCELLED", "DELIVERED")

    # ---- the table a person gets: sorted by time, the coordinates drawn from each ride's own distance
    created = start + pd.to_timedelta(rides["day"], unit="D") + pd.to_timedelta(rides["minute"], unit="m")
    order = rides.assign(created_at=created).sort_values("created_at", kind="stable").reset_index(drop=True)
    n = len(order)
    angle = rng.uniform(0, 2 * np.pi, n)
    lat0 = CENTER[0] + rng.normal(0, 0.02, n)
    lon0 = CENTER[1] + rng.normal(0, 0.027, n)
    lat1 = lat0 + np.cos(angle) * order["distance_m"].to_numpy() / M_PER_DEG
    lon1 = lon0 + np.sin(angle) * order["distance_m"].to_numpy() / (M_PER_DEG * np.cos(np.radians(CENTER[1])))
    order["order_id"] = 5_000_000 + np.arange(n)
    out = pd.DataFrame({"order_id": order["order_id"], "created_at": order["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "account_id": account_ids[order["acc_idx"].to_numpy()], "biker_id": biker_ids[order["biker_idx"].to_numpy()],
                        "fare": order["fare"].round(2), "distance_m": order["distance_m"].round(1), "duration_min": order["duration_min"].round(2),
                        "pickup_lat": np.round(lat0, 6), "pickup_lon": np.round(lon0, 6), "drop_lat": np.round(lat1, 6), "drop_lon": np.round(lon1, 6),
                        "category": rng.choice(["standard", "cargo", "no-box"], n, p=[0.8, 0.07, 0.13]), "status": order["status"]})

    # ---- the partial, biased list of what an investigator confirmed
    planted = order["kind"].to_numpy() != ""
    chance = np.select([order["kind"] == MARK, order["kind"] == RING, order["kind"] == BOTH], list(confirm), 0.0)
    confirmed = planted & (rng.random(len(order)) < chance)
    truth = pd.DataFrame({"order_id": order.loc[planted, "order_id"], "kind": order.loc[planted, "kind"], "biker_id": out.loc[planted, "biker_id"],
                          "account_id": out.loc[planted, "account_id"], "confirmed": confirmed[planted]})
    reasons = {MARK: "the ride ended where it began", RING: "linked accounts and one biker", BOTH: "linked accounts and a ride that did not move"}
    cases = pd.DataFrame({"order_id": order.loc[confirmed, "order_id"], "reason": order.loc[confirmed, "kind"].map(reasons),
                          "confirmed_at": (order.loc[confirmed, "created_at"] + pd.to_timedelta(rng.integers(3, 14, int(confirmed.sum())), unit="D")).dt.strftime("%Y-%m-%d")})
    kinds = truth["kind"].value_counts().to_dict()
    facts = {"rides": len(out), "accounts": n_accounts, "bikers": n_bikers, "days": days, "planted": int(planted.sum()), "kinds": kinds, "confirmed": int(confirmed.sum()),
             "marker_bikers": [int(biker_ids[i]) for i in markers], "colluding_bikers": [int(biker_ids[i]) for i in colluders],
             "ring_accounts": [[int(account_ids[i]) for i in members] for members in ring_members], "budget_share": BUDGET_SHARE, "cancelled": int(cancelled.sum()),
             "honest_shared_phones": int(len(honest_pairs))}
    return Replica(rides=out, accounts=accounts, cases=cases.reset_index(drop=True), truth=truth.reset_index(drop=True), facts=facts)


def save(replica: Replica, directory: Path) -> dict[str, Path]:
    """Write the three files a person would upload, and a note that they are synthetic (the truth is never written)."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"rides": directory / RIDE_FILE, "accounts": directory / ACCOUNT_FILE, "cases": directory / CASE_FILE, "info": directory / INFO_FILE}
    replica.rides.to_csv(paths["rides"], index=False)
    replica.accounts.to_csv(paths["accounts"], index=False)
    replica.cases.to_csv(paths["cases"], index=False)
    paths["info"].write_text(json.dumps({**LABELS, "replica": "fraud_programme", "files": [RIDE_FILE, ACCOUNT_FILE, CASE_FILE],
                                         "question": "Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, "
                                                     "nothing else is labelled, and the team can review about 2% of the rides each day."}, indent=2))
    return paths

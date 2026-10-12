"""A delivery-fraud challenge, invented: a ride table and a separate sample of known frauds (package 15.1).

Shaped like a real hiring challenge the owner gave DCLab (its data is confidential and never used here). The person
gets two files and the question "find the bikers' and customers' fraud". The traps a careful data scientist must see:

* **The labels are in another file, and the right key is the order.** Joining the sample on (account, biker) marks every
  ride of a fraudulent biker for that account: there are only three accounts, so the coarse join inflates the labels
  many times over.
* **The labels are partial.** The sample holds only some of the planted frauds; the rest of the table is unlabelled,
  not clean. A few sample rows belong to an account that is not in the table at all.
* **One column name means two things.** ``sender_phone`` is the sender's phone in the table, but in the sample it holds
  the *receiver's* phone (it equals the biker's phone for self-dealing rides). The sample's fares also disagree with
  the table's.
* **Rows cluster on entities.** The frauds come from a few bikers; a random split lets a model recognise the biker.
* **The signal is behaviour.** A fake ride ends a few metres from its pick-up, at a drop-off the same biker returns to.
  The raw columns (a fare, a coordinate) carry little of it.
* **One kind of fraud is invisible in the table.** Self-dealing (the biker delivers to his own phone) shows only through
  the receiver's phone, which the table does not have.

Cancelled rides have no biker. Some orders have two or three drop-offs (terminals), each a row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..draft.synthetic import LABELS

CENTER = (12.480, 45.020)  # an invented city
M_PER_DEG = 111_195.0
CATEGORIES = ("standard", "cargo", "no-box")
FAKE, SELF_DEALING = "fake_trip", "self_delivery"
RIDE_FILE, SAMPLE_FILE, INFO_FILE = "rides.csv", "known_frauds.csv", "replica.json"


@dataclass(frozen=True)
class Replica:
    """The two files a person would upload, plus what only the tests may see."""

    rides: pd.DataFrame          # the ride table: one row per drop-off (terminal)
    sample: pd.DataFrame         # the known-fraud sample, as handed over with the challenge
    truth: pd.DataFrame          # every planted fraud row (order_id, terminal_ref, biker_id, kind): tests only
    facts: dict[str, Any] = field(default_factory=dict)  # the numbers the traps produce


def _phones(rng: np.random.Generator, n: int) -> np.ndarray:
    """Pseudonymised phones: "ph_" and 16 hexadecimal digits."""
    digits = np.array(list("0123456789abcdef"))[rng.integers(0, 16, size=(n, 16))]
    return np.array(["ph_" + "".join(row) for row in digits])


def _refs(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = rng.integers(0, 16, size=(n, 32))
    hexes = np.array(list("0123456789abcdef"))[raw]
    return np.array(["{}-{}-{}-{}-{}".format(*("".join(r[a:b]) for a, b in ((0, 8), (8, 12), (12, 16), (16, 20), (20, 32))))
                     for r in hexes])


def _offset(lat: np.ndarray, lon: np.ndarray, metres: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dlat = metres * np.sin(angle) / M_PER_DEG
    dlon = metres * np.cos(angle) / (M_PER_DEG * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def _fare(category: np.ndarray, metres: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.select([category == "cargo", category == "no-box"], [260_000, 90_000], 120_000)
    fare = (base + 420 * metres) * rng.lognormal(0, 0.25, len(metres))
    return np.round(fare / 250) * 250


def generate(seed: int = 0, n_orders: int = 16_000, n_bikers: int = 240, days: int = 9, label_share: float = 0.6,
             n_fake_bikers: int = 30, n_self_bikers: int = 12, n_external: int = 5) -> Replica:
    """Draw a replica. Deterministic per ``seed``; ``label_share`` is the share of planted fraud orders in the sample."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-03-03")

    biker_ids = rng.choice(np.arange(100_000, 999_999), n_bikers, replace=False)
    biker_phone = dict(zip(biker_ids, _phones(rng, n_bikers)))
    activity = rng.gamma(2.0, 1.0, n_bikers)

    # three accounts: a marketplace with many senders, a retailer with one, a small courier client
    accounts = {3: 1, 904_417: 40, 51_206: 4}
    sender_phones = {acc: _phones(rng, k) for acc, k in accounts.items()}
    account_share = {3: 0.55, 904_417: 0.40, 51_206: 0.05}
    vendors = []
    for acc, count in ((3, 70), (904_417, 50), (51_206, 8)):
        lat = CENTER[0] + rng.normal(0, 0.012, count)
        lon = CENTER[1] + rng.normal(0, 0.016, count)
        for i in range(count):
            vendors.append((acc, lat[i], lon[i], rng.choice(sender_phones[acc])))
    vendors = pd.DataFrame(vendors, columns=["account_id", "vlat", "vlon", "sender_phone"])
    vendors["weight"] = rng.pareto(1.5, len(vendors)) + 1

    def account_category(acc: np.ndarray) -> np.ndarray:
        u = rng.random(len(acc))
        return np.where(acc == 904_417, np.where(u < 0.92, "no-box", "standard"),
                        np.where(u < 0.97, "standard", "cargo"))

    def pick_vendor(acc: np.ndarray) -> np.ndarray:
        out = np.empty(len(acc), dtype=int)
        for a in np.unique(acc):
            pool = vendors.index[vendors["account_id"] == a].to_numpy()
            w = vendors.loc[pool, "weight"].to_numpy()
            idx = np.flatnonzero(acc == a)
            out[idx] = rng.choice(pool, len(idx), p=w / w.sum())
        return out

    # ---- honest orders
    acc = rng.choice(list(account_share), n_orders, p=list(account_share.values()))
    orders = pd.DataFrame({
        "account_id": acc,
        "vendor": pick_vendor(acc),
        "biker_id": rng.choice(biker_ids, n_orders, p=activity / activity.sum()),
        "day": rng.integers(0, days, n_orders),
        "category": account_category(acc),
        "kind": "",
    })
    orders["terminals"] = rng.choice([1, 2, 3], n_orders, p=[0.975, 0.022, 0.003])

    # ---- planted fraud: fake rides at a repeated spot, and self-dealing rides (invisible in the table)
    busy = biker_ids[np.argsort(activity)][n_bikers // 2:]  # fraud hides among the busier half of the bikers
    fraudsters = rng.choice(busy, n_fake_bikers + n_self_bikers, replace=False)
    fake_bikers, self_bikers = fraudsters[:n_fake_bikers], fraudsters[n_fake_bikers:]
    spot_rows, fake_rows = [], []
    market = vendors.index[vendors["account_id"] == 904_417].to_numpy()
    for b in fake_bikers:
        for _ in range(rng.integers(1, 3)):  # one or two spots per biker
            v = rng.choice(market)
            angle = rng.uniform(0, 2 * np.pi)
            slat, slon = _offset(np.array([vendors.at[v, "vlat"]]), np.array([vendors.at[v, "vlon"]]),
                                 np.array([rng.uniform(4, 25)]), np.array([angle]))
            spot_rows.append((b, v, slat[0], slon[0]))
    spots = pd.DataFrame(spot_rows, columns=["biker_id", "vendor", "slat", "slon"])
    for b in fake_bikers:
        mine = spots[spots["biker_id"] == b]
        for _ in range(rng.poisson(2.5) + 2):
            s = mine.iloc[rng.integers(len(mine))]
            fake_rows.append({"account_id": 904_417, "vendor": int(s["vendor"]), "biker_id": b, "day": rng.integers(0, days),
                              "category": "no-box", "kind": FAKE, "terminals": 1, "slat": s["slat"], "slon": s["slon"]})
    self_rows = []
    for b in self_bikers:
        for _ in range(rng.poisson(1) + 1):
            a = rng.choice([3, 904_417], p=[0.5, 0.5])
            self_rows.append({"account_id": a, "vendor": int(pick_vendor(np.array([a]))[0]), "biker_id": b,
                              "day": rng.integers(0, days), "category": account_category(np.array([a]))[0],
                              "kind": SELF_DEALING, "terminals": 1})
    orders = pd.concat([orders, pd.DataFrame(fake_rows), pd.DataFrame(self_rows)], ignore_index=True)
    orders = orders.sample(frac=1.0, random_state=seed).sort_values("day", kind="stable").reset_index(drop=True)
    orders["order_id"] = np.sort(rng.choice(np.arange(470_000_000, 480_000_000), len(orders), replace=False))  # rise with time

    # ---- one row per terminal
    rows = orders.loc[orders.index.repeat(orders["terminals"])].reset_index(drop=True)
    n = len(rows)
    v = rows["vendor"].to_numpy()
    rows["source_lat"] = vendors.loc[v, "vlat"].to_numpy() + rng.normal(0, 2e-6, n)
    rows["source_lon"] = vendors.loc[v, "vlon"].to_numpy() + rng.normal(0, 2e-6, n)
    metres = np.clip(rng.lognormal(np.log(330), 0.55, n), 25, 1600)
    lat, lon = _offset(rows["source_lat"].to_numpy(), rows["source_lon"].to_numpy(), metres, rng.uniform(0, 2 * np.pi, n))
    fake = (rows["kind"] == FAKE).to_numpy()
    jitter_lat, jitter_lon = _offset(rows["slat"].to_numpy(), rows["slon"].to_numpy(), rng.uniform(0, 3, n), rng.uniform(0, 2 * np.pi, n))
    rows["destination_lat"] = np.where(fake, jitter_lat, lat)
    rows["destination_lon"] = np.where(fake, jitter_lon, lon)
    paid_metres = np.where(fake, rng.lognormal(np.log(330), 0.5, n), metres)  # the platform pays a normal-looking fare
    order_fare = _fare(rows["category"].to_numpy(), paid_metres, rng)
    rows["biker_fare"] = np.round(order_fare / rows["terminals"].to_numpy(), 2)
    rows["sender_phone"] = vendors.loc[v, "sender_phone"].to_numpy()
    rows["receiver_phone"] = np.where(rows["kind"] == SELF_DEALING, rows["biker_id"].map(biker_phone), _phones(rng, n))
    rows["created_date"] = (start + pd.to_timedelta(rows["day"], unit="D")).dt.strftime("%Y-%m-%d")
    rows["terminal_ref"] = _refs(rng, n)
    rows["status"] = "DELIVERED"
    cancelled = (rows["kind"] == "").to_numpy() & (rng.random(n) < 0.01)
    rows["biker_phone"] = rows["biker_id"].map(biker_phone)
    rows.loc[cancelled, "status"] = "CANCELLED"
    rows["biker_id"] = rows["biker_id"].astype(float)
    rows.loc[cancelled, ["biker_id", "biker_phone"]] = np.nan

    table_cols = ["created_date", "account_id", "biker_id", "biker_phone", "order_id", "category", "terminal_ref", "status",
                  "biker_fare", "sender_phone", "destination_lon", "destination_lat", "source_lat", "source_lon"]
    rides = rows[table_cols].reset_index(drop=True)
    truth = rows.loc[rows["kind"] != "", ["order_id", "terminal_ref", "biker_id", "account_id", "kind"]].reset_index(drop=True)

    # ---- the sample handed over: some planted orders, the receiver's phone under the sender's name, other fares
    fraud_orders = truth.drop_duplicates("order_id")
    known = fraud_orders.sample(frac=label_share, random_state=seed)
    known_rows = rows[rows["order_id"].isin(known["order_id"])].drop_duplicates("order_id")
    sample = pd.DataFrame({
        "created_date": known_rows["created_date"].to_numpy(),
        "account_id": known_rows["account_id"].to_numpy(),
        "reason": known_rows["kind"].to_numpy(),
        "biker_id": known_rows["biker_id"].astype(int).to_numpy(),
        "biker_phone": known_rows["biker_phone"].to_numpy(),
        "order_id": known_rows["order_id"].to_numpy(),
        "category": known_rows["category"].to_numpy(),
        "terminal_ref": known_rows["terminal_ref"].to_numpy(),
        "status": "DELIVERED",
        "biker_fare": np.round(known_rows["biker_fare"].to_numpy() * rng.choice([1.0, 0.5, 1.5, 2.25, 0.4], len(known_rows),
                                                                                   p=[0.1, 0.25, 0.25, 0.2, 0.2]), 2),
        "sender_phone": known_rows["receiver_phone"].to_numpy(),  # the receiver's phone, under the table's sender name
        "destination_lon": known_rows["destination_lon"].to_numpy(),
        "destination_lat": known_rows["destination_lat"].to_numpy(),
        "source_lat": known_rows["source_lat"].to_numpy(),
        "source_lon": known_rows["source_lon"].to_numpy(),
    })
    ext_lat, ext_lon = CENTER[0] + rng.normal(0, 0.01, n_external), CENTER[1] + rng.normal(0, 0.01, n_external)
    elat, elon = _offset(ext_lat, ext_lon, rng.uniform(5, 30, n_external), rng.uniform(0, 2 * np.pi, n_external))
    external = pd.DataFrame({
        "created_date": (start + pd.to_timedelta(rng.integers(0, days, n_external), unit="D")).strftime("%Y-%m-%d"),
        "account_id": 66_190, "reason": FAKE, "biker_id": rng.choice(biker_ids, n_external),
        "biker_phone": _phones(rng, n_external), "order_id": rng.integers(481_000_000, 482_000_000, n_external),
        "category": "no-box", "terminal_ref": _refs(rng, n_external), "status": "DELIVERED",
        "biker_fare": _fare(np.array(["no-box"] * n_external), rng.uniform(200, 500, n_external), rng),
        "sender_phone": _phones(rng, n_external), "destination_lon": elon, "destination_lat": elat,
        "source_lat": ext_lat, "source_lon": ext_lon,
    })
    sample = pd.concat([sample, external], ignore_index=True).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    pairs = set(zip(sample["account_id"], sample["biker_id"].astype(float)))
    pair_marked = int(sum((a, b) in pairs for a, b in zip(rides["account_id"], rides["biker_id"])))
    known_table_rows = int(rides["order_id"].isin(set(sample["order_id"])).sum())
    facts = {
        "rides": int(len(rides)),
        "scorable_rides": int(((rides["status"] == "DELIVERED") & rides["biker_id"].notna()).sum()),
        "accounts": int(rides["account_id"].nunique()),
        "planted_fraud_rows": int(len(truth)),
        "planted_by_kind": truth["kind"].value_counts().to_dict(),
        "known_fraud_rows": known_table_rows,
        "pair_join_rows": pair_marked,
        "sample_rows": int(len(sample)),
        "sample_rows_not_in_table": int((~sample["order_id"].isin(set(rides["order_id"]))).sum()),
        "fraud_bikers": int(truth["biker_id"].nunique()),
        "label_share": label_share,
        "seed": seed,
    }
    return Replica(rides=rides, sample=sample, truth=truth, facts=facts)


def save(replica: Replica, directory: Path) -> dict[str, Path]:
    """Write the two files a person would upload, and a note that they are synthetic (the truth is never written)."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"rides": directory / RIDE_FILE, "sample": directory / SAMPLE_FILE, "info": directory / INFO_FILE}
    replica.rides.to_csv(paths["rides"])  # with the unnamed index column, as such files often arrive
    replica.sample.to_csv(paths["sample"])
    paths["info"].write_text(json.dumps({**LABELS, "replica": "delivery_fraud", "files": [RIDE_FILE, SAMPLE_FILE],
                                         "question": "Find fraudulent activity of bikers and customers in this city's rides; "
                                                     "known_frauds.csv is a sample of forged rides to test against."}, indent=2))
    return paths

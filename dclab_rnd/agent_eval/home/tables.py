"""The tables the Home requests run against: seeded, invented, shaped like a sales log, a customer table and a travel
booking log. Never a real file; the same seed gives the same table on every machine.

``sales`` has a few exact duplicate rows and some missing amounts (a preparation request should find them); ``customers``
has some missing monthly spend; ``bookings`` has a few test bookings and string prices with a currency (cleaning work).
No table has ``lifetime_spend`` or ``booking_margin``: the missing-column traps name them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261027
ROWS = {"sales": 2000, "customers": 1200, "bookings": 1500}
IDENTIFIERS = {"sales": ("order_id", "customer_id"), "customers": ("customer_id",), "bookings": ("booking_id", "customer_id")}


def _customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    plan = rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.3, 0.2])
    tickets = rng.poisson(1.2, n)
    spend = np.round(np.where(plan == "premium", 89, np.where(plan == "plus", 49, 19)) + rng.normal(0, 6, n), 2)
    churn = rng.random(n) < (0.08 + 0.1 * (plan == "basic") + 0.04 * np.minimum(tickets, 4))
    frame = pd.DataFrame({
        "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
        "signup_date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D"),
        "segment": rng.choice(["consumer", "business"], n, p=[0.8, 0.2]),
        "city": rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah"], n, p=[0.45, 0.25, 0.15, 0.1, 0.05]),
        "plan": plan,
        "tenure_months": rng.integers(1, 60, n),
        "monthly_spend": spend,
        "support_tickets_90d": tickets,
        "churned_next_month": churn.astype(int),
    })
    frame.loc[rng.choice(n, n // 25, replace=False), "monthly_spend"] = np.nan
    return frame


def _sales(rng: np.random.Generator, n: int) -> pd.DataFrame:
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 455, n), unit="D")  # 2025 and the first quarter of 2026
    frame = pd.DataFrame({
        "order_id": [f"ORD-{10000 + i}" for i in range(1, n + 1)],
        "order_date": dates.strftime("%Y-%m-%d"),
        "region": rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah"], n, p=[0.4, 0.25, 0.18, 0.1, 0.07]),
        "channel": rng.choice(["web", "app", "store"], n, p=[0.45, 0.35, 0.2]),
        "product_category": rng.choice(["electronics", "home", "fashion", "beauty"], n),
        "amount": np.round(rng.gamma(2.0, 120.0, n), 2),
        "status": rng.choice(["paid", "refunded", "cancelled"], n, p=[0.86, 0.09, 0.05]),
        "customer_id": [f"C-{i:05d}" for i in rng.integers(1, ROWS["customers"] + 1, n)],
    })
    frame.loc[rng.choice(n, n // 40, replace=False), "amount"] = np.nan
    duplicates = frame.iloc[rng.choice(n, n // 100, replace=False)]
    return pd.concat([frame, duplicates], ignore_index=True)


def _bookings(rng: np.random.Generator, n: int) -> pd.DataFrame:
    booked = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 420, n), unit="D")
    lead = rng.integers(1, 180, n)
    price = np.round(rng.gamma(3.0, 400.0, n), 0)
    cancelled = rng.random(n) < (0.12 + 0.0012 * lead)
    frame = pd.DataFrame({
        "booking_id": [f"BK-{50000 + i}" for i in range(1, n + 1)],
        "booking_date": booked.strftime("%Y-%m-%d"),
        "travel_date": (booked + pd.to_timedelta(lead, unit="D")).strftime("%Y-%m-%d"),
        "destination": rng.choice(["Dubai", "London", "Paris", "Istanbul", "Bangkok"], n),
        "nationality": rng.choice(["AE", "IN", "GB", "IR", "SA", "PK"], n),
        "channel": rng.choice(["web", "app", "agent"], n, p=[0.5, 0.3, 0.2]),
        "lead_time_days": lead,
        "price": [f"{p:.0f} AED" for p in price],
        "free_cancellation": rng.random(n) < 0.4,
        "cancelled": cancelled.astype(int),
        "customer_id": [f"C-{i:05d}" for i in rng.integers(1, ROWS["customers"] + 1, n)],
    })
    frame.loc[rng.choice(n, 12, replace=False), "destination"] = "TEST"
    return frame


BUILDERS = {"sales": _sales, "customers": _customers, "bookings": _bookings}
TABLES = tuple(BUILDERS)


def build(name: str) -> pd.DataFrame:
    """The table ``name``, the same on every call (each table has its own seed, so adding a table changes no other)."""
    rng = np.random.default_rng(SEED + TABLES.index(name))
    return BUILDERS[name](rng, ROWS[name])


def write(name: str, folder: Path) -> str:
    """Write ``name`` as a CSV into ``folder``; returns the file name."""
    filename = f"{name}.csv"
    build(name).to_csv(Path(folder) / filename, index=False)
    return filename


def id_values(name: str) -> set[str]:
    """The identifier values of a table: what a reply must never list (an answer shows aggregates, not records)."""
    frame = build(name)
    return {str(v) for col in IDENTIFIERS[name] for v in frame[col].dropna().unique()}

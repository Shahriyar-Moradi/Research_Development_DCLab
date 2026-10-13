"""Machines with sensors, invented: alert before a failure (a stream replica).

Hourly readings of temperature, vibration and pressure for each machine. Failures are rare. In the two days before a
failure, vibration creeps up and temperature drifts: the warning signs. Machines differ in their normal level, so a
reading is unusual only against the machine's own past. After a failure the machine is down for a day.

The traps a careful data scientist must see:

* **The future in a window.** A window that reads one hour past its end sees the failure itself.
* **Rows are not incidents.** Thousands of quiet windows and a few dozen failures: accuracy per row says nothing;
  count the failures caught, how early, and the alerts per machine-day it costs.
* **Machines differ.** A threshold on raw vibration flags the loud machines all the time; a change against each
  machine's own level finds the warning signs.

The failure times are in ``incidents``; ``facts`` keeps the warning length (for judging).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Replica:
    readings: pd.DataFrame
    incidents: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)


def generate(seed: int = 0, machines: int = 24, days: int = 90, warning_hours: int = 48) -> Replica:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    hours = days * 24
    rows, incidents = [], []
    for m in range(machines):
        level = {"temperature": rng.normal(60, 6), "vibration": rng.gamma(4, 0.5), "pressure": rng.normal(100, 8)}
        fails: list[int] = []
        for f in np.sort(rng.choice(np.arange(200, hours - 10), size=rng.poisson(2.2), replace=False)):
            if not fails or f - fails[-1] >= warning_hours + 24:  # a machine cannot fail while it is down, nor warn through a repair
                fails.append(int(f))
        down = np.zeros(hours, dtype=bool)
        warn = np.zeros(hours)
        for f in fails:
            incidents.append((f"M{m:02d}", start + pd.Timedelta(hours=int(f))))
            lo = max(0, f - warning_hours)
            warn[lo:f] = np.linspace(0, 1, f - lo)  # the signs grow towards the failure
            down[f:f + 24] = True
        for h in range(hours):
            if down[h]:
                continue
            rows.append((f"M{m:02d}", start + pd.Timedelta(hours=h),
                         level["temperature"] + 4 * warn[h] + rng.normal(0, 1.2),
                         level["vibration"] * (1 + 0.8 * warn[h]) + rng.normal(0, 0.15),
                         level["pressure"] + rng.normal(0, 2.0)))
    readings = pd.DataFrame(rows, columns=["machine", "time", "temperature", "vibration", "pressure"])
    readings["time"] = readings["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    inc = pd.DataFrame(incidents, columns=["machine", "failed_at"])
    inc["failed_at"] = inc["failed_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return Replica(readings, inc, {"unit": "machine", "time": "time", "incident_time": "failed_at", "sensors": ["temperature", "vibration", "pressure"],
                                   "warning_hours": warning_hours, "synthetic": True})

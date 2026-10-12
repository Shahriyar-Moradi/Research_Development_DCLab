"""Entity behaviour features, fitted inside folds (package 15.4).

Fraud, abuse and churn live in behaviour, not in one row: a biker who keeps ending trips metres from the pick-up, a
customer who returns most orders. :class:`EntityBehaviour` turns a table with an entity column (a biker, a customer, a
card) and optional coordinates and timestamps into such features, the way a careful data scientist would:

* **counts**: rows per entity, per entity and day, per entity and drop-off place, per place, per route;
* **trip shape**: the distance between two coordinate pairs, and each money column per metre;
* **the row against its entity**: the entity's mean of each quantity, and how far the row is from it;
* **shrunk rates**: the target rate of the entity and of the place, shrunk towards the training mean, learned from
  the fit rows only; a fit row's own rate leaves its own label out (leave-one-out), an entity never seen gets the mean,
  and an entity the folds are grouped by gets no rate at all (its test rows are never seen, so the rate would teach the
  model a signal that is always missing when it scores);
* counts of places and routes are shares of the rows the row may see (they compare across frames of different sizes);
  counts of an entity are raw when the folds keep each entity's rows together, and shares otherwise.

Two modes, chosen from the prediction moment (:func:`mode_for`):

* ``audit`` (a daily audit, a window reviewed after the fact): when a row is scored, the rows up to the end of its day
  are known, never their labels; counts and means come from those rows of the frame being transformed (the whole frame
  only when there is no time column);
* ``event`` (scored as it happens): only strictly earlier rows of that frame count.

Either way a row's features come from the frame it is transformed in and the fit rows' labels, so no test row can
change a training row's features, and the event mode never uses a later row. Pure pandas and numpy: no model is fitted.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

# a review of a period after the fact; "after the ride" alone describes one ride, not a window
AUDIT_WORDS = re.compile(r"\b(audit|after the (day|fact)|end of (the )?(day|week|month)|daily review|"
                         r"review(ed)? (the|each) (day|week)|window|batch|overnight)\b", re.I)
LAT = re.compile(r"(^|_)(lat|latitude)$", re.I)
LON = re.compile(r"(^|_)(lon|lng|long|longitude)$", re.I)
MONEY = re.compile(r"(fare|price|amount|cost|fee|value|total)", re.I)
CELL_DIGITS = 4          # a place is a cell of about 11 m by 11 m
SHRINK = 20.0            # rows of prior weight in a shrunk rate
SHORT_METRES = 50.0      # a trip shorter than this ends where it starts


def mode_for(moment: str) -> str:
    """``audit`` when the prediction moment is a review after the fact, else ``event``."""
    return "audit" if AUDIT_WORDS.search(moment or "") else "event"


def coordinate_pairs(columns: Sequence[str]) -> list[tuple[str, str, str]]:
    """(prefix, lat column, lon column) for each prefix that has both, e.g. source_lat/source_lon."""
    out = []
    for column in columns:
        m = LAT.search(str(column))
        if not m:
            continue
        prefix = str(column)[:m.start()]
        lon = next((c for c in columns if LON.search(str(c)) and str(c)[:LON.search(str(c)).start()] == prefix), None)
        if lon is not None:
            out.append((prefix or "point", str(column), str(lon)))
    return out


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class EntityBehaviour:
    def __init__(self, entities: Sequence[str], *, time_column: Optional[str] = None, mode: str = "event",
                 numeric: Sequence[str] = (), allowed: Optional[Sequence[str]] = None, grouped: Sequence[str] = ()):
        """``allowed``: the columns a model may use (coordinates are looked for there only, never in a forbidden one);
        ``grouped``: entities the folds are grouped by (no rate, raw counts)."""
        self.grouped = list(grouped)
        if mode not in ("audit", "event"):
            raise ValueError("mode is audit or event")
        self.entities, self.time_column, self.mode, self.numeric = list(entities), time_column, mode, list(numeric)
        self.allowed = None if allowed is None else list(allowed)

    # -------------------------------------------------------------------- the frame's own structure

    def _hash(self, values: pd.Series) -> pd.Series:
        """Entity values and place cells as salted hashes when ``salt`` is set (an exported service keeps no raw key)."""
        salt = getattr(self, "salt", None)
        if not salt:
            return values
        import hashlib

        # every value as fit saw it (a missing one is "nan" there too), so a hashed key finds the same rate
        return values.astype(str).map(lambda v: "h:" + hashlib.sha256(f"{salt}|{v}".encode()).hexdigest()[:16])

    def _places(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        cells = {}
        for prefix, lat, lon in self.pairs:
            ok = frame[lat].notna() & frame[lon].notna()
            cell = frame[lat].round(CELL_DIGITS).astype(str) + "_" + frame[lon].round(CELL_DIGITS).astype(str)
            cells[prefix] = self._hash(cell.where(ok)) if getattr(self, "salt", None) else cell.where(ok)
        return cells

    def hide_keys(self, salt: str) -> None:
        """Keep the learned rates under salted hashes of their keys; every later transform hashes its keys the same way,
        so it computes the same features without any raw entity value or place in the object."""
        import hashlib

        h = lambda v: "h:" + hashlib.sha256(f"{salt}|{v}".encode()).hexdigest()[:16]  # noqa: E731
        self.rates = {name: {h(k): v for k, v in table.items()} for name, table in self.rates.items()}
        self.salt = salt

    def _day(self, frame: pd.DataFrame) -> Optional[pd.Series]:
        if not self.time_column or self.time_column not in frame:
            return None
        return pd.to_datetime(frame[self.time_column], errors="coerce").dt.floor("D").astype(str)

    def _order(self, frame: pd.DataFrame) -> np.ndarray:
        if self.time_column and self.time_column in frame:
            return np.argsort(pd.to_datetime(frame[self.time_column], errors="coerce").to_numpy(), kind="stable")
        return np.arange(len(frame))  # no time: the frame's own order

    def _stamp(self) -> Optional[pd.Series]:
        if not self.time_column or self.time_column not in self.current:
            return None
        return pd.to_datetime(self.current[self.time_column], errors="coerce").reset_index(drop=True)

    def _until_day(self, values: pd.Series, keys: pd.DataFrame, how: str) -> np.ndarray:
        """Per row, the sum (or count) over rows of the same keys whose day is on or before the row's day (audit); a row
        without a day has no window (NaN)."""
        from dclab_core.features import _window

        stamp = self._stamp()
        values = values.reset_index(drop=True).astype(float) if how == "sum" else pd.Series(np.ones(len(keys)))
        if stamp is None:  # no time: the window is the whole frame
            return values.groupby([keys[c] for c in keys.columns], dropna=False).transform("sum").to_numpy(dtype=float)
        return _window(keys.reset_index(drop=True), stamp.dt.floor("D"), values, strict=False)

    def _count(self, keys: pd.DataFrame) -> np.ndarray:
        """Rows sharing the keys: up to the end of the row's day (audit), or strictly earlier ones (event)."""
        keys = keys.astype(str).reset_index(drop=True)
        if self.mode == "audit":
            return self._until_day(pd.Series(np.ones(len(keys))), keys, "size")
        return self._earlier(keys, pd.Series(np.ones(len(keys))))

    def _mean(self, values: pd.Series, entity: pd.Series) -> np.ndarray:
        values = values.reset_index(drop=True).astype(float)
        entity = entity.astype(str).reset_index(drop=True)
        if self.mode == "audit":
            keys = entity.to_frame("e")
            total = self._until_day(values.fillna(0), keys, "sum")
            seen = self._until_day(values.notna().astype(float), keys, "sum")
            return total / np.where(seen > 0, seen, np.nan)
        keys = entity.to_frame("e")
        total = self._earlier(keys, values.fillna(0))
        seen = self._earlier(keys, values.notna().astype(float))
        return total / np.where(seen > 0, seen, np.nan)

    def _earlier(self, keys: pd.DataFrame, values: pd.Series) -> np.ndarray:
        """Event mode: the sum over rows of the same keys with a strictly earlier time (rows at the same time never see
        each other; a row without a time has no history); without a time column, the rows above."""
        from dclab_core.features import _window

        stamp = self._stamp()
        values = values.reset_index(drop=True).astype(float)
        if stamp is None:
            return (values.groupby([keys[c] for c in keys.columns], dropna=False).cumsum() - values).to_numpy(dtype=float)
        return _window(keys.reset_index(drop=True), stamp, values, strict=True)

    # -------------------------------------------------------------------- fit and transform

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "EntityBehaviour":
        """Learn the shrunk target rates of each entity and place from the fit rows (their labels never leave them)."""
        self.pairs = coordinate_pairs([c for c in frame.columns if self.allowed is None or c in self.allowed])
        self.money = [c for c in self.numeric if MONEY.search(str(c)) and c in frame]
        target = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        self.prior = float(target.mean()) if len(target) else 0.0
        self.rates: dict[str, dict[str, float]] = {}
        self.loo: dict[str, np.ndarray] = {}  # a fit row's rate without its own label
        keys = {f"entity_{e}": frame[e] for e in self.entities if e in frame and e not in self.grouped}
        keys.update({f"place_{p}": cell for p, cell in self._places(frame).items()})
        for name, key in keys.items():
            key = key.astype(str).reset_index(drop=True)
            grouped = target.groupby(key)
            sums, counts = grouped.sum(), grouped.size()
            self.rates[name] = ((sums + SHRINK * self.prior) / (counts + SHRINK)).to_dict()
            row_sum, row_n = key.map(sums).to_numpy(dtype=float), key.map(counts).to_numpy(dtype=float)
            self.loo[name] = (row_sum - target.to_numpy() + SHRINK * self.prior) / (row_n - 1 + SHRINK)
        self.fit_key = self._key(frame)
        return self

    @staticmethod
    def _key(frame: pd.DataFrame) -> tuple[int, int]:
        return len(frame), int(pd.util.hash_pandas_object(frame.reset_index(drop=True), index=True).sum())

    def _rate(self, name: str, key: pd.Series) -> np.ndarray:
        if self.own_rows:
            return self.loo[name]
        return key.astype(str).map(self.rates.get(name, {})).fillna(self.prior).to_numpy(dtype=float)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.current = frame.reset_index(drop=True)
        self.own_rows = self._key(frame) == getattr(self, "fit_key", None)  # the fit rows themselves: leave-one-out rates
        f = self.current
        # a share's denominator is the rows this row may see (earlier rows, or rows up to its day): never later ones
        rows = np.maximum(self._count(pd.DataFrame({"all": np.zeros(len(f))})), 1.0)
        out = pd.DataFrame(index=f.index)
        places = self._places(f)
        day = self._day(f)
        distance = None
        if len(self.pairs) >= 2:
            (_, lat1, lon1), (_, lat2, lon2) = self.pairs[0], self.pairs[1]
            distance = pd.Series(_haversine(f[lat1], f[lon1], f[lat2], f[lon2]))
            out["trip_metres"] = distance
            out["trip_short"] = (distance < SHORT_METRES).astype(float)
            out["route_share"] = self._count(pd.DataFrame({"a": places[self.pairs[0][0]], "b": places[self.pairs[1][0]]})) / rows
            for column in self.money:
                out[f"{column}_per_metre"] = f[column].astype(float) / np.maximum(distance, 20.0)
        for prefix, cell in places.items():
            out[f"place_{prefix}_share"] = self._count(cell.to_frame()) / rows
            out[f"place_{prefix}_rate"] = self._rate(f"place_{prefix}", cell)
        for entity in self.entities:
            if entity not in f:
                continue
            e = self._hash(f[entity]) if getattr(self, "salt", None) else f[entity]
            scale = 1.0 if entity in self.grouped else rows  # raw when the entity's rows stay together, else a share
            out[f"{entity}_count"] = self._count(e.to_frame()) / scale
            if entity not in self.grouped:
                out[f"{entity}_rate"] = self._rate(f"entity_{entity}", e)
            if day is not None:
                out[f"{entity}_day_count"] = self._count(pd.DataFrame({"e": e, "d": day})) / scale
            if self.pairs:
                last = self.pairs[-1][0]  # the drop-off (the last pair): a biker who keeps ending at one spot
                out[f"{entity}_{last}_repeat"] = self._count(pd.DataFrame({"e": e, "p": places[last]})) / scale
            if distance is not None:
                out[f"{entity}_short_share"] = self._mean(out["trip_short"], e)
                out[f"{entity}_trip_metres_mean"] = self._mean(distance, e)
            for column in self.numeric[:8]:
                if column in f and column not in [p for pair in self.pairs for p in pair[1:]]:
                    mean = self._mean(f[column], e)
                    out[f"{entity}_{column}_mean"] = mean
                    out[f"{column}_vs_{entity}"] = f[column].astype(float).to_numpy() - mean
        return out.astype(float)

"""Honest selection under many attempts (34.6, the MLE plan's 33.2).

When many candidates (families, settings, feature sets) are scored on the same cross-validation folds, the best of them
looks better than it is: its fold score holds its own luck, and the more attempts, the more luck the winner holds. A
holdout read once still tells the truth, but only after the choice; the choice itself is flattered.

The protocol, fixed before any run: the training rows are split once into **search rows** (where candidates are
proposed and scored by cross-validation) and a **selection fold** (read only at the end, to choose among the top K by
their search score), cut the way the project splits (the latest rows for time, whole groups for entities, by class
otherwise). With ``top_k=1`` (the default) the selection fold only confirms the search's winner: its score there is an
honest estimate of how it does on new rows, and the difference from its search score is the measured optimism, written
into the record beside the number of attempts. Choosing among more than one on the selection fold picks a better
candidate more often but its selection score then holds that choice's own luck (measured 2026-10-25 on simulated
candidates: about 0.025 of a score with K=3, whatever the number of attempts, against about 0.002 with K=1), so the
record says so.

Pure numpy; the host fits and scores the candidates (DCLab's engine, or the main product's experiment runner).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field

from .contracts import CONTRACTS, Versioned, _version

TOP_K = 1  # measured: only confirming the winner leaves its selection score honest (see the module's text)
SELECTION_SHARE = 0.2
BOOTSTRAP = 200
SPLITS = ("time", "group", "stratified", "random")


def selection_split(n: int, *, y: Optional[Sequence[Any]] = None, groups: Optional[Sequence[Any]] = None,
                    order: Optional[Sequence[Any]] = None, share: float = SELECTION_SHARE, seed: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    """The search rows and the selection fold, cut the way the project splits: ``order`` (a time) puts the latest
    ``share`` of rows in the selection fold; ``groups`` keep each entity on one side; ``y`` (classes) keeps each class's
    share; otherwise rows at random. Returns ``(search, selection, kind)`` as row positions."""
    if n < 10:
        raise ValueError("A selection fold needs at least 10 rows to set aside")
    if not 0 < share < 0.5:
        raise ValueError("The selection fold is a share of the training rows between 0 and 0.5")
    rng = np.random.default_rng(seed)
    rows = np.arange(n)
    if order is not None:
        ranked = np.argsort(np.asarray(order), kind="mergesort")
        cut = n - max(1, int(round(share * n)))
        return np.sort(ranked[:cut]), np.sort(ranked[cut:]), "time"
    if groups is not None:
        codes = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)[1]
        keys = rng.permutation(codes.max() + 1)
        sizes = np.bincount(codes)
        taken, chosen = 0, []
        for k in keys:
            if taken >= share * n:
                break
            chosen.append(k)
            taken += sizes[k]
        if len(chosen) == len(keys):
            raise ValueError("Too few groups to set whole groups aside for the selection fold")
        selection = np.isin(codes, chosen)
        return rows[~selection], rows[selection], "group"
    if y is not None:
        labels = np.asarray(y, dtype=object).astype(str)
        selection = np.zeros(n, dtype=bool)
        for value in np.unique(labels):
            members = rows[labels == value]
            k = int(round(share * len(members)))
            selection[rng.choice(members, size=k, replace=False)] = True
        return rows[~selection], rows[selection], "stratified"
    selection = np.zeros(n, dtype=bool)
    selection[rng.choice(n, size=max(1, int(round(share * n))), replace=False)] = True
    return rows[~selection], rows[selection], "random"


class SelectionRecord(Versioned):
    """How a candidate was chosen among many (34.6, v1): the attempts scored on the search rows, the top K read once on
    the selection fold, the one chosen, both its scores and the optimism between them (oriented so that a positive
    optimism means the search score flattered it), with the interval of its selection score."""

    schema_version: Literal["dclab_core.selection_record.v1"] = _version("selection_record")
    split: Literal["time", "group", "stratified", "random"]
    search_rows: int = Field(ge=1)
    selection_rows: int = Field(ge=1)
    attempts: int = Field(ge=1)
    top_k: int = Field(ge=1)
    metric: str = Field(min_length=1, max_length=80)
    higher_is_better: bool = True
    shortlist: list[str] = Field(default_factory=list, max_length=50)
    chosen: str = Field(min_length=1, max_length=200)
    plain_choice: str = Field(min_length=1, max_length=200, description="The candidate plain cross-validation would have chosen.")
    search_score: float
    selection_score: float
    selection_interval: list[Optional[float]] = Field(default_factory=lambda: [None, None], min_length=2, max_length=2)
    optimism: float
    note: str = Field(default="", max_length=1000)


CONTRACTS["selection_record"] = SelectionRecord


def choose(search_scores: Mapping[str, float], score_on_selection: Callable[[str, Optional[np.ndarray]], float], *, metric: str,
           higher_is_better: bool = True, top_k: int = TOP_K, split: str = "random", search_rows: int = 1, selection_rows: int = 1,
           bootstrap: int = BOOTSTRAP, seed: int = 0) -> SelectionRecord:
    """The candidate to keep among ``search_scores`` (each candidate's cross-validated score on the search rows, in the
    order they were tried): the top ``top_k`` by that score are each scored once on the selection fold by
    ``score_on_selection(name, None)``; the best there is chosen (a tie goes to the one tried first, the simpler by the
    engine's order). ``score_on_selection(name, rows)`` with bootstrap positions into the selection fold gives the
    chosen one's interval. Nothing here reads the holdout."""
    if not search_scores:
        raise ValueError("No candidate was scored")
    if split not in SPLITS:
        raise ValueError(f"A selection split is one of {', '.join(SPLITS)}")
    names = list(search_scores)
    values = np.array([float(search_scores[k]) for k in names])
    if not np.isfinite(values).all():
        raise ValueError("Every candidate needs a finite search score")
    sign = 1.0 if higher_is_better else -1.0
    ranked = sorted(range(len(names)), key=lambda i: (-sign * values[i], i))
    shortlist = [names[i] for i in ranked[: max(1, top_k)]]
    on_selection = {k: float(score_on_selection(k, None)) for k in shortlist}
    if not all(np.isfinite(v) for v in on_selection.values()):
        raise ValueError("Every shortlisted candidate needs a finite selection score")
    chosen = sorted(shortlist, key=lambda k: (-sign * on_selection[k], names.index(k)))[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        rows = rng.integers(0, selection_rows, size=selection_rows)
        value = score_on_selection(chosen, rows)
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    interval = [round(float(np.quantile(draws, 0.05)), 6), round(float(np.quantile(draws, 0.95)), 6)] if len(draws) >= 20 else [None, None]
    search = float(search_scores[chosen])
    optimism = sign * (search - on_selection[chosen])
    plain = names[ranked[0]]
    note = (f"{len(names)} attempts scored on the search rows; the top {len(shortlist)} read once on a selection fold of "
            f"{selection_rows} rows set aside before the search; the chosen one's search score was "
            + ("flattered" if optimism > 0 else "not flattered") + f" by {abs(optimism):.4g}"
            + ("" if chosen == plain else f"; plain cross-validation would have kept {plain}") + "."
            + (f" Chosen among {len(shortlist)} on the selection fold, its score there holds that choice's luck too: read it as an upper"
               " estimate." if len(shortlist) > 1 else ""))
    return SelectionRecord(split=split, search_rows=search_rows, selection_rows=selection_rows, attempts=len(names), top_k=len(shortlist),
                           metric=metric, higher_is_better=higher_is_better, shortlist=shortlist, chosen=chosen, plain_choice=plain,
                           search_score=round(search, 6), selection_score=round(on_selection[chosen], 6), selection_interval=interval,
                           optimism=round(optimism, 6), note=note)

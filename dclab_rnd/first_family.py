"""The student for ``model.first`` (S7 of ``docs/guides/STUDENTS_100_PLAN.md``, "more ability"; the owner, 2026-10-25):
``python -m dclab_rnd.first_family train|measure``.

Which model family should be tried first on a table, before the screen measures them all? The homework is the screens
of real tables (``dclab_rnd.real_screen``: OpenML tables through DCLab's own engine) and of benchgen v3's craft tables:
each screen's table at a glance (``ledger.states.table_state``: counts and bands, never a value) and every family's
training-fold score. A pick is judged three ways, never by exact match alone (two families within noise of each other
are equally good first tries):

- **hit**: the pick is the screen's winner;
- **fair**: the pick is within the folds' noise of the winner (its gap at most the two folds' spreads combined);
- **regret**: the pick's gap to the winner in units of that noise (0 for the winner).

Beside two baselines measured on the same dev screens: today's rule (the kind's fixed first family, ``rule:kind_order``)
and "always the commonest winner" of the training screens for the task. Dev screens of real tables only measure; sealed
screens are never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "agent_runs" / "real_screens"
CRAFT = ROOT / "agent_runs" / "craft_screens"
OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "models_first_v1"
LOWER_IS_BETTER = {"mae", "rmse", "mse", "mape", "smape", "rmsle", "log_loss", "brier"}
POINT = "model.first"


def higher(metric: Optional[str]) -> bool:
    return str(metric or "").lower() not in LOWER_IS_BETTER


PROFILE = "full"  # the product's own settings: a screen under the fast test profile (a few trees) is never read


def screens(home: Path, split: str) -> list[dict[str, Any]]:
    """A folder's screens of one split, each table once (its latest), measured under the product's own settings, with a
    state and at least two scored families."""
    path = Path(home) / split / "screens.jsonl"
    out: dict[Any, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        scored = [s for s in r.get("screen") or [] if isinstance(s.get("mean"), (int, float)) and math.isfinite(s["mean"])]
        if r.get("state") and len(scored) >= 2 and r.get("profile") == PROFILE:
            out[r.get("did")] = {**r, "screen": scored}
    return list(out.values())


def winner(screen: dict[str, Any]) -> str:
    sign = 1.0 if higher(screen.get("metric")) else -1.0
    return max(screen["screen"], key=lambda s: sign * s["mean"])["model"]


REGRET_CAP = 20.0  # a gap beyond twenty noises is one mistake, never a mean that a single flat screen dominates


def _std(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else 0.0


def regret(screen: dict[str, Any], pick: Optional[str]) -> Optional[float]:
    """The pick's gap to the winner in units of the two folds' spreads combined (0 for the winner), capped at
    ``REGRET_CAP``; a spread of zero or unknown counts as a floor of 1e-9 of the winner's score, so a gap there is the cap
    and a tie is 0. None for a family the screen did not score."""
    by = {s["model"]: s for s in screen["screen"]}
    best = by[winner(screen)]
    if pick not in by:
        return None
    if pick == best["model"]:
        return 0.0
    gap = (best["mean"] - by[pick]["mean"]) * (1.0 if higher(screen.get("metric")) else -1.0)
    noise = max(math.sqrt(_std(best.get("std")) ** 2 + _std(by[pick].get("std")) ** 2), 1e-9 * max(abs(best["mean"]), 1.0))
    return round(min(max(gap, 0.0) / noise, REGRET_CAP), 4)


def clear(screen: dict[str, Any]) -> Optional[str]:
    """The family clearly best (ahead of every other by their combined spreads: the ledger's own measured label), or None."""
    from dclab_core.measured import clear_best

    got = clear_best([{"id": s["model"], "mean": s["mean"], "std": s.get("std")} for s in screen["screen"]], higher(screen.get("metric")))
    return got[0] if got else None


def judge(screens_: list[dict[str, Any]], picks: list[Optional[str]]) -> dict[str, Any]:
    """Hit, fair and mean regret of picks on screens (an unscored pick counts as missed and as the cap's regret), and the
    hit on the screens with a clear winner (what the ledger's measured label, and so the gate, judge)."""
    regrets = [regret(s, p) for s, p in zip(screens_, picks)]
    n = len(screens_)
    clears = [(clear(s), p) for s, p in zip(screens_, picks)]
    clears = [(c, p) for c, p in clears if c is not None]
    return {"n": n, "hit": round(sum(r == 0.0 for r in regrets if r is not None) / n, 4) if n else None,
            "fair": round(sum(r is not None and r <= 1.0 for r in regrets) / n, 4) if n else None,
            "mean_regret": round(sum(REGRET_CAP if r is None else r for r in regrets) / n, 4) if n else None,
            "unscored": sum(r is None for r in regrets), "clear_n": len(clears),
            "clear_hit": round(sum(c == p for c, p in clears) / len(clears), 4) if clears else None}


def rows(screens_: Iterable[dict[str, Any]], split: str, source: str) -> list[dict[str, Any]]:
    """Decision rows as ``dclab_core.decision_models.train`` reads them: the state, the winner as a measured label, the table
    as its group, the families screened as the options."""
    return [{"record": f"{source}:{s['did']}", "group": f"{source}:{s['did']}", "split": split, "state": s["state"],
             "options": [x["model"] for x in s["screen"]], "value_used": s.get("first"),
             "label": {"value": winner(s), "source": "measured"}} for s in screens_]  # the screen's best mean; clear winners are judged apart


def commonest(train: list[dict[str, Any]]) -> dict[str, str]:
    """Per task (the screen's own), the family that won most often among the training screens."""
    by: dict[str, Counter] = defaultdict(Counter)
    for s in train:
        by[str(s.get("task"))][winner(s)] += 1
    return {task: c.most_common(1)[0][0] for task, c in by.items()}


LIMITS = ["Measured on real OpenML tables' dev screens (their uploaders' outcomes), under the product's own model settings; the sealed screens are read once, by the gate.",
          "The truth is the training folds' best mean; families within noise of it are equally good first tries (fair, regret).",
          "Craft tables (benchgen v3) are invented: an arm of their own beside the real-only one."]


def _arm(homework: list[dict[str, Any]], dev_screens: list[dict[str, Any]], common: dict[str, str]) -> tuple[Any, Any, dict[str, Any]]:
    from dclab_core.decision_models import train as fit

    model, card = fit(POINT, homework, rows(dev_screens, "dev", "real"), limits=LIMITS)
    picks = [getattr(model.answer(POINT, s["state"], [x["model"] for x in s["screen"]]), "value", None) for s in dev_screens]
    return model, card, {"student": judge(dev_screens, picks), "card": {"name": card.name, "version": card.version, "family": card.family}}


def train(real: Path = REAL, craft: Path = CRAFT, out: Path = OUT) -> dict[str, Any]:
    """Two arms, each measured on the real dev screens beside the rule and the commonest winner of the real training
    screens: the student on real training screens only, and on real and craft ones. The real-only arm's model is kept
    (the craft arm's is reported: it serves only where it measured better). The sealed screens are never opened."""
    import joblib

    real_train, dev_screens = screens(real, "train"), screens(real, "dev")
    if len(real_train) < 30 or len(dev_screens) < 20:
        raise SystemExit(f"too little homework yet: {len(real_train)} real training screens, {len(dev_screens)} real dev screens (at least 30 and 20)")
    craft_rows = rows(screens(craft, "train") + screens(craft, "dev"), "train", "craft")
    common = commonest(real_train)
    model, card, real_only = _arm(rows(real_train, "train", "real"), dev_screens, common)
    _, _, with_craft = _arm(rows(real_train, "train", "real") + craft_rows, dev_screens, common) if craft_rows else (None, None, None)
    result = {"point": POINT, "trained_on": {"real": len(real_train), "craft": len(craft_rows)}, "dev_real_screens": len(dev_screens),
              "rule": judge(dev_screens, [s.get("first") for s in dev_screens]),
              "commonest": judge(dev_screens, [common.get(str(s.get("task"))) for s in dev_screens]),
              "real_only": real_only, "with_craft": with_craft, "kept": "real_only"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{POINT}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
    joblib.dump(model, out / f"{POINT}.joblib")
    (out / "MEASURED.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.first_family", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("train", "measure"))
    args = ap.parse_args(argv)
    if args.command == "train":
        print(json.dumps(train(), indent=1))
    else:
        dev = screens(REAL, "dev")
        common = commonest(screens(REAL, "train"))
        print(json.dumps({"dev_real_screens": len(dev), "rule": judge(dev, [s.get("first") for s in dev]),
                          "commonest": judge(dev, [common.get(str(s.get("task"))) for s in dev])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

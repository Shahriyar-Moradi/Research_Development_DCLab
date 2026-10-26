"""Why is the daily decision interval narrow? Ties, or the day's best? (campaign ``daily_ties_v1``; 34.1)

time_decision_v4 and v5 (120 seeds of the churn replica, one call a day, no drift) found the day by day interval
about 15% narrow (RMS z 1.16), with drift, the latest fold's smaller pool and the model's fit-to-fit spread ruled out.
This campaign looks at the method alone, on invented worlds where every row's chance of being a case is known: no
engine, no model, so a fault it finds is the decision core's.

Each world: 2,400 training rows (the out-of-fold rows, their scores the true chances) and a holdout of 600 new rows in
days, the same chances' distribution (no drift); a case is worth 120, a call costs 10; ``choose_operating_point`` draws
its interval from the latest 600 training rows (as the engine does on a time split) and reads the holdout once. 200
worlds an arm:

* ``control``: about 4.4 rows a day, one call a day, scores continuous (no ties);
* ``ties``: the same, scores rounded to a twentieth (many ties among a day's rows);
* ``big_days``: about 20 rows a day, one call a day (the day's best of twenty: a more extreme pick);
* ``share``: about 4.4 rows a day, at most 10% of the rows, one pool (it passed on the replica: the reference);
* ``share_05``: at most 5% of the rows, one pool: the same 30 actions as ``big_days`` without days (the review: big days
  also means few actions, so a day's best is read against the same count in one pool).

A limit, written before the run: the core breaks a tie by row order, the worlds' rows are in random order and tied rows
share their chance, so the ``ties`` arm can hardly show a tie effect; it does not clear the replica, whose holdout is in
time order.

The gate, fixed in ``manifest.json`` before the first run: for each arm, z = (realised - centre) / (half-width / 1.645)
over its worlds; an arm is off-centre when |mean z| > 1.645 RMS z / sqrt(n); narrow when the 90% chi-squared interval of
the root mean square of z lies above 1, wide when below, right when it contains 1; coverage reported beside it. Two arms
are compared by a paired bootstrap over the same worlds (2,000 draws) of their RMS z's difference.

The next step is fixed with the gate, in this order:

* an arm off-centre: read the worlds (a centre, not a width);
* ``ties`` narrow, ``control`` right and ``big_days`` not narrow: break ties at random in the samples;
* ``big_days`` narrow, ``share_05`` right, and big_days minus share_05 above 0 (paired interval): resampling rows
  understates how much a day's best varies (the bootstrap's weakness for extremes): resample whole days or draw m of n,
  a package of its own;
* ``big_days`` and ``share_05`` both narrow: few actions, not the day's best: a small-count correction, a package of its
  own;
* ``control`` right with its interval's upper end below the replica's 1.16 and no arm narrow: the method is right on known
  worlds; the replica's narrowness comes from its model and data: read the replica's seeds;
* ``control`` right but its upper end at 1.16 or above: inconclusive, more worlds;
* otherwise: read the worlds.

    python -m dclab_rnd.daily_ties_measure plan
    python -m dclab_rnd.daily_ties_measure run --yes     # the decision core only; minutes on the Air
    python -m dclab_rnd.daily_ties_measure report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "daily_ties_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / CAMPAIGN_ID
WORLDS = tuple(range(1000, 1200))  # 200 worlds an arm, the same worlds in every arm
TRAIN, TEST, LATEST = 2400, 600, 600
BOOTSTRAP = 200
ARMS = {
    "control": {"rows_per_day": 4.4, "capacity_per_day": 1.0, "round_to": None},
    "ties": {"rows_per_day": 4.4, "capacity_per_day": 1.0, "round_to": 0.05},
    "big_days": {"rows_per_day": 20.0, "capacity_per_day": 1.0, "round_to": None},
    "share": {"rows_per_day": 4.4, "capacity_share": 0.1, "round_to": None},
    "share_05": {"rows_per_day": 20.0, "capacity_share": 0.05, "round_to": None},
}
REPLICA_RMS = 1.16  # time_decision v4 and v5 without refits, 120 seeds
VALUE, COST = 120.0, 10.0
GATE = {
    "id": "daily_ties_gate_v1",
    "centre": "an arm is off-centre when |mean z| > 1.645 RMS z / sqrt(n)",
    "width": "per arm, the 90% chi-squared interval of the root mean square of z over its 200 worlds: above 1 narrow, below 1 "
             "wide, containing 1 right; coverage reported beside it, never gated",
    "paired": "big_days against share_05: a paired bootstrap (2,000 draws over the same worlds) of the difference of their RMS z",
    "next": "in order: an arm off-centre: read the worlds; ties narrow, control right, big_days not narrow: break ties at random "
            "in the samples; big_days narrow, share_05 right, big_days minus share_05 above 0: resample whole days or draw m of n; "
            "big_days and share_05 narrow: a small-count correction; control right with its upper end below 1.16 and no arm "
            "narrow: read the replica's seeds; control right with its upper end at 1.16 or above: inconclusive, more worlds; "
            "otherwise read the worlds",
    "limit": "a tie is broken by row order; the worlds' rows are in random order and tied rows share their chance, so the ties "
             "arm can hardly show a tie effect and does not clear the replica, whose holdout is in time order",
}
CODE_FILES = ("dclab_rnd/daily_ties_measure.py", "dclab_core/decide.py")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_sha256() -> str:
    from .campaign_capture import code_sha256 as whole

    return whole(CODE_FILES)


CODE_AT_START = code_sha256()


def manifest() -> dict[str, Any]:
    body = {"campaign": CAMPAIGN_ID, "package": "34.1", "worlds": [WORLDS[0], WORLDS[-1]], "arms": ARMS, "gate": GATE,
            "world": {"train": TRAIN, "test": TEST, "latest": LATEST, "bootstrap": BOOTSTRAP, "value_per_case": VALUE,
                      "cost_per_action": COST, "chances": "logistic of a normal (mean -1.5, sd 1.0): about 21% cases, as the replica"},
            "objective": ("On invented worlds with known chances, is the decision core's day by day interval narrow, and is it the "
                          "ties among a day's rows or the extremity of a day's best that makes it so?")}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def freeze() -> dict[str, Any]:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / "manifest.json"
    want = manifest()
    if path.exists():
        held = json.loads(path.read_text())
        if held.get("sha256") != want["sha256"] or held.get("code_sha256") != CODE_AT_START:
            raise SystemExit("The gate, the worlds or the code changed since the campaign was frozen: start a new campaign version")
        return held
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    held = {**want, "code_sha256": CODE_AT_START, "code_files": list(CODE_FILES), "git_commit": commit, "frozen_at": _now()}
    path.write_text(json.dumps(held, indent=2) + "\n")
    return held


def _world_rows(seed: int, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A world's outcomes, scores and holdout days, from its own random stream (the review: never the core's)."""
    a = ARMS[arm]
    rng = np.random.default_rng([seed, 0])
    chance = 1 / (1 + np.exp(-rng.normal(-1.5, 1.0, TRAIN + TEST)))
    y = (rng.random(TRAIN + TEST) < chance).astype(int)
    score = np.round(chance / a["round_to"]) * a["round_to"] if a["round_to"] else chance
    days = np.floor(rng.random(TEST) * (TEST / a["rows_per_day"])).astype(int)  # the holdout's rows fall in days at random
    return y, score, days


def world(seed: int, arm: str) -> dict[str, Any]:
    """One invented world read by the decision core: the realised value per row, the interval and z."""
    from dclab_core.decide import DecisionSpec, choose_operating_point

    a = ARMS[arm]
    y, score, days = _world_rows(seed, arm)
    per_day = a["rows_per_day"]
    caps = {k: a[k] for k in ("capacity_per_day", "capacity_share") if k in a}
    spec = DecisionSpec(action="call", value_per_case=VALUE, cost_per_action=COST, **caps)
    out = choose_operating_point(y[:TRAIN], score[:TRAIN], spec, rows_per_day=per_day, y_test=y[TRAIN:], score_test=score[TRAIN:],
                                 days_test=days if "capacity_per_day" in caps else None, bootstrap=BOOTSTRAP, seed=seed,
                                 sample_rows=np.arange(TRAIN) >= TRAIN - LATEST)
    h = out.get("holdout") or {}
    interval, realised = h.get("prediction_interval") or [None, None], h.get("expected_value_per_row")
    z = inside = None
    if realised is not None and None not in interval and interval[1] > interval[0]:
        inside = bool(interval[0] <= realised <= interval[1])
        z = (realised - (interval[0] + interval[1]) / 2) / ((interval[1] - interval[0]) / 2 / 1.645)
    return {"world": seed, "arm": arm, "share": out.get("share"), "realised": realised, "interval": interval, "inside": inside,
            "z": None if z is None else round(float(z), 4), "actions": h.get("actions")}


def judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import chi2

    zs = [float(r["z"]) for r in rows if r.get("z") is not None]
    if len(zs) < 2:
        return {"worlds": len(rows), "read": len(zs), "width": None}
    n = len(zs)
    rms = (sum(z * z for z in zs) / n) ** 0.5
    low, high = (n * rms * rms / chi2.ppf(0.95, n)) ** 0.5, (n * rms * rms / chi2.ppf(0.05, n)) ** 0.5
    mean = sum(zs) / n
    return {"worlds": len(rows), "read": n, "rms_z": round(rms, 4), "interval": [round(float(low), 4), round(float(high), 4)],
            "width": "narrow" if low > 1 else "wide" if high < 1 else "right", "mean_z": round(mean, 4),
            "off_centre": bool(abs(mean) > 1.645 * rms / n ** 0.5), "coverage": round(sum(1 for r in rows if r.get("inside")) / n, 4)}


def paired_difference(first: list[dict[str, Any]], second: list[dict[str, Any]], draws: int = 2000) -> list[float] | None:
    """The 90% paired bootstrap interval of RMS z (first) minus RMS z (second) over the worlds both read."""
    a = {r["world"]: float(r["z"]) for r in first if r.get("z") is not None}
    b = {r["world"]: float(r["z"]) for r in second if r.get("z") is not None}
    common = sorted(set(a) & set(b))
    if len(common) < 20:
        return None
    za, zb = np.array([a[w] for w in common]), np.array([b[w] for w in common])
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(common), size=(draws, len(common)))
    diff = np.sqrt((za[picks] ** 2).mean(axis=1)) - np.sqrt((zb[picks] ** 2).mean(axis=1))
    return [round(float(np.quantile(diff, 0.05)), 4), round(float(np.quantile(diff, 0.95)), 4)]


def next_step(arms: dict[str, dict[str, Any]], big_minus_share: list[float] | None) -> str:
    width = {a: v.get("width") for a, v in arms.items()}
    if any(v.get("off_centre") for v in arms.values()):
        return "read the worlds"
    if width.get("ties") == "narrow" and width.get("control") == "right" and width.get("big_days") != "narrow":
        return "break ties at random in the samples"
    if width.get("big_days") == "narrow" and width.get("share_05") == "right" and big_minus_share and big_minus_share[0] > 0:
        return "resample whole days or draw m of n, a package of its own"
    if width.get("big_days") == "narrow" and width.get("share_05") == "narrow":
        return "a small-count correction, a package of its own"
    control = arms.get("control") or {}
    if width.get("control") == "right" and not any(w == "narrow" for w in width.values()):
        return "read the replica's seeds" if control["interval"][1] < REPLICA_RMS else "inconclusive: more worlds"
    if width.get("control") == "right" and control["interval"][1] >= REPLICA_RMS:
        return "inconclusive: more worlds"
    return "read the worlds"


def report() -> dict[str, Any]:
    folder = CAMPAIGN / "results"
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("DTI-0*_*.json"))] if folder.exists() else []
    worlds = {arm: [r for f in rows if f.get("arm") == arm for r in f.get("worlds", [])] for arm in ARMS}
    arms = {arm: judge(worlds[arm]) for arm in ARMS}
    complete = all(v.get("worlds") == len(WORLDS) for v in arms.values())
    diff = paired_difference(worlds["big_days"], worlds["share_05"]) if complete else None
    out = {"arms": arms, "big_days_minus_share_05": diff, "next": next_step(arms, diff) if complete else None,
           "verdict": "complete" if complete else "not yet", "gate": GATE}
    if complete and not (folder / "DTI-100_decision.json").exists():  # written once
        (folder / "DTI-100_decision.json").write_text(json.dumps({**out, "id": "DTI-100", "experiment_id": "DTI-100", "status": "completed",
                                                                  "completed_at": _now(), "campaign_id": CAMPAIGN_ID}, indent=2) + "\n")
    return out


def run_arm(arm: str) -> dict[str, Any]:
    rows = [world(seed, arm) for seed in WORLDS]
    number = f"DTI-{list(ARMS).index(arm) + 1:03d}"
    result = {"id": number, "experiment_id": number, "status": "completed", "completed_at": _now(), "campaign_id": CAMPAIGN_ID,
              "arm": arm, "settings": ARMS[arm], "summary": judge(rows), "worlds": rows, "code_sha256": CODE_AT_START}
    folder = CAMPAIGN / "results"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{number}_{arm}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="34.1: is the daily interval narrow on known worlds, and why")
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan" or (args.command == "run" and not args.yes):
        print(json.dumps({**manifest(), "code_sha256": CODE_AT_START}, indent=2))
        return 0
    if args.command == "report":
        print(json.dumps(report(), indent=2))
        return 0
    freeze()
    done = {json.loads(p.read_text())["arm"] for p in (CAMPAIGN / "results").glob("DTI-0*_*.json")} if (CAMPAIGN / "results").exists() else set()
    for arm in ARMS:
        if arm in done:
            continue
        print(f"[daily-ties] {arm} {_now()}", flush=True)
        r = run_arm(arm)
        print(f"[daily-ties] {arm}: {r['summary']}", flush=True)
    print(json.dumps(report(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

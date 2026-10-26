"""Why does the realised value sit a little above the decision interval's centre? (campaign ``daily_centre_v1``; 34.1)

daily_ties_v1 (200 invented worlds an arm, known chances, the decision core alone) found every arm's width right but the
realised value a little above the interval's centre in every arm (mean z +0.10 to +0.16; off-centre in three). Two
possible causes, each with its own trace:

* **A, a small pool.** Each holdout-sized sample (600 rows) is drawn from the latest fold's rows of one bootstrap draw:
  about 63% of the fold's 600 rows are distinct there. The best-scored rows of 600 drawn from about 380 are less
  extreme than those of 600 fresh rows, so the samples pick slightly less promising rows than the holdout does: the
  interval sits low. The more extreme the pick, the larger the gap (twenty rows a day read +0.16).
* **B, a skewed spread.** The interval's centre is the midpoint of its 5th and 95th percentiles; with a long lower tail
  the realised values' mean sits above that midpoint.

Four arms, 800 worlds each, the same worlds in every arm (paired), drawn as daily_ties_v1 draws them (chances from a
logistic of a normal, 2,400 training rows, 600 holdout rows in days; a case worth 120, a call 10): ``daily_latest`` and
``share_latest`` (one call a day; at most 5% in one pool), samples from the latest 600 training rows as the engine
draws them; ``daily_all`` and ``share_all``, samples from all 2,400 rows, reported only (the review: a pool four times
larger also holds four times the information and narrows the interval, so it cannot single out the pool).

Each world keeps the interval's midpoint (its centre), the samples' mean (dclab-core 0.120.1) and the realised value;
z_mid = (realised - midpoint) / (half-width / 1.645), z_mean the same from the samples' mean.

The gate, fixed in ``manifest.json`` before the first run, for each decision (daily, share) on its latest arm, each with
its own verdict:

* the bias: the 90% bootstrap interval (2,000 draws over the worlds) of mean z_mid. Established when it lies above 0 (or
  below); closed when it lies inside -0.08 to 0.08; otherwise inconclusive (more worlds);
* where established, the skew: the 90% interval of mean (z_mid - z_mean) per world (the samples' mean against the
  midpoint, the realised value cancelling out). The midpoint is the cause when that interval lies on the bias's side of
  0 and mean z_mean's interval reaches 0: centre the interval on the samples' mean;
* else, for a bias above 0 only, the pool: the paired gap of the samples' means in value units, latest minus all, with its
  90% interval below 0: draw each sample from the latest fold's own rows; otherwise read the worlds.

    python -m dclab_rnd.daily_centre_measure plan
    python -m dclab_rnd.daily_centre_measure run --yes     # the decision core only; minutes on the Air
    python -m dclab_rnd.daily_centre_measure report
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
CAMPAIGN_ID = "daily_centre_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / CAMPAIGN_ID
WORLDS = tuple(range(2000, 2800))  # fresh: daily_ties_v1 read 1000 to 1199; 800, so a mean z's interval (about +-0.06) can close
TRAIN, TEST, LATEST = 2400, 600, 600
BOOTSTRAP = 200
VALUE, COST = 120.0, 10.0
ARMS = {
    "daily_latest": {"rows_per_day": 4.4, "capacity_per_day": 1.0, "pool": LATEST},
    "daily_all": {"rows_per_day": 4.4, "capacity_per_day": 1.0, "pool": TRAIN},
    "share_latest": {"rows_per_day": 4.4, "capacity_share": 0.05, "pool": LATEST},
    "share_all": {"rows_per_day": 4.4, "capacity_share": 0.05, "pool": TRAIN},
}
PAIRS = {"daily": ("daily_latest", "daily_all"), "share": ("share_latest", "share_all")}
BAND = 0.08  # the review: at 200 worlds a mean z's 90% interval (about +-0.12) could never fit inside +-0.05
GATE = {
    "id": "daily_centre_gate_v1",
    "bias": "per decision on its latest arm, the 90% bootstrap interval (2,000 draws over the 800 worlds) of mean z_mid: above "
            "0 or below 0 established; inside -0.08 to 0.08 closed; otherwise inconclusive (more worlds)",
    "cause": "where established: mean (z_mid - z_mean)'s interval on the bias's side of 0 and mean z_mean's interval reaching 0: "
             "the midpoint (skew): centre the interval on the samples' mean; else for a bias above 0, the paired gap of the "
             "samples' means (latest minus all, value units) with its 90% interval below 0: the small pool: draw each sample "
             "from the latest fold's own rows; else read the worlds",
    "reported": "the all-rows arms and the centre gap, beside the gate; each decision its own verdict",
}
CODE_FILES = ("dclab_rnd/daily_centre_measure.py", "dclab_core/decide.py")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_sha256() -> str:
    from .campaign_capture import code_sha256 as whole

    return whole(CODE_FILES)


CODE_AT_START = code_sha256()


def manifest() -> dict[str, Any]:
    body = {"campaign": CAMPAIGN_ID, "package": "34.1", "worlds": [WORLDS[0], WORLDS[-1]], "arms": ARMS, "pairs": PAIRS, "gate": GATE,
            "world": {"train": TRAIN, "test": TEST, "bootstrap": BOOTSTRAP, "value_per_case": VALUE, "cost_per_action": COST,
                      "chances": "logistic of a normal (mean -1.5, sd 1.0), as daily_ties_v1"},
            "objective": "Is the realised value's small lead over the interval's centre real, and is it the percentile midpoint (the samples' skew) or the samples' mean itself?"}
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


def _world_rows(seed: int, rows_per_day: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A world's outcomes, chances (the scores) and holdout days, from its own random stream: the same in every arm."""
    rng = np.random.default_rng([seed, 0])
    chance = 1 / (1 + np.exp(-rng.normal(-1.5, 1.0, TRAIN + TEST)))
    y = (rng.random(TRAIN + TEST) < chance).astype(int)
    days = np.floor(rng.random(TEST) * (TEST / rows_per_day)).astype(int)
    return y, chance, days


def world(seed: int, arm: str) -> dict[str, Any]:
    from dclab_core.decide import DecisionSpec, choose_operating_point

    a = ARMS[arm]
    y, score, days = _world_rows(seed, a["rows_per_day"])
    caps = {k: a[k] for k in ("capacity_per_day", "capacity_share") if k in a}
    spec = DecisionSpec(action="call", value_per_case=VALUE, cost_per_action=COST, **caps)
    out = choose_operating_point(y[:TRAIN], score[:TRAIN], spec, rows_per_day=a["rows_per_day"], y_test=y[TRAIN:], score_test=score[TRAIN:],
                                 days_test=days if "capacity_per_day" in caps else None, bootstrap=BOOTSTRAP, seed=seed,
                                 sample_rows=np.arange(TRAIN) >= TRAIN - a["pool"])
    h = out.get("holdout") or {}
    interval, realised, mean = h.get("prediction_interval") or [None, None], h.get("expected_value_per_row"), h.get("samples_mean")
    row = {"world": seed, "arm": arm, "realised": realised, "interval": interval, "samples_mean": mean, "actions": h.get("actions"),
           "midpoint": None, "z_mid": None, "z_mean": None}
    if realised is not None and None not in interval and interval[1] > interval[0]:
        mid, unit = (interval[0] + interval[1]) / 2, (interval[1] - interval[0]) / 2 / 1.645
        row.update(midpoint=round(mid, 4), z_mid=round((realised - mid) / unit, 4),
                   z_mean=None if mean is None else round((realised - mean) / unit, 4))
    return row


def mean_interval(values: list[float], draws: int = 2000) -> list[float] | None:
    """The 90% bootstrap interval of the mean over the worlds."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) < 20:
        return None
    means = v[np.random.default_rng(0).integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
    return [round(float(np.quantile(means, 0.05)), 4), round(float(np.quantile(means, 0.95)), 4)]


def judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    zs = [r["z_mid"] for r in rows if r.get("z_mid") is not None]
    skew = [r["z_mid"] - r["z_mean"] for r in rows if r.get("z_mid") is not None and r.get("z_mean") is not None]
    return {"worlds": len(rows), "read": len(zs), "mean_z_mid": round(float(np.mean(zs)), 4) if zs else None,
            "mean_z_mid_interval": mean_interval(zs), "mean_z_mean_interval": mean_interval([r.get("z_mean") for r in rows]),
            "skew_interval": mean_interval(skew),
            "rms_z_mid": round(float(np.sqrt(np.mean(np.square(zs)))), 4) if zs else None}


def centre_gap(latest: list[dict[str, Any]], every: list[dict[str, Any]]) -> list[float] | None:
    """The 90% interval of the paired gap of the samples' means in value units, latest minus all, over the worlds (the
    review: the samples' own mean, not the midpoints, which carry each arm's skew)."""
    a = {r["world"]: r["samples_mean"] for r in latest if r.get("samples_mean") is not None}
    b = {r["world"]: r["samples_mean"] for r in every if r.get("samples_mean") is not None}
    return mean_interval([a[w] - b[w] for w in sorted(set(a) & set(b))])


def verdict(latest: dict[str, Any], gap: list[float] | None) -> str:
    """One decision's verdict, in the gate's order."""
    bias = latest.get("mean_z_mid_interval")
    if bias is None:
        return "not read"
    if -BAND <= bias[0] and bias[1] <= BAND:
        return f"closed: no bias beyond {BAND}"
    if not (bias[0] > 0 or bias[1] < 0):
        return "inconclusive: more worlds"
    by_mean, skew = latest.get("mean_z_mean_interval"), latest.get("skew_interval")
    up = bias[0] > 0
    skew_on_side = skew is not None and (skew[0] > 0 if up else skew[1] < 0)
    if skew_on_side and by_mean is not None and by_mean[0] <= 0 <= by_mean[1]:
        return "the midpoint (the samples' skew): centre the interval on the samples' mean"
    if up and gap is not None and gap[1] < 0:  # the pool can only put the samples low, the realised value above
        return "the small pool: draw each sample from the latest fold's own rows"
    return "read the worlds"


def report() -> dict[str, Any]:
    folder = CAMPAIGN / "results"
    files = [json.loads(p.read_text()) for p in sorted(folder.glob("DCE-0*_*.json"))] if folder.exists() else []
    worlds = {arm: [r for f in files if f.get("arm") == arm for r in f.get("worlds", [])] for arm in ARMS}
    arms = {arm: judge(worlds[arm]) for arm in ARMS}
    complete = all(v.get("worlds") == len(WORLDS) for v in arms.values())
    gaps = {d: centre_gap(worlds[latest], worlds[every]) for d, (latest, every) in PAIRS.items()} if complete else {}
    verdicts = {d: verdict(arms[latest], gaps.get(d)) for d, (latest, _) in PAIRS.items()} if complete else {}
    out = {"arms": arms, "centre_gap_latest_minus_all": gaps, "verdicts": verdicts, "verdict": "complete" if complete else "not yet", "gate": GATE}
    if complete and not (folder / "DCE-100_decision.json").exists():  # written once
        (folder / "DCE-100_decision.json").write_text(json.dumps({**out, "id": "DCE-100", "experiment_id": "DCE-100", "status": "completed",
                                                                  "completed_at": _now(), "campaign_id": CAMPAIGN_ID}, indent=2) + "\n")
    return out


def run_arm(arm: str) -> dict[str, Any]:
    rows = [world(seed, arm) for seed in WORLDS]
    number = f"DCE-{list(ARMS).index(arm) + 1:03d}"
    result = {"id": number, "experiment_id": number, "status": "completed", "completed_at": _now(), "campaign_id": CAMPAIGN_ID,
              "arm": arm, "settings": ARMS[arm], "summary": judge(rows), "worlds": rows, "code_sha256": CODE_AT_START}
    folder = CAMPAIGN / "results"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{number}_{arm}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="34.1: why the realised value sits above the decision interval's centre")
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
    done = {json.loads(p.read_text())["arm"] for p in (CAMPAIGN / "results").glob("DCE-0*_*.json")} if (CAMPAIGN / "results").exists() else set()
    for arm in ARMS:
        if arm in done:
            continue
        print(f"[daily-centre] {arm} {_now()}", flush=True)
        r = run_arm(arm)
        print(f"[daily-centre] {arm}: {r['summary']}", flush=True)
    print(json.dumps(report(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

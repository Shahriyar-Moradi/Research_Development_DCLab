"""The screening cascade on the retail returns replica (package 30.17): ``python -m dclab_rnd.cascade [--seeds 0 1 2] [--write]``.

The returns are screened day by day as the loss-prevention team would (``dclab_core.cascade``): the late-evening rule,
the link checks the matching kind names (a return by the cashier who sold the bill, the same bill returned twice, a
near-identical bill sold shortly before), the anomaly detector on the amount, the items and the hour, a queue of the
team's capacity with a random audit sample, and every thirty days a model fitted on what the team confirmed. Recall is
measured per stage on the last ninety days, with the learning on and off; the summaries are the template's (no model is
asked). ``--write`` stores the result in ``evidence/campaigns/cascade_v1``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "cascade_v1"
LINKS = ("by_seller", "returned_twice", "near_copy")
FEATURES = ("amount", "items", "hour", "cash", *LINKS)


def links_of(returns: Any, sales: Any, minutes: int = 90, tolerance: float = 0.05) -> Any:
    """Each return with the link checks of its bill, as known on the day it is screened (the matching kind's exact links
    and its blocking rule): returned by the cashier who sold it, returned before (another return of the bill on that day
    or earlier; a later one is the future), or sold minutes after a near-identical bill (the same items, the amount
    within ``tolerance``; the bill's own sale is before its return)."""
    import numpy as np
    import pandas as pd

    out = returns.copy()
    seller = sales.set_index("bill_id")["cashier_id"]
    stamp = pd.to_datetime(out["returned_at"])
    out["hour"] = stamp.dt.hour + stamp.dt.minute / 60
    out["by_seller"] = (out["bill_id"].map(seller) == out["cashier_id"]).astype(int)
    day = stamp.dt.normalize()
    upto = out.assign(__day=day).groupby("bill_id")["__day"].transform(lambda d: d.apply(lambda x: int((d <= x).sum())))
    out["returned_twice"] = (upto > 1).astype(int)  # the review: a whole-log count flagged a first return by a later one
    s = sales.assign(t=pd.to_datetime(sales["sold_at"])).sort_values("t")
    near = set()
    for _, g in s.groupby("items"):
        t, a, b = g["t"].to_numpy(), g["amount"].to_numpy(), g["bill_id"].to_numpy()
        for i in range(len(g)):
            j = i - 1
            while j >= 0 and (t[i] - t[j]) <= np.timedelta64(minutes, "m"):
                if abs(a[i] - a[j]) <= tolerance * max(a[i], a[j]):
                    near.add(b[i])
                    break
                j -= 1
    out["near_copy"] = out["bill_id"].isin(near).astype(int)
    out["cash"] = (out["refund_method"] == "cash").astype(int)
    return out


def replica_run(seed: int, learn: bool = True, summarize: Any = None, private: bool = False, net_share: Optional[float] = None) -> dict[str, Any]:
    from dclab_core import cascade
    from dclab_core.replicas import retail_returns

    r = retail_returns.generate(seed=seed)
    sales = r.sales[[c for c in r.sales.columns if not c.startswith("_")]]
    keep_out = ["label", *r.facts["keep_out"]]  # the label, the review's verdict, a rule computed with the future: never read
    frame = links_of(r.returns.drop(columns=keep_out), sales)
    truth = dict(zip(r.truth["return_id"], r.truth["fraud"].astype(int)))  # the reviewers confirm from it, row by row; the measurement reads it after
    return cascade.run(frame, day="returned_at", key="return_id", review=truth, measure=truth, rules=["rule_flag"], anomaly=["amount", "items", "hour"],
                       links=list(LINKS), features=list(FEATURES), keep_out=keep_out, k=retail_returns.CAPACITY, audit=1,
                       holdout_days=retail_returns.DRIFT_DAYS, retrain_days=30, learn=learn, summarize=summarize, private=private,
                       net_share=cascade.NET_SHARE if net_share is None else net_share, seed=seed)


DETAIL = ("queued", "audited", "label_of", "summaries", "cases", "shown")  # every reviewed row, its label, its summary whole with why: beside the record


def run(seeds: tuple[int, ...] = (0, 1, 2), log=print) -> dict[str, Any]:
    out, detail = [], []
    for seed in seeds:
        on, off = replica_run(seed), replica_run(seed, learn=False)
        row = {"seed": seed, "stages": on["stages"], "without_learning": off["stages"], "holdout": on["holdout"], "rounds": on["rounds"],
               "labels": on["labels"], "learned": on["learned"], "trained_on": on["trained_on"], "k": on["k"], "audit": on["audit"]}
        out.append(row)
        detail += [{"seed": seed, "learning": arm, **{k: r[k] for k in DETAIL}} for arm, r in (("on", on), ("off", off))]
        log(f"seed {seed}: net {on['stages']['net']['recall']}, queue {on['stages']['queue']['recall']} (learning off {off['stages']['queue']['recall']}), "
            f"audit estimate {(on['stages']['audit']['estimated_recall'] or {}).get('recall')}, cases {on['holdout']['cases']}")
    reviewed = sorted({r["holdout"]["reviewed_share"] for r in out})
    return {"runs": out, "detail": detail, "limits": [
        "Invented data: the replica's truth stands for the reviewers, who confirm every reviewed row correctly.",
        f"The team reviews five returns of about nineteen a day: {reviewed[0]:.0%} to {reviewed[-1]:.0%} of the holdout's rows are reviewed, far more "
        "than a real team of five among thousands; the queue's recall here says the method works, not how much a real queue would catch.",
        "The audit is one row a day: its estimates are wide (their intervals are in each run).",
        "The summaries are the template's words: no model was asked."]}


def write(doc: dict[str, Any], campaign: Path = CAMPAIGN) -> Path:
    """The record (``results/CSC-<n>_retail.json``: the stages, rounds and limits) and, beside it, every run's detail whole
    (``runs/CSC-<n>_detail.jsonl.gz``: each queued row, each confirmed label, each summary), read back by ``detail``."""
    import gzip

    results, runs = campaign / "results", campaign / "runs"
    results.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("CSC-*.json")] or [0])
    ident = f"CSC-{n:03d}"
    path, kept = results / f"{ident}_retail.json", runs / f"{ident}_detail.jsonl.gz"
    if path.exists() or kept.exists():
        raise SystemExit(f"{ident} exists: results are never overwritten")
    with gzip.open(kept, "wt", encoding="utf-8") as fh:
        for row in doc.get("detail", []):
            fh.write(json.dumps(row, default=str) + "\n")
    record = {k: v for k, v in doc.items() if k != "detail"}
    path.write_text(json.dumps({"id": ident, "experiment_id": ident, "campaign_id": "cascade_v1", "kind": "screening_cascade", "status": "completed",
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record,
                                "detail": f"runs/{kept.name}",  # relative to the campaign's folder
                                "rule": "The screening cascade run day by day on the retail returns replica (three seeds), recall measured per stage on the "
                                        "last ninety days with the learning on and off: evidence about the method on invented data, not about any "
                                        "real team's returns."}, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def detail(path: Path) -> list[dict[str, Any]]:
    """A record's runs read back whole (the record's ``detail`` path, or the file itself)."""
    import gzip

    p = Path(path)
    if p.suffix == ".json":  # the record: its detail sits in the campaign's folder
        p = p.parent.parent / json.loads(p.read_text(encoding="utf-8"))["detail"]
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.cascade", description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    doc = run(tuple(args.seeds))
    if args.write:
        print(write(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

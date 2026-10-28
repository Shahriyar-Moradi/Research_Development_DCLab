"""35.8 v2: does memory help, measured on the intern's own mistakes, with the gate sized before it is frozen.

v1 (memory_value.py, MV-001) chose its treated cases from another model's mistakes: the intern's model went wrong on
only 3 of the 8 without memory, so its sign-flip test could never reach p < 0.05 whatever memory did. v2 fixes the
design in three steps, every rule written in manifest.json before the first step runs:

1. pilot: every candidate in the pool runs with memory off, a few times. Nothing here enters the verdict; it only
   shows which columns this model, on its own, judges against what a person would correct.
2. freeze: the candidates it got wrong often enough become the treated cases; a simulation from the pilot's rates
   (shrunk toward the middle) and an assumed rate with memory on sizes the repeats; if no allowed repeat count gives
   the power the manifest asks, nothing more runs. The choice is written once to frozen.json.
3. run: the treated and control cases, both arms, fresh sessions, paired and alternating as in v1; the verdict uses
   v1's paired judge, with the leak guard read on every case that holds a planted leak.

    python -m dclab_rnd.agent_eval.memory_value_v2                     # the plan; nothing sent
    python -m dclab_rnd.agent_eval.memory_value_v2 pilot --yes --cap-eur 1
    python -m dclab_rnd.agent_eval.memory_value_v2 freeze
    python -m dclab_rnd.agent_eval.memory_value_v2 run --yes --cap-eur 1
    python -m dclab_rnd.agent_eval.memory_value_v2 summarize
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import memory_value as v1
from .cases import BASE_INPUTS, CASES, Case, _base, _flip

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "memory_value_v2"
DIR = ROOT / "evidence" / "campaigns" / CAMPAIGN
MANIFEST = DIR / "manifest.json"
FROZEN = DIR / "frozen.json"
RESULTS = DIR / "results"
RUNS = ROOT / "agent_runs" / "live_tests" / CAMPAIGN
PREFIX = "MV2"


# ------------------------------------------------------------------------------------------------ the candidate cases
def _sig(x):
    return 1 / (1 + np.exp(-x))


def _after_hours(rng):
    f = _base(rng)
    f["calls_after_hours"] = rng.binomial(f["support_calls"], 0.3)  # a part of the support calls, all before the snapshot
    return f


def _cancel_fee(rng):
    f = _base(rng)
    f["cancellation_fee_eur"] = np.where(f["plan"] == "basic", 0, np.where(f["plan"] == "plus", 50, 100)) + rng.integers(0, 3, len(f)) * 10
    return f


def _churn_band(rng):
    f = _base(rng)
    f["churn_risk_band"] = np.where((f["plan"] == "basic") & (f["age"] < 35), "high", np.where(f["plan"] == "pro", "low", "mid"))
    return f


def _final_offer(rng):
    f = _base(rng)
    f["final_offer_discount"] = rng.choice([0, 5, 10, 15], len(f))
    return f


def _exit_waived(rng):
    f = _base(rng)
    f["exit_fee_waived"] = (rng.random(len(f)) < 0.3).astype(int)
    return f


def _retention(rng):
    f = _base(rng)
    f["retention_offer_eligible"] = ((f["plan"] != "basic") & (f["age"] > 40)).astype(int)
    return f


def _prev_term(rng):
    f = _base(rng)
    n = len(f)
    prev = np.where(f["tenure_months"] < 12, "first_term", np.where(rng.random(n) < 0.7, "renewed", "lapsed"))
    f["previous_term_outcome"] = prev
    f["target"] = np.where(prev == "lapsed", (rng.random(n) < 0.6).astype(int), f["target"])  # a past term drives this one
    return f


def _constants(rng):
    f = _base(rng)
    f["region"] = "EU-West"
    f["data_feed"] = "crm_v2"
    return f


def _closing_prev(rng):
    f = _base(rng)
    f["closing_balance_prev_month"] = np.round(f["monthly_spend"] * rng.uniform(0.8, 1.2, len(f)), 2)
    return f


def _first_week_and_leak(rng):
    f = _base(rng)
    f["usage_first_week_after_signup"] = rng.poisson(5, len(f))
    f["final_balance_after"] = np.round(f["monthly_spend"] * (1 - 0.9 * f["target"]) + rng.normal(0, 3, len(f)), 2)  # the planted leak
    return f


def _renewal_and_leak(rng):
    f = _base(rng)
    d = np.round(rng.uniform(0, 365, len(f)))
    f["days_to_renewal"] = d
    f["target"] = (rng.random(len(f)) < _sig(1.8 - 0.02 * d)).astype(int)  # a strong, honest driver
    f["account_status_closed"] = _flip(rng, f["target"], 0.08)  # the planted leak
    return f


def _cancel_requests(rng):
    f = _base(rng)
    f["cancellations_requested_last_year"] = rng.poisson(0.3 + 0.5 * (f["plan"] == "basic"), len(f))
    return f


def _onboarding(rng):
    f = _base(rng)
    f["onboarding_outcome_score"] = rng.integers(1, 6, len(f))
    return f


POOL_CASES: tuple[Case, ...] = (
    Case("MEM-01", "control", "benign_post_name", "Support calls outside office hours", _after_hours, 101, innocent=("calls_after_hours",)),
    Case("MEM-02", "control", "benign_post_name", "A cancellation fee fixed in the contract", _cancel_fee, 102, innocent=("cancellation_fee_eur",)),
    Case("MEM-03", "control", "benign_post_name", "A risk band set at signup by a rule", _churn_band, 103, innocent=("churn_risk_band",)),
    Case("MEM-04", "control", "benign_post_name", "The discount of the last offer before signup", _final_offer, 104, innocent=("final_offer_discount",)),
    Case("MEM-05", "control", "benign_post_name", "An exit fee waived by a signup promotion", _exit_waived, 105, innocent=("exit_fee_waived",)),
    Case("MEM-06", "control", "benign_post_name", "Eligibility for an offer, set at signup", _retention, 106, innocent=("retention_offer_eligible",)),
    Case("MEM-07", "control", "strong_honest_driver", "How the previous contract term ended", _prev_term, 107, innocent=("previous_term_outcome",)),
    Case("MEM-08", "control", "constant_column", "Two columns with one value each", _constants, 108, innocent=("region", "data_feed")),
    Case("MEM-09", "control", "benign_post_name", "The balance at the end of the previous month", _closing_prev, 109, innocent=("closing_balance_prev_month",)),
    Case("MEM-10", "leakage", "post_outcome_name", "First-week use beside a balance written after the outcome", _first_week_and_leak, 110,
         leaks=("final_balance_after",), innocent=("usage_first_week_after_signup",)),
    Case("MEM-11", "leakage", "post_outcome_name", "A strong honest driver beside the outcome's own status", _renewal_and_leak, 111,
         leaks=("account_status_closed",), innocent=("days_to_renewal",)),
    Case("MEM-12", "control", "benign_post_name", "Cancellation requests in the year before", _cancel_requests, 112, innocent=("cancellations_requested_last_year",)),
    Case("MEM-13", "control", "benign_post_name", "The score of the first-week onboarding call", _onboarding, 113, innocent=("onboarding_outcome_score",)),
)


def case_of(case_id: str) -> Case:
    return next(c for c in (*POOL_CASES, *CASES) if c.id == case_id)


# ------------------------------------------------------------------------------------------------ the plan
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest_sha() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def scenario_of(case_id: str, m: dict[str, Any] | None = None) -> dict[str, Any]:
    m = m or manifest()
    return next(s for s in (*m["pool"], *m["controls"]) if s["case"] == case_id)


def pilot_schedule(m: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    m = m or manifest()
    # P- ids: a pilot row can never pass for a main-run row, whose off arm must be fresh sessions
    return [{"id": f"P-{s['case']}-r{r}-off", "case": s["case"], "repeat": r, "arm": "off", "role": "candidate"}
            for r in range(m["pilot"]["repeats"]) for s in m["pool"]]


def frozen() -> dict[str, Any] | None:
    return json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else None


def frozen_sha() -> str:
    return hashlib.sha256(FROZEN.read_bytes()).hexdigest()


def main_scenarios(f: dict[str, Any], m: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    m = m or manifest()
    return [{**scenario_of(c, m), "role": "treated"} for c in f["treated"]] + [{**s, "role": "control"} for s in m["controls"]]


def main_schedule(f: dict[str, Any], m: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every planned session of the main run: repeat by repeat, case by case, the two arms alternating which goes first."""
    out = []
    for r in range(f["repeats"]):
        for n, s in enumerate(main_scenarios(f, m)):
            arms = ("on", "off") if (r + n) % 2 == 0 else ("off", "on")
            out += [{"id": f"{s['case']}-r{r}-{arm}", "case": s["case"], "repeat": r, "arm": arm, "role": s["role"]} for arm in arms]
    return out


def setup() -> dict[str, Any]:
    out = v1.setup()
    out["manifest_sha256"] = manifest_sha()
    out["code"] = {**out["code"], **{n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()[:16]
                                     for n in ("dclab_rnd/agent_eval/memory_value.py", "dclab_rnd/agent_eval/memory_value_v2.py")}}
    return out


def session(gateway: Any, item: dict[str, Any], budget_eur: float | None, max_steps: int, home: Path) -> dict[str, Any]:
    m = manifest()
    s = {**scenario_of(item["case"], m), "role": item.get("role", "candidate")}  # the stored row names its true role
    row = v1.one_session(gateway, item, budget_eur, max_steps, home, scenario=s, case=case_of(item["case"]), m=m)
    return {**row, "campaign": CAMPAIGN}


# ------------------------------------------------------------------------------------------------ choose and size
def select(rows: dict[str, dict[str, Any]], m: dict[str, Any] | None = None) -> dict[str, Any]:
    """The pilot's tally per candidate and, by the manifest's rule, the treated cases (the most often wrong first)."""
    m = m or manifest()
    rule = m["selection"]
    tally = {}
    for n, s in enumerate(m["pool"]):
        rs = [r for r in rows.values() if r["case"] == s["case"] and r.get("ran")]
        wrong = sum(1 for r in rs if set(r.get("against") or []) & set(s["columns"]))  # the corrected column, not any column
        tally[s["case"]] = {"scored": len(rs), "against": wrong, "share": round(wrong / len(rs), 4) if rs else None, "order": n}
    num, den = rule["min_against_fraction"]  # whole counts: 2 of 3 is two thirds, never a rounded share
    ok = [c for c, t in tally.items() if t["scored"] >= rule["min_scored"] and t["against"] * den >= num * t["scored"]]
    ok.sort(key=lambda c: (-tally[c]["share"], tally[c]["order"]))
    return {"tally": tally, "treated": ok[: rule["max_treated"]]}


def _sign_flip_p_many(means: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """v1.sign_flip_p for many draws at once: means is (draws, k), signs every pattern (2^k, k)."""
    observed = means.mean(axis=1)
    flipped = means @ signs.T / means.shape[1]  # (draws, 2^k)
    return (flipped <= observed[:, None] + 1e-12).mean(axis=1)


def power(off_rates: list[float], on_rate: float, repeats: int, gate: dict[str, Any], draws: int, seed: int) -> float:
    """The share of simulated main runs whose treated cases pass the primary check: the mean paired difference at most
    the gate's bound and the exact sign-flip p below its bound."""
    import itertools

    k = len(off_rates)
    if not k:
        return 0.0
    rng = np.random.default_rng(seed)
    p_off = np.array(off_rates)[None, :, None]
    on = rng.random((draws, k, repeats)) < on_rate
    off = rng.random((draws, k, repeats)) < p_off
    means = (on.astype(float) - off.astype(float)).mean(axis=2)
    signs = np.array(list(itertools.product((1, -1), repeat=k)), dtype=float)
    p = _sign_flip_p_many(means, signs)
    ok = (means.mean(axis=1) <= gate["against_mean_at_most"]) & (p < gate["sign_flip_p_below"])
    return round(float(ok.mean()), 4)


def size(choice: dict[str, Any], m: dict[str, Any] | None = None) -> dict[str, Any]:
    """The manifest's sizing rule on the pilot's choice: the fewest allowed repeats whose power reaches the bar, or none."""
    m = m or manifest()
    rule = m["sizing"]
    treated = choice["treated"]
    t = choice["tally"]
    scored = sum(x["scored"] for x in t.values())
    mean = sum(x["against"] for x in t.values()) / scored if scored else 0.5  # the whole pool's pilot rate
    k = rule["prior_strength"]  # chosen for being wrong, a case looks worse than it is: pull it toward the pool
    rates = [(t[c]["against"] + k * mean) / (t[c]["scored"] + k) for c in treated]
    by_repeats = {r: power(rates, rule["assumed_on_rate"], r, m["gate"], rule["draws"], rule["seed"]) for r in rule["repeats_options"]}
    sensitivity = {r: power(rates, rule["sensitivity_on_rate"], r, m["gate"], rule["draws"], rule["seed"]) for r in rule["repeats_options"]}
    enough = [r for r in rule["repeats_options"] if by_repeats[r] >= rule["min_power"]]
    why = None
    if len(treated) < rule["min_cases"]:
        why = f"only {len(treated)} candidates were wrong often enough in the pilot; the sign-flip test needs at least {rule['min_cases']}"
    elif not enough:
        why = f"no allowed repeat count reaches power {rule['min_power']}"
    return {"off_rates": dict(zip(treated, [round(x, 4) for x in rates])), "power_by_repeats": by_repeats,
            "power_if_memory_on_rate_is": {"rate": rule["sensitivity_on_rate"], "by_repeats": sensitivity},
            "repeats": None if why else enough[0], "runnable": why is None, "why_not": why}


def _pilots(where: Path | None = None) -> set[Path]:
    """Every folder here that a pilot ran in, whatever its name: its run.json says so and it holds sessions."""
    where = where or RUNS
    found = set()
    for meta in where.glob("*/run.json") if where.exists() else []:
        try:
            ok = json.loads(meta.read_text(encoding="utf-8")).get("campaign") == f"{CAMPAIGN}:pilot"
        except (OSError, json.JSONDecodeError):
            continue
        if ok and any(meta.parent.glob("P-*.json")):
            found.add(meta.parent)
    return found


def freeze(pilot: Path, at: str | None = None) -> dict[str, Any]:
    """Write frozen.json once: the pilot's tally, the treated cases, the repeats and the power. Never overwritten."""
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN.name} exists: the main run's plan is frozen once")
    m = manifest()
    todo = v1.pending_of(pilot, pilot_schedule(m), m["gate"]["max_attempts"])
    if todo:
        raise SystemExit(f"the pilot has {len(todo)} sessions still to run: nothing was frozen")
    meta = json.loads((pilot / "run.json").read_text(encoding="utf-8"))
    if meta.get("campaign") != f"{CAMPAIGN}:pilot" or meta.get("manifest_sha256") != manifest_sha():
        raise SystemExit("the pilot ran under another campaign or manifest: the rules are fixed before the pilot; nothing was frozen")
    others = sorted(str(p) for p in _pilots() | _pilots(pilot.parent) if p.resolve() != pilot.resolve())
    if others:
        raise SystemExit(f"other pilots hold sessions ({', '.join(others)}): one pilot only, never the best of several; nothing was frozen")
    rows = v1.done(pilot)
    choice = select(rows, m)
    sized = size(choice, m)
    doc = {"campaign": CAMPAIGN, "frozen_at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"), "manifest_sha256": manifest_sha(),
           "pilot_run": pilot.name, "pilot_sessions": len(rows), "pilot_eur": round(sum(float(r.get("eur") or 0) for r in rows.values()), 4),
           "pilot_setup": {k: meta.get(k) for k in ("model", "thinking", "prompts", "code")},
           "tally": choice["tally"], "treated": choice["treated"], **sized}
    FROZEN.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def _checked_frozen() -> dict[str, Any]:
    f = frozen()
    if not f:
        raise SystemExit("frozen.json is missing: run the pilot and freeze first; nothing was sent")
    if f["manifest_sha256"] != manifest_sha():
        raise SystemExit("the manifest changed after the plan was frozen: one verdict never mixes two; nothing was sent")
    if not f["runnable"]:
        raise SystemExit(f"the frozen plan cannot reach its gate ({f['why_not']}): the main run is not started; nothing was sent")
    return f


# ------------------------------------------------------------------------------------------------ run and judge
def folder_of(kind: str, run: str | None) -> Path:
    if run:
        return Path(run) if Path(run).is_absolute() else RUNS / run
    tag = "PILOT" if kind == "pilot" else PREFIX
    started = sorted(p for p in RUNS.glob(f"{tag}-*") if p.is_dir()) if RUNS.exists() else []
    return started[-1] if started else RUNS / f"{tag}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"


def run_pilot(gateway: Any, folder: Path, cap_eur: float | None, max_requests: int, max_steps: int, log=print) -> dict[str, Any]:
    m = manifest()
    items = pilot_schedule(m)
    return v1.run_items(gateway, folder, lambda: v1.pending_of(folder, items, m["gate"]["max_attempts"]), lambda *a: session(*a),
                        campaign=f"{CAMPAIGN}:pilot", now=setup(), estimate=m["estimate_eur_per_session"], cap_eur=cap_eur,
                        max_requests=max_requests, max_steps=max_steps, log=log)


def run_main(gateway: Any, folder: Path, cap_eur: float | None, max_requests: int, max_steps: int, log=print) -> dict[str, Any]:
    m = manifest()
    f = _checked_frozen()
    now = {**setup(), "model": getattr(gateway.client("intern"), "model", None)}
    then = f.get("pilot_setup") or {}
    moved = sorted(k for k in ("model", "thinking", "prompts") if then.get(k) != now[k])
    moved += sorted(n for n in v1.setup()["code"] if (then.get("code") or {}).get(n) != now["code"].get(n))  # the intern and its scoring
    if moved:
        raise SystemExit(f"the main run differs from the pilot ({', '.join(moved)}): the cases were chosen on that setup's mistakes; nothing was sent")
    items = main_schedule(f, m)
    return v1.run_items(gateway, folder, lambda: v1.pending_of(folder, items, m["gate"]["max_attempts"]), lambda *a: session(*a),
                        campaign=CAMPAIGN, now={**setup(), "frozen_sha256": frozen_sha()},
                        estimate=m["estimate_eur_per_session"], cap_eur=cap_eur, max_requests=max_requests, max_steps=max_steps, log=log)


def summarize(folder: Path) -> dict[str, Any]:
    m = manifest()
    f = _checked_frozen()
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if (meta.get("campaign"), meta.get("frozen_sha256"), meta.get("manifest_sha256")) != (CAMPAIGN, frozen_sha(), manifest_sha()):
        raise SystemExit("this run was not made under the frozen plan as it is now: the verdict reads only the plan the run used")
    items = main_schedule(f, m)
    role = {s["case"]: s["role"] for s in main_scenarios(f, m)}
    left = len(v1.pending_of(folder, items, m["gate"]["max_attempts"]))
    rows = v1.done(folder)
    out = v1.judge(rows, items, role, m["gate"], left, leaks_over="all")
    if "checks" in out:  # each treated case that holds a leak, alone: a pooled share would hide one case's loss
        by = {}
        for i in items:
            r = rows.get(i["id"]) or {}
            if role[i["case"]] == "treated" and r.get("ran") and r.get("leak_caught") is not None:
                by.setdefault(i["case"], {"on": [], "off": []})[i["arm"]].append(bool(r["leak_caught"]))
        share = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731
        out["leak_by_treated_case"] = {c: {a: share(v[a]) for a in v} for c, v in by.items()}
        out["checks"]["treated_leaks_kept_each_case"] = all(share(v["on"]) >= share(v["off"]) for v in by.values())
        out["verdict"] = "memory_helps" if all(out["checks"].values()) else "memory_does_not_help"
    return {**out, "frozen": {k: f[k] for k in ("treated", "repeats", "power_by_repeats", "pilot_run")}}


LIMITS = [v1.SCOPE,
          "the treated cases were chosen because this model judged them wrongly without memory in the pilot: the effect is "
          "memory's on the mistakes it makes, not an average over every column",
          "the earlier session is scripted through the write paths; a real person's earlier session can say less, or say it worse",
          "some candidates are near copies of suite cases (MEM-08 of AE-14, MEM-01 and MEM-10 of AE-16, MEM-11 of AE-17): "
          "the cases are fewer kinds of mistake than their count",
          "the candidate tables are invented; the sealed read waits for the owner and needs cases no run has seen",
          "questions asked again are not measured: the intern asks none in this suite, and Home does not read memory yet"]


def store(folder: Path, summary: dict[str, Any]) -> Path:
    return v1.store_result(folder, summary, results=RESULTS, campaign=CAMPAIGN, prefix=PREFIX, limitations=LIMITS,
                           extra={"frozen": frozen(), "manifest_sha256": manifest_sha()})


def main(argv: list[str] | None = None, gateway: Any = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.memory_value_v2", description=__doc__.split("\n")[0])
    parser.add_argument("action", nargs="?", choices=("plan", "pilot", "freeze", "run", "summarize"), default="plan")
    parser.add_argument("--yes", action="store_true", help="send; without it the plan is printed and nothing is sent")
    parser.add_argument("--cap-eur", type=float, help="stop before this invocation costs more (required for a priced remote model)")
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--run", help="the run folder to resume (default: the latest started of its kind)")
    args = parser.parse_args(argv)
    m = manifest()
    if args.action == "freeze":
        doc = freeze(folder_of("pilot", args.run))
        print(json.dumps({k: doc[k] for k in ("treated", "repeats", "power_by_repeats", "runnable", "why_not")}, indent=1))
        return 0
    if args.action == "summarize":
        folder = folder_of("main", args.run)
        s = summarize(folder)
        print(json.dumps({k: v for k, v in s.items() if k != "by_case"}, indent=1))
        if s["verdict"] != "incomplete":
            print(f"Stored {store(folder, s)}")
        return 0
    f = frozen()
    print(f"Pilot: {len(m['pool'])} candidates × {m['pilot']['repeats']} repeats, memory off = {len(pilot_schedule(m))} sessions.")
    if f:
        print(f"Frozen: {len(f['treated'])} treated + {len(m['controls'])} controls × {f['repeats']} repeats × 2 arms = "
              f"{len(main_schedule(f, m)) if f['runnable'] else 0} sessions (runnable: {f['runnable']}).")
    print(f"About €{m['estimate_eur_per_session']} a session; each invocation stops before its cap. {v1.SCOPE}")
    if args.action == "plan":
        return 0
    if gateway is None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        from dclab_rnd.models.gateway import for_workspace

        gateway = for_workspace()
        gateway.shadows = None
    from dclab_rnd.models import prices, settings

    tier = settings.public(gateway.route("intern")[0])
    priced = prices.price_of(str(tier.get("model") or "")) is not None and not tier.get("local")
    if gateway.client("intern") is None:
        print("No model serves the intern (DCLAB_NO_LIVE_MODELS, or no key); nothing was sent.", file=sys.stderr)
        return 2
    if priced and args.cap_eur is None:
        print("The intern's model is priced: give --cap-eur; nothing was sent.", file=sys.stderr)
        return 2
    if not args.yes:
        print("Nothing was sent. Run again with --yes.", file=sys.stderr)
        return 2
    fn = run_pilot if args.action == "pilot" else run_main
    out = fn(gateway, folder_of(args.action if args.action == "pilot" else "main", args.run), args.cap_eur, args.max_requests, args.max_steps)
    print(f"Ran {len(out['ran'])} sessions · €{out['spent_eur']} · {out['requests']} requests · {out['left']} sessions left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

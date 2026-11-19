"""Reproducible synthetic price studies. Run with --out NEW_DIRECTORY; no models or network required."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dclab_core.price_scenarios import (
    StudySpec,
    effect_interval,
    fit_study,
    standardized,
    study_rows,
)

WORLDS = ("sales", "renewals", "no_effect", "confounding", "fixed", "late", "shock")
PRICES = [100, 105, 110]
METHODS = {"Flat": [], "Price": ["price"], "Price + history": ["price", "segment", "history_band"]}


def generate(world, seed):
    rng = np.random.default_rng(seed)
    n, n_train = 1600, 1200
    customer = np.arange(n)
    train = customer < n_train
    segment = rng.integers(0, 2, n)
    high = rng.integers(0, 2, n)
    arm = rng.integers(0, 3, n)
    if world == "fixed":
        arm = segment * 2
    elif world == "confounding":
        for ix in (0, 1):
            mask = segment == ix
            arm[mask] = rng.choice(3, mask.sum(), p=(0.6, 0.25, 0.15) if ix == 0 else (0.15, 0.25, 0.6))
    day = np.where(train, customer % 20, 80 + customer % 10)
    date = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(day, unit="D")
    baseline = 0.30 + 0.10 * segment + 0.18 * high
    step = np.zeros(n) if world == "no_effect" else 0.02 + 0.05 * high
    if world == "renewals":
        baseline, step = 1 - baseline, -step
    truth = baseline[:, None] - step[:, None] * np.arange(3)
    shock = rng.choice([-0.12, 0.12], 90)[day] if world == "shock" else 0
    y = (rng.random(n) < truth[customer, arm] + shock).astype(int)
    offers = pd.DataFrame(
        {
            "offer_id": customer,
            "customer_id": customer,
            "offered_at": date,
            "price": np.array(PRICES)[arm],
            "currency": "USD",
            "segment": segment,
            "split": np.where(train, "train", "score"),
        }
    )
    person = np.repeat(customer, 5)
    position = np.tile(np.arange(5), n)
    occurred = date[person] - pd.to_timedelta(6 - position, unit="D")
    available = occurred + pd.to_timedelta((world == "late") * (position == 4) * 10, unit="D")
    offers["post_offer_confirmation"] = y  # deliberately unavailable at offer time; synthetic diagnostic only
    events = pd.DataFrame(
        {
            "customer_id": person,
            "event_id": np.arange(len(person)),
            "occurred_at": occurred,
            "available_at": available,
            "discounted": (position < 1 + 3 * high[person]).astype(int),
        }
    )
    if world == "late":
        events = events[(events.customer_id % 13 != 0) | (position == 0)].copy()
    label_delay = 30 + (world == "late") * (customer % 8 == 0) * 100
    outcomes = pd.DataFrame(
        {"offer_id": customer, "available_at": date + pd.to_timedelta(label_delay, unit="D"), "outcome": y}
    )
    return offers, events, outcomes, truth


def spec_for(world):
    design = "observational" if world in ("fixed", "confounding") else "randomized"
    return StudySpec(
        base_price=100,
        horizon_days=30,
        outcome="churn" if world == "renewals" else "purchase",
        design=design,
        design_basis="Known synthetic assignment by segment"
        if design == "observational"
        else "Generator assigned each person independently to an offer price",
    )


def evaluate(world, seed, folder=None, prediction_file=None):
    offers, events, outcomes, oracle = generate(world, seed)
    spec = spec_for(world)
    if folder:
        folder.mkdir(parents=True)
        for name, frame in zip(("offers", "events", "outcomes"), (offers, events, outcomes)):
            path = folder / ("price_" + name + ".csv")
            frame.to_csv(path, index=False)
        offers, events, outcomes = (
            pd.read_csv(folder / ("price_" + name + ".csv")) for name in ("offers", "events", "outcomes")
        )
    train, cohort = study_rows(offers, events, outcomes, 30)
    n = len(cohort)
    y = outcomes.set_index("offer_id").loc[cohort.offer_id, "outcome"].to_numpy()
    # Align by explicit offer ID for both factual and oracle scoring.
    truth = oracle[cohort.offer_id.to_numpy()]
    observed = np.searchsorted(PRICES, cohort.price)
    summaries, predictions = [], []
    for method, fields in METHODS.items():
        if fields:
            cells = train.groupby(fields).outcome.agg(["mean", "size"])
            columns = []
            for price in PRICES:
                keys = cohort.assign(price=price)[fields]
                keys = pd.MultiIndex.from_frame(keys) if len(fields) > 1 else keys.price
                columns.append(cells["mean"].where(cells["size"] >= 10).reindex(keys).to_numpy())
            p = np.column_stack(columns)
        else:
            p = np.full((n, 3), train.outcome.mean())
        seen = p[np.arange(n), observed]
        complete = np.isfinite(p).all(axis=1)
        factual = np.isfinite(seen)
        scores = p.mean(axis=0) if complete.all() else [None] * 3
        estimates = dict(zip(("p100", "p105", "p110"), scores))
        delta = None if not complete.all() else 100 * (scores[2] - scores[0])
        summaries.append(
            dict(
                world=world,
                seed=seed,
                method=method,
                audience=n,
                supported=int(complete.sum()),
                factual=int(factual.sum()),
                brier=np.mean((seen[factual] - y[factual]) ** 2) if factual.any() else None,
                effect_pp=delta,
                truth_pp=100 * np.mean(truth[:, 2] - truth[:, 0]),
                **estimates,
            )
        )
        if folder or prediction_file:
            for j, price in enumerate(PRICES):
                predictions.append(
                    pd.DataFrame(
                        {
                            "offer_id": cohort.offer_id,
                            "method": method,
                            "price": price,
                            "seed": seed,
                            "prediction": p[:, j],
                            "source": "safe forecast",
                        }
                    )
                )
    # Isolated positive control for detecting post-outcome information; no fitted or deployable artifact.
    diagnostic = (
        offers.set_index("offer_id").loc[cohort.offer_id, "post_offer_confirmation"].to_numpy(dtype=float)
    )
    summaries.append(
        {
            "world": world,
            "seed": seed,
            "method": "Future label diagnostic",
            "audience": n,
            "supported": 0,
            "factual": n,
            "brier": np.mean((diagnostic - y) ** 2),
            "effect_pp": None,
            "truth_pp": None,
        }
    )
    if prediction_file:
        pd.concat(predictions).to_csv(prediction_file, index=False, header=(seed == 19), float_format="%.8g")
    if folder:
        pd.concat(predictions).to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame(truth, index=cohort.offer_id, columns=PRICES).to_csv(folder / "evaluator_truth.csv")
    report = fit_study(offers, events, outcomes, spec)
    effect = report.get("effects", [{}, {}])[1] or {}
    true_delta = 100 * np.mean(truth[:, 2] - truth[:, 0])
    interval = effect.get("interval95_pp")
    if world == "fixed" and interval:
        raise AssertionError("Fixed-by-segment prices produced an effect claim.")
    if world in ("sales", "renewals", "no_effect") and not interval:
        raise AssertionError("Randomized fixture did not produce the planned comparison.")
    row = {
        "world": world,
        "seed": seed,
        "source": report.get("effect_source", report["source"]),
        "effect_pp": effect.get("change_pp"),
        "truth_pp": true_delta,
        "audience": n,
        "mature_train": report["mature_training"],
        "covers_truth": None if not interval else interval[0] <= true_delta <= interval[1],
        "lo": None if not interval else interval[0],
        "hi": None if not interval else interval[1],
    }
    # Prespecified sensitivity contrast: change at 110 vs 100, high vs low prior discount usage.
    contrasts = []
    for band in ("high", "low"):
        g = cohort[cohort.history_band == band]
        s = standardized(train[train.history_band == band], g, PRICES, ["segment"]) if len(g) else None
        if not s:
            break
        counts, rates, sizes = s
        contrasts.append(effect_interval(counts, rates[0], rates[2], sizes[0], sizes[2]))
    if len(contrasts) == 2:
        diff = contrasts[0]["change_pp"] - contrasts[1]["change_pp"]
        width = np.hypot(*[(r["interval95_pp"][1] - r["interval95_pp"][0]) / 2 for r in contrasts])
        row.update(segment_contrast_pp=diff, segment_lo=diff - width, segment_hi=diff + width)
    if world == "shock":
        rng = np.random.default_rng(seed + 500_000)
        rates = truth[:, 2]
        # One common shock per deployment window in this deliberately stressed risk comparison.
        observed_counts = [(rng.random(n) < rates + rng.choice([-0.12, 0.12])).sum() for _ in range(300)]
        threshold = round(0.9 * rates.sum())
        independent_counts = [(rng.random(n) < rates).sum() for _ in range(300)]
        row.update(
            risk_shared=float(np.mean(np.array(observed_counts) < threshold)),
            risk_independent=float(np.mean(np.array(independent_counts) < threshold)),
            risk_target_revenue=threshold * 110,
        )
    return summaries, row


def run(out, seeds):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory.")
    out.mkdir(parents=True, exist_ok=True)
    results, effects = [], []
    for world in WORLDS:
        with (
            (out / (world + "_predictions.csv.gz")).open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream,
            io.TextIOWrapper(stream, encoding="utf-8", newline="") as predictions,
        ):
            for seed in range(19, 19 + seeds):
                rows, effect = evaluate(world, seed, out / world if seed == 19 else None, predictions)
                results.extend(rows)
                effects.append(effect)
    metrics = pd.DataFrame(results)
    metrics["absolute_effect_error_pp"] = (metrics.effect_pp - metrics.truth_pp).abs()
    metrics["split"] = "future"
    metrics["status"] = np.select(
        [metrics.method == "Future label diagnostic", metrics.supported < metrics.audience],
        ["diagnostic only", "insufficient common support"],
        default="scored",
    )
    for price in PRICES:
        events = metrics["p" + str(price)] * metrics.audience
        metrics[f"expected_events_{price}"] = events
        sold = events.where(metrics.world != "renewals", metrics.audience - events)
        metrics[f"expected_revenue_{price}"] = sold * price
    metrics.to_csv(out / "results.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "effects.csv", index=False)
    lines = [
        "# Price scenario simulation results",
        "",
        f"{seeds} paired seeds per world. Each seed uses 1,200 earlier and 400 later customers. Lower Brier is better.",
        "The interval is a paired bootstrap over independent generated worlds. Coverage counts scoreable people at all three prices.",
        "",
        "| World | Method | Coverage | Brier | Brier gain vs flat (95% interval) | Estimated change, pp |",
        "|---|---|---:|---:|---|---:|",
    ]
    for (world, method), group in metrics.groupby(["world", "method"], sort=False):
        baseline = metrics[(metrics.world == world) & (metrics.method == "Flat")].set_index("seed")
        gains = (baseline.brier - group.set_index("seed").brier).to_numpy()
        eligible = method != "Future label diagnostic" and (group.factual == group.audience).all()
        gain_text = "diagnostic or differing coverage"
        if eligible:
            rng = np.random.default_rng(119)
            interval = np.quantile(rng.choice(gains, (2000, len(gains))).mean(axis=1), [0.025, 0.975])
            gain_text = f"{gains.mean():.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"
        delta = group.effect_pp.mean()
        shown = "unavailable" if pd.isna(delta) else f"{delta:.2f}"
        lines.append(
            f"| {world} | {method} | {group.supported.sum()}/{group.audience.sum()} | {group.brier.mean():.4f} | {gain_text} | {shown} |"
        )
    lines += [
        "",
        "## Effect estimates",
        "",
        "| World | Estimates / seeds | Mean estimate, pp | Mean truth, pp | Intervals covering truth |",
        "|---|---:|---:|---:|---:|",
    ]
    for world, group in pd.DataFrame(effects).groupby("world", sort=False):
        valid = group.dropna(subset=["effect_pp"])
        mean = f"{valid.effect_pp.mean():.2f}" if len(valid) else "withheld"
        lines.append(
            f"| {world} | {len(valid)}/{len(group)} | {mean} | {group.truth_pp.mean():.2f} | {sum(int(x) for x in valid.covers_truth)}/{len(valid)} |"
        )
    detail = pd.DataFrame(effects)
    shock = detail[detail.world == "shock"]
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"Under a shared demand shock, the mean frequency of revenue missing the target was {shock.risk_shared.mean():.1%}. Independent resampling gave {shock.risk_independent.mean():.1%} for those same targets.",
            "The risk target was 90% of expected unit sales at 110, rounded to a whole unit and converted to revenue. Each estimate uses 300 futures per seed.",
            "",
            "The future-label control reads the planted confirmation on invented development rows. Its zero Brier score exposes leakage. It has no deployable model or counterfactual predictions.",
            "",
            "Per-seed effect intervals, subgroup contrasts and shock results are in effects.csv. Replicated seeds measure simulation uncertainty. Interval coverage is descriptive with only 30 worlds.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    entries = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            entries[str(path.relative_to(out))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    (out / "manifest.json").write_text(
        json.dumps({"version": 1, "first_seed": 19, "seeds": seeds, "files": entries}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    run(args.out, args.seeds)

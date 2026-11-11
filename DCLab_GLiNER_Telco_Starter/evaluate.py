"""Score one GLiNER checkpoint alongside both baselines; open TEST only once explicitly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd
from common import FEATURES, TASK, LABELS, read_data, metrics, dump


def normalize(result):
    item = (result or {}).get(TASK)
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        return item.get("label"), item.get("confidence")
    return None, None


def gliner_predictions(model, items):
    labels, proxy_probs, examples = [], [], []
    for row in items:
        ans = model.classify_text(row["input"], {TASK: LABELS}, include_confidence=True)
        label, confidence = normalize(ans)
        if label not in LABELS:
            raise RuntimeError(f"Unexpected GLiNER output: {ans!r}")
        labels.append(1 if label == "churn" else 0)
        if isinstance(confidence, (int, float)) and 0 <= float(confidence) <= 1:
            proxy_probs.append(float(confidence) if label == "churn" else 1 - float(confidence))
        else:
            proxy_probs.append(None)
        examples.append({"label": label, "confidence_of_selected_label": confidence})
    return labels, proxy_probs, examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    p.add_argument("--processed", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/processed"))
    p.add_argument("--baselines", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/runs/baselines"))
    p.add_argument("--model", default="research/churn-prediction/experiments/gliner_decide/runs/gliner/final")
    p.add_argument("--out", type=Path, default=Path("research/churn-prediction/experiments/gliner_decide/runs/evaluation"))
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("Refusing sealed test without --final. Use dev until the model and settings are frozen.")
    if args.final and args.split != "test":
        raise SystemExit("--final only applies to --split test")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / "SEALED_TEST_OPENED.json"
    if args.split == "test" and lock.exists():
        raise SystemExit("Sealed test already opened. Do not tune on it; use a fresh independent dataset later.")

    # Load models before the one-time test marker, so installation errors do not consume the test.
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(args.model)
    base = {name: joblib.load(args.baselines / f"{name}.joblib") for name in ("logistic", "histboost")}
    if args.split == "test":
        dump(lock, {"opened_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
                    "purpose": "One final, same-split comparison"})
    data = read_data(args.csv)
    split_map = pd.read_csv(args.processed / "splits.csv").sort_values("source_index").reset_index(drop=True)
    chosen = split_map["split"] == args.split
    X = data.loc[chosen, FEATURES]
    y = (data.loc[chosen, "Churn"] == "Yes").astype(int).tolist()
    scores = {}
    for name, pipeline in base.items():
        probs = pipeline.predict_proba(X)[:, 1].tolist()
        scores[name] = metrics(y, probs)

    with (args.processed / f"{args.split}.jsonl").open(encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    if len(entries) != len(y):
        raise RuntimeError("Mismatched target/test example count")
    g_labels, g_probs, raw = gliner_predictions(model, entries)
    if all(prob is not None for prob in g_probs):
        scores["gliner"] = metrics(y, g_probs, g_labels)
        scores["gliner"]["probability_note"] = "Proxy from winning-class score; NOT independently calibrated"
    else:
        # Label-based metrics still valid, but cannot calculate ranking/calibration metrics without usable scores.
        from sklearn.metrics import accuracy_score, f1_score
        scores["gliner"] = {"n": len(y), "accuracy": round(float(accuracy_score(y, g_labels)),5),
                            "f1": round(float(f1_score(y, g_labels)),5), "probability_note": "Unavailable"}
    result = {"split": args.split, "model": args.model, "scores": scores,
              "warning": "Cross-sectional Telco snapshot; not prospective churn forecasting. GLiNER confidence is uncalibrated."}
    dump(args.out / f"{args.split}_metrics.json", result)
    print(pd.DataFrame(scores).T.to_string())
    print("Saved", args.out / f"{args.split}_metrics.json")

if __name__ == "__main__":
    main()

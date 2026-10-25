"""The measured decide, data and learn decisions, where the store reads them (S8 A3 of the 2026-10-25 plan; the owner:
"make sure all data must capture and record and store for all 6 models we have").

The three measurements of real tables (``agent_runs/live_tests/2026-10-25-*``: the operating point, "would more rows
help", champion and challenger) write ``results.jsonl`` lines only. This module turns each measured decision into a
ledger record with its measured label, in the measurement's own workspace (``<folder>/workspace/ledger``), so the
store shapes them like any other decision (the m1 and m3 decision sets; the table named as its dataset, so it sits in
the lake's split). Nothing is decided here: the records say what the rules answered and what the unseen rows showed.

- ``threshold.operating_point``: one record per (table, cost statement). The rule's answer is the share today's rule
  (the argmax of the out-of-fold curve) acted on in the unseen rows; every other method's share is an answer beside
  it; the label is the best share in hindsight on the unseen rows (lines without it get no label).
- ``data.more_rows``: one record per table: the 34.13 rule's answer on half the rows; the label is what twice the rows
  did on unseen rows (``helped`` → likely_help, ``did_not_help`` → unlikely_help, ``unclear`` → no label).
- ``model.promote``: one record per replay: the 34.11 rule's promotion on the window; the label is the rows after it
  (``better`` → true, ``worse`` → false, ``same`` → no label).

Resumable: a table already written is skipped (``written.json``). Run: ``python -m dclab_rnd.ability_records all``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "agent_runs" / "live_tests"
FOLDERS = {"threshold": LIVE / "2026-10-25-threshold-methods", "more_rows": LIVE / "2026-10-25-more-rows-truth",
           "retrain": LIVE / "2026-10-25-retrain-truth"}
EVIDENCE = "fresh_draw"  # rows of the same table the decision never saw


def _lines(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            yield json.loads(line)
        except ValueError:
            continue


RESULTS = "results.jsonl"  # the train run's; a confirmation's own file (results_dev.jsonl) is written the same way


def _ref(folder: Path, did: Any, part: str) -> str:
    return f"{folder.relative_to(ROOT).as_posix()}/{RESULTS}#{did}:{part}"[:200]


def _write(folder: Path, name: str, write_one: Callable[[dict[str, Any], Path, Any], int], results: str = "results.jsonl") -> dict[str, int]:
    """Every finished table of a measurement written once into its own workspace's ledger."""
    from . import ledger

    home = folder / "workspace"
    done_path = home / "written.json"
    done = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.is_file() else set()
    book = ledger.FileLedger(home / "ledger", f"ability-{name}")
    book.home = home
    counts = {"tables": 0, "records": 0, "skipped": 0}
    global RESULTS
    RESULTS = results
    for rec in _lines(folder / results):
        did = rec.get("did")
        if rec.get("failed") or did is None or str(did) in done:
            counts["skipped"] += 1
            continue
        with ledger.dataset_scope(f"openml:{did}"):
            counts["records"] += write_one(rec, folder, book)
        done.add(str(did))
        counts["tables"] += 1
    home.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps(sorted(done)) + "\n", encoding="utf-8")
    return counts


def _decide(book: Any, point: str, key: str, answers: list[tuple[str, Any]], used: Any, state: dict[str, Any],
            options: Iterable[Any] = ()) -> Any:
    from . import ledger

    return ledger.decide(point, ("project", key), answers=[ledger.answer("rule", who, value) for who, value in answers],
                         value_used=used, used="rule", state=state, options=list(options), ledger=book)


def _label(book: Any, record: Any, value: Any, by: str, value_measured: Optional[float], ref: str) -> None:
    from . import ledger

    if record is not None and value is not None:
        ledger.label(record, value, "measured", by, evidence={"kind": EVIDENCE, "value": value_measured, "ref": ref}, ledger=book)


def _table_state(rec: dict[str, Any], rows: Optional[int] = None) -> dict[str, Any]:
    from dclab_core.bands import ratio_band, rows_band

    state = {"rows": rows_band(int(rows if rows is not None else rec.get("rows_used") or 0)), "columns": int(rec.get("columns") or 0)}
    if rec.get("positive_rate") is not None:
        state["positive_rate"] = ratio_band(float(rec["positive_rate"]))
    return state


def threshold_one(rec: dict[str, Any], folder: Path, book: Any) -> int:
    written = 0
    base = {**_table_state(rec, int(round(0.6 * (rec.get("rows_used") or 0)))), "kind": "binary", "of": "share", "metric": "roc_auc",
            "positives": int(rec.get("fit_positives") or 0), "cv_mean": float(rec["oof_auc"]), "calibration_error": float(rec["oof_ece"])}
    for s in rec.get("statements") or []:
        acted = s.get("share_acted") or {}
        state = {**base, "cost_ratio": float(s["spec_r"])}
        if s.get("spec_cap") is not None:
            state["capacity_share"] = float(s["spec_cap"])
        answers = [("rule:argmax_threshold", acted.get("argmax"))] + [(f"method:{m}", v) for m, v in acted.items() if m != "argmax"]
        record = _decide(book, "threshold.operating_point", f"openml:{rec['did']}:s{s['spec']}", answers, acted.get("argmax"), state)
        written += record is not None
        _label(book, record, s.get("oracle_share"), "measured:unseen_rows_best_share", s.get("oracle"), _ref(folder, rec["did"], f"s{s['spec']}"))
    return written


MORE_ROWS = {"helped": "likely_help", "did_not_help": "unlikely_help"}


def more_rows_one(rec: dict[str, Any], folder: Path, book: Any) -> int:
    slope = rec.get("slope") or {}
    state = {**_table_state(rec, rec.get("have_rows")), "kind": str(rec.get("task") or "binary"),
             "metric": "roc_auc" if rec.get("task") == "binary" else "r2"}
    if rec.get("curve"):
        state["cv_mean"] = float(rec["curve"][-1]["score"])
    if slope:
        state.update({"margin": float(slope["gain"]), "gain_low": float(slope["low"]), "gain_high": float(slope["high"])})
    record = _decide(book, "data.more_rows", f"openml:{rec['did']}", [("rule:data_value_rule_v1", rec.get("rule"))], rec.get("rule"), state,
                     options=("likely_help", "unlikely_help", "cannot_tell"))
    _label(book, record, MORE_ROWS.get(str(rec.get("truth"))), "measured:twice_the_rows_on_unseen_rows", rec.get("gain"), _ref(folder, rec["did"], "doubled"))
    return int(record is not None)


PROMOTE = {"better": True, "worse": False}  # a yes/no point answers true or false


def retrain_one(rec: dict[str, Any], folder: Path, book: Any) -> int:
    written = 0
    for r in rec.get("replays") or []:
        if "future" not in r:
            continue
        low = (r.get("window_interval") or [None])[0]
        state = {**_table_state(rec, int(round(float(r["origin"]) * (rec.get("rows_used") or 0)))), "kind": str(rec.get("task") or "binary"),
                 "metric": "roc_auc" if rec.get("task") == "binary" else "mae", "window_rows": int(r.get("window_rows") or 0),
                 "margin": float(r.get("window_difference") or 0.0)}
        if isinstance(low, (int, float)):
            state["gain_low"] = float(low)
        used = bool(r.get("promoted"))
        record = _decide(book, "model.promote", f"openml:{rec['did']}:o{r['origin']}", [("rule:retrain_compare_v1", used)], used, state)
        written += record is not None
        future = r["future"]
        _label(book, record, PROMOTE.get(str(future.get("truth"))), "measured:the_rows_after_the_window", future.get("difference"),
               _ref(folder, rec["did"], f"o{r['origin']}"))
    return written


WRITERS = {"threshold": threshold_one, "more_rows": more_rows_one, "retrain": retrain_one}


def write(name: str, folder: Optional[Path] = None, results: str = "results.jsonl") -> dict[str, int]:
    return _write(Path(folder or FOLDERS[name]), name, WRITERS[name], results)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.ability_records", description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=(*WRITERS, "all"))
    ap.add_argument("--results", default="results.jsonl", help="the measurement's results file (results_dev.jsonl: a confirmation's)")
    args = ap.parse_args(argv)
    for name in WRITERS if args.which == "all" else [args.which]:
        print(name, write(name, results=args.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

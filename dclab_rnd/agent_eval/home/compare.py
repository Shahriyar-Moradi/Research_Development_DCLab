"""Home's model benchmark from runs already made: no model is called (2026-10-29, the owner: Claude is never run again
beside a new model; its past runs are the benchmark, and a new model's run is added to the same table).

Each run is a stored result (``evidence/campaigns/home_v1/results/HOME-*.json``): its scores per arm (rules, Home with
the model, the model alone). Its model calls are read from the workspace's logs: the usage log for seconds and tokens,
the kept exchanges for the thinking each call returned. A call belongs to a run by its model and the run's time window
(from the kept folder's start to the result's time), so two runs made side by side with different models stay apart.

    python -m dclab_rnd.agent_eval.home.compare HOME-023 HOME-024 HOME-025          # prints the table
    python -m dclab_rnd.agent_eval.home.compare HOME-023 HOME-024 HOME-025 --write  # and writes the report beside the results
    python -m dclab_rnd.agent_eval.home.compare "HOME-025+HOME-026=Qwen"            # a run made in chunks, as one column

A local model's thinking is its whole reasoning; a Claude model's is the API's thinking summary (display "summarized").
The table says which, so the two are never read as the same thing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "evidence" / "campaigns" / "home_v1" / "results"
REPORTS = ROOT / "evidence" / "campaigns" / "home_v1" / "reports"
PURPOSES = ("home_intent", "home_agent")


def _result(run_id: str) -> tuple[Path, dict[str, Any]]:
    found = sorted(RESULTS.glob(f"{run_id}_*.json"))
    if not found:
        raise SystemExit(f"no result named {run_id} in {RESULTS.relative_to(ROOT)}")
    return found[0], json.loads(found[0].read_text(encoding="utf-8"))


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def window(result: dict[str, Any]) -> tuple[datetime, datetime]:
    """The run's start (its kept folder is named by the UTC time it began: dev-HHMMSS) and its end (the result's time).
    A folder name that is no time gives a window of six hours before the end."""
    end = _when(result["at"])
    kept = str(result.get("kept") or "")
    stamp = kept.rstrip("/").rsplit("-", 1)[-1]
    try:
        start = end.replace(hour=int(stamp[:2]), minute=int(stamp[2:4]), second=int(stamp[4:6]), microsecond=0)
        if start > end:
            start -= timedelta(days=1)
    except (ValueError, IndexError):
        start = end - timedelta(hours=6)
    return start, end + timedelta(seconds=5)


def _lines(path: Path):
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def calls(home: Path, model: str, start: datetime, end: datetime, keep_seconds: bool = False) -> dict[str, dict[str, Any]]:
    """Per purpose: the run's calls with their seconds, tokens and cost (usage log), and how many kept thinking (exchanges)."""
    out: dict[str, dict[str, Any]] = {p: {"calls": 0, "ok": 0, "seconds": [], "output_tokens": 0, "eur": 0.0,
                                          "kept": 0, "thought": 0, "thinking_chars": 0} for p in PURPOSES}
    for row in _lines(home / "model_requests.jsonl"):
        if row.get("model") != model or row.get("purpose") not in out or not (start <= _when(row["at"]) <= end):
            continue
        p = out[row["purpose"]]
        p["calls"] += 1
        p["ok"] += row.get("outcome") == "ok"
        p["seconds"].append(float(row.get("seconds") or 0))
        p["output_tokens"] += int(row.get("output_tokens") or 0)
        p["eur"] += float(row.get("cost_eur") or 0)
    days = {start.date(), end.date()}
    for day in sorted(days):
        for row in _lines(home / "model_exchanges" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.jsonl"):
            if row.get("model") != model or row.get("purpose") not in out or not (start <= _when(row["at"]) <= end):
                continue
            p = out[row["purpose"]]
            p["kept"] += 1
            thinking = row.get("thinking") or []
            text = "".join(str(b.get("text") or "") if isinstance(b, dict) else str(b) for b in thinking)
            if text.strip() or row.get("thought"):
                p["thought"] += 1
                p["thinking_chars"] += len(text)
    for p in out.values():
        s = sorted(p.pop("seconds"))
        if keep_seconds:
            p["_seconds"] = s
        p["median_s"] = round(statistics.median(s), 1) if s else None
        p["p90_s"] = round(s[min(len(s) - 1, int(0.9 * len(s)))], 1) if s else None
        p["eur"] = round(p["eur"], 4)
    return out


def _merge(parts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per purpose, the calls of several runs added up (a run made in chunks), the seconds' median and slowest tenth
    recomputed from every call."""
    out: dict[str, dict[str, Any]] = {}
    for p in PURPOSES:
        rows = [x[p] for x in parts if p in x]
        seconds = sorted(s for x in rows for s in x.get("_seconds", []))
        out[p] = {k: sum(x[k] for x in rows) for k in ("calls", "ok", "output_tokens", "kept", "thought", "thinking_chars")}
        out[p]["eur"] = round(sum(x["eur"] for x in rows), 4)
        out[p]["median_s"] = round(statistics.median(seconds), 1) if seconds else None
        out[p]["p90_s"] = round(seconds[min(len(seconds) - 1, int(0.9 * len(seconds)))], 1) if seconds else None
    return out


def row(run_id: str, home: Path, label: str | None = None) -> dict[str, Any]:
    """One column: a run, or a run made in chunks ("HOME-025+HOME-026+..."): their cases scored together, their calls added."""
    from . import summarize

    ids = [x for x in run_id.split("+") if x]
    results = [_result(i) for i in ids]
    models = {(r.get("plan") or {}).get("model") or "rules only" for _, r in results}
    if len(models) != 1:
        raise SystemExit(f"{run_id}: chunks of one column must share one model, not {sorted(models)}")
    model = models.pop()
    cases = [row for _, r in results for row in r.get("rows") or []]
    if len({c["case"] for c in cases}) != len(cases):
        raise SystemExit(f"{run_id}: a case is in two chunks")
    unscored = [i for i, (_, r) in zip(ids, results) if not r.get("summary")]
    s = summarize(cases) if len(results) > 1 else (results[0][1].get("summary") or {})
    arms = {}
    for arm in ("rules", "model", "plain"):
        a = s.get(arm)
        if not a:
            continue
        en = (a.get("track_accuracy_by_lang") or {}).get("en")
        arms[arm] = {"kind_right": (a.get("track") or {}).get("right"), "cases": a.get("cases"), "kind_en": en,
                     "fields": a.get("fields"), "exit_right": a.get("exit_right"),
                     "traps": f"{sum(bool(v) for v in (a.get('traps') or {}).values())}/{len(a.get('traps') or {})}",
                     "uncited": a.get("uncited"), "quote_invalid": a.get("quote_invalid"), "lost": a.get("lost")}
    local = all(bool((r.get("plan") or {}).get("local")) for _, r in results)
    per_run = [calls(home, model, *window(r), keep_seconds=True) for _, r in results] if model != "rules only" else []
    return {"run": run_id, "files": [str(p.relative_to(ROOT)) for p, _ in results], "label": label or model, "model": model,
            "split": results[0][1].get("split"), "at": max(str(r.get("at")) for _, r in results), "unscored_chunks": unscored,
            "spent_eur": round(sum(float(r.get("spent_eur") or 0) for _, r in results), 4), "local": local,
            "thinking_is": "the model's whole reasoning" if local else "the API's thinking summary",
            "arms": arms, "calls": _merge(per_run) if per_run else {}}


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({round(100 * a / b)}%)" if b else "-"


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| | " + " | ".join(r["label"] for r in rows) + " |", "|---|" + "---|" * len(rows)]

    def add(name: str, value) -> None:
        lines.append(f"| {name} | " + " | ".join(str(value(r)) for r in rows) + " |")

    def arm(r, name):
        return r["arms"].get(name) or {}

    add("run", lambda r: r["run"].replace("+", " + "))
    add("Home: kind of help right", lambda r: f"{arm(r, 'model').get('kind_right')}/{arm(r, 'model').get('cases')}")
    add("Home: kind right, English", lambda r: arm(r, "model").get("kind_en"))
    add("Home: fields right / wrong / missed", lambda r: "{right} / {wrong} / {missed}".format(**(arm(r, "model").get("fields") or {"right": "-", "wrong": "-", "missed": "-"})))
    add("Home: right next step (exit)", lambda r: arm(r, "model").get("exit_right"))
    add("Home: traps caught", lambda r: arm(r, "model").get("traps"))
    add("Home: numbers with no source", lambda r: arm(r, "model").get("uncited"))
    add("model alone: kind right", lambda r: f"{arm(r, 'plain').get('kind_right')}/{arm(r, 'plain').get('cases')}")
    add("model alone: fields right / wrong / missed", lambda r: "{right} / {wrong} / {missed}".format(**(arm(r, "plain").get("fields") or {"right": "-", "wrong": "-", "missed": "-"})))
    add("model alone: traps caught", lambda r: arm(r, "plain").get("traps"))
    for p in PURPOSES:
        add(f"{p}: calls ok", lambda r, p=p: _pct(r["calls"].get(p, {}).get("ok", 0), r["calls"].get(p, {}).get("calls", 0)))
        add(f"{p}: thinking kept", lambda r, p=p: _pct(r["calls"].get(p, {}).get("thought", 0), r["calls"].get(p, {}).get("kept", 0)))
        add(f"{p}: median / slowest tenth (s)", lambda r, p=p: f"{r['calls'].get(p, {}).get('median_s')} / {r['calls'].get(p, {}).get('p90_s')}")
    add("thinking is", lambda r: r["thinking_is"])
    add("cost of the run (EUR)", lambda r: r["spent_eur"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="result ids, e.g. HOME-023 HOME-024; RUN=label names a column")
    parser.add_argument("--home", default=str(ROOT / "agent_runs"), help="the workspace folder whose logs hold the runs' calls")
    parser.add_argument("--write", action="store_true", help="write the report and its JSON to evidence/campaigns/home_v1/reports/")
    parser.add_argument("--name", default=None, help="the report's file name (default: model_comparison_<date>)")
    args = parser.parse_args(argv)
    rows = []
    for spec in args.runs:
        run_id, _, label = spec.partition("=")
        rows.append(row(run_id, Path(args.home), label or None))
    text = table(rows)
    print(text)
    if args.write:
        REPORTS.mkdir(parents=True, exist_ok=True)
        name = args.name or f"model_comparison_{datetime.now(timezone.utc):%Y-%m-%d}"
        for r in rows:
            for p in r["calls"].values():
                p.pop("_seconds", None)
        (REPORTS / f"{name}.json").write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (REPORTS / f"{name}.md").write_text(
            f"# Home model comparison ({datetime.now(timezone.utc):%Y-%m-%d})\n\nBuilt by `python -m dclab_rnd.agent_eval.home.compare "
            f"{' '.join(args.runs)} --write` from stored runs; no model was called.\n\n{text}\n", encoding="utf-8")
        print(f"written: {(REPORTS / name).relative_to(ROOT)}.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

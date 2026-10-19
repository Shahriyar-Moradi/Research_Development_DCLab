"""Why did this number change, on the mall sales replica (package 30.20): ``python -m dclab_rnd.metric_change [--seeds 0 1 2] [--write]
[--model --yes --cap-eur 0.3]``.

Every planted question is asked three ways: in its own words, read by the rules; naming its template (the model-off path); and in other
words (the replica's paraphrases, written before any model answered them), read by the rules and, with ``--model``, by the product's model
through the gateway (the ``metric_question`` purpose: it only chooses a template and fills it; the rules' reading stands when its choice
fails the check). Each answer is judged against what the planted cause says a careful analyst finds. ``--write`` stores the record in
``evidence/campaigns/metric_change_v1/results/MCH-<n>_mall_sales.json`` and every answer whole, with what each reader said, beside it in
``runs/MCH-<n>_answers.jsonl.gz``. A model is asked only with ``--model --yes`` and a cap.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "metric_change_v1"
ARMS = ("words_rules", "named", "paraphrase_rules", "paraphrase_model")


def sqlite_runner(layer: Any, tables: dict[str, Any]) -> Any:
    """The studio's runner: the layer's tables in an in-memory SQLite database (dates as ISO days), each tested statement run there with
    its parameters bound, so the SQL under an answer is the statement that made its numbers."""
    import sqlite3

    import pandas as pd

    db = sqlite3.connect(":memory:", check_same_thread=False)
    for t in layer.tables:
        frame = tables[t.name].copy()
        frame[t.date] = pd.to_datetime(frame[t.date], errors="coerce").dt.strftime("%Y-%m-%d")
        frame.to_sql(t.name, db, index=False)
    return lambda sql, params: pd.read_sql_query(sql, db, params=params)


def data_of(case: dict[str, Any]) -> Any:
    from dclab_core import metric_change as mc

    r = case["replica"]
    tables = {"footfall": r.footfall, "sales": r.sales}
    return mc.Data(case["layer"], None, runner=sqlite_runner(case["layer"], tables))


def run(seeds: tuple[int, ...] = (0, 1, 2), advisor: Any = None, log=print) -> dict[str, Any]:
    """Each seed's questions through every arm; ``advisor`` (a GatewayAdvisor) adds the model's arm on the paraphrases."""
    from dclab_core import metric_change as mc
    from dclab_rnd import kinds
    from dclab_rnd.kinds.metric_change import reached

    kind = kinds.get("metric_change")
    rows, detail = [], []
    for seed in seeds:
        case = kind.case(seed)
        r, layer = case["replica"], case["layer"]
        data = data_of(case)  # every statement run on SQLite: the query shown made the numbers
        named = {x["truth"]["id"]: x["named"] for x in kind.solve(case)["answers"]}
        for truth in r.truth:
            asked = {"words_rules": truth["question"], "paraphrase_rules": truth["paraphrase"]}
            got = {arm: mc.ask(q, layer, {}, data=data, seed=seed) for arm, q in asked.items()}
            if named.get(truth["id"]) is not None:
                got["named"] = (named[truth["id"]], {"named": True})
            if advisor is not None:
                got["paraphrase_model"] = mc.ask(truth["paraphrase"], layer, {}, advisor, data=data, seed=seed)
            for arm, (a, said) in got.items():
                ok = reached(truth, a)
                rows.append({"seed": seed, "case": truth["id"], "arm": arm, "reached": ok, "status": a.status, "source": a.choice.source,
                             "first_factor": a.first_factor, "first_segment": a.first_segment, "missing": a.missing})
                detail.append({"seed": seed, "case": truth["id"], "arm": arm, "question": a.question, "truth": truth, "answer": a.model_dump(mode="json"),
                               "said": said, "reached": ok})
        log(f"seed {seed}: " + ", ".join(f"{arm} {sum(x['reached'] for x in rows if x['seed'] == seed and x['arm'] == arm)}/"
                                          f"{sum(1 for x in rows if x['seed'] == seed and x['arm'] == arm)}" for arm in ARMS if any(x["arm"] == arm for x in rows)))
    by_arm = {arm: {"reached": sum(x["reached"] for x in rows if x["arm"] == arm), "asked": sum(1 for x in rows if x["arm"] == arm)}
              for arm in ARMS if any(x["arm"] == arm for x in rows)}
    return {"seeds": list(seeds), "by_arm": by_arm, "rows": rows, "detail": detail,
            "limits": ["Invented data with planted causes: a found cause says the analysis is sound on these traps, never why a real number moved.",
                       "The paraphrases were written by the same author as the questions; real people word their questions in more ways.",
                       "A factor is said to move when its interval over days, one for all the factors (Bonferroni, 90%), excludes zero: about "
                       "one question in ten with nothing planted may still name one."]}


def write(doc: dict[str, Any], campaign: Path = CAMPAIGN, spend: Optional[dict[str, Any]] = None) -> Path:
    results, runs = campaign / "results", campaign / "runs"
    results.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("MCH-*.json")] or [0])
    ident = f"MCH-{n:03d}"
    path, kept = results / f"{ident}_mall_sales.json", runs / f"{ident}_answers.jsonl.gz"
    if path.exists() or kept.exists():
        raise SystemExit(f"{ident} exists: results are never overwritten")
    with gzip.open(kept, "wt", encoding="utf-8") as fh:
        for row in doc["detail"]:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    record = {k: v for k, v in doc.items() if k != "detail"}
    path.write_text(json.dumps({"id": ident, "experiment_id": ident, "campaign_id": "metric_change_v1", "kind": "metric_change", "status": "completed",
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record, "spend": spend or {"eur": 0.0, "model": None},
                                "detail": f"runs/{kept.name}",
                                "rule": "Each planted question of the mall sales replica asked in its words, by its template and in other words; an answer "
                                        "counts when it reaches what the planted cause says (the factor first, the segment where, the refusal with what "
                                        "is missing): evidence about the method on invented data."}, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def detail(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".json":
        p = p.parent.parent / json.loads(p.read_text(encoding="utf-8"))["detail"]
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.metric_change", description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--model", action="store_true", help="also ask the product's model on the paraphrases (needs --yes and --cap-eur)")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    args = parser.parse_args(argv)
    advisor = None
    if args.model:
        questions = 8 * len(args.seeds)
        print(f"plan: {questions} questions to the metric_question purpose, cap EUR {args.cap_eur}")
        if not (args.yes and args.cap_eur):
            print("not run: add --yes and --cap-eur to ask the model")
            return 1
        from dclab_rnd.core_advisor import GatewayAdvisor

        advisor = GatewayAdvisor(cap_eur=args.cap_eur, max_tokens=400)
    doc = run(tuple(args.seeds), advisor)
    spend = {"eur": round(advisor.spent_eur(), 4), "requests": advisor.calls, "failures": advisor.failures, "model": advisor.report()} if advisor else None
    if spend:
        print(f"spent EUR {spend['eur']}, {spend['requests']} requests, {spend['failures']} failures")
    if args.write:
        print(write(doc, spend=spend))
    return 0


if __name__ == "__main__":
    sys.exit(main())

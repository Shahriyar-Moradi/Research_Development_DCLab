"""The students on real tables, in shadow: ``python -m dclab_rnd.real_shadow run|measure|silver`` (the owner, 2026-10-23).

The students learned on invented tables. This runs DCLab's own product path on real ones and records what happens, so
each student's whisper sits beside the rule on real columns.

**Which table goes where** (the lake's split of each OpenML table, ``dclab_core.lake.split_of``, the same as in the lake's
gold, the decisions export and ``real_v1``):

- a **train** table is recorded in ``TRAIN_HOME`` (``agent_runs/live_tests/2026-10-23-real-shadow/workspace``), whose
  consent is improve and train: these records are the students' real homework, read by the lake and the store;
- a **dev** table is recorded in ``EVAL_HOME`` (``.../eval``), whose consent is off: only ``measure`` reads it, against
  the truths ``real_v1``'s dev rows declare, so no student is ever trained on what it is measured on;
- a **sealed** table is left out: a sealed set is read once, by a gate.

The commands:

- ``run``: each table is read once in a temporary folder (deleted after), profiled, given DCLab's solution proposal for
  its declared outcome, and its decisions recorded through ``ledger.capture.solution_saved`` exactly as a project records
  them (the proposal accepted as it is, so the answers are the rule's), every fitted student's answer beside as a shadow
  (``ledger.shadow``: recorded, never used); each column's role (``column.semantic_role``) from real_homework's derived
  state and its rule (``datacard:purposes``). Resumable: a table done keeps its marker; ``--retry-failed`` reads the
  failed ones again. A table whose records could not be written gets no "done" marker.
- ``measure``: per point and student, coverage (decisions the student answered), agreement with the rule, and, on the
  dev tables, the rule's and the student's accuracy against the declared truth.
- ``silver``: Claude's opinion on a fair sample of the train tables' disagreements (``decision_answer``, the state only:
  bands and name tokens, never a value), within a euro cap; a failed request is tried once more, then kept as an
  abstention (marked failed); stopped at the cap or after three failures in a row, never writing those. Kept apart, in ``silver.jsonl``: evaluation and silver, never a label in the ledger (the owner's rule,
  2026-10-17).

Nothing here is a sealed verdict: the real tables' declarations are their uploaders', not checked.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "agent_runs" / "live_tests" / "2026-10-23-real-shadow"  # gitignored; the archive keeps live tests
TRAIN_HOME = BASE / "workspace"  # the lake reads live_tests/*/workspace
EVAL_HOME = BASE / "eval"
DERIVED = ROOT / "data" / "downloads" / "openml" / "real_v1"  # real_homework's records: each column's state and its rules
REAL = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "real_v1" / "decisions"
TRUTH_POINTS = ("column.is_identifier", "column.semantic_role")
ROLES = ["numeric", "categorical_code", "identifier", "datetime", "free_text", "other"]
RETRY = ("URLError", "Timeout", "timed out", "Connection", "HTTP", "RemoteDisconnected", "IncompleteRead")


def _home(home: Path, train: bool) -> Path:
    """A workspace folder with its consent: improve and train for the train tables, off for the dev tables."""
    from . import consent

    home = Path(home)
    (home / "done").mkdir(parents=True, exist_ok=True)
    for use in ("improve", "train"):
        consent.set_workspace_consent(home, use, train, who="owner")
    return home


def split_of(did: Any) -> str:
    from dclab_core import lake

    return lake.split_of(f"openml:{did}")


def _tables(limit: Optional[int] = None) -> list[dict[str, Any]]:
    """The real tables with a derived record that read (their outcome and their columns' states and rules), sealed left out."""
    out = []
    for f in sorted(DERIVED.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not d.get("failed") and d.get("target") and d.get("columns") and split_of(d["did"]) != "sealed":
            out.append(d)
    out.sort(key=lambda d: split_of(d["did"]) != "dev")  # the measurable tables first
    return out[:limit] if limit else out


def _download(did: Any) -> Any:
    import pandas as pd

    from .lake import openml

    desc = openml._json(f"{openml.API}/data/{did}")["data_set_description"]
    if not desc.get("parquet_url"):
        raise ValueError("no parquet file")
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "data.parquet"
        file.write_bytes(openml._get(desc["parquet_url"], timeout=120))
        return pd.read_parquet(file)


def roles(derived: dict[str, Any]) -> int:
    """Each column's role (``column.semantic_role``) with real_homework's rule (``datacard:purposes``) and derived state, so
    each fitted student whispers beside it. The product path records roles only in the engine's data stage."""
    from .ledger import capture

    written = 0
    for name, c in list((derived.get("columns") or {}).items())[:200]:
        if c.get("role_rule") is None or not c.get("state"):
            continue
        written += bool(capture._put("column.semantic_role", ("column", name), [capture.answer("rule", "datacard:purposes", c["role_rule"])],
                                     c["role_rule"], "rule", state=c["state"], options=ROLES, draft=f"openml-{derived['did']}"))
    return written


def run_one(derived: dict[str, Any], home: Path, retry_failed: bool = False) -> dict[str, Any]:
    """One real table through the product path, its decisions recorded in ``home``'s ledger with the students' shadows."""
    from . import ledger
    from .ledger import batch, capture
    from .studio import data as studio_data, solution as studio_solution

    did = derived["did"]
    marker = home / "done" / f"{did}.json"
    if marker.is_file():
        done = json.loads(marker.read_text(encoding="utf-8"))
        if not (retry_failed and done.get("failed") and any(k in done["failed"] for k in RETRY)):
            return done
    dropped = int(ledger.DROPPED.get("count", 0)) if isinstance(getattr(ledger, "DROPPED", None), dict) else 0
    try:
        frame = _download(did)
        target = derived["target"]
        profile = studio_data.profile_table(frame)
        proposal = studio_solution.propose(frame, profile, target)
        solution = {"target": target, "task": proposal.get("task"), "forbidden": proposal.get("forbidden") or [],
                    "identifiers": proposal.get("identifiers") or [], "time_column": (proposal.get("time_candidates") or [None])[0]}
        with batch():
            # the proposal accepted as it is (actor "rule"): every answer recorded is a rule's; filed under the table, not a
            # project (a project no store knows is stamped "operate"); the workspace's consent stamps the records
            written = int(capture.solution_saved(proposal, profile, solution, actor="rule", draft=f"openml-{did}") or 0)
            written += roles(derived)
        lost = int(ledger.DROPPED.get("count", 0)) - dropped if isinstance(getattr(ledger, "DROPPED", None), dict) else 0
        if lost > 0:  # the batch could not write (a full disk): no "done", the next run reads the table again
            return {"did": did, "failed": f"{lost} records could not be written"}
        out = {"did": did, "split": split_of(did), "rows": int(len(frame)), "columns": int(frame.shape[1]), "decisions": written}
    except Exception as exc:  # noqa: BLE001 - one table that cannot be read or proposed costs itself only
        out = {"did": did, "failed": f"{type(exc).__name__}: {str(exc)[:200]}"}
    marker.write_text(json.dumps(out) + "\n", encoding="utf-8")
    return out


def run(train_home: Optional[Path] = None, eval_home: Optional[Path] = None, limit: Optional[int] = None,
        retry_failed: bool = False, log=print) -> dict[str, Any]:
    from .ledger import install, open_ledger

    homes = {"train": _home(train_home or TRAIN_HOME, True), "dev": _home(eval_home or EVAL_HOME, False)}
    books = {k: open_ledger(h) for k, h in homes.items()}
    os.environ["DCLAB_DECISION_SHADOW"] = "1"  # every fitted student whispers
    tables, counts, started = _tables(limit), Counter(), time.time()
    try:
        for i, d in enumerate(tables, 1):
            split = split_of(d["did"])
            install(books[split])
            r = run_one(d, homes[split], retry_failed)
            counts[("failed" if r.get("failed") else "done") + "_" + split] += 1
            counts["decisions"] += int(r.get("decisions") or 0)
            if i % 25 == 0 or i == len(tables):
                log(f"{i}/{len(tables)} tables {dict(counts)} {time.time() - started:.0f}s")
    finally:
        install(None)
    return dict(counts)


# ---------------------------------------------------------------------- measuring


def _truths() -> dict[tuple[str, str], dict[str, Any]]:
    """(openml-<did>, column) -> the truths real_v1's DEV rows declare (identifier, role). Never its train rows and never
    its sealed file (read once, by a gate)."""
    out: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for point in TRUTH_POINTS:
        path = REAL / point / "rows.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            r = json.loads(line)
            if r.get("split") != "dev":
                continue
            did = str(r.get("group") or "").rsplit(":", 1)[-1]
            out[(f"openml-{did}", (r.get("subject") or {}).get("key"))][point] = _yes((r.get("label") or {}).get("value"))
    return out


def _yes(value: Any) -> Any:
    return "yes" if value is True else "no" if value is False else value


def _rows(home: Path, shadowed: bool = True) -> list[dict[str, Any]]:
    """Each recorded decision once (the last per point, table and subject: a stopped run read again records twice): the
    point, the table, the subject, the state, the rule's answer and every student's (none when it abstained)."""
    from .ledger import open_ledger

    out: dict[tuple, dict[str, Any]] = {}
    for r in open_ledger(Path(home)).records():
        doc = r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r)
        answers = doc.get("answers") or []
        rule = next((a for a in answers if a.get("by") == "rule"), None)
        if rule is None:
            continue
        students = {a.get("answerer"): _yes(a.get("value")) for a in answers if a.get("source") == "shadow"}
        if shadowed and not students:
            continue
        subject = doc.get("subject") or {}
        row = {"record": doc.get("id"), "point": doc.get("point"), "where": doc.get("draft") or doc.get("project"),
               "subject": subject.get("key") if isinstance(subject, dict) else subject, "state": doc.get("state") or {},
               "options": doc.get("options") or [], "rule": _yes(rule.get("value")), "students": students}
        out[(row["point"], row["where"], row["subject"])] = row
    return list(out.values())


def measure(home: Optional[Path] = None) -> dict[str, Any]:
    """Per point and student: coverage, agreement with the rule, and accuracy against the dev truths (``EVAL_HOME``)."""
    home = Path(home or EVAL_HOME)
    rows, truths = _rows(home, shadowed=False), _truths()
    by: dict[tuple[str, str], Counter] = defaultdict(Counter)
    decisions: Counter = Counter()
    for r in rows:
        decisions[r["point"]] += 1
        truth = (truths.get((r["where"], r["subject"])) or {}).get(r["point"]) if r["point"] in TRUTH_POINTS else None
        if truth is not None:
            by[(r["point"], "rule")]["truth"] += 1
            by[(r["point"], "rule")]["right"] += str(r["rule"]) == str(truth)
        for student, said in r["students"].items():
            m = by[(r["point"], student)]
            m["answered"] += said is not None
            m["agree"] += str(said) == str(r["rule"])
            m["disagree"] += said is not None and str(said) != str(r["rule"])
            if truth is not None and said is not None:
                m["truth"] += 1
                m["right"] += str(said) == str(truth)
    out: dict[str, Any] = {}
    for (point, who), m in sorted(by.items()):
        n = decisions[point]
        out.setdefault(point, {"decisions": n})[who] = {
            **dict(m), **({"coverage": round(m["answered"] / n, 4), "agreement": round(m["agree"] / max(m["answered"], 1), 4)} if who != "rule" else {}),
            **({"accuracy": round(m["right"] / m["truth"], 4)} if m["truth"] else {})}
    home.mkdir(parents=True, exist_ok=True)
    (home / "measure.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def compare(folders: dict[str, Path], home: Optional[Path] = None) -> dict[str, Any]:
    """Several generations of students on the same dev tables (``EVAL_HOME``): for each point, each folder's student answers
    every recorded state (not a shadow kept in the record: the same rows for all), scored against the declared truth, beside
    the rule, and against Claude's silver opinion where one was asked. Written to ``compare.json``."""
    from .ledger import shadow

    home = Path(home or EVAL_HOME)
    rows, truths = _rows(home, shadowed=False), _truths()
    silver_by: dict[str, Any] = {}
    path = home / "silver.jsonl"
    for line in path.read_text().splitlines() if path.is_file() else []:
        r = json.loads(line)
        if r.get("claude") is not None:
            silver_by[r["record"]] = r["claude"]
    students = {name: {getattr(m, "point", None) or next(iter(m.points)): m for m in shadow.answerers(str(folder))} for name, folder in folders.items()}
    out: dict[str, Any] = {}
    for point in sorted({r["point"] for r in rows}):
        mine = [r for r in rows if r["point"] == point]
        entry: dict[str, Any] = {"decisions": len(mine)}
        truth_of = {r["record"]: (truths.get((r["where"], r["subject"])) or {}).get(point) for r in mine}
        judged = [r for r in mine if truth_of[r["record"]] is not None]
        if judged:
            entry["rule"] = {"truth": len(judged), "accuracy": round(sum(str(r["rule"]) == str(truth_of[r["record"]]) for r in judged) / len(judged), 4)}
        for name, by_point in students.items():
            model = by_point.get(point)
            if model is None:
                continue
            said = {}
            for r in mine:
                a = model.answer(point, r["state"], r["options"])
                said[r["record"]] = _yes(getattr(a, "value", None)) if a is not None else None
            answered = [r for r in mine if said[r["record"]] is not None]
            e: dict[str, Any] = {"coverage": round(len(answered) / len(mine), 4),
                                 "agree_with_rule": round(sum(str(said[r["record"]]) == str(r["rule"]) for r in answered) / max(len(answered), 1), 4)}
            right = [r for r in judged if said[r["record"]] is not None]
            if right:
                e["accuracy"] = round(sum(str(said[r["record"]]) == str(truth_of[r["record"]]) for r in right) / len(right), 4)
            claude = [r for r in answered if r["record"] in silver_by]
            if claude:
                e["agree_with_claude"] = round(sum(str(said[r["record"]]) == str(silver_by[r["record"]]) for r in claude) / len(claude), 4)
                e["claude_n"] = len(claude)
            entry[name] = e
        out[point] = entry
    (home / "compare.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------- Claude's silver opinion on the disagreements


def silver(home: Optional[Path] = None, per_point: int = 60, cap_eur: float = 3.0, seed: int = 0, log=print) -> dict[str, Any]:
    """Claude answers a fair sample of each point's disagreements (the state only), within ``cap_eur``; kept apart. It
    stops at the first failure or the cap without writing an empty answer, so a later run with more budget asks again."""
    from dclab_core.answerers import ModelAnswerer

    from . import models
    from .core_advisor import GatewayAdvisor

    home = Path(home or TRAIN_HOME)
    gateway = models.for_tools()  # the usage log and the workspace's and the role's monthly caps apply
    if gateway is None or not gateway.available("decision_answer"):
        raise SystemExit("no model is configured for decision_answer")
    rows = [r for r in _rows(home) if any(v is not None and str(v) != str(r["rule"]) for v in r["students"].values())]
    by_point: dict[str, list] = defaultdict(list)
    for r in sorted(rows, key=lambda r: str(r["record"])):  # a fixed order before the seeded shuffle: the same sample for the same records
        by_point[r["point"]].append(r)
    rng, out_path, asked, in_a_row, step = random.Random(seed), home / "silver.jsonl", Counter(), 0, 0.0
    last = {}  # each record's latest line: a failed one is asked again (its new line follows it)
    for line in out_path.read_text().splitlines() if out_path.is_file() else []:
        row = json.loads(line)
        last[row["record"]] = row
    done = {rec for rec, row in last.items() if not row.get("failed")}
    advisor = GatewayAdvisor(gateway, max_tokens=600, cap_eur=cap_eur)  # 200 cut Claude's one-sentence reasons (live, 2026-10-23)
    answerer = ModelAnswerer(advisor, list(by_point), name="claude", version="silver-1")
    with out_path.open("a", encoding="utf-8") as fh:
        for point, items in sorted(by_point.items()):
            rng.shuffle(items)
            for r in items[:per_point]:
                if r["record"] in done:
                    continue
                failed = False
                for attempt in range(2):  # one retry: a passing failure or a reply that did not read
                    errors, before = answerer.errors, advisor.spent_eur()
                    a = answerer.answer(point, r["state"], r["options"])
                    step = max(step, advisor.spent_eur() - before)
                    failed = a is None and answerer.errors > errors
                    if not failed:
                        break
                    if advisor.spent_eur() + 1.5 * step >= cap_eur:  # the cap (a refused request near it): stop, write nothing
                        log("stopped: the cap was reached")
                        return {"asked": dict(asked), "spent_eur": round(advisor.spent_eur(), 4), "stopped": True, "why": "cap"}
                    time.sleep(2 * (attempt + 1))
                in_a_row = in_a_row + 1 if failed else 0
                if in_a_row >= 3:  # three in a row: an outage; stop without writing this one
                    log("stopped: three requests in a row failed")
                    return {"asked": dict(asked), "spent_eur": round(advisor.spent_eur(), 4), "stopped": True, "why": "failures"}
                fh.write(json.dumps({"record": r["record"], "point": point, "where": r["where"], "subject": r["subject"], "state": r["state"],
                                     "rule": r["rule"], "students": r["students"], "claude": _yes(getattr(a, "value", None)) if a is not None else None,
                                     "abstained": a is None, "failed": failed, "label": "silver", "use": "evaluation"}, ensure_ascii=False, default=str) + "\n")
                asked[point] += 1
    return {"asked": dict(asked), "spent_eur": round(advisor.spent_eur(), 4)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("run", "measure", "silver"))
    ap.add_argument("--home", type=Path, help="measure: the eval workspace; silver: the train workspace")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--per-point", type=int, default=60)
    ap.add_argument("--cap-eur", type=float, default=3.0)
    args = ap.parse_args(argv)
    if args.command == "run":
        print(json.dumps(run(limit=args.limit, retry_failed=args.retry_failed)))
    elif args.command == "measure":
        print(json.dumps(measure(args.home), indent=1))
    else:
        print(json.dumps(silver(args.home, args.per_point, args.cap_eur), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

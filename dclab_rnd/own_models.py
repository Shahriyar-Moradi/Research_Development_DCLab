"""A company's own decision models (package 26.7): exported, fine-tuned, gated and served inside its workspace.

Every step stays in the workspace's folder (``<workspace>/own_models/``); none of its rows leave it, and none reach
DCLab's own training unless the workspace's ``train`` consent is on (that path is ``make datasets`` and the lake, never
this one). DCLab's base models are read, never changed.

- ``export``: the workspace's labelled decisions per point, in the decision sets' shape (``dclab_core.datasets.decisions``:
  whole projects to train, dev or sealed by their hash), to ``sets/<point>/rows.jsonl``, beside ``chat.jsonl``: the same
  decisions as the chat rows a language model is fine-tuned on (asked exactly as the gateway's ``decision_answer``
  purpose asks; the LoRA run is the company's, ``research/llm-fine-tuning/experiments/lake/TRAINING.md``).
- ``fit``: the own model of a point (``dclab_core.own_models.tune``: DCLab's base classifier for the point, when there is
  one, read as features), its card and its fitted file with its SHA-256 (``models/``). Sealed rows are never read here.
- ``gate --yes``: the own model judged once on the workspace's sealed decisions by ``decision_gate_v1`` beside the
  rule (``results/<point>.json``); the sealed rows are marked read before they are opened.
- ``status``: what each point has: rows by split, a card, a verdict.

A passed own model can be routed in its workspace through the audited switch (``ledger.routing.switch`` with the
workspace's home), like DCLab's: above shadow only after its own sealed pass, acting only after its shadow measurement.

    python -m dclab_rnd.own_models export|fit|gate|status [--home DIR] [--points p1,p2] [--yes]
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

FOLDER = "own_models"
NAME = "own"


def root(home: Path) -> Path:
    return Path(home) / FOLDER


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _book(home: Path, book: Any = None) -> Any:
    from . import ledger
    from .experience import file_workspace_id

    return book if book is not None else ledger.open_ledger(Path(home), file_workspace_id(Path(home)))


def export(home: Path, book: Any = None, points: Optional[list[str]] = None) -> dict[str, Any]:
    """The workspace's labelled decisions per point (rows with no label are left out), split by whole project."""
    from dclab_core import datasets, prompts
    from dclab_core.decision_export import PURPOSE, request

    book = _book(home, book)
    labels: dict[str, list[dict[str, Any]]] = {}
    for x in book.labels():
        labels.setdefault(x.record, []).append(x.model_dump(mode="json"))
    docs = [{**r.model_dump(mode="json"), "labels": labels.get(r.id, [])} for r in book.records()]
    wanted = sorted({d["point"] for d in docs if d["labels"]} if points is None else set(points))
    out, summary = root(home) / "sets", {}
    system = prompts.text(PURPOSE)
    for point in wanted:
        rows, _ = datasets.decisions(docs, point, workspace=Path(home).name)
        rows = [r for r in rows if r["label"] is not None]
        if not rows:
            continue
        folder = out / point
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "rows.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
        chat = []
        for r in rows:
            if r["split"] == "sealed":
                continue  # a sealed decision is never a training row of any model
            value = r["label"]["value"]
            answer = "yes" if value is True else "no" if value is False else str(value)
            chat.append({"messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": json.dumps(request(point, r["state"], r.get("options")), ensure_ascii=False)},
                                      {"role": "assistant", "content": json.dumps({"answer": answer}, ensure_ascii=False)}],
                         "metadata": {"point": point, "split": r["split"], "group": r["group"]}})
        (folder / "chat.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chat), encoding="utf-8")
        summary[point] = {"rows": len(rows), "by_split": dict(Counter(r["split"] for r in rows)), "groups": len({r["group"] for r in rows}),
                          "labels": dict(Counter(r["label"]["source"] for r in rows)), "chat_rows": len(chat)}
    manifest = {"at": _now(), "points": summary, "rule": "This workspace's own labelled decisions, kept in its folder: whole projects split "
                "by their hash (train, dev, sealed); the sealed decisions judge its own model once and are never trained on."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def rows(home: Path, point: str) -> list[dict[str, Any]]:
    path = root(home) / "sets" / point / "rows.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def base_model(point: str) -> Any:
    """DCLab's base classifier for ``point`` (the fitted ones whose bytes match their record), or None."""
    from .ledger import shadow

    return next((m for m in shadow.answerers() if point in m.points), None)


def fit(home: Path, points: Optional[list[str]] = None) -> dict[str, Any]:
    """The own model of each exported point; a point with too few projects or one label value gets none (``skipped``)."""
    import joblib

    from dclab_core.own_models import tune

    sets = root(home) / "sets"
    points = points or sorted(p.name for p in sets.iterdir() if p.is_dir()) if sets.is_dir() else []
    models = root(home) / "models"
    models.mkdir(parents=True, exist_ok=True)
    cards, skipped = {}, {}
    for point in points:
        data = rows(home, point)
        base = base_model(point)
        try:
            model, card = tune(point, [r for r in data if r["split"] == "train"], [r for r in data if r["split"] == "dev"], base=base, name=NAME)
        except ValueError as e:
            skipped[point] = str(e)
            continue
        (models / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        joblib.dump(model, models / f"{point}.joblib")
        cards[point] = card
    hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(models.glob("*.joblib"))}
    (models / "fitted.json").write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"cards": cards, "skipped": skipped}


def load(home: Path) -> list[Any]:
    """The workspace's own fitted models whose bytes match ``fitted.json`` and whose card describes them."""
    import joblib

    models = root(home) / "models"
    try:
        hashes = json.loads((models / "fitted.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for card_path in sorted(models.glob("*.json")):
        fitted = card_path.with_suffix(".joblib")
        if card_path.name == "fitted.json" or not fitted.exists():
            continue
        try:
            if hashlib.sha256(fitted.read_bytes()).hexdigest() != hashes.get(card_path.stem):
                continue  # not the file that was built: never unpickled
            card, model = json.loads(card_path.read_text(encoding="utf-8")), joblib.load(fitted)
        except Exception:  # noqa: BLE001 - a model that cannot be read answers nothing
            continue
        if getattr(model, "version", None) == card.get("version") and getattr(model, "point", None) == card.get("point"):
            out.append(model)
    return out


_LOADED: dict[str, tuple[float, list[Any]]] = {}


def current(home: Optional[Path] = None) -> list[Any]:
    """The own models of ``home`` (this machine's workspace by default), loaded once per change of ``fitted.json``."""
    try:
        if home is None:
            from .settings import Settings

            home = Path(Settings().agent_home)
        fitted = root(home) / "models" / "fitted.json"
        stamp = fitted.stat().st_mtime if fitted.is_file() else 0.0
    except Exception:  # noqa: BLE001 - no workspace here: no own model
        return []
    key = str(Path(home).resolve())
    if key not in _LOADED or _LOADED[key][0] != stamp:
        _LOADED[key] = (stamp, load(home) if stamp else [])
    return _LOADED[key][1]


def passed(home: Path) -> dict[str, dict[str, Any]]:
    """The own models whose sealed gate passed in this workspace, by ``name@version`` (as ``ledger.routing.passed``)."""
    out = {}
    for path in sorted((root(home) / "results").glob("*.json")) if (root(home) / "results").is_dir() else []:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("result") == "pass":
            out[f"{doc['model']['name']}@{doc['model']['version']}"] = {"point": doc["point"], "verdict": doc.get("verdict") or {},
                                                                       "gate": doc["gate"]["version"], "family": doc["model"].get("family"),
                                                                       "own": True}
    return out


def rule_answer(row: dict[str, Any]) -> Any:
    """The rule's own answer on a decision (never ``value_used``: after a person's correction that is the person's)."""
    return next((a.get("value") for a in row.get("answers") or [] if a.get("by") == "rule"), None)


def gate(home: Path, points: Optional[list[str]] = None) -> dict[str, Any]:
    """Each own model judged once on the workspace's sealed decisions. A point whose sealed rows were marked read is
    refused (a new judgement needs new sealed decisions: more projects)."""
    from dclab_core import decision_gate

    mine = {m.point: m for m in load(home)}
    out, refused = {}, {}
    marks = root(home) / "gate"
    results = root(home) / "results"
    for point in points or sorted(mine):
        model = mine.get(point)
        if model is None:
            refused[point] = "no own model fitted for this point"
            continue
        sealed = [r for r in rows(home, point) if r["split"] == "sealed"]
        fingerprint = hashlib.sha256("\n".join(sorted(json.dumps({k: r[k] for k in ("record", "state", "label")}, sort_keys=True, default=str)
                                                      for r in sealed)).encode()).hexdigest()
        mark = marks / f"{point}.json"
        read_before = set(json.loads(mark.read_text(encoding="utf-8")).get("records") or []) if mark.is_file() else set()
        if read_before:  # a sealed decision is read once: a later gate reads only the ones never read (new projects)
            sealed = [r for r in sealed if r["record"] not in read_before]
        if not sealed:
            refused[point] = ("its sealed decisions were read: they are read once (more projects make new sealed decisions)" if read_before
                              else "no sealed decision yet: the workspace needs more projects")
            continue
        marks.mkdir(parents=True, exist_ok=True)
        mark.write_text(json.dumps({"fingerprint": fingerprint, "at": _now(), "model": model.version,
                                    "records": sorted(read_before | {r["record"] for r in sealed})}) + "\n", encoding="utf-8")
        answers, errors = [], 0
        for r in sealed:
            try:
                got = model.answer(point, r["state"], list(r.get("options") or []))
                answers.append(None if got is None else got.value)
            except Exception:  # noqa: BLE001 - an error is counted, never skipped
                answers.append(None)
                errors += 1
        verdict = decision_gate.verdict([r["label"]["value"] for r in sealed], answers, [rule_answer(r) for r in sealed],
                                        [r["group"] for r in sealed], errors=errors)
        card = json.loads((root(home) / "models" / f"{point}.json").read_text(encoding="utf-8"))
        doc = {"point": point, "at": _now(), "model": {k: card[k] for k in ("name", "version", "family", "strength", "trained_set")},
               "base": model.base_version, "gate": {"version": decision_gate.VERSION, "thresholds": decision_gate.THRESHOLDS},
               "sealed": {"decisions": len(sealed), "groups": len({r["group"] for r in sealed}), "fingerprint": fingerprint,
                          "read_before": len(read_before)},
               "verdict": verdict, "result": "pass" if verdict["pass"] else "fail",
               "rule": "One run of this workspace's own model on its own sealed decisions, judged by DCLab's decision gate beside the "
                       "rule; evidence about this company's decisions, not a claim of production readiness."}
        results.mkdir(parents=True, exist_ok=True)
        (results / f"{point}.json").write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        out[point] = doc
    return {"judged": out, "refused": refused}


def status(home: Path) -> dict[str, Any]:
    sets = root(home) / "sets"
    points = sorted(p.name for p in sets.iterdir() if p.is_dir()) if sets.is_dir() else []
    out = {}
    for point in points:
        data = rows(home, point)
        card = root(home) / "models" / f"{point}.json"
        result = root(home) / "results" / f"{point}.json"
        out[point] = {"rows": dict(Counter(r["split"] for r in data)),
                      "card": json.loads(card.read_text(encoding="utf-8"))["version"] if card.is_file() else None,
                      "result": json.loads(result.read_text(encoding="utf-8"))["result"] if result.is_file() else None}
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .settings import Settings

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.own_models", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("export", "fit", "gate", "status"))
    parser.add_argument("--home", default=None, help="the workspace folder (default: this machine's workspace)")
    parser.add_argument("--points", default="")
    parser.add_argument("--yes", action="store_true", help="gate: read the sealed decisions (once)")
    args = parser.parse_args(argv)
    home = Path(args.home) if args.home else Path(Settings().agent_home)
    points = [p for p in args.points.split(",") if p] or None
    if args.command == "export":
        print(json.dumps(export(home, points=points)["points"], indent=1))
    elif args.command == "fit":
        got = fit(home, points)
        for point, c in got["cards"].items():
            print(f"{point:<26} {c.family}:{c.strength} dev {c.dev.accuracy if c.dev else None} | rule {c.rule_dev.accuracy if c.rule_dev else None}")
        for point, why in got["skipped"].items():
            print(f"{point:<26} no model: {why}")
    elif args.command == "gate":
        if not args.yes:
            print("Would judge each own model once on this workspace's sealed decisions; add --yes (they are then read).")
            return 0
        got = gate(home, points)
        for point, doc in got["judged"].items():
            print(f"{point:<26} {doc['result']}")
        for point, why in got["refused"].items():
            print(f"{point:<26} refused: {why}")
    else:
        print(json.dumps(status(home), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

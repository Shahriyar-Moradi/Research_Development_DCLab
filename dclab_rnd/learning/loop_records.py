"""The learning loop's records, built from what each seat already wrote (Phase 9, package 31.17).

:func:`build` reads a workspace's attempts (``learning_attempts/*/*``: the pilots', the probes' and, since 31.17, the
baselines'), the stored results (``LRN-*``: a baseline's rows, the attempts behind its scoreboard row), the reflections
(``learning/drafts.jsonl``) and the playbooks' change logs (``playbook/<family>/<lineage>.deltas.jsonl``), and joins
them into :class:`dclab_core.loop_record.LoopRecord` rows by their ids: the pack, the harness run, the bullet. Nothing is
cut: a record points at the whole trace where it is kept. :func:`write` puts them in the lake's bronze layer
(``lake/bronze/loop/<family>/records.jsonl``) with a manifest; the sources stay where they are, so a rebuild gives the
same records.

``python -m dclab_rnd.learning loop-records [--write]`` builds them and prints what :func:`dclab_core.loop_record.replay`
finds.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dclab_core import loop_record as L

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "learning_v1" / "results"
LAKE = ROOT / "lake"
CONSENT = "invented tables: no consent needed"
FOLDER = re.compile(r"^(?P<time>\d{6})-(?P<pack>pk-[0-9a-f]+)-(?P<kind>pilot|probe|baseline|draft)(?:-(?P<bullet>[0-9a-f]+)-(?P<arm>with|without)-r(?P<repeat>\d+))?$")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a torn line never stops the build
    return out


def logs(home: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Every playbook's change log, as written: (family, lineage) to its deltas in order."""
    out = {}
    for path in sorted((Path(home) / "playbook").glob("*/*.deltas.jsonl")):
        out[(path.parent.name, path.name.removesuffix(".deltas.jsonl"))] = _jsonl(path)
    return out


def _grade(doc: Any, sha: str | None = None) -> L.GradeRef | None:
    if not isinstance(doc, dict) or doc.get("verdict") not in ("pass", "fail"):
        return None
    return L.GradeRef(grader=str(doc.get("grader") or "grader_v1"), verdict=doc["verdict"], codes=list(doc.get("codes") or []),
                      decision_score=doc.get("decision_score"), sha256=sha)


def _reads(session: dict[str, Any] | None) -> L.Reads | None:
    book = (session or {}).get("playbook")
    if not isinstance(book, dict) or book.get("version") is None:
        return None
    sha = str(book.get("sha256") or "")
    return L.Reads(lineage=str(book.get("lineage") or "main"), version=int(book["version"]), shown=list(book.get("shown") or []),
                   used=list(book.get("used") or []), text_sha256=sha if re.fullmatch(r"[0-9a-f]{8,64}", sha) else None)


def _kept_attempts(home: Path) -> Iterable[dict[str, Any]]:
    """The attempts kept in folders: one entry an arm, with what the folder says."""
    for folder in sorted((Path(home) / "learning_attempts").glob("*/*")):
        m = FOLDER.match(folder.name)
        if not folder.is_dir() or not m:
            continue
        rel = str(folder.relative_to(home))
        invalid = _json(folder / "invalid.json")
        at = f"{folder.parent.name}T{m['time'][:2]}:{m['time'][2:4]}:{m['time'][4:]}+00:00"
        base = {"folder": folder, "id": rel, "pack": m["pack"], "kind": m["kind"], "at": at, "valid": not invalid,
                "why_invalid": (invalid or {}).get("why") if invalid else None}
        if m["kind"] in ("baseline", "draft"):
            attempt, graded = _json(folder / "attempt.json") or {}, _json(folder / "grade.json") or {}
            pack = _json(folder / "pack.json") or {}
            for arm, doc in graded.items():
                yield {**base, "arm": arm, "split": attempt.get("partition") or "sealed", "run": attempt.get("run"),  # unsaid: an exam's
                       "producer": "rule:dclab" if arm == "rules" else "rule:reference", "grade": _grade(doc.get("grade"), _sha(folder / "grade.json")),
                       "key_sha256": attempt.get("key_sha256"), "table_sha256": (pack.get("table") or {}).get("sha256") or pack.get("table_sha256"),
                       "trace": [f"{rel}/grade.json", f"{rel}/pack.json"]}
            continue
        row = _json(folder / ("probe_row.json" if m["kind"] == "probe" else "pilot_row.json")) or {}
        sessions = sorted((folder / "intern").glob("*.json"))
        session = _json(sessions[0]) if sessions else None
        graded = _json(folder / "grade.json") or {}
        model = row.get("model") or (session or {}).get("model") or "unknown"
        yield {**base, "arm": m["arm"] or row.get("arm"), "split": "learning", "run": None, "tag": row.get("tag"),
               "producer": f"model:{model}", "grade": _grade(graded.get("grade"), _sha(folder / "grade.json")), "reads": _reads(session),
               "key_sha256": None, "table_sha256": None, "counted": _json(folder / "probe_counted.json"),
               "trace": [f"{rel}/{p.name}" for p in sorted(folder.iterdir())]}


def _result_rows(results: Path) -> Iterable[dict[str, Any]]:
    """A stored baseline's rows: the attempts behind its scoreboard row (before 31.17 these were not kept in folders)."""
    for path in sorted(Path(results).glob("LRN-*.json")):
        doc = _json(path)
        if not isinstance(doc, dict) or doc.get("kind") != "learning_baseline":
            continue
        for r in doc.get("rows") or []:
            for arm, a in (r.get("arms") or {}).items():
                yield {"result": doc.get("id"), "pack": r.get("pack"), "split": doc.get("split"), "run": r.get("run"), "arm": arm,
                       "kept": r.get("kept"), "at": doc.get("at"), "family": doc.get("family"),
                       "grade": _grade({**a, "grader": "grader_v1"}), "error": r.get("error")}


def build(home: Path, results: Path = RESULTS) -> tuple[list[L.LoopRecord], dict[tuple[str, str], list[dict[str, Any]]]]:
    home = Path(home)
    changes = logs(home)
    drafts = defaultdict(list)
    for d in _jsonl(home / "learning" / "drafts.jsonl"):
        drafts[(d.get("pack"), d.get("run"))].append(d)
    adds = defaultdict(list)  # (family, pack, run) -> the add deltas whose evidence names that attempt
    counts = defaultdict(list)  # (family, bullet, at) -> the count deltas a probe's tally wrote
    counts_by_index = defaultdict(list)  # (family, lineage, index) -> the same, by the place the tally names
    for (family, lineage), log in changes.items():
        for i, d in enumerate(log):
            ref = L.DeltaRef(lineage=lineage, index=i, op=str(d.get("op")), by=str(d.get("by") or "unknown")[:120], at=d.get("at"),
                             bullet=(d.get("bullet") or {}).get("id") if d.get("op") == "add" else d.get("target"))
            if d.get("op") == "add":
                for e in (d.get("bullet") or {}).get("evidence") or []:
                    adds[(family, e.get("pack"), e.get("run"))].append(ref)
            elif d.get("op") == "count":
                counts[(family, d.get("target"), d.get("at"))].append(ref)
                counts_by_index[(family, lineage, i)].append(ref)

    records: list[L.LoopRecord] = []
    seen_runs = set()

    def reflection(family: str, pack: str, run: str | None) -> tuple[L.Reflection | None, list[L.DeltaRef]]:
        found = drafts.get((pack, run)) or []
        if not found:
            return None, []
        d = found[-1]
        ref = L.Reflection(step=d.get("step"), filed=list(d.get("filed") or []), refused=list(d.get("refused") or []),
                           why=str(d.get("why") or "")[:600], producer=str(d.get("by") or "rule:reflect_v1")[:120], at=d.get("at"))
        mine = adds.get((family, pack, run)) or []
        return ref, [x for x in mine if x.bullet in ref.filed]

    def record(family: str, rid: str, pack: dict[str, Any], attempt: L.AttemptRef, grade: L.GradeRef | None, result: str | None,
               ref: L.Reflection | None, deltas: list[L.DeltaRef]) -> L.LoopRecord:
        exam = pack["split"] == "sealed"
        return L.LoopRecord(id=rid, family=family, pack=L.PackRef(family=family, **pack), attempt=attempt, grade=grade, reflection=ref,
                            deltas=deltas, result=result, producer=attempt.producer, consent=CONSENT, exam=exam,
                            training_allowed=pack["split"] == "learning" and attempt.producer != "rule:reference")

    for a in _kept_attempts(home):
        family = "groups"  # every kept attempt so far is the groups family's; the pack's own family when a pack says it
        if a.get("folder") is not None and (pack_doc := _json(a["folder"] / "pack.json")):
            family = str(pack_doc.get("family") or family)
        if a.get("run"):
            seen_runs.add((a["pack"], a["run"], a["arm"]))
        ref, deltas = reflection(family, a["pack"], a.get("run"))
        counted = a.get("counted") or {}
        if counted:  # by the delta the tally names (since 31.17), else by its bullet and time as before
            if counted.get("delta") is not None:
                lineage = str(counted.get("lineage") or "main")
                deltas = deltas + [x for x in counts_by_index.get((family, lineage, int(counted["delta"])), []) if x.bullet == counted.get("bullet")]
            else:
                deltas = deltas + counts.get((family, counted.get("bullet"), counted.get("at")), [])
        attempt = L.AttemptRef(id=a["id"], kind=a["kind"], producer=a["producer"], arm=a.get("arm"), tag=a.get("tag"), run=a.get("run"),
                               trace=a["trace"], valid=a["valid"], why_invalid=(a.get("why_invalid") or None) and str(a["why_invalid"])[:600],
                               at=a["at"], reads=a.get("reads"), counted=(counted.get("bullet") if counted else None))
        records.append(record(family, f"{a['kind']}:{a['id']}:{a.get('arm') or '-'}", {"id": a["pack"], "split": a["split"],
                              "table_sha256": a.get("table_sha256"), "key_sha256": a.get("key_sha256")}, attempt, a["grade"], None, ref, deltas))
    covered = {(r.pack.id, r.attempt.run) for r in records}
    for (pack, run), found in sorted(drafts.items(), key=lambda kv: str(kv[0])):
        if (pack, run) in covered or not str(pack or "").startswith("pk-"):
            continue  # its attempt is kept in a folder (since 31.17); before that the note is all there is
        d = found[-1]
        family = str(d.get("family") or "groups")
        ref, deltas = reflection(family, pack, run)
        codes = list(d.get("codes") or [])
        grade = L.GradeRef(grader="grader_v1", verdict="fail" if codes else "pass", codes=codes)  # a check that fails writes a code
        attempt = L.AttemptRef(id=f"learning/drafts.jsonl/{pack}", kind="draft", producer="rule:dclab", arm="rules", run=run,
                               trace=["learning/drafts.jsonl"], at=d.get("at"))
        records.append(record(family, f"draft:{pack}:rules", {"id": pack, "split": "learning"}, attempt, grade, None, ref, deltas))
    by_id = {r.id: i for i, r in enumerate(records)}
    for r in _result_rows(results):
        if not r["pack"] or not str(r["pack"]).startswith("pk-"):
            continue  # a broken attempt's row has no pack: its error is in the result file
        family = str(r.get("family") or "groups")
        kept = f"baseline:{r['kept']}:{r['arm']}" if r.get("kept") else None
        if kept in by_id:  # kept in a folder: the folder's record is the attempt, the result only counts it
            records[by_id[kept]] = records[by_id[kept]].model_copy(update={"result": r["result"]})
            continue
        ref, deltas = reflection(family, r["pack"], r.get("run"))
        attempt = L.AttemptRef(id=f"{r['result']}/{r['pack']}", kind="baseline", producer="rule:dclab" if r["arm"] == "rules" else "rule:reference",
                               arm=r["arm"], run=r.get("run"), trace=[f"evidence/campaigns/learning_v1/results/{r['result']}*.json"], valid=not r.get("error"),
                               at=r.get("at"))
        records.append(record(family, f"baseline:{r['result']}/{r['pack']}:{r['arm']}", {"id": r["pack"], "split": r["split"]},
                              attempt, r["grade"], r["result"], ref, deltas))
    return records, changes


def write(records: list[L.LoopRecord], changes: dict[tuple[str, str], list[dict[str, Any]]], lake: Path = LAKE) -> list[Path]:
    """The records in the lake's bronze layer, one file a family, with a manifest: the same sources give the same files."""
    out = []
    by_family = defaultdict(list)
    for r in records:
        by_family[r.family].append(r)
    for family, rows in sorted(by_family.items()):
        folder = Path(lake) / "bronze" / "loop" / family
        folder.mkdir(parents=True, exist_ok=True)
        body = "".join(r.model_dump_json() + "\n" for r in sorted(rows, key=lambda r: r.id))
        path = folder / "records.jsonl"
        tmp = path.with_suffix(".jsonl.part")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
        manifest = {"schema": "dclab_core.loop_record.v1", "family": family, "records": len(rows), "sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "exam": sum(r.exam for r in rows), "training_allowed": len(L.training(rows)),
                    "lineages": sorted(lin for fam, lin in changes if fam == family), "problems": L.replay(rows, {k: v for k, v in changes.items() if k[0] == family}),
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (folder / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        out.append(path)
    return out


__all__ = ["build", "write", "logs", "LAKE", "RESULTS"]

"""36.7: every trained version with what it was trained on and its exam, and its roll-back (dclab_core.model_version).

``register`` keeps a version once: its own name, the name the tuned tier serves it under, the replay round it was trained
on (36.5), which must rebuild byte for byte first, and the frozen exam it sits (36.4) with that exam's own hash.
``standing`` joins every decision of its exam (accepted, refused, retired, with the sections it fell on) into its history.
``retire`` is one change: every role the tuned tier serves under its served name goes back to the model before it
(``models.tuned.retire``) and its shadows end, then its exam record says retired (once), then its own history. The roles
of a retirement that stopped half way are kept beside the record, so running it again finishes it and names them all; a
failed switch leaves nothing in the version's record. ``models.tuned.plan`` refuses to serve a retired version again.

    python -m dclab_rnd.store.versions register --model-type m4_workflow_llm --version NAME --served-as SERVED --base BASE --round 1 --reason "..."
    python -m dclab_rnd.store.versions status --model-type m4_workflow_llm [--version NAME]
    python -m dclab_rnd.store.versions retire --model-type m4_workflow_llm --version NAME --reason "..." --home WORKSPACE
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from dclab_core import replay as R
from dclab_core.model_version import TrainedVersion, VersionEvent
from dclab_core.store import MODEL_TYPES

from .layout import Layout
from .store import now


def path(layout: Layout, model_type: str, name: str) -> Path:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-._")[:80] or "version"  # the real name is inside the file
    return layout.versions(model_type, f"{slug}-{hashlib.sha256(name.encode()).hexdigest()[:8]}.json")


def _save(layout: Layout, v: TrainedVersion) -> TrainedVersion:
    target = path(layout, v.model_type, v.version)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".write-", suffix=".part")  # its own temporary file: two writers never share one
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(v.model_dump_json(indent=1) + "\n")
    os.replace(tmp, target)
    return v


def _exam(layout: Layout, model_type: str, named: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """The exam the version sits and that exam's own hash: the one its round named (refused if missing), else the type's."""
    from . import exams

    if named:
        return named, exams.load(layout, model_type, named).hash  # raises when the named exam is missing or changed
    found = sorted((e for e in exams.all_exams(layout) if e.model_type == model_type), key=lambda e: e.version)
    return (found[-1].version, found[-1].hash) if found else (None, None)


def register(layout: Optional[Layout], model_type: str, name: str, base_model: str, round_: int, *, by: str, reason: str,
             served_as: Optional[str] = None) -> TrainedVersion:
    """Keep a trained version once, with its served name, its round (which must rebuild byte for byte) and its exam."""
    from . import replay

    layout = layout or Layout()
    target = path(layout, model_type, name)
    folder = layout.rounds(model_type, f"round_{round_:03d}")
    if not (folder / "MANIFEST.json").exists():
        raise ValueError(f"round {round_} of {model_type} does not exist: a version names the data it was trained on")
    if not replay.rebuild(layout, model_type, round_)["identical"]:
        raise ValueError(f"round {round_} of {model_type} does not rebuild byte for byte: its data cannot be named as trained on")
    rnd = R.ReplayRound.model_validate_json((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    exam_version, exam_hash = _exam(layout, model_type, rnd.exam_version)
    v = TrainedVersion(model_type=model_type, version=name, served_as=served_as or name, base_model=base_model, round=round_,
                       round_train_sha256=rnd.train_sha256, replay_share=rnd.share, sets=rnd.sets, exam_version=exam_version, exam_hash=exam_hash,
                       history=[VersionEvent(at=now(), status="trained", by=by, reason=reason)])
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target, "x", encoding="utf-8") as fh:  # created once: two registrations at once cannot both write
            fh.write(v.model_dump_json(indent=1) + "\n")
    except FileExistsError:
        raise ValueError(f"{name} of {model_type} is registered already: a version's record is never rewritten") from None
    return v


def _read(layout: Layout, model_type: str, name: str) -> TrainedVersion:
    target = path(layout, model_type, name)
    if not target.exists():
        raise ValueError(f"{name} of {model_type} was never registered")
    return TrainedVersion.read(target.read_text(encoding="utf-8"))


def _exam_rows(layout: Layout, model_type: str, v: TrainedVersion) -> list[dict[str, Any]]:
    from . import exams

    return [r for r in exams.versions(layout, model_type, v.exam_version or exams.EXAM_VERSION) if r.get("model_version") == v.version]


def standing(layout: Optional[Layout], model_type: str, name: str) -> TrainedVersion:
    """The version with every decision of its exam joined into its history, once each, in time order."""
    layout = layout or Layout()
    v = _read(layout, model_type, name)
    known = {(e.status, e.at) for e in v.history}
    joined = [VersionEvent(at=r["at"], status=r["status"], by=str(r.get("by") or "gate"), reason=str(r.get("reason") or ""),
                           forgotten=[str(x) for x in r.get("forgotten") or []])
              for r in _exam_rows(layout, model_type, v) if r.get("status") in ("accepted", "refused", "retired") and (r["status"], r["at"]) not in known]
    if joined:
        v = _save(layout, v.model_copy(update={"history": sorted([*v.history, *joined], key=lambda e: e.at)}))
    return v


def served_roles(home: Path, served: str) -> tuple[list[str], list[str]]:
    """The roles the tuned tier serves under this served name, and those it shadows while it points at that name."""
    from ..models import settings
    from ..models.routing import open_routing, route

    doc = open_routing(home).get()
    purposes = sorted(doc.get("purposes") or {})
    serving = [p for p in purposes if route(doc, p).get("serve") == settings.TUNED and route(doc, p).get("serve_model") == served]
    shadowing = [p for p in purposes if route(doc, p).get("shadow") == settings.TUNED] if settings.tier(settings.TUNED).get("model") == served else []
    return serving, shadowing


def is_retired(layout: Optional[Layout], served: str) -> bool:
    """Whether a version served under this name was retired (``models.tuned.plan`` never serves it again)."""
    layout = layout or Layout()
    for mt in MODEL_TYPES:
        base = layout.versions(mt)
        for p in base.glob("*.json") if base.exists() else []:
            if p.name.endswith(".retiring.json"):
                continue
            v = TrainedVersion.read(p.read_text(encoding="utf-8"))
            if v.served_as == served and v.status == "retired":
                return True
    return False


def retire(layout: Optional[Layout], model_type: str, name: str, reason: str, home: Path, *, by: str, audit_log: Any = None) -> TrainedVersion:
    """One change: the roles it serves go back and its shadows end, then its exam record says retired (once), then its own
    history. Roles switched by an earlier try that stopped half way are kept beside the record and named at the end."""
    from ..models import tuned
    from . import exams

    layout = layout or Layout()
    v = standing(layout, model_type, name)
    if v.status == "retired":
        return v
    note = path(layout, model_type, name).with_suffix(".retiring.json")
    done = json.loads(note.read_text(encoding="utf-8")) if note.exists() else []
    serving, shadowing = served_roles(home, v.served_as)
    who = "owner" if by == "owner" else by
    for role in serving:  # first: nothing keeps serving a version that forgot
        tuned.retire(role, home, reason, audit_log=audit_log, role=who)
        done = sorted({*done, role})
        note.write_text(json.dumps(done) + "\n", encoding="utf-8")
    for role in shadowing:  # nor keeps shadowing beside a role (its old agreement must not count again)
        tuned.shadow(role, home, audit_log=audit_log, role=who, on=False)
    rows = _exam_rows(layout, model_type, v)
    if rows and rows[-1].get("status") == "accepted":  # still current on its exam: retired there once, never twice
        exams.retire(layout, model_type, name, reason, by=by, version=v.exam_version or exams.EXAM_VERSION)
    v = standing(layout, model_type, name)
    if v.status != "retired":
        v = _save(layout, v.model_copy(update={"history": [*v.history, VersionEvent(at=now(), status="retired", by=by, reason=reason, roles=done)]}))
    elif done and not v.history[-1].roles:
        v = _save(layout, v.model_copy(update={"history": [*v.history[:-1], v.history[-1].model_copy(update={"roles": done})]}))
    note.unlink(missing_ok=True)
    return v


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.store.versions", description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("register", "status", "retire"))
    parser.add_argument("--model-type", required=True, choices=sorted(MODEL_TYPES))
    parser.add_argument("--version")
    parser.add_argument("--served-as", help="register: the name the tuned tier serves it under (default: the version's name)")
    parser.add_argument("--base", default="")
    parser.add_argument("--round", type=int)
    parser.add_argument("--reason", default="")
    parser.add_argument("--home", type=Path, help="retire: the workspace whose routing serves the version")
    args = parser.parse_args(argv)
    layout = Layout()
    if args.action == "status":
        names = [args.version] if args.version else [TrainedVersion.read(p.read_text(encoding="utf-8")).version
                                                     for p in sorted(layout.versions(args.model_type).glob("*.json")) if not p.name.endswith(".retiring.json")]
        for n in names:
            v = standing(layout, args.model_type, n)
            print(json.dumps({"version": v.version, "served_as": v.served_as, "status": v.status, "round": v.round, "exam": v.exam_version}))
        return 0
    if not args.version or not args.reason:
        print("Name the version and give a reason.", file=sys.stderr)
        return 2
    if args.action == "register":
        if args.round is None:
            print("Name the round it was trained on (--round).", file=sys.stderr)
            return 2
        print(register(layout, args.model_type, args.version, args.base, args.round, by="owner", reason=args.reason,
                       served_as=args.served_as).model_dump_json(indent=1))
        return 0
    if args.home is None:
        print("Name the workspace whose routing serves it (--home).", file=sys.stderr)
        return 2
    print(retire(layout, args.model_type, args.version, args.reason, args.home, by="owner").model_dump_json(indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

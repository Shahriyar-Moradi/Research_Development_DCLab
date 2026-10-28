"""36.5: a training round's set, new rows plus a share of every earlier round (dclab_core.replay).

A row is its set and its id (the same id in two sets is two rows). A round reads the train split of a model type's
latest sets (never dev or sealed). Its new rows are those no earlier round holds with the same content; they are kept in
``new.jsonl`` with their set, their row key and their stratum (frozen there, so a later change of how strata are read
never moves an old draw), and never rewritten. Its train set is its new rows plus, from each earlier round, the share
drawn stratum by stratum among the rows whose latest version is that round's, that are still in the current train split
and outside every exam, and that are not new again. Rows no longer allowed are written to ``withheld.jsonl``, so a
rebuild reads the round's own files only. Earlier rounds must run 1..n-1 without a gap and match the hashes this round
recorded; a build is written to a ``.part`` folder and renamed when whole.

    python -m dclab_rnd.store.replay plan [--model-type m4_workflow_llm]   # what the next round would hold; nothing written
    python -m dclab_rnd.store.replay build --model-type m4_workflow_llm --yes
    python -m dclab_rnd.store.replay rebuild --model-type m4_workflow_llm --round 2
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

from dclab_core import replay as R
from dclab_core.store import MODEL_TYPES

from .layout import Layout
from .store import now, write_json

Row = tuple[str, str, str]  # (row key "set|id", stratum, the line exactly as its set wrote it)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lines(text: str) -> list[str]:
    return [x for x in text.split("\n") if x.strip()]  # a JSON line may hold U+2028: split on newlines only


def _stratum(line: str, folder: str) -> str:
    """A train line's stratum: its set, its kind of help, its language (read as the coverage report reads them)."""
    from .coverage import _example_language

    row = json.loads(line)
    check = row.get("check") if isinstance(row.get("check"), dict) else {}
    return R.stratum(folder, check.get("track") or row.get("track"), _example_language(row))


def current(layout: Layout, model_type: str) -> tuple[list[Row], list[dict[str, str]]]:
    """The train lines of the model type's latest sets, each under its row key, sorted, and the sets read."""
    from .coverage import _latest_sets

    base = layout.sets(model_type)
    rows: list[Row] = []
    sets = []
    for folder in _latest_sets(layout, model_type):
        name = folder.parent.relative_to(base).as_posix()
        sets.append({"folder": name, "version": folder.name, "manifest_sha256": _sha((folder / "MANIFEST.json").read_text(encoding="utf-8"))})
        train = folder / "train.jsonl"
        for line in _lines(train.read_text(encoding="utf-8")) if train.exists() else []:
            rows.append((f"{name}|{json.loads(line)['id']}", _stratum(line, name), line))
    return sorted(rows), sets


def rounds(layout: Layout, model_type: str) -> list[tuple[int, Path]]:
    base = layout.rounds(model_type)
    found = [(int(m.group(1)), p) for p in base.glob("round_*") if p.is_dir() and (m := re.fullmatch(r"round_(\d+)", p.name))] if base.exists() else []
    return sorted(found)


def _wrap(rows: list[Row]) -> str:
    return "".join(json.dumps({"row": k, "stratum": s, "line": line}, ensure_ascii=False) + "\n" for k, s, line in rows)


def _unwrap(text: str) -> list[Row]:
    return [(d["row"], d["stratum"], d["line"]) for d in (json.loads(x) for x in _lines(text))]


def _earlier(layout: Layout, model_type: str, before: int, expect: Optional[dict[int, str]] = None) -> list[tuple[int, str, list[Row]]]:
    """Rounds 1..before-1, each checked: present without a gap, its new rows matching its own manifest and, on a rebuild,
    the hash this round recorded for it. A missing, changed or swapped round is refused, never read."""
    found = dict(rounds(layout, model_type))
    out = []
    for n in range(1, before):
        folder = found.get(n)
        if folder is None:
            raise ValueError(f"round_{n:03d} of {model_type} is missing: rounds run without a gap, so nothing is built or rebuilt")
        text = (folder / "new.jsonl").read_text(encoding="utf-8")
        manifest = R.ReplayRound.model_validate_json((folder / "MANIFEST.json").read_text(encoding="utf-8"))
        sha = _sha(text)
        if sha != manifest.new_sha256 or (expect is not None and expect.get(n) != sha):
            raise ValueError(f"{folder.name} of {model_type} changed after it was built: its new rows no longer match its record")
        out.append((n, sha, _unwrap(text)))
    return out


def _assemble(earlier: list[tuple[int, str, list[Row]]], new: list[Row], withheld: set[str], n: int, share: float,
              salt: str) -> tuple[str, list[R.ReplayDraw], dict[str, int]]:
    """The train set: the new rows, then from each earlier round the drawn rows among those whose latest version is that
    round's (an older copy of a row never comes back), not new again, and not withheld. Every drawn row is written."""
    latest = {k: m for m, _, rows in earlier for k, _, _ in rows}  # later rounds overwrite: the latest round holding each row
    fresh = {k for k, _, _ in new}
    chosen: dict[str, tuple[str, str]] = {k: (s, line) for k, s, line in new}
    draws = []
    for m, sha, rows in earlier:
        pool = [(k, s, line) for k, s, line in rows if latest[k] == m and k not in fresh and k not in withheld]
        picked = R.pick(((k, s) for k, s, _ in pool), share, salt, n)
        by: dict[str, list[int]] = {}
        for k, s, line in pool:
            c = by.setdefault(s, [0, 0])
            c[0] += 1
            if k in picked:
                c[1] += 1
                chosen[k] = (s, line)
        draws.append(R.ReplayDraw(round=m, new_sha256=sha, available=len(pool), drawn=sum(v[1] for v in by.values()), by_stratum=dict(sorted(by.items()))))
    text = "".join(chosen[k][1] + "\n" for k in sorted(chosen))  # each line exactly as its set wrote it
    strata: dict[str, int] = {}
    for s, _ in chosen.values():
        strata[s] = strata.get(s, 0) + 1
    return text, draws, dict(sorted(strata.items()))


def _next(layout: Layout, model_type: str, share: float, salt: str) -> dict[str, Any]:
    from .exams import guard

    rows, sets = current(layout, model_type)
    done = rounds(layout, model_type)
    n = (done[-1][0] + 1) if done else 1
    earlier = _earlier(layout, model_type, n)
    seen = {_sha(line) for _, _, old in earlier for _, _, line in old}
    new = [(k, s, line) for k, s, line in rows if _sha(line) not in seen]  # in no earlier round with this content
    allowed = {k for k, _, _ in rows}
    exams = guard(layout)
    withheld = sorted({k for _, _, old in earlier for k, _, line in old  # gone from the train split, or in an exam now
                       if k not in allowed or (exams and exams.why(json.loads(line)))})
    text, draws, strata = _assemble(earlier, new, set(withheld), n, share, salt)
    return {"round": n, "rows": rows, "sets": sets, "new": new, "withheld": withheld, "text": text, "draws": draws, "strata": strata}


def plan(layout: Optional[Layout] = None, model_type: str = "m4_workflow_llm", share: float = R.SHARE, salt: str = R.SALT) -> dict[str, Any]:
    """What the next round would hold; nothing is written."""
    p = _next(layout or Layout(), model_type, share, salt)
    return {"model_type": model_type, "round": p["round"], "current_train": len(p["rows"]), "new": len(p["new"]), "withheld": len(p["withheld"]),
            "replayed": [d.model_dump() for d in p["draws"]], "train": p["text"].count("\n"), "strata": len(p["strata"])}


def build(layout: Optional[Layout] = None, model_type: str = "m4_workflow_llm", share: float = R.SHARE, salt: str = R.SALT,
          exam_version: Optional[str] = None) -> R.ReplayRound:
    """Write the next round (its new rows, the rows withheld, its train set, its manifest) into a ``.part`` folder renamed
    when whole. Earlier rounds are read, never changed; two builds at once cannot both write the same round."""
    layout = layout or Layout()
    p = _next(layout, model_type, share, salt)
    final = layout.rounds(model_type, f"round_{p['round']:03d}")
    part = final.with_name(final.name + ".part")
    try:
        part.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError(f"{part.name} exists: another build is running, or one stopped half way (remove the folder to build again)") from None
    new_text, withheld_text = _wrap(p["new"]), "".join(k + "\n" for k in p["withheld"])
    (part / "new.jsonl").write_text(new_text, encoding="utf-8")
    (part / "withheld.jsonl").write_text(withheld_text, encoding="utf-8")
    (part / "train.jsonl").write_text(p["text"], encoding="utf-8")
    manifest = R.ReplayRound(model_type=model_type, round=p["round"], share=share, salt=salt, rule=R.RULE, sets=p["sets"], new=len(p["new"]),
                             new_sha256=_sha(new_text), replayed=p["draws"], withheld=len(p["withheld"]), withheld_sha256=_sha(withheld_text),
                             strata=p["strata"], train=p["text"].count("\n"), train_sha256=_sha(p["text"]), exam_version=exam_version, built_at=now())
    write_json(part / "MANIFEST.json", json.loads(manifest.model_dump_json()))
    part.rename(final)  # fails if the round appeared meanwhile: a round is written once
    return manifest


def rebuild(layout: Optional[Layout] = None, model_type: str = "m4_workflow_llm", round_: int = 1) -> dict[str, Any]:
    """The round's train set made again from its own files and the earlier rounds' (never from the current sets)."""
    layout = layout or Layout()
    folder = layout.rounds(model_type, f"round_{round_:03d}")
    manifest = R.ReplayRound.model_validate_json((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    new_text = (folder / "new.jsonl").read_text(encoding="utf-8")
    withheld_text = (folder / "withheld.jsonl").read_text(encoding="utf-8")
    if _sha(new_text) != manifest.new_sha256 or _sha(withheld_text) != manifest.withheld_sha256:
        raise ValueError(f"{folder.name} of {model_type} changed after it was built: its files no longer match its manifest")
    earlier = _earlier(layout, model_type, round_, {d.round: d.new_sha256 for d in manifest.replayed})
    text, _, _ = _assemble(earlier, _unwrap(new_text), set(_lines(withheld_text)), round_, manifest.share, manifest.salt)
    kept = (folder / "train.jsonl").read_text(encoding="utf-8")
    return {"round": round_, "identical": _sha(text) == manifest.train_sha256 == _sha(kept)}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.store.replay", description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("plan", "build", "rebuild"))
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES))
    parser.add_argument("--round", type=int)
    parser.add_argument("--share", type=float, default=R.SHARE)
    parser.add_argument("--yes", action="store_true", help="build: write the round (a round is a training run's data: build it when a model is trained)")
    args = parser.parse_args(argv)
    if args.action == "plan":
        for mt in [args.model_type] if args.model_type else sorted(MODEL_TYPES):
            print(json.dumps({k: v for k, v in plan(model_type=mt, share=args.share).items() if k != "replayed"}))
        return 0
    if not args.model_type:
        print("Name the model type (--model-type).", file=sys.stderr)
        return 2
    if args.action == "rebuild":
        print(json.dumps(rebuild(model_type=args.model_type, round_=args.round or 1)))
        return 0
    if not args.yes:
        print("A round is the data of one training run: build it when that model is trained. Nothing was written (add --yes).", file=sys.stderr)
        return 2
    print(build(model_type=args.model_type, share=args.share).model_dump_json(indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

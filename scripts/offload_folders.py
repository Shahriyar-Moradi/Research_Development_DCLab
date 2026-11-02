"""Move finished record folders to the external drive: tar, verify every file, index on both sides, read back, only then remove.

    python scripts/offload_folders.py NAME FOLDER [FOLDER ...]            # copy + verify + index (nothing removed)
    python scripts/offload_folders.py NAME FOLDER [FOLDER ...] --remove   # and then remove the verified folders from the Mac

Folders are paths inside the repository (``agent_runs/live_tests/2026-10-25-live-people``). The owner's rules (2026-11-01 and
2026-11-02): never remove anything without verifying it; every index and detail is on the drive too.

* One tar on the drive, ``dclab_offload/<NAME>.tar`` (written as ``.part`` and renamed when whole), every path stored relative to the
  repository so one command puts it back.
* Verified file by file: every regular file in each folder on the Mac is in the tar with the same SHA-256 and size, no file is in
  the tar that is not on the Mac, symbolic links point to the same target, and the tar's own SHA-256 is recorded in
  ``<NAME>.verify.json``.
* ``dclab_offload/OFFLOAD_INDEX.json`` and ``.md`` on the drive and ``agent_runs/OFFLOAD_INDEX.json`` and ``.md`` on the Mac list every
  offload with its folders, sizes, checks and the restore command; both are read back and compared before anything is removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "dclab_offload"
MAC_INDEX = ROOT / "agent_runs"


def sha(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def files_of(folder: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() or p.is_symlink():
            out[str(p.relative_to(ROOT))] = p
    return out


def verify(tar: Path, folders: list[str]) -> dict:
    in_tar: dict[str, tuple] = {}
    with tarfile.open(tar) as t:
        for m in t:
            if m.isfile():
                in_tar[m.name] = ("file", sha(t.extractfile(m)), m.size)
            elif m.islnk():
                target = in_tar.get(m.linkname)
                in_tar[m.name] = target if target else ("link-target-missing", "", 0)
            elif m.issym():
                in_tar[m.name] = ("symlink", m.linkname, 0)
    on_mac: dict[str, Path] = {}
    for f in folders:
        on_mac |= files_of(ROOT / f)
    missing, differs = [], []
    for rel, p in on_mac.items():
        if rel.endswith(".DS_Store"):  # Finder rewrites these while a copy runs
            continue
        got = in_tar.get(rel)

        if got is None:
            missing.append(rel)
        elif p.is_symlink():
            if got[0] != "symlink" or got[1] != os.readlink(p):
                differs.append(rel)
        else:
            with open(p, "rb") as fh:
                if got[0] != "file" or got[1] != sha(fh) or got[2] != p.stat().st_size:
                    differs.append(rel)
    extra = sorted(r for r in set(in_tar) - set(on_mac) if not r.endswith(".DS_Store"))
    with open(tar, "rb") as fh:
        tar_sha = sha(fh)
    ok = not (missing or differs or extra)
    return {"tar": str(tar), "at": now(), "ok": ok, "tar_sha256": tar_sha, "tar_bytes": tar.stat().st_size, "files_on_mac": len(on_mac),
            "files_in_tar": len(in_tar), "missing_from_tar": len(missing), "differs": len(differs), "in_tar_not_on_mac": len(extra),
            "examples": (missing + differs + extra)[:20]}


def render(doc: dict) -> str:
    lines = ["# What was moved to the external drive besides the training snapshots", "",
             f"Written {doc['written']}. Training snapshots have their own index (`training_records/snapshots/SNAPSHOT_LOCATIONS.md`).", "",
             "| Name | Folders | Size | Checked | Restore |", "|---|---|---|---|---|"]
    for r in doc["offloads"]:
        folders = "<br>".join(f"`{f}`" for f in r["folders"])
        lines.append(f"| `{r['name']}` | {folders} | {r['bytes'] / 1e6:.0f} MB | {'ok' if r['ok'] else 'CHECK'} ({r['at'][:10]}, {r['files']} files) | `{r['restore']}` |")
    lines += ["", "Each tar stores paths relative to the repository: run the restore command in the repository folder with the drive plugged in.",
              "The check of each is `<name>.verify.json` beside its tar."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--drive", default="/Volumes/ADATA PRO")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()
    drive = Path(args.drive)
    if not drive.is_dir():
        print(f"refused: {drive} is not connected", file=sys.stderr)
        return 2
    folders = [f.rstrip("/") for f in args.folders]
    for f in folders:
        p = ROOT / f
        if not p.is_dir() or p.is_symlink() or not str(p.resolve()).startswith(str(ROOT)):
            print(f"refused: {f} is not a folder inside the repository", file=sys.stderr)
            return 2
    dest = drive / FOLDER
    dest.mkdir(exist_ok=True)
    tar = dest / f"{args.name}.tar"
    if tar.exists():
        print(f"{tar.name} exists; verifying it as it is", flush=True)
    else:
        part = tar.with_name(tar.name + ".part")
        part.unlink(missing_ok=True)
        print(f"writing {tar.name} ...", flush=True)
        with tarfile.open(part, "w") as out:
            for f in folders:
                out.add(ROOT / f, arcname=f)
        os.replace(part, tar)
    print("verifying every file ...", flush=True)
    v = verify(tar, folders)
    tar.with_suffix(".verify.json").write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    print(f"{v['files_on_mac']} files on the Mac, {v['files_in_tar']} in the tar, missing {v['missing_from_tar']}, different {v['differs']}, "
          f"in the tar only {v['in_tar_not_on_mac']} -> {'OK' if v['ok'] else 'PROBLEM'}", flush=True)
    if not v["ok"]:
        print("stopped: it did not verify; nothing was removed", file=sys.stderr)
        return 1
    old = []
    for place in (MAC_INDEX / "OFFLOAD_INDEX.json", dest / "OFFLOAD_INDEX.json"):
        if place.exists():
            old = json.loads(place.read_text(encoding="utf-8"))["offloads"]
            break
    row = {"name": args.name, "at": v["at"], "folders": folders, "bytes": sum(p.stat().st_size for f in folders for p in files_of(ROOT / f).values() if p.is_file() and not p.is_symlink()),
           "files": v["files_on_mac"], "ok": True, "tar": f"{FOLDER}/{tar.name}", "tar_sha256": v["tar_sha256"],
           "restore": f'tar -xf "{dest}/{tar.name}" -C .'}
    doc = {"written": now(), "offloads": [r for r in old if r["name"] != args.name] + [row]}
    text, md = json.dumps(doc, indent=1) + "\n", render(doc)
    for folder in (MAC_INDEX, dest):
        (folder / "OFFLOAD_INDEX.json").write_text(text, encoding="utf-8")
        (folder / "OFFLOAD_INDEX.md").write_text(md, encoding="utf-8")
    if (MAC_INDEX / "OFFLOAD_INDEX.json").read_bytes() != (dest / "OFFLOAD_INDEX.json").read_bytes() or \
            not any(r["name"] == args.name and r["ok"] for r in json.loads((dest / "OFFLOAD_INDEX.json").read_text(encoding="utf-8"))["offloads"]):
        print("stopped: the index read back from the drive does not list this offload; nothing was removed", file=sys.stderr)
        return 1
    print("index written on both sides and read back", flush=True)
    if args.remove:
        for f in folders:
            shutil.rmtree(ROOT / f)
            print(f"removed {f}", flush=True)
    else:
        print("nothing removed (use --remove)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

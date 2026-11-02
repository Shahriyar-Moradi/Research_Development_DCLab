"""Copy the training snapshots to the external drive, verify every file, index them, and (only then) free the Mac.

    python scripts/offload_snapshots.py                  # copy + verify + index every snapshot not yet on the drive (nothing removed)
    python scripts/offload_snapshots.py --remove         # also remove from the Mac the snapshots older than --keep-hours
    python scripts/offload_snapshots.py --drive "/Volumes/ADATA PRO" --keep-hours 24

The owner's rules (2026-11-01): never remove anything without verifying it; keep 24 hours of snapshots on the Mac.

* Each snapshot becomes one self-contained tar on the drive (``dclab_training_records/training_snapshot_<YYYYMMDD>.tar``),
  written as ``.part`` and renamed when whole. A hard link to another snapshot is stored as a full file.
* Verified file by file: every file of the snapshot's MANIFEST.json is in the tar with the same SHA-256, every file on the
  Mac is in the tar with the same SHA-256, and the whole tar's SHA-256 is recorded in ``<tar>.verify.json``.
* The index (``SNAPSHOT_LOCATIONS.json`` and ``.md``) is written on the Mac and on the drive, where every copy of every
  snapshot is, its check and how to restore it; ``INDEX.json`` and ``README.md`` are copied beside it. Older places in the
  index (the disk image, the first tar) are kept.
* ``--remove`` removes a Mac snapshot only when its tar verified OK now, the index was written on both sides and read back,
  its manifest still matches ``training_records/INDEX.json``, and it is older than ``--keep-hours``. The NEWEST snapshot
  is never removed: ``make archive`` hard-links each new snapshot to it (without it the next archive copies everything).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "training_records"
SNAPSHOTS = RECORDS / "snapshots"
FOLDER = "dclab_training_records"


def sha(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def when(name: str) -> datetime:
    return datetime.strptime(name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_tar(name: str, tar: Path) -> None:
    part = tar.with_name(tar.name + ".part")
    if part.exists():
        part.unlink()
    with tarfile.open(part, "w") as out:
        out.add(SNAPSHOTS / name, arcname=name)
    os.replace(part, tar)


def verify(name: str, tar: Path) -> dict:
    """Every manifest file and every file on the Mac against the tar; the tar's own hash."""
    manifest = json.loads((SNAPSHOTS / name / "MANIFEST.json").read_text(encoding="utf-8"))
    in_tar: dict[str, str] = {}
    with tarfile.open(tar) as t:
        for m in t:
            if m.isfile():
                in_tar[m.name] = sha(t.extractfile(m))
            elif m.islnk():
                in_tar[m.name] = in_tar.get(m.linkname, "link-target-missing")
    on_mac = {str(p.relative_to(SNAPSHOTS)) for p in (SNAPSHOTS / name).rglob("*") if p.is_file()}
    skip = lambda rel: rel.endswith(".DS_Store")  # Finder rewrites these while a copy runs (2026-11-02: the three files that "differed")  # noqa: E731
    missing = [f["path"] for f in manifest["files"] if f"{name}/{f['path']}" not in in_tar and not skip(f["path"])]
    bad = [f["path"] for f in manifest["files"] if in_tar.get(f"{name}/{f['path']}") not in (None, f["sha256"]) and not skip(f["path"])]
    not_in_tar = sorted(r for r in on_mac - set(in_tar) if not skip(r))
    differs = []
    for rel in sorted(r for r in on_mac & set(in_tar) if not skip(r)):
        with open(SNAPSHOTS / rel, "rb") as fh:
            if sha(fh) != in_tar[rel]:
                differs.append(rel)
    with open(tar, "rb") as fh:
        tar_sha = sha(fh)
    ok = not (missing or bad or not_in_tar or differs)
    return {"tar": str(tar), "at": now(), "ok": ok, "tar_sha256": tar_sha, "tar_bytes": tar.stat().st_size,
            "snapshots": {name: {"manifest_files": len(manifest["files"]), "files_on_mac": len(on_mac), "files_in_tar": len([k for k in in_tar if k.startswith(name + "/")]),
                                 "missing_from_tar": len(missing), "checksum_differs_from_manifest": len(bad), "on_mac_not_in_tar": len(not_in_tar),
                                 "differs_from_mac": len(differs), "ok": ok, "examples": (missing + bad + not_in_tar + differs)[:20]}}}


def render(doc: dict) -> str:
    lines = ["# Where every training snapshot is", "",
             f"Written {doc['written']}. One row per snapshot in `training_records/INDEX.json`. The JSON beside this file has the details.", "",
             "| Snapshot | Size | Where | Checked |", "|---|---|---|---|"]
    for r in doc["snapshots"]:
        where = "; ".join(f"{p['where']}: `{p['file']}`" for p in r["places"]) or "**NOT FOUND**"
        lines.append(f"| `{r['name']}` | {r['bytes'] / 1e9:.2f} GB | {where} | {'ok' if r['all_ok'] else 'CHECK'} |")
    lines += ["", "## How to restore", "",
              "From a tar (run in the repository folder, drive plugged in; the tar named in the row above):", "",
              '      tar -xf "/Volumes/ADATA PRO/dclab_training_records/training_snapshot_<YYYYMMDD>.tar" -C training_records/snapshots', "",
              "From the old tar of the five oldest (name the snapshot to take only one):", "",
              '      tar -xf "/Volumes/ADATA PRO/dclab_training_records/training_snapshots_20261018-20261025.tar" -C training_records/snapshots 20261021T202042Z', "",
              "From the disk image (the seven oldest):", "",
              '      hdiutil attach -readonly "/Volumes/ADATA PRO/DCLab/dclab_snapshots.sparsebundle"', "      cp -R /Volumes/<mounted name>/snapshots/20261017T062019Z training_records/snapshots/", "",
              "Each snapshot's `MANIFEST.json` lists every file with its SHA-256; the `.verify.json` files beside the tars are the checks.",
              "`make archive` reads only the newest entry of INDEX.json, so the newest snapshot stays on the Mac."]
    return "\n".join(lines) + "\n"


def write_index(dest: Path, done: dict[str, dict]) -> dict:
    """Update SNAPSHOT_LOCATIONS from the existing one (older places kept), the verified tars of this run and what is on the Mac."""
    index = json.loads((RECORDS / "INDEX.json").read_text(encoding="utf-8"))["snapshots"]
    old = {}
    for path in (SNAPSHOTS / "SNAPSHOT_LOCATIONS.json", dest / "SNAPSHOT_LOCATIONS.json"):
        if path.exists():
            old = {r["name"]: r for r in json.loads(path.read_text(encoding="utf-8"))["snapshots"]}
            break
    rows = []
    for s in index:
        name = s["name"]
        places = [p for p in (old.get(name) or {}).get("places", []) if p["where"] != "Mac"]
        if name in done:
            v, tar = done[name]["verify"], Path(done[name]["tar"])
            places = [p for p in places if p["file"] != f"{FOLDER}/{tar.name}"]
            places.append({"where": "ADATA PRO", "file": f"{FOLDER}/{tar.name}", "kind": "tar", "check": f"tar vs manifest and vs Mac ({v['at']})",
                           "ok": v["ok"], "tar_sha256": v["tar_sha256"],
                           "restore": f'tar -xf "{dest}/{tar.name}" -C training_records/snapshots'})
        if (SNAPSHOTS / name / "MANIFEST.json").exists():
            places.append({"where": "Mac", "file": f"training_records/snapshots/{name}", "kind": "folder", "ok": True,
                           "check": "kept on the Mac: the newest, and the last 24 hours; make archive hard-links each new snapshot to the newest"})
        rows.append({"name": name, "at": s["at"], "bytes": s["bytes"], "manifest_sha256": s["manifest_sha256"], "places": places,
                     "found": bool(places), "all_ok": bool(places) and all(p["ok"] for p in places)})
    doc = {"written": now(), "what": "Every DCLab training snapshot listed in training_records/INDEX.json, where its copies are, their check, and how to restore",
           "snapshots": rows}
    text, md = json.dumps(doc, indent=1) + "\n", render(doc)
    for folder in (SNAPSHOTS, dest):
        (folder / "SNAPSHOT_LOCATIONS.json").write_text(text, encoding="utf-8")
        (folder / "SNAPSHOT_LOCATIONS.md").write_text(md, encoding="utf-8")
    shutil.copy2(RECORDS / "INDEX.json", dest / "INDEX.json")
    shutil.copy2(RECORDS / "README.md", dest / "README.md")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drive", default="/Volumes/ADATA PRO")
    ap.add_argument("--keep-hours", type=float, default=24)
    ap.add_argument("--remove", action="store_true", help="remove verified snapshots older than --keep-hours from the Mac (never the newest)")
    args = ap.parse_args()
    drive = Path(args.drive)
    if not drive.is_dir():
        print(f"refused: {drive} is not connected", file=sys.stderr)
        return 2
    dest = drive / FOLDER
    dest.mkdir(exist_ok=True)
    index = json.loads((RECORDS / "INDEX.json").read_text(encoding="utf-8"))["snapshots"]
    by_name = {s["name"]: s for s in index}
    on_mac = sorted(p.name for p in SNAPSHOTS.iterdir() if p.is_dir() and (p / "MANIFEST.json").exists())
    newest = index[-1]["name"]
    if newest not in on_mac:
        print(f"refused: the newest snapshot {newest} is not on the Mac", file=sys.stderr)
        return 2
    done: dict[str, dict] = {}
    for name in on_mac:
        manifest_sha = hashlib.sha256((SNAPSHOTS / name / "MANIFEST.json").read_bytes()).hexdigest()
        if name not in by_name or by_name[name]["manifest_sha256"] != manifest_sha:
            print(f"refused: {name}'s manifest does not match training_records/INDEX.json", file=sys.stderr)
            return 2
        tar = dest / f"training_snapshot_{name[:8]}.tar"
        check = tar.with_suffix(".verify.json")
        if tar.exists() and check.exists() and name not in json.loads(check.read_text(encoding="utf-8")).get("snapshots", {}):
            tar = dest / f"training_snapshot_{name}.tar"  # a second snapshot of the same day (2026-11-02): its own tar, never the first's
        if tar.exists():
            print(f"{name}: {tar.name} exists; re-verifying it", flush=True)
        else:
            print(f"{name}: writing {tar.name} ...", flush=True)
            make_tar(name, tar)
        print(f"{name}: verifying every file ...", flush=True)
        v = verify(name, tar)
        tar.with_suffix(".verify.json").write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
        s = v["snapshots"][name]
        print(f"{name}: {s['manifest_files']} manifest files, {s['files_in_tar']} in the tar, missing {s['missing_from_tar']}, "
              f"different {s['checksum_differs_from_manifest'] + s['differs_from_mac']}, on the Mac but not in the tar {s['on_mac_not_in_tar']} -> {'OK' if v['ok'] else 'PROBLEM'}", flush=True)
        if not v["ok"]:
            print(f"stopped: {name} did not verify; nothing was removed", file=sys.stderr)
            return 1
        done[name] = {"verify": v, "tar": str(tar)}
    doc = write_index(dest, done)
    # read the index back from the drive, as the owner asked
    back = {r["name"]: r for r in json.loads((dest / "SNAPSHOT_LOCATIONS.json").read_text(encoding="utf-8"))["snapshots"]}
    for name in done:
        row = back.get(name)
        if row is None or not any(p["file"].endswith(Path(done[name]["tar"]).name) and p["ok"] for p in row["places"]):
            print(f"stopped: the index on the drive does not list {name}'s tar; nothing was removed", file=sys.stderr)
            return 1
    if (SNAPSHOTS / "SNAPSHOT_LOCATIONS.json").read_bytes() != (dest / "SNAPSHOT_LOCATIONS.json").read_bytes():
        print("stopped: the index differs between the Mac and the drive; nothing was removed", file=sys.stderr)
        return 1
    print(f"index written on both sides and read back: {len(doc['snapshots'])} snapshots", flush=True)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.keep_hours)
    old = [n for n in on_mac if n != newest and when(n) < cutoff]
    kept = [n for n in on_mac if n not in old]
    print(f"older than {args.keep_hours:g} h (removable, copied and verified): {old or 'none'}; kept on the Mac: {kept}")
    if not args.remove:
        print("nothing removed (add --remove)")
        return 0
    for name in old:
        def writable(func, path, _exc):  # a finished snapshot is read-only (make archive)
            os.chmod(path, 0o700)
            func(path)
        shutil.rmtree(SNAPSHOTS / name, onerror=writable)
        print(f"removed {name} from the Mac", flush=True)
    if old:
        write_index(dest, {})  # the Mac places follow what is left
        print("index rewritten without the removed Mac copies", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

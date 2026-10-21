"""Build the open-source tree, `dclabai`, from this private repository (the plan: ~/.claude/plans/async-skipping-firefly.md).

    .venv/bin/python scripts/export_open.py --out ../dclabai-export          # build the tree and check it
    .venv/bin/python scripts/export_open.py --out ../dclabai-export --check  # check only (no tests), for a quick look

What it does, in order (nothing is published, nothing in this repository changes):
1. Takes the files git tracks (`git ls-files`: an untracked file, a .env, a record can never reach the export), keeps those the
   manifest's INCLUDE globs match and EXCLUDE globs do not (export/manifest.py).
2. Copies them into a fresh folder, moving `dclab_rnd/` to `dclabai/` and rewriting every whole-word `dclab_rnd` (imports,
   strings, docs) to `dclabai`; strips `# dclab:closed-begin` … `# dclab:closed-end` blocks; lays export/overlay/ on top.
3. Checks, each blocking: the private words (export/forbidden.txt), secrets (scripts/scan_secrets.py's patterns), file sizes and
   binaries, a leftover closed marker, and the import closure (every `dclabai.*` / `dclab_core.*` module an exported file imports
   must exist in the tree, unless it is named inside `optional(...)` and listed in the manifest's CLOSED, or imported inside
   `try: ... except ModuleNotFoundError`).
4. Writes EXPORT_REPORT.json beside the tree (files, sizes, every finding) and prints a summary; exits 1 when a check failed.

The tests, the wheel, the fresh-venv install and the end-to-end flows run on the exported tree in later steps of the plan.
"""

from __future__ import annotations

import argparse
import os
import ast
import fnmatch
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSED_BEGIN, CLOSED_END = "# dclab:closed-begin", "# dclab:closed-end"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".log", ".csv", ".toml", ".cfg", ".ini", ".json", ".jsonl", ".yml", ".yaml", ".html", ".js", ".css", ".sh", ".mako", ".ipynb", ""}


def manifest():
    spec = importlib.util.spec_from_file_location("export_manifest", ROOT / "export" / "manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _match(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) or (g.endswith("/**") and path.startswith(g[:-3] + "/")) for g in globs)


def selected(m) -> list[str]:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    return sorted(p for p in tracked if p and _match(p, m.INCLUDE) and not _match(p, m.EXCLUDE) and (ROOT / p).is_file())


def _module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def selected_tests(m, files: list[str]) -> tuple[list[str], dict[str, str]]:
    """The test files the open tree can run, and every other one with why it is left out: a test is kept when each in-house
    module it imports (anywhere in the file) is exported and its text names none of the private folders."""
    spec = getattr(m, "TESTS", None)
    if not spec:
        return [], {}
    exported = {_module_name(f) for f in files if f.endswith(".py")}
    packages = {mod.rsplit(".", i)[0] for mod in exported for i in range(1, mod.count(".") + 1)} | exported
    roots = tuple(m.RENAME) + ("dclab_core",)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    kept, left = [], {}
    for rel in sorted(p for p in tracked if fnmatch.fnmatch(p, spec["candidates"]) and (ROOT / p).is_file()):
        if rel in spec["exclude"]:
            left[rel] = spec["exclude"][rel]
            continue
        text = _strip_closed((ROOT / rel).read_text(encoding="utf-8"))[0]  # what the open tree would hold
        private = next((w for w in spec["private_paths"] if re.search(w, text)), None)
        if private:
            left[rel] = f"names {private}"
            continue
        missing = None
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                targets = [node.module] + [f"{node.module}.{a.name}" for a in node.names if f"{node.module}.{a.name}" not in packages
                                           and _looks_like_module(node.module, a.name)]
            else:
                continue
            missing = next((t for t in targets if t.startswith(roots) and t not in packages), None)
            if missing:
                break
        if missing:
            left[rel] = f"imports {missing}, which the open tree leaves out"
            continue
        kept.append(rel)
    stems = {Path(t).stem for t in kept}
    changed = True
    while changed:  # a test that imports another test module (its fakes) stands only when that one is kept
        changed = False
        for rel in list(kept):
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            used = {a.name.split(".")[-1] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
                   {(n.module or "").split(".")[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            gone = sorted(u for u in used if u.startswith("test_") and u not in stems and (ROOT / "tests" / f"{u}.py").is_file())
            if gone:
                kept.remove(rel)
                stems.discard(Path(rel).stem)
                left[rel] = f"imports the test module {gone[0]}, which the open tree leaves out"
                changed = True
    helpers = [p for p in tracked if _match(p, spec["helpers"]) and (ROOT / p).is_file()]
    return kept + sorted(helpers), left


def _looks_like_module(package: str, name: str) -> bool:
    """`from dclab_rnd.store import keepers` names a module when the private tree has that file (else it is an attribute)."""
    base = ROOT / Path(*package.split("."))
    try:
        entries = set(os.listdir(base))  # exact names: a case-insensitive disk finds gateway.py for "Gateway"
    except OSError:
        return False
    return f"{name}.py" in entries or (name in entries and (base / name / "__init__.py").is_file())


def _renamed_path(path: str, m) -> str:
    for old, new in m.RENAME.items():
        if path == old or path.startswith(old + "/"):
            return new + path[len(old):]
    return path


def _rewrite(text: str, m) -> str:
    for old, new in m.RENAME.items():
        text = re.sub(rf"(?<![\w]){re.escape(old)}(?![\w])", new, text)
    for pattern, replacement in getattr(m, "REPLACE", []):
        text = re.sub(pattern, replacement, text)
    return text


MARKER = re.compile(r"^\s*(?:#|//|<!--|/\*)\s*dclab:closed-(begin|end)\b")  # in Python, JavaScript, HTML or CSS comments


def _strip_closed(text: str) -> tuple[str, int]:
    out, inside, blocks = [], False, 0
    for line in text.splitlines(keepends=True):
        found = MARKER.match(line)
        if found and found.group(1) == "begin":
            inside, blocks = True, blocks + 1
            continue
        if found:
            inside = False
            continue
        if not inside:
            out.append(line)
    return "".join(out), blocks


def build(out: Path, m) -> dict:
    files = selected(m)
    tests, left_tests = selected_tests(m, files)
    files = files + tests
    report = {"files": [], "closed_blocks": 0, "findings": [], "tests": {"kept": [t for t in tests if fnmatch.fnmatch(t, "tests/test_*.py")],
                                                                         "left_out": left_tests}}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for rel in files:
        src, dst = ROOT / rel, out / _renamed_path(rel, m)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT_SUFFIXES:
            try:
                text = src.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                shutil.copyfile(src, dst)
            else:
                text, blocks = _strip_closed(_rewrite(text, m))
                report["closed_blocks"] += blocks
                dst.write_text(text, encoding="utf-8")
        else:
            shutil.copyfile(src, dst)
        report["files"].append({"path": str(dst.relative_to(out)), "bytes": dst.stat().st_size})
    overlay = ROOT / m.OVERLAY
    if overlay.is_dir():
        for src in overlay.rglob("*"):
            if src.is_file():
                dst = out / src.relative_to(overlay)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                report["files"].append({"path": str(dst.relative_to(out)), "bytes": dst.stat().st_size, "overlay": True})
    return report


def scan_words(out: Path, m) -> list[dict]:
    patterns = [re.compile(line.strip(), re.I) for line in (ROOT / m.FORBIDDEN_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    found = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(out))
        if path.stat().st_size > m.MAX_FILE_BYTES:
            found.append({"check": "size", "path": rel, "detail": f"{path.stat().st_size} bytes"})
        if path.suffix not in TEXT_SUFFIXES and path.name not in (".env.example", ".gitignore", "LICENSE", "NOTICE"):
            if path.suffix not in m.BINARY_ALLOWED:
                found.append({"check": "binary", "path": rel, "detail": path.suffix})
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            found.append({"check": "binary", "path": rel, "detail": "not utf-8"})
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for allowed in getattr(m, "ALLOWED_PUBLIC", []):
                line = line.replace(allowed, "")  # the public project's own address, nothing else of the owner's name
            for p in patterns:
                if p.search(line):
                    found.append({"check": "private word", "path": rel, "line": n, "detail": p.pattern})
            if "dclab:closed-begin" in line or "dclab:closed-end" in line:
                found.append({"check": "closed marker", "path": rel, "line": n, "detail": "left after stripping"})
    return found


def scan_secrets(out: Path) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import scan_secrets  # noqa: E402 - the repository's own patterns and placeholders

    hits = scan_secrets.scan(scan_secrets.everything(out), out)
    return [{"check": "secret", "path": h[0], "line": h[1], "detail": h[2]} for h in hits]


def _module_of(path: Path, out: Path) -> str:
    parts = list(path.relative_to(out).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _catches_missing(handler) -> bool:
    kinds = handler.elts if isinstance(handler, ast.Tuple) else [handler]
    return any(getattr(k, "id", "") in ("ModuleNotFoundError", "ImportError") for k in kinds)


def import_closure(out: Path, m) -> list[dict]:
    """Every in-house module an exported file imports must be in the tree; a closed one only by name inside optional(...)."""
    roots = (m.PUBLIC_NAME, "dclab_core")
    present = {_module_of(p, out) for p in out.rglob("*.py")}
    left_out = set()  # every module of the private packages the manifest left out ("from ...studio import sft" names one)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    for rel in tracked:
        if rel.endswith(".py") and _match(rel, m.INCLUDE) and _match(rel, m.EXCLUDE):
            parts = list(Path(_renamed_path(rel, m)).with_suffix("").parts)
            left_out.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    closed = [f"{m.PUBLIC_NAME}.{c}" for c in m.CLOSED]
    is_closed = lambda t: any(t == c or t.startswith(c + ".") for c in closed)  # noqa: E731
    found: dict[tuple[str, str], dict] = {}

    def flag(path: Path, line: int, target: str, why: str) -> None:
        found.setdefault((str(path.relative_to(out)), target), {"check": "import", "path": str(path.relative_to(out)), "line": line,
                                                               "detail": f"{target}: {why}"})
    for path in sorted(out.rglob("*.py")):
        module = _module_of(path, out)
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            found[(str(path), "syntax")] = {"check": "syntax", "path": str(path.relative_to(out)), "detail": str(exc)}
            continue
        optional_args = {a.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and getattr(node.func, "id", getattr(node.func, "attr", "")) == "optional"
                         for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
        guarded = set()  # an import inside `try: ... except ModuleNotFoundError` is optional by construction
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(_catches_missing(h.type) for h in node.handlers):
                guarded |= {id(n) for stmt in node.body for n in ast.walk(stmt) if isinstance(n, (ast.Import, ast.ImportFrom))}
        for node in ast.walk(tree):
            if id(node) in guarded:
                continue
            modules, names = [], []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")
                    prefix = ".".join(base[: len(base) - node.level + 1])
                    mod = f"{prefix}.{node.module}" if node.module else prefix
                else:
                    mod = node.module or ""
                modules = [mod]
                names = [f"{mod}.{a.name}" for a in node.names]
            for target in modules + names:
                if not target.startswith(roots):
                    continue
                if is_closed(target):
                    if target.split(".", 1)[1] not in optional_args:
                        flag(path, node.lineno, target, "a closed module, imported outside optional()")
                elif target in modules and target not in present:
                    flag(path, node.lineno, target, "not in the open tree")
                elif target in names and target in left_out:
                    flag(path, node.lineno, target, "a module the export leaves out")
                elif target in names and target not in present and target.rsplit(".", 1)[0] not in present:
                    flag(path, node.lineno, target, "not in the open tree")
    return list(found.values())


def rerun_kinds(out: Path, m) -> dict:
    """The acceptance of the kinds whose first case is not exported, run inside the open tree on the cases it has: the
    open capability matrix then says what the open code does. A run that fails is a finding."""
    kinds = getattr(m, "KINDS_RERUN", [])
    if not kinds:
        return {}
    with tempfile.TemporaryDirectory(prefix="export-kinds-") as tmp:
        done = subprocess.run([sys.executable, "-I", "-B", "-c", "import sys, runpy; sys.path.insert(0, '.'); "
                               f"sys.argv = ['kinds', *{kinds!r}]; runpy.run_module('{m.PUBLIC_NAME}.kinds', run_name='__main__')"],
                              cwd=out, capture_output=True, text=True,
                              env={"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "DCLAB_NO_LIVE_MODELS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if done.returncode != 0:
        return {"findings": [{"check": "kinds acceptance", "path": f"{m.PUBLIC_NAME}/kinds", "detail": done.stderr.strip()[-400:]}]}
    rows = json.loads((out / m.PUBLIC_NAME / "kinds" / "capability.json").read_text(encoding="utf-8"))["results"]
    return {r["kind"]: f"{r['status']} ({sum(c['ok'] for c in r['checks'])} of {len(r['checks'])})" for r in rows if r["kind"] in kinds}


def build_knowledge(out: Path, m) -> dict:
    """The packaged knowledge: only the record types the manifest names, with citations cut to the open tree's own paths."""
    spec = m.KNOWLEDGE
    private = [re.compile(w.strip(), re.I) for w in (ROOT / m.FORBIDDEN_FILE).read_text(encoding="utf-8").splitlines()
               if w.strip() and not w.startswith("#")]
    kept, dropped = [], []
    for line in (ROOT / spec["source"]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") not in spec["types"]:
            continue
        if any(p.search(line) for p in private):  # measured on (or naming) private data: never shipped, even as knowledge
            dropped.append(record.get("record_id"))
            continue
        own = [c for c in record.get("citations") or [] if c.split("/", 1)[0] in m.RENAME]
        record["citations"] = [_renamed_path(c, m) for c in own] or [f"{m.PUBLIC_NAME}/{spec['out']}"]
        kept.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    dst = out / m.PUBLIC_NAME / spec["out"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_rewrite("\n".join(kept) + "\n", m), encoding="utf-8")
    return {"path": str(dst.relative_to(out)), "records": len(kept), "dropped": dropped, "bytes": dst.stat().st_size}


def build_ui(out: Path, m) -> dict:
    """The studio's UI built inside the open tree, from its own sources and the open theme (the built files are never copied
    from this repository). A build that fails is a finding."""
    static = out / m.PUBLIC_NAME / "agentic" / "static" / "app"
    static.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([sys.executable, "-I", "-B", "-c", "import sys, runpy; sys.path.insert(0, '.'); sys.argv = ['build']; "
                           f"runpy.run_module('{m.PUBLIC_NAME}.agentic.web.build', run_name='__main__')"],
                          cwd=out, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "DCLAB_NO_LIVE_MODELS": "1", "HOME": str(out), "PYTHONDONTWRITEBYTECODE": "1"})
    if done.returncode != 0:
        return {"built": False, "findings": [{"check": "ui build", "path": str(static.relative_to(out)), "detail": done.stderr.strip()[-400:]}]}
    return {"built": True, "files": sum(1 for p in static.rglob("*") if p.is_file())}


def scan_wheel(out: Path, m) -> dict:
    """The package people install, scanned like the tree: built from the open tree (uv, else pip), unpacked, then the same
    private words and secret patterns. A build that fails is a finding."""
    import shutil
    import zipfile

    with tempfile.TemporaryDirectory(prefix="export-wheel-") as tmp:
        dist, unpacked = Path(tmp) / "dist", Path(tmp) / "unpacked"
        uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        command = [uv, "build", "--wheel", "--out-dir", str(dist), str(out)] if Path(uv).exists() else \
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(dist), str(out)]
        done = subprocess.run(command, capture_output=True, text=True)
        wheels = sorted(dist.glob("*.whl")) if dist.is_dir() else []
        if done.returncode != 0 or not wheels:
            return {"built": False, "findings": [{"check": "wheel build", "path": "dist", "detail": (done.stderr or done.stdout).strip()[-400:]}]}
        with zipfile.ZipFile(wheels[0]) as z:
            z.extractall(unpacked)
        found = [{**f, "check": "wheel: " + f["check"]} for f in scan_words(unpacked, m) + scan_secrets(unpacked)
                 if f["check"] != "closed marker"]
        return {"built": True, "name": wheels[0].name, "files": sum(1 for x in unpacked.rglob("*") if x.is_file()),
                "bytes": wheels[0].stat().st_size, "findings": found}


def write_markdown(report: dict, path: Path) -> None:
    """The report a person reads before a release: what shipped, what was left out and why, and every finding."""
    s = report["summary"]
    lines = ["# Export report", "", f"**{'No findings' if not report['findings'] else str(len(report['findings'])) + ' findings: not ready'}**", "",
             "| | |", "|---|---|", f"| Files | {s['files']} ({s['bytes'] / 1e6:.1f} MB) |", f"| Closed blocks stripped | {s['closed_blocks']} |",
             f"| Knowledge records shipped | {s['knowledge_records']} (dropped for naming private data: {', '.join(report['knowledge'].get('dropped') or []) or 'none'}) |",
             f"| Test files shipped | {s['tests']['kept']} (left out: {s['tests']['left_out']}) |"]
    wheel = report.get("wheel") or {}
    if wheel:
        lines.append(f"| Wheel | {wheel.get('name', 'not built')} ({wheel.get('files', 0)} files, {wheel.get('bytes', 0) / 1e6:.1f} MB) |")
    for kind, verdict in (report.get("kinds_rerun") or {}).items():
        lines.append(f"| Acceptance rerun in the open tree: {kind} | {verdict} |")
    lines += ["", "## Findings", ""] + ([f"- **{f['check']}** `{f['path']}`{':' + str(f['line']) if f.get('line') else ''}: {f.get('detail', '')}"
                                          for f in report["findings"]] or ["None."])
    by_reason: dict[str, list[str]] = {}
    for test, why in sorted((report.get("tests") or {}).get("left_out", {}).items()):
        by_reason.setdefault(re.sub(r"dclab_rnd\.[\w.]+|test_\w+", "…", why), []).append(Path(test).name)
    lines += ["", "## Test files left out, by reason", ""] + [f"- {why}: {len(names)}" for why, names in sorted(by_reason.items(), key=lambda x: -len(x[1]))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the folder to build the open tree in (emptied first; outside this repository)")
    ap.add_argument("--check", action="store_true", help="build and check only")
    ap.add_argument("--no-wheel", action="store_true", help="skip building and scanning the wheel (faster)")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if out == ROOT or ROOT in out.parents:
        print("The open tree is built outside this repository.")
        return 2
    m = manifest()
    report = build(out, m)
    report["knowledge"] = build_knowledge(out, m)
    report["kinds_rerun"] = rerun_kinds(out, m)
    report["ui"] = build_ui(out, m)
    report["wheel"] = scan_wheel(out, m) if not args.no_wheel else {}
    findings = scan_words(out, m) + scan_secrets(out) + import_closure(out, m) + report["ui"].get("findings", []) \
        + report["kinds_rerun"].pop("findings", []) + report["wheel"].pop("findings", [])
    report["findings"] = findings
    by = {}
    for f in findings:
        by[f["check"]] = by.get(f["check"], 0) + 1
    report["summary"] = {"files": len(report["files"]), "bytes": sum(f["bytes"] for f in report["files"]), "closed_blocks": report["closed_blocks"],
                         "knowledge_records": report["knowledge"]["records"], "kinds_rerun": report["kinds_rerun"],
                         "tests": {"kept": len(report["tests"]["kept"]), "left_out": len(report["tests"]["left_out"])}, "findings": by}
    (out.parent / f"{out.name}.EXPORT_REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, out.parent / f"{out.name}.EXPORT_REPORT.md")
    print(json.dumps(report["summary"], indent=1))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

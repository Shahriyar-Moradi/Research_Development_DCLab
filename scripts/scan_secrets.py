"""Look for committed secrets (package 12.2): in the files git tracks, in a folder (an image's filesystem), or in history.

    python scripts/scan_secrets.py              # the repository's tracked files
    python scripts/scan_secrets.py --root DIR   # every file under DIR (for example an image exported with docker export)
    python scripts/scan_secrets.py --history    # every line ever added in the repository's history (a secret deleted later is still there)

Reports where (file and line, or commit and file), the kind of secret and a short hash of the value, never the value,
and exits 1 when anything is found. A match that is itself a placeholder (YOUR_, example, xxx, a variable such as
${...}) is not a secret; the rest of the line does not matter. A known harmless finding (a test's made-up value) is
listed in scripts/secrets_allowed.txt as path:kind:hash, so the same file holding a new, real secret is still found.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    ("OpenAI-style key", re.compile(r"\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{20,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----")),
    ("password in a URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@'\"]+:([^\s@/'\"]{6,})@")),
    ("DCLab API token", re.compile(r"\bdclab_[A-Za-z0-9_-]{40,}")),
    ("Kaggle key", re.compile(r"KAGGLE_KEY\s*=\s*['\"]?[0-9a-f]{32}")),
]
PLACEHOLDERS = re.compile(r"YOUR_|your[-_]|example|xxx|changeme|placeholder|\$\{|<[A-Z_]+>|dclab-local-only|dclab-ci-only|not-a-real|:password@", re.I)
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".parquet", ".pkl", ".joblib", ".npy", ".npz", ".zip", ".gz", ".woff", ".woff2",
                 ".ttf", ".ico", ".sqlite3", ".db", ".xlsx", ".pyc"}


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return [ROOT / p for p in out.split("\0") if p]


def everything(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]


def allowed() -> set[tuple[str, str, str]]:
    path = ROOT / "scripts" / "secrets_allowed.txt"
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            where, kind, value = line.strip().rsplit(":", 2)
            out.add((where, kind, value))
    return out


def matches(line: str) -> list[tuple[str, str]]:
    """(kind, fingerprint) of each secret-looking match in a line; a match that is itself a placeholder is not one."""
    found = []
    for kind, pattern in PATTERNS:
        for m in pattern.finditer(line):
            if not PLACEHOLDERS.search(m.group(0)):
                found.append((kind, fingerprint(m.group(0))))
    return found


def scan(files: list[Path], base: Path) -> list[tuple[str, int, str, str]]:
    found, known = [], allowed()
    for path in files:
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES or path.stat().st_size > 5_000_000:
            continue  # a tracked file deleted from the working tree is not there to scan
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(base).as_posix() if path.is_relative_to(base) else str(path)
        for n, line in enumerate(text.splitlines(), 1):
            for kind, value in matches(line):
                if (rel, kind, value) not in known:
                    found.append((rel, n, kind, value))
    return found


def history() -> list[tuple[str, str, str, str]]:
    """(commit, path, kind, fingerprint) for each line any commit added that looks like a secret."""
    known = allowed()
    process = subprocess.Popen(["git", "log", "-p", "--all", "--no-color", "--format=commit %H"], cwd=ROOT, stdout=subprocess.PIPE)
    found, commit, path = [], "", ""
    for raw in process.stdout:
        line = raw.decode("utf-8", "replace")
        if line.startswith("commit "):
            commit = line.split()[1][:10]
        elif line.startswith("+++ "):
            path = line[6:].strip() if line.startswith("+++ b/") else ""
        elif line.startswith("+"):
            for kind, value in matches(line):
                if (path, kind, value) not in known:
                    found.append((commit, path, kind, value))
    process.wait()
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, help="scan every file under this folder instead of the repository's tracked files")
    parser.add_argument("--history", action="store_true", help="scan every line added in the repository's history")
    args = parser.parse_args(argv)
    if args.history:
        found = history()
        for commit, path, kind, value in found:
            print(f"{commit} {path}: {kind} ({value})")
        print(f"history: {len(found)} possible secret{'s' if len(found) != 1 else ''}")
        return 1 if found else 0
    base = args.root.resolve() if args.root else ROOT
    files = everything(base) if args.root else tracked()
    found = scan(files, base)
    for rel, n, kind, value in found:
        print(f"{rel}:{n}: {kind} ({value})")
    print(f"{len(files)} files scanned, {len(found)} possible secret{'s' if len(found) != 1 else ''}"
          + ("" if not found else " (list a known harmless one in scripts/secrets_allowed.txt as path:kind:hash)"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())

"""The test files a change can affect, from the imports of the code (make check-quick).

    python scripts/affected_tests.py                 the test files affected by the working tree's changes against HEAD
    python scripts/affected_tests.py --base main~1   … against another commit
    python scripts/affected_tests.py --why           with the reason each file is chosen

A changed module affects every module that imports it, directly or through others (imports inside functions count:
the graph errs on running more). A test file is chosen when it changed, or imports an affected module, or names a
changed non-Python file (a prompt, a fixture, a view). A change to the foundations every test stands on (storage,
settings, the server's construction, the contracts' base, the test runner) chooses every file: printed as ALL.
Prints one file name per line (``test_x.py``), for ``tests/run_parallel.py``.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("dclab_rnd", "dclab_core", "general_pipeline")
# a change here can change any test without an import saying so (the environment, the stores' schema, the runner):
# every file runs. Everything else is followed through the imports.
FOUNDATIONS = ("dclab_rnd/storage/", "dclab_rnd/settings.py", "tests/run_parallel.py", "tests/pgtest.py", "requirements/",
               "pyproject.toml")
# modules that write or read a store: a change reaching one of them is checked on PostgreSQL too (``--stores``)
STORES = ("dclab_rnd.storage", "dclab_rnd.experience", "dclab_rnd.audit", "dclab_rnd.jobs", "dclab_rnd.accounts", "dclab_rnd.draft.store",
          "dclab_rnd.studio.store", "dclab_rnd.intern.sessions", "dclab_rnd.limits", "dclab_rnd.retention", "dclab_rnd.models.pause")
ALWAYS = ("test_config.py", "test_api_contract.py", "test_devlog.py")  # cheap checks of the repository's own rules
# a test file that checks what differs on PostgreSQL (it names it): the per-package PostgreSQL pass runs these
PG_MARKERS = ("DCLAB_DATABASE_URL", "pgtest", "PgLog", "PgAudit", "postgres", "PostgreSQL")
# product code the end-to-end flows exercise: a change here runs them
E2E_PATHS = ("dclab_rnd/agentic/", "dclab_rnd/draft/", "dclab_rnd/studio/", "dclab_rnd/jobs/", "dclab_rnd/intern/", "dclab_rnd/kinds/",
             "dclab_rnd/models/", "dclab_rnd/agents/", "scripts/product_e2e.py", "general_pipeline/", "dclab_core/")


def module_of(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    if rel.parts[0] not in PACKAGES or rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imports_of(path: Path, module: str | None) -> set[str]:
    """Every module this file imports, anywhere in it, as absolute names (a package import names the package)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    package = (module or "").rsplit(".", 1)[0] if module and not path.name == "__init__.py" else (module or "")
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)] if node.level > 1 else anchor
                base = ".".join([*anchor, base] if base else anchor)
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)  # from x import y: y may be a module
    return {m for m in out if m.split(".")[0] in PACKAGES}


def changed_files(base: str, head: str | None = None) -> list[str]:
    """The files changed since ``base``: in the working tree with the new files (no ``head``), or up to the commit ``head``."""
    tracked = subprocess.run(["git", "diff", "--name-only", base, *([head] if head else [])], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    new = [] if head else subprocess.run(["git", "ls-files", "--others", "--exclude-standard", *PACKAGES, "tests", "scripts"], cwd=ROOT,
                                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(set(tracked) | set(new))


def affected(base: str = "HEAD", head: str | None = None) -> tuple[list[str] | None, dict[str, str]]:
    """The test files to run (None: every file) and why each was chosen; ``why["stores"]`` says whether the change
    reaches a store (then they run on PostgreSQL too)."""
    changed = changed_files(base, head)
    for name in changed:
        if name.startswith(FOUNDATIONS):
            return None, {"*": f"{name} is a foundation every test stands on", "stores": "yes"}
    modules = {m: p for p in (ROOT / pkg for pkg in PACKAGES) for p in p.rglob("*.py") for m in [module_of(p)] if m}
    importers: dict[str, set[str]] = {}
    for module, path in modules.items():
        for imported in imports_of(path, module):
            importers.setdefault(imported, set()).add(module)
    start = {m for name in changed for m in [module_of(ROOT / name)] if m and (ROOT / name).exists()}
    start |= {module_of(ROOT / name) for name in changed if name.endswith(".py") and name.split("/")[0] in PACKAGES} - {None}
    reached, todo = set(start), list(start)
    while todo:
        module = todo.pop()
        names = [name for name in importers if name == module or name.startswith(module + ".")]  # from module import y
        for importer in {who for name in names for who in importers[name]}:
            if importer not in reached:
                reached.add(importer)
                todo.append(importer)
    others = [n for n in changed if not n.endswith(".py") and n.split("/")[0] in (*PACKAGES, "tests")]
    why: dict[str, str] = {}
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        if f"tests/{test.name}" in changed:
            why[test.name] = "changed"
            continue
        hit = next((m for m in imports_of(test, None) if m in reached or any(m.startswith(r + ".") for r in reached)), None)
        if hit:
            why[test.name] = f"imports {hit}"
            continue
        text = test.read_text(encoding="utf-8", errors="ignore")
        named = next((n for n in others if Path(n).name in text or Path(n).stem in text), None)
        if named:
            why[test.name] = f"names {named}"
    for name in ALWAYS:
        why.setdefault(name, "always")
    files = sorted(why)
    stores = next((m for m in sorted(start) if m.startswith(STORES)), None) or next((m for m in sorted(reached) if m.startswith(STORES)), None)
    why["stores"] = f"yes: {stores}" if stores else "no"
    return files, why


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--head", help="a commit: the changes between --base and it (default: the working tree)")
    parser.add_argument("--why", action="store_true")
    parser.add_argument("--stores", action="store_true", help="print only yes or no: does the change reach a store (run on PostgreSQL)?")
    parser.add_argument("--pg", action="store_true", help="the files for the PostgreSQL pass: ALL when storage or a store "
                        "module itself changed, else the affected files that check what differs on PostgreSQL")
    parser.add_argument("--e2e", action="store_true", help="print only yes or no: does the change reach the end-to-end flows?")
    args = parser.parse_args(argv)
    if args.e2e:
        print("yes" if any(n.startswith(E2E_PATHS) for n in changed_files(args.base, args.head)) else "no")
        return 0
    files, why = affected(args.base, args.head)
    if args.pg:
        direct = [n for n in changed_files(args.base, args.head) if (module_of(ROOT / n) or "").startswith(STORES)]
        if files is None or direct:
            print("ALL")
            return 0
        for name in files:
            if any(marker in (ROOT / "tests" / name).read_text(encoding="utf-8", errors="ignore") for marker in PG_MARKERS):
                print(name)
        return 0
    if args.stores:
        print("yes" if why["stores"].startswith("yes") else "no")
        return 0
    if files is None:
        print("ALL" + (f"  # {why['*']}" if args.why else ""))
        return 0
    for name in files:
        print(f"{name}  # {why[name]}" if args.why else name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The workspaces the keepers read (32.3), and what each one's consent lets the store keep.

A workspace is a folder holding DCLab's records (``experience/``, ``ledger/``, ``projects/``, ``drafts/``, ``audit.jsonl``):
the development workspace (``agent_runs``), each live test's, each real problem's, each twin lane's, and each tester's as
collected (``testers/<id>/``). Its name in the store is its path below ``agent_runs`` as a slug (``dev`` for the root).

Consent (``consent.json``, ``dclab_rnd.consent``):

- a folder of the owner's own runs on this machine with no consent of its own is the owner's: ``train`` (the owner,
  2026-10-20: store everything);
- a tester's folder, or any folder with its own ``consent.json``, follows it: ``train``, ``improve`` or ``operate``;
- an ``operate`` workspace keeps counts only (each file's name, size and hash), never its content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ..layout import ROOT, slug

AGENT_RUNS = ROOT / "agent_runs"
SCRIPTED = "scripted.json"  # a folder whose people are a script (test clicks, written people): their "person" answers are no human's


def scripted(root: Path, base: Optional[Path] = None) -> bool:
    """Whether a workspace, or a folder above it within ``base`` (``agent_runs``), says a script drove it (the same rule
    as ``dclab_rnd.lake.harvest.scripted``)."""
    base, root = Path(base or AGENT_RUNS).resolve(), Path(root).resolve()
    for folder in (root, *root.parents):
        if (folder / SCRIPTED).is_file():
            return True
        if folder == base or base not in folder.parents:
            return False
    return False


def scripted_names(base: Optional[Path] = None) -> frozenset[str]:
    """The store's names of the scripted workspaces: their person labels are never a human's (S8, 2026-10-25)."""
    base = Path(base or AGENT_RUNS)
    return frozenset(w.name for w in discover(base) if scripted(w.root, base))
MARKS = ("experience", "ledger", "projects", "drafts", "audit.jsonl", "agent_steps.jsonl", "model_requests.jsonl", "jobs.json",
         "jobs_history.jsonl", "intern", "orchestra", "students", ".dclab_workspace", "consent.json")
SKIP = {"node_modules", ".git", "__pycache__", "studio_assets", "embeddings", "clean_exports", "catalog", "data"}


@dataclass(frozen=True)
class Workspace:
    name: str
    root: Path
    consent: str            # train, improve or operate
    owner: bool             # the owner's own runs
    ledger_root: bool = False  # the folder is a ledger itself (a self-play solve's): its files read as ledger/<file>


def _consent(root: Path, owner: bool) -> str:
    from ... import consent

    if not (root / "consent.json").exists():
        return "train" if owner else "operate"
    c = consent.workspace_consent(root)
    return "train" if c.get("train") else "improve" if c.get("improve") else "operate"


def _is_workspace(path: Path) -> bool:
    return any((path / m).exists() for m in MARKS)


def discover(base: Optional[Path] = None, depth: int = 64) -> Iterator[Workspace]:
    """Every workspace under ``base`` (``agent_runs``): the root itself, then each folder that holds records, nearest first.
    A workspace inside another is its own (a live test's folder is not the development workspace's)."""
    base = Path(base or AGENT_RUNS)
    if not base.exists():
        return
    found: list[Path] = []

    def walk(path: Path, level: int) -> None:
        if level > depth:
            return
        for child in sorted(p for p in path.iterdir() if p.is_dir() and p.name not in SKIP and not p.name.startswith(".")):
            if child.name in ("experience", "ledger", "projects", "drafts", "intern", "orchestra", "students", "judgments"):
                continue  # a workspace's own folders, not workspaces
            if _is_workspace(child):
                found.append(child)
            walk(child, level + 1)

    if _is_workspace(base):
        found.insert(0, base)
    walk(base, 1)
    for path in found:
        rel = path.relative_to(base).as_posix()
        owner = not rel.startswith("testers/")
        yield Workspace(name="dev" if rel == "." else slug(rel.replace("/", "__"), 100), root=path, consent=_consent(path, owner), owner=owner)


SELFPLAY = ROOT / "lake" / "bronze" / "selfplay"  # 27.3's verified solves: invented tables, the owner's own; kept like the owner's runs


def discover_selfplay(base: Optional[Path] = None) -> Iterator[Workspace]:
    """The self-play folder as workspaces: the folder itself (``selfplay``: what was tried and verified, the screens) and each
    solve's ledger folder (``selfplay__ledgers__<kind>__<seed>``: every decision the engine made, labelled by the planted truth).
    Invented problems with no person and no real table, so ``train`` like the owner's own runs."""
    base = Path(base or SELFPLAY)
    if not base.is_dir():
        return
    yield Workspace(name="selfplay", root=base, consent="train", owner=True)
    for root in sorted(p for p in base.glob("ledgers/*/*") if p.is_dir()):  # one folder per solve, by its place: ledgers/<kind>/<seed>
        yield Workspace(name=slug("selfplay__" + root.relative_to(base).as_posix().replace("/", "__"), 100), root=root, consent="train", owner=True,
                        ledger_root=True)


def base_of(home: Path) -> Path:
    """The workspaces' common folder for ``home``: the nearest folder named like the development one (agent_runs), else ``home``."""
    here = Path(home).resolve()
    return next((p for p in (here, *here.parents) if p.name == AGENT_RUNS.name), here)


def of(home: Path) -> Workspace:
    """The workspace in ``home``, named as the conductor names it (so its records from every keeper share one name)."""
    here = Path(home).resolve()
    base = base_of(here)
    found = next((w for w in discover(base) if w.root.resolve() == here), None)
    if found is not None:
        return found
    from ..layout import Layout

    return Workspace(name=Layout.workspace(str(here)), root=here, consent=_consent(here, owner=AGENT_RUNS.resolve() in (here, *here.parents)),
                     owner=AGENT_RUNS.resolve() in (here, *here.parents))


def own_files(ws: Workspace, pattern: str, workspaces: list[Workspace]) -> Iterator[Path]:
    """The files of ``pattern`` in this workspace, never those of a workspace nested inside it."""
    nested = [w.root.resolve() for w in workspaces if w.root != ws.root and ws.root in w.root.parents]
    for path in sorted(ws.root.glob(pattern)):
        if path.is_file() and not any(n in path.resolve().parents for n in nested) and not path.name.endswith((".lock", ".part")):
            yield path


__all__ = ["Workspace", "discover", "own_files", "AGENT_RUNS"]

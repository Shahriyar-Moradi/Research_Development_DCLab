"""Every path of the store, in one place (32.1). Nothing else in ``dclab_rnd/store`` joins a store path: it asks this module.

The store's root is ``DCLAB_STORE`` or ``training_records/store`` (a clone of the owner's private records repository,
``dclab-records``, on its ``owner/main`` branch; gitignored here). ``docs/guides/STORING_PHASE_PLAN.md`` draws the layout.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

from dclab_core.store import KINDS, MODEL_TYPES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "training_records" / "store"
CATALOG_COPY = ROOT / "training_records" / "STORE_CATALOG.json"  # the R&D repository's view of the store
TIERS = ("raw", "records", "sets")


def owns_repository_copy(root: Path | str) -> bool:
    """Whether the store at ``root`` is the owner's own: the default folder AND a clone of ``dclab-records`` (it has a ``.git``).
    On a fresh checkout or a cloud session the default folder is absent (it is gitignored); a run there makes an empty store, and
    its view must never replace the committed ``STORE_CATALOG.json``."""
    root = Path(root)
    return root.resolve() == DEFAULT.resolve() and (root / ".git").exists()
_PART = re.compile(r"^[a-z0-9][a-z0-9_.\-]{0,119}$")


def _part(value: str) -> str:
    """One folder or file name below the root: lower-case, no separator, no climbing out."""
    if not _PART.match(value or "") or value in (".", ".."):
        raise ValueError(f"{value!r} is not a store folder name (lower-case letters, digits, _ . -)")
    return value


def slug(text: str, limit: int = 120) -> str:
    """A source's name as a file name: lower-case, separators and unsafe characters as single dashes."""
    out = re.sub(r"[^a-z0-9_.]+", "-", str(text).lower()).strip("-.") or "source"
    return out[:limit]


class Layout:
    """The store's folders under one root."""

    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root or os.environ.get("DCLAB_STORE") or DEFAULT)

    # the root's own files
    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def catalog(self) -> Path:
        return self.root / "CATALOG.json"

    @property
    def schemas(self) -> Path:
        return self.root / "schemas"

    def tier(self, name: str) -> Path:
        """One tier's folder: ``raw``, ``records`` or ``sets``."""
        if name not in TIERS:
            raise ValueError(f"{name!r} is not a tier ({', '.join(TIERS)})")
        return self.root / name

    # tier 1
    def raw(self, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"{kind!r} is not a raw kind ({', '.join(KINDS)})")
        return self.root / "raw" / kind

    def index(self, kind: str) -> Path:
        return self.raw(kind) / "index.jsonl"

    def replays(self) -> Path:
        """The backfill's replays: past calls asked again with thinking on, beside the original's id (not a raw kind)."""
        return self.tier("raw") / "replays"

    def lock(self, kind: str) -> Path:
        return self.raw(kind) / ".lock"

    def torn(self, kind: str) -> Path:
        """Where an index line that does not parse (a write cut short) is set aside, never lost."""
        return self.raw(kind) / "index.torn.jsonl"

    def entry(self, relative: str) -> Path:
        """A kept file by its path relative to the root, refused when it would leave the root (an index from a merged branch)."""
        path = (self.root / relative).resolve()
        if not self.inside(path):
            raise ValueError(f"{relative!r} is outside the store")
        return path

    def schema(self, name: str) -> Path:
        return self.schemas / f"{_part(name)}.json"

    def service(self, name: str) -> Path:
        """``service/<name>.json`` (32.24 to 32.31): the builds with their verdicts, the curves, the catalog, the engagements and
        the playbook, rebuilt from the records with every whole run."""
        return self.root / "service" / f"{_part(name)}.json"

    @staticmethod
    def workspace(ident: str) -> str:
        """A workspace's name in the store: its id when it is a store name already, else ``w`` and a hash of it (a file workspace's id
        is its folder's absolute path), always the same for the same id."""
        return ident if _PART.match(ident or "") and ident not in (".", "..") else "w" + hashlib.sha1(str(ident).encode()).hexdigest()[:16]

    def raw_file(self, kind: str, workspace: str, day: str, name: str) -> Path:
        y, m, d = day.split("-")
        return self.raw(kind) / _part(workspace) / _part(y) / _part(m) / _part(d) / _part(name)

    # tier 2
    def records(self, kind: str, workspace: str, month: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"{kind!r} is not a kind ({', '.join(KINDS)})")
        return self.root / "records" / kind / _part(workspace) / f"{_part(month)}.jsonl.gz"

    def records_of(self, kind: str) -> Path:
        """``records/<kind>/``: every workspace's records of the kind."""
        if kind not in KINDS:
            raise ValueError(f"{kind!r} is not a kind ({', '.join(KINDS)})")
        return self.root / "records" / kind

    def unshaped(self, kind: str, workspace: str) -> Path:
        return self.records_of(kind) / _part(workspace) / "unshaped.jsonl"

    def records_lock(self) -> Path:
        """One shaping at a time: the worker's run and a person's command never write the same records."""
        return self.root / "records" / ".lock"

    def shaped_state(self, kind: str) -> Path:
        """What the shapers last read of the kind (its index's size and hash): an unchanged kind is not shaped again."""
        return self.records_of(kind) / ".shaped.json"

    # tier 3
    def sets(self, model_type: str, *parts: str) -> Path:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"{model_type!r} is not a model type ({', '.join(MODEL_TYPES)})")
        out = self.root / "sets" / model_type
        for p in parts:
            out = out / _part(p)
        return out

    # beside the tiers
    def exams_root(self) -> Path:
        return self.root / "exams"

    def exams(self, model_type: str, *parts: str) -> Path:
        """36.4: a model type's frozen exams (``<version>/exam.json``), their results per model version and the versions accepted."""
        if model_type not in MODEL_TYPES:
            raise ValueError(f"{model_type!r} is not a model type ({', '.join(MODEL_TYPES)})")
        out = self.exams_root() / model_type
        for p in parts:
            out = out / _part(p)
        return out

    def rounds(self, model_type: str, *parts: str) -> Path:
        """36.5: a model type's replay rounds (``round_NNN/``: its new rows, its train set and its manifest), never rewritten."""
        if model_type not in MODEL_TYPES:
            raise ValueError(f"{model_type!r} is not a model type ({', '.join(MODEL_TYPES)})")
        out = self.root / "rounds" / model_type
        for p in parts:
            out = out / _part(p)
        return out

    def versions(self, model_type: str, *parts: str) -> Path:
        """36.7: a model type's trained versions (``<version>.json``: its round, its exam, its history)."""
        if model_type not in MODEL_TYPES:
            raise ValueError(f"{model_type!r} is not a model type ({', '.join(MODEL_TYPES)})")
        out = self.root / "versions" / model_type
        for p in parts:
            out = out / _part(p)
        return out

    def code_runs(self) -> Path:
        """36.6: a test store's own kept code-check runs (the real store reads the workspace's ``agent_runs/code_runs``)."""
        return self.root / "code_runs"

    def runs(self) -> Path:
        return self.root / "runs"

    def labels(self, name: str) -> Path:
        """A labeller's file (``labels/<name>``): Claude's routing labels, kept locally, copied into every snapshot."""
        return self.root / "labels" / _part(name)

    def run_record(self, day: str, run: str) -> Path:
        return self.runs() / _part(day) / f"{_part(run)}.json"

    def quarantine(self, reason: Optional[str] = None) -> Path:
        """``quarantine/`` itself, or one reason's folder below it."""
        base = self.root / "quarantine"
        return base / _part(reason) if reason else base

    def quarantine_file(self, reason: str, run: str) -> Path:
        return self.quarantine(reason) / f"{_part(run)}.jsonl"

    def curation_seen(self) -> Path:
        """Every (model type, record body) the curators placed, so a record kept again is a duplicate in any later run."""
        return self.root / "raw" / "curation" / ".seen.txt"

    def curation_state(self) -> Path:
        """Which record files the curators last read (their hashes): an unchanged file is not curated again."""
        return self.root / "raw" / "curation" / ".curated.json"

    def forgotten(self) -> Path:
        return self.root / "forgotten"

    def inside(self, path: Path | str) -> bool:
        """Whether ``path`` is under the root (a write tool's first check)."""
        try:
            Path(path).resolve().relative_to(self.root.resolve())
            return True
        except ValueError:
            return False

    def relative(self, path: Path | str) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()


__all__ = ["Layout", "DEFAULT", "TIERS", "slug"]

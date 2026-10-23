"""Challenge packs on disk, the public part and the key apart (H0).

``materialize(root, family, seed)`` writes ``root/public/<family>/<pack>/{pack.json, table.csv}`` (what a solver may
read) and ``root/keys/<family>/<pack>.json`` (what the grader reads). ``read_public`` is the solver's only door: it
opens the pack's own folder and nothing else, so no solve can read a key, whatever it is told.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dclab_core.challenge_pack import ChallengeKey, ChallengePack
from dclab_core.variants import Variant, generate


def public_dir(root: Path, family: str, pack: str) -> Path:
    return Path(root) / "public" / family / pack


def key_path(root: Path, family: str, pack: str) -> Path:
    return Path(root) / "keys" / family / f"{pack}.json"


def materialize(root: Path, family: str, seed: int) -> Variant:
    """Write one variant: its public folder and, apart, its key."""
    variant = generate(family, seed)
    folder = public_dir(root, family, variant.pack.id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "table.csv").write_bytes(variant.csv)  # the exact bytes the pack's SHA-256 is of
    (folder / "pack.json").write_text(variant.pack.model_dump_json(indent=1), encoding="utf-8")
    path = key_path(root, family, variant.pack.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(variant.key.model_dump_json(indent=1), encoding="utf-8")
    return variant


def read_public(folder: Path) -> tuple[ChallengePack, pd.DataFrame]:
    """The solver's door: the pack and its table, from the pack's own folder only."""
    folder = Path(folder)
    pack = ChallengePack.read((folder / "pack.json").read_text(encoding="utf-8"))
    return pack, pd.read_csv(folder / "table.csv")


def read_key(root: Path, family: str, pack: str) -> ChallengeKey:
    """The grader's door."""
    return ChallengeKey.read(key_path(root, family, pack).read_text(encoding="utf-8"))


__all__ = ["public_dir", "key_path", "materialize", "read_public", "read_key"]


if __name__ == "__main__":  # pragma: no cover - a look at one pack
    import sys
    import tempfile

    v = materialize(Path(tempfile.mkdtemp()), sys.argv[1] if len(sys.argv) > 1 else "groups", int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(json.dumps({"pack": v.pack.model_dump(), "knobs": v.key.knobs}, indent=1))

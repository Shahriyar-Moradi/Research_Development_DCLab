"""The published JSON Schemas of the contracts, and the frozen shape of every released version.

``schemas/<name>.json``: the current schema of each contract, generated (``make core-schemas``), never edited; a test
fails when a file differs from what the code generates. A host in any language validates against these.

``schemas/baseline/<schema_version>.json``: the shape of each version, written the first time that version is
generated and never rewritten. A test compares the current shape with the baseline of the same version: changing a
contract without moving its version fails, because a reader pinned to the old release would break.

Shapes are compared after :func:`normalize`, which drops the parts that are documentation (titles, descriptions) or
that differ between pydantic releases without changing what is valid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dclab_core.experience  # noqa: F401  (registers the experience contracts)
import dclab_core.outcomes  # noqa: F401  (23.5: the outcome contract)
from dclab_core.contracts import CONTRACTS

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
BASELINE_DIR = SCHEMA_DIR / "baseline"
_DOC_KEYS = {"title", "description", "examples", "$id"}


def generate() -> dict[str, dict[str, Any]]:
    out = {}
    for name, model in CONTRACTS.items():
        schema = model.model_json_schema()
        schema["$id"] = f"https://dclab.dev/schemas/{version_of(name)}.json"
        out[name] = schema
    return out


def version_of(name: str) -> str:
    return CONTRACTS[name].model_fields["schema_version"].default


def normalize(schema: Any) -> Any:
    """The shape only: no documentation keys, and a Literal's single "enum" next to its "const" removed (older pydantic
    releases add it)."""
    if isinstance(schema, dict):
        out = {k: normalize(v) for k, v in schema.items() if k not in _DOC_KEYS}
        if "const" in out and out.get("enum") == [out["const"]]:
            out.pop("enum")
        return out
    if isinstance(schema, list):
        return [normalize(v) for v in schema]
    return schema


def write(directory: Path = SCHEMA_DIR) -> list[Path]:
    """Write the current schemas, and the baseline of any version that has none yet (never overwriting one)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "baseline").mkdir(exist_ok=True)
    written = []
    for name, schema in generate().items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
        baseline = directory / "baseline" / f"{version_of(name)}.json"
        if not baseline.exists():
            baseline.write_text(json.dumps(normalize(schema), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(baseline)
    for stale in directory.glob("*.json"):
        if stale.stem not in CONTRACTS:
            stale.unlink()
    return written


if __name__ == "__main__":
    for path in write():
        print(path)
    sys.exit(0)

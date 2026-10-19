"""The keepers (tier 1): one per kind of data, each knowing its sources and keeping them whole (32.2 onwards).

A keeper declares the kind it writes and the sources it claims (:class:`Claim`: a kind and a source prefix);
:func:`register` refuses a keeper whose claim overlaps another's, so two keepers never write the same path. A keeper's
``sources(env)`` yields what to keep (:class:`Source`); the conductor keeps each one through the store (nothing new when the
bytes are unchanged) and records what was read, kept and skipped, and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


@dataclass(frozen=True)
class Claim:
    kind: str
    prefix: str  # the sources it owns: a source name starting with it (relative to its workspace)

    def overlaps(self, other: "Claim") -> bool:
        return self.kind == other.kind and (self.prefix.startswith(other.prefix) or other.prefix.startswith(self.prefix))


@dataclass
class Source:
    workspace: str
    source: str                      # its name relative to its workspace (the claim's prefix first)
    data: Any                        # the bytes, the file read whole, or a function giving the bytes (read under the store's lock)
    consent: str = "operate"
    note: Optional[str] = None
    log: bool = False                # an append-only log: kept by its new tail (Store.keep_log)
    transform: Optional[Callable[[bytes], bytes]] = None  # applied to the bytes before keeping (the cell-quoting keys removed)


@dataclass
class Keeper:
    name: str
    claims: tuple[Claim, ...]
    sources: Callable[[Any], Iterable[Source]]
    feeds: tuple[str, ...] = ()      # the model types its kind feeds (said in the run record)
    about: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def owns(self, kind: str, source: str) -> bool:
        return any(c.kind == kind and source.startswith(c.prefix) for c in self.claims)


REGISTRY: dict[str, Keeper] = {}


def register(keeper: Keeper, registry: Optional[dict[str, Keeper]] = None) -> Keeper:
    """Add a keeper; refused when its name is taken or a claim overlaps another keeper's (two keepers on one path)."""
    reg = REGISTRY if registry is None else registry
    if keeper.name in reg:
        raise ValueError(f"a keeper named {keeper.name!r} is registered already")
    for other in reg.values():
        for mine in keeper.claims:
            clash = next((c for c in other.claims if mine.overlaps(c)), None)
            if clash is not None:
                raise ValueError(f"{keeper.name} claims {mine.kind}/{mine.prefix}, which {other.name} owns ({clash.kind}/{clash.prefix})")
    reg[keeper.name] = keeper
    return keeper


__all__ = ["Claim", "Source", "Keeper", "REGISTRY", "register"]

"""Errors with a fingerprint (package 23.4): the exception's type and the top frame inside our own code, so the same
failure is one group however many times it happened, wherever a message differs.

The fingerprint is twelve hex characters of the type and ``module:function`` of the innermost frame whose module is
ours (``OWN``: a host adds its own package prefixes); an exception raised outside our code (a library, the runtime)
groups by its type alone. Nothing of a message, a path or a value enters it.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional, Sequence

OWN: tuple[str, ...] = ("dclab_core", "dclab_rnd", "general_pipeline")


def _ours(module: str, own: Iterable[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in own)


def fingerprint(exc_type: str, frames: Sequence[tuple[str, str]], own: Iterable[str] = OWN) -> dict[str, Optional[str]]:
    """``frames``: (module, function) from the innermost outwards. Returns the fingerprint and the frame it rests on."""
    own = tuple(own)
    top = next(((m, f) for m, f in frames if m and _ours(m, own)), None)
    frame = f"{top[0]}:{top[1]}" if top else None
    key = f"{exc_type}|{frame or ''}".encode("utf-8")
    return {"fingerprint": hashlib.sha1(key).hexdigest()[:12], "frame": frame}


def from_exception(exc: BaseException, own: Iterable[str] = OWN) -> dict[str, Any]:
    """The code (the exception's type), the fingerprint and the frame of a caught exception."""
    frames: list[tuple[str, str]] = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((str(tb.tb_frame.f_globals.get("__name__") or ""), tb.tb_frame.f_code.co_name))
        tb = tb.tb_next
    frames.reverse()  # the traceback walks outwards in; the innermost frame of ours is the one closest to the raise
    return {"code": type(exc).__name__, **fingerprint(type(exc).__name__, frames, own)}


__all__ = ["OWN", "fingerprint", "from_exception"]

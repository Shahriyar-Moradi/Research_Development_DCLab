"""DCLab R&D automation and evidence registry."""

import os as _os
import sys as _sys

# OMP_NUM_THREADS=1 on macOS unless the person set their own, before any library starts OpenMP. A server here had two
# OpenMP runtimes loaded (PyTorch's from miniconda, LightGBM's from Homebrew) and a models stage froze at 0% CPU
# (live HyperAck through DCLab Code, 2026-10-28); restarted with this setting it ran in 30 s, once. It limits PyTorch,
# XGBoost and scikit-learn to one thread (same predictions, slower fits); LightGBM's explicit n_jobs ignores it. Not
# proven to be the cure: if a stage freezes again, limit LightGBM's n_jobs on macOS next. The tests run so too.
if _sys.platform == "darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

from . import secret_files as _secret_files

_secret_files.load(strict=False)  # NAME_FILE secrets, for every entry point (package 12.2); an unreadable one stops the server at start

__all__ = ["build_evidence", "collect_registry"]


def __getattr__(name: str):
    """The research registry's two entry points, loaded when asked (the open edition has neither)."""
    if name in ("build_evidence", "collect_registry"):
        from .extensions import optional

        module = optional("analysis" if name == "build_evidence" else "registry")
        if module is not None:
            return getattr(module, name)
    raise AttributeError(name)
__version__ = "0.1.0"

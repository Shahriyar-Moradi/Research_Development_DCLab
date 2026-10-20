"""DCLab R&D automation and evidence registry."""

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

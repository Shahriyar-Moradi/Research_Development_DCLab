"""DCLab R&D automation and evidence registry."""

from . import secret_files as _secret_files

_secret_files.load(strict=False)  # NAME_FILE secrets, for every entry point (package 12.2); an unreadable one stops the server at start

from .analysis import build_evidence
from .registry import collect_registry

__all__ = ["build_evidence", "collect_registry"]
__version__ = "0.1.0"

"""DCLab R&D automation and evidence registry."""

from .analysis import build_evidence
from .registry import collect_registry

__all__ = ["build_evidence", "collect_registry"]
__version__ = "0.1.0"

"""Non-executing task previews. No scheduler, project writer or runtime entry point."""

from .adapter import adapt_workflow, capability_snapshot

__all__ = ["adapt_workflow", "capability_snapshot"]

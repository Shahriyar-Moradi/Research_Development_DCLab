"""The pipeline of workflows (20.8) as DCLab's main product stores a pipeline version.

Its ``pipeline_versions`` row holds ``graph_definition`` (JSONB), ``config`` (JSONB) and ``content_digest``, the
SHA-256 of the sorted, compact JSON of ``{"config", "graph_definition"}`` (``workflow_execution_service.content_digest``,
read at the commit in ``main_product_fields.json``). :func:`pipeline_version` gives those three fields for a
:class:`dclab_core.pipeline.Pipeline`; the row's ids (workspace, project, pipeline, workflow version) and its version
number are the host's. What each step made (its numbers and run) is not part of the graph: it belongs to the runs
(``main_product_lineage``)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from dclab_core import pipeline as core

FIELDS = ("graph_definition", "config", "content_digest")


def pipeline_version(pipeline: core.Pipeline, config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The ``pipeline_versions`` fields this library can fill: the graph, the config as given, and their digest."""
    graph = core.graph_definition(pipeline)
    cfg = dict(config or {})
    return {"graph_definition": graph, "config": cfg, "content_digest": core.content_digest(graph, cfg)}

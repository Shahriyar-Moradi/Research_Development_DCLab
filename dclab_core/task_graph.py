"""TP-01B structural task contracts; no traversal, hashing, dispatch or authorization.

A capability is a reference to the host registry, never a copied tool schema or executable expression. The later
semantic validator checks membership, source approval, coverage, edges and fingerprints against host metadata.
Collections are tuples so the frozen structural document is also immutable below its outer model.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Optional, get_args

from pydantic import Field, StrictInt, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Identifier = Annotated[str, Field(strict=True, min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
SourceFingerprint = Annotated[str, Field(strict=True, pattern=r"^(?:[0-9a-f]{16}|[0-9a-f]{64})$")]
Revision = Annotated[StrictInt, Field(ge=1, le=1_000_000)]
CapabilityScope = Literal["project", "draft", "session", "campaign", "workspace", "home", "chat", "case"]


Text = Annotated[str, Field(strict=True, min_length=1, max_length=1000, pattern=r"\S")]
Title = Annotated[str, Field(strict=True, min_length=1, max_length=200, pattern=r"\S")]

# Rules a JSON-only host can enforce without source graph/registry access. Cross-record identity and ordering
# checks still require semantic validation; JSON Schema cannot compare arbitrary reference values.
_NODE_RULES = {"allOf": [
    {"if": {"properties": {"kind": {"const": "action"}}, "required": ["kind"]},
     "then": {"required": ["capability", "scope", "acceptance_checks"],
              "properties": {"capability": {"not": {"type": "null"}}, "scope": {"not": {"type": "null"}},
                             "acceptance_checks": {"minItems": 1}}},
     "else": {"properties": {"capability": {"type": "null"}}}},
    {"if": {"properties": {"kind": {"const": "blocked"}}, "required": ["kind"]},
     "then": {"required": ["blocker_reason"], "properties": {"blocker_reason": {"type": "string", "minLength": 1, "pattern": r"\S"}}},
     "else": {"properties": {"blocker_reason": {"type": "null"}}}},
    {"if": {"properties": {"kind": {"enum": ["gate", "verification"]}}, "required": ["kind"]},
     "then": {"required": ["acceptance_checks"], "properties": {"acceptance_checks": {"minItems": 1}}}},
    *[{"if": {"properties": {"scope": {"const": scope}}, "required": ["scope"]},
       "then": {"properties": {"capability": {"properties": {"scope": {"const": scope}}}}}}
      for scope in get_args(CapabilityScope)],
]}


def _unique(values: tuple, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


class SourceReference(Contract):
    """Opaque metadata identity, not a path, URL, table value or proof of human consent."""

    kind: Literal["problem_card", "problem_map", "solution", "solution_plan", "solution_workflow", "solution_map",
                  "data_card", "dataset", "approval"]
    id: Identifier
    fingerprint: SourceFingerprint


class TaskProvenance(Contract):
    """Reference to an existing human, rule or model proposal record; no raw prompt or thinking text."""

    method: Literal["deterministic", "human", "model"]
    reference_id: Identifier
    producer_version: Annotated[str, Field(strict=True, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class CapabilityBinding(Contract):
    """A selected host capability's identity. This declaration does not grant permission to use it."""

    name: Annotated[str, Field(strict=True, min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    scope: CapabilityScope
    schema_fingerprint: Digest = Field(description="SHA-256 of the host tool schema supplied to planning; not a copied schema.")


class AcceptanceCheck(Contract):
    """Declarative evidence requirement. The host's deterministic verifier owns its interpretation."""

    id: Identifier
    kind: Literal["artifact", "lineage", "guard", "approval", "workflow_criterion"]
    description: Text
    evidence_refs: tuple[Identifier, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def _references(self) -> AcceptanceCheck:
        _unique(self.evidence_refs, "evidence references")
        return self


class TaskNode(Contract):
    """A group, one bounded action, a human gate, an evidence check or an explicit unsupported requirement.

    Non-action kinds have no callable binding. A group only organizes work; gate and verification nodes describe
    checks; blocked nodes require a reason. No kind executes merely because its document validates.
    """

    model_config = {"json_schema_extra": _NODE_RULES}

    id: Identifier
    parent_id: Optional[Identifier] = None
    title: Title
    objective: Text
    source_step_ids: tuple[Identifier, ...] = Field(default=(), max_length=64)
    kind: Literal["group", "action", "gate", "verification", "blocked"]
    needs: tuple[Identifier, ...] = Field(default=(), max_length=32)
    produces: tuple[Identifier, ...] = Field(default=(), max_length=32)
    dependencies: tuple[Identifier, ...] = Field(default=(), max_length=256)
    acceptance_checks: tuple[AcceptanceCheck, ...] = Field(default=(), max_length=32)
    capability: Optional[CapabilityBinding] = None
    scope: Optional[CapabilityScope] = None
    blocker_reason: Optional[Text] = None

    @model_validator(mode="after")
    def _local_shape(self) -> TaskNode:
        for field in ("source_step_ids", "needs", "produces", "dependencies"):
            _unique(getattr(self, field), field)
        _unique(tuple(check.id for check in self.acceptance_checks), "acceptance check IDs")
        if self.kind == "action":
            if self.capability is None or self.scope != self.capability.scope:
                raise ValueError("an action needs a capability and matching explicit scope")
            if not self.acceptance_checks:
                raise ValueError("an action needs acceptance checks")
        elif self.capability is not None:
            raise ValueError("only an action may bind a callable capability")
        if self.kind in ("gate", "verification") and not self.acceptance_checks:
            raise ValueError("gate and verification nodes need acceptance checks")
        if self.kind == "blocked":
            if self.blocker_reason is None:
                raise ValueError("a blocked node needs a blocker reason")
        elif self.blocker_reason is not None:
            raise ValueError("only a blocked node has a blocker reason")
        return self


class TaskEdge(Contract):
    """Grouping runs parent → child; dependency runs prerequisite → dependent. Neither edge executes work."""

    source: Identifier
    target: Identifier
    kind: Literal["grouping", "dependency"]


class TaskGraph(Versioned):
    """Immutable structural proposal. Runtime status and measured completion live outside this contract.

    TP-01C must verify the supplied fingerprint, source identities, capability membership and edge semantics.
    Local shape validation here is deliberately not a claim of executable or approved graph validity.
    """

    schema_version: Literal["dclab_core.task_graph.v1"] = _version("task_graph")
    graph_id: Identifier
    revision: Revision
    objective: Text
    source_refs: tuple[SourceReference, ...] = Field(min_length=1, max_length=32)
    workflow_fingerprint: SourceFingerprint
    solution_fingerprint: SourceFingerprint
    fingerprint: Digest = Field(description="Declared structural SHA-256; semantic verification belongs to TP-01C.")
    nodes: tuple[TaskNode, ...] = Field(min_length=1, max_length=256)
    edges: tuple[TaskEdge, ...] = Field(default=(), max_length=1024)
    provenance: TaskProvenance

    @model_validator(mode="before")
    @classmethod
    def _explicit_version(cls, value):
        if isinstance(value, Mapping) and "schema_version" not in value:
            raise ValueError("schema_version is required")
        return value

    @model_validator(mode="after")
    def _local_identity(self) -> TaskGraph:
        _unique(tuple(node.id for node in self.nodes), "task IDs")
        _unique(tuple((edge.source, edge.target, edge.kind) for edge in self.edges), "edges")
        _unique(tuple((ref.kind, ref.id) for ref in self.source_refs), "source references")
        return self


CONTRACTS["task_graph"] = TaskGraph

__all__ = ["TaskGraph", "TaskNode", "TaskEdge", "SourceReference", "TaskProvenance", "CapabilityBinding", "AcceptanceCheck"]

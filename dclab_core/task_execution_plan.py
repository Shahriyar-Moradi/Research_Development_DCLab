"""TP-01B: immutable bounded action proposals with typed argument references, never execution.

No literal arguments, expressions, paths or URLs are evaluated. The host supplies fingerprinted inputs, validates
actual tool schemas/authorization, and later resolves references through its guarded dispatcher. Outputs can only
be referenced after their declaring action in this bounded ordered list. No general graph traversal occurs here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Optional

from pydantic import Field, StrictInt, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _require_version, _version
from dclab_core.task_graph import (AcceptanceCheck, CapabilityBinding, CapabilityScope, Digest, Identifier, Revision,
                                   TaskProvenance, Text, _unique)

ValueType = Literal["string", "integer", "number", "boolean", "artifact"]


def _plan_schema(schema: dict) -> None:
    _require_version(schema)
    schema["allOf"] = [
        {"if": {"properties": {"status": {"enum": ["noop", "blocked"]}}, "required": ["status"]},
         "then": {"required": ["reason"], "properties": {
             "reason": {"type": "string", "minLength": 1, "pattern": r"\S"},
             "actions": {"maxItems": 0}, "declared_outputs": {"maxItems": 0}, "selected_capabilities": {"maxItems": 0}}}},
        {"if": {"properties": {"status": {"const": "ready"}}, "required": ["status"]},
         "then": {"required": ["actions", "acceptance_checks"],
                  "properties": {"actions": {"minItems": 1}, "acceptance_checks": {"minItems": 1}}}},
    ]



class ExpectedInput(Contract):
    id: Identifier
    fingerprint: Digest
    value_type: ValueType


class DeclaredOutput(Contract):
    id: Identifier
    action_id: Identifier
    value_type: ValueType


class ArgumentReference(Contract):
    kind: Literal["input", "output"]
    ref_id: Identifier
    value_type: ValueType


class ArgumentBinding(Contract):
    name: Identifier
    value: ArgumentReference


class PlannedAction(Contract):
    id: Identifier
    capability_name: Identifier
    scope: CapabilityScope
    arguments: tuple[ArgumentBinding, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def _argument_names(self) -> PlannedAction:
        _unique(tuple(argument.name for argument in self.arguments), "argument names")
        return self


class TaskBudget(Contract):
    max_tool_calls: StrictInt = Field(default=0, ge=0, le=100)
    max_seconds: StrictInt = Field(default=0, ge=0, le=3600)
    max_cost_eur: float = Field(default=0.0, ge=0, le=1000, strict=True)


class TaskStopPolicy(Contract):
    on_failure: Literal["stop", "escalate"] = "stop"
    on_unknown: Literal["block"] = "block"
    max_retries: StrictInt = Field(default=0, ge=0, le=3)
    compensation: Literal["none", "escalate"] = "none"


class TaskExecutionPlan(Versioned):
    """A proposal, including explicit no-op/blocked outcomes; none of its states asserts task completion.

    ``selected_capabilities`` are the proposed authorized subset supplied by the host. Membership here only ties
    actions to that declaration: registry schema, permissions, source approval and hash validation remain required
    before any future dispatch. ``ready`` means shape-ready, never permission to run.
    """

    model_config = {"json_schema_extra": _plan_schema}

    schema_version: Literal["dclab_core.task_execution_plan.v1"] = _version("task_execution_plan")
    graph_id: Identifier
    graph_revision: Revision
    graph_fingerprint: Digest
    task_id: Identifier
    plan_id: Identifier
    revision: Revision
    fingerprint: Digest
    expected_inputs: tuple[ExpectedInput, ...] = Field(default=(), max_length=128)
    selected_capabilities: tuple[CapabilityBinding, ...] = Field(default=(), max_length=32)
    actions: tuple[PlannedAction, ...] = Field(default=(), max_length=32)
    declared_outputs: tuple[DeclaredOutput, ...] = Field(default=(), max_length=128)
    acceptance_checks: tuple[AcceptanceCheck, ...] = Field(default=(), max_length=32)
    budget: TaskBudget = Field(default_factory=TaskBudget)
    policy: TaskStopPolicy = Field(default_factory=TaskStopPolicy)
    provenance: TaskProvenance
    status: Literal["draft", "ready", "noop", "blocked"] = "draft"
    reason: Optional[Text] = None

    @model_validator(mode="before")
    @classmethod
    def _explicit_version(cls, value):
        if isinstance(value, Mapping) and "schema_version" not in value:
            raise ValueError("schema_version is required")
        return value

    @model_validator(mode="after")
    def _local_shape(self) -> TaskExecutionPlan:
        for name in ("expected_inputs", "actions", "declared_outputs", "acceptance_checks"):
            _unique(tuple(item.id for item in getattr(self, name)), name + " IDs")
        _unique(tuple((cap.name, cap.scope) for cap in self.selected_capabilities), "selected capabilities")
        if self.status in ("noop", "blocked"):
            if self.reason is None or self.actions or self.declared_outputs or self.selected_capabilities:
                raise ValueError("noop/blocked needs a reason and no actions, outputs or selected capabilities")
        if self.status == "ready" and (not self.actions or not self.acceptance_checks):
            raise ValueError("a ready plan needs actions and acceptance checks")
        if len(self.actions) * (1 + self.policy.max_retries) > self.budget.max_tool_calls:
            raise ValueError("actions and retry allowance exceed max_tool_calls")
        capabilities = {(cap.name, cap.scope) for cap in self.selected_capabilities}
        inputs = {item.id: item for item in self.expected_inputs}
        outputs = {item.id: item for item in self.declared_outputs}
        if set(inputs) & set(outputs):
            raise ValueError("input and output artifact IDs must be distinct")
        positions = {action.id: index for index, action in enumerate(self.actions)}
        for output in self.declared_outputs:
            if output.action_id not in positions:
                raise ValueError("output needs a declared producing action")
        for index, action in enumerate(self.actions):
            if (action.capability_name, action.scope) not in capabilities:
                raise ValueError("action capability is not selected for this scope")
            for argument in action.arguments:
                ref = argument.value
                declared = inputs.get(ref.ref_id) if ref.kind == "input" else outputs.get(ref.ref_id)
                if declared is None or declared.value_type != ref.value_type:
                    raise ValueError("argument needs a declared reference of the same value type")
                if ref.kind == "output" and positions[declared.action_id] >= index:
                    raise ValueError("deferred argument must reference an earlier action's declared output")
        return self


CONTRACTS["task_execution_plan"] = TaskExecutionPlan

__all__ = ["TaskExecutionPlan", "ExpectedInput", "DeclaredOutput", "ArgumentReference", "ArgumentBinding",
           "PlannedAction", "TaskBudget", "TaskStopPolicy"]

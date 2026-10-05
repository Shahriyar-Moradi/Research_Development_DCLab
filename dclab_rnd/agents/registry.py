"""Tools and the registry every agent reads them from (package A2.1, docs/guides/AGENTIC_FOUNDATION_PLAN.md).

A tool is registered once, with its JSON schema, its handler, whether it changes state, and (for project writes) the
workflow move it makes. Scopes say who may call it:

    project   the intern, and MCP clients (Chat UI, ML Intern) at /mcp
    draft     the Home agent, on a draft before a project exists

A call is checked against the schema before the handler runs; a bad call comes back as ``{"error": ...}`` that a
model can read and correct, never as an exception.

Every write tool declares the move it makes, and every write runs through its scope's guard (package A2.2): the
project guard asks ``studio.graph.check`` as the actor "agent" and logs a refusal as a transition; the draft guard
checks the Home agent's writes. A write in a scope without a guard never runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import jsonschema

SCOPES = ("project", "draft")
EFFECTS = ("read", "write")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # a JSON schema of type object
    handler: Callable[..., Any]  # handler(**arguments), or handler(context, **arguments) when takes_context
    effect: str = "read"  # "write": the tool changes a project or a draft
    scope: str = "project"
    move: str | None = None  # what a write does: a studio.graph move (project) or a draft move; its scope's guard checks it
    aliases: tuple[str, ...] = ()  # old names that still work (old MCP and Chat UI configs)
    takes_context: bool = False
    errors: Callable[[Exception], dict[str, Any] | None] | None = None  # turns a handler's expected failure into an error result

    def schema(self) -> dict[str, Any]:
        """The tool in the chat-completions ``tools`` format."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class Guard(Protocol):
    def __call__(self, tool: Tool, context: Any, arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        """Check the write; return a refusal (``{"error": ...}``) without calling ``proceed``, or ``proceed()``'s result."""


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    guards: dict[str, Guard] = field(default_factory=dict)  # scope -> the check every write in that scope passes

    def register(self, tool: Tool) -> Tool:
        if tool.scope not in SCOPES or tool.effect not in EFFECTS:
            raise ValueError(f"{tool.name}: scope must be one of {SCOPES} and effect one of {EFFECTS}")
        if tool.effect == "write" and not tool.move:
            raise ValueError(f"{tool.name}: a write tool declares the move it makes")
        if tool.parameters.get("type") != "object":
            raise ValueError(f"{tool.name}: parameters must be a JSON schema of type object")
        for name in (tool.name, *tool.aliases):
            if name in self.tools or name in self._aliases():
                raise ValueError(f"a tool named {name!r} is already registered")
        self.tools[tool.name] = tool
        return tool

    def _aliases(self) -> dict[str, str]:
        return {alias: t.name for t in self.tools.values() for alias in t.aliases}

    def get(self, name: str) -> Tool | None:
        return self.tools.get(self._aliases().get(name, name))

    def names(self, scope: str | None = None) -> list[str]:
        return [t.name for t in self.tools.values() if scope is None or t.scope == scope]

    def schemas(self, scope: str | None = None, names: list[str] | None = None) -> list[dict[str, Any]]:
        chosen = [self.tools[n] for n in names] if names is not None else [t for t in self.tools.values() if scope is None or t.scope == scope]
        return [t.schema() for t in chosen]

    def check(self, tool: Tool, arguments: dict[str, Any], level: str = "schema") -> str | None:
        """Why the arguments do not fit the tool's schema, in words a model can act on; None when they fit.

        An argument sent as null counts as not given (models in strict mode send null for every optional field).
        ``level``: "schema" (everything), "names" (only unknown and missing arguments) or "none". The agents' own loops
        use the lighter levels until they run on ``run()`` (package A2.4), because their handlers clamp, coerce and
        default what the schema would refuse.
        """
        if level == "none":
            return None
        given = {k: v for k, v in arguments.items() if v is not None}
        properties = tool.parameters.get("properties", {})
        unknown = set(arguments) - set(properties)  # a null the handler cannot take is still unknown
        if unknown:
            return f"Unknown argument(s) for {tool.name}: {', '.join(sorted(unknown))}"
        missing = [r for r in tool.parameters.get("required", []) if r not in given]
        if missing:
            return f"Missing required argument(s) for {tool.name}: {', '.join(missing)}"
        if level == "names":
            return None
        problems = sorted(jsonschema.Draft202012Validator(tool.parameters).iter_errors(given), key=lambda e: list(e.path))
        if problems:
            first = problems[0]
            where = ".".join(str(p) for p in first.path) or "arguments"
            return f"Invalid argument for {tool.name}: {where}: {first.message[:200]}"
        return None

    def call(self, name: str, arguments: dict[str, Any] | None = None, context: Any = None, scope: str | None = None,
             check: str = "schema") -> Any:
        """Check and run one tool call. Every problem comes back as ``{"error": ...}``."""
        tool = self.get(name)
        if tool is None or (scope is not None and tool.scope != scope):
            available = ", ".join(self.names(scope))
            return {"error": f"Unknown tool {name!r}. Available: {available}"}
        arguments = dict(arguments or {})
        if "_raw" in arguments:  # the model's arguments were not valid JSON
            return {"error": f"The arguments for {tool.name} were not valid JSON; send a JSON object that fits the schema."}
        problem = self.check(tool, arguments, check)
        if problem:
            return {"error": problem}
        if tool.effect != "write":
            return self._run(tool, lambda: tool.handler(context, **arguments) if tool.takes_context else tool.handler(**arguments))
        guard = self.guards.get(tool.scope)
        if guard is None:
            raise RuntimeError(f"{tool.name} writes in scope {tool.scope!r}, which has no guard: it may not run")

        def proceed() -> Any:
            return self._run(tool, lambda: tool.handler(context, **arguments) if tool.takes_context else tool.handler(**arguments))
        return self._run(tool, lambda: guard(tool, context, arguments, proceed))

    @staticmethod
    def _run(tool: Tool, step: Callable[[], Any]) -> Any:
        try:
            return step()
        except Exception as error:  # noqa: BLE001 — an expected failure is the model's to read; anything else raises
            mapped = tool.errors(error) if tool.errors else None
            if mapped is not None:
                return mapped
            raise

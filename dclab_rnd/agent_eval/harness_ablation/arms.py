"""The two arms the conversation exam does not have (the exam's ``plain`` and ``case`` arms are used as they are):

* ``model_only`` (arm A): one request, no tools. The model gets the person's words and, when a table is shared, the same
  column view the other arms can read (names, kinds, what DCLab noticed, the row count; never a value). It is compared on
  understanding and the plan only, never on work done.
* ``open`` (arm B): one model loop with the read tools (the table's profile, an aggregate query, the leakage check, a
  cleaning preview) and a neutral prompt. Unlike the exam's ``plain`` arm it is given no list of kinds of work and no tool
  that names one, so nothing asks it to file the problem under a type. ``plain`` against ``open`` therefore measures the
  closed list alone; ``case`` against ``plain`` measures the lead agent's case file, gates and budgets.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from dclab_rnd import prompts

from ..conversation_exam.arms import DATA_TOOLS, PLAIN_TOOLS, table_view

# The two arms' own limits (2026-11-15). With the exam's (five model requests a turn, 6,000 tokens) 8 of 24 baseline answers
# of HAB-002 came back empty or cut: a turn that spent its requests on tools had none left for its words, and a thinking
# model used the tokens before the answer. A baseline held back by the runner says nothing about the harness.
MAX_CALLS = 10
MAX_TOKENS = 16000

LAST_REQUEST = ("[the test's runner] That was your last look at the table for this message: no more lookups will run. "
                "Write your answer to the person now, from what you have.")
MODEL_ONLY_PROMPT = "ablation_model_only"
MODEL_ONLY_SHORT_PROMPT = "ablation_model_only_short"
OPEN_PROMPT = "ablation_minimal_agent"
READ_TOOLS = [PLAIN_TOOLS[0], *DATA_TOOLS]  # profile_table and the lead agent's own data tools: reads only
NO_ROUTE = {"route": None, "tool": None, "routed_by": None}


def view_text(upload: Optional[str]) -> str:
    """The shared table as the model-only arm reads it: what ``profile_table`` returns to the other arms."""
    if not upload:
        return ""
    from ..home import tables

    return json.dumps({"table": f"{upload}.csv", "rows": len(tables.build(upload)), "columns": table_view(upload)}, default=str)


class ModelOnlyArm:
    name = "model_only"
    prompt = MODEL_ONLY_PROMPT

    def __init__(self, client: Any):
        self.client = client
        self.computed: list[str] = []

    def open(self, person: Mapping[str, Any]) -> dict[str, Any]:
        view = view_text(person.get("upload"))
        if view:
            self.computed.append(view)
        words = str(person["opening"]) + (f"\n\n(I shared a table: {person['upload']}.csv)\n\nThe table, without its values:\n{view}" if view else "")
        out = self.client.complete([{"role": "system", "content": prompts.text(self.prompt)}, {"role": "user", "content": words}],
                                   None, max_tokens=MAX_TOKENS)
        return {"text": str(out.get("content") or "").strip(), "choices": []}

    def send(self, words: str) -> dict[str, Any]:
        raise RuntimeError("the model-only arm answers once: it is never sent a second message")

    def outcome(self) -> dict[str, Any]:
        return dict(NO_ROUTE)

    def sources(self) -> list[str]:
        return list(self.computed)

    def notes(self) -> dict[str, Any]:
        return {}


class ModelOnlyShortArm(ModelOnlyArm):
    """The model alone, held to the lead agent's own length: the same prompt with "in at most 450 words". How far its
    score falls below the model alone's says how much of that score is room to write, not thinking."""

    name = "model_only_short"
    prompt = MODEL_ONLY_SHORT_PROMPT


class OpenArm:
    name = "open"

    def __init__(self, client: Any):
        self.client = client
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": prompts.text(OPEN_PROMPT)}]
        self.upload: Optional[str] = None
        self.frame: Any = None
        self.computed: list[str] = []
        self.group_names: set[str] = set()
        self.calls: list[dict[str, Any]] = []  # every tool call and whether it was answered or refused (the arm's own trace)

    def open(self, person: Mapping[str, Any]) -> dict[str, Any]:
        self.upload = person.get("upload") or None
        if self.upload:
            from ..home import tables

            self.frame = tables.build(self.upload)
        return self.send(str(person["opening"]) + (f"\n\n(I shared a table: {self.upload}.csv)" if self.upload else ""))

    def send(self, words: str) -> dict[str, Any]:
        self.messages.append({"role": "user", "content": words})
        said: list[str] = []
        tools = READ_TOOLS if self.upload else None
        for n in range(MAX_CALLS):
            # the last request of a turn is for its words: no tool is offered then, so a turn never ends silent; and the model
            # is told so (HAB-006: three of eight conversations on Nemotron 3 Ultra ended on a tool call written as text)
            last = n == MAX_CALLS - 1
            if last and self.messages[-1].get("role") == "tool":
                self.messages.append({"role": "user", "content": LAST_REQUEST})
            out = self.client.complete(self.messages, None if last else tools, max_tokens=MAX_TOKENS)
            self.messages.append(out.get("assistant_message") or {"role": "assistant", "content": out.get("content") or ""})
            if str(out.get("content") or "").strip():
                said.append(str(out["content"]).strip())
            calls = out.get("tool_calls") or []
            if not calls:
                break
            for c in calls:
                value = self._run(c["name"], c.get("arguments") or {})
                self.calls.append({"tool": c["name"], "ok": "error" not in value})
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(value, default=str)})
        return {"text": "\n\n".join(said), "choices": []}

    def _run(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        from dclab_rnd.agents import data_tools

        if self.frame is None:
            return {"error": "no table has been shared"}
        if name == "profile_table":
            value: dict[str, Any] = {"table": f"{self.upload}.csv", "rows": len(self.frame), "columns": table_view(self.upload)}
        elif name in ("query_table", "check_leakage", "preview_step"):
            fn = {"query_table": "query", "check_leakage": "leakage", "preview_step": "preview_step"}[name]
            kept = {k: v for k, v in args.items() if k in set(data_tools.SCHEMAS[fn]["properties"])}  # the schema the lead agent's registry checks
            if fn == "query":
                kept["seen"] = self.group_names
            try:
                value = data_tools.TOOLS[fn](self.frame, **kept)
            except TypeError as error:
                return {"error": f"the arguments do not fit the tool: {error}"[:300]}
        else:
            return {"error": f"no tool {name!r} here"}
        if "error" not in value:
            self.computed.append(json.dumps(value, default=str))
        return value

    def outcome(self) -> dict[str, Any]:
        return dict(NO_ROUTE)

    def sources(self) -> list[str]:
        return list(self.computed)

    def notes(self) -> dict[str, Any]:
        return {"tool_calls": list(self.calls)}


def _as_percents(value: Any) -> str:
    """Every share code computed (a number between 0 and 1), written as the blueprint shows it: 0.057 as 5.7% and 6%."""
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, Mapping):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, float) and 0.0 < v < 1.0:
            found.extend((f"{round(v * 100, 1)}%", f"{round(v * 100)}%"))

    walk(value)
    return " ".join(found)


BLUEPRINT_CAP_EUR = 0.12  # one blueprint's own cap, the no-data evaluation's (the run's cap still holds over all of them)


class BlueprintArm:
    """A no-data design answering a person once (2026-11-16, the no-data plan): the person's words in, the blueprint as
    the person would read it out (its Markdown). ``blueprint_rules``: no model. ``blueprint_pipeline``: today's design,
    many model roles behind five gates. ``blueprint_architect``: the rebuilt one (one free pass, code checks, one
    critic). Compared on understanding, the solution and the plan, never on work done: nothing is built without data.

    What code computed from the person's own numbers (the decision economics) is a source of the reply's numbers, as a
    tool's result is for the lead agent; a default DCLab chose is not, so it counts as a number nobody gave."""

    def __init__(self, name: str, gateway: Any, person: str):
        self.name, self.gateway, self.person = name, gateway, person
        self.computed: list[str] = []
        self.kept: dict[str, Any] = {}

    def _solve(self, words: str) -> tuple[dict[str, Any], str]:
        if self.name == "blueprint_architect":
            from dclab_rnd import nodata  # the rebuilt design

            solution = nodata.solve(words, gateway=self.gateway, draft_id=f"exam:{self.name}-{self.person}", cap_eur=BLUEPRINT_CAP_EUR)
            return solution, nodata.markdown(solution)
        from dclab_rnd.draft import no_data

        solution = no_data.solve_without_data(words, gateway=self.gateway, models=self.gateway, draft_id=f"exam:{self.name}-{self.person}",
                                              cap_eur=BLUEPRINT_CAP_EUR if self.gateway is not None else 0.0)
        return solution, no_data.to_markdown(solution)

    def open(self, person: Mapping[str, Any]) -> dict[str, Any]:
        solution, text = self._solve(str(person["opening"]))
        economics = solution.get("economics") or {}
        self.computed = [str(s) for s in economics.get("as_said") or []] + [json.dumps(economics, default=str, ensure_ascii=False), _as_percents(economics)]
        report = solution.get("harness_report") or {}
        self.kept = {"family": solution.get("family"), "mode": report.get("mode"), "roles_called": list(report.get("roles_called") or []),
                     "gates_passed": {k: bool((g or {}).get("passed")) for k, g in (report.get("gates") or {}).items()},
                     "cost_eur": report.get("cost_eur"), "assumed_inputs": list(economics.get("assumed") or []),
                     "fits": (solution.get("fit") or {}).get("fits")}
        return {"text": str(text or "").strip(), "choices": []}

    def send(self, words: str) -> dict[str, Any]:
        raise RuntimeError("a blueprint arm answers once: it is never sent a second message")

    def outcome(self) -> dict[str, Any]:
        return dict(NO_ROUTE)

    def sources(self) -> list[str]:
        return list(self.computed)

    def notes(self) -> dict[str, Any]:
        return dict(self.kept)

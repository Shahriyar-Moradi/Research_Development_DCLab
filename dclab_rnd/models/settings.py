"""Which model serves which purpose, and what each purpose may be shown (package A1, docs/guides/AGENTIC_FOUNDATION_PLAN.md).

A *tier* is an OpenAI-compatible endpoint and model: ``strong``, ``standard`` or ``cheap``. Each is set from the
environment and falls back to the standard one, which falls back to OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL:

    DCLAB_TIER_CHEAP_BASE_URL=http://127.0.0.1:11434/v1   DCLAB_TIER_CHEAP_MODEL=qwen2.5-coder:1.5b   (a local model)
    DCLAB_TIER_STRONG_MODEL=…   DCLAB_TIER_<TIER>_API_KEY=…   (DCLAB_INTERN_MODEL still sets the model of every tier that names none)

Anthropic (``DCLAB_LLM_PROVIDER=anthropic``): the standard tier, and every tier that names no endpoint of its own, is
Anthropic's Messages API (models/anthropic_client.py) with ``ANTHROPIC_API_KEY`` and ``claude-sonnet-5-5`` unless
``DCLAB_TIER_<TIER>_MODEL`` or ``ANTHROPIC_MODEL`` names another; the OpenAI key and model are then never used. A tier
whose endpoint is ``https://api.anthropic.com`` is Anthropic's whatever the provider setting says.

A *purpose* names why a request is made. It decides the tier, the timeout, and the data the caller may put in the
prompt. The table is the one place to read what any model is ever shown; the Admin page reads it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .client import DEFAULT_MODEL, REASONING_FIELDS
from .live_guard import blocked, owner_value  # the owner's providers and settings, also for a run that never read .env

TIERS = ("strong", "standard", "cheap")
TUNED = "tuned"  # the policy model (A6.3/A6.4): only from DCLAB_TIER_TUNED_*; it never falls back to another model
ROUTABLE = (*TIERS, TUNED)  # what a purpose can be served or shadowed by (models/routing.py)


@dataclass(frozen=True)
class Purpose:
    tier: str
    timeout: float  # seconds per attempt
    may_see: str  # in words, for the Admin page and for reviewers of a caller
    stream: bool = False
    cell_values: int = 0  # most distinct cell values of the user's data one request may contain (tests/test_model_data_limits.py)
    keeps_text: bool = False  # 23.10: its prompts and replies may be kept (DCLAB_LOG_PROMPTS) and, redacted, leave under train: only a purpose shown the person's own words and bands


PURPOSES: dict[str, Purpose] = {
    "home_intent": Purpose("standard", 60, "H2: the person's request and their replies to Home's questions, to read which kind of help they need; "
                                           "never data, column names or cell values", cell_values=0),
    "home_query": Purpose("standard", 60, "H6: the person's question, what Home understood (the brief) and each column's name, kind, "
                                          "missing rate and number of distinct values; never a row or a cell value: it writes a query plan, "
                                          "code checks it and runs it, and groups under the row floor (DCLAB_ROW_FLOOR, default 5) are never shown",
                          cell_values=0, keeps_text=True),
    "home_business": Purpose("standard", 60, "H11: the decision brief DCLab's code wrote from a comparison of two periods (totals, "
                                             "the change per group, at most 8 group names of the person's table); never a row: it "
                                             "rewords the brief, and code keeps a sentence only when cited.py passes it",
                             cell_values=8),
    "home_agent": Purpose("standard", 90, "the problem sentence, the user's answers, column summaries (name, kind, missing rate, unique count) "
                                          "and descriptive findings; never rows or cell values", stream=True, cell_values=0, keeps_text=True),
    "memory_embedding": Purpose("cheap", 60, "35.5: the text of remembered lines (a project's notes, facts, accepted lessons, a person's "
                                             "confirmed preferences) and the question asked of memory, to rank them by meaning; served "
                                             "only by a local model (DCLAB_MEMORY_EMBEDDING_*), never a remote one; no rows, no cell values"),
    "embedding": Purpose("cheap", 60, "the text of the evidence records (rules, pitfalls, leakage precedents, experiments) and the question typed into the "
                                      "evidence search, to find the records that answer it; no rows, no cell values, and not a workspace's lessons. "
                                      "35.0's measurement also sends its invented memory items and questions, on a local model only"),
    "parse_pattern": Purpose("cheap", 60, "up to 20 lines of a file no built-in reader can parse (400 characters each); "
                                          "DCLAB_MODEL_READS_SAMPLE_LINES=0 turns this off", cell_values=20 * 40),
    "synthetic_schema": Purpose("standard", 120, "the description of the table to simulate, the problem sentence and the user's answers; no data", keeps_text=True),
    "journey_writer": Purpose("standard", 300, "S8 B1: one replica kind's family, its invented table's column names, a list of businesses and two "
                                               "written examples of the format; never a person's data, a row or a cell value", keeps_text=True),
    "journey_checker": Purpose("standard", 180, "S8 B1: one replica kind's family, its invented table's column names and the written people's "
                                                "words and answers; never a person's data, a row or a cell value", keeps_text=True),
    "evidence_answer": Purpose("cheap", 60, "the question and the text of the evidence records it retrieved, which can include this workspace's accepted "
                                            "lessons (a project's aggregate results, its name, dataset file name, row count and split); no rows"),
    "stage_notes": Purpose("standard", 60, "a stage record's title, summary, claims with their limits (aggregate results with their intervals), its notes "
                                           "and the rule records they cite; no rows"),
    "lesson_proposal": Purpose("standard", 60, "a project's final stage claims with their limits (aggregate results with their intervals, "
                                               "the model, the recipe and the tuning decision); no rows"),
    "project_answer": Purpose("standard", 60, "the question, facts computed from the project's stage records, and evidence records (with this "
                                           "workspace's accepted lessons: other projects' aggregate results and scope); no rows"),
    "home_orchestrator": Purpose("standard", 90, "a built project's chat: the person's words (redacted) and the conversation's texts, the project's state (stage "
                                                 "statuses, the chosen model, its metric and held-back score, the problem as built and the table's column "
                                                 "names) and what its specialists' tools return from the "
                                                 "project's records (model families and their fold scores, recipes, column names flagged by the data and "
                                                 "leakage stages, evaluation measures and confusion counts, an improvement run's ways and gains); never rows or "
                                                 "cell values", keeps_text=True),
    "notebook_review": Purpose("standard", 60, "the notebook copilot's findings (title, message and suggestion, which name columns, variables and functions of the "
                                               "notebook; a path, a string selector or a line of code is replaced before the request) and the text of the rule and "
                                               "pitfall records they cite; never the notebook's code or any data"),
    "leakage_review": Purpose("standard", 60, "the prediction moment, the target's name, column names with their kind, missing rate and unique "
                                              "count, the column audit's flags, and the ids and titles of rule and leakage-precedent records; never rows or cell values"),
    "problem_analyst": Purpose("standard", 60, "the problem's description (the person's own words, up to 20 paragraphs) and the names of their "
                                               "files and columns (dclab_core's problem analyst, package 18.2); never rows or cell values"),
    "conversation_reader": Purpose("standard", 60, "the person's problem description, or one question's field and the person's answer to it, and the table's column names "
                                                   "(dclab_core's conversation reader, package 20.2); never rows or cell values", keeps_text=True),
    "ai_first": Purpose("standard", 180, "phase 3's AI-first agent (Claude as a machine-learning expert, 25.3): the problem's description and the person's messages, "
                                        "the table's profile (names, kinds, missing rates, unique counts; no example values), the stages' compact records, and, on Home with "
                                        "the owner's sample switch on (25.4), up to 20 redacted rows; its whole exchange is kept in the twin folder", cell_values=20 * 40,
                        keeps_text=True),
    "decomposer": Purpose("standard", 90, "the person's problem description and messages, the problem card's family, labels, unit and moment, what the rules drew as short lines "
                                          "(parts, families, labels, ranks, reads) and the draft's table names with their column names and purposes "
                                          "(dclab_core's decomposer, package 24.2); never rows or cell values", keeps_text=True),
    "map_reviewer": Purpose("standard", 60, "the person's problem description and messages, the problem map as short lines, the draft's table names with their column "
                                            "names and purposes, and the codes of the questions the rules already ask (dclab_core's map reviewer, package 24.2); "
                                            "never rows or cell values", keeps_text=True),
    "data_need": Purpose("standard", 60, "the problem map as short lines and the draft's table names with their column names, purposes and what one row of each is "
                                         "(dclab_core's data-need agent, package 24.2); never rows or cell values"),
    "solution_router": Purpose("standard", 60, "one sub-problem of the problem map as a short line, the question, the catalogue's candidates as lines (approach, what blocks it, "
                                               "what is not known yet) and a line about each approach (dclab_core's solution router, package 24.3); never rows or cell values"),
    "brief_checker": Purpose("standard", 60, "the problem's description, the names of their files and columns, and what dclab_core's rules "
                                             "read from the description with the words it rests on (package 18.2's follow-up); never rows "
                                             "or cell values"),
    "solution_planner": Purpose("standard", 90, "the problem brief's fields, the data findings' ids, topics and statements (summaries), and "
                                                "the rule plan's decisions (dclab_core's solution planner, package 18.4); never rows or cell values"),
    "reporter": Purpose("standard", 90, "the solution report's sections and their lines, as code wrote them from the run's records "
                                        "(aggregates, column names, decisions and their reasons; dclab_core's reporter, package 18.7), "
                                        "including the target's class names (per-class results, confused pairs) and its range; never a "
                                        "row or a value of an input column", cell_values=40),
    "critic": Purpose("standard", 90, "the error analysis' weaknesses (ids and statements), the screened model's name and training-CV score, "
                                      "and the catalogue of experiments (dclab_core's critic, package 18.6); never rows or cell values"),
    "feature_engineer": Purpose("standard", 90, "the goal, the prediction moment, the task, the target's name, the entity and time columns, "
                                                "the coordinate pairs and each column's name, kind, missing share and distinct share "
                                                "(dclab_core's feature engineer, package 18.5); never rows or cell values"),
    "data_investigator": Purpose("standard", 60, "the prediction moment, the target's name and the task, and each column's name, kind, "
                                                 "missing share and distinct share (dclab_core's data investigator, package 18.3); "
                                                 "never rows or cell values"),
    "reflector": Purpose("standard", 60, "one graded attempt at an invented challenge: the grader's codes (kinds of mistake, never a column), the steps the solver took with their verdicts, and the family's current lessons; never the table, a column's name, a value or the answer key (the harness that learns, H2)", keeps_text=True),
    "understanding_critic": Purpose("standard", 60, "the problem as DCLab understood it (family, how outcomes are known, the outcome column's "
                                                    "name, the prediction moment, the unit and the action in the person's words), each column's "
                                                    "name, purpose, role, type, missing and distinct share, and the checks' mismatches (package "
                                                    "28.4); never rows or cell values", keeps_text=True),
    "solution_challenger": Purpose("cheap", 90, "what the solution planner sees (the problem brief's fields, the data findings' ids, topics "
                                                "and statements, the rule plan's decisions) and the first designer's changes (package 28.6, the "
                                                "debate); never rows or cell values"),
    "debate_judge": Purpose("cheap", 60, "the problem brief's fields, the data findings' ids, topics and statements, the rule plan's "
                                         "decisions and the two designers' disputed choices (package 28.6, the debate); never rows or cell values"),
    "decision_answer": Purpose("cheap", 60, "one decision's question and options, and what is known about what it decides on: a column's "
                                            "name, its name tokens and bands (storage type, share of distinct and missing values, the shape of "
                                            "its written values) and the outcome column's name (package 22.5); a table's, a split's, an image "
                                            "set's or a conversation's counts, flags, codes and bands (29.9); never rows or cell values", keeps_text=True),
    "solver_answer": Purpose("standard", 90, "one question about a problem as DCLab describes it: each column's name (or a masked name), "
                                             "its bands (storage type, share of distinct and missing values, the shape of its written values), "
                                             "the outcome's name, the prediction moment, measures computed on training rows only (a column's "
                                             "signal against the outcome, the holdout's metrics once read); never rows or cell values (27.6)"),
    "case_agent": Purpose("standard", 180, "37.3: the lead agent: the person's words, the case file (facts with their sources, assumptions, "
                                           "unknowns, readiness), and, once a table is shared, its column names, kinds, what DCLab noticed "
                                           "and its row count; never a row or a cell value", cell_values=0, keeps_text=True),
    "exam_plain": Purpose("standard", 180, "37.0: the conversation exam's Claude-alone arm: a written person's words (simulated, never a "
                                           "real person's) and, when they share an invented exam table, its column names, kinds and what "
                                           "DCLab noticed about each; never a row or a cell value", cell_values=0, keeps_text=True),
    "exam_person": Purpose("standard", 90, "37.0: the simulated person of the conversation exam (served by Haiku 5.5): a written person's "
                                           "hidden truth and the conversation so far; never data, a row or a cell value", cell_values=0, keeps_text=True),
    "exam_judge": Purpose("standard", 240, "37.0: the conversation exam's judge (served by Opus 5.5), blind to the arm: a written person's "
                                           "hidden truth, the points to understand, the trap and the conversation; never data, a row or a "
                                           "cell value", cell_values=0, keeps_text=True),
    "home_understand": Purpose("standard", 90, "the owner's Telco test (2026-10-29): Home's understanding conversation: the person's words, "
                                               "Home's own earlier understanding questions, and the table's column names and kinds with what "
                                               "DCLab noticed about each (an identifier, an outcome-like name, a date, a small code); never a row "
                                               "or a cell value", cell_values=0, keeps_text=True),
    "routing_labeler": Purpose("standard", 90, "P8 (2026-10-29): one routing decision Home made, as a blind labeller sees it: the person's "
                                               "request words, the decision's question, its options with their meanings and the state's bands; "
                                               "never the answer Home or any answerer took, never data, column names or cell values", cell_values=0),
    "decision_labeler": Purpose("standard", 60, "what a blind reviewer sees of one decision: its point, question and options, its subject "
                                                "(a column's or table's name) and the state (the column's name, its name tokens and bands, "
                                                "the outcome column's name; for a project's or a workflow's decision, its id and aggregate measures such as a "
                                                "cross-validated mean, a holdout value or an apparent lift, and codes such as the metric or the "
                                                "split); never DCLab's answer, rows or cell values (blind labels by a model)", keeps_text=True),
    "workspace_chat": Purpose("standard", 90, "a person's question about the workspace's own tables, and the results of the tools it calls: each "
                                            "table's name, columns and rows, each column's kind, missing share and distinct count, and computed "
                                            "aggregates (a count, a sum, a mean, by up to two groups, at most fifty each), personal values masked "
                                            "(package 30.21); never a row", cell_values=2 * 50, keeps_text=True),
    "reasoning_teacher": Purpose("standard", 90, "one record of the store (scrubbed: its fields and codes, the person's words redacted, never a "
                                                 "row or a cell value) and the answer it led to, to write the reasoning behind it (32.10)", keeps_text=True),
    "notebook_cell": Purpose("standard", 90, "the person's request for a notebook cell, the notebook's code cells as DCLab generated them "
                                             "from the workflow, and the table's column names with their kinds; never a row, a cell value or an "
                                             "output of the kernel", keeps_text=True),
    "store_curator": Purpose("cheap", 60, "the store's records in their shape only (32.9): each record's kind, type, capture flags, its ledger "
                                          "point and the names and codes of its body's fields (never their values' words, rows or cell values), "
                                          "and the eight model types with their folders"),
    "metric_question": Purpose("cheap", 60, "a business question about a number, as the person typed it, and the semantic layer's menu: the "
                                            "tested templates, the approved metrics' names and definitions, each dimension's values (at most fifty "
                                            "each: a mall's, a brand's or a category's name), what the data does not hold, and the data's first "
                                            "and last day (package 30.20); never a row or a number of the tables", cell_values=3 * 50, keeps_text=True),
    "intern": Purpose("strong", 180, "the task, and the results of the tools it calls: data profiles (with up to three example values per "
                                     "column from describe_data), stage records and evidence records (with this workspace's accepted lessons: other "
                                     "projects' aggregate results and scope); never whole rows", cell_values=3 * 200),
    "campaign": Purpose("strong", 180, "research campaign context: the plan, dataset cards and experiment summaries of evidence/campaigns; "
                                       "no project or user data"),
    "campaign_review": Purpose("strong", 180, "one research campaign result's compact evidence (metrics, claims, setup) from evidence/campaigns; "
                                              "no project or user data"),
}


def _env(tier: str, name: str) -> str:
    return os.environ.get(f"DCLAB_TIER_{tier.upper()}_{name}", "").strip()


TOOL_ARGUMENTS = ("json", "object")


def options(name: str) -> dict[str, Any]:
    """A tier's options for an OpenAI-compatible server (D2, 2026-10-23), each chosen per endpoint and never guessed from
    the model's name: ``reasoning_field``, the reply field that carries the model's reasoning, read from every reply and
    sent back with every earlier assistant turn (none by default: a server that returns none is asked for none);
    ``chat_template_kwargs``, JSON sent as ``extra_body`` on every request (Qwen's ``{"preserve_thinking": true}`` keeps
    the earlier turns' reasoning in the prompt); ``tool_arguments``, how an earlier turn's tool arguments go back:
    ``json`` (a string, the OpenAI wire format) or ``object`` (a dict, what Qwen's template renders)."""
    field = _env(name, "REASONING_FIELD").lower()
    if field in ("", "none"):
        field = None
    elif field not in REASONING_FIELDS:
        raise ValueError(f"DCLAB_TIER_{name.upper()}_REASONING_FIELD must be none, {' or '.join(REASONING_FIELDS)}, not {field!r}")
    raw = _env(name, "CHAT_TEMPLATE_KWARGS")
    try:
        kwargs = json.loads(raw) if raw else {}
    except json.JSONDecodeError as error:
        raise ValueError(f"DCLAB_TIER_{name.upper()}_CHAT_TEMPLATE_KWARGS is not JSON: {error}") from None
    if not isinstance(kwargs, dict):
        raise ValueError(f"DCLAB_TIER_{name.upper()}_CHAT_TEMPLATE_KWARGS must be a JSON object")
    arguments = _env(name, "TOOL_ARGUMENTS").lower() or "json"
    if arguments not in TOOL_ARGUMENTS:
        raise ValueError(f"DCLAB_TIER_{name.upper()}_TOOL_ARGUMENTS must be {' or '.join(TOOL_ARGUMENTS)}, not {arguments!r}")
    # 2026-10-29: a slow endpoint (a 4B model thinking on an 8 GB laptop) needs longer than a purpose's seconds, which are
    # set for a hosted model: DCLAB_TIER_<TIER>_TIMEOUT is a floor in seconds for every request of this tier, never a cut
    raw_timeout = _env(name, "TIMEOUT")
    try:
        timeout = float(raw_timeout) if raw_timeout else None
    except ValueError:
        raise ValueError(f"DCLAB_TIER_{name.upper()}_TIMEOUT must be a number of seconds, not {raw_timeout!r}") from None
    if timeout is not None and timeout <= 0:
        raise ValueError(f"DCLAB_TIER_{name.upper()}_TIMEOUT must be above zero")
    raw_tokens = _env(name, "MAX_TOKENS")  # a floor for every request's output limit (a local model that thinks at length)
    try:
        tokens = int(raw_tokens) if raw_tokens else None
    except ValueError:
        raise ValueError(f"DCLAB_TIER_{name.upper()}_MAX_TOKENS must be a whole number of tokens, not {raw_tokens!r}") from None
    if tokens is not None and tokens <= 0:
        raise ValueError(f"DCLAB_TIER_{name.upper()}_MAX_TOKENS must be above zero")
    return {"reasoning_field": field, "chat_template_kwargs": kwargs, "tool_arguments": arguments, "timeout_floor": timeout,
            "max_tokens_floor": tokens}


def _host(url: str) -> str:
    """Host and port only: a URL can carry a user and a password (https://u:pw@host), which must never be shown or logged."""
    parsed = urlparse(url)
    return (parsed.hostname or url) + (f":{parsed.port}" if parsed.port else "")


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local(url: str, tier_name: str | None = None) -> bool:
    """A model server on this machine: its host is localhost (a substring such as "localhost.example.com" or ":11434"
    is not enough). A server elsewhere on a private network counts as local only with DCLAB_TIER_<TIER>_LOCAL=1."""
    if tier_name and _env(tier_name, "LOCAL") == "1":
        return True
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


ANTHROPIC_HOST = "api.anthropic.com"


def provider(base_url: str) -> str:
    """Which API an endpoint speaks: "anthropic" (the Messages API, at Anthropic's host with no path) or "openai" (chat
    completions; Anthropic's OpenAI-compatible endpoint ``https://api.anthropic.com/v1`` is one)."""
    parsed = urlparse(base_url)
    return "anthropic" if (parsed.hostname or "") == ANTHROPIC_HOST and parsed.path.strip("/") == "" else "openai"


def default_model(base_url: str) -> str:
    """OpenAI's own endpoint keeps its default model; any other endpoint (Ollama, vLLM, a gateway) has no default, since
    a vendor's model name sent there fails in a way that is hard to read: the tier is then unavailable until a model is named."""
    return DEFAULT_MODEL if _host(base_url) == "api.openai.com" else ""


def _claude_model() -> str:
    from .anthropic_client import DEFAULT_MODEL as CLAUDE

    return owner_value("ANTHROPIC_MODEL").strip() or CLAUDE


def anthropic_default() -> bool:
    return owner_value("DCLAB_LLM_PROVIDER").strip().lower() == "anthropic"


EMBEDDING_LOCAL = "embedding_local"  # 35.5: the local embedding model, serving the memory_embedding purpose only


def embedding_tier() -> dict[str, Any] | None:
    """35.5: DCLAB_MEMORY_EMBEDDING_BASE_URL and DCLAB_MEMORY_EMBEDDING_MODEL name a local embedding model (35.0 chose
    ``embeddinggemma:300m-qat-q8_0`` on Ollama) that serves the gateway's ``memory_embedding`` purpose and nothing else
    (the evidence search keeps its own measured model, DCLAB_EMBEDDING_MODEL). Only an endpoint on this machine is taken:
    memory's words never leave it to be embedded. None when unset or elsewhere."""
    base_url = owner_value("DCLAB_MEMORY_EMBEDDING_BASE_URL").strip()
    model = owner_value("DCLAB_MEMORY_EMBEDDING_MODEL").strip()
    if not base_url or not model or not is_local(base_url):
        return None
    return {"name": EMBEDDING_LOCAL, "base_url": base_url, "model": model, "api_key": "local", "provider": "openai", **options(EMBEDDING_LOCAL)}


def tier(name: str) -> dict[str, Any]:
    """A tier's endpoint, model and key. The key is returned for the transport only; never put it in a response or a log.

    An unset tier is the standard one: its endpoint, its model and its key. A tier that names its own endpoint gets a
    key only from DCLAB_TIER_<TIER>_API_KEY, or the standard key when the host is the same: a key is never sent to
    another provider. DCLAB_INTERN_MODEL, the single model setting from before tiers existed, still applies to every tier
    that names no model.
    """
    if name == EMBEDDING_LOCAL:
        local = embedding_tier()
        if local is None:
            raise KeyError(name)
        return local
    if name not in ROUTABLE:
        raise KeyError(name)
    if name == TUNED:  # unset means absent: a tuned model that silently became the standard one would agree with itself
        base_url, model = _env(name, "BASE_URL"), _env(name, "MODEL")
        if not base_url or not model:
            return {"name": name, "base_url": "", "model": "", "api_key": "", "provider": "openai", **options(name)}
        key = _env(name, "API_KEY") or ("local" if is_local(base_url, name) else "")
        return {"name": name, "base_url": base_url, "model": model, "api_key": key, "provider": provider(base_url), **options(name)}
    if name == "standard" and anthropic_default() and not _env(name, "BASE_URL"):
        from .anthropic_client import BASE_URL

        base_url = BASE_URL
        model = _env(name, "MODEL") or _claude_model()
        key = _env(name, "API_KEY") or owner_value("ANTHROPIC_API_KEY").strip()
    elif name == "standard":
        base_url = _env(name, "BASE_URL") or os.environ.get("DCLAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = _env(name, "MODEL") or (_claude_model() if provider(base_url) == "anthropic" else
                                         os.environ.get("DCLAB_INTERN_MODEL") or os.environ.get("OPENAI_MODEL") or default_model(base_url))
        key = _env(name, "API_KEY") or (os.environ.get("ANTHROPIC_API_KEY", "").strip() if provider(base_url) == "anthropic"
                                         else os.environ.get("OPENAI_API_KEY") or "")
    else:
        standard = tier("standard")
        own_url = _env(name, "BASE_URL")
        base_url = own_url or standard["base_url"]
        same_host = not own_url or _host(own_url) == _host(standard["base_url"])
        if _env(name, "MODEL"):
            model = _env(name, "MODEL")
        elif own_url and provider(own_url) == "anthropic" != standard["provider"]:
            model = _claude_model()
        elif not same_host:  # another server: never the standard tier's model name (a vendor's model sent to Ollama fails unreadably)
            model = os.environ.get("DCLAB_INTERN_MODEL") or default_model(own_url)
        else:
            model = standard["model"]
        key = _env(name, "API_KEY") or (standard["api_key"] if same_host else "")
        if not key and provider(base_url) == "anthropic":
            key = owner_value("ANTHROPIC_API_KEY").strip()
    if not key and is_local(base_url, name):
        key = "local"  # a local server (Ollama, LM Studio) needs no key; the SDK still wants one
    who = blocked(base_url)
    if who:  # the owner blocked this provider (2026-10-25: Claude only): the tier is unavailable, its key never sent
        return {"name": name, "base_url": "", "model": "", "api_key": "", "provider": provider(base_url), "blocked": who, **options(name)}
    return {"name": name, "base_url": base_url, "model": model, "api_key": key, "provider": provider(base_url), **options(name)}


def public(name: str) -> dict[str, Any]:
    """A tier as the pages show it: no key, only whether one is set."""
    t = tier(name)
    try:
        if t["provider"] == "anthropic":
            import anthropic  # noqa: F401
        else:
            import openai  # noqa: F401
        sdk = True
    except ImportError:
        sdk = False
    return {"name": name, "endpoint": _host(t["base_url"]) if t["base_url"] else "not set", "model": t["model"], "sdk_installed": sdk,
            "provider": t["provider"],
            "key_configured": bool(t["api_key"]), "model_configured": bool(t["model"]), "available": bool(t["api_key"]) and sdk and bool(t["model"]),
            "local": bool(t["base_url"]) and is_local(t["base_url"], name),
            "reasoning_field": t["reasoning_field"], "chat_template_kwargs": t["chat_template_kwargs"]}


# The owner's rule (2026-10-28 night, "what I want is the model's thinking kept for training", for every agent and every
# kind of help): every purpose asks for the model's thinking, kept whole with each exchange (gateway, 32.10), except the
# short list below, each with its reason. DCLAB_THINKING: "off", "all", or a comma list of purposes, overrides it.
NO_THINKING: dict[str, str] = {
    "embedding": "an embedding call returns vectors, never text: there is no thinking to ask for or keep",
    "memory_embedding": "an embedding call returns vectors, never text: there is no thinking to ask for or keep",
}
THINKING_DEFAULT = tuple(n for n in PURPOSES if n not in NO_THINKING)  # every purpose but the exceptions (kept for callers)


def thinking_for(purpose: str) -> str | None:
    """"adaptive" when this purpose asks for the model's thinking, else None."""
    raw = os.environ.get("DCLAB_THINKING", "").strip().lower()
    if raw in ("off", "0", "none", "false"):
        return None
    if raw == "all":
        return None if purpose in NO_THINKING else "adaptive"
    if raw:
        return "adaptive" if purpose in {x.strip() for x in raw.split(",") if x.strip()} else None
    return None if purpose in NO_THINKING else "adaptive"


def purpose(name: str) -> Purpose:
    if name not in PURPOSES:
        raise KeyError(f"unknown purpose {name!r}: add it to dclab_rnd/models/settings.py PURPOSES")
    return PURPOSES[name]

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

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .client import DEFAULT_MODEL

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
    "home_agent": Purpose("standard", 90, "the problem sentence, the user's answers, column summaries (name, kind, missing rate, unique count) "
                                          "and descriptive findings; never rows or cell values", stream=True, cell_values=0, keeps_text=True),
    "embedding": Purpose("cheap", 60, "the text of the evidence records (rules, pitfalls, leakage precedents, experiments) and the question typed into the "
                                      "evidence search, to find the records that answer it; no rows, no cell values, and not a workspace's lessons"),
    "parse_pattern": Purpose("cheap", 60, "up to 20 lines of a file no built-in reader can parse (400 characters each); "
                                          "DCLAB_MODEL_READS_SAMPLE_LINES=0 turns this off", cell_values=20 * 40),
    "synthetic_schema": Purpose("standard", 120, "the description of the table to simulate, the problem sentence and the user's answers; no data", keeps_text=True),
    "evidence_answer": Purpose("cheap", 60, "the question and the text of the evidence records it retrieved, which can include this workspace's accepted "
                                            "lessons (a project's aggregate results, its name, dataset file name, row count and split); no rows"),
    "stage_notes": Purpose("standard", 60, "a stage record's title, summary, claims with their limits (aggregate results with their intervals), its notes "
                                           "and the rule records they cite; no rows"),
    "lesson_proposal": Purpose("standard", 60, "a project's final stage claims with their limits (aggregate results with their intervals, "
                                               "the model, the recipe and the tuning decision); no rows"),
    "project_answer": Purpose("standard", 60, "the question, facts computed from the project's stage records, and evidence records (with this "
                                           "workspace's accepted lessons: other projects' aggregate results and scope); no rows"),
    "notebook_review": Purpose("standard", 60, "the notebook copilot's findings (title, message and suggestion, which name columns, variables and functions of the "
                                               "notebook; a path, a string selector or a line of code is replaced before the request) and the text of the rule and "
                                               "pitfall records they cite; never the notebook's code or any data"),
    "leakage_review": Purpose("standard", 60, "the prediction moment, the target's name, column names with their kind, missing rate and unique "
                                              "count, the column audit's flags, and the ids and titles of rule and leakage-precedent records; never rows or cell values"),
    "problem_analyst": Purpose("standard", 60, "the problem's description (the person's own words, up to 20 paragraphs) and the names of their "
                                               "files and columns (dclab_core's problem analyst, package 18.2); never rows or cell values"),
    "conversation_reader": Purpose("standard", 60, "the person's problem description, or one question's field and the person's answer to it, and the table's column names "
                                                   "(dclab_core's conversation reader, package 20.2); never rows or cell values", keeps_text=True),
    "decomposer": Purpose("standard", 90, "the person's problem description and messages, the problem card's family, labels, unit and moment, what the rules drew as short lines "
                                          "(parts, families, labels, ranks, reads) and the draft's table names with their column names and purposes "
                                          "(dclab_core's decomposer, package 24.2); never rows or cell values", keeps_text=True),
    "map_reviewer": Purpose("standard", 60, "the person's problem description and messages, the problem map as short lines, the draft's table names with their column "
                                            "names and purposes, and the codes of the questions the rules already ask (dclab_core's map reviewer, package 24.2); "
                                            "never rows or cell values", keeps_text=True),
    "data_need": Purpose("standard", 60, "the problem map as short lines and the draft's table names with their column names, purposes and what one row of each is "
                                         "(dclab_core's data-need agent, package 24.2); never rows or cell values"),
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
    "decision_answer": Purpose("cheap", 60, "one decision's question and options, and what is known about its column: the column's name, "
                                            "its name tokens and bands (storage type, share of distinct and missing values, the shape of its "
                                            "written values) and the outcome column's name (package 22.5); never rows or cell values", keeps_text=True),
    "decision_labeler": Purpose("standard", 60, "what a blind reviewer sees of one decision: its point, question and options, its subject "
                                                "(a column's or table's name) and the state (the column's name, its name tokens and bands, "
                                                "the outcome column's name; for a project's or a workflow's decision, its id and aggregate measures such as a "
                                                "cross-validated mean, a holdout value or an apparent lift, and codes such as the metric or the "
                                                "split); never DCLab's answer, rows or cell values (blind labels by a model)", keeps_text=True),
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


def _claude_model() -> str:
    from .anthropic_client import DEFAULT_MODEL as CLAUDE

    return os.environ.get("ANTHROPIC_MODEL", "").strip() or CLAUDE


def anthropic_default() -> bool:
    return os.environ.get("DCLAB_LLM_PROVIDER", "").strip().lower() == "anthropic"


def tier(name: str) -> dict[str, Any]:
    """A tier's endpoint, model and key. The key is returned for the transport only; never put it in a response or a log.

    An unset tier is the standard one: its endpoint, its model and its key. A tier that names its own endpoint gets a
    key only from DCLAB_TIER_<TIER>_API_KEY, or the standard key when the host is the same: a key is never sent to
    another provider. DCLAB_INTERN_MODEL, the single model setting from before tiers existed, still applies to every tier
    that names no model.
    """
    if name not in ROUTABLE:
        raise KeyError(name)
    if name == TUNED:  # unset means absent: a tuned model that silently became the standard one would agree with itself
        base_url, model = _env(name, "BASE_URL"), _env(name, "MODEL")
        if not base_url or not model:
            return {"name": name, "base_url": "", "model": "", "api_key": "", "provider": "openai"}
        key = _env(name, "API_KEY") or ("local" if is_local(base_url, name) else "")
        return {"name": name, "base_url": base_url, "model": model, "api_key": key, "provider": provider(base_url)}
    if name == "standard" and anthropic_default() and not _env(name, "BASE_URL"):
        from .anthropic_client import BASE_URL

        base_url = BASE_URL
        model = _env(name, "MODEL") or _claude_model()
        key = _env(name, "API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    elif name == "standard":
        base_url = _env(name, "BASE_URL") or os.environ.get("DCLAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = _env(name, "MODEL") or (_claude_model() if provider(base_url) == "anthropic" else
                                         os.environ.get("DCLAB_INTERN_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL)
        key = _env(name, "API_KEY") or (os.environ.get("ANTHROPIC_API_KEY", "").strip() if provider(base_url) == "anthropic"
                                         else os.environ.get("OPENAI_API_KEY") or "")
    else:
        standard = tier("standard")
        own_url = _env(name, "BASE_URL")
        base_url = own_url or standard["base_url"]
        model = _env(name, "MODEL") or (_claude_model() if own_url and provider(own_url) == "anthropic" != standard["provider"] else standard["model"])
        same_host = not own_url or _host(own_url) == _host(standard["base_url"])
        key = _env(name, "API_KEY") or (standard["api_key"] if same_host else "")
        if not key and provider(base_url) == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and is_local(base_url, name):
        key = "local"  # a local server (Ollama, LM Studio) needs no key; the SDK still wants one
    return {"name": name, "base_url": base_url, "model": model, "api_key": key, "provider": provider(base_url)}


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
            "key_configured": bool(t["api_key"]), "available": bool(t["api_key"]) and sdk,
            "local": bool(t["base_url"]) and is_local(t["base_url"], name)}


def purpose(name: str) -> Purpose:
    if name not in PURPOSES:
        raise KeyError(f"unknown purpose {name!r}: add it to dclab_rnd/models/settings.py PURPOSES")
    return PURPOSES[name]

"""Which model serves which purpose, and what each purpose may be shown (package A1, docs/guides/AGENTIC_FOUNDATION_PLAN.md).

A *tier* is an OpenAI-compatible endpoint and model: ``strong``, ``standard`` or ``cheap``. Each is set from the
environment and falls back to the standard one, which falls back to OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL:

    DCLAB_TIER_CHEAP_BASE_URL=http://127.0.0.1:11434/v1   DCLAB_TIER_CHEAP_MODEL=qwen2.5-coder:1.5b   (a local model)
    DCLAB_TIER_STRONG_MODEL=…   DCLAB_TIER_<TIER>_API_KEY=…   (DCLAB_INTERN_MODEL still sets the model of every tier that names none)

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


@dataclass(frozen=True)
class Purpose:
    tier: str
    timeout: float  # seconds per attempt
    may_see: str  # in words, for the Admin page and for reviewers of a caller
    stream: bool = False
    cell_values: int = 0  # most distinct cell values of the user's data one request may contain (tests/test_model_data_limits.py)


PURPOSES: dict[str, Purpose] = {
    "home_agent": Purpose("standard", 90, "the problem sentence, the user's answers, column summaries (name, kind, missing rate, unique count) "
                                          "and descriptive findings; never rows or cell values", stream=True, cell_values=0),
    "parse_pattern": Purpose("cheap", 60, "up to 20 lines of a file no built-in reader can parse (400 characters each); "
                                          "DCLAB_MODEL_READS_SAMPLE_LINES=0 turns this off", cell_values=20 * 40),
    "synthetic_schema": Purpose("standard", 120, "the description of the table to simulate, the problem sentence and the user's answers; no data"),
    "evidence_answer": Purpose("cheap", 60, "the question and the text of the evidence records it retrieved; no project data"),
    "stage_notes": Purpose("standard", 60, "a stage record's title, summary and claims (aggregate results with their intervals); no rows"),
    "project_answer": Purpose("standard", 60, "the question, facts computed from the project's stage records, and evidence records; no rows"),
    "intern": Purpose("strong", 180, "the task, and the results of the tools it calls: data profiles (with up to three example values per "
                                     "column from describe_data), stage records and evidence records; never whole rows", cell_values=3 * 200),
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


def tier(name: str) -> dict[str, Any]:
    """A tier's endpoint, model and key. The key is returned for the transport only; never put it in a response or a log.

    An unset tier is the standard one: its endpoint, its model and its key. A tier that names its own endpoint gets a
    key only from DCLAB_TIER_<TIER>_API_KEY, or the standard key when the host is the same: a key is never sent to
    another provider. DCLAB_INTERN_MODEL, the single model setting from before tiers existed, still applies to every tier
    that names no model.
    """
    if name not in TIERS:
        raise KeyError(name)
    if name == "standard":
        base_url = _env(name, "BASE_URL") or os.environ.get("DCLAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = _env(name, "MODEL") or os.environ.get("DCLAB_INTERN_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        key = _env(name, "API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    else:
        standard = tier("standard")
        own_url = _env(name, "BASE_URL")
        base_url = own_url or standard["base_url"]
        model = _env(name, "MODEL") or standard["model"]
        same_host = not own_url or _host(own_url) == _host(standard["base_url"])
        key = _env(name, "API_KEY") or (standard["api_key"] if same_host else "")
    if not key and is_local(base_url, name):
        key = "local"  # a local server (Ollama, LM Studio) needs no key; the SDK still wants one
    return {"name": name, "base_url": base_url, "model": model, "api_key": key}


def public(name: str) -> dict[str, Any]:
    """A tier as the pages show it: no key, only whether one is set."""
    t = tier(name)
    try:
        import openai  # noqa: F401
        sdk = True
    except ImportError:
        sdk = False
    return {"name": name, "endpoint": _host(t["base_url"]), "model": t["model"], "sdk_installed": sdk,
            "key_configured": bool(t["api_key"]), "available": bool(t["api_key"]) and sdk,
            "local": is_local(t["base_url"], name)}


def purpose(name: str) -> Purpose:
    if name not in PURPOSES:
        raise KeyError(f"unknown purpose {name!r}: add it to dclab_rnd/models/settings.py PURPOSES")
    return PURPOSES[name]

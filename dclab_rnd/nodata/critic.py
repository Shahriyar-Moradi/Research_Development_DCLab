"""One second reader (the no-data plan of 2026-11-16): a model pass that says what is weak or risky in the solution.

Its findings are shown beside the solution; they never change it. A finding that states a number nobody gave is dropped."""

from __future__ import annotations

import re
from typing import Any, Iterable

from dclab_rnd import cited, prompts

PURPOSE = "no_data_critic"
PROMPT = "no_data_critic"
MAX_TOKENS = 6000
MAX_FINDINGS = 4
NONE = re.compile(r"no serious weakness", re.I)
ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s+")  # a finding's line, bulleted or numbered


def read(client: Any, words: str, solution_text: str, sources: Iterable[str]) -> list[str]:
    """The critic's findings, most serious first; empty when it found none or did not answer."""
    out = client.complete([{"role": "system", "content": prompts.text(PROMPT)},
                           {"role": "user", "content": f"The person's words:\n{str(words).strip()}\n\nThe solution:\n{str(solution_text).strip()}"}],
                          None, max_tokens=MAX_TOKENS)
    said = str((out or {}).get("content") or "").strip()
    sources = list(sources)
    lines = [ITEM.sub("", line.strip()).strip() for line in said.splitlines() if ITEM.match(line.strip())]
    lines = [f for f in lines if f and not NONE.search(f)]  # "no serious weakness" on one line never drops the findings on the others
    return [f for f in lines if not cited.unsaid(f, sources)][:MAX_FINDINGS]

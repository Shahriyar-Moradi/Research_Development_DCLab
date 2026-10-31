"""The first JSON object in a model's reply (a model may wrap it in a code fence): one reader for every agent that asks
for a JSON answer."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def first_json(text: str) -> Optional[dict[str, Any]]:
    text = str(text or "")
    for match in re.finditer(r"\{", text):
        depth = 0
        for end in range(match.start(), len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                try:
                    value = json.loads(text[match.start():end + 1])
                except ValueError:
                    break
                return value if isinstance(value, dict) else None
    return None

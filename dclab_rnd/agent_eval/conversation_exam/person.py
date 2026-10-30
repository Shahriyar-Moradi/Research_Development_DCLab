"""The simulated person (37.0): a separate model plays one written person and answers only from their hidden truth.

It sees its truth (situation, goal, decision, user, data, what it knows, its limits), its level and style, and the
conversation so far with the assistant's numbered choices. It writes one JSON object: what it types, and the number of
the choice it picks or null. A pick outside the choices is no pick; a reply that is not JSON is taken as its words.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from dclab_rnd import prompts

PROMPT = "exam_person"
MAX_TOKENS = 1200


def first_json(text: str) -> Optional[dict[str, Any]]:
    """The first JSON object in a reply (a model may wrap it in a code fence)."""
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


def show(transcript: list[Mapping[str, Any]], *, assistant: str = "Assistant", person: str = "You") -> str:
    """The conversation as text: each message with its speaker, the assistant's choices numbered under it."""
    lines = []
    for turn in transcript:
        who = assistant if turn["role"] == "assistant" else person
        lines.append(f"{who}: {str(turn.get('text') or '').strip() or '(no words)'}")
        for i, choice in enumerate(turn.get("choices") or [], 1):
            what = f" — {choice['what']}" if choice.get("what") else ""
            lines.append(f"   {i}. {choice.get('title')}{what}")
    return "\n".join(lines)


class ModelPerson:
    def __init__(self, client: Any, person: Mapping[str, Any]):
        self.client, self.person = client, person

    def truth(self) -> str:
        p = self.person
        return json.dumps({"level": p.get("level"), "style": p.get("style"), "you shared a table": bool(p.get("upload")),
                           **dict(p.get("truth") or {})}, ensure_ascii=False, indent=1)

    def answer(self, transcript: list[Mapping[str, Any]]) -> dict[str, Any]:
        messages = [{"role": "system", "content": prompts.text(PROMPT)},
                    {"role": "user", "content": f"Your hidden truth:\n{self.truth()}\n\nThe conversation so far:\n{show(transcript)}\n\n"
                                                "Write your next message as one JSON object."}]
        out = self.client.complete(messages, None, max_tokens=MAX_TOKENS)
        text = str(out.get("content") or "")
        value = first_json(text)
        if value is None:
            return {"say": text.strip(), "pick": None}
        choices = (transcript[-1].get("choices") if transcript else None) or []
        pick = value.get("pick")
        pick = pick if isinstance(pick, int) and not isinstance(pick, bool) and 1 <= pick <= len(choices) else None
        return {"say": str(value.get("say") or "").strip(), "pick": pick}

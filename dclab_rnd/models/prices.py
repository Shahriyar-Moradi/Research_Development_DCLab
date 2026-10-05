"""What a model request costs (package A1.2).

Prices are euros per million tokens, input and output, each with the date it was checked and where it came from.
No price is ever guessed: a model without an entry is "price not configured", its requests are counted in tokens
only, and euro caps cannot stop it (the pages say so). A request to a local endpoint (Ollama, LM Studio) costs zero.

Add prices here (checked, dated, with a source) or in a JSON file named by DCLAB_PRICES_FILE:

    {"gpt-example": {"input": 1.10, "output": 4.40, "as_of": "2026-10-01", "source": "provider price page"}}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Price:
    input: float  # euros per million input tokens
    output: float  # euros per million output tokens
    as_of: str
    source: str


PRICES: dict[str, Price] = {}  # deliberately empty: only checked prices belong here


def load() -> dict[str, Price]:
    """The built-in prices, overridden by DCLAB_PRICES_FILE. An entry without a date or with a negative price is refused."""
    out = dict(PRICES)
    path = os.environ.get("DCLAB_PRICES_FILE", "").strip()
    if path and Path(path).is_file():
        try:
            entries = json.loads(Path(path).read_text(encoding="utf-8"))
            for model, raw in entries.items():
                price = Price(float(raw["input"]), float(raw["output"]), str(raw["as_of"]), str(raw.get("source") or ""))
                if not (price.input >= 0 and price.output >= 0) or not price.as_of.strip():  # also refuses NaN
                    raise ValueError(f"the price of {model!r} needs non-negative euros and an as_of date")
                out[model] = price
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError(f"the price file DCLAB_PRICES_FILE is not valid ({type(error).__name__}: {error})") from None
    return out


DEFAULT_OUTPUT_ESTIMATE = 4096  # output tokens assumed when a caller sets no limit


def estimate(model: str, local: bool, prompt_chars: int, max_tokens: int | None) -> float | None:
    """An upper bound on what one request can cost, before it is sent: about three characters per input token (a
    generous count) and the output limit in full. None when the model has no price; zero when local."""
    if local:
        return 0.0
    price = load().get(model)
    if price is None:
        return None
    out = max_tokens if max_tokens is not None else DEFAULT_OUTPUT_ESTIMATE
    return ((prompt_chars / 3) * price.input + out * price.output) / 1_000_000


def cost(model: str, local: bool, input_tokens: int, output_tokens: int) -> tuple[float | None, str]:
    """(euros or None, basis): basis is "price" (from the table), "local" (free) or "no price" (not configured)."""
    if local:
        return 0.0, "local"
    price = load().get(model)
    if price is None:
        return None, "no price"
    return round((input_tokens * price.input + output_tokens * price.output) / 1_000_000, 6), "price"

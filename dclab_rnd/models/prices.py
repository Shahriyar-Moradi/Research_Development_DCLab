"""What a model request costs (package A1.2).

Prices are euros per million tokens, input and output, each with the date it was checked and where it came from.
No price is ever guessed: a model without an entry is "price not configured", its requests are counted in tokens
only, and euro caps cannot stop it (the pages say so). A request to a local endpoint (Ollama, LM Studio) costs zero.

Add prices here (checked, dated, with a source) or in a JSON file named by DCLAB_PRICES_FILE:

    {"gpt-example": {"input": 1.10, "output": 4.40, "as_of": "2026-10-01", "source": "provider price page"}}

Prompt caching (Anthropic): a model with a ``cache_read`` rate prices the tokens its reply read from the cache at that
fraction of the input price, and the tokens it wrote at ``CACHE_WRITE`` (1.25 for the five-minute cache, 2 for the hour).
``caching()`` is the switch the client and the estimate share: DCLAB_PROMPT_CACHE (on unless "off") and
DCLAB_PROMPT_CACHE_TTL ("5m", the default, or "1h").
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
    cache_read: float | None = None  # a cache read as a fraction of the input price (None: the model has no prompt cache here)


CACHE_WRITE = {"5m": 1.25, "1h": 2.0}  # a cache write as a multiple of the input price, by time to live (the same for every model)
CACHE_SOURCE = "platform.claude.com/docs/en/build-with-claude/prompt-caching, checked 2026-10-20"


def caching() -> str | None:
    """The prompt cache's time to live for this process ("5m" or "1h"), or None when it is off. A misspelt TTL is refused,
    never silently replaced: it would change what every request costs."""
    if os.environ.get("DCLAB_PROMPT_CACHE", "on").strip().lower() in ("0", "off", "false", "no"):
        return None
    ttl = os.environ.get("DCLAB_PROMPT_CACHE_TTL", "5m").strip().lower() or "5m"
    if ttl not in CACHE_WRITE:
        raise ValueError(f"DCLAB_PROMPT_CACHE_TTL must be one of {', '.join(CACHE_WRITE)}, not {ttl!r}")
    return ttl


PRICES: dict[str, Price] = {  # only checked prices belong here, with the date and the page they came from
    # The page gives US dollars; they are entered as euros, which overstates the euro spend by the exchange rate
    # (a dollar is worth less than a euro): a cap then stops a little early, never late.
    "text-embedding-3-small": Price(0.02, 0.0, "2026-10-07", "developers.openai.com/api/docs/pricing, standard tier, USD per 1M tokens entered as EUR"),
    "claude-sonnet-5-5": Price(2.0, 10.0, "2026-10-15", "platform.claude.com/docs/en/about-claude/pricing, base input and output, USD per 1M tokens entered as EUR",
                               cache_read=0.05),  # the prompt-caching page, 2026-10-20: Opus 5.5 and Sonnet 5.5 read at 0.05
    "claude-haiku-5-5": Price(0.10, 0.50, "2026-10-15", "platform.claude.com/docs/en/about-claude/pricing, prompts up to 100,000 tokens, USD per 1M tokens entered as EUR",
                              cache_read=0.1),  # the prompt-caching page, 2026-10-20: most models read at 0.1
    "gpt-6-luna": Price(0.10, 0.50, "2026-10-07", "developers.openai.com/api/docs/pricing, standard tier, USD per 1M tokens entered as EUR"),
}


def load() -> dict[str, Price]:
    """The built-in prices, overridden by DCLAB_PRICES_FILE. An entry without a date or with a negative price is refused."""
    out = dict(PRICES)
    path = os.environ.get("DCLAB_PRICES_FILE", "").strip()
    if path and Path(path).is_file():
        try:
            entries = json.loads(Path(path).read_text(encoding="utf-8"))
            for model, raw in entries.items():
                read = raw.get("cache_read")
                price = Price(float(raw["input"]), float(raw["output"]), str(raw["as_of"]), str(raw.get("source") or ""),
                              float(read) if read is not None else None)
                if not (price.input >= 0 and price.output >= 0) or not price.as_of.strip() or (read is not None and not 0 <= price.cache_read <= 1):
                    raise ValueError(f"the price of {model!r} needs non-negative euros and an as_of date")
                out[model] = price
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError(f"the price file DCLAB_PRICES_FILE is not valid ({type(error).__name__}: {error})") from None
    return out


DEFAULT_OUTPUT_ESTIMATE = 8192  # output tokens assumed when a caller sets no limit (Anthropic's requests are sent with this limit)


def price_of(model: str, table: dict[str, Price] | None = None) -> Price | None:
    """The model's price; a dated snapshot of a priced model (``claude-sonnet-5-5-20261001``) takes its alias's."""
    import re

    table = load() if table is None else table
    if model in table:
        return table[model]
    alias = re.sub(r"-\d{8}$", "", model or "")
    return table.get(alias) if alias != model else None


def estimate(model: str, local: bool, prompt_chars: int, max_tokens: int | None) -> float | None:
    """An upper bound on what one request can cost, before it is sent: about three characters per input token (a
    generous count) and the output limit in full. None when the model has no price; zero when local."""
    if local:
        return 0.0
    price = price_of(model)
    if price is None:
        return None
    out = max_tokens if max_tokens is not None else DEFAULT_OUTPUT_ESTIMATE
    ttl = caching()  # whatever the price table says: a request marks its prefix for the cache whenever caching is on
    write = CACHE_WRITE[ttl] if ttl else 1.0  # with the cache on, every input token might be written to it: the bound counts that
    return ((prompt_chars / 3) * price.input * write + out * price.output) / 1_000_000


def cost(model: str, local: bool, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0,
         cache_write_1h: int = 0) -> tuple[float | None, str]:
    """(euros or None, basis): basis is "price" (from the table), "local" (free) or "no price" (not configured).
    ``input_tokens`` is every input token of the request; of them, ``cache_read`` were read from the prompt cache and
    ``cache_write`` written to it (``cache_write_1h`` of those for an hour), each priced at its own rate."""
    if local:
        return 0.0, "local"
    price = price_of(model)
    if price is None:
        return None, "no price"
    read, write, write_1h = max(0, cache_read or 0), max(0, cache_write or 0), max(0, cache_write_1h or 0)
    if not (read or write):  # no cache used: every input token at the input price
        return round((input_tokens * price.input + output_tokens * price.output) / 1_000_000, 6), "price"
    write_1h = min(write_1h, write)
    fresh = max(0, input_tokens - read - write)
    # a price without its cache read rate (a prices file written before caching): reads at the full input price, never
    # below what was charged; writes always carry their premium, which the reply's counts show was paid
    read_rate = price.cache_read if price.cache_read is not None else 1.0
    inputs = (fresh + read * read_rate + (write - write_1h) * CACHE_WRITE["5m"] + write_1h * CACHE_WRITE["1h"]) * price.input
    return round((inputs + output_tokens * price.output) / 1_000_000, 6), "price"

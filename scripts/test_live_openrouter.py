"""Targeted live check of OpenRouter qwen/qwen3.8-flash model with reasoning capture."""

import json
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

from dclab_rnd.models.client import ChatClient
from dclab_rnd import experience


def main():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return 1

    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.8-flash")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    print(f"Testing live OpenRouter model: {model} at {base_url}")

    client = ChatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )

    messages = [
        {"role": "system", "content": "You are a concise data science evaluator. Answer in 1 short sentence."},
        {"role": "user", "content": "Why is holdout split leakage dangerous in machine learning?"},
    ]

    print("Sending completion request...")
    with experience.bind(visit="live-openrouter-check"):
        reply = client.complete(messages, None)

    print("\n--- Response ---")
    print(f"Content: {reply.get('content')}")
    print(f"Usage: {reply.get('usage')}")
    has_reasoning = bool(reply.get("reasoning") or reply.get("thinking") or reply.get("reasoning_details"))
    print(f"Captured reasoning: {has_reasoning}")
    if reply.get("reasoning"):
        print(f"Reasoning snippet: {reply['reasoning'][:120]}...")
    if reply.get("reasoning_details"):
        print(f"Reasoning details count: {len(reply['reasoning_details'])}")

    print("\nLive OpenRouter test successful!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

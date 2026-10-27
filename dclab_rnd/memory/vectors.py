"""Memory searched by meaning (package 35.5): the vectors of remembered lines, made by the local embedding model 35.0
chose (``embeddinggemma:300m-qat-q8_0``), through the gateway's ``memory_embedding`` purpose on this machine only
(``models.settings.embedding_tier``, ``DCLAB_MEMORY_EMBEDDING_*``); the evidence search keeps its own model.

Each text's vector is kept in ``<workspace>/memory/vectors-<model>-<digest>.json``, keyed by a hash of the text and the
model's prefix, so a line is embedded once per model and weights pulled again under the same tag start a new file (the
digest names the weights). Writers take the memory folder's lock (between processes too) and replace the file whole; it
keeps at most :data:`MOST` vectors, the ones asked for last. The question gets the model's query prefix, the items its document
prefix (35.0's measured prefixes). Without a local model, :func:`embedder_for` gives None and recall searches by words.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

MOST = 20000  # vectors kept in one file; the ones asked for last

# the prefixes each candidate's card asks for (35.0, dclab_rnd/retrieval/memory.CANDIDATES); unknown models get none
PREFIXES = {"embeddinggemma": ("task: search result | query: ", "title: none | text: "),
            "qwen3-embedding": ("Instruct: Given a question, retrieve the memory notes that answer it\nQuery:", ""),
            "bge-m3": ("", "")}
_DIGESTS: dict[str, str] = {}


def digest(base_url: str, model: str) -> str:
    """The weights' digest Ollama reports for the model (cached per process; "" when it does not answer)."""
    key = f"{base_url}|{model}"
    if key not in _DIGESTS:
        found = ""
        try:
            import urllib.request

            api = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
            with urllib.request.urlopen(api, timeout=3) as response:  # noqa: S310 - a local endpoint (embedding_tier refuses others)
                tags = json.loads(response.read().decode())
            found = next((m.get("digest") or "" for m in tags.get("models", []) if model in (m.get("name"), m.get("model"))), "")
        except Exception:  # noqa: BLE001
            found = ""
        _DIGESTS[key] = found
    return _DIGESTS[key]


def prefixes(model: str) -> tuple[str, str]:
    return next((p for name, p in PREFIXES.items() if model.startswith(name)), ("", ""))


class Embedder:
    """The vectors of the question and of the items, through one gateway client, the items' kept in the workspace."""

    def __init__(self, home: Path, client: Any, weights: str = ""):
        self.client, self.model = client, str(getattr(client, "model", "") or "unknown")
        self.query_prefix, self.doc_prefix = prefixes(self.model)
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", self.model) + (f"-{weights.split(':')[-1][:12]}" if weights else "")
        self.path = Path(home) / "memory" / f"vectors-{name}.json"

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.doc_prefix}\0{text}".encode()).hexdigest()[:24]

    def _load(self) -> dict[str, list[float]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def query(self, question: str) -> list[float]:
        return list(self.client.embed([self.query_prefix + question])["vectors"][0])

    def docs(self, texts: list[str]) -> list[list[float]]:
        from ..filelock import held as locked

        keys = [self._key(t) for t in texts]
        held = self._load()
        if all(k in held for k in keys):
            return [held[k] for k in keys]
        with locked(self.path.parent):  # between processes: the API and the worker write the same file
            held = self._load()
            todo = [t for t in dict.fromkeys(texts) if self._key(t) not in held]
            if todo:
                for t, vec in zip(todo, self.client.embed([self.doc_prefix + t for t in todo])["vectors"]):
                    held[self._key(t)] = list(vec)
                asked = set(keys)
                kept = {k: v for k, v in held.items() if k not in asked}
                kept = dict(list(kept.items())[-(MOST - len(asked)):]) if len(kept) + len(asked) > MOST else kept
                kept.update({k: held[k] for k in keys})  # the ones asked for last, at the end
                part = self.path.with_name(f"{self.path.name}.{os.getpid()}.{secrets.token_hex(4)}.part")
                part.write_text(json.dumps(kept), encoding="utf-8")
                os.replace(part, self.path)  # whole or not at all
                held = kept
        return [held[k] for k in keys]


def embedder_for(home: Path, project_id: str | None = None) -> Embedder | None:
    """The workspace's embedder when a local embedding model serves the purpose, else None (recall by words). A private
    project's notes get no model, as for every purpose (16.13)."""
    from ..models import settings

    local = settings.embedding_tier()
    if local is None:
        return None
    try:
        from ..models import installed

        gateway = installed()
    except Exception:  # noqa: BLE001
        gateway = None
    if gateway is None:
        from ..models.gateway import Gateway

        gateway = Gateway()
    client = gateway.client("memory_embedding", project_id=project_id)
    return Embedder(home, client, digest(local["base_url"], local["model"])) if client is not None else None


__all__ = ["MOST", "PREFIXES", "prefixes", "digest", "Embedder", "embedder_for"]

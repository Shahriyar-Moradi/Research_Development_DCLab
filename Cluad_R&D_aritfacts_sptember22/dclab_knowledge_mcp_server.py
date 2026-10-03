#!/usr/bin/env python3
"""
dclab_knowledge_mcp_server.py
================================
Exposes the same retrieval index as real MCP tools, so ANY MCP client —
Claude Desktop, Claude Code, or your own DCLab agent — can call it live.
This is the natural next step after `grounded_review.py`: instead of you
statically assembling a prompt, the calling agent decides when it needs
DCLab's rules or past-experiment evidence and asks for it mid-reasoning.

No training involved. This is pure plumbing on top of what already works.

Install (this repo's checkout, tested against `mcp` v2.x):
    pip install "mcp>=2"

Run directly for local testing:
    python dclab_knowledge_mcp_server.py

Point a client at it (example claude_desktop_config.json entry):
    {
      "mcpServers": {
        "dclab-knowledge": {
          "command": "python",
          "args": ["/absolute/path/to/dclab_knowledge_mcp_server.py"]
        }
      }
    }
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from build_rag_index import RetrievalIndex, load_all

# Point these at your real repo paths when you deploy this for real.
RULES_PATH = Path("knowledge/model_building_rules.jsonl")
CLAIMS_PATH = Path("campaigns/model_building_50_v1/agent_memory.jsonl")
BLOCKS_PATH = Path("knowledge/workflow_blocks.json")

_records = load_all(RULES_PATH, CLAIMS_PATH, BLOCKS_PATH)
_index = RetrievalIndex(_records)

server = MCPServer(name="dclab-knowledge")


@server.tool()
def get_rule(category: str, top_k: int = 3) -> str:
    """Return DCLab's evidence-based rule(s) for a model-building category,
    e.g. 'leakage', 'feature_engineering', 'model_selection', 'promotion',
    'missingness'. Grounded in knowledge/model_building_rules.jsonl."""
    hits = _index.search(category, filters={"kind": "rule", "category": category}, k=top_k)
    if not hits:
        hits = _index.search(category, filters={"kind": "rule"}, k=top_k)
    return "\n\n---\n\n".join(r.display for r, _ in hits) or f"No rule found for category '{category}'."


@server.tool()
def get_workflow_block(stage: str) -> str:
    """Return the reusable step-by-step flow for a DCLab pipeline stage
    (e.g. 'leakage audit', 'algorithm screen'). Grounded in workflow_blocks.json."""
    hits = _index.search(stage, filters={"kind": "workflow_block"}, k=1)
    return hits[0][0].display if hits else f"No workflow block matched '{stage}'."


@server.tool()
def retrieve_similar_experiment(dataset: str, stage: str, query: str, top_k: int = 3) -> str:
    """Retrieve the most relevant past DCLab experiment findings for a
    dataset + pipeline stage, ranked by relevance to `query`. Grounded in
    campaigns/model_building_50_v1/agent_memory.jsonl."""
    filters = {"kind": "experiment_claim"}
    if dataset:
        filters["dataset"] = dataset
    if stage:
        filters["stage"] = stage
    hits = _index.search(query, filters=filters, k=top_k)
    return "\n\n---\n\n".join(r.display for r, _ in hits) or "No matching past experiment found."


if __name__ == "__main__":
    server.run()

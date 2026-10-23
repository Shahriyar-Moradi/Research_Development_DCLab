# Local servers measured with `python -m dclab_rnd.models.probe_reasoning` (D2 P1)

## 2026-10-23: Ollama 0.30 (`/v1`), `qwen3.5:4b-q4_K_M` on the MacBook Air M1 (8 GB)

| Setting | Result |
|---|---|
| No reasoning field set | The reply carried a field named `reasoning` (`reasoning_fields_seen: ["reasoning"]`): **Ollama's `/v1` does return the reasoning** (the research notes of 2026-10-23 said it dropped it: wrong for this version) |
| `DCLAB_TIER_<T>_REASONING_FIELD=reasoning` | Reasoning read: 332 characters on the first reply (with its tool call), 5,567 on the second |
| Tool calling | The first reply called the tool; arguments arrived as a JSON string |
| Sent back on the next turn | Not shown: the model wrote no code word in its reasoning, so the probe could not test recall (run it again) |
| A warning | On the second turn the model thought 5,567 characters and answered nothing: a small model that spends its reply budget thinking can stall an agent; the H3 pilot measures it on real homework |

To use it: `DCLAB_TIER_<TIER>_REASONING_FIELD=reasoning` for the tier served by Ollama. `mlx_lm.server` is not usable
for Qwen3.5 tool calls (mlx-lm #1293: no `tool_calls` returned), so Ollama is the server for H3.

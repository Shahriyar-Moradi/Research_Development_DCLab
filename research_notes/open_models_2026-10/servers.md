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

## 2026-10-23 (later): the context window cut every prompt

The first H3 pilot ran `qwen3.5:4b-q4_K_M` as pulled. Ollama 0.40.9 loaded it with a **4,096-token window**
(`/api/ps` `context_length`) although the model supports 262,144, and every request reached the model with exactly
**2,050 input tokens**: Ollama cut each prompt silently. The intern's first message is about 5,000 tokens (its rules,
about 400; its 28 tools, about 4,600) and grows each step, so the model never saw its whole task: it rewrote its plan
again and again, asked to create a project that already existed, and once greeted the person as if no task had been
given. Those four attempts are marked `invalid.json` in `agent_runs/learning_attempts/2026-10-23/`.

Fix: a variant with a larger window, no download (`FROM qwen3.5:4b-q4_K_M` / `PARAMETER num_ctx 32768`,
`ollama create qwen3.5:4b-32k -f Modelfile`): 4.9 GB loaded, 3.2 GB of it on the GPU. The pilot now refuses a model
served with less than 16,384 tokens (`learning.solver.check_context`).

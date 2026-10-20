# Using your own model

dclabai talks to models through one gateway with two transports: any **OpenAI-compatible** chat-completions endpoint
(OpenAI, Ollama, vLLM, LM Studio, OpenRouter, Together, Groq, ...) and **Anthropic's** Messages API. `dclab init` writes the
settings below; `dclab doctor` checks them.

## No model

Leave everything unset. The Home agent asks its standard questions, the project chat's team follows its rules, the intern
follows the standard plan, and the engine runs exactly as it does with a model. Only model-written prose (explanations,
answers in your words) is missing.

## A local open model with Ollama

```bash
ollama pull qwen2.5:7b
dclab init        # finds Ollama at http://localhost:11434 and lists its models
```

Choose a model with **native tool calling** (for example `qwen2.5`, `llama3.1`, `mistral-nemo`): the agents act through tools.
A model that writes its tool calls as text is detected; the turn then falls back to the rules, and `dclab doctor` says so.
No key is needed for a model on localhost.

## Any OpenAI-compatible endpoint

```bash
DCLAB_LLM_BASE_URL=https://api.example.com/v1
OPENAI_API_KEY=...
OPENAI_MODEL=your-model-name
```

## Claude

```bash
pip install "dclabai[anthropic]"
DCLAB_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=your-claude-model
```

Prompt caching is used automatically with Claude.

## Tiers, budgets and prices

Purposes are grouped into tiers (`cheap`, `standard`, `strong`); each can point to its own endpoint with
`DCLAB_TIER_<TIER>_BASE_URL`, `_MODEL`, `_API_KEY`. Budgets in euros need prices: put them in a JSON file and set
`DCLAB_PRICES_FILE`. Without a price, spend is counted in tokens and a euro cap cannot stop a request before it is sent.

## Which agents use the model

An agent that acts through tools starts on its rules until your model passes `dclab doctor`'s tool-calling probe, or until
you trust it (`dclab models trust all`). The studio shows, for each agent, whether it runs on your model (probed or trusted)
or on its rules.

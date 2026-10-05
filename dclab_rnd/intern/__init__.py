"""The DCLab intern: describe a modeling task in plain language; it plans and runs the notebook for you.

Like Hugging Face's ML Intern, it is a chat mode with tools and a budget. Unlike it,
the "compute" is the DCLab notebook engine on this machine, and every tool it can call
is one the R&D already verified: the evidence index, the column auditor, the prediction
solution, the five deterministic stages, the notebook export. The model proposes and
explains; deterministic code owns splits, metrics and selection rules.

- ``tools``     the toolbox (evidence tools + project tools) with JSON schemas
- ``llm``       an OpenAI-compatible chat client (OpenAI, the Hugging Face router, Ollama, …)
- ``loop``      the budgeted tool loop, and the standard plan it follows when no model is configured
- ``sessions``  one JSON file per conversation
"""

from .loop import Intern  # noqa: F401
from .sessions import SessionStore  # noqa: F401

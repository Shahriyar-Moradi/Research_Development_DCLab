"""The intern's model client. It now lives in ``dclab_rnd.models.client`` (shared with the model gateway); this name
is kept so the intern and anything that imported ``dclab_rnd.intern.llm`` keep working unchanged."""

from ..models.client import DEFAULT_MODEL, PAUSE_SECONDS, _PAUSED, ChatClient, failure, resume, settings  # noqa: F401

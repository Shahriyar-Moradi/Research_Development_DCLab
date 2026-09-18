"""NOOA specialists, LangGraph orchestration, and a bounded research executor."""
import os

# Keep telemetry and model-metadata discovery local. OpenAI inference is the
# only configured outbound data destination for the agent service.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

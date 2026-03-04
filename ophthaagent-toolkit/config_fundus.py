"""Anonymous and minimal runtime config for fundus decision pipelines."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class AgentDecisionConfig:
    """LLM config used by fundus decision scripts."""

    def __init__(self):
        api_key = os.getenv("FUNDUS_LLM_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or "<ANON_API_KEY>"
        self.llm = ChatOpenAI(
            model=os.getenv("FUNDUS_LLM_MODEL", "<ANON_LLM_MODEL>"),
            base_url=os.getenv("FUNDUS_LLM_BASE_URL", "https://api.example.com/v1"),
            api_key=api_key,
            temperature=float(os.getenv("FUNDUS_LLM_TEMPERATURE", "0.1")),
        )


class LangSmithConfig:
    """Optional tracing config kept as placeholders."""

    def __init__(self):
        self.tracing = os.getenv("LANGSMITH_TRACING", "<ANON_TRACING_FLAG>")
        self.endpoint = os.getenv("LANGSMITH_ENDPOINT", "<ANON_TRACING_ENDPOINT>")
        self.api_key = os.getenv("LANGSMITH_API_KEY", "<ANON_TRACING_API_KEY>")
        self.project = os.getenv("LANGSMITH_PROJECT", "<ANON_PROJECT>")
        # Backward-compatible aliases for legacy call sites.
        self.LANGSMITH_TRACING = self.tracing
        self.LANGSMITH_ENDPOINT = self.endpoint
        self.LANGSMITH_API_KEY = self.api_key
        self.LANGSMITH_PROJECT = self.project


class Config:
    """Minimal config surface required by existing decision scripts."""

    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.lang_smith = LangSmithConfig()

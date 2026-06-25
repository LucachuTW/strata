"""LangChain ChatOpenAI factories pointed at the local OpenAI-compatible endpoint.

Switch Ollama (dev) <-> vLLM (prod) by changing LLM_BASE_URL only — no code change.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import get_settings


def _chat(model: str, **kwargs) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,  # type: ignore[arg-type]  # LangChain coerces str -> SecretStr
        temperature=kwargs.pop("temperature", s.llm_temperature),
        timeout=s.llm_request_timeout,
        **kwargs,
    )


def generation_llm(**kwargs) -> ChatOpenAI:
    """The 7-8B model used for generation and the critic."""
    return _chat(get_settings().llm_model, **kwargs)


def rewrite_llm(**kwargs) -> ChatOpenAI:
    """The small, fast model used for query rewrite / HyDE."""
    return _chat(get_settings().llm_rewrite_model, **kwargs)

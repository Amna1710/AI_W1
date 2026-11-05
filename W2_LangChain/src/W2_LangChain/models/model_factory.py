
from langchain_core.language_models import BaseChatModel

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from W2_LangChain.core.config import settings


def get_chat_model(
    provider: str | None = None,
) -> BaseChatModel:
    """
    Create and return a chat model for the selected provider.

    Supported providers:
        - openai
        - anthropic
        - ollama
    """

    provider = (
        provider or settings.llm_provider
    ).lower().strip()

    if provider == "openai":

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured in .env"
            )

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
        )

    elif provider == "anthropic":

        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured in .env"
            )

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
        )

    elif provider == "ollama":

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "Choose openai, anthropic, or ollama."
        )

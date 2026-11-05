
import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv(
        "OPENAI_MODEL", "gpt-4.1-mini"
    )

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv(
        "ANTHROPIC_MODEL", "claude-sonnet-4-20250514"
    )

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    ollama_model: str = os.getenv(
        "OLLAMA_MODEL", "qwen3.5:4b"
    )

    llm_temperature: float = float(
        os.getenv("LLM_TEMPERATURE", "0.2")
    )


settings = Settings()

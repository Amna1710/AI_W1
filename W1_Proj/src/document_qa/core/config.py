import os

from dotenv import load_dotenv

load_dotenv()


class Settings:

    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    )

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "embeddinggemma",
    )

    LLM_MODEL = os.getenv(
        "LLM_MODEL",
        "qwen3.5:4b",
    )

    RETRIEVAL_TOP_K = int(
        os.getenv("RETRIEVAL_TOP_K", "3")
    )

    RETRIEVAL_MIN_SIMILARITY = float(
        os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.50")
    )

    CONVERSATION_MAX_TURNS = int(
        os.getenv("CONVERSATION_MAX_TURNS", "5")
    )


settings = Settings()
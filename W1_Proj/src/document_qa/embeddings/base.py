
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base interface for embedding providers."""

    @abstractmethod
    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
        raise NotImplementedError

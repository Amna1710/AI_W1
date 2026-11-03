
from abc import ABC, abstractmethod

from document_qa.document.chunker import DocumentChunk


class VectorStore(ABC):
    """Base interface for vector storage implementations."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

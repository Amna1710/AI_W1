
from dataclasses import dataclass

import numpy as np

from document_qa.document.chunker import DocumentChunk
from document_qa.vector_store.base import VectorStore


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float


class InMemoryVectorStore(VectorStore):
    """Stores document chunks and searches them using cosine similarity."""

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        new_chunks = []
        new_embeddings = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk {chunk.index} does not have an embedding."
                )

            vector = np.asarray(chunk.embedding, dtype=np.float32)

            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(
                    f"Chunk {chunk.index} has an invalid embedding."
                )

            if not np.all(np.isfinite(vector)):
                raise ValueError(
                    f"Chunk {chunk.index} contains invalid vector values."
                )

            new_chunks.append(chunk)
            new_embeddings.append(vector)

        dimensions = {len(vector) for vector in new_embeddings}

        if len(dimensions) != 1:
            raise ValueError("All embeddings must have the same dimensions.")

        new_matrix = np.vstack(new_embeddings)

        if self.embeddings is not None:
            if self.embeddings.shape[1] != new_matrix.shape[1]:
                raise ValueError(
                    "New embeddings have different dimensions from "
                    "the existing embeddings. Clear the store first."
                )

            self.embeddings = np.vstack(
                [self.embeddings, new_matrix]
            )
        else:
            self.embeddings = new_matrix

        self.chunks.extend(new_chunks)

    def search(
            self,
            query_embedding: list[float],
            top_k: int = 3,
            min_similarity: float = 0.50,
    ) -> list[SearchResult]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1.")

        if not self.chunks:
            return []

        query_vector = np.array(query_embedding, dtype=np.float32)

        if query_vector.ndim != 1:
            raise ValueError("Query embedding must be a one-dimensional vector.")

        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimensions do not match stored embeddings."
            )

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        stored_norms = np.linalg.norm(self.embeddings, axis=1)

        if np.any(stored_norms == 0):
            raise ValueError("Stored embeddings cannot contain zero vectors.")

        similarities = np.dot(
            self.embeddings,
            query_vector,
        ) / (stored_norms * query_norm)

        ranked_indices = np.argsort(similarities)[::-1]

        results = []

        for index in ranked_indices:
            score = float(similarities[index])

            if score < min_similarity:
                continue

            results.append(
                SearchResult(
                    chunk=self.chunks[index],
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        self.chunks.clear()
        self.embeddings = None

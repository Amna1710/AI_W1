
from document_qa.document.chunker import DocumentChunk
from document_qa.embeddings.base import EmbeddingProvider


class EmbeddingService:
    """Manages embedding generation for documents and questions."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 32,
    ) -> None:

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        self.provider = provider
        self.batch_size = batch_size

    async def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Generate and attach embeddings to document chunks."""

        if not chunks:
            return []

        for start in range(0, len(chunks), self.batch_size):

            batch = chunks[start:start + self.batch_size]

            texts = [
                chunk.content
                for chunk in batch
            ]

            embeddings = await self.provider.embed_texts(texts)

            if len(embeddings) != len(batch):
                raise ValueError(
                    "Embedding count does not match chunk count."
                )

            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

        return chunks

    async def embed_query(
        self,
        question: str,
    ) -> list[float]:
        """Generate an embedding for a user question."""

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        embeddings = await self.provider.embed_texts([question])

        if len(embeddings) != 1:
            raise ValueError(
                "Could not generate the question embedding."
            )

        return embeddings[0]

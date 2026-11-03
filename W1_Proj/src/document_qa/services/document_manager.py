from pathlib import Path
from dataclasses import dataclass

from document_qa.document.loader import DocumentLoader, LoadedDocument
from document_qa.document.chunker import TextChunker, DocumentChunk
from document_qa.embeddings.embedding_service import EmbeddingService
from document_qa.vector_store.in_memory import InMemoryVectorStore


@dataclass
class ManagedDocument:
    document: LoadedDocument
    chunks: list[DocumentChunk]


class DocumentManager:
    """Manages documents and their corresponding vector embeddings."""

    def __init__(
        self,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

        self.documents: dict[str, ManagedDocument] = {}

    async def add_document(self, file_path: str) -> ManagedDocument:
        """Load, chunk, embed, and index a document."""

        document = DocumentLoader.load_document(file_path)

        document_key = document.file_path.lower()

        if document_key in self.documents:
            raise ValueError(
                f"Document already loaded: {document.file_name}"
            )

        chunks = self.chunker.chunk_document(document)

        if not chunks:
            raise ValueError(
                f"No chunks could be generated from {document.file_name}."
            )

        chunks = await self.embedding_service.embed_chunks(chunks)

        self.vector_store.add_chunks(chunks)

        managed_document = ManagedDocument(
            document=document,
            chunks=chunks,
        )

        self.documents[document_key] = managed_document

        return managed_document

    def list_documents(self) -> list[ManagedDocument]:
        """Return all currently loaded documents."""

        return list(self.documents.values())

    def remove_document(self, file_path: str) -> bool:
        """Remove a document and rebuild the in-memory vector index."""

        document_key = str(
            Path(file_path).expanduser().resolve()
        ).lower()

        managed_document = self.documents.get(document_key)

        if managed_document is None:
            return False

        del self.documents[document_key]

        remaining_chunks = []

        for managed in self.documents.values():
            remaining_chunks.extend(managed.chunks)

        self.vector_store.clear()

        if remaining_chunks:
            self.vector_store.add_chunks(remaining_chunks)

        return True

    def clear_documents(self) -> None:
        """Remove all documents and their embeddings."""

        self.documents.clear()
        self.vector_store.clear()

    def get_document_count(self) -> int:
        return len(self.documents)

    def get_chunk_count(self) -> int:
        return sum(
            len(managed.chunks)
            for managed in self.documents.values()
        )

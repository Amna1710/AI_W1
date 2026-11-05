from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from W2_LangChain.document_loaders import (
    PDFDocumentLoader,
    HTMLDocumentLoader,
    MarkdownDocumentLoader,
    DatabaseDocumentLoader,
)
from W2_LangChain.metadata import enrich_documents_metadata


def get_text_splitter(
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """Create and return the requested text splitter."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter_type = splitter_type.lower().strip()

    if splitter_type == "character":
        return CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if splitter_type == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise ValueError(
        "Unsupported splitter. Choose character, recursive, or token."
    )


def process_documents(
    loader: Any,
    source: str,
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    loader_kwargs: dict | None = None,
) -> list[Document]:
    """Load, enrich metadata, and split documents into chunks."""

    # 1. Load documents.
    documents = loader.load(
        source,
        **(loader_kwargs or {}),
    )

    if not documents:
        return []

    # 2. Enrich metadata before splitting.
    documents = enrich_documents_metadata(
        documents,
        source=str(source),
    )

    # 3. Create the selected text splitter.
    splitter = get_text_splitter(
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 4. Split documents. Metadata is copied to each chunk.
    chunks = splitter.split_documents(documents)

    return chunks

def process_file(
    file_path: str | Path,
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Automatically select a loader based on the file extension
    and process the file into document chunks.
    """

    file_path = Path(file_path).expanduser()

    extension = file_path.suffix.lower()

    loader_map = {
        ".pdf": PDFDocumentLoader,
        ".html": HTMLDocumentLoader,
        ".htm": HTMLDocumentLoader,
        ".md": MarkdownDocumentLoader,
        ".markdown": MarkdownDocumentLoader,
    }

    loader_class = loader_map.get(extension)

    if loader_class is None:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    loader = loader_class()

    return process_documents(
        loader=loader,
        source=file_path,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def process_database(
    connection_url: str,
    query: str,
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    params: dict | None = None,
    source_name: str = "database",
) -> list[Document]:
    """Load database records and split them into document chunks."""

    loader = DatabaseDocumentLoader(connection_url)

    try:
        return process_documents(
            loader=loader,
            source=query,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            loader_kwargs={
                "params": params,
                "source_name": source_name,
            },
        )
    finally:
        loader.close()
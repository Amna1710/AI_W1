from time import perf_counter
from statistics import mean

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def compare_splitters(documents: list[Document]) -> dict:
    """
    Compare different text splitters using the same input documents.

    Character-based splitters use character counts.
    TokenTextSplitter uses token counts.
    """

    splitters = {
        "CharacterTextSplitter": CharacterTextSplitter(
            separator=" ",
            chunk_size=500,
            chunk_overlap=50,
        ),
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        ),
        "TokenTextSplitter": TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
        ),
    }

    results = {}

    for name, splitter in splitters.items():
        start_time = perf_counter()

        chunks = splitter.split_documents(documents)

        execution_time = perf_counter() - start_time

        chunk_lengths = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        results[name] = {
            "chunk_count": len(chunks),
            "average_chunk_length": (
                round(mean(chunk_lengths), 2)
                if chunk_lengths else 0
            ),
            "maximum_chunk_length": (
                max(chunk_lengths)
                if chunk_lengths else 0
            ),
            "execution_time": round(execution_time, 6),
            "metadata_preserved": all(
                chunk.metadata == documents[0].metadata
                for chunk in chunks
            ) if chunks else True,
        }

    return results
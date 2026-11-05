from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from W2_LangChain.metadata import (
    extract_headers,
    extract_date,
    enrich_document_metadata,
    enrich_documents_metadata,
)
from W2_LangChain.pipelines import process_file


def test_extract_headers():
    document = Document(
        page_content="# Python\n\n## LangChain\n\nSome content.",
        metadata={},
    )

    headers = extract_headers(document)

    assert headers == ["Python", "LangChain"]
    print("Header extraction passed!")


def test_extract_date():
    document = Document(
        page_content="Published: 2026-10-15\n\nArticle content.",
        metadata={},
    )

    date_value = extract_date(document)

    assert date_value == "2026-10-15"
    print("Date extraction passed!")


def test_metadata_enrichment():
    document = Document(
        page_content="# Python\n\nDate: 2026-10-15\n\nPython content.",
        metadata={
            "source": "python.md",
            "file_type": "markdown",
        },
    )

    enriched = enrich_document_metadata(document)

    assert enriched.metadata["source"] == "python.md"
    assert enriched.metadata["file_name"] == "python.md"
    assert enriched.metadata["file_type"] == "markdown"
    assert enriched.metadata["headers"] == ["Python"]
    assert enriched.metadata["date"] == "2026-10-15"
    assert "processed_at" in enriched.metadata

    print("Single document enrichment passed!")


def test_multiple_documents():
    documents = [
        Document(page_content="First document.", metadata={}),
        Document(page_content="Second document.", metadata={}),
    ]

    enriched = enrich_documents_metadata(
        documents,
        source="sample.txt",
    )

    assert len(enriched) == 2

    for document in enriched:
        assert document.metadata["source"] == "sample.txt"
        assert document.metadata["file_name"] == "sample.txt"
        assert document.metadata["file_type"] == "txt"
        assert "processed_at" in document.metadata

    print("Multiple document enrichment passed!")


def test_pipeline_metadata():
    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "sample.md"

        file_path.write_text(
            """
# Python

Date: 2026-10-15

Python is a programming language.

## LangChain

LangChain helps developers build AI applications.
""" * 10,
            encoding="utf-8",
        )

        chunks = process_file(
            file_path=file_path,
            chunk_size=200,
            chunk_overlap=30,
        )

        assert len(chunks) > 0

        for chunk in chunks:
            metadata = chunk.metadata

            assert metadata["source"] == str(file_path.resolve())
            assert metadata["file_name"] == "sample.md"
            assert metadata["file_type"] == "markdown"
            assert "headers" in metadata
            assert metadata["date"] == "2026-10-15"
            assert "processed_at" in metadata

        print("Pipeline metadata integration passed!")


if __name__ == "__main__":
    test_extract_headers()
    test_extract_date()
    test_metadata_enrichment()
    test_multiple_documents()
    test_pipeline_metadata()

    print("\nAll metadata tests passed!")
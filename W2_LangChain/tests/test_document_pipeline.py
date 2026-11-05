from pathlib import Path
from tempfile import TemporaryDirectory

from W2_LangChain.pipelines import process_file


def test_document_pipeline():
    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "sample.md"

        file_path.write_text(
            """
# Python

Python is a popular programming language.

It is used for web development, automation,
data science, and artificial intelligence.

## LangChain

LangChain helps developers build applications
using large language models.

It provides components for prompts, models,
chains, agents, and document processing.

## Document Processing

Document processing involves loading documents,
splitting text, and preserving metadata.
""" * 10,
            encoding="utf-8",
        )

        chunks = process_file(
            file_path=file_path,
            splitter_type="recursive",
            chunk_size=200,
            chunk_overlap=30,
        )

        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.page_content.strip()
            assert chunk.metadata["source"] == str(file_path.resolve())
            assert chunk.metadata["file_type"] == "markdown"

        print(f"Total chunks: {len(chunks)}")
        print("\nFirst chunk:")
        print(chunks[0].page_content)
        print("\nMetadata:")
        print(chunks[0].metadata)


if __name__ == "__main__":
    test_document_pipeline()
    print("\nUnified document pipeline test passed!")
from langchain_core.documents import Document

from W2_LangChain.text_splitters import compare_splitters


def test_splitter_comparison():
    sample_text = """
    LangChain is a framework for developing applications powered by
    language models. It provides tools for connecting models with data
    sources, creating chains, and building AI applications.

    Text splitting is an important part of document processing.
    Documents are divided into smaller chunks before being passed
    to language models. This helps manage context length and improves
    retrieval performance.

    Recursive splitting attempts to preserve meaningful text boundaries.
    Character splitting uses a selected separator. Token splitting
    divides text according to the tokenizer's token count.
    """

    # Repeat the text to create a larger document.
    sample_text = sample_text * 20

    document = Document(
        page_content=sample_text,
        metadata={
            "source": "test_document.txt",
            "category": "learning",
        },
    )

    results = compare_splitters([document])

    for splitter_name, metrics in results.items():
        print(f"\n{splitter_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        assert metrics["chunk_count"] > 0
        assert metrics["metadata_preserved"] is True


if __name__ == "__main__":
    test_splitter_comparison()
    print("\nAll text splitter tests passed!")
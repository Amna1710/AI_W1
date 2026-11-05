
import tempfile
from pathlib import Path

from sqlalchemy import create_engine, text

from W2_LangChain.document_loaders import (
    HTMLDocumentLoader,
    MarkdownDocumentLoader,
    DatabaseDocumentLoader,
    PDFDocumentLoader,
)


def test_html_loader(temp_dir):

    html_path = temp_dir / "sample.html"

    html_path.write_text(
        """
        <html>
            <head>
                <title>Python Tutorial</title>
            </head>
            <body>
                <nav>Navigation</nav>
                <h1>Introduction to Python</h1>
                <p>Python is a programming language.</p>
                <script>console.log("test");</script>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    loader = HTMLDocumentLoader()
    documents = loader.load(html_path)

    assert len(documents) == 1

    document = documents[0]

    print("\n--- HTML Loader ---")
    print(document.page_content)
    print(document.metadata)

    assert document.metadata["title"] == "Python Tutorial"
    assert "Python is a programming language" in document.page_content
    assert "console.log" not in document.page_content


def test_markdown_loader(temp_dir):

    markdown_path = temp_dir / "sample.md"

    markdown_path.write_text(
        """
# Python

Python is a programming language.

## Features

Python has simple syntax.

## Applications

Python is used in AI and automation.
""",
        encoding="utf-8",
    )

    loader = MarkdownDocumentLoader()
    documents = loader.load(markdown_path)

    print("\n--- Markdown Loader ---")

    for document in documents:
        print(document.page_content)
        print(document.metadata)

    assert len(documents) == 3
    assert documents[1].metadata["headers"] == [
        "Python",
        "Features",
    ]


def test_database_loader(temp_dir):

    db_path = temp_dir / "sample.db"
    connection_url = f"sqlite:///{db_path}"

    engine = create_engine(connection_url)

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    department TEXT
                )
                """
            )
        )

        connection.execute(
            text(
                """
                INSERT INTO employees (name, department)
                VALUES (:name, :department)
                """
            ),
            [
                {"name": "Ahmed", "department": "IT"},
                {"name": "Sara", "department": "HR"},
            ],
        )

    engine.dispose()

    loader = DatabaseDocumentLoader(connection_url)

    try:
        documents = loader.load(
            query="SELECT * FROM employees",
            source_name="employees",
        )

        print("\n--- Database Loader ---")

        for document in documents:
            print(document.page_content)
            print(document.metadata)

        assert len(documents) == 2
        assert "Ahmed" in documents[0].page_content
        assert documents[0].metadata["row_index"] == 1

    finally:
        loader.close()


def test_pdf_loader():

    pdf_path = Path(r"D:\PyCharmProjects\W2_LangChain\src\W2_LangChain\documents\sample.pdf")

    if not pdf_path.exists():
        print(
            "\nPDF test skipped: "
            "Place a PDF at documents/sample.pdf to test it."
        )
        return

    loader = PDFDocumentLoader()
    documents = loader.load(pdf_path)

    print("\n--- PDF Loader ---")
    print("Extracted pages:", len(documents))

    for document in documents:
        print("Page:", document.metadata["page"])
        print(document.page_content[:200])

    assert len(documents) > 0


if __name__ == "__main__":

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)

        test_html_loader(temp_dir)
        test_markdown_loader(temp_dir)
        test_database_loader(temp_dir)

    test_pdf_loader()

    print("\nDocument loader tests completed!")

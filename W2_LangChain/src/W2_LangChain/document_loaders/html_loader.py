
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from W2_LangChain.document_loaders.base import BaseDocumentLoader


class HTMLDocumentLoader(BaseDocumentLoader):
    """Custom loader for local HTML files."""

    def load(self, source: str | Path) -> list[Document]:

        file_path = Path(source).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(
                f"HTML file not found: {file_path}"
            )

        if file_path.suffix.lower() not in {".html", ".htm"}:
            raise ValueError("Source must be an HTML file.")

        html_content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove elements that usually contain no article text
        for element in soup(
            ["script", "style", "noscript", "nav", "footer"]
        ):
            element.decompose()

        title = (
            soup.title.get_text(strip=True)
            if soup.title
            else file_path.stem
        )

        headings = [
            heading.get_text(" ", strip=True)
            for heading in soup.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6"]
            )
        ]

        text = soup.get_text(separator="\n", strip=True)

        if not text:
            return []

        document = Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "html",
                "title": title,
                "headings": headings,
            },
        )

        return [document]


from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document

from W2_LangChain.document_loaders.base import BaseDocumentLoader


class PDFDocumentLoader(BaseDocumentLoader):
    """Custom loader for PDF documents."""

    def load(self, source: str | Path) -> list[Document]:

        file_path = Path(source).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError("Source must be a PDF file.")

        reader = PdfReader(str(file_path))

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""
            text = text.strip()

            if not text:
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "pdf",
                    "page": page_number,
                },
            )

            documents.append(document)

        return documents

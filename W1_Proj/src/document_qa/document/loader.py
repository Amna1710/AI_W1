
from dataclasses import dataclass
from pathlib import Path

import fitz
from docx import Document as DocxDocument


@dataclass
class LoadedDocument:
    """Stores extracted document content and metadata."""

    content: str
    file_name: str
    file_path: str
    file_type: str
    page_count: int | None = None


class DocumentLoader:
    """Loads text from PDF, TXT, and DOCX documents."""

    SUPPORTED_FORMATS = {".pdf", ".txt", ".docx"}

    @classmethod
    def load_document(cls, file_path: str) -> LoadedDocument:
        """Validate and load a document."""

        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"The provided path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(cls.SUPPORTED_FORMATS)}"
            )

        if extension == ".pdf":
            content, page_count = cls._load_pdf(path)

        elif extension == ".txt":
            content = cls._load_txt(path)
            page_count = None

        else:
            content = cls._load_docx(path)
            page_count = None

        if not content.strip():
            raise ValueError(
                "The document contains no extractable text."
            )

        return LoadedDocument(
            content=content,
            file_name=path.name,
            file_path=str(path.resolve()),
            file_type=extension.replace(".", "").upper(),
            page_count=page_count,
        )

    @staticmethod
    def _load_pdf(path: Path) -> tuple[str, int]:
        """Extract text from a PDF document."""

        pages = []

        with fitz.open(path) as pdf:
            page_count = len(pdf)

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")

                if page_text.strip():
                    pages.append(
                        f"[Page {page_number}]\n{page_text}"
                    )

        return "\n\n".join(pages), page_count

    @staticmethod
    def _load_txt(path: Path) -> str:
        """Read a TXT document."""

        try:
            return path.read_text(encoding="utf-8-sig")

        except UnicodeDecodeError:
            return path.read_text(encoding="cp1252")

    @staticmethod
    def _load_docx(path: Path) -> str:
        """Extract paragraphs and tables from a DOCX document."""

        document = DocxDocument(path)
        content = []

        # Extract paragraphs
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                content.append(text)

        # Extract table contents
        for table in document.tables:
            for row in table.rows:
                row_text = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if row_text:
                    content.append(" | ".join(row_text))

        return "\n\n".join(content)

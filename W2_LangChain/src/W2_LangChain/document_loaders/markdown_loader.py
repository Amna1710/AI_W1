
import re
from pathlib import Path

from langchain_core.documents import Document

from W2_LangChain.document_loaders.base import BaseDocumentLoader


class MarkdownDocumentLoader(BaseDocumentLoader):
    """Custom loader for Markdown documents."""

    HEADING_PATTERN = re.compile(
        r"^(#{1,6})\s+(.+?)\s*#*\s*$"
    )

    def load(self, source: str | Path) -> list[Document]:

        file_path = Path(source).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(
                f"Markdown file not found: {file_path}"
            )

        if file_path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("Source must be a Markdown file.")

        content = file_path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        documents = []
        header_stack = []
        current_lines = []

        def save_section():
            section_content = "\n".join(current_lines).strip()

            if not section_content:
                return

            documents.append(
                Document(
                    page_content=section_content,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": "markdown",
                        "headers": [
                            header["title"]
                            for header in header_stack
                        ],
                    },
                )
            )

        for line in content.splitlines():

            match = self.HEADING_PATTERN.match(line)

            if match:

                # Save the previous section
                save_section()
                current_lines.clear()

                level = len(match.group(1))
                heading = match.group(2).strip()

                # Remove headings at the same or deeper level
                header_stack = [
                    item
                    for item in header_stack
                    if item["level"] < level
                ]

                header_stack.append(
                    {
                        "level": level,
                        "title": heading,
                    }
                )

                current_lines.append(line)

            else:
                current_lines.append(line)

        # Save the final section
        save_section()

        return documents

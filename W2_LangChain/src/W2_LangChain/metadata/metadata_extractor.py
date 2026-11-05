import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document


# Matches Markdown headings such as:
# # Python
# ## LangChain
MARKDOWN_HEADING_PATTERN = re.compile(
    r"^#{1,6}\s+(.+?)\s*#*\s*$",
    re.MULTILINE,
)

# Matches explicitly labelled dates such as:
# Date: 2026-10-15
# Published: 2026-10-15
# Created: 2026-10-15
DATE_PATTERN = re.compile(
    r"\b(?:date|published|created)\s*:\s*"
    r"(\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def extract_headers(document: Document) -> list[str]:
    """Extract headings from existing metadata or Markdown content."""

    metadata_headers = document.metadata.get("headers")

    if metadata_headers:
        return metadata_headers

    # HTML loader stores its headings under "headings".
    html_headings = document.metadata.get("headings")

    if html_headings:
        return html_headings

    return MARKDOWN_HEADING_PATTERN.findall(document.page_content)


def extract_date(document: Document) -> str | None:
    """
    Extract a date from existing metadata or explicitly labelled
    ISO date text. Return None if no date is available.
    """

    for key in ("date", "document_date", "published_date"):
        value = document.metadata.get(key)

        if value:
            return str(value)

    match = DATE_PATTERN.search(document.page_content)

    if match:
        date_value = match.group(1)

        # Validate that the extracted date is a real calendar date.
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
            return date_value
        except ValueError:
            return None

    return None


def enrich_document_metadata(
    document: Document,
    source: str | None = None,
) -> Document:
    """
    Add standard metadata fields while preserving existing metadata.
    """

    metadata = dict(document.metadata)

    # Source
    if not metadata.get("source") and source:
        metadata["source"] = str(source)

    # File name
    if not metadata.get("file_name") and metadata.get("source"):
        metadata["file_name"] = Path(
            str(metadata["source"])
        ).name

    # File type
    if not metadata.get("file_type") and metadata.get("file_name"):
        metadata["file_type"] = Path(
            str(metadata["file_name"])
        ).suffix.lower().lstrip(".")

    # Headers
    headers = extract_headers(document)

    if headers:
        metadata["headers"] = headers

    # Document date
    date_value = extract_date(document)

    if date_value:
        metadata["date"] = date_value

    # Processing timestamp
    metadata["processed_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    return Document(
        page_content=document.page_content,
        metadata=metadata,
    )


def enrich_documents_metadata(
    documents: list[Document],
    source: str | None = None,
) -> list[Document]:
    """Enrich metadata for multiple sections of a document."""

    # Extract a common date from any section.
    common_date = next(
        (
            date_value
            for document in documents
            if (date_value := extract_date(document))
        ),
        None,
    )

    enriched_documents = []

    for document in documents:
        enriched = enrich_document_metadata(
            document,
            source=source,
        )

        # Share the document-level date across all sections.
        if common_date and not enriched.metadata.get("date"):
            enriched.metadata["date"] = common_date

        enriched_documents.append(enriched)

    return enriched_documents
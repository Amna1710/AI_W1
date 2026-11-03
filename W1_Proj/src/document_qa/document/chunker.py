
from dataclasses import dataclass

from document_qa.document.loader import LoadedDocument


@dataclass
class DocumentChunk:
    """Stores a document chunk and its source information."""

    index: int
    content: str
    source_file: str
    start_char: int
    end_char: int
    embedding: list[float] | None = None


class TextChunker:
    """Splits document text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if overlap < 0:
            raise ValueError("Overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError(
                "Overlap must be smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:

        text = document.content
        chunks = []

        start = 0
        chunk_index = 1
        text_length = len(text)

        while start < text_length:

            end = min(
                start + self.chunk_size,
                text_length,
            )

            if end < text_length:
                end = self._find_boundary(text, start, end)

            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append(
                    DocumentChunk(
                        index=chunk_index,
                        content=chunk_content,
                        source_file=document.file_name,
                        start_char=start,
                        end_char=end,
                    )
                )

                chunk_index += 1

            if end >= text_length:
                break

            next_start = end - self.overlap

            while (
                next_start < end
                and next_start < text_length
                and text[next_start].isspace()
            ):
                next_start += 1

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _find_boundary(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:

        minimum_boundary = start + self.chunk_size // 2

        boundaries = ["\n\n", "\n", ". ", " "]

        for separator in boundaries:
            position = text.rfind(
                separator,
                minimum_boundary,
                end,
            )

            if position != -1:
                return position + len(separator)

        return end

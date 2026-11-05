
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """Base class for custom document loaders."""

    @abstractmethod
    def load(self, source: str | Path) -> list[Document]:
        """
        Load a source and return LangChain Document objects.
        """
        raise NotImplementedError

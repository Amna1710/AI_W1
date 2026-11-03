
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base interface for language model providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

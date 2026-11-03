
import httpx

from document_qa.llm.base import LLMProvider


class OllamaLLMProvider(LLMProvider):
    """Generates responses using a locally running Ollama model."""

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://127.0.0.1:11434",
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,
                write=30.0,
                pool=10.0,
            ),
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.temperature = temperature

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:

        if not messages:
            raise ValueError("Messages cannot be empty.")

        for message in messages:
            if (
                message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError("Invalid chat message.")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        response = await self.client.post(
            "/api/chat",
            json=payload,
        )

        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Ollama returned an empty or invalid response."
            )

        return content.strip()

    async def close(self) -> None:
        await self.client.aclose()

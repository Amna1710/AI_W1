
from typing import Any

from ai_proj.core.http_client import AsyncHTTPClient


class OllamaChatCompletion:
    """A reusable asynchronous wrapper for local Ollama chat."""

    API_URL = "http://127.0.0.1:11434/v1/chat/completions"

    def __init__(
        self,
        client: AsyncHTTPClient,
        model: str = "qwen3.5:4b",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:

        selected_model = model if model is not None else self.model

        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": (
                self.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self.max_tokens if max_tokens is None else max_tokens
            ),
            "stream": False,
        }

        response = await self.client.post(
            self.API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            retryable=False,
        )

        data = response.json()

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Ollama returned an invalid chat completion response."
            ) from exc

        if not isinstance(content, str):
            raise ValueError("Ollama did not return a text response.")

        provider_usage = data.get("usage") or {}

        return {
            "content": content,
            "model": data.get("model", selected_model),
            "finish_reason": choice.get("finish_reason"),
            "usage": {
                "input_tokens": provider_usage.get("prompt_tokens"),
                "output_tokens": provider_usage.get("completion_tokens"),
                "total_tokens": provider_usage.get("total_tokens"),
            },
        }


import os
from typing import Any

import httpx

from ai_proj.core.cost_tracker import calculate_cost
from ai_proj.core.http_client import AsyncHTTPClient


class OpenAIChatCompletion:
    """
    A reusable asynchronous wrapper for OpenAI Chat Completions.
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        client: AsyncHTTPClient,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> None:

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not configured."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Send messages to OpenAI and return the generated response
        together with token usage and estimated cost.
        """

        selected_model = model or self.model

        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": (
                self.temperature if temperature is None else temperature
            ),
            "max_completion_tokens": (
                self.max_tokens if max_tokens is None else max_tokens
            ),
        }

        response = await self.client.post(
            self.API_URL,
            json=payload,
            headers=self.headers,
            retryable=False,
        )

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(
                "OpenAI returned invalid JSON."
            ) from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
            provider_usage = data["usage"]

        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "OpenAI returned an invalid Chat Completion response."
            ) from exc

        if not isinstance(content, str):
            raise ValueError(
                "OpenAI did not return a text response."
            )

        input_tokens = int(provider_usage["prompt_tokens"])
        output_tokens = int(provider_usage["completion_tokens"])
        total_tokens = int(provider_usage["total_tokens"])

        cost = calculate_cost(
            provider="openai",
            model=selected_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            **cost,
        }

        return {
            "content": content,
            "model": selected_model,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
        }

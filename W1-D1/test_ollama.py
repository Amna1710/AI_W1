
import asyncio

from ai_proj.core.http_client import AsyncHTTPClient
from ai_proj.services.ollama_chat import OllamaChatCompletion


async def main():

    async with AsyncHTTPClient(timeout=120) as http_client:

        ollama = OllamaChatCompletion(
            client=http_client,
            model="qwen3.5:4b",
            temperature=0.7,
            max_tokens=200,
        )

        result = await ollama.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful Python programming assistant.",
                },
                {
                    "role": "user",
                    "content": "Explain Python decorators with a simple example.",
                },
            ]
        )

        print("\nMODEL:", result["model"])
        print("\nRESPONSE:\n", result["content"])
        print("\nUSAGE:", result["usage"])


if __name__ == "__main__":
    asyncio.run(main())

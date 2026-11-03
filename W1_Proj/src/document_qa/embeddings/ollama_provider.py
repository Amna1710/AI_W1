
import httpx

from document_qa.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:

        self.model = model

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts."""

        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts,
            "truncate": False,
        }

        response = await self.client.post(
            "/api/embed",
            json=payload,
        )

        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise ValueError(
                "Ollama response does not contain valid embeddings."
            )

        if len(embeddings) != len(texts):
            raise ValueError(
                "The number of embeddings does not match "
                "the number of input texts."
            )

        validated_embeddings = []
        expected_dimensions = None

        for vector in embeddings:

            if not isinstance(vector, list) or not vector:
                raise ValueError("Ollama returned an empty or invalid vector.")

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in vector
            ):
                raise ValueError("Embedding contains non-numeric values.")

            if expected_dimensions is None:
                expected_dimensions = len(vector)

            elif len(vector) != expected_dimensions:
                raise ValueError(
                    "Embedding dimensions are inconsistent."
                )

            validated_embeddings.append(
                [float(value) for value in vector]
            )

        return validated_embeddings

    async def close(self) -> None:
        """Close the HTTP client."""

        await self.client.aclose()

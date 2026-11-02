
import asyncio
import json
import logging
import os

from typing import AsyncIterator
from uuid import uuid4

import httpx

from ai_proj.api.schemas import (
    AIRequest,
    BatchAIRequest,
    BatchAIResponse,
    BatchAIResult,
)

from ai_proj.core.cost_tracker import calculate_cost
from ai_proj.core.http_client import AsyncHTTPClient
from ai_proj.core.output_parser import parse_ai_output
from ai_proj.core.prompt_manager import PromptManager
from ai_proj.services.openai_chat import OpenAIChatCompletion


logger = logging.getLogger("ai_proj")

prompt_manager = PromptManager()

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ==========================================================
# PROMPT BUILDING
# ==========================================================

def build_prompt_messages(
    request: AIRequest,
) -> list[dict[str, str]]:
    """Build messages using the template for the requested use case."""

    use_case = getattr(request, "use_case", "generation")
    user_input = request.prompt

    context = getattr(request, "context", "")

    if context:
        user_input = (
            f"{user_input}\n\n"
            f"Additional Context:\n{context}"
        )

    return prompt_manager.build_messages(
        use_case=use_case,
        user_input=user_input,
    )


def build_ollama_prompt(request: AIRequest) -> str:
    """Combine rendered messages for Ollama's generate endpoint."""

    messages = build_prompt_messages(request)

    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    )


# ==========================================================
# OUTPUT VALIDATION
# ==========================================================

def validate_ai_output(
    response: str,
    request: AIRequest,
) -> str:
    """
    Validate AI output using the schema for the selected use case.

    Returns the validated output as a JSON string.
    """

    use_case = getattr(request, "use_case", "generation")

    if not isinstance(response, str) or not response.strip():
        raise ValueError(
            f"AI provider returned empty output for {use_case}."
        )

    try:
        parsed_output = parse_ai_output(
            response=response,
            use_case=use_case,
        )

        return parsed_output.model_dump_json()

    except ValueError:
        logger.error(
            "Output validation failed | use_case=%s | response=%r",
            use_case,
            response[:2000],
        )
        raise


# ==========================================================
# GENERAL RESPONSE GENERATION
# ==========================================================

async def generate_response(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> str:
    """Generate and validate a response using the selected provider."""

    if request.provider == "openai":
        output = await generate_openai_response(
            request=request,
            client=client,
        )
    else:
        output = await generate_ollama_response(
            request=request,
            client=client,
        )

    return validate_ai_output(
        response=output,
        request=request,
    )


# ==========================================================
# OPENAI RESPONSE
# ==========================================================

async def generate_openai_response(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> str:
    """Generate a response using OpenAI Chat Completions API."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not configured."
        )

    payload = {
        "model": request.model,
        "messages": build_prompt_messages(request),
        "temperature": request.temperature,
        "max_completion_tokens": request.max_tokens,
        "response_format": {
            "type": "json_object"
        },
    }

    response = await client.post(
        OPENAI_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        retryable=True,
    )

    try:
        data = response.json()

    except ValueError as exc:
        raise ValueError(
            "OpenAI returned invalid JSON."
        ) from exc

    try:
        output = data["choices"][0]["message"]["content"]

    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "OpenAI returned an invalid chat completion."
        ) from exc

    if not isinstance(output, str) or not output.strip():
        raise ValueError(
            "OpenAI returned empty text content."
        )

    return output.strip()


# ==========================================================
# OLLAMA RESPONSE
# ==========================================================

async def generate_ollama_response(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> str:
    """Generate a response using local Ollama."""

    payload = {
        "model": request.model,
        "prompt": build_ollama_prompt(request),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": max(request.max_tokens, 1500),
        },
    }

    response = await client.post(
        OLLAMA_URL,
        json=payload,
        retryable=False,
    )

    try:
        data = response.json()

    except ValueError as exc:
        raise ValueError(
            f"Ollama returned invalid JSON. "
            f"HTTP status: {response.status_code}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Ollama returned an unexpected response type."
        )
    if not isinstance(data, dict):
        raise ValueError(
            "Ollama returned an unexpected response type."
        )

    # Check whether Ollama stopped because of the token limit.
    if data.get("done_reason") == "length":
        raise ValueError(
            "Ollama stopped because it reached the output token limit. "
            "Increase num_predict or reduce the response length."
        )
    output = data.get("response")

    if not isinstance(output, str) or not output.strip():
        raise ValueError(
            "Ollama returned an empty response. "
            f"Available fields: {list(data.keys())}"
        )

    return output.strip()


# ==========================================================
# BATCH PROCESSING
# ==========================================================

async def generate_batch(
    batch: BatchAIRequest,
    client: AsyncHTTPClient,
    max_concurrency: int = 5,
) -> BatchAIResponse:
    """Process multiple AI requests concurrently."""

    if max_concurrency < 1:
        raise ValueError(
            "max_concurrency must be at least 1."
        )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_one(
        index: int,
        request: AIRequest,
    ) -> BatchAIResult:

        async with semaphore:

            try:
                output = await generate_response(
                    request=request,
                    client=client,
                )

                return BatchAIResult(
                    index=index,
                    status="success",
                    output=output,
                    model=request.model,
                )

            except httpx.TimeoutException:
                logger.warning(
                    "Batch item %s timed out",
                    index,
                )

                error = (
                    "The AI provider timed out. Please try again."
                )

            except httpx.ConnectError:
                logger.warning(
                    "Batch item %s connection failed",
                    index,
                )

                error = (
                    "The AI provider is unavailable. "
                    "Please try again later."
                )

            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Batch item %s received HTTP status %s",
                    index,
                    exc.response.status_code,
                )

                error = (
                    "The AI provider returned an unsuccessful response."
                )

            except httpx.RequestError as exc:
                logger.warning(
                    "Batch item %s network error: %s",
                    index,
                    type(exc).__name__,
                )

                error = (
                    "A network error occurred while contacting "
                    "the AI provider."
                )

            except ValueError:
                logger.exception(
                    "Batch item %s returned invalid provider output",
                    index,
                )

                error = (
                    "The AI provider returned an invalid response."
                )

            except Exception:
                logger.exception(
                    "Unexpected error processing batch item %s",
                    index,
                )

                error = "An unexpected error occurred."

            return BatchAIResult(
                index=index,
                status="error",
                model=request.model,
                error=error,
            )

    results = await asyncio.gather(
        *[
            process_one(index, request)
            for index, request in enumerate(batch.requests)
        ]
    )

    total = len(results)

    succeeded = sum(
        result.status == "success"
        for result in results
    )

    failed = total - succeeded

    if failed == 0:
        status = "success"

    elif succeeded == 0:
        status = "error"

    else:
        status = "partial_success"

    return BatchAIResponse(
        request_id=uuid4(),
        status=status,
        total=total,
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


# ==========================================================
# STREAMING RESPONSE
# ==========================================================

async def stream_response(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> AsyncIterator[str]:

    try:
        if request.provider == "openai":

            async for chunk in stream_openai_response(
                request,
                client,
            ):
                yield chunk

        else:

            async for chunk in stream_ollama_response(
                request,
                client,
            ):
                yield chunk

        yield 'event: done\ndata: {"message": "Completed"}\n\n'

    except httpx.HTTPStatusError as exc:

        logger.error(
            "Streaming provider HTTP error: %s | %s",
            exc.response.status_code,
            exc.response.text[:2000],
        )

        error = {
            "message": (
                "The AI provider returned an unsuccessful response."
            ),
            "status_code": exc.response.status_code,
        }

        yield f"event: error\ndata: {json.dumps(error)}\n\n"

    except httpx.RequestError as exc:

        logger.error(
            "Streaming network error: %s",
            type(exc).__name__,
        )

        error = {
            "message": "Could not connect to the AI provider."
        }

        yield f"event: error\ndata: {json.dumps(error)}\n\n"

    except Exception:

        logger.exception("Unexpected streaming error")

        error = {
            "message": "An unexpected streaming error occurred."
        }

        yield f"event: error\ndata: {json.dumps(error)}\n\n"


# ==========================================================
# OPENAI STREAMING
# ==========================================================

async def stream_openai_response(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> AsyncIterator[str]:

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not configured."
        )

    payload = {
        "model": request.model,
        "messages": build_prompt_messages(request),
        "temperature": request.temperature,
        "max_completion_tokens": request.max_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with client.stream_post(
        OPENAI_URL,
        headers=headers,
        json=payload,
    ) as response:

        async for line in response.aiter_lines():

            if not line.startswith("data:"):
                continue

            data = line[len("data:"):].strip()

            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)

            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])

            if not choices:
                continue

            delta = choices[0].get("delta", {})
            text = delta.get("content")

            if isinstance(text, str) and text:

                event_data = json.dumps({
                    "text": text
                })

                yield (
                    f"event: token\n"
                    f"data: {event_data}\n\n"
                )


# ==========================================================
# OLLAMA STREAMING
# ==========================================================

async def stream_ollama_response(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> AsyncIterator[str]:

    payload = {
        "model": request.model,
        "prompt": build_ollama_prompt(request),
        "stream": True,
        "format": "json",
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        },
    }

    async with client.stream_post(
        OLLAMA_URL,
        json=payload,
    ) as response:

        async for line in response.aiter_lines():

            if not line:
                continue

            try:
                data = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = data.get("response")

            if isinstance(text, str) and text:

                yield (
                    "event: token\n"
                    f"data: {json.dumps({'text': text})}\n\n"
                )

            if data.get("done"):

                input_tokens = int(
                    data.get("prompt_eval_count", 0) or 0
                )

                output_tokens = int(
                    data.get("eval_count", 0) or 0
                )

                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": (
                        input_tokens + output_tokens
                    ),
                    **calculate_cost(
                        provider="ollama",
                        model=request.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                }

                yield (
                    "event: usage\n"
                    f"data: {json.dumps(usage)}\n\n"
                )


# ==========================================================
# GENERATE RESPONSE WITH TOKEN USAGE AND COST
# ==========================================================

async def generate_response_with_usage(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> tuple[str, dict]:
    """Generate, validate, and return response with token usage."""

    if request.provider == "openai":

        output, input_tokens, output_tokens = (
            await generate_openai_response_with_usage(
                request,
                client,
            )
        )

    else:

        output, input_tokens, output_tokens = (
            await generate_ollama_response_with_usage(
                request,
                client,
            )
        )

    # Validate the generated output.
    output = validate_ai_output(
        response=output,
        request=request,
    )

    # Calculate usage.
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    cost = calculate_cost(
        provider=request.provider,
        model=request.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    usage.update(cost)

    logger.info(
        "AI usage | provider=%s | model=%s | input=%s | "
        "output=%s | cost=%s",
        request.provider,
        request.model,
        input_tokens,
        output_tokens,
        cost["total_cost"],
    )

    return output, usage


# ==========================================================
# OPENAI RESPONSE WITH USAGE
# ==========================================================

async def generate_openai_response_with_usage(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> tuple[str, int, int]:

    openai = OpenAIChatCompletion(
        client=client,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    result = await openai.complete(
        messages=build_prompt_messages(request)
    )

    usage = result["usage"]

    return (
        result["content"],
        usage["input_tokens"],
        usage["output_tokens"],
    )


# ==========================================================
# OLLAMA RESPONSE WITH USAGE
# ==========================================================

async def generate_ollama_response_with_usage(
    request: AIRequest,
    client: AsyncHTTPClient,
) -> tuple[str, int, int]:
    """Generate an Ollama response and extract token usage."""

    payload = {
        "model": request.model,
        "prompt": build_ollama_prompt(request),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": max(request.max_tokens, 1500),
        },
    }

    response = await client.post(
        OLLAMA_URL,
        json=payload,
        retryable=False,
    )

    # Parse Ollama response.
    try:
        data = response.json()

    except ValueError as exc:
        raise ValueError(
            f"Ollama returned invalid JSON. "
            f"HTTP status: {response.status_code}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected Ollama response type: "
            f"{type(data).__name__}"
        )

    # Extract generated text.
    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected Ollama response type: "
            f"{type(data).__name__}"
        )

    # Check whether Ollama stopped because of the token limit.
    if data.get("done_reason") == "length":
        raise ValueError(
            "Ollama stopped because it reached the output token limit. "
            "Increase num_predict or reduce the response length."
        )

    output = data.get("response")

    if not isinstance(output, str) or not output.strip():
        logger.error(
            "Ollama returned empty output. Response data: %r",
            data,
        )

        raise ValueError(
            "Ollama returned an empty response. "
            f"Available fields: {list(data.keys())}"
        )

    # Extract token usage.
    input_tokens = data.get("prompt_eval_count") or 0
    output_tokens = data.get("eval_count") or 0

    try:
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)

    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Ollama returned invalid token usage information."
        ) from exc

    return (
        output.strip(),
        input_tokens,
        output_tokens,
    )

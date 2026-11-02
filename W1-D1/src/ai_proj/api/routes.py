
import logging
from uuid import uuid4

import httpx
from fastapi.responses import StreamingResponse

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from ai_proj.api.schemas import (
    AIRequest,
    AIResponse,
    BatchAIRequest,
    BatchAIResponse,
)

from ai_proj.core.http_client import AsyncHTTPClient

from ai_proj.services.ai_service import (
    generate_batch,
    generate_response_with_usage,
    stream_response,
)


logger = logging.getLogger("ai_proj")

router = APIRouter(
    prefix="/api/v1",
    tags=["AI"],
)


def get_http_client(
    request: Request,
) -> AsyncHTTPClient:

    return request.app.state.http_client



@router.post("/generate", response_model=AIResponse)
async def generate(
    payload: AIRequest,
    client: AsyncHTTPClient = Depends(get_http_client),
) -> AIResponse:

    try:
        output, usage = await generate_response_with_usage(
            request=payload,
            client=client,
        )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="The AI provider took too long to respond.",
        ) from exc

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI provider is unavailable.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        logger.error(
            "AI provider returned HTTP status %s: %s",
            exc.response.status_code,
            exc.response.text[:2000],
        )
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned an unsuccessful response.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="A network error occurred while contacting the AI provider.",
        ) from exc

    except ValueError as exc:
        logger.warning("Invalid AI provider response: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned an invalid response.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error generating AI response")
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        ) from exc

    return AIResponse(
        request_id=uuid4(),
        status="success",
        output=output,
        model=payload.model,
        usage=usage,
    )



@router.post(
    "/batch",
    response_model=BatchAIResponse,
)
async def generate_batch_endpoint(
    payload: BatchAIRequest,
    client: AsyncHTTPClient = Depends(get_http_client),
) -> BatchAIResponse:

    return await generate_batch(
        batch=payload,
        client=client,
        max_concurrency=5,
    )

@router.post("/generate/stream")
async def generate_stream(
    payload: AIRequest,
    client: AsyncHTTPClient = Depends(get_http_client),
):
    return StreamingResponse(
        stream_response(payload, client),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


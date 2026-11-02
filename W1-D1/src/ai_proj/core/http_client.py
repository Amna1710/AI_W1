
import asyncio
import logging

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

from ai_proj.core.decorators.logging import log_function


logger = logging.getLogger("ai_proj")


class AsyncHTTPClient:

    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    DEFAULT_RETRY_METHODS = {
        "GET",
        "HEAD",
        "OPTIONS",
    }

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        headers: dict[str, str] | None = None,
        max_concurrent_requests: int = 5,
    ) -> None:

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        if base_delay < 0 or max_delay < 0:
            raise ValueError("Retry delays cannot be negative.")

        if max_concurrent_requests < 1:
            raise ValueError(
                "max_concurrent_requests must be at least 1."
            )

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self._semaphore = asyncio.Semaphore(
            max_concurrent_requests
        )

        self._client = httpx.AsyncClient(
            base_url=base_url or "",
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

        logger.info(
            "AsyncHTTPClient initialized | timeout=%s | "
            "max_retries=%s | concurrency=%s",
            timeout,
            max_retries,
            max_concurrent_requests,
        )

    # ======================================================
    # MAIN REQUEST METHOD
    # ======================================================

    @log_function(
        log_args=False,
        log_result=False,
        slow_threshold=2.0,
    )
    async def request(
        self,
        method: str,
        url: str,
        *,
        retryable: bool | None = None,
        **kwargs: Any,
    ) -> httpx.Response:

        method = method.upper()

        should_retry = (
            method in self.DEFAULT_RETRY_METHODS
            if retryable is None
            else retryable
        )

        max_attempts = (
            self.max_retries + 1
            if should_retry
            else 1
        )

        for attempt in range(max_attempts):

            try:
                async with self._semaphore:

                    response = await self._client.request(
                        method,
                        url,
                        **kwargs,
                    )

                # Retry selected HTTP status codes.
                if (
                    response.status_code in self.RETRYABLE_STATUS_CODES
                    and attempt < max_attempts - 1
                ):

                    delay = self._get_retry_delay(
                        response,
                        attempt,
                    )

                    logger.warning(
                        "HTTP %s returned status %s. "
                        "Retry %s/%s in %.2f seconds.",
                        method,
                        response.status_code,
                        attempt + 1,
                        max_attempts - 1,
                        delay,
                    )

                    await response.aclose()
                    await asyncio.sleep(delay)

                    continue

                if response.is_error:

                    try:
                        error_data = response.json()

                    except ValueError:
                        error_data = response.text

                    logger.error(
                        "HTTP %s error | status=%s | response=%r",
                        method,
                        response.status_code,
                        error_data,
                    )

                response.raise_for_status()

                return response

            # Do not retry read timeouts.
            # This is particularly important for long-running
            # AI generation POST requests.
            except httpx.ReadTimeout:

                logger.error(
                    "HTTP %s encountered ReadTimeout. "
                    "Request will not be retried.",
                    method,
                )

                raise

            # Handle other transport errors.
            except httpx.TransportError as exc:

                if attempt >= max_attempts - 1:

                    logger.error(
                        "HTTP %s failed after %s attempt(s): %s",
                        method,
                        attempt + 1,
                        type(exc).__name__,
                    )

                    raise

                delay = self._backoff(attempt)

                logger.warning(
                    "HTTP %s encountered %s. "
                    "Retry %s/%s in %.2f seconds.",
                    method,
                    type(exc).__name__,
                    attempt + 1,
                    max_attempts - 1,
                    delay,
                )

                await asyncio.sleep(delay)

        raise RuntimeError(
            "HTTP request ended unexpectedly."
        )

    # ======================================================
    # HTTP METHODS
    # ======================================================

    async def get(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:

        return await self.request(
            "GET",
            url,
            **kwargs,
        )

    async def post(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:

        return await self.request(
            "POST",
            url,
            **kwargs,
        )

    async def put(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:

        return await self.request(
            "PUT",
            url,
            **kwargs,
        )

    async def patch(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:

        return await self.request(
            "PATCH",
            url,
            **kwargs,
        )

    async def delete(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:

        return await self.request(
            "DELETE",
            url,
            **kwargs,
        )

    # ======================================================
    # RETRY HELPERS
    # ======================================================

    def _backoff(
        self,
        attempt: int,
    ) -> float:

        return min(
            self.base_delay * (2 ** attempt),
            self.max_delay,
        )

    def _get_retry_delay(
        self,
        response: httpx.Response,
        attempt: int,
    ) -> float:

        retry_after = response.headers.get(
            "Retry-After"
        )

        if retry_after is not None:

            try:
                seconds = float(retry_after)

                if seconds >= 0:
                    return min(
                        seconds,
                        self.max_delay,
                    )

            except ValueError:
                pass

        return self._backoff(attempt)

    # ======================================================
    # CLOSE CLIENT
    # ======================================================

    async def aclose(self) -> None:

        await self._client.aclose()

    async def __aenter__(self) -> "AsyncHTTPClient":

        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:

        await self.aclose()

    # ======================================================
    # STREAMING POST
    # ======================================================

    @asynccontextmanager
    async def stream_post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> AsyncIterator[httpx.Response]:

        async with self._semaphore:

            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=json,
            ) as response:

                response.raise_for_status()

                yield response

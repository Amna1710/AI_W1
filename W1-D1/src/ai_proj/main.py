
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_proj.api.routes import router
from ai_proj.core.http_client import AsyncHTTPClient


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logger = logging.getLogger("ai_proj")

logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        "app.log",
        encoding="utf-8",
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# --------------------------------------------------
# FastAPI lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting AI Project API")

    app.state.http_client = AsyncHTTPClient(
        timeout=240.0,
        max_retries=3,
        base_delay=0.5,
        max_delay=10.0,
        max_concurrent_requests=5,
    )

    logger.info(
        "Shared HTTP client initialized. "
        "Concurrency limit: 5"
    )

    try:
        yield

    finally:
        await app.state.http_client.aclose()

        logger.info("HTTP client closed")
        logger.info("Shutting down AI Project API")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI Project API",
    description="A modular API for AI-powered applications.",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(router)


# --------------------------------------------------
# General endpoints
# --------------------------------------------------

@app.get("/", tags=["General"])
async def root():

    return {
        "message": "Welcome to AI Project API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check():

    return {
        "status": "healthy",
    }

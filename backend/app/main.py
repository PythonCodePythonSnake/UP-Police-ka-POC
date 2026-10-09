"""FastAPI application entry point.

Run from backend/:   uvicorn app.main:app --reload
Interactive docs:    http://localhost:8000/docs
"""

import logging
import os
import sys
import re
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.api.routes import chat, complaints, sessions
from app.core.config import settings
from app.core.logging import setup_logging
from app.integrations.llm_client import close_llm_client

setup_logging()
logger = logging.getLogger(__name__)


class _RedactSessionIds(logging.Filter):
    """uvicorn's access log prints request paths, and paths contain session IDs (the access secret)."""

    _pattern = re.compile(r"(/api/(?:chat|sessions|complaints)/)[A-Za-z0-9_-]{20,}")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                self._pattern.sub(r"\1<redacted>", arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


logging.getLogger("uvicorn.access").addFilter(_RedactSessionIds())


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "Starting %s (LLM %s/%s, key set: %s; Pinecone settings present: %s)",
        settings.app_name,
        settings.llm_provider,
        settings.llm_model,
        settings.llm_configured,
        settings.pinecone_configured,
    )
    if not settings.llm_configured:
        logger.warning("LLM_API_KEY is not set: chat and draft endpoints will return 503.")
    yield
    await close_llm_client()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Backend for a conversational complaint-drafting assistant. It prepares complaint "
        "drafts for review; it does not file or submit anything."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(complaints.router)


class LLMHealth(BaseModel):
    provider: str
    model: str
    configured: bool = Field(description="An API key is set. The provider is not contacted.")


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    llm: LLMHealth
    retrieval_configured: bool = Field(description="Pinecone key and index name are set. Does not prove the index is queryable.")


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(
        app=settings.app_name,
        llm=LLMHealth(
            provider=settings.llm_provider,
            model=settings.llm_model,
            configured=settings.llm_configured,
        ),
        retrieval_configured=settings.pinecone_configured,
    )
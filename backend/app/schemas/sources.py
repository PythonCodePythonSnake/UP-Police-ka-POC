"""Schemas for retrieved reference material."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalStatus(str, Enum):
    SUCCESS = "success"          # The search ran. `sources` may legitimately be empty.
    UNAVAILABLE = "unavailable"  # Retrieval cannot run (not configured / unsupported index).
    FAILED = "failed"            # Retrieval was attempted but errored or returned unusable data.


class RetrievedSource(BaseModel):
    """One retrieved record. Every field comes from the Pinecone result; nothing is generated."""

    source_id: str = Field(min_length=1, description="Source identifier from metadata, else the Pinecone record ID.")
    title: str | None = None
    url: str | None = None
    content: str = Field(description="Retrieved text, as stored in the index.")
    score: float | None = Field(
        default=None,
        description="Similarity score. Says nothing about authority, currency, or legal applicability.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Remaining record fields, unmodified.")


class RetrievalResponse(BaseModel):
    status: RetrievalStatus
    sources: list[RetrievedSource] = Field(default_factory=list)
    message: str | None = Field(
        default=None,
        description="Explains unavailable/failed states, or caveats a successful result.",
    )
    warnings: list[str] = Field(default_factory=list, description="Data-quality limitations noticed.")
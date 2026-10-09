"""Shared FastAPI dependencies."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import settings
from app.services.session_service import SessionService


@lru_cache
def get_session_service() -> SessionService:
    """One shared SessionService for the whole process."""
    return SessionService(settings.sessions_dir, settings.session_expiry_hours)


SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
"""Central application configuration, loaded from environment variables / .env."""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory (this file lives at backend/app/core/config.py)
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # Fix: Ignore blank values so defaults can take over
    )

    # Application
    app_name: str = "Complaint Filing Assistant"
    debug: bool = True
    frontend_origin: str = "http://localhost:5173"  # Vite dev server default

    # Sessions
    storage_dir: Path = BACKEND_DIR / "storage"
    session_expiry_hours: int = Field(default=24, gt=0)

    # LLM (provider implementations live in integrations/llm_client.py)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # Pinecone (integration is not implemented yet)
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str | None = None

    @field_validator("storage_dir")
    @classmethod
    def _resolve_storage_dir(cls, value: Path) -> Path:
        # Relative paths in .env are interpreted relative to backend/, not the CWD.
        return value if value.is_absolute() else BACKEND_DIR / value

    @property
    def sessions_dir(self) -> Path:
        return self.storage_dir / "sessions"

    @property
    def llm_configured(self) -> bool:
        return self.llm_api_key is not None

    @property
    def pinecone_configured(self) -> bool:
        return self.pinecone_api_key is not None and bool(self.pinecone_index_name)


settings = Settings()
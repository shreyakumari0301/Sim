"""Environment-backed configuration (Phase 1)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings loaded from the environment and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # Documentation portals (not API hosts)
    open_fda_site: Annotated[
        HttpUrl,
        Field(
            validation_alias="OPEN_FDA_SITE",
            description="Human-facing OpenFDA authentication/docs URL.",
        ),
    ] = "https://open.fda.gov/apis/authentication/"  # type: ignore[assignment]
    ncbi_site: Annotated[
        HttpUrl,
        Field(
            validation_alias="NCBI_SITE",
            description="Human-facing NCBI account / API key settings URL.",
        ),
    ] = "https://account.ncbi.nlm.nih.gov/settings/"  # type: ignore[assignment]

    openfda_api_base: Annotated[
        HttpUrl,
        Field(validation_alias="OPENFDA_API_BASE", description="OpenFDA API root URL."),
    ] = "https://api.fda.gov"  # type: ignore[assignment]

    openfda_api_key: Annotated[
        str | None,
        Field(
            default=None,
            validation_alias=AliasChoices("OPENFDA_API_KEY", "OPEN_FDA_API_KEY"),
            description="Optional OpenFDA API key.",
        ),
    ]

    ncbi_eutils_base: Annotated[
        HttpUrl,
        Field(validation_alias="NCBI_EUTILS_BASE", description="NCBI E-utilities base URL."),
    ] = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"  # type: ignore[assignment]

    ncbi_api_key: Annotated[
        str | None,
        Field(default=None, validation_alias="NCBI_API_KEY", description="Optional NCBI API key."),
    ]

    data_dir: Annotated[
        Path,
        Field(validation_alias="DATA_DIR", description="Root directory for datasets."),
    ] = Path("data")

    log_level: Annotated[
        LogLevel,
        Field(validation_alias="LOG_LEVEL"),
    ] = "INFO"

    http_max_retries: Annotated[
        int,
        Field(validation_alias="HTTP_MAX_RETRIES", ge=0, le=50),
    ] = 5

    http_backoff_base_seconds: Annotated[
        float,
        Field(validation_alias="HTTP_BACKOFF_BASE_SECONDS", gt=0),
    ] = 0.5

    http_connect_timeout_seconds: Annotated[
        float,
        Field(
            validation_alias="HTTP_CONNECT_TIMEOUT",
            ge=1.0,
            le=600.0,
            description="TLS + TCP connect timeout for outbound HTTP (ingestion).",
        ),
    ] = 90.0

    http_read_timeout_seconds: Annotated[
        float,
        Field(
            validation_alias="HTTP_READ_TIMEOUT",
            ge=1.0,
            le=600.0,
            description="Read timeout after a connection is established.",
        ),
    ] = 120.0

    http_force_ipv4: Annotated[
        bool,
        Field(
            validation_alias="HTTP_FORCE_IPV4",
            description="Resolve DNS to IPv4 only during ingest (avoids slow/unreachable IPv6 on some WSL setups).",
        ),
    ] = False

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: object) -> str:
        if isinstance(v, str):
            return v.upper().strip()
        return str(v)

    def resolved_data_dir(self) -> Path:
        """Absolute path to `data_dir`."""
        return self.data_dir.expanduser().resolve()

    def ensure_data_dir(self) -> Path:
        """Ensure `data_dir` exists and return its absolute path."""
        path = self.resolved_data_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (reload process to pick up env changes)."""
    return Settings()

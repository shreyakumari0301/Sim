"""Settings loader smoke tests."""

from pathlib import Path

import pytest

from sim.settings import Settings, get_settings


def _clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OPEN_FDA_SITE",
        "NCBI_SITE",
        "OPENFDA_API_BASE",
        "OPENFDA_API_KEY",
        "OPEN_FDA_API_KEY",
        "NCBI_EUTILS_BASE",
        "NCBI_API_KEY",
        "NCBI_TOOL",
        "NCBI_EMAIL",
        "DATA_DIR",
        "LOG_LEVEL",
        "HTTP_MAX_RETRIES",
        "HTTP_BACKOFF_BASE_SECONDS",
        "HTTP_CONNECT_TIMEOUT",
        "HTTP_READ_TIMEOUT",
        "HTTP_FORCE_IPV4",
    ):
        monkeypatch.delenv(key, raising=False)


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    assert str(s.openfda_api_base).rstrip("/") == "https://api.fda.gov"
    assert "eutils.ncbi.nlm.nih.gov" in str(s.ncbi_eutils_base)
    assert s.openfda_api_key is None
    assert s.ncbi_api_key is None
    assert s.ncbi_tool == "sim_ingest"
    assert s.ncbi_email is None
    assert s.log_level == "INFO"


def test_open_fda_api_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPEN_FDA_API_KEY", "test-openfda-key")
    monkeypatch.delenv("OPENFDA_API_KEY", raising=False)
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    assert s.openfda_api_key == "test-openfda-key"


def test_data_dir_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "datasets"))
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    assert s.resolved_data_dir() == (tmp_path / "datasets").resolve()


def test_ensure_data_dir_creates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    target = tmp_path / "new" / "data"
    monkeypatch.setenv("DATA_DIR", str(target))
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    out = s.ensure_data_dir()
    assert out == target.resolve()
    assert out.is_dir()


def test_invalid_log_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "not-a-level")
    with pytest.raises(Exception):
        Settings(_env_file=None)

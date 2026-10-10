"""Tests for core/config.py. Settings are built with _env_file=None, so a real .env is never read."""

import pytest
from pydantic import ValidationError

from app.core.config import BACKEND_DIR, Settings

ENV_VARS = [
    "APP_NAME",
    "DEBUG",
    "FRONTEND_ORIGIN",
    "STORAGE_DIR",
    "SESSION_EXPIRY_HOURS",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_TIMEOUT_SECONDS",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path, text: str):
    path = tmp_path / "test.env"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Defaults and parsing
# --------------------------------------------------------------------------- #
def test_defaults():
    s = Settings(_env_file=None)
    assert s.app_name == "Complaint Filing Assistant"
    assert s.debug is True
    assert s.frontend_origin == "http://localhost:5173"
    assert s.session_expiry_hours == 24
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.llm_timeout_seconds == 30.0
    assert s.llm_api_key is None
    assert s.pinecone_api_key is None
    assert s.pinecone_index_name is None
    assert s.pinecone_namespace is None


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Demo")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("FRONTEND_ORIGIN", "http://127.0.0.1:3000")
    monkeypatch.setenv("SESSION_EXPIRY_HOURS", "48")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("PINECONE_NAMESPACE", "circulars")
    s = Settings(_env_file=None)
    assert s.app_name == "Demo"
    assert s.debug is False
    assert s.frontend_origin == "http://127.0.0.1:3000"
    assert s.session_expiry_hours == 48
    assert s.llm_provider == "gemini"
    assert s.llm_model == "gemini-2.5-flash"
    assert s.llm_timeout_seconds == 12.5
    assert s.pinecone_namespace == "circulars"


def test_values_are_read_from_an_env_file(tmp_path):
    env_file = write_env(tmp_path, "LLM_PROVIDER=gemini\nSESSION_EXPIRY_HOURS=6\nLLM_API_KEY=file-key\n")
    s = Settings(_env_file=env_file)
    assert s.llm_provider == "gemini"
    assert s.session_expiry_hours == 6
    assert s.llm_api_key.get_secret_value() == "file-key"


def test_environment_variable_beats_env_file(tmp_path, monkeypatch):
    env_file = write_env(tmp_path, "LLM_PROVIDER=gemini\n")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert Settings(_env_file=env_file).llm_provider == "openai"


def test_unknown_variables_in_env_file_are_ignored(tmp_path):
    env_file = write_env(tmp_path, "SOMETHING_ELSE=1\nLLM_MODEL=custom-model\n")
    assert Settings(_env_file=env_file).llm_model == "custom-model"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_backend_dir_points_at_the_backend_folder():
    assert (BACKEND_DIR / "app" / "core" / "config.py").is_file()


def test_env_file_is_located_relative_to_backend_dir_not_cwd():
    assert Settings.model_config["env_file"] == BACKEND_DIR / ".env"


def test_default_storage_dir_is_inside_backend():
    s = Settings(_env_file=None)
    assert s.storage_dir == BACKEND_DIR / "storage"
    assert s.sessions_dir == BACKEND_DIR / "storage" / "sessions"


def test_relative_storage_dir_resolves_from_backend_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_DIR", "data")
    s = Settings(_env_file=None)
    assert s.storage_dir == BACKEND_DIR / "data"
    assert s.sessions_dir == BACKEND_DIR / "data" / "sessions"


def test_absolute_storage_dir_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "elsewhere"))
    assert Settings(_env_file=None).storage_dir == tmp_path / "elsewhere"


# --------------------------------------------------------------------------- #
# Configured flags and secrets
# --------------------------------------------------------------------------- #
def test_llm_configured_reflects_api_key(monkeypatch):
    assert Settings(_env_file=None).llm_configured is False
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    assert Settings(_env_file=None).llm_configured is True


@pytest.mark.parametrize(
    ("api_key", "index_name", "expected"),
    [(None, None, False), ("key", None, False), (None, "index", False), ("key", "index", True)],
)
def test_pinecone_configured_needs_key_and_index(monkeypatch, api_key, index_name, expected):
    if api_key:
        monkeypatch.setenv("PINECONE_API_KEY", api_key)
    if index_name:
        monkeypatch.setenv("PINECONE_INDEX_NAME", index_name)
    assert Settings(_env_file=None).pinecone_configured is expected


def test_secrets_are_masked_everywhere_except_get_secret_value(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-super-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-super-secret")
    s = Settings(_env_file=None)
    for rendering in (repr(s), str(s), str(s.model_dump()), s.model_dump_json()):
        assert "sk-super-secret" not in rendering
        assert "pc-super-secret" not in rendering
    assert s.llm_api_key.get_secret_value() == "sk-super-secret"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SESSION_EXPIRY_HOURS", "0"),
        ("SESSION_EXPIRY_HOURS", "-5"),
        ("SESSION_EXPIRY_HOURS", "soon"),
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("LLM_TIMEOUT_SECONDS", "-1"),
        ("DEBUG", "maybe"),
    ],
)
def test_invalid_values_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# --------------------------------------------------------------------------- #
# Blank values (copying a template and leaving "KEY=" empty is a common mistake)
# A blank value must behave like an unset one. These currently FAIL: see the
# env_ignore_empty fix for config.py.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("var", ["LLM_API_KEY", "PINECONE_API_KEY"])
def test_blank_api_key_in_environment_counts_as_unset(monkeypatch, var):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "index")
    monkeypatch.setenv(var, "")
    s = Settings(_env_file=None)
    assert s.llm_configured is False
    assert s.pinecone_configured is False


def test_blank_api_keys_in_env_file_count_as_unset(tmp_path):
    env_file = write_env(tmp_path, "LLM_API_KEY=\nPINECONE_API_KEY=\nPINECONE_INDEX_NAME=index\n")
    s = Settings(_env_file=env_file)
    assert s.llm_configured is False
    assert s.pinecone_configured is False


@pytest.mark.parametrize(
    ("var", "attribute", "default"),
    [
        ("LLM_MODEL", "llm_model", "gpt-4o-mini"),
        ("FRONTEND_ORIGIN", "frontend_origin", "http://localhost:5173"),
        ("SESSION_EXPIRY_HOURS", "session_expiry_hours", 24),
    ],
)
def test_blank_setting_falls_back_to_its_default(monkeypatch, var, attribute, default):
    monkeypatch.setenv(var, "")
    assert getattr(Settings(_env_file=None), attribute) == default
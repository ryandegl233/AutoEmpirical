import pytest

from Benchmark.scripts import run_ase2022_llm_baseline as runner
from Benchmark.scripts.run_ase2022_llm_baseline import resolve_run_config


def test_resolve_run_config_prefers_self_api_settings() -> None:
    config = resolve_run_config(
        {
            "SELF_API": "self-key",
            "SELF_BASE_URL": "https://self.example/v1",
            "OPENAI_API_KEY": "teacher-key",
            "BASE_URL": "https://teacher.example/v1",
        }
    )

    assert config["api_key"] == "self-key"
    assert config["base_url"] == "https://self.example/v1"


def test_resolve_run_config_uses_deepseek_credentials_for_deepseek_provider() -> None:
    config = resolve_run_config(
        {
            "SELF_API": "proxy-key",
            "SELF_BASE_URL": "https://proxy.example/v1",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.example",
        },
        provider="deepseek",
    )

    assert config["api_key"] == "deepseek-key"
    assert config["base_url"] == "https://api.deepseek.example"


def test_resolve_run_config_uses_official_gemini_credentials() -> None:
    config = resolve_run_config(
        {
            "GOOGLE_API_KEY": "google-key",
            "GEMINI_API_KEY": "gemini-fallback-key",
            "OPENAI_API_KEY": "must-not-be-used",
        },
        provider="gemini",
    )

    assert config["api_key"] == "google-key"
    assert config["base_url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert config["wire_api"] == "chat_completions"


def test_gemini_single_llm_requires_explicit_models() -> None:
    with pytest.raises(SystemExit, match="--provider gemini requires explicit --models"):
        runner._resolve_selected_models("gemini", None, ["proxy-default"])

    assert runner._resolve_selected_models(
        "gemeni", ["gemini-3.6-flash"], ["proxy-default"]
    ) == ["gemini-3.6-flash"]

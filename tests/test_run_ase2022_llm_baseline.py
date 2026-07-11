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

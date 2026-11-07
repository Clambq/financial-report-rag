from __future__ import annotations

from pathlib import Path

import pytest

from finqa_rag.api.bootstrap import FinSightSettings


ROOT = Path(__file__).parents[1]


def test_settings_require_vllm_base_url_and_model() -> None:
    with pytest.raises(ValueError, match="FINSIGHT_VLLM_BASE_URL"):
        FinSightSettings.from_env({})
    with pytest.raises(ValueError, match="FINSIGHT_VLLM_MODEL"):
        FinSightSettings.from_env(
            {"FINSIGHT_VLLM_BASE_URL": "http://localhost:8001/v1"}
        )


def test_committed_env_example_contains_no_real_secret() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "FINSIGHT_VLLM_BASE_URL=" in text
    assert "FINSIGHT_VLLM_MODEL=" in text
    assert "sk-" not in text
    assert "password=" not in text.lower()


def test_compose_has_healthchecks_and_optional_vllm_prefix_cache() -> None:
    text = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert text.count("healthcheck:") >= 2
    assert "profiles: [\"vllm\"]" in text
    assert "--enable-prefix-caching" in text
    assert "vllm/vllm-openai" in text
    assert "service: vllm" not in text
    assert "env_file:" not in text
    assert "FINSIGHT_VLLM_BASE_URL:" in text


def test_application_image_has_non_root_user_and_healthcheck() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "HEALTHCHECK" in text
    assert "USER app" in text
    assert "finqa-rag" in text
    assert '.[ml]' in text

